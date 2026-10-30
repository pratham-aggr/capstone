"""Train the CLAS sensing vectors.

Only the per-layer vectors c_L are learned; every model weight stays frozen. The loss is
next-token cross-entropy on prompt->completion pairs, masked to the COMPLETION tokens.

Training data deliberately mixes both classes, so the gate must learn context rather than
a constant:
    risky prompt  (MedSafetyBench TRAIN) -> config.REFUSAL_TARGET
    benign prompt (MedQuAD TRAIN)        -> its (truncated) reference answer
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

import config
import data as data_mod
from model_utils import format_prompt
from steering import ClasGates, clas_addition


def build_pairs(splits: Dict, seed: int = config.SEED,
                benign_completions: List[str] = None,
                risky_completions: List[str] = None) -> List[Tuple[str, str, str]]:
    """Return shuffled (prompt, completion, class_label) triples.

    The risky class is always trained toward config.REFUSAL_TARGET. The benign class is
    trained either toward its MedQuAD reference answer or toward the model's own unsteered
    answer, depending on config.CLAS_BENIGN_TARGET -- see the note there for why the
    reference answer provides no usable gradient signal.
    """
    if config.CLAS_RISKY_TARGET == "steered_self":
        assert risky_completions is not None, "steered_self requires risky_completions"
        pairs = [(p, c.strip(), "risky")
                 for p, c in zip(splits["risky_train"], risky_completions) if c.strip()]
    else:
        pairs = [(p, config.REFUSAL_TARGET, "risky") for p in splits["risky_train"]]
    if config.CLAS_BENIGN_TARGET == "on_policy":
        assert benign_completions is not None, "on_policy targets require benign_completions"
        pairs += [
            (r["question"], c.strip(), "benign")
            for r, c in zip(splits["benign_train_rows"], benign_completions)
            if c.strip()
        ]
    else:
        pairs += [
            (r["question"], data_mod.truncate_answer(r["answer"], max_words=60), "benign")
            for r in splits["benign_train_rows"]
        ]
    random.Random(seed).shuffle(pairs)
    return pairs


def _encode_batch(tokenizer, batch: List[Tuple[str, str, str]], device: str):
    """Right-pad prompt+completion; label only the completion tokens."""
    input_rows, label_rows = [], []
    for prompt, completion, _ in batch:
        p_ids = tokenizer(format_prompt(tokenizer, prompt),
                          add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        c_ids = c_ids[: config.CLAS_MAX_TARGET_TOKENS] + [tokenizer.eos_token_id]
        ids = (p_ids + c_ids)[-config.CLAS_MAX_LEN:]
        labels = ([-100] * len(p_ids) + c_ids)[-config.CLAS_MAX_LEN:]
        input_rows.append(ids)
        label_rows.append(labels)

    width = max(len(r) for r in input_rows)
    pad = tokenizer.eos_token_id
    input_ids = torch.tensor([r + [pad] * (width - len(r)) for r in input_rows])
    labels = torch.tensor([r + [-100] * (width - len(r)) for r in label_rows])
    attention_mask = torch.tensor(
        [[1] * len(r) + [0] * (width - len(r)) for r in input_rows]
    )
    return (input_ids.to(device), labels.to(device), attention_mask.to(device))


def _masked_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean, then mean over sequences.

    NOT a flat token mean: benign reference answers are several times longer than the
    12-token refusal string, so a flat mean would weight the benign class far more
    heavily and the gate would simply learn to stay near zero.
    """
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none", ignore_index=-100,
    ).view(shift_labels.shape)
    valid = (shift_labels != -100).float()
    per_seq = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    return per_seq.mean()


@torch.no_grad()
def gate_stats(model, tokenizer, gates: ClasGates, directions, pairs, device, n: int = 8):
    """Report mean/std/max gate magnitude per layer, split by class.

    std/|mean| is the diagnostic for the failure mode where the gate collapses into a
    constant: if the gate barely varies with input, CLAS has degenerated into fixed
    addition and the comparison is null by construction.
    """
    out = {}
    for label in ("risky", "benign"):
        subset = [p for p in pairs if p[2] == label][:n]
        if not subset:
            continue
        ids, labels, mask = _encode_batch(tokenizer, subset, device)
        trace: Dict[int, list] = {}
        holder = {"attention_mask": mask}
        with clas_addition(model, directions, gates, holder, trace=trace):
            model(input_ids=ids, attention_mask=mask, use_cache=False)
        stats = {}
        for layer, chunks in trace.items():
            g = torch.cat([c.flatten() for c in chunks])
            stats[layer] = (float(g.mean()), float(g.std()), float(g.abs().max()))
        out[label] = stats
    return out


def train_gates(model, tokenizer, directions, splits, device: str = config.DEVICE,
                epochs: int = None, verbose: bool = True,
                benign_completions: List[str] = None,
                risky_completions: List[str] = None) -> ClasGates:
    epochs = config.CLAS_EPOCHS if epochs is None else epochs
    pairs = build_pairs(splits, benign_completions=benign_completions,
                        risky_completions=risky_completions)
    gates = ClasGates(config.STEER_LAYERS, model.config.hidden_size).to(device)

    # The CLAS paper gives the bias entry of c its own, much larger learning rate
    # (1e-1 for Llama vs 3e-3 for the vector part); with zero init this is load-bearing.
    optimiser = torch.optim.AdamW([
        {"params": gates.vec_parameters(), "lr": config.CLAS_LR},
        {"params": gates.bias_parameters(), "lr": config.CLAS_BIAS_LR},
    ])

    if verbose:
        print(f"  CLAS: {len(pairs)} pairs "
              f"({sum(p[2] == 'risky' for p in pairs)} risky / "
              f"{sum(p[2] == 'benign' for p in pairs)} benign), "
              f"{epochs} epochs, lr={config.CLAS_LR} bias_lr={config.CLAS_BIAS_LR}")

    history = []
    for epoch in range(epochs):
        random.Random(config.SEED + epoch).shuffle(pairs)
        total, n_batches = 0.0, 0
        for start in range(0, len(pairs), config.CLAS_BATCH_SIZE):
            batch = pairs[start:start + config.CLAS_BATCH_SIZE]
            ids, labels, mask = _encode_batch(tokenizer, batch, device)
            holder = {"attention_mask": mask}
            with clas_addition(model, directions, gates, holder,
                               max_abs_gate=config.CLAS_MAX_ABS_GATE):
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            loss = _masked_loss(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("CLAS loss became non-finite")
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gates.parameters(), config.CLAS_GRAD_CLIP)
            optimiser.step()
            total += float(loss)
            n_batches += 1
        mean_loss = total / max(n_batches, 1)
        history.append(mean_loss)
        if verbose:
            stats = gate_stats(model, tokenizer, gates, directions, pairs, device)
            msg = " | ".join(
                f"{lab}: mean g@20={stats[lab][20][0]:+.2f} std={stats[lab][20][1]:.2f}"
                for lab in stats if 20 in stats[lab]
            )
            print(f"  epoch {epoch + 1}/{epochs}  loss={mean_loss:.4f}   {msg}")
    return gates, history

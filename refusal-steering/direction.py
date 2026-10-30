"""Extract the refusal direction by difference-in-means (Arditi et al., 2024).

  r_L    = mean(harmful activations at layer L) - mean(harmless activations at layer L)
  r_hat_L = r_L / ||r_L||

Activations are taken at the LAST token position of the chat-formatted prompt, from the
OUTPUT of layer L (== hidden_states[L+1]) -- the same tensor the hooks later modify.

Only GENERAL data is used (AdvBench harmful vs Alpaca harmless), so the direction is
independent of the medical evaluation distribution. One forward pass yields every layer,
so we extract a direction at each steered layer for free.
"""

from __future__ import annotations

from typing import Dict, List

import torch

import config
from model_utils import tokenize_prompts


@torch.no_grad()
def last_token_activations(model, tokenizer, prompts: List[str],
                           batch_size: int = 16) -> torch.Tensor:
    """Returns [n_layers+1, n_prompts, hidden] on CPU in fp32.
    Index L+1 is the output of decoder layer L; index 0 is the embedding output."""
    chunks = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tokenize_prompts(tokenizer, batch)
        hidden = model(**enc, output_hidden_states=True).hidden_states
        chunks.append(torch.stack([h[:, -1, :] for h in hidden], 0).float().cpu())
    return torch.cat(chunks, dim=1)


def extract_directions(model, tokenizer, harmful: List[str], harmless: List[str]) -> Dict:
    """Compute r at every layer; return the per-layer unit directions plus diagnostics."""
    acts_harmful = last_token_activations(model, tokenizer, harmful)
    acts_harmless = last_token_activations(model, tokenizer, harmless)
    r_all = acts_harmful.mean(dim=1) - acts_harmless.mean(dim=1)   # [n_layers+1, hidden]

    directions, norms, resid_norms = {}, {}, {}
    for layer in config.STEER_LAYERS:
        r = r_all[layer + 1]
        norms[layer] = float(r.norm())
        directions[layer] = r / r.norm()
        resid_norms[layer] = float(acts_harmful[layer + 1].norm(dim=-1).mean())

    # Split-half reliability: if the direction is not stable across disjoint halves of
    # the extraction set, nothing downstream is meaningful.
    half = len(harmful) // 2
    rel = {}
    for layer in config.STEER_LAYERS:
        a = acts_harmful[layer + 1, :half].mean(0) - acts_harmless[layer + 1, :half].mean(0)
        b = acts_harmful[layer + 1, half:].mean(0) - acts_harmless[layer + 1, half:].mean(0)
        rel[layer] = float(torch.nn.functional.cosine_similarity(a, b, dim=0))

    # How much the direction rotates with depth -- the reason we extract per layer.
    unit_all = {l: (r_all[l + 1] / r_all[l + 1].norm()) for l in range(32)}
    rotation = {l: round(float(unit_all[20] @ unit_all[l]), 3) for l in config.STEER_LAYERS}

    return {
        "directions": directions,
        "r_raw_norm": norms,
        "resid_norm": resid_norms,
        "split_half_cos": rel,
        "cos_with_layer20": rotation,
        "r_all": r_all,
    }


def random_directions(hidden_size: int, seed: int = config.SEED) -> Dict[int, torch.Tensor]:
    """Norm-matched random unit directions, for the optional control condition."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for layer in config.STEER_LAYERS:
        v = torch.randn(hidden_size, generator=gen)
        out[layer] = v / v.norm()
    return out


def to_device(directions: Dict[int, torch.Tensor], device: str, dtype) -> Dict:
    return {l: v.to(device=device, dtype=dtype) for l, v in directions.items()}

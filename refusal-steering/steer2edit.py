"""Steer2Edit (Sun, Yan, Wang, Weng, arXiv:2602.09870): steering vectors -> rank-1 weight edits.

Ported from the official code (github.com/Trustworthy-ML-Lab/Steer2Edit,
safety_alignment/extract_directions.py + edit.py), using this repo's model loading, chat
template and data. No hooks at inference time: the edit is baked into the weights.

Components (each writes into the residual stream):
  attention head j at layer l : the column block W_o[:, j*D:(j+1)*D] of o_proj
  MLP neuron j at layer l     : column j of down_proj

Steering vectors, one per layer per block type (attn / mlp):
  v_l^b = mean(block output | refusal responses) - mean(block output | compliant responses)
averaged over RESPONSE tokens (official span [start-1, end-1)), then over samples.

Per component i with weight slice W_i and mean input mu_i:
  g_i     = cos(v, W_i mu_i)                                 importance
  lam_i   = sign(g_i) * max(|g_i| - rho*alpha, 0) / (rho*(1-alpha))   elastic-net closed form
  k_hat_i = W_i^T v / ||W_i^T v||                            input direction
  dW_i    = lam_i * v_hat k_hat_i^T
For an MLP neuron W_i is a single column, so k_hat_i = sign(w_i . v) and
dw_i = lam_i * sign(w_i . v) * v_hat (exactly as the official code stores it).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

import config
from model_utils import format_prompt


# ---------------------------------------------------------------- 1. block-output means
@torch.no_grad()
def collect_block_means(model, tokenizer, questions: List[str], responses: List[str],
                        layers: Iterable[int], batch_size: int = 8) -> Dict[str, Dict[int, torch.Tensor]]:
    """Mean (over response tokens, then over samples) of the inputs and outputs of o_proj
    and down_proj at each layer in `layers`. Returns fp32 CPU tensors:
      {"attn_pre": {l: [H]}, "attn_post": {l: [H]}, "mlp_pre": {l: [I]}, "mlp_post": {l: [H]}}

    Tokens predicting the response are averaged, matching the official span
    `[start-1 : end-1)` where start/end are the token lengths of prompt and prompt+response.
    Batched with left padding; each row's span is located from its own lengths.
    """
    assert len(questions) == len(responses) and len(questions) > 0
    assert tokenizer.padding_side == "left"
    layers = list(layers)
    blocks = {"attn": lambda l: model.model.layers[l].self_attn.o_proj,
              "mlp": lambda l: model.model.layers[l].mlp.down_proj}
    sums = {f"{b}_{w}": {} for b in blocks for w in ("pre", "post")}
    state: dict = {}

    def make_hook(block: str, layer: int):
        def hook(module, inputs, output):
            w = state["weights"]                                    # [B, S, 1] fp32
            for kind, t in (("pre", inputs[0]), ("post", output)):
                m = (t.float() * w).sum(dim=1).sum(dim=0)           # sum of per-row span means
                key = f"{block}_{kind}"
                sums[key][layer] = sums[key].get(layer, 0) + m.cpu()
        return hook

    handles = [blocks[b](l).register_forward_hook(make_hook(b, l)) for b in blocks for l in layers]
    try:
        for s in range(0, len(questions), batch_size):
            qs, rs = questions[s:s + batch_size], responses[s:s + batch_size]
            prompts = [format_prompt(tokenizer, q) for q in qs]
            fulls = [p + r for p, r in zip(prompts, rs)]
            p_len = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]
            enc = tokenizer(fulls, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            S = enc["input_ids"].shape[1]
            f_len = enc["attention_mask"].sum(dim=1).tolist()
            weights = torch.zeros(len(qs), S, 1)
            for i, (P, Fl) in enumerate(zip(p_len, f_len)):
                lo, hi = S - Fl + P - 1, S - 1                      # [start-1, end-1) after left pad
                if hi <= lo:
                    raise ValueError(f"empty response span for sample {s + i}")
                weights[i, lo:hi, 0] = 1.0 / (hi - lo)
            state["weights"] = weights.to(model.device)
            model(**enc)
    finally:
        for h in handles:
            h.remove()
    n = len(questions)
    return {k: {l: v / n for l, v in d.items()} for k, d in sums.items()}


def build_directions(pos: Dict, neg: Dict) -> Dict:
    """Official layout: directions['pre'|'post']['pos_mean'|'neg_mean'|'direction'][block][layer]."""
    out = {"pre": {}, "post": {}}
    for kind in ("pre", "post"):
        out[kind] = {"pos_mean": {}, "neg_mean": {}, "direction": {}}
        for block in ("attn", "mlp"):
            key = f"{block}_{kind}"
            out[kind]["pos_mean"][block] = pos[key]
            out[kind]["neg_mean"][block] = neg[key]
            out[kind]["direction"][block] = {l: pos[key][l] - neg[key][l] for l in pos[key]}
    return out


# ---------------------------------------------------------------- 2. the edit
def elastic_net_lambda(g: torch.Tensor, rho: float, alpha: float) -> torch.Tensor:
    """lam_i = sign(g_i) * max(|g_i| - rho*alpha, 0) / (rho*(1-alpha)). rho<=0 disables."""
    if rho is None or rho <= 0:
        return torch.zeros_like(g)
    assert 0 <= alpha < 1, "alpha must be in [0, 1)"
    return torch.sign(g) * torch.clamp(g.abs() - rho * alpha, min=0.0) / (rho * (1.0 - alpha))


@torch.no_grad()
def compute_edits(model, directions: Dict, rho_attn: float, rho_mlp: float, alpha: float,
                  layers: Optional[Iterable[int]] = None,
                  v_override: Optional[Dict[str, Dict[int, torch.Tensor]]] = None) -> Dict:
    """Compact rank-1 factors, same structure as the official edit pack:
      {"attn": {(l, head): {"lambda", "k_hat"}}, "mlp": {(l, neuron): {"lambda"}},
       "meta": {"vhat_attn": {l: [H]}, "vhat_mlp": {l: [H]}, ...}}
    `v_override[block][l]` replaces the steering vector (e.g. the residual-stream r_hat, or
    a random control). The importance score still uses the measured mean inputs mu_i.
    """
    cfg = model.config
    H, A = cfg.hidden_size, cfg.num_attention_heads
    D = getattr(cfg, "head_dim", None) or H // A
    layers = sorted(directions["post"]["direction"]["attn"]) if layers is None else list(layers)

    attn, mlp, vhat_attn, vhat_mlp = {}, {}, {}, {}
    for l in layers:
        v = {b: (v_override[b][l] if v_override else directions["post"]["direction"][b][l]).float()
             for b in ("attn", "mlp")}
        va, vm = F.normalize(v["attn"], dim=0), F.normalize(v["mlp"], dim=0)
        vhat_attn[l], vhat_mlp[l] = va.cpu(), vm.cpu()
        mu = {b: 0.5 * (directions["pre"]["pos_mean"][b][l] + directions["pre"]["neg_mean"][b][l]).float()
              for b in ("attn", "mlp")}

        if rho_attn and rho_attn > 0:
            Wo = model.model.layers[l].self_attn.o_proj.weight.detach().float().cpu().view(H, A, D)
            head_out = torch.einsum("had,ad->ha", Wo, mu["attn"].view(A, D))       # W_j mu_j
            g = F.cosine_similarity(va.unsqueeze(1), head_out, dim=0)              # [A]
            lam = elastic_net_lambda(g, rho_attn, alpha)
            k = torch.einsum("had,h->ad", Wo, va)                                  # W_j^T v
            for j in torch.nonzero(lam).flatten().tolist():
                kn = k[j].norm()
                if kn > 0:
                    attn[(l, j)] = {"lambda": float(lam[j]), "k_hat": (k[j] / kn).clone(),
                                    "g": float(g[j])}

        if rho_mlp and rho_mlp > 0:
            Wd = model.model.layers[l].mlp.down_proj.weight.detach().float().cpu()  # [H, I]
            g = F.cosine_similarity(vm.unsqueeze(1), Wd * mu["mlp"].unsqueeze(0), dim=0)  # [I]
            lam = elastic_net_lambda(g, rho_mlp, alpha)
            ks = Wd.T @ vm
            for j in torch.nonzero((lam != 0) & (ks != 0)).flatten().tolist():
                mlp[(l, j)] = {"lambda": float(lam[j]) * (1.0 if ks[j] > 0 else -1.0),
                               "g": float(g[j])}

    return {"attn": attn, "mlp": mlp,
            "meta": {"vhat_attn": vhat_attn, "vhat_mlp": vhat_mlp, "layers": layers,
                     "rho_attn": rho_attn, "rho_mlp": rho_mlp, "alpha": alpha,
                     "hidden_size": H, "num_heads": A, "head_dim": D}}


def dense_deltas(model, pack: Dict):
    """Yield ((l, "o_proj"|"down_proj"), dense fp32 dW) one matrix at a time, only for
    matrices with at least one edit -- so peak memory is one dW, not all of them."""
    D = pack["meta"]["head_dim"]
    by_layer: Dict = {}
    for (l, j), p in pack["attn"].items():
        by_layer.setdefault((l, "o_proj"), []).append((j, p))
    for (l, j), p in pack["mlp"].items():
        by_layer.setdefault((l, "down_proj"), []).append((j, p))
    for (l, name), items in sorted(by_layer.items()):
        d = torch.zeros(_module(model, l, name).weight.shape, dtype=torch.float32)
        if name == "o_proj":
            v = pack["meta"]["vhat_attn"][l]
            for j, p in items:
                d[:, j * D:(j + 1) * D] += p["lambda"] * torch.outer(v, p["k_hat"])
        else:
            idx = torch.tensor([j for j, _ in items])
            lam = torch.tensor([p["lambda"] for _, p in items], dtype=torch.float32)
            d[:, idx] = pack["meta"]["vhat_mlp"][l].unsqueeze(1) * lam.unsqueeze(0)
        yield (l, name), d


def _module(model, l: int, name: str):
    layer = model.model.layers[l]
    return layer.self_attn.o_proj if name == "o_proj" else layer.mlp.down_proj


@contextmanager
@torch.no_grad()
def applied(model, pack: Dict):
    """Apply the edits in place for the duration of the block, then restore the original
    weights bit-exactly from a CPU copy (subtracting in bf16 would not be exact)."""
    backup = {}
    try:
        for (l, name), d in dense_deltas(model, pack):
            W = _module(model, l, name).weight
            backup[(l, name)] = W.detach().cpu().clone()
            W.copy_((W.float() + d.to(W.device)).to(W.dtype))
        yield
    finally:
        for (l, name), W0 in backup.items():
            _module(model, l, name).weight.copy_(W0.to(_module(model, l, name).weight.device))


@torch.no_grad()
def fingerprint(model, layers: Iterable[int]) -> tuple:
    """Exact checksum of every editable weight, to prove `applied` restored them."""
    out = []
    for l in layers:
        for n in ("o_proj", "down_proj"):
            w = _module(model, l, n).weight.detach().cpu().double()   # MPS has no float64
            out.append(float(w.sum()) + float(w.abs().sum()))
    return tuple(out)


def edit_stats(model, pack: Dict) -> Dict:
    """How many components were touched, where, and how large the edit is."""
    A = pack["meta"]["num_heads"]
    I = model.config.intermediate_size
    layers = pack["meta"]["layers"]
    per_layer = {l: {"heads": 0, "neurons": 0} for l in layers}
    for (l, _) in pack["attn"]:
        per_layer[l]["heads"] += 1
    for (l, _) in pack["mlp"]:
        per_layer[l]["neurons"] += 1
    rel = []
    for (l, name), d in dense_deltas(model, pack):
        W = _module(model, l, name).weight.detach().float().cpu()
        rel.append(float(d.norm() / W.norm()))
    return {
        "n_heads": len(pack["attn"]), "frac_heads": len(pack["attn"]) / (A * len(layers)),
        "n_neurons": len(pack["mlp"]), "frac_neurons": len(pack["mlp"]) / (I * len(layers)),
        "max_rel_dW": max(rel) if rel else 0.0,
        "mean_abs_lambda_attn": (sum(abs(p["lambda"]) for p in pack["attn"].values()) / len(pack["attn"])
                                 if pack["attn"] else 0.0),
        "mean_abs_lambda_mlp": (sum(abs(p["lambda"]) for p in pack["mlp"].values()) / len(pack["mlp"])
                                if pack["mlp"] else 0.0),
        "per_layer": per_layer,
    }


def random_override(directions: Dict, seed: int = config.SEED) -> Dict:
    """CONTROL: random unit vectors in place of each block's steering vector."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for b in ("attn", "mlp"):
        out[b] = {}
        for l, v in sorted(directions["post"]["direction"][b].items()):
            r = torch.randn(v.shape[0], generator=gen)
            out[b][l] = r / r.norm()
    return out

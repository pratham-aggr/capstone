"""Forward hooks implementing the two interventions.

Convention: a hook on `model.model.layers[L]` fires on that layer's OUTPUT, which is
exactly the tensor the directions were extracted from (`hidden_states[L + 1]`).

  fixed addition : h <- h + alpha * r_hat_L
  CLAS           : h <- h + (c_L . [h, 1]) * r_hat_L

Both at all token positions, on the same layers, with the same directions. Plain
torch forward hooks, so the mechanism stays visible.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn as nn


def _split_output(out):
    """LlamaDecoderLayer returns a bare tensor on transformers >=4.54 but a
    (hidden_states, ...) tuple on older versions. A hook written for only one of these
    silently no-ops on the other, so handle both explicitly."""
    if isinstance(out, tuple):
        return out[0], out[1:]
    return out, None


def _rebuild_output(h: torch.Tensor, rest):
    return (h,) + rest if rest is not None else h


class ActivationCapture:
    """Record the residual stream at the last token position, for every layer."""

    def __init__(self) -> None:
        self.hidden_states: Optional[torch.Tensor] = None


@contextmanager
def fixed_addition(model, directions: Dict[int, torch.Tensor], alpha: float,
                   mask_holder: Optional[dict] = None):
    """Experiment 1. `directions` maps layer index -> unit direction on the model device."""
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer: int):
        def hook(module, inputs, out):
            h, rest = _split_output(out)
            delta = (alpha * directions[layer]).to(h.dtype).expand_as(h)
            delta = _mask_delta(delta, h, mask_holder)
            return _rebuild_output(h + delta, rest)
        return hook

    try:
        for layer in directions:
            block = model.model.layers[layer]
            handles.append(block.register_forward_hook(make_hook(layer)))
        yield
    finally:
        for handle in handles:
            handle.remove()


class ClasGates(nn.Module):
    """One sensing vector c_L in R^(d+1) per steered layer, zero-initialised so that
    training begins as an exact no-op. Held in fp32 while the model runs in bf16.

    c_L is stored as two tensors -- the vector part c_L[:-1] and the bias entry c_L[-1] --
    purely so they can sit in separate optimiser param groups. The CLAS paper gives the
    bias its own, much larger learning rate, and under Adam that cannot be emulated by
    rescaling the gradient (Adam's m/sqrt(v) is invariant to gradient scale).
    """

    def __init__(self, layers: List[int], hidden_size: int) -> None:
        super().__init__()
        self.layers = list(layers)
        self.vec = nn.ParameterDict(
            {str(l): nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
             for l in self.layers}
        )
        self.bias = nn.ParameterDict(
            {str(l): nn.Parameter(torch.zeros(1, dtype=torch.float32))
             for l in self.layers}
        )

    def gate(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        """g = c_L . [h, 1]; shape matches h without its last dim."""
        return h.float() @ self.vec[str(layer)] + self.bias[str(layer)]

    def vec_parameters(self):
        return list(self.vec.parameters())

    def bias_parameters(self):
        return list(self.bias.parameters())


@contextmanager
def clas_addition(model, directions: Dict[int, torch.Tensor], gates: ClasGates,
                  mask_holder: Optional[dict] = None,
                  trace: Optional[Dict[int, list]] = None,
                  max_abs_gate: Optional[float] = None):
    """Experiment 3. Differentiable w.r.t. `gates`; do NOT wrap in torch.no_grad when
    training. `trace`, if given, collects per-layer gate statistics."""
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer: int):
        def hook(module, inputs, out):
            h, rest = _split_output(out)
            g = gates.gate(layer, h)                       # [B, T] fp32
            if not torch.isfinite(g).all():
                raise FloatingPointError(f"non-finite CLAS gate at layer {layer}")
            # Compare the MEAN |gate| against the threshold: a single outlier position
            # (the attention sink) does not displace the stream the way a sustained gate
            # does, and the threshold is calibrated on sustained displacement.
            if max_abs_gate is not None and g.abs().mean() > max_abs_gate:
                raise FloatingPointError(
                    f"CLAS gate blew up at layer {layer}: mean|g|={g.abs().mean():.2f} "
                    f"> {max_abs_gate} (see config.CLAS_MAX_ABS_GATE)"
                )
            if trace is not None:
                trace.setdefault(layer, []).append(g.detach().float().cpu())
            delta = g.unsqueeze(-1) * directions[layer].float()
            delta = _mask_delta(delta, h, mask_holder)
            return _rebuild_output(h + delta.to(h.dtype), rest)
        return hook

    try:
        for layer in gates.layers:
            block = model.model.layers[layer]
            handles.append(block.register_forward_hook(make_hook(layer)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _mask_delta(delta: torch.Tensor, h: torch.Tensor, mask_holder: Optional[dict]):
    """Zero the steering delta at padding positions.

    Padded positions are already excluded by the attention mask, so this does not change
    the result -- but it makes outputs exactly invariant to batch composition, which
    matters when six conditions are compared on the same prompts. During incremental
    decoding h has length 1 and every position is real, so skip."""
    if mask_holder is None or h.shape[1] == 1:
        return delta
    mask = mask_holder.get("attention_mask")
    if mask is None or mask.shape[1] != h.shape[1]:
        return delta
    return delta * mask.unsqueeze(-1).to(delta.dtype)


@contextmanager
def no_steering():
    """Baseline condition: an explicit no-op, so all three conditions read alike."""
    yield

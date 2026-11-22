"""Quantisation-aware training: the forward pass sees exactly the values the Spectrum will see.

Every quantiser uses the straight-through estimator: the backward pass treats rounding as the identity.

- Weights: ternary per output row, w_q = clamp(round(w / s), -1, 1) * s with s = mean |w| of the row (as BitNet
  b1.58). On the Z80 the trits are packed five to a byte and s becomes the row's fixed-point scale.
- Two planes (the output heads): a second ternary matrix quantises what the first one missed; on the Z80 each plane
  is an ordinary ternary matrix with its own scale, and the two results are added.
- Matrix inputs: integers in [-m, m], m = 25 (five weights to a lookup page, so every page entry fits a signed byte).
- Table lookups: a pre-activation becomes an 8-bit table index, and the table's 8-bit output is rounded too.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

M = 25  # the input bound: 5 * 25 = 125 fits a signed byte


def ste(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return x + (q - x).detach()


def ternary(w: torch.Tensor) -> torch.Tensor:
    s = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
    return ste(w, (w / s).round().clamp(-1, 1) * s)


def plane_values(w: torch.Tensor, planes: int) -> list[torch.Tensor]:
    """The planes (detached): plane k quantises what the earlier planes missed."""
    out, q = [], torch.zeros_like(w)
    for _ in range(planes):
        r = (w - q).detach()
        s = r.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        p = (r / s).round().clamp(-1, 1) * s
        out.append(p)
        q = q + p
    return out


def ternary_planes(w: torch.Tensor, planes: int = 2) -> torch.Tensor:
    return ste(w, sum(plane_values(w, planes)))


def q_unit(x: torch.Tensor, m: int = M) -> torch.Tensor:
    """x in [-1, 1] to multiples of 1/m (the gradient passes inside the range, as a clipped straight-through)."""
    return torch.fake_quantize_per_tensor_affine(x, 1.0 / m, 0, -m, m)


def q_index(x: torch.Tensor, step: float) -> torch.Tensor:
    """An 8-bit table index: x rounded to multiples of `step`, clamped to -128..127 steps."""
    return torch.fake_quantize_per_tensor_affine(x, step, 0, -128, 127)


def q_out(x: torch.Tensor, levels: int, lo: int, hi: int) -> torch.Tensor:
    return torch.fake_quantize_per_tensor_affine(x, 1.0 / levels, 0, lo, hi)


def sigmoid_t(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid through a 256-entry table: index step 1/16 (range +-8), output 0..255 in 256ths (one byte)."""
    return q_out(torch.sigmoid(q_index(x, 1 / 16)), 256, 0, 255)


TANH_MODE = "sig"  # "sig": tanh read from the sigmoid table (one table fewer); "table": its own table


def tanh_t(x: torch.Tensor) -> torch.Tensor:
    """Tanh through a table, index step 1/32 (range +-4). "sig": the sigmoid table at the same index, tanh(x) =
    2 sigmoid(2x) - 1, output (SIG - 128) / 128. "table": its own table, output in 127ths."""
    if TANH_MODE == "sig":
        return 2 * q_out(torch.sigmoid(2 * q_index(x, 1 / 32)), 256, 0, 255) - 1
    return q_out(torch.tanh(q_index(x, 1 / 32)), 127, -127, 127)


class TLinear(nn.Module):
    """A linear layer with ternary weights ("ternary"), two ternary planes ("planes2") or float weights ("float");
    inputs are quantised by the caller."""

    def __init__(self, k: int, n: int, bias: bool = True, kind: str = "ternary"):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n, k) / k ** 0.5)
        self.b = nn.Parameter(torch.zeros(n)) if bias else None
        self.kind = kind

    def qweight(self) -> torch.Tensor:
        if self.kind == "ternary":
            return ternary(self.w)
        if self.kind.startswith("planes"):
            return ternary_planes(self.w, int(self.kind[6:]))
        if self.kind == "float":
            return self.w
        raise ValueError(self.kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.qweight(), self.b)

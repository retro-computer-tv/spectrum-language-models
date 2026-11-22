"""The lookup tables every network here uses, and the sampler, exactly as the Z80 computes them.

  SIG       256 x sigmoid(i / 16), i = -128..127, at most 255         the GRU's gates (256ths)
  TANH      127 x tanh(i / 32)                                       tanh, when a network has its own table
  EXP       255 x exp(-i / 16), i = 0..255 (zero from 100 on)        the sampler's weights; attention
  LOGSIG    16 x log sigmoid(i / 16), at least -147                  J's notebook: a note's write gate
  SOFTPLUS  16 x log(1 + exp(i / 16))                                J's notebook: how far to trust it

Sampling (the Z80's rule): greedy at temperature 0; otherwise Tinv = round(64 / temp), e = ((max - logit) * Tinv)
>> 6, w = EXP[e] for e < 256 else 0, w = 0 if w < minp; r = (rand16 * sum(w)) >> 16; the first symbol whose running
total of w exceeds r. rand16 is xorshift16 (7, 9, 8). `>>` is arithmetic (floor) throughout, as on the Z80.
"""
from __future__ import annotations

import math

import numpy as np

SIG = np.array([min(255, round(256 / (1 + math.exp(-i / 16)))) for i in range(-128, 128)], dtype=np.int64)
TANH = np.array([round(127 * math.tanh(i / 32)) for i in range(-128, 128)], dtype=np.int64)
EXP = np.array([round(255 * math.exp(-i / 16)) for i in range(256)], dtype=np.int64)
LOGSIG = np.array([max(-147, round(16 * math.log(1 / (1 + math.exp(-i / 16))))) for i in range(-128, 128)], np.int64)
SOFTPLUS = np.array([round(16 * math.log1p(math.exp(i / 16))) for i in range(-128, 128)], np.int64)


def xorshift16(x: int) -> int:
    x ^= (x << 7) & 0xFFFF
    x ^= x >> 9
    x ^= (x << 8) & 0xFFFF
    return x


def tinv_of(temp: float) -> int:
    """The sampler's inverse temperature in 64ths (0: greedy)."""
    return 0 if temp == 0 else round(64 / temp)


def sample(logits: np.ndarray, rng: int, tinv: int, minp: int) -> tuple[int, int]:
    """One symbol from integer logits, exactly as the Z80 picks it. Returns (symbol, new rng state)."""
    if tinv == 0:
        return int(np.argmax(logits)), rng
    e = ((logits.max() - logits) * tinv) >> 6
    w = np.where(e < 256, EXP[np.minimum(e, 255)], 0)
    w = np.where(w < minp, 0, w)
    rng = xorshift16(rng)
    r = (rng * int(w.sum())) >> 16
    return int(np.searchsorted(np.cumsum(w), r, side="right")), rng

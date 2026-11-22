"""The GRU the story writer and J share, written as the Z80 computes it (train/qat.py has the quantisers), and the
streaming that trains it.

A GRU with ternary weights and batch-normalised pre-activations, the fix for binary and ternary recurrent networks
of Ardakani et al. (ICLR 2019): W_x x and W_h h are normalised separately, row by row. At inference each
normalisation is a scale and a bias per row, which the Z80 applies anyway (folded into M, sh and B at export), so it
costs nothing at run time beyond keeping the x and h parts of r and z in separate accumulators.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import qat
from train.qat import M, TLinear, q_unit, sigmoid_t, tanh_t

VOCABS = {"plain": 37, "slots": 41, "chat": 43}  # the story alphabet; with cast slots; with J's two turn markers


class GRUBN(nn.Module):
    def __init__(self, d: int, e: int = 48, vocab: str = "slots", head: str = "planes2", tanh: str = "sig",
                 m: int = M, momentum: float = 0.01):
        super().__init__()
        qat.TANH_MODE = tanh
        V = VOCABS[vocab]
        self.d, self.e, self.m, self.vocab, self.tanh = d, e, m, vocab, tanh
        self.emb = nn.Parameter(torch.randn(V, e) * 0.5)
        self.wx = TLinear(e, 3 * d, bias=False)  # r, z, n from the input
        self.wh = TLinear(d, 3 * d, bias=False)  # r, z, n from the state
        self.bnx = nn.BatchNorm1d(3 * d, momentum=momentum)
        self.bnh = nn.BatchNorm1d(3 * d, momentum=momentum)
        with torch.no_grad():
            self.bnx.weight.fill_(0.5)
            self.bnh.weight.fill_(0.5)
        self.head = TLinear(d, V, kind=head)

    def init_state(self, b: int, device) -> list:
        return [torch.zeros(b, self.d, device=device)]

    def cell(self, xa: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        d = self.d
        ah = self.bnh(q_unit(h, self.m) @ self.wh.qweight().t())
        r, z = sigmoid_t(xa[:, : 2 * d] + ah[:, : 2 * d]).chunk(2, dim=-1)
        n = tanh_t(xa[:, 2 * d:] + r * ah[:, 2 * d:])
        return torch.lerp(n, h, z)

    def forward(self, idx: torch.Tensor, state: list):
        B, L = idx.shape
        d = self.d
        x = q_unit(self.emb, self.m)[idx]
        ax = self.bnx(self.wx(x).reshape(B * L, 3 * d)).reshape(B, L, 3 * d)
        h = state[0]
        outs = []
        for t in range(L):
            h = self.cell(ax[:, t], h)
            outs.append(h)
        x = torch.stack(outs, 1)
        return self.head(q_unit(x, self.m)), [h]


def detach(state):
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, (list, tuple)):
        return type(state)(detach(s) for s in state)
    return state


class Streams:
    """B parallel streams through a file of symbols, each from its own offset, advancing L symbols a step; the
    network's state is carried from one step to the next and detached."""

    def __init__(self, path: Path, b: int, l: int, seed: int):
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        n = len(self.data)
        rnd = np.random.default_rng(seed)
        self.pos = (np.arange(b) * (n // b) + rnd.integers(0, n // b, b)) % (n - l - 2)
        self.l = l

    def next(self, device):
        x = np.stack([self.data[p: p + self.l + 1] for p in self.pos]).astype(np.int64)
        self.pos = (self.pos + self.l) % (len(self.data) - self.l - 2)
        return (torch.from_numpy(np.ascontiguousarray(x[:, :-1])).to(device, non_blocking=True),
                torch.from_numpy(np.ascontiguousarray(x[:, 1:])).to(device, non_blocking=True))


@torch.no_grad()
def evaluate(model, path: Path, symbols: float, device, b: int = 64, l: int = 256, warm: int = 512,
             per_char: float = 1.0) -> float:
    """Bits per character over `symbols` symbols of a split, in b evenly spaced streams; the first `warm` symbols of
    each stream only build up state. per_char: symbols per original character (cast slots change the length)."""
    was_training = model.training
    model.eval()
    data = np.memmap(path, dtype=np.uint8, mode="r")
    n = len(data)
    per = int(symbols) // b
    starts = np.arange(b) * (n // b)
    state = model.init_state(b, device)
    tot, cnt = 0.0, 0
    for off in range(0, warm + per, l):
        x = np.stack([data[s + off: s + off + l + 1] for s in starts]).astype(np.int64)
        t = torch.from_numpy(x).to(device)
        logits, state = model(t[:, :-1], state)
        if off >= warm:
            tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), t[:, 1:].reshape(-1), reduction="sum").item()
            cnt += t[:, 1:].numel()
    model.train(was_training)
    return tot / cnt / math.log(2) * per_char


def load_qat(path: Path, cls=GRUBN) -> nn.Module:
    """A network from a QAT checkpoint (weights/qat/*.npz, or what train/*/train.py writes), in evaluation mode."""
    import json  # noqa: PLC0415

    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    model = cls(**{k: meta[k] for k in meta if k in cls.ARGS})
    model.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files if k != "meta"})
    model.eval()
    return model


GRUBN.ARGS = ("d", "e", "vocab", "head", "tanh")

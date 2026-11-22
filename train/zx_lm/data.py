"""ZX-LM conversations as steps: what the network reads, writes and is scored on, step by step.

A user turn is <u> and its words. A Spectrum turn is <s>, N_THINK think steps, then its words: the last think step
predicts the first word, each word the next. A turn's last step writes the turn into memory. References are stored
by order of first appearance (local numbers) and given random reference numbers each time a batch is made.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from train.zx_lm.vocab import Vocab, words

N_THINK = 2
MAX_SLOTS = 64
NO_TARGET = -10_000


@dataclass
class Script:
    x: list[int]  # word index, or -1 - local reference
    tgt: list[int]  # output word index, -1 - local reference, or NO_TARGET
    w: list[float]  # loss weight of the target
    end: list[int]  # 1: the turn ends after this step
    user: list[int]  # 1: a user-turn step (the reader's, when reading and writing are split)
    ev: list[int]  # question number whose evidence this step's read is supervised toward, or -1
    supports: list[list[int]]  # per question: the turns its answer rests on ([]: nothing does)
    n_local: int
    answers: list[int]  # per Spectrum answer to a question: the conversation turn it answers


def script(vocab: Vocab, turns: list[dict], ack_weight: float = 0.1) -> Script:
    local: dict[str, int] = {}

    def tok(w: str) -> int:
        if w in vocab.index:
            return vocab.index[w]
        local.setdefault(w, len(local))
        return -1 - local[w]

    def out(w: str) -> int:
        if w in vocab.out_index:
            return vocab.out_index[w]
        local.setdefault(w, len(local))
        return -1 - local[w]

    s = Script([], [], [], [], [], [], [], 0, [])
    for i, t in enumerate(turns):
        ws = words(t["text"])
        if t["speaker"] == "u":
            xs = [vocab.index["<u>"]] + [tok(w) for w in ws]
            for j, x in enumerate(xs):
                s.x.append(x)
                s.tgt.append(NO_TARGET)
                s.w.append(0.0)
                s.end.append(int(j == len(xs) - 1))
                s.user.append(1)
                s.ev.append(-1)
            continue
        q = -1
        if not t.get("ack"):
            q = len(s.supports)
            s.supports.append(list(turns[t["q_turn"]].get("support", [])))
            s.answers.append(i)
        wt = ack_weight if t.get("ack") else 1.0
        xs = [vocab.index["<s>"]] + [vocab.index["<think>"]] * N_THINK + [tok(w) for w in ws]
        ys = [NO_TARGET] * N_THINK + [out(w) for w in ws] + [NO_TARGET]
        for j, (x, y) in enumerate(zip(xs, ys)):
            s.x.append(x)
            s.tgt.append(y)
            s.w.append(wt if y != NO_TARGET else 0.0)
            s.end.append(int(j == len(xs) - 1))
            s.user.append(0)
            s.ev.append(q if 1 <= j <= N_THINK else -1)
    s.n_local = len(local)
    return s


def mention_count(s: Script) -> int:
    return sum(x < 0 for x in s.x)


class Batch:
    """Scripts padded to fixed shapes (T steps, N answer targets, E evidence reads, n_turns turns), references
    numbered at random. Fixed shapes let a training step be captured once as a CUDA graph and replayed."""

    def __init__(self, vocab: Vocab, scripts: list[Script], rng: random.Random, T: int = 300, N: int | None = None,
                 E: int | None = None, n_turns: int = 64):
        B = len(scripts)
        N, E = N or 40 * B, E or 14 * B
        a = {"x": np.zeros((B, T), np.int64), "ref": np.full((B, T), -1, np.int64),
             "end": np.zeros((B, T), np.bool_), "user": np.zeros((B, T), np.bool_),
             "active": np.zeros((B, T), np.bool_),
             "tt": np.zeros(N, np.int64), "tb": np.zeros(N, np.int64), "ty": np.zeros(N, np.int64),
             "tw": np.zeros(N, np.float32), "et": np.zeros(E, np.int64), "eb": np.zeros(E, np.int64),
             "esup": np.zeros((E, n_turns), np.bool_), "ew": np.zeros(E, np.float32)}
        n = e = 0
        for b, s in enumerate(scripts):
            assert len(s.x) <= T and sum(s.end) <= n_turns, (len(s.x), sum(s.end))
            perm = rng.sample(range(16), s.n_local)
            for t, xv in enumerate(s.x):
                if xv < 0:
                    a["ref"][b, t] = perm[-1 - xv]
                    a["x"][b, t] = vocab.ref(perm[-1 - xv])
                else:
                    a["x"][b, t] = xv
                y = s.tgt[t]
                if y != NO_TARGET:
                    assert n < N, "more answer targets than the batch holds"
                    a["tt"][n], a["tb"][n], a["tw"][n] = t, b, s.w[t]
                    a["ty"][n] = vocab.n_out + perm[-1 - y] if y < 0 else y
                    n += 1
                if s.ev[t] >= 0:
                    assert e < E, "more evidence reads than the batch holds"
                    a["et"][e], a["eb"][e], a["ew"][e] = t, b, 1.0
                    for tt in s.supports[s.ev[t]]:
                        a["esup"][e, tt] = True
                    e += 1
            L = len(s.x)
            a["end"][b, :L], a["user"][b, :L], a["active"][b, :L] = s.end, s.user, True
        self.arrays, self.B, self.T = a, B, T

    def to(self, device: str, into: dict | None = None) -> "Batch":
        """Tensors on the device; with into, copied into those (a captured graph's static inputs)."""
        for k, v in self.arrays.items():
            t = torch.from_numpy(v)
            if into is None:
                setattr(self, k, t.to(device))
            else:
                into[k].copy_(t, non_blocking=True)
                setattr(self, k, into[k])
        return self

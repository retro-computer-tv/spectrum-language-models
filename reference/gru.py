"""The story writer's network (and J's, underneath its notebook) as the Spectrum computes it: integers only.

    python -m reference.gru weights/story-writer.npz "once upon a time, there was a little girl named Zara. "

prints what the Spectrum writes after that opening (the film's take: seed 1, temperature 0.7, 280 symbols).

A GRU with batch-normalised gates, its matrices ternary. One step, for input symbol c and state h (int16, 1.0 = 16384):

  hq     = clamp((h * m + 8192) >> 14, -m, m)               the state as int8, the lookup-table kernel's inputs
  X      = ((Wx emb[c]) * Mx >> shx) + Bx                    r, z and n rows, each in its table's units
  H      = ((Wh hq) * Mh >> shh) + Bh
  r, z   = SIG[clamp(X_r + H_r)], SIG[clamp(X_z + H_z)]      0..255 (256ths); the sigmoid's index step is 1/16
  n14    = (SIG[clamp(X_n + ((H_n * r) >> 8))] - 128) * 128  tanh from the sigmoid table, on the state's scale
  h'     = n14 + ((z * (h - n14)) >> 8)
  logit  = sum over the head's two planes of ((Wp hq) * Mp >> shp), + B     units of 1/16 nat

`>>` is arithmetic (floor), as on the Z80; clamp is to -128..127 for table indices.

Generation (the Z80's protocol): state 0, feed '#' then the opening, then repeatedly sample from the logits (with the
cast slots' rule: impossible symbols get the lowest logit) and feed the sample back, stopping at '#' or after n.
"""
from __future__ import annotations

import argparse

import numpy as np

from reference import text
from reference.tables import SIG, TANH, sample, tinv_of
from reference.weights import load


def clamp8(x):
    return np.clip(x, -128, 127)


def scaled(acc, ex, name):
    return ((acc * ex[f"{name}_M"]) >> ex[f"{name}_sh"]) + ex[f"{name}_B"]


def tanh14(idx, ex):
    """tanh on the state's scale (1.0 = 16384): the sigmoid table less 128, times 128, or its own table times 129."""
    if ex.get("tanh") == "sig":
        return (SIG[idx + 128] - 128) * 128
    return TANH[idx + 128] * 129


def head_logits(hq, ex) -> np.ndarray:
    out = scaled(hq @ ex["head_T"].T, ex, "head")
    for k in range(2, int(ex.get("head_planes", 1)) + 1):
        out = out + ((hq @ ex[f"head{k}_T"].T * ex[f"head{k}_M"]) >> ex[f"head{k}_sh"])
    return out


class IntGRUBN:
    def __init__(self, ex: dict):
        self.ex = ex
        self.m, self.d, self.e = ex["m"], ex["d"], ex["e"]

    def init_state(self, b: int) -> np.ndarray:
        return np.zeros((b, self.d), dtype=np.int64)

    def hq(self, h):
        return np.clip((h * self.m + 8192) >> 14, -self.m, self.m)

    def logits(self, h) -> np.ndarray:
        return head_logits(self.hq(h), self.ex)

    def step(self, c: np.ndarray, h: np.ndarray) -> np.ndarray:
        ex, d = self.ex, self.d
        x = ex["emb"][c]
        X = scaled(x @ ex["x_T"].T, ex, "x")
        H = scaled(self.hq(h) @ ex["h_T"].T, ex, "h")
        rz = SIG[clamp8(X[:, : 2 * d] + H[:, : 2 * d]) + 128]
        r, z = rz[:, :d], rz[:, d:]
        n14 = tanh14(clamp8(X[:, 2 * d:] + ((H[:, 2 * d:] * r) >> 8)), ex)
        return n14 + ((z * (h - n14)) >> 8)


def prompt_symbols(ex: dict, prompt: str) -> list[int]:
    """The opening's symbols as the Spectrum feeds them (after its '#')."""
    if ex.get("vocab") == "slots":
        return [text.SSTOI[c] for c in text.encode_typed(prompt)]
    return [text.STOI[c] for c in prompt.lower()]


def display(ex: dict, syms: list[int]) -> str:
    """Symbols as the Spectrum prints them."""
    if ex.get("vocab") == "slots":
        d = text.Decoder(caps=True)
        return "".join(d.push(text.SVOCAB[c]) for c in syms)
    return "".join(text.VOCAB[c] for c in syms)


def generate_syms(ex: dict, prompt: str, seed: int, temp: float, minp: int, n: int) -> tuple[list, list]:
    """(opening symbols, written symbols) for an opening as typed."""
    net = IntGRUBN(ex)
    ps = prompt_symbols(ex, prompt)
    h = net.init_state(1)
    for c in [0] + ps:
        h = net.step(np.array([c]), h)
    dec = None
    if ex.get("vocab") == "slots":
        dec = text.Decoder()
        for c in ps:
            dec.push(text.SVOCAB[c])
    out, rng, ti = [], seed, tinv_of(temp)
    for _ in range(n):
        lg = net.logits(h)[0]
        if dec is not None:
            lg = lg.copy()
            lg[dec.banned()] = -32768
        c, rng = sample(lg, rng, ti, minp)
        if c == 0:
            break
        out.append(c)
        if dec is not None:
            dec.push(text.SVOCAB[c])
        h = net.step(np.array([c]), h)
    return ps, out


def generate(ex: dict, prompt: str, seed: int, temp: float = 0.7, minp: int = 13, n: int = 280) -> str:
    """The text the Spectrum prints after the opening."""
    ps, out = generate_syms(ex, prompt, seed, temp, minp, n)
    return display(ex, ps + out)[len(display(ex, ps)):]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("weights")
    ap.add_argument("opening")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--minp", type=int, default=13)
    ap.add_argument("--n", type=int, default=280)
    a = ap.parse_args()
    print(a.opening + generate(load(a.weights), a.opening, a.seed, a.temp, a.minp, a.n))


if __name__ == "__main__":
    main()

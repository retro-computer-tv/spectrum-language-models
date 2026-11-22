"""ZX-LM, who owns what, as the Spectrum computes it: integers only, word by word.

    python -m reference.zx_lm weights/zx-lm.npz "Ana has a kite.|Ana gave the kite to Tom.|who has the kite?"

prints the Spectrum's reply to each line, exactly as the tape writes them (seed 1). An empty line on the Spectrum
starts a new conversation; here, each run is one conversation.

Words: lower case, "'s" split off, punctuation as words. A word in the vocabulary is its own token; any other word (a
name, an object, an adjective) becomes one of 16 reference numbers, the same number for the same spelling. Which
number a new word gets is chosen at random (xorshift16 & 15, or the next free one), so the network can't learn
anything from it. The Spectrum keeps each reference's spelling and prints it when the network points at it.

One step for word x:

  hq      = clamp((h * 25 + 8192) >> 14, -25, 25)                  the state as the lookup-table kernel's inputs
  P       = ((hq @ T_proj) * M >> sh) + B                            108 rows, each in its own units
  attention, from the previous step's P (q, rec) and the memory:
    s     = ((q . k) >> 7) + ((rec * min(age, 15)) >> 2); "nothing": ((q . k0) >> 7) + b0
    w     = EXP[max - s] (0 from 100); alpha = (w << 8) // sum(w)
    read  = ((sum alpha k) >> 8) -> (r * 51 + 128) >> 8;  ((sum alpha e[ref]) >> 8)
  x       = [emb[x], read]                                           64 numbers in [-25, 25]
  h'      = the batch-normalised GRU (as reference/gru.py, with its own tanh table)
  mention: pre = key_at rows of P(h');  turn end: g = SIG[clamp((k . z) >> 7)], t = TANH[clamp((k + c) >> 2)],
           k += (g * (t - k)) >> 8 for every older slot; new keys TANH[clamp(pre + key_end)]
  answer: gen >= 0 -> argmax head (first of equals); else the reference whose slots hold the most alpha (first)

A user turn is <u> and its words; a Spectrum turn is <s>, two think steps (reads with no word), then its words.
"""
from __future__ import annotations

import argparse
import re

import numpy as np

from reference.tables import EXP, SIG, TANH, xorshift16
from reference.weights import load

N_REFS = 16
MAX_SLOTS = 64
K = 16
GROUPS = [("q", K), ("rec", 1), ("key_at", K), ("key_end", K), ("upd_c", K), ("upd_z", K), ("gen", 1)]
WORD_RE = re.compile(r"[a-z]+n't|[a-z]+|'s|[.,?:!]")


def words(text: str) -> list[str]:
    """Lower case, "'s" split off, punctuation as words."""
    return WORD_RE.findall(text.lower().replace("\u2019", "'"))


class Vocab:
    """The network's words, from its weights file: what it reads (words) and what it can write itself (out_words)."""

    def __init__(self, ex: dict):
        self.words, self.out_words = list(ex["words"]), list(ex["out_words"])
        self.index = {w: i for i, w in enumerate(self.words)}
        self.n_words, self.n_out = len(self.words), len(self.out_words)


def clamp8(x):
    return np.clip(x, -128, 127)


def scaled(acc, ex, name):
    return ((acc * ex[f"{name}_M"]) >> ex[f"{name}_sh"]) + ex[f"{name}_B"]


class IntMem:
    def __init__(self, B: int):
        z = lambda *s: np.zeros(s, dtype=np.int64)  # noqa: E731
        self.keys, self.pre = z(B, MAX_SLOTS, K), z(B, MAX_SLOTS, K)
        self.valid, self.pending = z(B, MAX_SLOTS).astype(bool), z(B, MAX_SLOTS).astype(bool)
        self.ref, self.birth, self.turn, self.n = z(B, MAX_SLOTS), z(B, MAX_SLOTS), z(B), z(B)


class IntZX:
    def __init__(self, ex: dict):
        self.ex, self.m, self.d = ex, int(ex["m"]), int(ex["d"])
        self.sl, i = {}, 0
        for name, n in GROUPS + [("head", int(ex["n_out"]))]:
            self.sl[name] = slice(i, i + n)
            i += n

    def hq(self, h):
        return np.clip((h * self.m + 8192) >> 14, -self.m, self.m)

    def project(self, h) -> dict:
        r = scaled(self.hq(h) @ self.ex["p_T"].T, self.ex, "p")
        g = {name: r[:, s] for name, s in self.sl.items()}
        return {"q": np.clip(g["q"], -16, 16), "rec": np.clip(g["rec"][:, 0], -128, 127), "key_at": g["key_at"],
                "key_end": g["key_end"], "c": TANH[clamp8(g["upd_c"]) + 128], "z": np.clip(g["upd_z"], -16, 16),
                "gen": g["gen"][:, 0], "head": g["head"]}

    def start(self, B: int):
        h = np.zeros((B, self.d), dtype=np.int64)
        return h, IntMem(B), self.project(h)

    def alpha(self, P: dict, mem: IntMem) -> np.ndarray:
        """[B, 1 + slots] in 256ths."""
        age = np.minimum(mem.turn[:, None] - mem.birth, 15)
        s = (np.einsum("bmk,bk->bm", mem.keys, P["q"]) >> 7) + ((P["rec"][:, None] * age) >> 2)
        s0 = ((P["q"] @ self.ex["null_k"]) >> 7) + self.ex["null_b"]
        s = np.concatenate([s0[:, None], np.where(mem.valid, s, -(1 << 30))], 1)
        e = s.max(1, keepdims=True) - s
        w = np.where(e < 100, EXP[np.minimum(e, 255)], 0)
        return (w << 8) // w.sum(1, keepdims=True)

    def read(self, a: np.ndarray, mem: IntMem) -> np.ndarray:
        a = a[:, 1:, None]
        r127 = (a * mem.keys).sum(1) >> 8
        e25 = (a * self.ex["eref"][mem.ref]).sum(1) >> 8
        return np.concatenate([(r127 * 51 + 128) >> 8, e25], 1)

    def cell(self, x, h):
        ex, d = self.ex, self.d
        X = scaled(x @ ex["x_T"].T, ex, "x")
        Hh = scaled(self.hq(h) @ ex["h_T"].T, ex, "h")
        rz = SIG[clamp8(X[:, : 2 * d] + Hh[:, : 2 * d]) + 128]
        r, z = rz[:, :d], rz[:, d:]
        n14 = TANH[clamp8(X[:, 2 * d:] + ((Hh[:, 2 * d:] * r) >> 8)) + 128] * 129
        return n14 + ((z * (h - n14)) >> 8)

    def step(self, x, ref, active, h, mem: IntMem, P: dict):
        a = self.alpha(P, mem)
        inp = np.concatenate([self.ex["emb"][x], self.read(a, mem)], 1)
        h = np.where(active[:, None], self.cell(inp, h), h)
        Pn = self.project(h)
        P = {k: np.where(active.reshape((-1,) + (1,) * (v.ndim - 1)), v, P[k]) for k, v in Pn.items()}
        is_ref = (ref >= 0) & active
        slot = ((mem.n % MAX_SLOTS)[:, None] == np.arange(MAX_SLOTS)) & is_ref[:, None]
        mem.pre = np.where(slot[..., None], P["key_at"][:, None, :], mem.pre)
        mem.ref = np.where(slot, np.maximum(ref, 0)[:, None], mem.ref)
        mem.valid &= ~slot
        mem.pending |= slot
        mem.n = mem.n + is_ref
        return a, h, P

    def write(self, end, P: dict, mem: IntMem) -> None:
        e = end[:, None]
        g = SIG[clamp8(np.einsum("bmk,bk->bm", mem.keys, P["z"]) >> 7) + 128][..., None]
        t = TANH[clamp8((mem.keys + P["c"][:, None]) >> 2) + 128]
        upd = mem.keys + ((g * (t - mem.keys)) >> 8)
        new = TANH[clamp8(mem.pre + P["key_end"][:, None]) + 128]
        keys = np.where((e & mem.valid)[..., None], upd, mem.keys)
        fresh = e & mem.pending
        mem.keys = np.where(fresh[..., None], new, keys)
        mem.birth = np.where(fresh, mem.turn[:, None], mem.birth)
        mem.valid |= fresh
        mem.pending &= ~e
        mem.turn = mem.turn + end

    def choose(self, P: dict, mem: IntMem) -> np.ndarray:
        a = self.alpha(P, mem)[:, 1:] * mem.valid
        refp = np.zeros((a.shape[0], N_REFS), dtype=np.int64)
        np.add.at(refp, (np.arange(a.shape[0])[:, None].repeat(MAX_SLOTS, 1), mem.ref), a)
        return np.where(P["gen"] >= 0, P["head"].argmax(1), int(self.ex["n_out"]) + refp.argmax(1))


class Z80Refs:
    """The Spectrum's reference numbers: a new word gets xorshift16 & 15, or the next free one after it (all 16 in
    use: that one is replaced). Spellings kept to 15 letters."""

    def __init__(self, seed: int):
        self.seed, self.spelling = seed, {}

    def get(self, w: str) -> int:
        w = w[:15]
        for r, s in self.spelling.items():
            if s == w:
                return r
        self.seed = xorshift16(self.seed)
        c = self.seed & 15
        for _ in range(16):
            if c not in self.spelling:
                break
            c = (c + 1) & 15
        self.spelling[c] = w
        return c


def converse(ex: dict, lines: list[str], seed: int = 1, max_reply: int = 14) -> list[str]:
    """The Spectrum's replies to these lines, one conversation, exactly as the tape writes them."""
    v = Vocab(ex)
    net, refs = IntZX(ex), Z80Refs(seed)
    h, mem, P = net.start(1)
    one = np.ones(1, dtype=bool)

    def feed(tok: int, end: bool):
        nonlocal h, P
        ref = np.array([tok - v.n_words if tok >= v.n_words else -1])
        _, h, P = net.step(np.array([tok]), ref, one, h, mem, P)
        if end:
            net.write(one, P, mem)

    out = []
    for line in lines:
        toks = [v.index[w] if w in v.index else v.n_words + refs.get(w) for w in words(line)][:60]
        seq = [v.index["<u>"]] + toks
        for i, t in enumerate(seq):
            feed(t, i == len(seq) - 1)
        for t in (v.index["<s>"], v.index["<think>"], v.index["<think>"]):
            feed(t, False)
        reply = []
        while True:
            c = int(net.choose(P, mem)[0])
            tok = v.index[v.out_words[c]] if c < v.n_out else v.n_words + c - v.n_out
            reply.append(tok)
            last = tok == v.index["."] or len(reply) >= max_reply
            feed(tok, last)
            if last:
                break
        said = [v.words[t] if t < v.n_words else refs.spelling[t - v.n_words] for t in reply]
        out.append(" ".join(said).replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" :", ":")
                   .replace(" !", "!").replace(" '", "'"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("weights")
    ap.add_argument("lines", help="your lines, separated by |")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    lines = a.lines.split("|")
    for you, said in zip(lines, converse(load(a.weights), lines, a.seed)):
        print(f"{you}\n  {said}")


if __name__ == "__main__":
    main()

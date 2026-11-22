"""J, the conversation network, as the Spectrum computes it: the story writer's GRU plus a notebook of 64 words.

    python -m reference.notebook weights/j-dg1.npz "hello! my name is Ottoline.|i have a little cat called Pip."

prints the conversation the Spectrum would hold with those lines (seed 1, temperature 0.6; --margin 1 for the
margin-1 builds). Capitals only for names: a capital makes a word a name, as in training.

The GRU and its head are reference/gru.py's. The notebook adds, per symbol, with hq the state as int8 in [-m, m] and
hp the previous symbol's:

  K      = clamp(((Wk hp) * Mk + half) >> shk, -m, m)    16 numbers, the key of a word starting here
  Q      = clamp(((Wq hq) * Mq + half) >> shq, -m, m)    16 numbers, the query
  gi     = clamp(((wg . hp) * Mg >> shg) + Bg)           the write gate's index (step 1/16)
  bi     = clamp(((wc . hq) * Mc >> shc) + Bc)           the trust's index (step 1/16)
  score  = ((Q . K) * A >> shA) + LOGSIG[gi]             1/16 nat
  w      = EXP[min(255, best - score)]                   over the notes that fit what the reply has of its word
           (0 when best - score >= cut: 100 by default, EXP's own zero; 17 for "margin 1", the notes within a nat)
  bonus  = (SOFTPLUS[bi] * w) / sum(w)                   1/16 nat, added to the logit of the letter each note proposes

Every word you type is noted (key and gate from the state before its first letter, its letters as they come, a
name's first spelling as its slot digit); the last 64 are kept.
"""
from __future__ import annotations

import argparse

import numpy as np

from reference import text
from reference.gru import IntGRUBN
from reference.tables import EXP, LOGSIG, SOFTPLUS, sample, tinv_of
from reference.weights import load

W = 64  # notes kept: the last 64 words you typed
MAXREPLY = 120  # a reply longer than this is ended for it
MARGIN1_CUT = 17


def wordchar(c: int) -> bool:
    return 11 <= c <= 40  # letters, '*' and the slot digits; the apostrophe is not part of a word


def rng_start(seed: int) -> int:
    """The sampler's state at the start of a conversation."""
    return (seed * 7919 + 1) & 0xFFFF or 1


def _round_shift(acc, M, sh):
    return (acc * M + (1 << (sh - 1))) >> sh


class IntNotebook:
    def __init__(self, ex: dict, cut: int | None = None):
        self.ex, self.gru = ex, IntGRUBN(ex)
        self.m = ex["m"]
        self.cut = int(cut if cut is not None else np.asarray(ex.get("nb_cut", 100)).reshape(-1)[0])

    def init_state(self, b: int) -> dict:
        return {"h": self.gru.init_state(b), "hp": np.zeros((b, self.gru.d), np.int64), "lg": None,
                "rows": [{"cells": [], "typing": False, "prevw": False, "stars": 0, "partial": []} for _ in range(b)]}

    def step(self, c, st) -> dict:
        ex, m = self.ex, self.m
        c = np.asarray(c, dtype=np.int64)
        h = self.gru.step(c, st["h"])
        hq, hp = self.gru.hq(h), st["hp"]
        K = np.clip(_round_shift(hp @ ex["k_T"].T, ex["k_M"], ex["k_sh"]), -m, m)
        Q = np.clip(_round_shift(hq @ ex["q_T"].T, ex["q_M"], ex["q_sh"]), -m, m)
        gi = np.clip(((hp @ ex["g_T"].T * ex["g_M"]) >> ex["g_sh"]) + ex["g_B"], -128, 127)[:, 0]
        bi = np.clip(((hq @ ex["c_T"].T * ex["c_M"]) >> ex["c_sh"]) + ex["c_B"], -128, 127)[:, 0]
        lg = self.gru.logits(h).copy()
        rows = [dict(r, cells=[dict(x, seq=list(x["seq"])) for x in r["cells"]], partial=list(r["partial"]))
                for r in st["rows"]]
        for b, r in enumerate(rows):
            s = int(c[b])
            if s == 0:
                r.update(cells=[], typing=True, prevw=False, stars=0, partial=[])
                continue
            word = wordchar(s)
            if s == 37:
                r["stars"] += 1
            last = r["cells"][-1] if r["cells"] and r["cells"][-1]["open"] else None
            if last is not None and not (r["typing"] and word):
                last["open"], last = False, None
            if r["typing"] and word and not r["prevw"]:
                r["cells"].append({"key": K[b], "lsig": int(LOGSIG[gi[b] + 128]), "open": s != 37,
                                   "seq": [37 + r["stars"]] if s == 37 else [s]})
                r["cells"] = r["cells"][-W:]
            elif last is not None and len(last["seq"]) < 15:
                last["seq"].append(s)
            r["prevw"] = r["typing"] and word
            r["partial"] = (r["partial"] + [s])[-15:] if word else []
            if s == text.YOU_END:
                r["typing"] = False
            elif s == text.SPEC_END:
                r["typing"] = True
            o = len(r["partial"])
            fits = [x for x in r["cells"] if not x["open"] and o < len(x["seq"]) and x["seq"][:o] == r["partial"]]
            if not fits:
                continue
            sc = np.array([((int(Q[b] @ x["key"]) * int(ex["att_M"][0])) >> int(ex["att_sh"][0])) + x["lsig"]
                           for x in fits])
            d = sc.max() - sc
            w = np.where(d < self.cut, EXP[np.minimum(255, d)], 0)
            beta = int(SOFTPLUS[bi[b] + 128])
            tot = int(w.sum())
            for x, wi in zip(fits, w):
                lg[b, x["seq"][o]] += (beta * int(wi)) // tot
        return {"h": h, "hp": hq, "lg": lg, "rows": rows}

    def logits(self, st) -> np.ndarray:
        return st["lg"]


def talk(ex: dict, lines: list[str], seed: int = 1, temp: float = 0.6, minp: int = 13,
         cut: int | None = None) -> list[str]:
    """One conversation, exactly as the Spectrum holds it: J's reply to each line, as it appears on the screen."""
    net, stream = IntNotebook(ex, cut), text.Stream()
    st = net.step(np.array([0]), net.init_state(1))
    stream.push(0)
    rng, ti, replies = rng_start(seed), tinv_of(temp), []
    for line in lines:
        for s in stream.typed(line):
            stream.push(s)
            st = net.step(np.array([s]), st)
        shown, n = "", 0
        while True:
            lg = net.logits(st)[0].copy()
            lg[stream.banned()] = -32768
            if n >= MAXREPLY:
                c = text.SPEC_END
            else:
                c, rng = sample(lg, rng, ti, minp)
            n += 1
            disp = stream.push(c)
            if c != text.SPEC_END:
                shown += disp
            st = net.step(np.array([c]), st)
            if c == text.SPEC_END:
                break
        replies.append(shown)
    return replies


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("weights")
    ap.add_argument("lines", help="your lines, separated by |")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--minp", type=int, default=13)
    ap.add_argument("--margin", type=int, choices=[0, 1], default=0, help="1: only notes within a nat of the best")
    a = ap.parse_args()
    lines = a.lines.split("|")
    for you, j in zip(lines, talk(load(a.weights), lines, a.seed, a.temp, a.minp, MARGIN1_CUT if a.margin else None)):
        print(f"you: {you}\n{j}")


if __name__ == "__main__":
    main()

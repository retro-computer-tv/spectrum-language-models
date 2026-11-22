"""J: the GRU (train/gru.py) plus a notebook, an exact memory of the words you typed. The network decides what to
look up and when to use it; nothing in it knows about questions.

Writing. At the start of every word you type, the network's state just before it ("my dog is called") gives a write
gate g = sigmoid(w_g . h) and a key K h (16 numbers). The word's letters are kept with its key (a name's first
spelling, "*rex", as its slot digit, so the notebook can find names as well as other words). The Spectrum keeps the
latest 64 words.

Reading. At every step the network forms a query Q h. Every noted word that fits what the reply has written of the
current word so far (all of them at a word's start; after "re", those that begin "re") proposes its next letter. The
proposals are weighed by softmax(Q h . K + log g), and each proposed letter's logit gets beta times its weight, beta =
softplus(w_c . h), the network's trust in its notes. So the network chooses the word (by its query), whether to copy
at all (beta), and writes every other word itself.

Training sees every typed word with its soft gate (a small penalty on the gates), in windows that start at a
conversation's '#', so a fact and its question fall in the same window.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train.gru import GRUBN
from train.qat import TLinear, q_unit

W = 64  # notes the Spectrum keeps: the last 64 words you typed
WMAX = 256  # with ring=True, the most words a training window notes (the latest W of them take part at each step)
GATE_PENALTY = 0.02


class NotebookGRU(GRUBN):
    ARGS = GRUBN.ARGS + ("k",)

    def __init__(self, d: int = 150, e: int = 48, vocab: str = "chat", head: str = "planes2", tanh: str = "sig",
                 k: int = 16, **kw):
        super().__init__(d, e, vocab, head, tanh, **kw)
        self.k = k
        self.wk = TLinear(d, k, bias=False)
        self.wq = TLinear(d, k, bias=False)
        self.wg = TLinear(d, 1)
        self.wc = TLinear(d, 1)
        self.att_scale = nn.Parameter(torch.tensor(4.0))
        with torch.no_grad():
            self.wg.b.fill_(1.0)
            self.wc.b.fill_(0.5)

    def forward(self, idx: torch.Tensor, state: list, aux=None):
        B, L = idx.shape
        d = self.d
        x = q_unit(self.emb, self.m)[idx]
        ax = self.bnx(self.wx(x).reshape(B * L, 3 * d)).reshape(B, L, 3 * d)
        h = state[0]
        outs = []
        for t in range(L):
            h = self.cell(ax[:, t], h)
            outs.append(h)
        hq = q_unit(torch.stack(outs, 1), self.m)  # B, L, d
        logits = self.head(hq)
        self.gate_loss = torch.zeros((), device=idx.device)
        if aux is None:
            return logits, [h]
        wpos, wvalid, elig, prop = aux  # (B, N) (B, N) (B, L, N) (B, L, N)
        hw = hq.gather(1, wpos.clamp(min=0)[..., None].expand(B, wpos.shape[1], d))  # the state before each word
        keys = q_unit(self.wk(hw), self.m)  # B, N, k
        g = torch.sigmoid(self.wg(hw)[..., 0]) * wvalid  # B, N
        self.gate_loss = GATE_PENALTY * g.sum() / wvalid.sum().clamp(min=1)
        q = q_unit(self.wq(hq), self.m)  # B, L, k
        s = torch.einsum("blk,bwk->blw", q, keys) * self.att_scale + torch.log(g.clamp(min=1e-4))[:, None, :]
        s = s.masked_fill(~elig, -1e4)
        a = torch.softmax(s, -1) * elig.any(-1, keepdim=True)  # B, L, N (no notes that fit: no bonus)
        beta = F.softplus(self.wc(hq))  # B, L, 1
        bonus = torch.zeros_like(logits).scatter_add(2, prop.long(), a * beta)
        return logits + bonus, [h]


_RUN = None


def precompute(win: np.ndarray, typed: np.ndarray, ring: bool = False):
    """Word notes for windows that start at a '#': win (B, L+1) symbols, typed (B, L+1) 1 where you typed.
    Returns wpos, wvalid (B, N) and elig, prop (B, L, N) for NotebookGRU.forward. ring=False: N = W, the window's
    first W words (how J's first two builds were trained). ring=True: N = WMAX, every word, with only the latest W
    eligible at each step, as the Spectrum keeps them."""
    global _RUN
    if _RUN is None:
        _RUN = _compile()
    n = WMAX if ring else W
    B, L1 = win.shape
    wpos = np.full((B, n), -1, np.int64)
    wvalid = np.zeros((B, n), np.float32)
    elig = np.zeros((B, L1 - 1, n), np.bool_)
    prop = np.zeros((B, L1 - 1, n), np.int64)
    _RUN(win.astype(np.int64), typed.astype(np.int64), wpos, wvalid, elig, prop, W if ring else 0)
    return wpos, wvalid, elig, prop


def _compile():
    from numba import njit  # noqa: PLC0415

    @njit(cache=True)
    def run(win, typed, wpos, wvalid, elig, prop, keep):
        B, L1 = win.shape
        L = L1 - 1
        cap = wpos.shape[1]
        for b in range(B):
            n = 0
            conv = np.zeros(L1, np.int64)
            stars = np.zeros(L1, np.int64)
            cid, ns = 0, 0
            for t in range(L1):
                if win[b, t] == 0:
                    cid += 1
                    ns = 0
                if win[b, t] == 37:
                    ns += 1
                conv[t], stars[t] = cid, ns
            starts = np.zeros(cap, np.int64)
            ends = np.zeros(cap, np.int64)
            seqs = np.zeros((cap, 16), np.int64)
            lens = np.zeros(cap, np.int64)
            t = 1
            while t < L and n < cap:
                c = win[b, t]
                if typed[b, t] and 11 <= c <= 40 and not (typed[b, t - 1] and 11 <= win[b, t - 1] <= 40):
                    e = t
                    while e < L and typed[b, e] and 11 <= win[b, e] <= 40 and e - t < 15:
                        e += 1
                    starts[n], ends[n] = t, e
                    if c == 37:  # a name's first spelling: noted as its slot digit
                        seqs[n, 0], lens[n] = 37 + stars[t], 1
                    else:
                        for k in range(e - t):
                            seqs[n, k] = win[b, t + k]
                        lens[n] = e - t
                    wpos[b, n], wvalid[b, n] = t - 1, 1.0
                    n += 1
                    t = e
                else:
                    t += 1
            started = 0  # notes begun at or before t: with keep, only the latest keep of them are held
            for t in range(L):
                while started < n and starts[started] <= t:
                    started += 1
                o = 0
                while o < t and 11 <= win[b, t - o] <= 40 and o < 15:
                    o += 1
                for i in range(n):
                    if ends[i] > t or conv[starts[i]] != conv[t] or o >= lens[i] or (keep and i < started - keep):
                        continue
                    ok = True
                    for k in range(o):
                        if seqs[i, k] != win[b, t - o + 1 + k]:
                            ok = False
                            break
                    if ok:
                        elig[b, t, i] = True
                        prop[b, t, i] = seqs[i, o]

    return run

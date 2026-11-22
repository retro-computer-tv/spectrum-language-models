"""ZX-LM in floats: the network the Spectrum's one was converted from (train/zx_lm/convert.py).

One GRU controller reads and writes, one step per word. Memory is one slot per reference mention (up to 64): the
slot keeps where the word is (its reference number) and a 16-number key the network writes. A mention's key is
made when its turn ends, from the state just after the mention and the state at the turn's end, so it can say
what role the word played in the whole sentence. Every turn's end also rewrites the older keys through a learned
gate: this is how a later statement marks an earlier one as out of date. Every step reads the memory by attention
(content, plus a learned preference for recent or old turns, plus "nothing" as a slot of its own) and feeds what it
read back in. Writing, each word is either a word of its own (26 of them) or a pointer to a slot: the reference in
that slot is written out by copying its spelling. Copying is mechanical; which slot to point at is learned.

split=True gives reading (user turns) and writing (the Spectrum's turns) a GRU each, of the same total size.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from train.zx_lm.data import MAX_SLOTS
from train.zx_lm.vocab import N_REFS, Vocab


class Memory:
    def __init__(self, B: int, K: int, device):
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        self.keys, self.pre = z(B, MAX_SLOTS, K), z(B, MAX_SLOTS, K)
        self.valid = z(B, MAX_SLOTS, dt=torch.bool)
        self.pending = z(B, MAX_SLOTS, dt=torch.bool)
        self.ref = z(B, MAX_SLOTS, dt=torch.long)
        self.birth = z(B, MAX_SLOTS, dt=torch.long)
        self.agef = z(B, MAX_SLOTS)  # each slot's age in turns, capped at 15, over 8
        self.turn = z(B, dt=torch.long)
        self.n = z(B, dt=torch.long)


class ZXLM(nn.Module):
    def __init__(self, vocab: Vocab, H: int = 96, D: int = 32, K: int = 16, DR: int = 16, split: bool = False):
        super().__init__()
        self.vocab, self.H, self.K, self.split = vocab, H, K, split
        self.emb = nn.Embedding(vocab.size, D)
        self.eref = nn.Embedding(N_REFS, DR)
        self.cell = nn.GRUCell(D + K + DR, H)
        self.cell_w = nn.GRUCell(D + K + DR, H) if split else None
        self.q, self.recency = nn.Linear(H, K), nn.Linear(H, 1)
        self.key_at, self.key_end = nn.Linear(H, K), nn.Linear(H, K)
        self.upd_c, self.upd_z = nn.Linear(H, K), nn.Linear(H, K)
        self.null = nn.Parameter(torch.zeros(K + 1))
        self.out, self.gen = nn.Linear(H, vocab.n_out), nn.Linear(H, 1)

    def start(self, B: int, device) -> tuple[torch.Tensor, Memory]:
        return torch.zeros(B, self.H, device=device), Memory(B, self.K, device)

    def attend(self, h: torch.Tensor, m: Memory) -> torch.Tensor:
        """[B, 1 + slots]: column 0 is "nothing"."""
        q = self.q(h)
        s = torch.einsum("bmk,bk->bm", m.keys, q) / 4
        s = (s + self.recency(h) * m.agef).masked_fill(~m.valid, -1e9)
        s0 = (q @ self.null[:-1] / 4 + self.null[-1])[:, None]
        return torch.softmax(torch.cat([s0, s], 1), 1)

    def read(self, a: torch.Tensor, m: Memory) -> torch.Tensor:
        a = a[:, 1:, None]
        return torch.cat([(a * m.keys).sum(1), (a * self.eref(m.ref)).sum(1)], 1)

    def step(self, x, ref, user, active, h, m: Memory):
        """Read the memory, take one word in. Returns the read attention and the new state."""
        a = self.attend(h, m)
        inp = torch.cat([self.emb(x), self.read(a, m)], 1)
        hn = self.cell(inp, h)
        if self.split:
            hn = torch.where(user[:, None], hn, self.cell_w(inp, h))
        h = torch.where(active[:, None], hn, h)
        is_ref = (ref >= 0) & active
        slot = ((m.n % MAX_SLOTS)[:, None] == torch.arange(MAX_SLOTS, device=h.device)) & is_ref[:, None]  # a ring
        m.pre = torch.where(slot[..., None], self.key_at(h)[:, None, :], m.pre)
        m.ref = torch.where(slot, ref.clamp(min=0)[:, None], m.ref)
        m.valid = m.valid & ~slot
        m.pending = m.pending | slot
        m.n = m.n + is_ref.long()
        return a, h

    def predict(self, h: torch.Tensor, m: Memory) -> torch.Tensor:
        """Log-probabilities over the 26 own words, then the 16 references (by pointing)."""
        return self.predict_from(h, m.keys, m.valid, m.ref, m.agef)

    def predict_from(self, h, keys, valid, ref, agef) -> torch.Tensor:
        m = Memory.__new__(Memory)
        m.keys, m.valid, m.ref, m.agef = keys, valid, ref, agef
        a = self.attend(h, m)[:, 1:] * valid
        refp = torch.zeros(h.shape[0], N_REFS, device=h.device).scatter_add(1, ref, a)
        g = self.gen(h)
        return torch.cat([F.logsigmoid(g) + F.log_softmax(self.out(h), 1),
                          F.logsigmoid(-g) + torch.log(refp + 1e-6)], 1)

    def write(self, end: torch.Tensor, h: torch.Tensor, m: Memory) -> None:
        """At a turn's end: rewrite older keys through the learned gate, then file the turn's mentions."""
        e = end[:, None]
        c, z = torch.tanh(self.upd_c(h)), self.upd_z(h)
        g = torch.sigmoid(torch.einsum("bmk,bk->bm", m.keys, z))[..., None]
        upd = (1 - g) * m.keys + g * torch.tanh(m.keys + c[:, None])
        new = torch.tanh(m.pre + self.key_end(h)[:, None])
        keys = torch.where((e & m.valid)[..., None], upd, m.keys)
        fresh = e & m.pending
        m.keys = torch.where(fresh[..., None], new, keys)
        m.birth = torch.where(fresh, m.turn[:, None], m.birth)
        m.valid = m.valid | fresh
        m.pending = m.pending & ~e
        m.turn = m.turn + end.long()
        m.agef = (m.turn[:, None] - m.birth).clamp(0, 15).float() / 8

    def forward(self, bt, ev_weight: float = 0.5) -> dict:
        """Teacher-forced, static shapes and no host syncs (so a whole training step can be one CUDA graph). The
        loop steps and writes; every prediction and evidence term is computed afterwards from the memory as it
        stood at the start of its step (a step never changes keys)."""
        h, m = self.start(bt.B, bt.x.device)
        hs, alphas, keys, valid, ref, agef, birth = [], [], [], [], [], [], []
        for t in range(bt.T):
            keys.append(m.keys), valid.append(m.valid), ref.append(m.ref), agef.append(m.agef), birth.append(m.birth)
            a, h = self.step(bt.x[:, t], bt.ref[:, t], bt.user[:, t], bt.active[:, t], h, m)
            hs.append(h)
            alphas.append(a)
            self.write(bt.end[:, t], h, m)
        at = lambda xs, t, b: torch.stack(xs, 0)[t, b]  # noqa: E731
        tt, tb = bt.tt, bt.tb
        lp = self.predict_from(at(hs, tt, tb), at(keys, tt, tb), at(valid, tt, tb), at(ref, tt, tb), at(agef, tt, tb))
        nll = -(lp.gather(1, bt.ty[:, None]).squeeze(1) * bt.tw).sum()
        right = (lp.argmax(1) == bt.ty) & (bt.tw > 0)
        ans = bt.tw >= 1
        et, eb = bt.et, bt.eb
        a = at(alphas, et, eb)
        on = torch.gather(bt.esup, 1, at(birth, et, eb).clamp(max=bt.esup.shape[1] - 1)) & at(valid, et, eb)
        mass = torch.where(bt.esup.any(1), (a[:, 1:] * on).sum(1), a[:, 0])
        evl = -(torch.log(mass + 1e-6) * bt.ew).sum()
        stats = torch.stack([right.sum(), (bt.tw > 0).sum(), (right & ans).sum(), ans.sum()]).float()
        stats = torch.cat([stats, torch.stack([(mass * bt.ew).sum(), bt.ew.sum()])]).detach()
        return {"loss": (nll + ev_weight * evl) / bt.B, "stats": stats, "lp": lp, "aev": a}

    def weights(self) -> dict[str, int]:
        """Weight counts by matrix (biases and row parameters not included)."""
        out = {}
        for name, p in self.named_parameters():
            if p.dim() == 2:
                out[name] = p.numel()
        return out

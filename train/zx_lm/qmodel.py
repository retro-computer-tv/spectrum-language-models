"""ZX-LM as the Spectrum computes it, for quantisation-aware training.

The controller is J's ternary GRU (train/gru.py, train/qat.py), so the Z80 reuses J's lookup-table kernel and gates
unchanged; its input is 64 numbers: the word (32) and the memory read (16 key, 16 reference). One
ternary matrix of 108 rows reads the quantised state for everything else, each row with its own scale and bias:

  q 16 (1/16, |q| <= 16)  rec 1 (1/64 nat an age step)  key_at 16, key_end 16 (tanh index, 1/32)  upd_c 16 (1/32)
  upd_z 16 (1/16, |z| <= 16)  gen 1 (1/16 nat)  head 26 (1/16 nat)

Memory values are integers held in floats: keys in 127ths, attention weights in 256ths. Every rounding the Z80 does
is done here with the straight-through estimator, in the same order (reference/zx_lm.py is the integer version):

  score  = floor(q . k / 128) + floor(rec * min(age, 15) / 4)          1/16 nat; "nothing": q . k0 / 128 + b0
  w      = EXP[max - score] (round(255 exp(-e/16)), 0 from e = 100)   alpha = floor(256 w / sum w)
  read   = floor(sum alpha k / 256) in 127ths -> (r * 51 + 128) >> 8 in 25ths; floor(sum alpha e / 256) for e
  rewrite at a turn's end: g = SIG[floor(k . z / 128)], t = TANH[floor((k + c) / 4)], k += floor(g (t - k) / 256)
  new key = TANH[key_at (at the mention) + key_end (at the turn's end)]
  answer word: gen >= 0 -> the best of the 26 words; else the reference whose slots hold the most attention
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from train import qat
from train.qat import M, TLinear, q_unit, sigmoid_t, tanh_t
from train.zx_lm.data import MAX_SLOTS
from train.zx_lm.model import Memory
from train.zx_lm.vocab import N_REFS, Vocab

K = 16
GROUPS = [("q", K), ("rec", 1), ("key_at", K), ("key_end", K), ("upd_c", K), ("upd_z", K), ("gen", 1)]


class _Floor(torch.autograd.Function):
    """Floor forward, identity backward (the straight-through estimator), one kernel instead of three."""

    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, g):
        return g


class _Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def fl(x: torch.Tensor) -> torch.Tensor:
    return _Floor.apply(x)


def rd(x: torch.Tensor) -> torch.Tensor:
    return _Round.apply(x)


def table_sig(idx: torch.Tensor) -> torch.Tensor:
    """SIG[idx] for an integer index (1/16 steps), 0..255."""
    i = idx.clamp(-128, 127)
    return rd(torch.clamp(256 * torch.sigmoid(i / 16), max=255))


def table_tanh(idx: torch.Tensor) -> torch.Tensor:
    """TANH[idx] for an integer index (1/32 steps), -127..127."""
    i = idx.clamp(-128, 127)
    return rd(127 * torch.tanh(i / 32))


class ZXQ(nn.Module):
    def __init__(self, vocab: Vocab, H: int = 96, D: int = 32, DR: int = 16, m: int = M, bn: bool = False):
        """bn=False (the deployed design): no batch norm; the gates get a bias each instead. With batch norm the
        ternary network reached 42.5% on development after 1,000 batches, without it 57.1%."""
        super().__init__()
        qat.TANH_MODE = "table"  # ZX-LM's GRU has its own tanh table (J's reads tanh from the sigmoid table)
        self.vocab, self.H, self.m, self.K = vocab, H, m, K
        self.emb = nn.Parameter(torch.randn(vocab.size, D) * 0.5)
        self.eref = nn.Parameter(torch.randn(N_REFS, DR) * 0.5)
        wk = "ternary"
        self.no_bn = not bn
        self.wx = TLinear(D + K + DR, 3 * H, bias=self.no_bn, kind=wk)
        self.wh = TLinear(H, 3 * H, bias=self.no_bn, kind=wk)
        self.bnx, self.bnh = nn.BatchNorm1d(3 * H, momentum=0.01), nn.BatchNorm1d(3 * H, momentum=0.01)
        with torch.no_grad():
            self.bnx.weight.fill_(0.5)
            self.bnh.weight.fill_(0.5)
        self.n_proj = sum(n for _, n in GROUPS) + vocab.n_out
        self.proj = TLinear(H, self.n_proj, kind=wk)
        self.null_k = nn.Parameter(torch.zeros(K))
        self.null_b = nn.Parameter(torch.zeros(1))
        sl, i = {}, 0
        for name, n in GROUPS + [("head", vocab.n_out)]:
            sl[name] = slice(i, i + n)
            i += n
        self.sl = sl

    # ---- the h side: everything read from the quantised state, as integers in their units
    def project(self, h: torch.Tensor) -> dict:
        p = self.proj(q_unit(h, self.m))
        s = self.sl
        return {"q": fl(16 * p[:, s["q"]]).clamp(-16, 16), "rec": fl(64 * p[:, s["rec"]]).clamp(-128, 127),
                "key_at": fl(32 * p[:, s["key_at"]]), "key_end": fl(32 * p[:, s["key_end"]]),
                "c": table_tanh(fl(32 * p[:, s["upd_c"]])), "z": fl(16 * p[:, s["upd_z"]]).clamp(-16, 16),
                "gen": fl(16 * p[:, s["gen"]]), "head": fl(16 * p[:, s["head"]])}

    def norm(self, bn: nn.BatchNorm1d, v: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Batch norm over the conversations still running (padding left out of the statistics)."""
        if self.no_bn:
            return v
        if not self.training:
            return bn(v)
        w = act.float()[:, None]
        n = w.sum().clamp(min=2)
        mu = (v * w).sum(0) / n
        var = ((v - mu) ** 2 * w).sum(0) / n
        with torch.no_grad():
            bn.running_mean.mul_(1 - bn.momentum).add_(bn.momentum * mu)
            bn.running_var.mul_(1 - bn.momentum).add_(bn.momentum * var * n / (n - 1))
        return (v - mu) / torch.sqrt(var + bn.eps) * bn.weight + bn.bias

    def cell(self, x: torch.Tensor, h: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        H = self.H
        ax = self.norm(self.bnx, self.wx(x), act)
        ah = self.norm(self.bnh, self.wh(q_unit(h, self.m)), act)
        r, z = sigmoid_t(ax[:, : 2 * H] + ah[:, : 2 * H]).chunk(2, dim=-1)
        n = tanh_t(ax[:, 2 * H:] + r * ah[:, 2 * H:])
        return torch.lerp(n, h, z)

    def start(self, B: int, device) -> tuple[torch.Tensor, Memory, dict]:
        h = torch.zeros(B, self.H, device=device)
        m = Memory(B, K, device)
        return h, m, self.project(h)

    # ---- memory
    def weights(self, P: dict, keys, valid, agef) -> torch.Tensor:
        """[B, 1 + slots] attention in 256ths (column 0: nothing). agef holds min(age, 15)."""
        nk = rd(127 * torch.tanh(self.null_k))
        s = fl(torch.einsum("bmk,bk->bm", keys, P["q"]) / 128) + fl(P["rec"] * agef / 4)
        s0 = fl((P["q"] @ nk) / 128) + rd(16 * self.null_b)
        s = torch.cat([s0[:, None], s.masked_fill(~valid, -1e4)], 1)
        e = s.max(1, keepdim=True).values.detach() - s
        w = rd(255 * torch.exp(-e / 16)) * (e < 100)
        return fl(256 * w / w.sum(1, keepdim=True))

    def read(self, a: torch.Tensor, m: Memory) -> torch.Tensor:
        a = a[:, 1:, None]
        r127 = fl((a * m.keys).sum(1) / 256)
        r25 = fl((r127 * 51 + 128) / 256)
        per_ref = torch.zeros(a.shape[0], N_REFS, device=a.device).scatter_add(1, m.ref, a[..., 0])
        e25 = fl(per_ref @ rd(q_unit(self.eref, self.m) * self.m) / 256)  # = sum alpha e[ref], exactly
        return torch.cat([r25, e25], 1) / self.m

    def step(self, x, ref, user, active, h, m: Memory, P: dict, xe: torch.Tensor | None = None):
        """xe: the word's embedding row, when the caller gathered them all at once (training)."""
        a = self.weights(P, m.keys, m.valid, m.agef)
        if xe is None:
            xe = F.embedding(x, q_unit(self.emb, self.m))
        inp = torch.cat([xe, self.read(a, m)], 1)
        h = torch.where(active[:, None], self.cell(inp, h, active), h)
        P = self.project(h)
        is_ref = (ref >= 0) & active
        slot = ((m.n % MAX_SLOTS)[:, None] == torch.arange(MAX_SLOTS, device=h.device)) & is_ref[:, None]
        m.pre = torch.where(slot[..., None], P["key_at"][:, None, :], m.pre)
        m.ref = torch.where(slot, ref.clamp(min=0)[:, None], m.ref)
        m.valid = m.valid & ~slot
        m.pending = m.pending | slot
        m.n = m.n + is_ref.long()
        return a, h, P

    def write(self, end: torch.Tensor, P: dict, m: Memory) -> None:
        e = end[:, None]
        g = table_sig(fl(torch.einsum("bmk,bk->bm", m.keys, P["z"]) / 128))[..., None]
        t = table_tanh(fl((m.keys + P["c"][:, None]) / 4))
        upd = m.keys + fl(g * (t - m.keys) / 256)
        new = table_tanh((m.pre + P["key_end"][:, None]).clamp(-128, 127))
        keys = torch.where((e & m.valid)[..., None], upd, m.keys)
        fresh = e & m.pending
        m.keys = torch.where(fresh[..., None], new, keys)
        m.birth = torch.where(fresh, m.turn[:, None], m.birth)
        m.valid = m.valid | fresh
        m.pending = m.pending & ~e
        m.turn = m.turn + end.long()
        m.agef = (m.turn[:, None] - m.birth).clamp(0, 15).float()

    # ---- output
    def logp(self, P: dict, a: torch.Tensor, m_ref: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Training's mixture: the 26 own words, then the 16 references by attention (from 256ths)."""
        refp = torch.zeros(a.shape[0], N_REFS, device=a.device).scatter_add(1, m_ref, a[:, 1:] * valid) / 256
        g = P["gen"][:, 0] / 16
        return torch.cat([F.logsigmoid(g)[:, None] + F.log_softmax(P["head"] / 16, 1),
                          F.logsigmoid(-g)[:, None] + torch.log(refp + 1e-3)], 1)

    def choose(self, P: dict, a: torch.Tensor, m_ref: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """The Spectrum's rule: its own best word if gen >= 0, else the reference with the most attention."""
        refp = torch.zeros(a.shape[0], N_REFS, device=a.device).scatter_add(1, m_ref, a[:, 1:] * valid)
        word = P["head"].argmax(1)
        return torch.where(P["gen"][:, 0] >= 0, word, self.vocab.n_out + refp.argmax(1))


def forward(self: ZXQ, bt, ev_weight: float = 0.5, teacher: dict | None = None, kd: float = 1.0) -> dict:
    """Teacher-forced and static, as model.ZXLM.forward: predictions and evidence afterwards, from the memory as it
    stood at the start of each step. teacher: a float network's outputs on the same batch ("lp" at the answer
    targets, "aev" at the evidence reads); the student also matches them (distillation, weight kd)."""
    h, m, P = self.start(bt.B, bt.x.device)
    alphas, Ps, snaps = [], [], []
    xe = F.embedding(bt.x, q_unit(self.emb, self.m))  # one gather
    for t in range(bt.T):
        snaps.append((m.keys, m.valid, m.ref, m.agef, m.birth))
        a, h, P = self.step(bt.x[:, t], bt.ref[:, t], bt.user[:, t], bt.active[:, t], h, m, P, xe[:, t])
        alphas.append(a)
        Ps.append(torch.cat([P["q"], P["rec"], P["gen"], P["head"]], 1))
        self.write(bt.end[:, t], P, m)
    at = lambda xs, t, b: torch.stack(xs, 0)[t, b]  # noqa: E731
    tt, tb = bt.tt, bt.tb
    pp = at(Ps, tt, tb)
    Pt = {"q": pp[:, :K], "rec": pp[:, K:K + 1], "gen": pp[:, K + 1:K + 2], "head": pp[:, K + 2:]}
    keys, valid, ref, agef = (torch.stack([s[i] for s in snaps], 0)[tt, tb] for i in range(4))
    a = self.weights(Pt, keys, valid, agef)
    lp = self.logp(Pt, a, ref, valid)
    nll = -(lp.gather(1, bt.ty[:, None]).squeeze(1) * bt.tw).sum()
    right = (self.choose(Pt, a, ref, valid) == bt.ty) & (bt.tw > 0)
    ans = bt.tw >= 1
    et, eb = bt.et, bt.eb
    ae = at(alphas, et, eb) / 256
    birth = torch.stack([s[4] for s in snaps], 0)[et, eb]
    on = torch.gather(bt.esup, 1, birth.clamp(max=bt.esup.shape[1] - 1)) & torch.stack([s[1] for s in snaps], 0)[et, eb]
    mass = torch.where(bt.esup.any(1), (ae[:, 1:] * on).sum(1), ae[:, 0])
    evl = -(torch.log(mass + 1e-3) * bt.ew).sum()
    stats = torch.stack([right.sum(), (bt.tw > 0).sum(), (right & ans).sum(), ans.sum()]).float()
    stats = torch.cat([stats, torch.stack([(mass * bt.ew).sum(), bt.ew.sum()])]).detach()
    loss = nll + ev_weight * evl
    if teacher is not None:
        pt = teacher["lp"].exp()
        kl = (pt * (teacher["lp"] - lp)).sum(1)
        at_ = teacher["aev"]
        kla = (at_ * (torch.log(at_ + 1e-6) - torch.log(ae + 1e-3))).sum(1)
        loss = loss + kd * ((kl * (bt.tw > 0)).sum() + ev_weight * (kla * bt.ew).sum())
    return {"loss": loss / bt.B, "stats": stats, "lp": lp}


ZXQ.forward = forward


class Stepper:
    """ZXQ behind the float model's stepping interface (runner.py): start, step, predict, write."""

    def __init__(self, model: ZXQ):
        self.model, self.P = model, None

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def start(self, B: int, device):
        h, m, self.P = self.model.start(B, device)
        return h, m

    def step(self, x, ref, user, active, h, m):
        a, h, P = self.model.step(x, ref, user, active, h, m, self.P)
        self.P = {k: torch.where(active[:, None], v, self.P[k]) for k, v in P.items()}
        return a, h

    def predict(self, h, m) -> torch.Tensor:
        a = self.model.weights(self.P, m.keys, m.valid, m.agef)
        c = self.model.choose(self.P, a, m.ref, m.valid)
        return F.one_hot(c, self.model.vocab.n_out + N_REFS).float()

    def write(self, end, h, m) -> None:
        self.model.write(end, self.P, m)


def n_weights(model: ZXQ) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if p.dim() == 2)

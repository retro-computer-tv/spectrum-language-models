"""The trained float network (model.ZXLM) as the Spectrum's network (qmodel.ZXQ, no batch norm), weight for weight.

    python -m train.zx_lm.convert runs/float/best.npz runs/conv/best.npz      # ready for train.py --quant --init

Both compute the same thing, so the float one's weights carry over exactly (up to the null key's tanh), then
quantisation-aware fine-tuning adapts them (rounded to trits as they come, the converted weights answered 1.7% of
questions right; after 500 batches of fine-tuning, 68% of the development set):

- GRU: GRUCell's [r, z, n] rows are ZXQ's; b_ih is the x part's bias, b_hh the h part's.
- Inputs must lie in [-1, 1] (25ths on the Spectrum): the word and reference embeddings are divided by their
  largest value and the GRU's input columns multiplied by it.
- Projections, into ZXQ's units: ZXQ's score is q . k in nats (ZXLM's q . k / 4), so the query rows are divided by 4;
  ZXQ's recency is p x age (ZXLM's p x age / 8), so that row is divided by 8; the rest are equal.
"""
from __future__ import annotations

import sys

import torch

from train.zx_lm.model import ZXLM
from train.zx_lm.qmodel import ZXQ
from train.zx_lm.train import load, save


@torch.no_grad()
def convert(f: ZXLM, q: ZXQ) -> None:
    D = f.emb.weight.shape[1]
    se = float(f.emb.weight.abs().max())
    sr = float(f.eref.weight.abs().max())
    q.emb.copy_(f.emb.weight / se)
    q.eref.copy_(f.eref.weight / sr)
    wih = f.cell.weight_ih.clone()
    wih[:, :D] *= se
    wih[:, D + 16:] *= sr  # the read: 16 key numbers (already in [-1, 1]), then 16 reference numbers
    q.wx.w.copy_(wih)
    q.wx.b.copy_(f.cell.bias_ih)
    q.wh.w.copy_(f.cell.weight_hh)
    q.wh.b.copy_(f.cell.bias_hh)
    rows = [(f.q, 0.25), (f.recency, 0.125), (f.key_at, 1.0), (f.key_end, 1.0), (f.upd_c, 1.0), (f.upd_z, 1.0),
            (f.gen, 1.0), (f.out, 1.0)]
    i = 0
    for lin, k in rows:
        n = lin.weight.shape[0]
        q.proj.w[i: i + n] = lin.weight * k
        q.proj.b[i: i + n] = lin.bias * k
        i += n
    assert i == q.n_proj
    nk = f.null[:-1] / 4  # ZXLM's null score q . k0 / 4, ZXQ's q . tanh(k0)
    q.null_k.copy_(torch.atanh(nk.clamp(-0.99, 0.99)))
    q.null_b.copy_(f.null[-1:])


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    f, cfg = load(src)
    q = ZXQ(f.vocab, H=cfg["H"], bn=False)
    convert(f, q)
    save(q, dst, {**cfg, "quant": True, "bn": False}, 0)
    print("converted", src, "->", dst)


if __name__ == "__main__":
    main()

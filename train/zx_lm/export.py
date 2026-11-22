"""A trained ZX-LM (the Spectrum's network, train.py --quant) in the Spectrum's integers: the weights file build.py
reads.

    python -m train.zx_lm.export runs/spectrum/best.npz weights/mine.npz

The GRU's x and h parts and the 108-row projection as trits with a mantissa, shift and bias per row (as
train/export.py), the embeddings in 25ths, the null key in 127ths; the vocabulary goes in the file's meta.
reference/zx_lm.py says how the Spectrum uses them.
"""
from __future__ import annotations

import sys

import numpy as np

from reference.weights import save
from train.zx_lm.qmodel import GROUPS, K, ZXQ
from train.zx_lm.train import load

UNITS = {"q": 16, "rec": 64, "key_at": 32, "key_end": 32, "upd_c": 32, "upd_z": 16, "gen": 16, "head": 16}


def trits(w) -> tuple[np.ndarray, np.ndarray]:
    wq = w.detach().float().cpu().numpy()
    s = np.abs(wq).max(axis=1)
    s[s == 0] = 1.0
    t = np.rint(wq / s[:, None]).astype(np.int64)
    assert set(np.unique(t)) <= {-1, 0, 1}
    return t, s.astype(np.float64)


def mantissa(scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.minimum(scale, 255 / 256)  # the Z80's scale is at most 255 >> 8: a larger row is capped there
    sh = np.floor(np.log2(255.5 / scale)).astype(np.int64)
    M = np.rint(scale * 2.0 ** sh).astype(np.int64)
    over = M > 255
    sh[over] -= 1
    M[over] = np.rint(scale[over] * 2.0 ** sh[over]).astype(np.int64)
    assert (M <= 255).all() and (M >= 128).all() and (sh >= 8).all() and (sh <= 31).all(), (sh.min(), sh.max())
    return M, sh


def export(model: ZXQ) -> dict:
    assert model.no_bn, "the Spectrum's network has no batch norm (its gates have a bias each)"
    m, H = model.m, model.H
    ex = {"m": m, "d": H, "K": K, "n_out": model.vocab.n_out, "version": model.vocab.version,
          "words": model.vocab.words, "out_words": model.vocab.out_words}
    unit = lambda p: np.rint(np.clip(p.detach().cpu().numpy(), -1, 1) * m).astype(np.int64)  # noqa: E731
    ex["emb"], ex["eref"] = unit(model.emb), unit(model.eref)
    gate_units = np.array([16.0] * (2 * H) + [32.0] * H)
    for part, lin in (("x", model.wx), ("h", model.wh)):
        t, s = trits(lin.qweight())
        Kf = s / m * gate_units
        C = lin.b.detach().cpu().numpy().astype(np.float64) * gate_units
        ex[f"{part}_T"] = np.where(Kf[:, None] < 0, -t, t)
        ex[f"{part}_M"], ex[f"{part}_sh"] = mantissa(np.abs(Kf))
        b = np.rint(C).astype(np.int64) + (1 if part == "x" else 0)  # as J's: the pair's two floors
        if (np.abs(Kf) > 255 / 256).any() or (np.abs(b) > 127).any():
            print(f"{part}: {int((np.abs(Kf) > 255 / 256).sum())} scales capped, {int((np.abs(b) > 127).sum())} "
                  "biases clipped", file=sys.stderr)
        ex[f"{part}_B"] = np.clip(b, -128, 127)
    t, s = trits(model.proj.qweight())
    u = np.concatenate([[UNITS[name]] * n for name, n in GROUPS + [("head", model.vocab.n_out)]]).astype(np.float64)
    ex["p_T"] = t
    ex["p_M"], ex["p_sh"] = mantissa(s / m * u)
    b = np.rint(model.proj.b.detach().cpu().numpy() * u).astype(np.int64)
    ex["p_B"] = np.clip(b, -32768, 32767)
    ex["null_k"] = np.rint(127 * np.tanh(model.null_k.detach().cpu().numpy())).astype(np.int64)
    ex["null_b"] = int(np.rint(16 * float(model.null_b.detach())))
    return ex


def main() -> None:
    model, cfg = load(sys.argv[1])
    assert cfg.get("quant"), "the Spectrum's network (train.py --quant), not the float one"
    model.eval()
    save(export(model), sys.argv[2])
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()

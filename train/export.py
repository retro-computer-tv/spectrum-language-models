"""A trained GRU network (the story writer, or J) in the Spectrum's integers: the weights file build.py reads.

    python -m train.export runs/<run>/qat.npz weights/<name>.npz

Embedding int8 in [-m, m]; each matrix as trits plus, per row, an 8-bit mantissa M (128..255), a shift sh and a bias
B (int8 in table units; int16 for the output head), with the batch normalisation folded in. reference/gru.py and
reference/notebook.py say how the Spectrum uses them.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from reference.weights import save


def trits(lin) -> tuple[np.ndarray, np.ndarray]:
    """A TLinear's weights as integers (-1, 0, +1) and the per-row float scale that multiplies them."""
    wq = lin.qweight().detach().float().cpu().numpy()
    s = np.abs(wq).max(axis=1)
    s[s == 0] = 1.0
    t = np.rint(wq / s[:, None]).astype(np.int64)
    assert set(np.unique(t)) <= {-1, 0, 1}
    return t, s.astype(np.float64)


def mantissa(scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """scale ~= M / 2^sh with M in 128..255 (the largest sh that keeps M below 256)."""
    sh = np.floor(np.log2(255.5 / scale)).astype(np.int64)
    M = np.rint(scale * 2.0 ** sh).astype(np.int64)
    over = M > 255
    sh[over] -= 1
    M[over] = np.rint(scale[over] * 2.0 ** sh[over]).astype(np.int64)
    assert (M <= 255).all() and (M >= 128).all() and (sh >= 0).all() and (sh <= 31).all(), (M.min(), sh.min(), sh.max())
    return M, sh


def bn_fold(bn, s: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Batch norm (evaluation mode) after a ternary product: value = acc * K + C."""
    g = bn.weight.detach().cpu().numpy().astype(np.float64)
    b = bn.bias.detach().cpu().numpy().astype(np.float64)
    mu = bn.running_mean.detach().cpu().numpy().astype(np.float64)
    sd = np.sqrt(bn.running_var.detach().cpu().numpy().astype(np.float64) + bn.eps)
    return g * s / (m * sd), b - g * mu / sd


def head_planes(model, ex: dict, m: int) -> None:
    """The output head as planes: each plane its own trits and row scale; the bias rides on the first plane."""
    from train.qat import plane_values  # noqa: PLC0415

    lin = model.head
    vals = plane_values(lin.w.detach().float(), int(lin.kind[6:])) if lin.kind.startswith("planes") else \
        [lin.qweight().detach().float()]
    for k, pv in enumerate(vals):
        pv = pv.cpu().numpy().astype(np.float64)
        s = np.abs(pv).max(1)
        s[s == 0] = 1.0
        t = np.rint(pv / s[:, None]).astype(np.int64)
        assert set(np.unique(t)) <= {-1, 0, 1}
        tag = "head" if k == 0 else f"head{k + 1}"
        ex[f"{tag}_T"] = t
        ex[f"{tag}_M"], ex[f"{tag}_sh"] = mantissa(s / m * 16)
    ex["head_planes"] = len(vals)
    ex["head_B"] = np.clip(np.rint(lin.b.detach().cpu().numpy() * 16).astype(np.int64), -32768, 32767)


def export_gru(model) -> dict:
    """r, z and n each get an x part and an h part, with their own scale and bias (the folded batch norms)."""
    m, d, e = model.m, model.d, model.e
    ex = {"m": m, "d": d, "e": e, "quant": "ternary", "kind": "grubn", "tanh": model.tanh, "vocab": model.vocab,
          "x_planes": 1}
    ex["emb"] = np.rint(np.clip(model.emb.detach().cpu().numpy(), -1, 1) * m).astype(np.int64)
    units = np.array([16.0] * (2 * d) + [32.0] * d)  # the sigmoid's index step is 1/16, tanh's 1/32
    for part, lin, bn in (("x", model.wx, model.bnx), ("h", model.wh, model.bnh)):
        t, s = trits(lin)
        K, C = bn_fold(bn, s, m)
        K, C = K * units, C * units
        ex[f"{part}_M"], ex[f"{part}_sh"] = mantissa(np.abs(K))
        ex[f"{part}_T"] = np.where(K[:, None] < 0, -t, t)  # a negative gamma flips the row's trits instead
        # each part's scaled value is floored on the Z80: about half a table step lost per part, so the x part
        # carries +1 for the pair (it halves the integer network's loss against the trained one)
        b = np.rint(C).astype(np.int64) + (1 if part == "x" else 0)
        if (np.abs(b) > 127).any():
            print(f"{part}: {int((np.abs(b) > 127).sum())} biases clipped to a signed byte", file=sys.stderr)
        ex[f"{part}_B"] = np.clip(b, -128, 127)
    head_planes(model, ex, m)
    return ex


def export_notebook(model) -> dict:
    """J: the GRU, plus the notebook's key, query, gate and trust rows and its attention scale."""
    ex = export_gru(model)
    m = model.m
    for name, lin in (("k", model.wk), ("q", model.wq)):
        t, s = trits(lin)
        ex[f"{name}_T"] = t
        ex[f"{name}_M"], ex[f"{name}_sh"] = mantissa(s)  # K = s * acc on the int8 scale (acc counts units of 1/m)
    for name, lin in (("g", model.wg), ("c", model.wc)):
        t, s = trits(lin)
        ex[f"{name}_T"] = t
        ex[f"{name}_M"], ex[f"{name}_sh"] = mantissa(s / m * 16)
        ex[f"{name}_B"] = np.rint(lin.b.detach().cpu().numpy() * 16).astype(np.int64)
    a = float(model.att_scale.detach()) * 16 / m ** 2
    ex["att_M"], ex["att_sh"] = mantissa(np.array([a]))
    ex["kind"] = "notebook"
    return ex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("qat", help="a QAT checkpoint (.npz)")
    ap.add_argument("out", help="the weights file to write")
    a = ap.parse_args()
    import json  # noqa: PLC0415

    kind = json.loads(str(np.load(a.qat)["meta"]))["kind"]
    if kind == "notebook":
        from train.gru import load_qat  # noqa: PLC0415
        from train.j.model import NotebookGRU  # noqa: PLC0415

        ex = export_notebook(load_qat(a.qat, NotebookGRU))
    else:
        from train.gru import load_qat  # noqa: PLC0415

        ex = export_gru(load_qat(a.qat))
    save(ex, a.out)
    print(f"{a.out}: {', '.join(k for k in ex if isinstance(ex[k], np.ndarray))}")


if __name__ == "__main__":
    main()

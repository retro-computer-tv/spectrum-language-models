"""Reading and writing the deployed weights (weights/*.npz).

Each file is a numpy .npz archive: integer arrays, plus one string array `meta` holding a JSON object of the network's
sizes and settings. Every array is exactly what goes on the tape, before it is packed into bytes:

  emb                    the input embedding, int8 in [-m, m] (m = 25: the lookup-table kernel's input bound)
  <name>_T               a matrix of trits (-1, 0, +1), one row per output
  <name>_M, <name>_sh    per row, an 8-bit mantissa (128..255) and a shift: the row's scale is M / 2^sh
  <name>_B               per row, a bias in the units of the table the row indexes (int8; int16 for output rows)

`python -c "from reference.weights import load; print(load('weights/zx-lm.npz').keys())"` lists a file's contents. The
tape stores each matrix five trits to a byte (z80/kernel.py) and each row's M, sh and B as bytes beside it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load(path: str | Path) -> dict:
    """Every array as int64 (the reference's arithmetic must never overflow), and the meta keys alongside."""
    z = np.load(path)
    ex = {k: z[k].astype(np.int64) for k in z.files if k != "meta"}
    ex.update(json.loads(str(z["meta"])))
    return ex


def save(ex: dict, path: str | Path) -> None:
    """Arrays in the smallest integer type that holds them, compressed; everything else in meta."""
    arrays, meta = {}, {}
    for k, v in ex.items():
        if isinstance(v, np.ndarray) and v.ndim:
            v = v.astype(np.int64)
            for t in (np.int8, np.int16, np.int32, np.int64):
                if v.min(initial=0) >= np.iinfo(t).min and v.max(initial=0) <= np.iinfo(t).max:
                    arrays[k] = v.astype(t)
                    break
        else:
            meta[k] = v.item() if isinstance(v, np.generic | np.ndarray) else v
    np.savez_compressed(path, **arrays, meta=json.dumps(meta))

"""J's training text: one stream of symbols per conversation (reference/text.py has the alphabet).

    #hello! my name is *rosa.>hello 1! nice to meet you.<what is your name?>i'm a little computer.<

Every symbol has a role: 1 for what you typed (your text and the '>' that ends it), 2 for what the Spectrum writes
(its reply and the '<' that ends it), 0 for the '#' and for a reply training must not learn to write. Training weighs
the loss by role (train/j/train.py).
"""
from __future__ import annotations

import re

import numpy as np

from reference.text import CSTOI, MAXNAME, NSLOTS

PLAIN = set("\n !\"',-.:?abcdefghijklmnopqrstuvwxyz")
TABLE = {ord("\u2018"): "'", ord("\u2019"): "'", ord("\u201c"): '"', ord("\u201d"): '"', ord(";"): ",",
         ord("\u2014"): ", ", ord("\u2013"): "-", ord("\u2026"): "...", ord("\n"): " "}


def clean(text: str) -> str | None:
    """A turn in the story data's normalisation, case kept (names are found by case): straight quotes, ';' to ',',
    one space. None if anything is left outside the alphabet (digits, '*', '#', other symbols)."""
    t = re.sub(r"\s+", " ", text.translate(TABLE)).strip()
    return t if t and all(c.lower() in PLAIN for c in t) else None


def encode_turns(turns: list[tuple[str, str]], names: set[str]) -> tuple[str, str]:
    """(stream text, roles as 'n', 'u', 's') for one conversation of (role 'u' or 's', cleaned text) turns. A word is a
    name if its lower case is in `names`; the first three distinct names become slots in order of first mention,
    across all turns, and any later name is spelled every time (as in the stories)."""
    table: list[str] = []
    rx = re.compile(r"[A-Za-z]+")

    def sub(m):
        w = m.group().lower()
        if w not in names:
            return w
        if w in table:
            return str(table.index(w) + 1)
        if len(table) < NSLOTS and len(w) <= MAXNAME:
            table.append(w)
            return "*" + w
        return w

    text, roles = ["#"], ["n"]
    for role, t in turns:
        body = rx.sub(sub, t).lower() + (">" if role == "u" else "<")
        text.append(body)
        roles.append(role * len(body))
    return "".join(text), "".join(roles)


_LUT = np.zeros(256, dtype=np.uint8)
for _c, _i in CSTOI.items():
    _LUT[ord(_c)] = _i


def to_ids(text: str) -> np.ndarray:
    return _LUT[np.frombuffer(text.encode("ascii"), dtype=np.uint8)]


def roles(r: str) -> np.ndarray:
    """'u', 's', 'n' as 1, 2, 0."""
    role = np.frombuffer(r.encode("ascii"), dtype=np.uint8)
    return np.where(role == ord("u"), 1, np.where(role == ord("s"), 2, 0)).astype(np.uint8)

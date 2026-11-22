"""Long conversations for J's long1kr build (train/j/fine_tune.py says what is in them): built by
`python -m train.j.fine_tune long`. Made-up words, names and small talk for testing are never here.
"""
from __future__ import annotations

import random
import re

import numpy as np

from reference.text import CVOCAB, MAXNAME, NSLOTS
from train.j import facts as D
from train.j import pools as P
from train.j.fine_tune import (CTX_REL, DATA3, NONCE, SPELL_TRAIN, SPLITS, _filler, answer, choose_conv,
                               corrected_conv, ctx_subject, ctx_value, data3_half, fact_only, repeat_conv, spell_conv,
                               wordings, write)
from train.j.format import to_ids

MAXLEN = 1000
TALK = {"data3": 0.45, "filler": 0.15, "fact": 0.20, "instruction": 0.15, "corrected": 0.05}
ITEMS = {"context": 0.40, "correct": 0.20, "name": 0.15, "pet": 0.15, "likes": 0.10}
GAPS = [0, 1, 1, 2, 2, 3, 4] + list(range(5, 15))  # exchanges between telling and asking
BACKREF = re.compile(r"remember|told|tell me|said|name|again|earlier|before|talking about|[0-9*]")
NAME_TELL = [("my name is {N}.", "hello {N}! nice to meet you."), ("i'm {N}.", "hi {N}! what a lovely name."),
             ("hello! i am {N}.", "hello {N}! how are you?")]
NAME_ASK = [("tell me my name.", "your name is {N}."), ("what is my name?", "you are {N}!"),
            ("do you know who i am?", "yes! you are {N}.")]
PET_TELL = [("i have a {p} called {N}.", "{N} is a lovely name for a {p}!"), ("my {p} is called {N}.", "hello to {N}!")]
PET_ASK = [("what is my {p} called?", "your {p} is called {N}."), ("what is my {p}'s name?", "your {p} is {N}.")]
REL_TELL = [("my {r} is called {N}.", "hello to your {r} {N}!"), ("i have a {r} called {N}.", "{N} sounds nice!")]
REL_ASK = [("what is my {r} called?", "your {r} is called {N}."), ("who is my {r}?", "your {r} is {N}.")]
LIKE_TELL = ("my {r} {N} likes {x}.", "ok! {N} likes {x}.")
LIKE_ASK = ("what does {N} like?", "{N} likes {x}.")
FOLLOW = ("are you sure?", "yes! {a}")


def _lines(ids: np.ndarray) -> str:
    return np.frombuffer(CVOCAB.encode(), dtype=np.uint8)[ids].tobytes().decode()


def exchanges(split: str, skip: set[int], n: int, seed: str) -> list[tuple[str, str]]:
    """Up to n (your line, the Spectrum's reply) pairs from data3 conversations not in `skip` (conversation numbers):
    no names, nothing that refers back to the conversation they came from, 15 to 160 symbols."""
    ids = np.fromfile(DATA3 / f"{split}.u8", dtype=np.uint8)
    starts = np.flatnonzero(ids == 0)
    order = [i for i in range(len(starts) - 1) if i not in skip]
    random.Random(seed).shuffle(order)
    out = []
    for i in order:
        text = _lines(ids[starts[i] + 1: starts[i + 1]])
        for m in re.finditer(r"([^<>]*)>([^<>]*)<", text):
            u, s = m.group(1), m.group(2)
            if 15 <= len(u) + len(s) <= 160 and not BACKREF.search(u + " " + s):
                out.append((u, s))
        if len(out) >= n:
            break
    return out[:n]


def data3_used(total: int) -> set[int]:
    """The conversation numbers the data3 half draws (fine_tune.data3_half): never reused as talk."""
    ids = np.fromfile(DATA3 / "train.u8", dtype=np.uint8)
    starts = np.flatnonzero(ids == 0)
    order = list(range(len(starts) - 1))
    random.Random("diag-mix").shuffle(order)
    used, got = set(), 0
    for i in order:
        used.add(i)
        got += starts[i + 1] - starts[i]
        if got >= total:
            break
    return used


class Builder:
    def __init__(self, talk: list[tuple[str, str]]):
        self.talk = talk
        self.real = [r for r in D.REL if r != "arbitrary"]
        self.words = SPELL_TRAIN + NONCE["word-train"]
        self.names = P.train_names()
        self.held = set(NONCE["ctx-test"]) | set(NONCE["word-test"]) | set(P.TEST_NAMES_RAW)

    def name(self, rng: random.Random, avoid: set[str]) -> str:
        while True:
            w = rng.choice(self.names) if rng.random() < 0.75 else P.made_up_name(rng)
            if w not in avoid and w not in self.held and len(w) <= MAXNAME:
                return w.capitalize()

    def talk_ex(self, rng: random.Random) -> list:
        k = rng.choices(list(TALK), list(TALK.values()))[0]
        if k == "data3":
            u, s = rng.choice(self.talk)
            return [["u", u, True], ["s", s, True]]
        if k == "filler":
            return _filler(rng, 1)
        if k == "fact":
            return fact_only(rng, self.real if rng.random() < 0.6 else ["arbitrary"])
        if k == "instruction":
            f = rng.choices([spell_conv, repeat_conv, choose_conv], [10, 8, 6])[0]
            forms = {spell_conv: D.SPELL, repeat_conv: D.REPEAT, choose_conv: D.CHOOSE}[f][0]
            return f(rng, self.words, forms)[-2:]
        rel = rng.choice(self.real)
        trained = SPLITS[rel][0]
        i = rng.randrange(len(trained))
        s, v = trained[i]
        return corrected_conv(rng, rel, s, v, D.REL[rel]["q"][rng.choice(wordings(rel, i)[0])].format(s=s), False)

    def talk_n(self, rng: random.Random, n: int) -> list:
        return [t for _ in range(n) for t in self.talk_ex(rng)]

    def item(self, rng: random.Random, kind: str, used: set[str]) -> tuple[list, list, list]:
        """(what you tell, a correction or nothing, the question and its answer) for one thing to remember."""
        if kind == "context":
            rel = rng.choice(CTX_REL)
            s = ctx_subject(rel, rng.choice([w for w in NONCE["ctx-train"] if w not in used]))
            used.add(s.split()[-1])
            q = D.REL[rel]["q"][rng.choice(list(D.TRAIN_Q))].format(s=s)
            if rng.random() < 0.15:  # never told: decline
                return [], [], [["u", q, True], ["s", D.DECLINE_REPLY, True]]
            fact = answer(rel, s, ctx_value(rng, rel, NONCE["ctx-train"]))
            return [["u", fact, True], ["s", "ok! i will remember that.", True]], [], [["u", q, True], ["s", fact, True]]
        if kind == "correct":
            k = rng.randrange(len(D.TELL))
            v1, v2 = rng.sample(NONCE["word-train"] if k == 0 else D.COLOURS, 2)
            fixed = rng.random() < 0.7
            fix = [["u", D.FIX[k].format(v=v2), True], ["s", D.FIX_ACK.format(v=v2), True]] if fixed else []
            return ([["u", D.TELL[k].format(v=v1), True], ["s", D.TELL_ACK[k].format(v=v1), True]], fix,
                    [["u", D.TELL_ASK[k], True], ["s", D.TOLD[k].format(v=v2 if fixed else v1), True]])
        n = self.name(rng, used)
        used.add(n.lower())
        if kind == "name":
            (tu, ts), (au, as_) = rng.choice(NAME_TELL), rng.choice(NAME_ASK)
            f = {"N": n}
        elif kind == "pet" and rng.random() < 0.6:
            (tu, ts), (au, as_) = rng.choice(PET_TELL), rng.choice(PET_ASK)
            f = {"N": n, "p": rng.choice(P.PETS[0])}
        elif kind == "pet":
            (tu, ts), (au, as_) = rng.choice(REL_TELL), rng.choice(REL_ASK)
            f = {"N": n, "r": rng.choice(P.RELATIONS[0])}
        else:
            (tu, ts), (au, as_) = LIKE_TELL, LIKE_ASK
            f = {"N": n, "r": rng.choice(P.RELATIONS[0]), "x": rng.choice(P.LIKES[0])}
        return ([["u", tu.format(**f), True], ["s", ts.format(**f), True]], [],
                [["u", au.format(**f), True], ["s", as_.format(**f), True]])

    def conv(self, rng: random.Random) -> tuple[list, list[int]]:
        """Turns, and the reply positions (turn indices) of the answers to remembered things."""
        short = rng.random() < 0.2  # a fifth like the short mixture's: little before, one thing, asked soon
        turns = self.talk_n(rng, rng.randint(0, 2) if short else min(rng.randint(0, 16), rng.randint(0, 16)))
        due, used = [], set()
        kinds = rng.choices(list(ITEMS), list(ITEMS.values()), k=1 if short else rng.choice((1, 1, 2, 2, 3)))
        for kind in kinds:
            tell, fix, ask = self.item(rng, kind, used)
            turns += tell
            gap = rng.choice(GAPS[:7] if short else GAPS)
            at = len(turns) // 2 + gap
            if fix:
                due.append((len(turns) // 2 + rng.randint(0, max(0, gap - 1)), fix, False))
            due.append((at, ask, True))
            turns += self.talk_n(rng, rng.choice((0, 0, 1, 2)))
        answers = []
        for at, t, is_ask in sorted(due, key=lambda x: (x[0], x[2])):
            while len(turns) // 2 < at:
                turns += self.talk_ex(rng)
            turns += t
            if is_ask:
                answers.append(len(turns) - 1)
                if rng.random() < 0.1 and t[1][1] != D.DECLINE_REPLY:
                    turns += [["u", FOLLOW[0], True], ["s", FOLLOW[1].format(a=t[1][1]), True]]
        turns += self.talk_n(rng, rng.choice((0, 0, 1)))
        return turns, answers


def encode(turns: list) -> tuple[np.ndarray, np.ndarray]:
    """As fine_tune.encode, with names: a capitalised word is a name, a cast slot ('*' and its spelling the first
    time, then its digit), so the Spectrum's reply writes the digit."""
    table: list[str] = []

    def sub(m):
        w = m.group().lower()
        if w in table:
            return str(table.index(w) + 1)
        table.append(w)
        return "*" + w

    text, role = ["#"], [0]
    for r, t, loss in turns:
        body = re.sub(r"\b[A-Z][a-z]+\b", sub, t).lower() + (">" if r == "u" else "<")
        text.append(body)
        role += [1 if r == "u" else (2 if loss else 0)] * len(body)
    assert len(table) <= NSLOTS, table
    return to_ids("".join(text)), np.array(role, np.uint8)


def build_long(total: int = int(10e6)) -> dict:
    b = Builder(exchanges("train", data3_used(total // 2), 60000, "diag-long-talk"))
    rng = random.Random("diag-long")
    diag, n, lens = [], 0, []
    while n < total // 2:
        turns, _ = b.conv(rng)
        ids, role = encode(turns)
        if len(ids) > MAXLEN:
            continue
        diag.append((ids, role))
        n += len(ids)
        lens.append(len(ids))
    base, m = data3_half(total)
    mix = diag + base
    rng.shuffle(mix)
    meta = {"diag_symbols": n, "data3_symbols": m, "conversations": len(diag), "source": "train/j/fine_tune.py long",
            "length_median": int(np.median(lens)), "length_p90": int(np.percentile(lens, 90))}
    write("diag-long", mix, meta)
    return meta

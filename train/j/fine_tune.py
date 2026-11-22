"""The data J's two films builds were fine-tuned on: half conversations that teach facts, instructions and remembering,
half data3's own conversations. Everything is seeded, so a rebuild is identical.

    python -m train.j.fine_tune short    # data/chat/diag-ft:   10M symbols (the dg1 build's fine-tune)
    python -m train.j.fine_tune long     # data/chat/diag-long: 10M symbols (the long1kr build's)

A conversation is a list of turns [role, text, loss]: role 'u' you typed, 's' the Spectrum's reply; loss False marks
a reply training must not learn to write (the wrong answer you then correct): its symbols get role 0, read but not
predicted. The splits, all fixed here:
  facts         a quarter of each relation's facts never trained; each trained fact in two of its three training
                wordings; the last three wordings of every relation never trained
  arbitrary     300 made-up animals with random foods, 240 trained, 60 held (pure memory, nothing to infer)
  context       a fact about a made-up thing told in the conversation, asked 0 to 4 exchanges later (sometimes
                never told: the answer is to decline); made-up words for testing never in training
  instructions  spelling, repeating, choosing one of two, a corrected fact, the Spectrum corrected

Long conversations ("long"): 0-16 exchanges of talk first (data3 exchanges from conversations neither half trains on
whole, small talk, trained facts, instructions, the Spectrum corrected), then one to three things to remember (a
made-up thing's fact, a secret word or favourite colour perhaps corrected, your name, a pet's or a relative's name,
what someone likes), each asked 0-14 exchanges later with more talk between; now and then a follow-up. Kept under
1,000 symbols, so a 1,024-symbol window holds every one whole.

Made-up words are checked against data/words.json (train/story/tinystories.py), so none is a word of the stories.
"""
from __future__ import annotations

import json
import random
import sys

import numpy as np

from train import DATA
from train.j import facts as D
from train.j import pools as P
from train.j.format import PLAIN, to_ids

ROOT = DATA / "chat"
DATA3 = ROOT / "data3"
LEGS = ["two", "four", "six", "eight", "ten", "no"]
CTX_REL = ["capital", "colour", "legs"]
SHARES = {"facts": 0.24, "arbitrary": 0.16, "context": 0.20, "spell": 0.10, "repeat": 0.08, "choose": 0.06,
          "correct": 0.08, "corrected": 0.08}  # of the conversations that are not data3's
WORDS = json.loads((DATA / "words.json").read_text())
REAL = set(WORDS)


def nonce(n: int, seed: str, avoid: set) -> list[str]:
    """Made-up words: two or three syllables, none a word of the story data or of the facts."""
    rng, out = random.Random(seed), []
    while len(out) < n:
        w = "".join(rng.choice("bdfgklmnprstvz") + rng.choice("aeiou") for _ in range(rng.choice((2, 2, 3))))
        w += rng.choice("klmnprst") if rng.random() < 0.5 else ""
        if w not in REAL and w not in avoid:
            avoid.add(w)
            out.append(w)
    return out


FACT_WORDS = {w for k, r in D.REL.items() if k != "arbitrary" for f in r["facts"] for x in f for w in x.split()}
_USED = set(FACT_WORDS)
NONCE = {k: nonce(n, "diag-" + k, _USED) for k, n in (("arbitrary", 300), ("ctx-train", 600), ("ctx-test", 400),
                                                       ("word-train", 300), ("word-test", 120))}
D.REL["arbitrary"]["facts"] = [("a " + w, random.Random("diag-food-" + w).choice(D.FOODS))
                               for w in NONCE["arbitrary"]]
COMMON = [w for w, _ in sorted(WORDS.items(), key=lambda x: -x[1]) if w.isalpha() and 3 <= len(w) <= 7][:600]
SPELL_TRAIN, SPELL_TEST = COMMON[100:400], COMMON[400:500]  # the hundred commonest left out (a, the, ...)


def split(rel: str) -> tuple[list, list]:
    """(trained, held) facts of a relation: a seeded quarter held."""
    fs = list(D.REL[rel]["facts"])
    random.Random("diag-split-" + rel).shuffle(fs)
    held = fs[::4]
    return [f for f in fs if f not in held], held


SPLITS = {r: split(r) for r in D.REL}


def values(rel: str) -> list[str]:
    return sorted({v for _, v in D.REL[rel]["facts"]} | (set(D.FOODS) if rel == "arbitrary" else set()))


def answer(rel: str, s: str, v: str) -> str:
    return D.REL[rel]["a"].format(s=s, v=v)


def wordings(rel: str, i: int) -> tuple[list[int], int]:
    """A trained fact's two training wordings and its third one, by its place in the trained list."""
    return [t for t in D.TRAIN_Q if t != i % 3], i % 3


# ---- short conversations ---------------------------------------------------------------------------------------------

def _pad(rng: random.Random, turns: list) -> list:
    """A greeting before, sometimes."""
    return ([["u", g, True], ["s", r, True]] for g, r in [rng.choice(D.GREET)]).__next__() + turns \
        if rng.random() < 0.3 else turns


def _filler(rng: random.Random, n: int) -> list:
    out = []
    for _ in range(n):
        u, s = rng.choice(D.FILLER)
        out += [["u", u, True], ["s", s, True]]
    return out


def fact_conv(rng: random.Random, rel_pool: list[str]) -> list:
    turns = []
    for _ in range(rng.choice((1, 2, 3))):
        rel = rng.choice(rel_pool)
        trained = SPLITS[rel][0]
        i = rng.randrange(len(trained))
        s, v = trained[i]
        q = D.REL[rel]["q"][rng.choice(wordings(rel, i)[0])].format(s=s)
        turns += [["u", q, True], ["s", answer(rel, s, v), True]]
        if rng.random() < 0.2:
            turns += _filler(rng, 1)
    return _pad(rng, turns)


def ctx_value(rng: random.Random, rel: str, pool: list[str]) -> str:
    return rng.choice(pool) if rel == "capital" else rng.choice(D.COLOURS if rel == "colour" else LEGS)


def ctx_subject(rel: str, w: str) -> str:
    return w if rel == "capital" else "a " + w


def context_conv(rng: random.Random, pool: list[str]) -> list:
    rel = rng.choice(CTX_REL)
    s = ctx_subject(rel, rng.choice(pool))
    q = D.REL[rel]["q"][rng.choice(list(D.TRAIN_Q))].format(s=s)
    if rng.random() < 0.15:  # never told: decline
        return _pad(rng, _filler(rng, rng.choice((0, 1))) + [["u", q, True], ["s", D.DECLINE_REPLY, True]])
    v = ctx_value(rng, rel, pool)
    fact, dist = answer(rel, s, v), rng.choice((0, 1, 1, 2, 2, 3, 4))
    if dist == 0:
        return _pad(rng, [["u", f"{fact} {q}", True], ["s", fact, True]])
    return _pad(rng, [["u", fact, True], ["s", "ok! i will remember that.", True]] + _filler(rng, dist - 1)
                + [["u", q, True], ["s", fact, True]])


def spell_conv(rng: random.Random, words: list[str], forms: list[str]) -> list:
    w = rng.choice(words)
    return _pad(rng, [["u", rng.choice(forms).format(w=w), True], ["s", "-".join(w) + ".", True]])


def repeat_conv(rng: random.Random, words: list[str], forms: list[str]) -> list:
    p = " ".join(rng.choice(words) for _ in range(rng.choice((1, 2, 3))))
    return _pad(rng, [["u", rng.choice(forms).format(p=p), True], ["s", p + ".", True]])


def choose_conv(rng: random.Random, words: list[str], forms: list[str]) -> list:
    a, b = rng.sample(words, 2)
    x = rng.choice((a, b))
    return _pad(rng, [["u", rng.choice(forms).format(a=a, b=b), True],
                      ["s", rng.choice(D.CHOSEN).format(x=x), True]])


def correct_conv(rng: random.Random, pool: list[str]) -> list:
    k = rng.randrange(len(D.TELL))
    v1, v2 = rng.sample(pool if k == 0 else D.COLOURS, 2)
    fixed = rng.random() < 0.7
    turns = [["u", D.TELL[k].format(v=v1), True], ["s", D.TELL_ACK[k].format(v=v1), True]]
    turns += _filler(rng, rng.choice((0, 1)))
    if fixed:
        turns += [["u", D.FIX[k].format(v=v2), True], ["s", D.FIX_ACK.format(v=v2), True]]
    turns += _filler(rng, rng.choice((0, 1)))
    return _pad(rng, turns + [["u", D.TELL_ASK[k], True], ["s", D.TOLD[k].format(v=v2 if fixed else v1), True]])


def corrected_conv(rng: random.Random, rel: str, s: str, v: str, q: str, reask: bool) -> list:
    """You ask, the Spectrum answers wrongly (read, never trained), you correct it."""
    wrong = rng.choice([x for x in values(rel) if x != v])
    turns = [["u", q, True], ["s", answer(rel, s, wrong), False], ["u", D.WRONG_FIX.format(v=v), True],
             ["s", D.SORRY.format(a=answer(rel, s, v)), True]]
    if reask:
        turns += _filler(rng, rng.choice((0, 1))) + [["u", q, True], ["s", answer(rel, s, v), True]]
    return turns


def train_conv(rng: random.Random, kind: str) -> list:
    real = [r for r in D.REL if r != "arbitrary"]
    if kind == "facts":
        return fact_conv(rng, real)
    if kind == "arbitrary":
        return fact_conv(rng, ["arbitrary"])
    if kind == "context":
        return context_conv(rng, NONCE["ctx-train"])
    words = SPELL_TRAIN + NONCE["word-train"]
    if kind == "spell":
        return spell_conv(rng, words, D.SPELL[0])
    if kind == "repeat":
        return repeat_conv(rng, words, D.REPEAT[0])
    if kind == "choose":
        return choose_conv(rng, words, D.CHOOSE[0])
    if kind == "correct":
        return correct_conv(rng, NONCE["word-train"])
    rel = rng.choice(real)
    trained = SPLITS[rel][0]
    i = rng.randrange(len(trained))
    s, v = trained[i]
    return corrected_conv(rng, rel, s, v, D.REL[rel]["q"][rng.choice(wordings(rel, i)[0])].format(s=s),
                          rng.random() < 0.5)


def fact_only(rng: random.Random, rels: list[str]) -> list:
    """One trained fact's question and answer, in a training wording."""
    rel = rels[0] if len(rels) == 1 else rng.choice(rels)
    trained = SPLITS[rel][0]
    i = rng.randrange(len(trained))
    s, v = trained[i]
    return [["u", D.REL[rel]["q"][rng.choice(wordings(rel, i)[0])].format(s=s), True], ["s", answer(rel, s, v), True]]


def encode(turns: list) -> tuple[np.ndarray, np.ndarray]:
    """Symbols and roles (0 '#' and unlearned replies, 1 you, 2 the Spectrum)."""
    text, role = ["#"], [0]
    for r, t, loss in turns:
        body = t.lower() + (">" if r == "u" else "<")
        assert set(body[:-1]) <= PLAIN, body
        text.append(body)
        role += [1 if r == "u" else (2 if loss else 0)] * len(body)
    return to_ids("".join(text)), np.array(role, np.uint8)


# ---- data3's conversations, and writing a mixture ------------------------------------------------------------------

def data3_convs(rng: random.Random, chars: int) -> list[tuple[np.ndarray, np.ndarray]]:
    ids = np.fromfile(DATA3 / "train.u8", dtype=np.uint8)
    role = np.fromfile(DATA3 / "train.role", dtype=np.uint8)
    starts = np.flatnonzero(ids == 0)
    order = list(range(len(starts) - 1))
    rng.shuffle(order)
    out, got = [], 0
    for i in order:
        a, b = starts[i], starts[i + 1]
        out.append((ids[a:b], role[a:b]))
        got += b - a
        if got >= chars:
            break
    return out


def data3_half(total: int) -> tuple[list, int]:
    """Half the mixture: data3's conversations, drawn at random (the same draw for both mixtures), whole."""
    base, n = [], 0
    for c in data3_convs(random.Random("diag-mix"), total // 2):
        if n + len(c[0]) > total // 2:
            break
        base.append(c)
        n += len(c[0])
    return base, n


def write(name: str, convs: list, meta: dict):
    d = ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    np.concatenate([c[0] for c in convs]).tofile(d / "train.u8")
    np.concatenate([c[1] for c in convs]).tofile(d / "train.role")
    for ext in ("u8", "role"):
        (d / f"dev.{ext}").write_bytes((DATA3 / f"dev.{ext}").read_bytes())
    (d / "meta.json").write_text(json.dumps(meta, indent=1))


def build_short(total: int = int(10e6)) -> dict:
    rng = random.Random("diag-train")
    diag, got = [], {k: 0 for k in SHARES}
    for k, share in SHARES.items():
        while got[k] < share * total / 2:
            c = encode(train_conv(rng, k))
            diag.append(c)
            got[k] += len(c[0])
    base, n = data3_half(total)
    mix = diag + base
    rng.shuffle(mix)
    meta = {"diag_symbols": got, "data3_symbols": n, "source": "train/j/fine_tune.py short"}
    write("diag-ft", mix, meta)
    return meta


if __name__ == "__main__":
    from train.j.fine_tune_long import build_long  # noqa: PLC0415

    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which not in ("short", "long"):
        raise SystemExit("python -m train.j.fine_tune short|long")
    print(json.dumps(build_short() if which == "short" else build_long(), indent=1))

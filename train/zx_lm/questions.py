"""ZX-LM questions: what to ask, the exact answer, the statements it rests on, its tags.

generate() writes one whole conversation. Every question carries gold (the answer's content), accept (every answer
text scoring accepts), support (the user turns the answer rests on; empty when it can't be answered) and tags (the
categories to split results by).
"""
from __future__ import annotations

import itertools
import random
import re

from train.zx_lm.frames import ANSWERS, QUESTIONS
from train.zx_lm.world import Conversation, Obj

FOCUS_P = 0.5  # focus=True: how often a question that could be a hard one is
TYPE_WEIGHTS = {"has": 1.2, "what": 1.0, "yn": 1.0, "first": 0.5, "prev": 0.5, "before": 0.5, "giver": 0.6,
                "recip": 0.6, "unknown": 0.35, "follow": 1.5}


def mentioned(c: Conversation) -> list[str]:
    said = " ".join(t["text"] for t in c.turns if t["speaker"] == "u")
    return [p for p in c.people if re.search(rf"\b{p}\b", said)]


def ever_owned(c: Conversation, n: str) -> bool:
    return any(e.owner == n for o in c.introduced() for e in o.entries)


# ---- answers: (answer kind, format values, gold, support)
def a_has(o: Obj):
    e = o.entries[-1]
    if e.owner is None:
        return "has_nobody", {}, {"kind": "nobody"}, e.stmts
    return "has", {"O": e.owner, "X": o.noun}, {"kind": "ent", "items": [e.owner]}, e.stmts


def a_what(c: Conversation, n: str):
    owned = [o for o in c.introduced() if o.owner == n]
    if owned:
        sup = [t for o in owned for t in o.entries[-1].stmts]
        nouns = [o.noun for o in owned][:3]
        return f"what_{len(nouns)}", {"N": n, "objs": nouns}, {"kind": "set", "items": sorted(nouns)}, sup
    sup = [t for o in c.introduced() for i, e in enumerate(o.entries[:-1]) if e.owner == n
           for t in o.entries[i + 1].stmts]
    return "what_none", {}, {"kind": "nothing"}, sup


def a_yn(o: Obj, n: str):
    e = o.entries[-1]
    if e.owner == n:
        return "yes", {"N": n}, {"kind": "yes"}, e.stmts
    if e.owner is None:
        return "no_nobody", {}, {"kind": "no", "owner": None}, e.stmts
    return "no_owner", {"O": e.owner}, {"kind": "no", "owner": e.owner}, e.stmts


def a_first(o: Obj):
    e = o.entries[0]
    return "first", {"O": e.owner}, {"kind": "ent", "items": [e.owner]}, e.stmts


def a_prev(o: Obj):
    last = o.entries[-1]
    earlier = [e for e in o.entries[:-1] if e.owner and e.owner != last.owner]
    if not earlier:
        return "nobody_before", {}, {"kind": "nobody"}, o.entries[0].stmts
    e = earlier[-1]
    return "prev", {"O": e.owner}, {"kind": "ent", "items": [e.owner]}, e.stmts + last.stmts


def before_index(o: Obj, q: str) -> tuple[int, int] | None:
    js = [j for j, e in enumerate(o.entries) if j >= 1 and e.owner == q]
    if not js:
        return None
    ks = [k for k in range(js[-1]) if o.entries[k].owner and o.entries[k].owner != q]
    return (ks[-1], js[-1]) if ks else None


def a_before(o: Obj, q: str):
    k, j = before_index(o, q)
    e = o.entries[k]
    return "first" if False else "prev", {"O": e.owner}, {"kind": "ent", "items": [e.owner]}, \
        e.stmts + o.entries[j].stmts


def a_giver(o: Obj, q: str):
    e = [e for e in o.entries if e.how == "give" and e.owner == q][-1]
    return "giver", {"O": e.giver, "Q": q}, {"kind": "ent", "items": [e.giver]}, e.stmts


def a_recip(o: Obj, n: str):
    e = [e for e in o.entries if e.how == "give" and e.giver == n][-1]
    return "recip", {"O": e.owner}, {"kind": "ent", "items": [e.owner]}, e.stmts


def accept_texts(kind: str, fmt: dict) -> list[str]:
    if kind.startswith("what_") and kind != "what_none":
        out = []
        for perm in itertools.permutations(fmt["objs"]):
            vals = dict(zip("ABC", perm), N=fmt["N"])
            out += [f.format(**vals) for f in ANSWERS[kind]]
        return sorted(set(out))
    return [f.format(**fmt) for f in ANSWERS[kind]]


# ---- choosing a question
def options(c: Conversation) -> dict[str, list[tuple[dict, str | None]]]:
    """question type -> [(arguments, held composition or None)]"""
    intro, ppl = c.introduced(), mentioned(c)
    owners = [n for n in ppl if ever_owned(c, n)]
    corrected = lambda o: any(e.corrected for e in o.entries)  # noqa: E731
    gave = {e.giver for o in intro for e in o.entries if e.how == "give"}
    received = {o.owner for o in intro if o.entries[-1].how in ("give", "find")}
    opts: dict[str, list] = {t: [] for t in TYPE_WEIGHTS}
    for o in intro:
        opts["has"].append(({"o": o}, None))
        for n in ppl:
            opts["yn"].append(({"o": o, "n": n}, None))
        opts["first"].append(({"o": o}, "a" if corrected(o) else None))
        if o.owner:
            opts["prev"].append(({"o": o}, "a" if corrected(o) else None))
        for q in ppl:
            if before_index(o, q):
                opts["before"].append(({"o": o, "q": q}, "a" if corrected(o) else None))
            if any(e.how == "give" and e.owner == q for e in o.entries):
                opts["giver"].append(({"o": o, "q": q}, None))
            if any(e.how == "give" and e.giver == q for e in o.entries):
                opts["recip"].append(({"o": o, "n": q}, None))
    for n in owners:
        if sum(o.owner == n for o in intro) > 3:
            continue
        opts["what"].append(({"n": n}, "b" if n in gave and n in received else None))
    opts["unknown"] = [({"t": "has", "x": c.fresh[0]}, None), ({"t": "what", "n": c.stranger}, None)]
    if intro and ppl:
        opts["unknown"].append(({"t": "yn", "x": c.fresh[1], "n": c.rng.choice(ppl)}, None))
    p = c.prev
    if p.get("event") == "q" and p["type"] in ("has", "first", "prev"):
        comp = None if p["type"] == "has" else "c"
        opts["follow"] = [({"t": p["type"], "o": o}, comp) for o in intro
                          if o is not p.get("obj") and (p["type"] != "prev" or o.owner)]
    elif p.get("event") == "q" and p["type"] == "what":
        opts["follow"] = [({"t": "what", "n": n}, None) for n in owners if n != p.get("n")]
    return opts


def hard(t: str, a: dict, c: Conversation) -> bool:
    """What v2 trained on least: asked about a move the object has since moved on from; a list of two or more."""
    if t == "recip":
        return [e for e in a["o"].entries if e.how == "give" and e.giver == a["n"]][-1] is not a["o"].entries[-1]
    if t == "giver":
        return [e for e in a["o"].entries if e.how == "give" and e.owner == a["q"]][-1] is not a["o"].entries[-1]
    return t == "what" and sum(o.owner == a["n"] for o in c.introduced()) >= 2


def pick(c: Conversation) -> tuple[str, dict, str | None]:
    opts = options(c)
    if c.split == "train":
        opts = {t: [x for x in xs if x[1] is None] for t, xs in opts.items()}
    if c.unknown and c.rng.random() < c.unknown:
        return ("unknown", *c.rng.choice(opts["unknown"]))
    if c.focus:
        hs = [(t, x) for t in ("recip", "giver", "what") for x in opts[t] if hard(t, x[0], c)]
        if hs and c.rng.random() < c.focus:
            t, x = c.rng.choice(hs)
            return (t, *x)
    types = [t for t in TYPE_WEIGHTS if opts[t]]
    t = c.rng.choices(types, [TYPE_WEIGHTS[x] for x in types])[0]
    held = [x for x in opts[t] if x[1]]
    if held and c.rng.random() < c.comp_p:
        return (t, *c.rng.choice(held))
    return (t, *c.rng.choice(opts[t]))


def ask(c: Conversation) -> None:
    t, a, comp = pick(c)
    o, n, q = a.get("o"), a.get("n"), a.get("q")
    follow = t == "follow"
    base = a["t"] if t in ("follow", "unknown") else t
    if t == "unknown":
        qtext = c.frame(QUESTIONS, base).format(X=a.get("x", ""), N=n or "")
        kind, fmt, gold, sup = "unknown", {}, {"kind": "unknown"}, []
    else:
        if follow:
            qtext = c.frame(QUESTIONS, "follow_n" if base == "what" else "follow_x").format(
                X=c.obj_text(o, False) if o else "", N=n or "")
        elif base == "before":
            qtext = c.frame(QUESTIONS, "before").format(X=c.obj_text(o, False), Q=q)
        elif base == "giver":
            qtext = c.frame(QUESTIONS, "giver").format(X=c.obj_text(o, False), Q=q)
        else:
            qtext = c.frame(QUESTIONS, base).format(X=c.obj_text(o, False) if o else "", N=n or "")
        kind, fmt, gold, sup = {
            "has": lambda: a_has(o), "what": lambda: a_what(c, n), "yn": lambda: a_yn(o, n),
            "first": lambda: a_first(o), "prev": lambda: a_prev(o), "before": lambda: a_before(o, q),
            "giver": lambda: a_giver(o, q), "recip": lambda: a_recip(o, n)}[base]()
        fmt.setdefault("X", o.noun if o else "")
    stmts = [i for i, x in enumerate(c.turns) if x["speaker"] == "u" and "event" in x]
    events = {i: c.turns[i]["event"] for i in stmts}
    objs = [o] if o else ([x for x in c.introduced() if x.owner == n or any(e.owner == n for e in x.entries)]
                          if n else [])
    tags = {
        "type": base, "follow": follow, "unknown": t == "unknown", "comp": comp,
        "explicit": any(events.get(s) in ("correct", "correct_own") for s in sup),
        "inferred": any(events.get(s) in ("give", "lose", "find") for s in sup),
        "earlier": base in ("first", "prev", "before"),
        "chain": max([len(x.entries) for x in objs] or [0]),
        "moved": sum(len(x.entries) >= 2 for x in c.introduced()),
        "latest_irrelevant": bool(sup) and stmts[-1] not in sup,
        "negation": any(x.negs for x in objs),
        "held_frame": c.held_used,
        "moved_on": hard(t, a, c) if t in ("recip", "giver") else None,
    }
    accept = accept_texts(kind, fmt)
    if o and o.adj and "{X}" in "".join(ANSWERS[kind]):
        accept += accept_texts(kind, {**fmt, "X": f"{o.adj} {o.noun}"})
    qi = c.say(qtext, q=True, tags=tags, gold=gold, support=sorted(set(sup)))
    c.reply(sorted(set(accept)), answer_kind=kind, gold=gold, q_turn=qi)
    c.prev = {"event": "q", "type": base, "obj": o, "n": n}


def generate(rng: random.Random, pool: dict, split: str, held_p: float = 0.5, comp_p: float = 0.6,
             version: int = 1, focus: bool | float = False, unknown: float = 0.0) -> list[dict]:
    if version == 2:
        from train.zx_lm.world2 import Conversation2  # noqa: PLC0415
        c = Conversation2(rng, pool, split, held_p)
    else:
        c = Conversation(rng, pool, split, held_p)
    c.comp_p, c.focus, c.unknown = comp_p, FOCUS_P if focus is True else focus, unknown
    for _ in range(rng.randint(3, 10)):
        c.statement()
        c.reply(ANSWERS["ack"], ack=True)
        if rng.random() < 0.35:
            ask(c)
            while rng.random() < 0.3 and options(c)["follow"]:
                ask(c)
    for _ in range(rng.randint(1, 3)):
        ask(c)
    return c.turns

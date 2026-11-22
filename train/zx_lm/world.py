"""ZX-LM conversation generator: who owns what, how it changes, and questions about it.

Every conversation is written from a simulated world, so every answer and the statements it rests on are known
exactly. A conversation is a list of turns, user and Spectrum alternating: the Spectrum says "ok." to a statement and
answers a question. split="train" uses training frames and never makes a held-out composition (HELD_COMPOSITIONS);
split="test" draws held frames with probability held_p and prefers held compositions where one is possible.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from train.zx_lm.frames import STATEMENTS

HELD_COMPOSITIONS = {
    "a": "an earlier-owner question about an object whose history was corrected",
    "b": "what a person has, when they have both received something and given something away",
    "c": "a follow-up about another object after an earlier-owner question",
}


@dataclass
class Entry:
    owner: str | None
    how: str  # own, give, lose, find
    giver: str | None
    stmts: list[int]
    corrected: bool = False


@dataclass
class Obj:
    noun: str
    adj: str | None
    entries: list[Entry] = field(default_factory=list)
    negs: list[str] = field(default_factory=list)

    @property
    def owner(self) -> str | None:
        return self.entries[-1].owner if self.entries else None


class Conversation:
    def __init__(self, rng: random.Random, pool: dict, split: str, held_p: float = 0.5):
        self.rng, self.split, self.held_p = rng, split, held_p if split == "test" else 0.0
        self.people = rng.sample(pool["names"], rng.randint(3, 5))
        nouns = rng.sample(pool["objects"], 6)
        adjs = rng.sample(pool["adjs"], 2)
        self.objs = [Obj(n, rng.choice(adjs) if rng.random() < 0.2 else None) for n in nouns[: rng.randint(2, 4)]]
        self.fresh = nouns[4:]  # objects never mentioned: for questions it can't answer
        self.stranger = rng.choice([n for n in pool["names"] if n not in self.people])
        self.turns: list[dict] = []
        self.prev: dict = {}
        self.held_used = False

    # ---- writing
    def frame(self, table: dict, key: str) -> str:
        fs = table[key]
        held = [f for f, s in fs if s == "held"]
        if held and self.rng.random() < self.held_p:
            self.held_used = True
            return self.rng.choice(held)
        return self.rng.choice([f for f, s in fs if s == "train"])

    def obj_text(self, o: Obj, new: bool) -> str:
        if o.adj and (new or self.rng.random() < 0.3):
            return f"{o.adj} {o.noun}"
        return o.noun

    def say(self, text: str, **meta) -> int:
        text = re.sub(r"\ba ([aeiou])", r"an \1", text)
        self.turns.append({"speaker": "u", "text": text, **meta})
        return len(self.turns) - 1

    def reply(self, accept: list[str], **meta) -> None:
        self.turns.append({"speaker": "s", "text": self.rng.choice(accept), "accept": accept, **meta})

    def introduced(self) -> list[Obj]:
        return [o for o in self.objs if o.entries]

    # ---- statements
    def statement(self) -> None:
        rng, intro = self.rng, self.introduced()
        owned = [o for o in intro if o.owner]
        lost = [o for o in intro if not o.owner]
        pe = self.prev.get("event")
        cands = []
        if len(intro) < len(self.objs):
            cands.append(("own", 3.0 if not intro else 1.2))
        if owned:
            cands += [("give", 4.0), ("lose", 0.5)]
        if lost:
            cands.append(("find", 2.0))
        if pe == "give" and not self.prev.get("corrected"):
            cands.append(("correct", 1.5))
        if pe == "own" and not self.prev.get("corrected"):
            cands.append(("correct_own", 0.6))
        if intro:
            cands += [("neg", 0.7), ("want", 0.7)]
        ev = rng.choices([c for c, _ in cands], [w for _, w in cands])[0]
        getattr(self, "ev_" + ev)(owned, lost)

    def ev_own(self, owned, lost) -> None:
        o = next(o for o in self.objs if not o.entries)
        p = self.rng.choice(self.people)
        t = self.say(self.frame(STATEMENTS, "own").format(P=p, X=self.obj_text(o, True)), event="own")
        o.entries.append(Entry(p, "own", None, [t]))
        self.prev = {"event": "own", "obj": o, "P": p}

    def ev_give(self, owned, lost) -> None:
        o = self.rng.choice(owned)
        p, q = o.owner, self.rng.choice([x for x in self.people if x != o.owner])
        if self.prev.get("obj") is o and self.rng.random() < 0.4:
            f = self.frame(STATEMENTS, "give_it")
        else:
            f = self.frame(STATEMENTS, "give")
        t = self.say(f.format(P=p, Q=q, X=self.obj_text(o, False)), event="give")
        o.entries.append(Entry(q, "give", p, [t]))
        self.prev = {"event": "give", "obj": o, "P": p, "Q": q}

    def ev_lose(self, owned, lost) -> None:
        o = self.rng.choice(owned)
        t = self.say(self.frame(STATEMENTS, "lose").format(P=o.owner, X=self.obj_text(o, False)), event="lose")
        o.entries.append(Entry(None, "lose", o.owner, [t]))
        self.prev = {"event": "lose", "obj": o}

    def ev_find(self, owned, lost) -> None:
        o = self.rng.choice(lost)
        q = self.rng.choice(self.people)
        t = self.say(self.frame(STATEMENTS, "find").format(Q=q, X=self.obj_text(o, False)), event="find")
        o.entries.append(Entry(q, "find", None, [t]))
        self.prev = {"event": "find", "obj": o, "Q": q}

    def ev_correct(self, owned, lost) -> None:
        o, p, q = self.prev["obj"], self.prev["P"], self.prev["Q"]
        r = self.rng.choice([x for x in self.people if x not in (p, q)] or [None])
        if r is None:
            return self.ev_want(owned, lost)
        f = self.frame(STATEMENTS, "correct")
        t = self.say(f.format(P=p, Q=q, R=r, X=self.obj_text(o, False)), event="correct")
        e = o.entries[-1]
        e.owner, e.corrected = r, True
        e.stmts.append(t)
        self.prev = {"event": "correct", "obj": o, "corrected": True}

    def ev_correct_own(self, owned, lost) -> None:
        o, p = self.prev["obj"], self.prev["P"]
        r = self.rng.choice([x for x in self.people if x != p])
        t = self.say(self.frame(STATEMENTS, "correct_own").format(P=p, R=r, X=self.obj_text(o, False)),
                     event="correct_own")
        e = o.entries[-1]
        e.owner, e.corrected = r, True
        e.stmts.append(t)
        self.prev = {"event": "correct_own", "obj": o, "corrected": True}

    def ev_neg(self, owned, lost) -> None:
        o = self.rng.choice(self.introduced())
        others = [x for x in self.people if x != o.owner]
        p = self.rng.choice(others)
        t = self.say(self.frame(STATEMENTS, "neg").format(P=p, X=self.obj_text(o, False)), event="neg")
        o.negs.append(p)
        self.prev = {"event": "neg", "obj": o, "t": t}

    def ev_want(self, owned, lost) -> None:
        o = self.rng.choice(self.introduced())
        p = self.rng.choice(self.people)
        self.say(self.frame(STATEMENTS, "want").format(P=p, X=self.obj_text(o, False)), event="want")
        self.prev = {"event": "want", "obj": o}

    # ---- questions: see questions.py

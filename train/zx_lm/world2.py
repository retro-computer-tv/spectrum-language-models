"""ZX-LM v2 conversations (frames2.py): v1's world and questions in a much wider language.

Added to v1's generator: every v2 frame; a pronoun for the person who acted in the statement before (he, she or
they, fixed per person, and possessives his, her, their); fillers before or after a statement ("then", "yesterday");
two people getting objects in one sentence; one person getting a list; a swap (two changes of owner at once); and a
change of owner that doesn't say how ("the kite now belongs to Ana": no giver).
"""
from __future__ import annotations

import random

from train.zx_lm.frames2 import FILLERS_END, FILLERS_START, PRONOUNS, QUESTIONS2, STATEMENTS2
from train.zx_lm.world import Conversation, Entry, Obj


class Frame(str):
    """A frame that fills {PP} and {QP} (possessives) from {P} and {Q}."""

    conv: "Conversation2"

    def format(self, **kw) -> str:  # noqa: A003
        for who, key in (("P", "PP"), ("Q", "QP")):
            if who in kw and kw[who] in self.conv.pron:
                kw.setdefault(key, self.conv.pron[kw[who]][1])
        return str.format(self, **kw)


class Conversation2(Conversation):
    ST = STATEMENTS2

    def __init__(self, rng: random.Random, pool: dict, split: str, held_p: float = 0.5):
        super().__init__(rng, pool, split, held_p)
        self.pron = {p: rng.choice(PRONOUNS) for p in self.people}
        self.last_actor: str | None = None

    def frame(self, table: dict, key: str) -> str:
        it = key == "give_it"
        key = "give" if it else key
        fs = self.ST[key] if key in self.ST else QUESTIONS2[key]
        if key == "give":
            fs = [(f, s) for f, s in fs if (" it " in f) == it]
        held = [f for f, s in fs if s == "held"]
        if held and self.rng.random() < self.held_p:
            self.held_used = True
            f = self.rng.choice(held)
        else:
            f = self.rng.choice([f for f, s in fs if s == "train"])
        out = Frame(f)
        out.conv = self
        return out

    def say(self, text: str, **meta) -> int:
        if not meta.get("q"):
            actor = next((p for p in self.people if text.startswith(p + " ")), None)
            if actor and actor == self.last_actor and self.rng.random() < 0.3:
                text = self.pron[actor][0] + text[len(actor):]
            self.last_actor = actor
            if self.rng.random() < 0.12:
                text = self.rng.choice(FILLERS_START) + " " + text
            elif text.endswith(".") and self.rng.random() < 0.08:
                text = text[:-1] + self.rng.choice(FILLERS_END) + "."
        return super().say(self.dress(text, meta), **meta)

    def dress(self, text: str, meta: dict) -> str:
        """Conversation3 adds unfamiliar words here."""
        return text

    def extra(self, cands: list, owned: list[Obj]) -> None:
        """Conversation3 adds events here."""

    def ev_lose(self, owned, lost) -> None:
        o = self.rng.choice(owned)
        f = self.frame_it("lose", self.prev.get("obj") is o)
        t = self.say(f.format(P=o.owner, X=self.obj_text(o, False)), event="lose")
        o.entries.append(Entry(None, "lose", o.owner, [t]))
        self.prev = {"event": "lose", "obj": o}

    def ev_find(self, owned, lost) -> None:
        o = self.rng.choice(lost)
        q = self.rng.choice(self.people)
        f = self.frame_it("find", self.prev.get("obj") is o)
        t = self.say(f.format(Q=q, X=self.obj_text(o, False)), event="find")
        o.entries.append(Entry(q, "find", None, [t]))
        self.prev = {"event": "find", "obj": o, "Q": q}

    def frame_it(self, key: str, it_ok: bool) -> str:
        """A frame of key; one that says "it" only when the object is the one the last statement was about."""
        while True:
            f = self.frame(None, key)
            if it_ok or not f.endswith(" it."):
                return f

    def new_objs(self) -> list[Obj]:
        return [o for o in self.objs if not o.entries]

    def statement(self) -> None:
        rng, intro, new = self.rng, self.introduced(), self.new_objs()
        owned = [o for o in intro if o.owner]
        lost = [o for o in intro if not o.owner]
        pe = self.prev.get("event")
        cands = []
        if new:
            cands.append(("own", 2.5 if not intro else 1.0))
        if len(new) >= 2:
            cands += [("own_pair", 0.6), ("own_list", 0.5)]
        if owned:
            cands += [("give", 4.0), ("move", 0.8), ("lose", 0.5)]
        if len({o.owner for o in owned}) >= 2:
            cands.append(("swap", 0.5))
        if lost:
            cands.append(("find", 2.0))
        if pe == "give" and not self.prev.get("corrected"):
            cands.append(("correct", 1.5))
        if pe == "own" and not self.prev.get("corrected"):
            cands.append(("correct_own", 0.6))
        if intro:
            cands += [("neg", 0.7), ("want", 0.7)]
        self.extra(cands, owned)
        ev = rng.choices([c for c, _ in cands], [w for _, w in cands])[0]
        getattr(self, "ev_" + ev)(owned, lost)

    def ev_own_pair(self, owned, lost) -> None:
        x, y = self.new_objs()[:2]
        p, q = self.rng.sample(self.people, 2)
        f = self.frame(None, "own_pair")
        t = self.say(f.format(P=p, Q=q, X=self.obj_text(x, True), Y=self.obj_text(y, True)), event="own")
        x.entries.append(Entry(p, "own", None, [t]))
        y.entries.append(Entry(q, "own", None, [t]))
        self.prev = {"event": "own_multi"}

    def ev_own_list(self, owned, lost) -> None:
        new = self.new_objs()
        f = self.frame(None, "own_list")
        while "{Z}" in f and len(new) < 3:
            f = self.frame(None, "own_list")
        objs = new[: 3 if "{Z}" in f else 2]
        p = self.rng.choice(self.people)
        vals = dict(zip("XYZ", (self.obj_text(o, True) for o in objs)))
        t = self.say(f.format(P=p, **vals), event="own")
        for o in objs:
            o.entries.append(Entry(p, "own", None, [t]))
        self.prev = {"event": "own_multi"}

    def ev_swap(self, owned, lost) -> None:
        x = self.rng.choice(owned)
        y = self.rng.choice([o for o in owned if o.owner != x.owner])
        p, q = x.owner, y.owner
        f = self.frame(None, "swap")
        t = self.say(f.format(P=p, Q=q, X=self.obj_text(x, False), Y=self.obj_text(y, False)), event="give")
        x.entries.append(Entry(q, "give", p, [t]))
        y.entries.append(Entry(p, "give", q, [t]))
        self.prev = {"event": "swap"}

    def ev_move(self, owned, lost) -> None:
        o = self.rng.choice(owned)
        q = self.rng.choice([x for x in self.people if x != o.owner])
        t = self.say(self.frame(None, "move").format(Q=q, X=self.obj_text(o, False)), event="give")
        o.entries.append(Entry(q, "move", None, [t]))
        self.prev = {"event": "move", "obj": o}

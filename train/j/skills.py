"""The procedural conversation generator: conversations that plant facts and ask for them later, with ten kinds of
exchange (greetings and names, everyday questions, likes and activities, follow-ups, remembering what was said,
keeping to a topic, simple explanations, questions it can't answer, corrections, turn-taking), in many wordings,
from large pools.

    Conv(rng, names).random_plan()                        # one training conversation
    Conv(rng, names, words="test", values="test")         # held-out wordings and values, to build tests from

A Conv is a list of exchanges (your turn, the Spectrum's reply), the names it uses, and checks: for some of your
turns, what a right reply must contain ("want"), must not contain ("avoid"), or that it should decline. Training
uses the replies written here; a test keeps only your turns and the checks, and lets the network write the replies.
"""
from __future__ import annotations

import random

from train.j import pools as P
from train.j import wordings as W


def typo(r: random.Random, you: str) -> str:
    """What people type: a quarter leave off the final full stop or question mark; 4% swap or drop a letter in one
    lower-case word (names, typed with a capital, are left alone)."""
    if r.random() < 0.25 and you[-1] in ".?!":
        you = you[:-1]
    if r.random() < 0.04:
        ws = you.split(" ")
        k = r.randrange(len(ws))
        w = ws[k]
        if len(w) > 3 and w.isalpha() and w.islower():
            i = r.randrange(len(w) - 1)
            ws[k] = w[:i] + w[i + 1] + w[i] + w[i + 2:] if r.random() < 0.5 else w[:i] + w[i + 1:]
            you = " ".join(ws)
    return you


class Conv:
    def __init__(self, rng: random.Random, names: list[str], words: str = "train", values: str = "train",
                 filler=None, noisy: bool = False):
        self.r, self.pool = rng, names
        self.wi, self.vi = int(words == "test"), int(values == "test")
        self.filler = filler  # rng -> (you, spec) from public dialogue, or None
        self.noisy = noisy  # training: some missing full stops and typos in what you type
        self.turns: list[tuple[str, str]] = []
        self.moves: list[str] = []
        self.checks: list[dict] = []
        self.names: set[str] = set()
        self.user: str | None = None
        self.facts: dict[str, dict] = {}  # what has been planted, by key ("pet:dog", "fav:colour", ...)
        self.topic: tuple[str, list[str]] | None = None  # (phrase in the Spectrum's words, words a reply must use)
        self.pending: str | None = None  # the Spectrum asked something; your next turn may answer it

    # --- helpers
    def w(self, pair) -> str:
        return self.r.choice(pair[self.wi])

    def v(self, pair, avoid=()) -> str:
        opts = [x for x in pair[self.vi] if x not in avoid]
        return self.r.choice(opts)

    def new_name(self) -> str | None:
        if len(self.names) >= 3:
            return None
        n = None
        while n is None or n in self.names:
            n = P.made_up_name(self.r) if self.noisy and self.r.random() < 0.25 else self.r.choice(self.pool)
        self.names.add(n)
        return n.capitalize()

    def say(self, you: str, spec: str, move: str, check: dict | None = None):
        if self.noisy:
            you = typo(self.r, you)
        self.turns += [("u", you), ("s", spec)]
        self.moves.append(move)
        if check:
            self.checks.append({"turn": len(self.moves) - 1, "move": move, **check})

    def reply(self, core: str, ask: str | None = None) -> str:
        r = self.r
        if r.random() < 0.12:
            core = r.choice(["oh, ", "ok! ", "yes! ", "hmm, ", "wow! "]) + core
        if ask and r.random() < 0.25 and ask in W.ASK_BACK and not self.done(ask):
            core += " " + W.ASK_BACK[ask]
            self.pending = ask
        return core

    def done(self, ask: str) -> bool:
        return {"name": self.user is not None, "pet": any(k.startswith("pet:") for k in self.facts),
                "colour": "fav:colour" in self.facts, "food": "fav:food" in self.facts,
                "act": "act" in self.facts, "feel": False}[ask]

    # --- moves: opening and your name
    def greet(self, intro: bool):
        g = self.w(W.GREET)
        if intro and (n := self.new_name()):
            self.user = n
            spec = self.reply(f"{self.r.choice(W.HELLO)} {n}! {self.r.choice(W.MEET)}.", "feel")
            self.say(f"{g} {self.w(W.INTRO).format(N=n)}", spec, "greet+intro", {"use": [n]})
        else:
            self.pending = "name"
            self.say(g, f"{self.r.choice(W.HELLO)}! what is your name?", "greet", {"any": ["hello", "hi", "hey"]})

    def recall_name(self):
        n = self.user
        self.say(self.w(W.ASK_NAME), self.r.choice(W.RECALL["name"]).format(N=n), "recall:name",
                 {"want": [n], "avoid": [x.capitalize() for x in self.names if x != n.lower()]})

    def answer_pending(self):
        p, self.pending = self.pending, None
        r = self.r
        if p == "name" and (n := self.new_name()):
            self.user = n
            self.say(self.w(W.ANSWER_NAME).format(N=n), f"{r.choice(W.MEET)}, {n}!", "answer:name", {"use": [n]})
        elif p == "feel":
            self.say(self.w(W.FEEL_FINE), r.choice(W.FINE_REPLY), "answer:feel")
        elif p == "pet" and (n := self.new_name()):
            pet = self.v(P.PETS)
            self.facts[f"pet:{pet}"] = {"P": pet, "N": n}
            self.topic = (f"your {pet}", [pet])
            self.say(f"yes, a {pet} called {n}.", r.choice(W.ACK["pet"]).format(P=pet, N=n), "answer:pet")
        elif p in ("colour", "food"):
            val = self.v(P.FAVOURITES[p])
            self.facts[f"fav:{p}"] = {"K": p, "V": val}
            self.topic = (f"your favourite {p}", [p, *val.split()])
            you = r.choice([f"{val}.", f"i like {val}.", f"{val}, i think."])
            self.say(you, r.choice(W.ACK["fav"]).format(V=val), "answer:" + p)
        elif p == "act":
            a = self.v(P.ACTIVITIES)
            self.facts["act"] = {"A": a}
            self.say(f"i like to {a}.", r.choice(W.ACK["act"]).format(A=a.replace("my ", "your ")), "answer:act")

    # --- moves: facts
    def fields(self, kind: str, avoid: dict) -> dict | None:
        """Fresh values for one fact of a kind (None if no name slot is left)."""
        v = self.v
        if kind in ("pet", "rel"):
            n = self.new_name()
            if n is None:
                return None
            key = "P" if kind == "pet" else "R"
            return {key: v(P.PETS if kind == "pet" else P.RELATIONS, avoid.get(key, ())), "N": n}
        if kind == "fav":
            k = self.r.choice([k for k in P.FAVOURITES if f"fav:{k}" not in self.facts] or ["colour"])
            return {"K": k, "V": v(P.FAVOURITES[k])}
        if kind == "obj":
            return {"O": v(P.OBJECTS, avoid.get("O", ())), "C": v(P.COLOURS, avoid.get("C", ()))}
        return {"act": {"A": v(P.ACTIVITIES)}, "place": {"PL": v(P.PLACES)}, "age": {"AGE": v(P.AGES)}}[kind]

    @staticmethod
    def key(kind: str, f: dict) -> str:
        sub = f.get("P") or f.get("R") or f.get("K") or f.get("O")
        return f"{kind}:{sub}" if sub else kind

    def about(self, kind: str, f: dict):
        words = {"pet": [f.get("P")], "rel": [f.get("R")], "fav": [f.get("K")], "obj": [f.get("O")],
                 "act": f.get("A", "").replace("my ", "").split()[-1:], "place": f.get("PL", "").split()[-1:],
                 "age": ["old", "age", f.get("AGE")]}[kind]
        phrase = {"pet": "your {P}", "rel": "your {R}", "fav": "your favourite {K}", "obj": "your {O}",
                  "act": "what you like to do", "place": "where you live", "age": "how old you are"}[kind]
        self.topic = (phrase.format(**f), [x for x in words if x])

    def plant(self, kind: str) -> bool:
        f = self.fields(kind, {})
        if f is None:
            return False
        self.facts[self.key(kind, f)] = f
        self.about(kind, f)
        g = {**f, "A": f.get("A", "").replace("my ", "your ")}
        self.say(self.w(W.PLANT[kind]).format(**f), self.reply(self.r.choice(W.ACK[kind]).format(**g)),
                 "plant:" + kind)
        return True

    def two(self, kind: str, separate: bool = False) -> tuple[str, str] | None:
        """Two facts of the same kind; returns their keys. separate: one turn each (a held-out structure)."""
        f1 = self.fields(kind, {})
        if f1 is None:
            return None
        sub = "P" if kind == "pet" else "R" if kind == "rel" else "O"
        f2 = self.fields(kind, {sub: [f1[sub]], "C": [f1.get("C")]})
        if f2 is None:
            return None
        k1, k2 = self.key(kind, f1), self.key(kind, f2)
        self.facts[k1], self.facts[k2] = f1, f2
        if separate:
            for f in (f1, f2):
                self.say(self.w(W.PLANT[kind]).format(**f), self.r.choice(W.ACK[kind]).format(**f), "plant:" + kind)
        else:
            both = {**f1, **{k + "2": x for k, x in f2.items()}}
            self.say(self.w(W.TWO[kind]).format(**both), self.r.choice(["got it!", "i will remember that.", "ok!",
                                                                         "lovely!", "how nice!"]), "plant2:" + kind)
        self.topic = (f"your {f1[sub]} and your {f2[sub]}", [f1[sub], f2[sub]])
        return k1, k2

    def recall(self, key: str):
        kind = key.split(":")[0]
        f = self.facts[key]
        g = {**f, "A": f.get("A", "").replace("my ", "your ")}
        answer = [f["N"]] if "N" in f else [f.get("V") or f.get("C") or g.get("A") or f.get("PL") or f.get("AGE")]
        others = [o for k, o in self.facts.items() if k != key and k.split(":")[0] == kind]
        avoid = [o.get("N") or o.get("V") or o.get("C") for o in others if o.get("N") or o.get("V") or o.get("C")]
        if kind in ("pet", "rel"):
            avoid += [x.capitalize() for x in self.names if x != f["N"].lower() and x.capitalize() not in avoid]
        self.say(self.w(W.ASK[kind]).format(**f), self.r.choice(W.RECALL[kind]).format(**g), "recall:" + kind,
                 {"want": answer, "avoid": avoid, "select": bool(others)})

    def unknown(self) -> bool:
        """A question about something never said: the right reply says it doesn't know."""
        r = self.r
        kinds = [k for k in ("pet", "rel", "fav", "age", "place") if not any(x.startswith(k) for x in self.facts)]
        if not kinds:
            return False
        kind = r.choice(kinds)
        f = {"P": self.v(P.PETS), "R": self.v(P.RELATIONS), "K": r.choice(list(P.FAVOURITES))}
        self.say(self.w(W.ASK[kind]).format(**f), r.choice(W.UNKNOWN), "unknown:" + kind,
                 {"decline": True, "avoid": [x.capitalize() for x in self.names if x != (self.user or "").lower()]})
        return True

    def correct(self) -> bool:
        pets = [k for k in self.facts if k.startswith("pet:")]
        if not pets or (n := self.new_name()) is None:
            return False
        f = self.facts[pets[0]]
        f["old"], f["N"] = f["N"], n
        self.say(self.w(W.CORRECT).format(P=f["P"], N=n), self.r.choice(W.CORRECTED).format(P=f["P"], N=n),
                 "correct:pet")
        return True

    def reverse(self) -> bool:
        """From a name to who it is ("who is Rex?" "your dog!")."""
        named = [k for k, f in self.facts.items() if "N" in f and k.split(":")[0] in ("pet", "rel")]
        if not named:
            return False
        key = self.r.choice(named)
        f = self.facts[key]
        x = f.get("P") or f.get("R")
        others = [g.get("P") or g.get("R") for k, g in self.facts.items()
                  if k != key and k.split(":")[0] in ("pet", "rel")]
        self.say(self.w(W.REVERSE).format(N=f["N"]), self.r.choice(W.REVERSE_REPLY).format(N=f["N"], X=x), "reverse",
                 {"want": [x], "avoid": [o for o in others if o != x], "select": bool(others)})
        return True

    def ask_topic(self):
        t, words = self.topic
        self.say(self.w(W.TOPIC), self.r.choice(W.TOPIC_REPLY).format(T=t), "topic", {"want_any": words})

    # --- moves: chat
    def persona(self):
        r = self.r
        key = r.choice(list(W.PERSONA_Q))
        ans = r.choice(W.PERSON_ANSWER if key == "person" else P.PERSONA[key])
        check = {"any": {"name": ["computer", "spectrum"], "what": ["computer"], "person": ["no", "computer"],
                         "age": ["old", "eighty", "made"], "home": ["live", "desk", "box", "table"],
                         "feel": ["fine", "well", "good", "happy"]}.get(key, [])}
        self.say(self.w(W.PERSONA_Q[key]), self.reply(ans, r.choice(list(W.ASK_BACK))), "persona:" + key,
                 check if check["any"] else None)

    def can(self):
        r = self.r
        yes = r.random() < 0.5
        x = r.choice(P.PERSONA["can"] if yes else P.PERSONA["cannot"])
        self.say(self.w(W.CAN_Q).format(X=x), r.choice(W.CAN_REPLY[0 if yes else 1]).format(X=x), "can",
                 {"any": ["yes", "can"] if yes else ["no", "can't", "cannot"]})

    def chat(self):
        r = self.r
        kind = r.choice(["like_q", "user_likes", "outing", "feeling", "knowledge", "thanks"])
        if kind in ("like_q", "user_likes"):
            x = self.v(P.LIKES)
            you = self.w(W.LIKE_Q if kind == "like_q" else W.USER_LIKES).format(X=x)
            spec = r.choice(W.LIKE_REPLY if kind == "like_q" else W.USER_LIKES_REPLY).format(X=x)
            self.topic = (x, x.replace("the ", "").split())
            self.say(you, self.reply(spec, r.choice(list(W.ASK_BACK))), kind, {"any": self.topic[1]})
        elif kind == "outing":
            o = self.v(P.OUTINGS)
            self.topic = (f"the {o}", [o])
            self.say(self.w(W.OUTING).format(OUT=o), r.choice(W.OUTING_REPLY).format(OUT=o), kind, {"any": [o]})
        elif kind == "feeling":
            f = self.v(P.FEELINGS)
            spec = r.choice(W.FEELING_REPLY.get(f, ["oh! why is that?"]))
            self.say(self.w(W.FEELING).format(F=f), spec, kind)
        elif kind == "knowledge":
            q, a = r.choice(P.KNOWLEDGE[self.vi])
            self.say(q, a or "i don't know.", kind, {"decline": True} if a is None else None)
        else:
            self.say(self.w(W.THANKS), r.choice(W.THANKS_REPLY), kind)

    def bye(self):
        spec = (self.r.choice(W.BYE_REPLY).format(N=self.user) if self.user else
                self.r.choice(["goodbye!", "bye! come back soon.", "see you soon!"]))
        self.say(self.w(W.BYE), spec, "bye")

    # --- a whole training conversation
    def late_intro(self):
        """Your name after a pet's or a relative's, so you are not always slot 1."""
        n = self.new_name()
        if n is None:
            return
        self.user = n
        lead = self.r.choice(["also, ", "i forgot to say, ", "", "my name? "])
        self.say(lead + self.w(W.INTRO).format(N=n), self.reply(f"{self.r.choice(W.MEET)}, {n}!"), "late intro")

    def random_plan(self, late: float = 0.0, reverse: bool = False):
        r = self.r
        if late and r.random() < late:  # a named fact first, your name after it
            self.greet(intro=False)
            self.pending = None
            for _ in range(r.randint(1, 2)):
                self.plant(r.choice(["pet", "rel"]))
            self.late_intro()
        else:
            x = r.random()
            if x < 0.85:
                self.greet(intro=x < 0.6)
        for _ in range(r.randint(2, 7)):
            if self.pending and r.random() < 0.8:
                self.answer_pending()
                continue
            self.pending = None
            asked = [c["move"] for c in self.checks]
            planted = [k for k in self.facts if asked.count("recall:" + k.split(":")[0]) < 2]
            opts = [("plant", 3.0 if len(self.facts) < 3 else 0.3), ("two", 1.0), ("recall", 4.0 * bool(planted)),
                    ("name", 2.0 * bool(self.user and asked.count("recall:name") < 2)), ("unknown", 1.0),
                    ("correct", 0.8 if reverse else 0.4), ("topic", 1.0 * bool(self.topic)),
                    ("persona", 2.0), ("can", 0.6), ("chat", 2.5), ("filler", 1.5 * bool(self.filler)),
                    ("reverse", 1.5 if reverse else 0.0)]
            move = r.choices([o for o, _ in opts], [wt for _, wt in opts])[0]
            if move == "plant":
                self.plant(r.choice(list(W.PLANT)))
            elif move == "two":
                self.two(r.choice(list(W.TWO)))
            elif move == "recall":
                self.recall(r.choice(planted))
            elif move == "name":
                self.recall_name()
            elif move == "unknown":
                self.unknown()
            elif move == "correct":
                self.correct()
            elif move == "reverse":
                self.reverse()
            elif move == "topic":
                self.ask_topic()
            elif move == "persona":
                self.persona()
            elif move == "can":
                self.can()
            elif move == "chat":
                self.chat()
            else:
                you, spec = self.filler(r)
                self.say(you, spec, "filler")
        if r.random() < 0.3:
            self.bye()
        return self

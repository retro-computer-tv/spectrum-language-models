"""ZX-LM's training language, v2 (the film's network). v1's language was 2 to 14 frames a kind, and the network
learned the frames, not the grammar (held-out question wordings: 2-48% right). v2 trains on many more constructions per kind,
pronouns and possessives, fillers, two events in one sentence, lists and swaps. Its own held-out frames ("held")
are new constructions or new words; v1's held frames are mostly in v2's training.

Placeholders as frames.py, plus {PP} {QP}: the person's possessive (his, her, their, fixed per person).
"""
from __future__ import annotations

from train.zx_lm.frames import ANSWERS  # noqa: F401  (the same answer forms)


def _t(*fs: str) -> list[tuple[str, str]]:
    return [(f, "train") for f in fs]


def _h(*fs: str) -> list[tuple[str, str]]:
    return [(f, "held") for f in fs]


GIVE_VERBS = [("gave", "given"), ("handed", "handed"), ("passed", "passed"), ("sold", "sold"), ("sent", "sent")]
GET_VERBS = ["got", "received", "bought", "took", "collected"]


def _give() -> list[tuple[str, str]]:
    out = []
    for v, pp in GIVE_VERBS:
        out += _t(f"{{P}} {v} the {{X}} to {{Q}}.", f"{{P}} {v} {{Q}} the {{X}}.", f"{{P}} {v} it to {{Q}}.",
                  f"the {{X}} was {pp} to {{Q}} by {{P}}.", f"{{Q}} was {pp} the {{X}} by {{P}}.",
                  f"{{P}} {v} {{PP}} {{X}} to {{Q}}.", f"{{P}} {v} the {{X}} over to {{Q}}.")
    for v in GET_VERBS:
        out += _t(f"{{Q}} {v} the {{X}} from {{P}}.", f"{{Q}} {v} it from {{P}}.", f"{{Q}} {v} {{P}}'s {{X}}.")
    out += _t("the {X} went from {P} to {Q}.", "{P} gave the {X} away to {Q}.", "{P} let {Q} keep the {X}.")
    return out + _h("the {X} was taken from {P} by {Q}.", "{P} passed the {X} on to {Q}.",
                    "{Q} ended up with {P}'s {X}.")


STATEMENTS2: dict[str, list[tuple[str, str]]] = {
    "own": _t("{P} owns a {X}.", "{P} has a {X}.", "{P} has got a {X}.", "{P} got a {X}.", "{P} bought a {X}.",
              "{P} found a {X}.", "{P} keeps a {X}.", "{P} is the owner of a {X}.", "{P} is the proud owner of a {X}.",
              "the {X} belongs to {P}.", "a {X} belongs to {P}.", "the {X} is {P}'s.", "the owner of the {X} is {P}.",
              "there is a {X} and it belongs to {P}.", "{P} now has a {X}.", "{P} has a {X} now.",
              "{P} was given a {X}.", "{P} has got {PP} own {X}.")
    + _h("{P} picked up a {X} at the market.", "a {X} is what {P} owns."),
    "own_pair": _t("{P} has a {X} and {Q} has a {Y}.", "{P} owns a {X} and {Q} owns a {Y}.",
                   "{P} has a {X}, and {Q} has a {Y}."),
    "own_list": _t("{P} owns a {X}, a {Y} and a {Z}.", "{P} has a {X} and a {Y}.", "{P} has a {X}, a {Y} and a {Z}.",
                   "{P} owns a {X} and a {Y}."),
    "give": _give(),
    "move": _t("the {X} now belongs to {Q}.", "{Q} has the {X} now.", "{Q} now has the {X}.",
               "the {X} is {Q}'s now.", "{Q} is the new owner of the {X}.") + _h("the {X} ended up with {Q}."),
    "swap": _t("{P} swapped {PP} {X} for {Q}'s {Y}.", "{P} and {Q} swapped the {X} and the {Y}.",
               "{P} traded {PP} {X} for {Q}'s {Y}.") + _h("{P} and {Q} did a swap: the {X} for the {Y}."),
    "lose": _t("{P} lost the {X}.", "{P} has lost the {X}.", "{P} lost {PP} {X}.", "the {X} was lost by {P}.",
               "{P} lost it.") + _h("{P} dropped the {X} somewhere and lost it."),
    "find": _t("{Q} found the {X}.", "the {X} was found by {Q}.", "{Q} found it.", "{Q} picked up the {X}.")
    + _h("the {X} turned up and {Q} kept it."),
    "correct": _t("sorry, {P} gave the {X} to {R}, not {Q}.", "no, it was {R}, not {Q}.",
                  "actually, {R} got the {X}, not {Q}.", "correction: the {X} went to {R}.", "sorry, i meant {R}, not {Q}.",
                  "no, not {Q}. it was {R}.", "actually the {X} went to {R}, not {Q}.", "sorry, {R} got it, not {Q}.",
                  "wait, i mean {R}.", "no, {Q} didn't get it. {R} did.", "i got that wrong: {R} has the {X}.",
                  "sorry, not {Q} but {R}.") + _h("oops, i mixed them up. it was {R}, not {Q}."),
    "correct_own": _t("sorry, the {X} belongs to {R}, not {P}.", "no, {R} owns the {X}, not {P}.",
                      "actually, {R} has the {X}, not {P}.", "sorry, i meant {R}, not {P}.",
                      "actually, it is {R} who owns the {X}.", "no, the {X} is {R}'s.")
    + _h("wait, that was wrong: the {X} is really {R}'s."),
    "neg": _t("{P} doesn't have the {X}.", "the {X} doesn't belong to {P}.", "{P} does not own the {X}.",
              "{P} has not got the {X}.", "{P} hasn't got the {X}.", "the {X} isn't {P}'s.")
    + _h("{P} has never owned the {X}."),
    "want": _t("{P} wants the {X}.", "{P} likes the {X}.", "{P} asked for the {X}.", "{P} would like the {X}.",
               "{P} really wants the {X}.", "{P} loves the {X}.") + _h("{P} is jealous of whoever has the {X}."),
}

QUESTIONS2: dict[str, list[tuple[str, str]]] = {
    "has": _t("who has the {X}?", "who has the {X} now?", "who owns the {X}?", "who owns the {X} now?",
              "who has got the {X}?", "who's got the {X}?", "who is the owner of the {X}?",
              "who is the owner of the {X} now?", "who does the {X} belong to?", "who does the {X} belong to now?",
              "the {X} belongs to who?", "whose {X} is it?", "whose is the {X}?", "whose {X} is it now?",
              "who has the {X} at the moment?", "who has the {X} these days?", "who's the owner of the {X}?")
    + _h("which person is holding the {X}?", "tell me who has the {X}."),
    "what": _t("what does {N} have?", "what does {N} have now?", "what does {N} own?", "what does {N} own now?",
               "what has {N} got?", "what has {N} got now?", "what belongs to {N}?", "what is {N}'s?",
               "what does {N} have left?", "what's {N} got?", "which things does {N} have?")
    + _h("tell me what {N} owns.", "what is in {N}'s hands?"),
    "yn": _t("does {N} have the {X}?", "does {N} still have the {X}?", "does {N} own the {X}?",
             "does {N} own the {X} now?", "has {N} got the {X}?", "has {N} still got the {X}?",
             "does the {X} belong to {N}?", "is the {X} {N}'s?", "is {N} the owner of the {X}?",
             "is the {X} still {N}'s?") + _h("is {N} still the owner of the {X}?", "is the {X} with {N}?"),
    "first": _t("who had the {X} first?", "who owned the {X} first?", "who was the first owner of the {X}?",
                "who had the {X} originally?", "who owned the {X} at first?", "whose was the {X} first?")
    + _h("who had the {X} in the beginning?"),
    "prev": _t("who had the {X} before?", "who owned the {X} before?", "who was the previous owner of the {X}?",
               "who used to have the {X}?", "who had the {X} before that?")
    + _h("who did the {X} belong to before?"),
    "before": _t("who had the {X} before {Q}?", "who owned the {X} before {Q}?", "who had the {X} before {Q} got it?",
                 "before {Q}, who had the {X}?") + _h("who was the owner of the {X} before {Q}?"),
    "giver": _t("who gave the {X} to {Q}?", "who gave {Q} the {X}?", "who did {Q} get the {X} from?",
                "where did {Q} get the {X} from?", "who handed the {X} to {Q}?", "from whom did {Q} get the {X}?",
                "who sold the {X} to {Q}?") + _h("who passed {Q} the {X}?"),
    "recip": _t("who did {N} give the {X} to?", "who got the {X} from {N}?", "who did {N} hand the {X} to?",
                "who did {N} sell the {X} to?", "to whom did {N} give the {X}?", "who received the {X} from {N}?")
    + _h("who did {N} pass the {X} to?"),
    "follow_x": _t("and the {X}?", "what about the {X}?", "and what about the {X}?", "the {X}?")
    + _h("how about the {X}?"),
    "follow_n": _t("and {N}?", "what about {N}?", "and what about {N}?") + _h("how about {N}?"),
}

FILLERS_START = ["then", "later", "yesterday", "after that", "this morning", "so", "and then", "ok,", "well,", "then,"]
FILLERS_END = [" yesterday", " today", " this morning", " last week", " again"]
PRONOUNS = [("he", "his"), ("she", "her"), ("they", "their")]

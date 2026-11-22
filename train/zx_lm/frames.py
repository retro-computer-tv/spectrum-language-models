"""ZX-LM's training language, v1: every sentence the generator can write, and every answer form.

Placeholders: {P} {Q} {R} people, {X} {Y} objects (the generator writes an object as its noun, sometimes after an
adjective), {N} a person in a question. Each frame is "train" or "held": held frames never appear in training. Most
held frames recombine words training uses ("the owner of the {X} is {P}." from "who is the owner of the {X}?"); a
few bring a word training never uses (marked NEW), which the tokenizer then reads as a name-like word.
"""
from __future__ import annotations

# event -> [(frame, split)]
STATEMENTS: dict[str, list[tuple[str, str]]] = {
    # someone gets a new object (its first owner)
    "own": [
        ("{P} owns a {X}.", "train"), ("{P} has a {X}.", "train"), ("the {X} belongs to {P}.", "train"),
        ("{P} has got a {X}.", "train"), ("{P} is the owner of a {X}.", "train"),
        ("there is a {X} and it belongs to {P}.", "train"), ("{P} bought a {X}.", "train"),
        ("the owner of the {X} is {P}.", "held"), ("a {X} belongs to {P}.", "held"), ("{P} got a {X}.", "held"),
        ("{P} has got a {X} now.", "held"),
    ],
    # P's object moves to Q
    "give": [
        ("{P} gave the {X} to {Q}.", "train"), ("{P} gave {Q} the {X}.", "train"),
        ("the {X} was given to {Q} by {P}.", "train"), ("{Q} was given the {X} by {P}.", "train"),
        ("{P} handed the {X} to {Q}.", "train"), ("{P} passed the {X} to {Q}.", "train"),
        ("{P} passed {Q} the {X}.", "train"), ("{P} sold the {X} to {Q}.", "train"),
        ("the {X} was sold to {Q} by {P}.", "train"), ("{Q} got the {X} from {P}.", "train"),
        ("{Q} received the {X} from {P}.", "train"), ("{Q} bought the {X} from {P}.", "train"),
        ("{Q} took the {X} from {P}.", "train"), ("the {X} went from {P} to {Q}.", "train"),
        ("{P} handed {Q} the {X}.", "held"), ("{P} sold {Q} the {X}.", "held"),
        ("the {X} was handed to {Q} by {P}.", "held"), ("the {X} was passed to {Q} by {P}.", "held"),
        ("{Q} was sold the {X} by {P}.", "held"), ("{Q} got the {X} off {P}.", "held"),  # NEW: off
    ],
    # the same, with "it" for the object the previous statement was about
    "give_it": [
        ("{P} gave it to {Q}.", "train"), ("{P} handed it to {Q}.", "train"), ("{Q} took it from {P}.", "train"),
        ("{P} sold it to {Q}.", "held"), ("{Q} got it from {P}.", "held"),
    ],
    "lose": [("{P} lost the {X}.", "train"), ("{P} has lost the {X}.", "train"), ("the {X} was lost by {P}.", "held")],
    "find": [("{Q} found the {X}.", "train"), ("the {X} was found by {Q}.", "held")],
    # the previous statement said the object went to Q; it went to R
    "correct": [
        ("sorry, {P} gave the {X} to {R}, not {Q}.", "train"), ("no, it was {R}, not {Q}.", "train"),
        ("actually, {R} got the {X}, not {Q}.", "train"), ("correction: the {X} went to {R}.", "train"),
        ("sorry, i meant {R}, not {Q}.", "train"),
        ("no, not {Q}. it was {R}.", "held"), ("actually the {X} went to {R}, not {Q}.", "held"),
        ("sorry, {R} got it, not {Q}.", "held"),
    ],
    # the previous statement said P had a new object; R did
    "correct_own": [
        ("sorry, the {X} belongs to {R}, not {P}.", "train"), ("no, {R} owns the {X}, not {P}.", "train"),
        ("actually, {R} has the {X}, not {P}.", "held"), ("actually, it is {R} who owns the {X}.", "held"),
    ],
    # true statements that mention a person and an object without changing anything
    "neg": [
        ("{P} doesn't have the {X}.", "train"), ("the {X} doesn't belong to {P}.", "train"),
        ("{P} does not own the {X}.", "held"), ("{P} has not got the {X}.", "held"),
    ],
    "want": [
        ("{P} wants the {X}.", "train"), ("{P} likes the {X}.", "train"), ("{P} asked for the {X}.", "train"),
        ("{P} really wants the {X}.", "held"), ("{P} would like the {X}.", "held"),  # NEW: really, would
    ],
}

# question type -> [(frame, split)]; {X} the object asked about, {N} the person, {Q} a second person
QUESTIONS: dict[str, list[tuple[str, str]]] = {
    "has": [
        ("who has the {X}?", "train"), ("who has the {X} now?", "train"), ("who owns the {X}?", "train"),
        ("who owns the {X} now?", "train"), ("who has got the {X}?", "train"),
        ("who is the owner of the {X}?", "train"),
        ("who does the {X} belong to?", "held"), ("who is the owner of the {X} now?", "held"),
        ("who has the {X} at the moment?", "held"),  # NEW: at, moment
    ],
    "what": [
        ("what does {N} have?", "train"), ("what does {N} have now?", "train"), ("what does {N} own?", "train"),
        ("what has {N} got?", "train"),
        ("what does {N} own now?", "held"), ("what belongs to {N}?", "held"), ("what has {N} got now?", "held"),
    ],
    "yn": [
        ("does {N} have the {X}?", "train"), ("does {N} still have the {X}?", "train"),
        ("does {N} own the {X}?", "train"), ("has {N} got the {X}?", "held"), ("does the {X} belong to {N}?", "held"),
    ],
    "first": [
        ("who had the {X} first?", "train"), ("who owned the {X} first?", "train"),
        ("who was the first owner of the {X}?", "held"),
    ],
    "prev": [("who had the {X} before?", "train"), ("who owned the {X} before?", "held")],
    "before": [("who had the {X} before {Q}?", "train"), ("who owned the {X} before {Q}?", "held")],
    "giver": [
        ("who gave the {X} to {Q}?", "train"), ("who did {Q} get the {X} from?", "train"),
        ("who gave {Q} the {X}?", "held"),
    ],
    "recip": [
        ("who did {N} give the {X} to?", "train"), ("who got the {X} from {N}?", "train"),
        ("who did {N} hand the {X} to?", "held"),  # NEW: hand
    ],
    # follow-ups: the same question as the previous one, about another object or person
    "follow_x": [("and the {X}?", "train"), ("what about the {X}?", "train"), ("and what about the {X}?", "held")],
    "follow_n": [("and {N}?", "train"), ("what about {N}?", "train"), ("and what about {N}?", "held")],
}

# answer kind -> forms; the training target is one form at random, and scoring accepts any form exactly
ANSWERS: dict[str, list[str]] = {
    "has": ["{O}.", "{O} has it.", "{O} has the {X}.", "{O} has the {X} now."],
    "has_nobody": ["nobody has it.", "nobody."],
    "what_1": ["the {A}.", "{N} has the {A}."],
    "what_2": ["the {A} and the {B}.", "{N} has the {A} and the {B}."],
    "what_3": ["the {A}, the {B} and the {C}."],
    "what_none": ["nothing.", "nothing now."],
    "yes": ["yes.", "yes, {N} has it."],
    "no_owner": ["no.", "no, {O} has it."],
    "no_nobody": ["no.", "no, nobody has it."],
    "first": ["{O}.", "{O} had it.", "{O} had it first."],
    "prev": ["{O}.", "{O} had it.", "{O} had it before."],
    "giver": ["{O}.", "{O} did.", "{O} gave it to {Q}."],
    "recip": ["{O}.", "{O} got it."],
    "nobody_before": ["nobody.", "nobody had it before."],
    "unknown": ["i don't know.", "i don't know. you didn't tell me."],
    "ack": ["ok."],
}

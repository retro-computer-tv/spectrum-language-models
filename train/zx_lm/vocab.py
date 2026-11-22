"""ZX-LM's words: words in, words out, nothing lost.

A word in the vocabulary is one token. Any other word (a name, an object, an adjective, a word the vocabulary lacks)
becomes one of 16 reference tokens; the conversation's reference table keeps its spelling, so the text can be written
back exactly. Which of the 16 a new word gets is random (rng), so the network can't learn anything from the number.
The tokenizer decides only "same spelling, same reference": it never reads meaning, roles or answers.
"""
from __future__ import annotations

import random
import re

from train.zx_lm.frames import ANSWERS, QUESTIONS, STATEMENTS

N_REFS = 16
SPECIALS = ["<pad>", "<u>", "<s>", "<think>"]
WORD_RE = re.compile(r"[a-z]+n't|[a-z]+|'s|[.,?:!]")
PLACEHOLDER = re.compile(r"\{[A-Z]\}")


def words(text: str) -> list[str]:
    """Lower case, "'s" split off, punctuation as words."""
    t = text.lower().replace("\u2019", "'")
    return WORD_RE.findall(t)


def _frame_words(frames) -> set[str]:
    out: set[str] = set()
    for f in frames:
        out.update(words(PLACEHOLDER.sub(" ", f)))
    return out


def train_words(version: int = 1) -> set[str]:
    """Every word a training frame, question or answer uses (v2: frames2.py, its fillers and pronouns too)."""
    st, qs, extra = STATEMENTS, QUESTIONS, set()
    if version >= 2:
        from train.zx_lm.frames2 import FILLERS_END, FILLERS_START, PRONOUNS, QUESTIONS2, STATEMENTS2  # noqa: PLC0415
        st, qs = STATEMENTS2, QUESTIONS2
        extra = _frame_words(FILLERS_START + FILLERS_END) | {w for p in PRONOUNS for w in p}
    w = set(extra)
    for fs in st.values():
        w |= _frame_words(f for f, s in fs if s == "train")
    for fs in qs.values():
        w |= _frame_words(f for f, s in fs if s == "train")
    for fs in ANSWERS.values():
        w |= _frame_words(fs)
    return w | {"an"}  # "a" before a vowel


def answer_words() -> list[str]:
    """The words the network can write itself (its output rows); everything else it writes is a reference."""
    w = set()
    for fs in ANSWERS.values():
        w |= _frame_words(fs)
    return sorted(w)


class Vocab:
    def __init__(self, cap: int | None = None, counts: dict[str, int] | None = None, version: int = 1):
        """cap keeps the cap most frequent words (counts from training text); the rest become references."""
        self.version = version
        ws = sorted(train_words(version))
        if cap is not None:
            keep = set(answer_words())  # the output words must be readable as input too
            rest = sorted((w for w in ws if w not in keep), key=lambda w: (-(counts or {}).get(w, 0), w))
            ws = sorted(keep | set(rest[: max(0, cap - len(keep))]))
        self.words = SPECIALS + ws
        self.index = {w: i for i, w in enumerate(self.words)}
        self.n_words = len(self.words)
        self.size = self.n_words + N_REFS  # input tokens: words, then references
        self.out_words = answer_words()
        self.out_index = {w: i for i, w in enumerate(self.out_words)}
        self.n_out = len(self.out_words)

    def ref(self, r: int) -> int:
        return self.n_words + r

    def is_ref(self, tok: int) -> bool:
        return tok >= self.n_words


class Refs:
    """One conversation's reference table: spelling <-> reference number, numbers handed out at random."""

    def __init__(self, rng: random.Random, replace: bool = False):
        """replace: past 16 words, reuse a number as the Spectrum does (reference/zx_lm.py), rather than refuse."""
        self.rng, self.replace = rng, replace
        self.by_word: dict[str, int] = {}
        self.spelling: dict[int, str] = {}

    def get(self, w: str) -> int:
        if w not in self.by_word:
            free = [r for r in range(N_REFS) if r not in self.spelling]
            if not free and not self.replace:
                raise OverflowError("more than 16 unfamiliar words in one conversation")
            if not free:
                free = [self.rng.randrange(N_REFS)]
                del self.by_word[self.spelling[free[0]]]
            r = self.rng.choice(free)
            self.by_word[w] = r
            self.spelling[r] = w
        return self.by_word[w]


def encode(vocab: Vocab, refs: Refs, text: str) -> list[int]:
    return [vocab.index[w] if w in vocab.index else vocab.ref(refs.get(w)) for w in words(text)]


def decode(vocab: Vocab, refs: Refs, toks: list[int]) -> str:
    out = []
    for t in toks:
        w = refs.spelling.get(t - vocab.n_words, "<?>") if vocab.is_ref(t) else vocab.words[t]
        out.append(w)
    s = " ".join(out)
    return re.sub(r" ([.,?:!]|'s)", r"\1", s)


def normal(text: str) -> str:
    """The form decode() writes: what a round trip must give back."""
    return re.sub(r" ([.,?:!]|'s)", r"\1", " ".join(words(text)))

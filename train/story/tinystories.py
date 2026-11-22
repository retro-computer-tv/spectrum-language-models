"""TinyStories (V2, the GPT-4 stories) as 37-symbol text for the story writer, then names as cast slots.

    python -m train.story.tinystories        # data/stories/ and data/stories-slots/ (about half an hour)

Download TinyStoriesV2-GPT4-train.txt and TinyStoriesV2-GPT4-valid.txt from
https://huggingface.co/datasets/roneneldan/TinyStories into data/tinystories/ first (CDLA-Sharing-1.0, per the
dataset's card).

Normalisation: lower case; curly quotes to straight; ';' to ','; runs of spaces to one; blank lines dropped (a newline
is a paragraph break); each story ends with the end-of-story symbol '#'. A story with any other character (digits,
stray Unicode) is dropped whole, so no story is altered beyond the rules above.

Splits, by whole stories: train is the training file less 0.5% of its stories (chosen with a fixed seed), which are
dev; test is the validation file less any story that also appears in train.

Then the names: a word is a name if it occurs 20 or more times in the middle of a sentence (in the first 300M
characters of the cased training file) and at least 95% of those times with a capital, less calendar words and
titles ("lily", "ben" and "max" are names; "spot", "tiny" and "mom" are not). Within each story the first three
distinct names become cast slots (reference/text.py): "named *zara." the first time, then "1".

Also data/words.json: every word used at least five times in the first 300M characters of the training stories (J's
fine-tuning data checks its made-up words against it, and takes its spelling words from it).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter

import numpy as np

from reference.text import NSLOTS, SSTOI, STOI, VOCAB
from train import DATA

RAW = DATA / "tinystories"
OUT = DATA / "stories"
SLOTS = DATA / "stories-slots"
EOS = "#"
SEP = "<|endoftext|>"
TABLE = {ord("\u2018"): "'", ord("\u2019"): "'", ord("\u201c"): '"', ord("\u201d"): '"', ord(";"): ","}
OK = set(VOCAB) - {EOS}
NOT_NAMES = {"i", "mr", "mrs", "ms", "dr", "miss", "monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday", "january", "february", "march", "april", "may", "june", "july", "august",
             "september", "october", "november", "december", "christmas", "halloween", "easter", "god", "earth", "tv",
             "ok", "okay"}


def normalise(story: str) -> str | None:
    s = story.translate(TABLE).lower()
    s = re.sub(r"[ \t]+", " ", s)
    s = "\n".join(line.strip() for line in s.split("\n") if line.strip())
    if not s or not set(s) <= OK:
        return None
    return s + EOS


def stories(path):
    buf = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if SEP in line:
                head, _, tail = line.partition(SEP)
                buf.append(head)
                yield "".join(buf)
                buf = [tail]
            else:
                buf.append(line)
    if "".join(buf).strip():
        yield "".join(buf)


def encode(texts: list[str]) -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    for c, i in STOI.items():
        lut[ord(c)] = i
    return lut[np.frombuffer("".join(texts).encode("ascii"), dtype=np.uint8)]


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"vocab": VOCAB, "source": "roneneldan/TinyStories (TinyStoriesV2-GPT4), CDLA-Sharing-1.0"}
    train, dropped = [], 0
    for st in stories(RAW / "TinyStoriesV2-GPT4-train.txt"):
        s = normalise(st)
        if s is None:
            dropped += 1
        else:
            train.append(s)
    rnd = random.Random(1982)
    idx = list(range(len(train)))
    rnd.shuffle(idx)
    dev_idx = set(idx[: len(train) // 200])
    dev = [train[i] for i in sorted(dev_idx)]
    train = [s for i, s in enumerate(train) if i not in dev_idx]
    seen = {hashlib.blake2b(s.encode(), digest_size=8).digest() for s in train}
    test = []
    for st in stories(RAW / "TinyStoriesV2-GPT4-valid.txt"):
        s = normalise(st)
        if s is not None and hashlib.blake2b(s.encode(), digest_size=8).digest() not in seen:
            test.append(s)
    for name, texts in (("train", train), ("dev", dev), ("test", test)):
        arr = encode(texts)
        arr.tofile(OUT / f"{name}.u8")
        meta[name] = {"stories": len(texts), "chars": int(arr.size)}
    meta["train_dropped_stories"] = dropped
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def build_names(chars: int = 300_000_000) -> set[str]:
    raw = (RAW / "TinyStoriesV2-GPT4-train.txt").open(encoding="utf-8", errors="replace").read(chars)
    caps, allc = Counter(), Counter()
    for m in re.finditer(r"(?<=[a-z,;] )([A-Za-z]+)", raw):
        w = m.group(1)
        allc[w.lower()] += 1
        if w[0].isupper():
            caps[w.lower()] += 1
    names = {w: caps[w] for w, n in allc.items() if n >= 20 and caps[w] / n >= 0.95 and w not in NOT_NAMES}
    (DATA / "names.json").write_text(json.dumps(dict(sorted(names.items(), key=lambda kv: -kv[1]))))
    print(f"{len(names)} names; the commonest: {list(names)[:20]}")
    return set(names)


def build_words(chars: int = 300_000_000) -> None:
    data = np.memmap(OUT / "train.u8", dtype=np.uint8, mode="r")[:chars]
    lut = np.frombuffer(VOCAB.encode("ascii"), dtype=np.uint8)
    word, cnt = re.compile(r"[a-z]+(?:'[a-z]+)?"), Counter()
    for i in range(0, len(data), 20_000_000):
        for story in lut[data[i: i + 20_000_000]].tobytes().decode("ascii").split("#"):
            cnt.update(word.findall(story))
    vocab = {w: c for w, c in cnt.items() if c >= 5}
    (DATA / "words.json").write_text(json.dumps(vocab))
    print(f"{len(vocab)} words")


def lexicon() -> set[str]:
    return set(json.loads((DATA / "names.json").read_text()))


def to_slots(text: str, name_re: re.Pattern) -> str:
    """One text (stories separated by '#'): names to cast slots, story by story."""
    out = []
    for story in text.split("#"):
        slots: dict[str, str] = {}

        def sub(m, slots=slots):
            w = m.group()
            if w in slots:
                return slots[w]
            if len(slots) < NSLOTS:
                slots[w] = str(len(slots) + 1)
                return "*" + w
            return w

        out.append(name_re.sub(sub, story))
    return "#".join(out)


def convert(split: str, name_re: re.Pattern, step: int = 50_000_000) -> dict:
    lut = np.frombuffer(VOCAB.encode("ascii"), dtype=np.uint8)
    back = np.zeros(256, dtype=np.uint8)
    for c, i in SSTOI.items():
        back[ord(c)] = i
    src = np.memmap(OUT / f"{split}.u8", dtype=np.uint8, mode="r")
    n_in = n_out = 0
    with (SLOTS / f"{split}.u8").open("wb") as f:
        i = 0
        while i < len(src):
            j = min(len(src), i + step)
            if j < len(src):  # end the chunk on a story boundary
                j = i + int(np.flatnonzero(src[i:j] == 0)[-1]) + 1
            enc = to_slots(lut[src[i:j]].tobytes().decode("ascii"), name_re)
            f.write(back[np.frombuffer(enc.encode("ascii"), dtype=np.uint8)].tobytes())
            n_in, n_out, i = n_in + (j - i), n_out + len(enc), j
    print(f"{split}: {n_in} characters, {n_out} symbols ({n_out / n_in:.4f})")
    return {"chars": n_in, "symbols": n_out}


def main() -> None:
    prepare()
    build_words()
    names = build_names()
    name_re = re.compile(rf"\b(?:{'|'.join(sorted(names, key=len, reverse=True))})\b")
    SLOTS.mkdir(parents=True, exist_ok=True)
    meta = {"vocab": VOCAB + "*123", "slots": NSLOTS}
    for split in ("dev", "test", "train"):
        meta[split] = convert(split, name_re)
    (SLOTS / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

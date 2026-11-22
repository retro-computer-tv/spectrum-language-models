"""The public dialogue sets as two-role conversations: you, and the Spectrum.

    python -m train.j.public      # data/chat/public/{tinydialogues,soda,persona,everyday}.jsonl

Download them into data/dialogue/ first, saved under the names on the left (each set's licence is on its card):

  tinydialogues/train.txt   tinydialogue_train_ordered.txt from https://huggingface.co/datasets/styfeng/TinyDialogues
                            (MIT)
  soda/train.parquet        train.parquet from https://huggingface.co/datasets/allenai/soda (CC BY 4.0)
  persona/train.csv         data/Synthetic-Persona-Chat_train.csv and
  persona/new.csv           data/New-Persona-New-Conversations.csv from
                            https://huggingface.co/datasets/google/Synthetic-Persona-Chat (CC BY 4.0)
  everyday/train.parquet    data/train_sft-00000-of-00001.parquet from
                            https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k
                            (Apache 2.0)

Each line written: {"turns": [[role, text], ...], "names": [...]}, role "u" (you) or "s" (the Spectrum), text cleaned
as format.clean (case kept, so names can be found). Kept only: two speakers, at least two exchanges, every turn
inside the alphabet (no digits) and at most MAXTURN characters, no placeholder names, no test name anywhere. A
dialogue is cut at its first turn that fails, and kept if two exchanges are left.

  TinyDialogues  ages 5 and 10 only (the file is in age order: 2, 5, 10, 15; age 2 is toddler talk, 15 is beyond
                 the story vocabulary). The Spectrum takes the grown-up's part, you the child's.
  SODA           you are the first speaker. Names: the PersonX/Y/Z fields and the speakers.
  Persona chat   "[User 1's name]" placeholders get names from the training pool.
  everyday       user and assistant; the assistant's long replies are cut to whole sentences within MAXTURN.
"""
from __future__ import annotations

import csv
import json
import random
import re

from train import DATA
from train.j import pools as P
from train.j.format import clean

RAW = DATA / "dialogue"
OUT = DATA / "chat" / "public"
MAXTURN = 120
WORD = re.compile(r"[A-Za-z]+")


def finish(pairs: list[tuple[str, str]], names: set[str], test: set[str]) -> dict | None:
    """Alternating (role, raw text) turns from you: merge repeats, clean, cut at the first bad turn."""
    turns: list[list[str]] = []
    for role, raw in pairs:
        t = clean(raw.strip().strip('"').strip())
        if t is None or len(t) > MAXTURN or "[" in raw:
            break
        if turns and turns[-1][0] == role:
            if len(turns[-1][1]) + len(t) + 1 > MAXTURN:
                break
            turns[-1][1] += " " + t
        else:
            turns.append([role, t])
    while turns and turns[0][0] != "u":
        turns.pop(0)
    if turns and turns[-1][0] == "u":
        turns.pop()
    if len(turns) < 4:
        return None
    words = {w.lower() for _, t in turns for w in WORD.findall(t)}
    if words & test:
        return None
    return {"turns": turns, "names": sorted(names & words)}


def names_in(turns, known: set[str]) -> set[str]:
    """Capitalised words that are known names (not at the start of a sentence, where every word has a capital)."""
    out = set()
    for _, t in turns:
        for m in re.finditer(r"(?<=[a-z,] )([A-Z][a-z]+)", t):
            if m.group(1).lower() in known:
                out.add(m.group(1).lower())
    return out


def tinydialogues(known: set[str], test: set[str]):
    text = (RAW / "tinydialogues/train.txt").read_text(encoding="utf-8")
    convs = text.split("<|endoftext|>")
    for conv in convs[len(convs) // 4: 3 * len(convs) // 4]:
        turns = []
        for t in conv.split("\\n\\n"):
            m = re.match(r"\s*\*\*([^*]+)\*\*(?:\s*\([^)]*\))?:\s*(.*)", t.strip(), re.S)
            if not m or "**" in m.group(2):
                break
            turns.append((m.group(1).strip(), re.sub(r"\*[^*]*\*|\([^)]*\)", "", m.group(2))))
        speakers = {s for s, _ in turns}
        if len(speakers) != 2 or "Child" not in speakers:
            continue
        pairs = [("u" if s == "Child" else "s", t) for s, t in turns]
        yield finish(pairs, names_in(pairs, known), test)


def soda(known: set[str], test: set[str]):
    import pyarrow.parquet as pq  # noqa: PLC0415

    f = pq.ParquetFile(RAW / "soda/train.parquet")
    for batch in f.iter_batches(columns=["dialogue", "speakers", "PersonX", "PersonY", "PersonZ"], batch_size=20000):
        for r in batch.to_pylist():
            sp = r["speakers"]
            if len(set(sp)) != 2:
                continue
            first = sp[0]
            pairs = [("u" if s == first else "s", t) for s, t in zip(sp, r["dialogue"])]
            people = {x.lower() for x in (r["PersonX"], r["PersonY"], r["PersonZ"], *sp)
                      if x and x.isalpha() and 3 <= len(x) <= 10}
            yield finish(pairs, people, test)


def persona(known: set[str], test: set[str], rng: random.Random):
    pool = sorted(known)
    for name in ("train.csv", "new.csv"):
        with open(RAW / "persona" / name, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                conv = r.get("Best Generated Conversation") or next(iter(r.values()))
                a, b = rng.sample(pool, 2)
                conv = re.sub(r"\[User 1'?s? [Nn]ame\]", a.capitalize(), conv)
                conv = re.sub(r"\[User 2'?s? [Nn]ame\]", b.capitalize(), conv)
                pairs = [("u" if m.group(1) == "1" else "s", m.group(2))
                         for m in re.finditer(r"^User ([12]): (.*)$", conv, re.M)]
                yield finish(pairs, {a, b}, test)


def sentences_within(t: str, n: int) -> str:
    out = ""
    for s in re.findall(r"[^.!?]+[.!?]+", t):
        if len(out) + len(s) > n:
            break
        out += s
    return out.strip()


def everyday(known: set[str], test: set[str]):
    import pyarrow.parquet as pq  # noqa: PLC0415

    for r in pq.read_table(RAW / "everyday/train.parquet", columns=["messages"]).to_pylist():
        pairs = [("u" if m["role"] == "user" else "s",
                  m["content"] if m["role"] == "user" else sentences_within(m["content"], MAXTURN))
                 for m in r["messages"]]
        yield finish(pairs, set(), test)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    known, test = set(P.train_names()), set(P.TEST_NAMES)
    jobs = (("tinydialogues", tinydialogues(known, test)), ("soda", soda(known, test)),
            ("persona", persona(known, test, random.Random(1))), ("everyday", everyday(known, test)))
    for name, gen in jobs:
        n = chars = 0
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for c in gen:
                if c is None:
                    continue
                f.write(json.dumps(c) + "\n")
                n += 1
                chars += sum(len(t) + 1 for _, t in c["turns"])
        print(f"{name}: {n} conversations, {chars / 1e6:.1f}M characters", flush=True)


if __name__ == "__main__":
    main()

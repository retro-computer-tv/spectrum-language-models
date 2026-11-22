"""J's main training data: the public dialogue sets (train/j/public.py) plus the procedural conversations
(train/j/skills.py), mixed by conversation.

    python -m train.j.build_data      # data/chat/data3/{train,dev}.{u8,role} and meta.json (about 230M characters)

The mixture (characters): procedural conversations 70%, SODA 20%, persona chat, TinyDialogues and everyday all of
theirs. A third of the procedural conversations give your name after a pet's or a relative's (so you are not always
the first slot); some ask from a name to who it is ("who is Rex?"); corrections come up. Within half of them, the
chit-chat between a fact and its question is borrowed from persona chat and TinyDialogues (short, without names). 1%
of conversations, chosen at random, are dev. No test name occurs (public.py drops any dialogue with one; the
generator draws only from pools.train_names() and made-up names). .role is 1 where you typed, 2 where the Spectrum
wrote, 0 for '#'.
"""
from __future__ import annotations

import json
import random

from train import DATA
from train.j import pools as P
from train.j.format import encode_turns, roles, to_ids
from train.j.skills import Conv

ROOT = DATA / "chat"
OUT = ROOT / "data3"
TOTAL = 230e6
SODA, SKILLS, LATE = 0.20, 0.70, 0.33


def load(name: str) -> list[dict]:
    with open(ROOT / "public" / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    rng = random.Random(7)
    pub = {n: load(n) for n in ("soda", "persona", "tinydialogues", "everyday")}
    rng.shuffle(pub["soda"])
    pairs = [(c["turns"][i][1], c["turns"][i + 1][1]) for n in ("persona", "tinydialogues") for c in pub[n]
             if not c["names"] for i in range(0, len(c["turns"]) - 1, 2)
             if len(c["turns"][i][1]) <= 80 and len(c["turns"][i + 1][1]) <= 80]

    def filler(r):
        return r.choice(pairs)

    convs: list[tuple[str, list, set]] = []
    budget = {"soda": SODA * TOTAL}
    for n, cs in pub.items():
        used = 0
        for c in cs:
            if n in budget and used >= budget[n]:
                break
            convs.append((n, [tuple(t) for t in c["turns"]], set(c["names"])))
            used += sum(len(t) + 1 for _, t in c["turns"])
        print(f"{n}: {used / 1e6:.1f}M characters", flush=True)
    names = P.train_names()
    used = 0
    while used < SKILLS * TOTAL:
        c = Conv(rng, names, filler=filler if rng.random() < 0.5 else None, noisy=True).random_plan(LATE, True)
        convs.append(("skills", c.turns, c.names))
        used += sum(len(t) + 1 for _, t in c.turns)
    print(f"skills: {used / 1e6:.1f}M characters", flush=True)
    rng.shuffle(convs)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"source": "train/j/build_data.py", "total_target": TOTAL}
    for split in ("dev", "train"):
        part = convs[: len(convs) // 100] if split == "dev" else convs[len(convs) // 100:]
        text, role, count = [], [], {}
        for src, turns, nm in part:
            t, r = encode_turns(turns, {x.lower() for x in nm})
            text.append(t)
            role.append(r)
            count[src] = count.get(src, 0) + 1
        ids = to_ids("".join(text))
        ids.tofile(OUT / f"{split}.u8")
        roles("".join(role)).tofile(OUT / f"{split}.role")
        meta[split] = {"symbols": int(len(ids)), "conversations": count}
        print(split, json.dumps(meta[split]), flush=True)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

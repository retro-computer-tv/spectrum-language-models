"""ZX-LM conversations run free: the user's turns as written, every Spectrum turn the
network's own, fed back as it wrote it. Greedy: the likeliest word every time. Scored strictly: a reply counts only
if it is exactly one of the accepted answers.
"""
from __future__ import annotations

import random

import torch

from train.zx_lm.data import N_THINK
from train.zx_lm.vocab import Refs, Vocab, decode, normal, words

MAX_REPLY = 14


def conversation(vocab: Vocab, refs: Refs, turns: list[dict], replies: list):
    """Yields (x, ref, user, end, wants_prediction); receives the predicted token after a step that wanted one."""
    for t in turns:
        if t["speaker"] == "u":
            toks = [vocab.index["<u>"]] + [vocab.index[w] if w in vocab.index else vocab.ref(refs.get(w))
                                           for w in words(t["text"])]
            for j, x in enumerate(toks):
                yield x, (x - vocab.n_words if vocab.is_ref(x) else -1), True, j == len(toks) - 1, False
            continue
        yield vocab.index["<s>"], -1, False, False, False
        for k in range(N_THINK):
            pred = yield vocab.index["<think>"], -1, False, False, k == N_THINK - 1
        out = []
        while True:
            x = vocab.index[vocab.out_words[pred]] if pred < vocab.n_out else vocab.ref(pred - vocab.n_out)
            out.append(x)
            last = vocab.out_words[pred] == "." if pred < vocab.n_out else False
            last = last or len(out) >= MAX_REPLY
            pred = yield x, (x - vocab.n_words if vocab.is_ref(x) else -1), False, last, not last
            if last:
                break
        replies.append(decode(vocab, refs, out))


@torch.no_grad()
def run(model, vocab: Vocab, convs: list[list[dict]], device: str, seed: int = 0, batch: int = 256) -> list[list[str]]:
    """Every Spectrum turn's text, per conversation."""
    model.eval()
    out: list[list[str]] = []
    for i in range(0, len(convs), batch):
        chunk = convs[i: i + batch]
        replies = [[] for _ in chunk]
        procs = [conversation(vocab, Refs(random.Random(seed * 100_003 + i + j), replace=True), c, replies[j])
                 for j, c in enumerate(chunk)]
        cur = [next(p) for p in procs]
        B = len(chunk)
        h, m = model.start(B, device)
        live = [True] * B
        while any(live):
            x = torch.tensor([c[0] if live[b] else 0 for b, c in enumerate(cur)], device=device)
            ref = torch.tensor([c[1] if live[b] else -1 for b, c in enumerate(cur)], device=device)
            user = torch.tensor([bool(c[2]) for c in cur], device=device)
            end = torch.tensor([bool(c[3]) and live[b] for b, c in enumerate(cur)], device=device)
            want = [bool(c[4]) and live[b] for b, c in enumerate(cur)]
            active = torch.tensor(live, device=device)
            _, h = model.step(x, ref, user, active, h, m)
            preds = model.predict(h, m).argmax(1).tolist() if any(want) else None
            if end.any():
                model.write(end, h, m)
            for b in range(B):
                if not live[b]:
                    continue
                try:
                    cur[b] = procs[b].send(preds[b] if want[b] else None)
                except StopIteration:
                    live[b] = False
        out += replies
    model.train()
    return out


def strict(reply: str, accept: list[str]) -> bool:
    return normal(reply) in {normal(a) for a in accept}


def content(reply: str, gold: dict, vocab: Vocab) -> dict:
    """What a reply says, read as the answer to its question: who, which things, yes/no, nobody, nothing."""
    ws = words(reply)
    unfamiliar = lambda w: w not in vocab.index  # noqa: E731
    if ws[:3] == ["i", "don't", "know"]:
        return {"kind": "unknown"}
    if ws[:1] == ["yes"]:
        return {"kind": "yes"}
    if ws[:1] == ["no"]:
        return {"kind": "no", "owner": ws[2] if len(ws) > 2 and ws[1] == "," and unfamiliar(ws[2]) else None}
    if ws[:1] == ["nobody"]:
        return {"kind": "nobody"}
    if ws[:1] == ["nothing"]:
        return {"kind": "nothing"}
    if gold["kind"] == "set":
        items, run_ = [], []
        for w in ws + ["."]:
            if unfamiliar(w):
                run_.append(w)
            elif run_:
                items.append(run_[-1])
                run_ = []
        named = [w for w in items if w != ws[0]] if len(ws) > 1 and ws[1] == "has" else items
        return {"kind": "set", "items": sorted(set(named))}
    return {"kind": "ent", "items": ws[:1]} if ws and unfamiliar(ws[0]) else {"kind": "other"}


def lower(g: dict) -> dict:
    out = dict(g)
    if "items" in g:
        out["items"] = sorted(x.lower() for x in g["items"])
    if g.get("owner"):
        out["owner"] = g["owner"].lower()
    return out

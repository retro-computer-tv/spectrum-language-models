"""Train ZX-LM: the float network first, then the Spectrum's, converted from it and fine-tuned in its arithmetic.

    python -m train.zx_lm.train --tag float --v 2                                  # 1. float, 6,000 batches
    python -m train.zx_lm.convert runs/float/best.npz runs/conv/best.npz           # 2. weight for weight
    python -m train.zx_lm.train --tag spectrum --v 2 --quant --init runs/conv/best.npz \\
        --teacher runs/float/best.npz --steps 4000 --lr 1e-3                       # 3. fine-tuned, distilled
    python -m train.zx_lm.export runs/spectrum/best.npz weights/mine.npz            # 4. the Spectrum's integers

That is the film's network, seed 1 (step 1 took 55 minutes on an RTX 3060 Ti). Training the Spectrum's network from
nothing stalled near 57% right on the development set however much of it was made float; starting from the float
network's weights it reached 77.5%.

Training conversations come from the generator's training split (training frames, training names, no held-out
composition), made fresh for every batch by a background thread, with references renumbered at random every batch.
The whole training step (forward, backward, clipping, Adam) is captured once as a CUDA graph and replayed with each
new batch copied into its inputs. The checkpoint is chosen on 300 development conversations from the training split,
run free and scored strictly (a reply counts only if it is exactly an accepted answer). Writes runs/<tag>/: log.jsonl,
best.npz and final.npz.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import random
import threading
import time

import numpy as np
import torch

from train import RUNS
from train.zx_lm.data import MAX_SLOTS, Batch, mention_count, script
from train.zx_lm.model import ZXLM
from train.zx_lm.pools import pools
from train.zx_lm.qmodel import ZXQ, Stepper
from train.zx_lm.questions import generate
from train.zx_lm.runner import run, strict
from train.zx_lm.vocab import Vocab, train_words

T_MAX = 300


def make(rng: random.Random, pool: dict, split: str, vocab: Vocab, n: int, max_steps: int | None = None, **kw):
    """n conversations that fit: at most 16 unfamiliar words and 64 mentions (and, for training batches, at most
    max_steps steps; fewer than 1 in 200 are longer)."""
    convs, scripts = [], []
    while len(convs) < n:
        c = generate(rng, pool, split, **kw)
        s = script(vocab, c)
        if s.n_local > 16 or mention_count(s) > MAX_SLOTS or (max_steps and len(s.x) > max_steps):
            continue
        convs.append(c)
        scripts.append(s)
    return convs, scripts


def save(model, path, cfg: dict, step: int) -> None:
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(path, **arrays, meta=json.dumps({**cfg, "step": step}))


def load(path, device="cpu"):
    """A ZX-LM checkpoint (.npz): the float network, or the Spectrum's (meta "quant")."""
    z = np.load(path)
    cfg = json.loads(str(z["meta"]))
    vocab = Vocab(version=cfg["version"])
    model = ZXQ(vocab, H=cfg["H"], bn=cfg.get("bn", False)) if cfg.get("quant") else ZXLM(vocab, H=cfg["H"])
    model.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files if k != "meta"})
    return model.to(device), cfg


def dev_score(model, vocab, convs, device) -> float:
    replies = run(model, vocab, convs, device, seed=7)
    ok = n = 0
    for c, rs in zip(convs, replies):
        for t, r in zip([t for t in c if t["speaker"] == "s"], rs):
            if not t.get("ack"):
                ok += strict(r, t["accept"])
                n += 1
    return ok / n


def producer(q: queue.Queue, vocab: Vocab, pool: dict, seed: int, B: int, version: int) -> None:
    rng = random.Random(seed)
    try:
        while True:
            _, scripts = make(rng, pool, "train", vocab, B, max_steps=T_MAX, version=version)
            q.put(Batch(vocab, scripts, rng, T=T_MAX))
    except Exception as e:  # noqa: BLE001
        q.put(e)


def take(q: queue.Queue) -> Batch:
    b = q.get()
    if isinstance(b, Exception):
        raise b
    return b


def lr_at(step: int, total: int, peak: float) -> float:
    warm = max(1, int(0.05 * total))
    if step < warm:
        return peak * (step + 1) / warm
    return peak * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--every", type=int, default=500)
    ap.add_argument("--quant", action="store_true", help="the Spectrum's arithmetic (qmodel.py)")
    ap.add_argument("--v", type=int, default=2, help="the language: 1 (frames.py) or 2 (frames2.py, the film's)")
    ap.add_argument("--teacher", help="a float checkpoint to distil from (with --quant)")
    ap.add_argument("--init", help="start from this checkpoint (the same kind of network)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    device = "cuda"
    vocab, pool = Vocab(version=a.v), pools(train_words(a.v))
    model = (ZXQ(vocab, H=a.H) if a.quant else ZXLM(vocab, H=a.H)).to(device)
    if a.init:
        z = np.load(a.init)
        model.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files if k != "meta"})
    stepper = Stepper(model) if a.quant else model
    out = RUNS / a.tag
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"H": a.H, "version": a.v, "quant": a.quant, "bn": False, "seed": a.seed}
    dev, _ = make(random.Random(999), pool["train"], "train", vocab, 300, version=a.v)
    q: queue.Queue = queue.Queue(maxsize=4)
    threading.Thread(target=producer, args=(q, vocab, pool["train"], a.seed * 7919, a.batch, a.v), daemon=True).start()
    lr = torch.tensor(a.lr, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, capturable=True)
    static = take(q).to(device)
    teacher = None
    if a.teacher:
        teacher, _ = load(a.teacher, device)
        teacher.eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)

    def train_step():
        if teacher is not None:
            with torch.no_grad():
                t_out = teacher(static)
            r = model(static, teacher={"lp": t_out["lp"], "aev": t_out["aev"]})
        else:
            r = model(static)
        r["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
        opt.step()
        return r

    lr.fill_(0.0)  # warm-up steps on the first batch, with no update
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            train_step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        res = train_step()
    into = {k: getattr(static, k) for k in static.arrays}
    best, log, t0 = -1.0, open(out / "log.jsonl", "a"), time.time()
    acc = torch.zeros(6, device=device)
    for step in range(1, a.steps + 1):
        take(q).to(device, into)
        lr.fill_(lr_at(step - 1, a.steps, a.lr))
        graph.replay()
        acc += res["stats"]
        if step % 100 == 0:
            st = acc.tolist()
            acc.zero_()
            row = {"step": step, "loss": round(float(res["loss"].detach()), 3), "tok_acc": round(st[0] / st[1], 4),
                   "ans_acc": round(st[2] / st[3], 4), "ev_mass": round(st[4] / st[5], 4),
                   "lr": round(lr_at(step - 1, a.steps, a.lr), 6), "s": round(time.time() - t0)}
            if step % a.every == 0 or step == a.steps:
                row["dev_strict"] = round(dev_score(stepper, vocab, dev, device), 4)
                if row["dev_strict"] > best:
                    best = row["dev_strict"]
                    save(model, out / "best.npz", cfg, step)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
    save(model, out / "final.npz", cfg, a.steps)


if __name__ == "__main__":
    main()

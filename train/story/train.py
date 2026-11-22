"""Train the story writer with quantisation-aware training, streaming through TinyStories with cast slots
(train/story/tinystories.py makes the data).

    python -m train.story.train --tag mine [--d 180] [--e 48] [--chars 1e9] [--lr 3e-3] [--graph]

The network in the film: 180 units, a 48-number embedding, 1e9 characters (about 1 h 45 min on an RTX 3060 Ti with
--graph), batch 256, sequence 128, learning rate 3e-3 with a 2% warm-up then a cosine to a tenth, AdamW (weight decay
0.01), gradients clipped at 1, seed 1.

B parallel streams through train.u8, each from its own offset, advancing L symbols a step, with the state carried from
one step to the next. Writes runs/<tag>/: log.jsonl (step, characters, train and dev bits per character, time) and
qat.npz, the checkpoint train/export.py turns into the Spectrum's weights. Then:

    python -m train.export runs/<tag>/qat.npz weights/mine.npz
    python -m reference.gru weights/mine.npz "once upon a time, there was a dog named Rex. "
    python -m z80.story_writer weights/mine.npz build/mine
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from train import DATA, RUNS
from train.gru import GRUBN, Streams, detach, evaluate

SLOTS = DATA / "stories-slots"


def save_qat(model, path, extra: dict | None = None) -> None:
    """The checkpoint as .npz: every tensor (float32; counters int64) and the network's settings in meta."""
    arrays = {k: (v.cpu().numpy() if v.dtype == torch.int64 else v.detach().float().cpu().numpy())
              for k, v in model.state_dict().items()}
    meta = {"kind": "notebook" if hasattr(model, "wk") else "grubn", "d": model.d, "e": model.e,
            "vocab": model.vocab, "head": model.head.kind, "tanh": model.tanh, **(extra or {})}
    np.savez_compressed(path, **arrays, meta=json.dumps(meta))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--d", type=int, default=180)
    ap.add_argument("--e", type=int, default=48)
    ap.add_argument("--chars", type=float, default=1e9)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--eval-every", type=float, default=100e6)
    ap.add_argument("--eval-chars", type=float, default=1e6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init", help="start from this QAT checkpoint (.npz)")
    ap.add_argument("--graph", action="store_true", help="capture each step as a CUDA graph (much faster)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUBN(a.d, a.e, vocab="slots").to(dev)
    if a.init:
        z = np.load(a.init)
        model.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files if k != "meta"})
    out = RUNS / a.tag
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((SLOTS / "meta.json").read_text())
    per_char = meta["dev"]["symbols"] / meta["dev"]["chars"]  # bits per original character, names and all
    lr0 = torch.tensor(a.lr, device=dev) if a.graph else a.lr
    opt = torch.optim.AdamW(model.parameters(), lr=lr0, betas=(0.9, 0.95), weight_decay=a.wd, capturable=a.graph)
    steps = int(a.chars // (a.batch * a.seq))
    warm = max(1, steps // 50)

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        p = (s - warm) / max(1, steps - warm)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    streams = Streams(SLOTS / "train.u8", a.batch, a.seq, a.seed)
    state = model.init_state(a.batch, dev)

    def step_fn(x, y, state):
        logits, new_state = model(x, state)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss.detach(), detach(new_state)

    graph = None
    if a.graph:
        # static tensors: the batch and the carried state go in, the loss and the new state come out. A few eager
        # steps on a side stream first (they train on the first batch, as normal steps do)
        sx, sy = streams.next(dev)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                opt.zero_grad(set_to_none=False)
                step_fn(sx, sy, state)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        opt.zero_grad(set_to_none=True)  # so the captured backward writes the gradients rather than adding to them
        with torch.cuda.graph(graph):
            gloss, gstate = step_fn(sx, sy, state)
    log = open(out / "log.jsonl", "a")
    t0, run_loss, run_n, next_eval = time.time(), 0.0, 0, a.eval_every
    for s in range(steps):
        x, y = streams.next(dev)
        if graph is not None:
            opt.param_groups[0]["lr"].fill_(lr_at(s))
            sx.copy_(x)
            sy.copy_(y)
            graph.replay()
            for dst, src in zip(state, gstate):
                dst.copy_(src)
            loss = gloss
        else:
            for g in opt.param_groups:
                g["lr"] = lr_at(s)
            opt.zero_grad(set_to_none=True)
            loss, state = step_fn(x, y, state)
        run_loss += loss.item()
        run_n += 1
        chars = (s + 1) * a.batch * a.seq
        if chars >= next_eval or s == steps - 1:
            rec = {"step": s + 1, "chars": chars, "train_bpc": run_loss / run_n / math.log(2) * per_char,
                   "dev_bpc": evaluate(model, SLOTS / "dev.u8", a.eval_chars, dev, per_char=per_char),
                   "lr": lr_at(s), "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            run_loss, run_n, next_eval = 0.0, 0, next_eval + a.eval_every
    save_qat(model, out / "qat.npz")
    print("wrote", out / "qat.npz")


if __name__ == "__main__":
    main()

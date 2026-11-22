"""Train J, with quantisation-aware training, on conversation windows that start at a conversation's '#'.

    python -m train.j.train --tag mine --data data3 --chars 100e6                 # from nothing
    python -m train.j.train --tag mine-ft --data diag-ft --chars 10e6 --init runs/mine/qat.npz
    python -m train.j.train --tag mine-long --data diag-long --chars 10e6 --seq 1024 --batch 64 --ring \\
        --init runs/mine/qat.npz --seed 2

The film's builds, step by step: J (150 units) from nothing on data3, 100M symbols, sequence 512, batch 128,
learning rate 3e-3; then fine-tuned for 10M symbols at 1e-3, on diag-ft for dg1 (seed 1) and on diag-long with
1,024-symbol windows and --ring for long1kr (seed 2). train/j/build_data.py and
train/j/fine_tune.py make the data.

The loss weighs each predicted symbol by its role: what the Spectrum writes (and the '<' that ends it) counts fully,
what you type counts --user-weight (the network still reads it, it just isn't asked to guess it as hard), the '#'
between conversations not at all. Each window starts at a '#' with the state at zero; a conversation cut by the
window's end is read again whole in the next. --ring: the notebook notes every word you type and keeps the latest 64,
as the Spectrum does (without it, a window's first 64 words only). Writes runs/<tag>/log.jsonl and qat.npz; then
`python -m train.export runs/<tag>/qat.npz weights/<name>.npz`.
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
from train.gru import detach
from train.j.model import NotebookGRU, precompute
from train.story.train import save_qat

NEW, LIKE = 2, 1  # J's two turn markers (symbols 41 and 42) start as copies of '\n', the story's paragraph break


class AlignedStreams:
    """Windows that start at a conversation's '#'. Each stream reads seq symbols from a '#', then moves to the last
    '#' inside that window, so a conversation cut by the window's end is read again whole; the network's state starts
    from zero in every window. Each window's word notes too (model.precompute)."""

    def __init__(self, folder, b: int, l: int, seed: int, user_w: float, ring: bool):
        self.data = np.fromfile(folder / "train.u8", dtype=np.uint8)
        self.role = np.fromfile(folder / "train.role", dtype=np.uint8)
        starts = np.flatnonzero(self.data == 0)
        self.starts = starts[starts < len(self.data) - l - 2]
        self.pos = self.starts[np.random.default_rng(seed).integers(0, len(self.starts), b)]
        self.l, self.ring = l, ring
        self.wt = np.array([0.0, user_w, 1.0], dtype=np.float32)

    def next(self, device):
        win = np.stack([self.data[p: p + self.l + 1] for p in self.pos])
        role = np.stack([self.role[p: p + self.l + 1] for p in self.pos])
        out = [torch.from_numpy(win[:, :-1].astype(np.int64)), torch.from_numpy(win[:, 1:].astype(np.int64)),
               torch.from_numpy(self.wt[role[:, 1:]])]
        out += [torch.from_numpy(a) for a in precompute(win, role == 1, self.ring)]
        for i, p in enumerate(self.pos):
            z = np.flatnonzero(win[i, 1: self.l] == 0)
            nxt = p + 1 + z[-1] if len(z) else p + self.l
            self.pos[i] = nxt if nxt < len(self.data) - self.l - 2 else self.starts[0]
        return [t.to(device, non_blocking=True) for t in out]


@torch.no_grad()
def evaluate(model, folder, device, ring: bool, n: int = 1024, b: int = 128, l: int = 512) -> dict:
    """Bits per symbol on n dev windows that each start at a conversation's '#', state from zero: the Spectrum's
    replies, what you typed, and both."""
    was = model.training
    model.eval()
    data = np.fromfile(folder / "dev.u8", dtype=np.uint8)
    role = np.fromfile(folder / "dev.role", dtype=np.uint8)
    starts = np.flatnonzero(data == 0)
    starts = starts[starts < len(data) - l - 2]
    pick = starts[np.linspace(0, len(starts) - 1, n).astype(np.int64)]
    tot, cnt = {1: 0.0, 2: 0.0}, {1: 0, 2: 0}
    for i in range(0, n, b):
        win = np.stack([data[p: p + l + 1] for p in pick[i: i + b]])
        rl = np.stack([role[p: p + l + 1] for p in pick[i: i + b]])
        x = torch.from_numpy(win[:, :-1].astype(np.int64)).to(device)
        aux = [torch.from_numpy(a).to(device) for a in precompute(win, rl == 1, ring)]
        logits, _ = model(x, model.init_state(len(win), device), aux)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             torch.from_numpy(win[:, 1:].astype(np.int64)).to(device).reshape(-1),
                             reduction="none").cpu().numpy()
        r = rl[:, 1:].reshape(-1)
        for k in (1, 2):
            tot[k] += float(ce[r == k].sum())
            cnt[k] += int((r == k).sum())
    model.train(was)
    ln2 = math.log(2)
    return {"reply_bps": tot[2] / cnt[2] / ln2, "typed_bps": tot[1] / cnt[1] / ln2,
            "all_bps": (tot[1] + tot[2]) / (cnt[1] + cnt[2]) / ln2}


def init_from(model: torch.nn.Module, path) -> None:
    """Load a QAT checkpoint: J's own, or the story writer's (180 units and 41 symbols: the rows for J's two turn
    markers are copied from the paragraph break's, and the notebook starts fresh)."""
    z = np.load(path)
    own = model.state_dict()
    for k in z.files:
        if k == "meta":
            continue
        v, o = torch.from_numpy(z[k]), own[k]
        if o.shape == v.shape:
            own[k] = v.clone()
        else:
            n = v.shape[0]
            new = o.clone()
            new[:n] = v
            new[n: n + NEW] = v[LIKE]
            own[k] = new
    model.load_state_dict(own)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="data3", help="data/chat/<data>: data3, diag-ft or diag-long")
    ap.add_argument("--d", type=int, default=150, help="units (with --init, the checkpoint's)")
    ap.add_argument("--chars", type=float, default=100e6)
    ap.add_argument("--lr", type=float, default=None, help="default 3e-3 from nothing, 1e-3 fine-tuning")
    ap.add_argument("--user-weight", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--eval-every", type=float, default=25e6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ring", action="store_true", help="note every typed word, keep the latest 64 (as the Spectrum)")
    ap.add_argument("--init", help="start from this QAT checkpoint (.npz)")
    a = ap.parse_args()
    folder = DATA / "chat" / a.data
    lr = a.lr or (1e-3 if a.init else 3e-3)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    if a.init:
        a.d = json.loads(str(np.load(a.init)["meta"]))["d"]
    model = NotebookGRU(a.d).to(dev)
    if a.init:
        init_from(model, a.init)
    out = RUNS / a.tag
    out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=torch.tensor(lr, device=dev), betas=(0.9, 0.95), weight_decay=0.01,
                            capturable=True)
    steps = int(a.chars // (a.batch * a.seq))
    warm = max(1, steps // 50)

    def lr_at(s):
        if s < warm:
            return lr * (s + 1) / warm
        p = (s - warm) / max(1, steps - warm)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    streams = AlignedStreams(folder, a.batch, a.seq, a.seed, a.user_weight, a.ring)
    state = model.init_state(a.batch, dev)

    def step_fn(x, y, w, state, *aux):
        logits, new_state = model(x, state, aux)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
        loss = (ce * w.reshape(-1)).sum() / w.sum().clamp(min=1.0) + model.gate_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss.detach(), detach(new_state)

    # a CUDA graph: a few eager steps on a side stream, then capture one step and replay it with each batch
    static = streams.next(dev)  # x, y, w and the notebook's notes
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            opt.zero_grad(set_to_none=False)
            step_fn(*static[:3], state, *static[3:])
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        gloss, _ = step_fn(*static[:3], state, *static[3:])
    log = open(out / "log.jsonl", "a")
    t0, run_loss, run_n, next_eval = time.time(), 0.0, 0, a.eval_every
    for s in range(steps):
        batch = streams.next(dev)
        opt.param_groups[0]["lr"].fill_(lr_at(s))
        for dst, src in zip(static, batch):
            dst.copy_(src)
        for st in state:
            st.zero_()
        graph.replay()
        run_loss += gloss.item()
        run_n += 1
        chars = (s + 1) * a.batch * a.seq
        if chars >= next_eval or s == steps - 1:
            rec = {"step": s + 1, "chars": chars, "train_loss_bits": run_loss / run_n / math.log(2),
                   **evaluate(model, folder, dev, a.ring), "lr": lr_at(s), "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            run_loss, run_n, next_eval = 0.0, 0, next_eval + a.eval_every
    save_qat(model, out / "qat.npz", {"k": model.k})
    print("wrote", out / "qat.npz")


if __name__ == "__main__":
    main()

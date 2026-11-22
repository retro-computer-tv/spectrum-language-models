"""Build every tape in run/ from the weights in weights/ and the generators in z80/ (Python 3 and numpy, and the
sjasmplus assembler: on the PATH, or named by the SJASMPLUS environment variable).

    python build.py            ->  run/*.tap; build/<tape>/: the assembly source, its data blobs, the layout
    python build.py --check    ->  the same into build/ only, each tape compared byte for byte with run/

The film's tapes have the opening or the lines built in, exactly as the film recorded them; the -keyboard tapes are
the same networks waiting for you to type.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

from reference.weights import load
from z80 import j, story_writer, zx_lm

HERE = Path(__file__).resolve().parent
RUN, OUT, W = HERE / "run", HERE / "build", HERE / "weights"

ZARA = "once upon a time, there was a little girl named Zara. "
OTTOLINE = ["hello! my name is Ottoline.", "i have a little cat called Pip.", "my favourite colour is red.",
            "no, sorry, my favourite colour is green.", "what is my favourite colour?", "what is my name?"]
WENDELINE = ["hello there! i am Wendeline.", "i have a goat called Horatio.", "my cousin is called Isolde.",
             "my secret word is apple.", "no, sorry, my secret word is moon.", "what is my goat called?",
             "who is my cousin?", "what is my secret word?", "what is my name?"]
CASPIAN = ["hi! people call me Caspian.", "my favourite colour is purple.", "my sister is called Ophelia.",
           "actually, my favourite colour is orange.", "what is my favourite colour?", "what is my sister called?",
           "what is my name?"]

# tape: (generator, weights, the file it writes, settings)
TAPES = {
    "story-writer": (story_writer, "story-writer", "story.tap", dict(auto=ZARA, seed=1, temp=0.7, minp=13, n=280)),
    "story-writer-keyboard": (story_writer, "story-writer", "story.tap", dict(seed=1, temp=0.7, minp=13, n=280)),
    "j-dg1": (j, "j-dg1", "chat.tap", dict(auto=OTTOLINE, seed=1, temp=0.6, minp=13)),
    "j-dg1-margin1": (j, "j-dg1", "chat.tap", dict(auto=WENDELINE, seed=1, temp=0.6, minp=13, margin=1)),
    "j-long1kr": (j, "j-long1kr", "chat.tap", dict(auto=CASPIAN, seed=1, temp=0.6, minp=13, margin=1)),
    "j-dg1-keyboard": (j, "j-dg1", "chat.tap", dict(seed=1, temp=0.6, minp=13)),
    "j-long1kr-keyboard": (j, "j-long1kr", "chat.tap", dict(seed=1, temp=0.6, minp=13, margin=1)),
    "zx-lm": (zx_lm, "zx-lm", "zxlm.tap", dict(seed=1)),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="build into build/ only and compare with run/")
    ap.add_argument("tapes", nargs="*", help="only these (default: all)")
    a = ap.parse_args()
    RUN.mkdir(exist_ok=True)
    bad = 0
    for name in a.tapes or TAPES:
        gen, weights, made, settings = TAPES[name]
        folder = OUT / name
        gen.build(load(W / f"{weights}.npz"), folder, **settings)
        data = (folder / made).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if a.check:
            ours = RUN / f"{name}.tap"
            same = ours.exists() and ours.read_bytes() == data
            bad += not same
            print(f"run/{name}.tap  {'identical' if same else 'DIFFERENT'}  {digest}")
        else:
            shutil.copyfile(folder / made, RUN / f"{name}.tap")
            print(f"run/{name}.tap  {len(data)} bytes  sha256 {digest}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

# Language models on a 48K ZX Spectrum

The three neural networks from Retro Computer TV's film *Can a 48K ZX Spectrum Hold a Conversation?*, as tapes you can load, with the code that builds them, the trained weights they carry and the code that trained them.

They were trained on a PC. On the Spectrum, every letter they write is worked out by the Z80 itself, from the weights on the tape: no reply is stored anywhere, and nothing needs more than an unmodified 48K machine.

| Program | What it is | What to expect |
|---|---|---|
| The story writer | A 180-unit recurrent network (a GRU) with 134,669 parameters, trained on a billion characters of children's stories. | You give it the start of a story; it carries on, one letter at a time, just under a second a letter. It keeps track of names, but the story wanders and it makes up words. |
| J | A 150-unit GRU (104,560 parameters) trained on conversations, with a notebook: it keeps the last 64 words you typed and can copy from them. | You type a line, it reads it and writes a reply, letter by letter: a short reply takes half a minute or so. It sometimes remembers what you told it. |
| ZX-LM | A 96-unit GRU (61,472 weights) that reads and writes whole words and remembers who has what. | Tell it who owns what ("Ana has a kite.", "Ana gave the kite to Tom."), then ask ("Who has the kite?"). It says "ok." to a statement and answers a question in ten to twenty seconds. |

## Run them

Nothing to build: everything ready to run is in the [run](run) folder. Choose a 48K Spectrum in your emulator ([Fuse](https://fuse-emulator.sourceforge.net/) and most others) and open a tape; most emulators load and start it for you, and if not, type `LOAD ""` and press ENTER. Run it at normal speed: the programs take minutes, as on the real machine, and that is part of the point.

On a real 48K, play the `.wav` beside the tape into the EAR socket and type `LOAD ""`. Loading takes about four minutes for the story writer, three and a half for J and under three for ZX-LM.

| Tape | What it does |
|---|---|
| `story-writer.tap` | The film's story: it types "once upon a time, there was a little girl named Zara." itself, then writes the same 280 symbols you saw in the film (seed 1), in about four and a half minutes. |
| `story-writer-keyboard.tap` | The same network, waiting for you. Type the start of a story and press ENTER; it writes 280 symbols, then shows the seed it used. Any key starts another story. |
| `j-dg1.tap` | The film's first conversation with J ("hello! my name is Ottoline."): the lines are built in and typed for you, and J writes its own replies to them. About four and a half minutes. |
| `j-dg1-margin1.tap` | The goat and the cousin: the same J, with its notebook's margin of 1 (only the notes nearly as good as the best one count). About six minutes. |
| `j-long1kr.tap` | The later J build, trained on longer conversations ("hi! people call me Caspian."), margin 1. About five minutes. |
| `j-dg1-keyboard.tap`, `j-long1kr-keyboard.tap` | J, waiting for you to talk to it. |
| `zx-lm.tap` | ZX-LM, waiting for you (the tape the film's final session was typed into). |

**Typing.** Small letters as normal; CAPS SHIFT with a letter for a capital; SYMBOL SHIFT for punctuation (`.` is SYMBOL SHIFT and M, `?` is SYMBOL SHIFT and C, `,` is SYMBOL SHIFT and N, `'` is SYMBOL SHIFT and 7, `!` is SYMBOL SHIFT and 1). DELETE (CAPS SHIFT and 0) rubs out, ENTER sends. In Fuse, the PC's Shift is CAPS SHIFT and Ctrl is SYMBOL SHIFT. A line holds up to 120 characters.

**The story writer and J: capitals only for names.** A capital letter tells the network that a word is a name ("my dog is called Rex"), as in its training. "Hello" with a capital becomes a name too, and J will happily call you Hello.

**J:** after ENTER it reads your line (about 0.7 seconds a letter) and then writes its reply (about 0.8 seconds a letter). The square at the bottom right of the screen is magenta while it waits for you or reads, and red while it writes. An empty line starts a new conversation.

**ZX-LM:** after ENTER it reads and thinks for 8 to 16 seconds, then writes its reply in a second or three. Every word it doesn't know (any name, object or adjective) becomes one of 16 reference numbers, handed out at random from the moment you first press ENTER, so the same lines can come out a little differently from one go to the next. An empty line starts a new conversation.

The tapes hold none of the ROM: the programs use the 48K's own ROM for the screen and keyboard. Emulators include it under Amstrad's permission, and a real Spectrum has its own.

## Building

To change a program and build its tape again you need Python 3 with numpy, and the [sjasmplus](https://github.com/z00m128/sjasmplus) assembler (the tapes here were built with 1.24.0; on Windows take the ready-made download from its [releases page](https://github.com/z00m128/sjasmplus/releases), not the source code).

```sh
pip install numpy
python build.py              # every tape, into run/ (SJASMPLUS=/path/to/sjasmplus if it isn't on the PATH)
python build.py --check      # build them all into build/ and compare with run/, byte for byte
python build.py zx-lm        # just one
```

Each tape's assembly source is written to `build/<tape>/` (`engine.asm`, `chat.asm` or `zxlm.asm`, with the data files it includes and `layout.json`, every address and size), so you can read the Z80 code there. What's here:

| Folder | What |
|---|---|
| [z80](z80) | Python that writes each program's Z80 source from its weights: [story_writer.py](z80/story_writer.py), [j.py](z80/j.py), [zx_lm.py](z80/zx_lm.py), and the routines they share ([kernel.py](z80/kernel.py) is the matrix kernel). |
| [reference](reference) | The same three networks in Python, in the Spectrum's own integer arithmetic. Given the same weights and the same seed they write exactly what the Spectrum writes, so you can try a network on the PC first. |
| [weights](weights) | The weights each tape carries, and the checkpoints they came from ([qat](weights/qat)). |
| [train](train) | Training your own. |

Trying a network on the PC:

```sh
python -m reference.gru weights/story-writer.npz "once upon a time, there was a little girl named Zara. "
python -m reference.notebook weights/j-dg1.npz "hello! my name is Ottoline.|what is my name?"
python -m reference.zx_lm weights/zx-lm.npz "Kofi has a teapot.|Kofi gave the teapot to Paloma.|who has the teapot?"
```

**The weights.** Each `weights/*.npz` is a numpy archive of integer arrays plus a `meta` string of JSON with the network's sizes. Every matrix is stored as trits (-1, 0 or +1, one row per output) with three numbers per row: an 8-bit mantissa M, a shift and a bias, so the row's result is `((sum of the trits times the inputs) * M >> shift) + bias`. Embeddings are integers from -25 to 25. [reference/weights.py](reference/weights.py) reads and writes them, and the top of each file in `reference/` says exactly how the Spectrum uses them. `weights/qat/` holds the quantisation-aware training checkpoints each was exported from (floating-point latent weights and batch-norm statistics, also as `.npz`); `python -m train.export` turns one into the other. The story writer's export was run on a CUDA GPU: run on a CPU, one of its 7,380 second-plane head trits rounds the other way.

## Training your own

You need PyTorch (a CUDA graphics card for the full runs), numba and pyarrow (`pip install -r requirements.txt`). Data goes in `data/` and training runs in `runs/`, or wherever `SPECTRUM_LM_DATA` and `SPECTRUM_LM_RUNS` point. No dataset is included here: the scripts say what to download and where to put it.

**The story writer.** Download `TinyStoriesV2-GPT4-train.txt` and `TinyStoriesV2-GPT4-valid.txt` from [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (licence: CDLA-Sharing-1.0, as its card states) into `data/tinystories/`, then:

```sh
python -m train.story.tinystories                 # the 37-symbol text, the name list, names as cast slots
python -m train.story.train --tag mine --graph    # 180 units, 1e9 characters: about 1 h 45 min on an RTX 3060 Ti
python -m train.export runs/mine/qat.npz weights/mine.npz
python -m z80.story_writer weights/mine.npz build/mine      # build/mine/story.tap
```

**J.** Download the four public dialogue sets listed at the top of [train/j/public.py](train/j/public.py) (TinyDialogues, SODA, Synthetic-Persona-Chat and everyday-conversations, each with its own licence), and make the story data first (J's name lists come from it). Then the conversations: the public ones, plus thousands made by a generator ([train/j/skills.py](train/j/skills.py)) that plants facts (a pet's name, a favourite colour) and asks for them later, and the fine-tuning sets of facts, instructions and things to remember ([train/j/fine_tune.py](train/j/fine_tune.py)). The film's builds were trained like this:

```sh
python -m train.j.public && python -m train.j.build_data && python -m train.j.fine_tune short && python -m train.j.fine_tune long
python -m train.j.train --tag j --data data3 --chars 100e6
python -m train.j.train --tag j-dg1 --data diag-ft --chars 10e6 --init runs/j/qat.npz
python -m train.j.train --tag j-long1kr --data diag-long --chars 10e6 --seq 1024 --batch 64 --ring --seed 2 --init runs/j/qat.npz
python -m train.export runs/j-dg1/qat.npz weights/my-j.npz
python -m z80.j weights/my-j.npz build/my-j [--margin 1]
```

**ZX-LM.** Its conversations come from a little simulated world of people and objects ([train/zx_lm/world2.py](train/zx_lm/world2.py) and [frames2.py](train/zx_lm/frames2.py)), made fresh for every batch, so nothing needs downloading. A float network first, then the Spectrum's, converted from it weight for weight and fine-tuned in the Spectrum's arithmetic (from nothing, the Spectrum's network stalled near 57% on its development set; converted, it reached 77.5%):

```sh
python -m train.zx_lm.train --tag float                       # 6,000 batches: 55 minutes on an RTX 3060 Ti
python -m train.zx_lm.convert runs/float/best.npz runs/conv/best.npz
python -m train.zx_lm.train --tag spectrum --quant --init runs/conv/best.npz --teacher runs/float/best.npz --steps 4000 --lr 1e-3
python -m train.zx_lm.export runs/spectrum/best.npz weights/my-zx-lm.npz
python -m z80.zx_lm weights/my-zx-lm.npz build/my-zx-lm
```

Every training step computes what the Spectrum will compute: ternary weights, inputs rounded to 25ths, the gates read from 256-entry tables, with each rounding passed straight through in the backward pass ([train/qat.py](train/qat.py)). So a network that does well in training does as well on the Spectrum: in integers, the story writer scores 1.397 bits a character on held-out stories against 1.394 for the network it was exported from.

## How it fits in 48K

A Z80 has no multiply instruction, and the programs have 41,472 bytes, from address 24064 to the top of memory, for their code, weights and working space. What makes these networks fit and run:

- **Five weights to a byte.** Every weight is -1, 0 or +1, and five of them have 3^5 = 243 patterns, so a byte holds five. The story writer's matrices fit in 27,792 bytes.
- **Pages of sums instead of multiplying.** Before each pass over a matrix, the Spectrum takes the inputs five at a time and fills a 256-byte page with every sum those five inputs can make under the 243 patterns. A weight byte is then just an index into its page: one lookup and one add covers five weights, 7 to 10 T-states a weight at 3.5 MHz ([z80/kernel.py](z80/kernel.py)).
- **The stack pointer, borrowed.** The weight bytes stream through the stack pointer, two at a time with POP, the fastest way the Z80 has to read memory. Interrupts are off while it does, or the ROM's interrupt would push its return address over the weights.
- **Cast slots.** The story writer and J never have to remember how to spell a name. The first time a name appears the network writes `*` and spells it; after that it writes a slot number (1, 2 or 3), and the Spectrum keeps the spelling in a table of three 12-byte rows and prints it. To the network, Lily and Zara are both just "1".
- **J's notebook.** Every word you type is kept with a 16-number key the network writes: 64 notes of 33 bytes, 2,112 bytes. At each letter of its reply, J's query picks the notes that fit what it has written of the word so far, and how much it trusts them; their next letters get a bonus. Choosing is learned; the copying is exact.
- **ZX-LM's word references.** ZX-LM reads and writes words, not letters, from a vocabulary of 133. Any other word becomes one of 16 reference numbers, and the Spectrum keeps its spelling (16 bytes each). Its memory holds one slot per mention, 64 of them, each a reference number and a 16-number key the network writes, and to answer, it either writes one of its 26 own words or points at a remembered one, whose spelling the Spectrum copies. No weight spells anything.

The memory budget, in bytes (each from the assembler's own addresses, in `build/<tape>/layout.json`):

| | Story writer | J | ZX-LM |
|---|---|---|---|
| Weights, five to a byte | 27,792 | 21,600 | 11,952 |
| Row scales and biases, embedding, tables | 6,086 | 6,578 | 8,621 |
| Code | 2,179 | 3,310 | 3,540 |
| Working memory (lookup pages, sums, the state, notes) | 5,001 | 6,741 | 8,475 |
| Free | 414 | 3,243 | 8,884 |

(The story writer's figures are the film's Zara tape; the keyboard tape has 464 bytes free. J's are the keyboard tape's, and its stack never needs more than 30 bytes of the free space.)

## What didn't work

None of them holds a reliable conversation. The story writer keeps its characters' names, but its stories drift and it invents words ("Zara felt alerabed"). J can bring back something it was told, a corrected favourite colour or a name, but it mixes things up: asked "what is my goat called?", it gave the cousin's name, and it will greet you again in the middle of a conversation. ZX-LM answered 5 of 10 questions right on a frozen final set it had never seen, written without sight of its training language, and between 36 and 43 per cent on three independently written sets of 40 conversations. Its commonest mistake is pointing at the wrong statement: asked who someone gave a thing to, it names the thing's latest owner.

## Do better

There is room on the machine: ZX-LM has 8,884 bytes free, enough for a 128-unit network instead of 96, or a couple of hundred more words. Wider English, a better way to choose between two remembered facts, faster gates (they take 41% of J's time): if you make a 48K do better, we'd love to hear about it.

## Licence

MIT, see [LICENSE](LICENSE). The datasets used for training are not included and keep their own licences.

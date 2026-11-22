"""J, the conversation network, for a 48K Spectrum: a 150-unit ternary GRU with a notebook of 64 words, from its
weights (weights/*.npz).

    python -m z80.j weights/j-dg1.npz build/j-dg1 [--auto "line|line|..."] [--margin 1] [--seed 1] [--temp 0.6]
        [--minp 13]

Writes chat.asm, the data blobs it includes, chat.bin (the CODE block from 0x5E00), chat.tap (the BASIC loader plus
the CODE block) and layout.json (every address and size: the memory map) into the folder. Without --auto you type on
the Spectrum's keyboard: ENTER ends your line, and an empty line starts a new conversation. With --auto the lines are
built in and typed for you (the film's takes).

The turn protocol: '#' starts a conversation; each of your lines is fed symbol by symbol (names, a word with a capital,
become cast slots as the network's training wrote them), then '>'; the Spectrum writes until it writes '<' (or 120
symbols, then '<'). Every symbol of the reply is sampled from the network's own logits: the GRU's head plus the
notebook's bonus, nothing else. reference/notebook.py is the same network in Python.

Per symbol, one pass over the state computes the head, the GRU's h part and the notebook's 34 rows (its query, key,
gate and trust all read the same quantised state); the pass over the input and the gates follow as in the story
writer (story_writer.py). The notebook's routines are notebook.py's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reference.notebook import MARGIN1_CUT, rng_start
from reference.text import CSTOI, CVOCAB, MAXNAME, NSLOTS, SPEC_END, YOU_END
from reference.weights import load
from z80 import story_writer as E
from z80.cast_slots import VARS as SLOT_VARS
from z80.cast_slots import slots as slot_io
from z80.kernel import BIAS, P, build_lut_asm, encode_block, matvec_asm, pad_cols
from z80.notebook import MAXW, NB_ROWS, NNOTES, NOTE, NOTEBOOK, tables
from z80.notebook import VARS as NB_VARS
from z80.routines import MUL, RNG, SAMPLE
from z80.screen_keys import IO
from z80.tape import ORG, assemble, tap

MAXREPLY = 120  # reference/notebook.py
TITLE = "a neural network you can talk to"


def typed(line: str) -> list[int]:
    """A line as read_prompt stores it: each character's symbol, bit 7 set for a capital."""
    out = []
    for ch in line:
        s = CSTOI[ch.lower()]
        assert 2 <= s <= 36, f"{ch!r} can't be typed"
        out.append(s | (0x80 if ch.isupper() else 0))
    return out


def plan_and_blobs(ex: dict) -> tuple[dict, dict]:
    exb = dict(ex, kind="grubn")
    pl = E.plan(exb)
    pl.update(vocab=CVOCAB, V=len(CVOCAB), slots=True)
    bl = E.blobs(exb, pl)
    nb_rows = pad_cols(np.concatenate([ex["q_T"], ex["k_T"], ex["g_T"], ex["c_T"]]), pl["dpad"])
    assert nb_rows.shape[0] == NB_ROWS
    w, offs, g0 = bytearray(), [], 0
    for b in pl["hblocks"]:
        offs.append(len(w))
        w += encode_block(nb_rows, g0, b)
        g0 += b
    pnb = bytearray()
    for name in ("q", "k"):
        assert (ex[f"{name}_sh"] >= 9).all() and (ex[f"{name}_sh"] <= 23).all()
        for M, sh in zip(ex[f"{name}_M"], ex[f"{name}_sh"]):
            pnb += bytes([int(M), int(sh), 0])
    for name in ("g", "c"):
        assert int(ex[f"{name}_sh"][0]) >= 8
        pnb += bytes([int(ex[f"{name}_M"][0]), int(ex[f"{name}_sh"][0]), int(ex[f"{name}_B"][0]) & 0xFF])
    assert 9 <= int(ex["att_sh"][0]) <= 23
    bl.update(nbw=bytes(w), nboffs=offs, pnb=bytes(pnb), **tables())
    return pl, bl


def buffers(pl: dict) -> list[tuple[str, int]]:
    """The zeroed RAM, not on the tape: the engine's accumulators and state, the notebook's."""
    d, VH = pl["d"], pl["V"] * pl["hp"]
    return [("ACC", 2 * (VH + 6 * d)), ("NBACC", 2 * NB_ROWS), ("HSTATE", 3 * d), ("XIN", pl["epad"]),
            ("NOTES", NNOTES * NOTE)] + list(NB_VARS) + [("rlen", 1), ("cur_sym", 1), ("seeded", 1)]


def hpass_asm(pl: dict, bl: dict, label: str, head: bool) -> list[str]:
    """The pass over the state: (head,) the GRU's h rows and the notebook's rows, LUT block by LUT block. Without
    the head (while you type) the logits aren't wanted, so its rows are skipped."""
    d, VH = pl["d"], pl["V"] * pl["hp"]
    init_h = (-BIAS * pl["Gh"]) & 0xFFFF
    first, rows = (0, VH + 3 * d) if head else (VH, 3 * d)
    L = [f"{label}:", f"    ld hl,ACC + {2 * first}", f"    ld bc,{rows}", f"    ld de,{init_h}", "    call fill16",
         "    ld hl,NBACC", f"    ld bc,{NB_ROWS}", f"    ld de,{init_h}", "    call fill16"]
    g0 = 0
    for k, b in enumerate(pl["hblocks"]):
        page = f"LUTPAGE+{P - b}"
        L += [f"    ld ix,HQ+{g0 * pl['g']}", f"    ld h,{page}", f"    ld a,{b}", "    call bl_group",
              f"    ld hl,WEIGHTS+{bl['hoffs'][k] + first * b}", f"    ld ix,ACC + {2 * first}", f"    ld bc,{rows}",
              f"    ld de,mv_g{P - b}", f"    ld a,{page}", "    call matvec",
              f"    ld hl,NBW+{bl['nboffs'][k]}", "    ld ix,NBACC", f"    ld bc,{NB_ROWS}", f"    ld de,mv_g{P - b}",
              f"    ld a,{page}", "    call matvec"]
        g0 += b
    return L + ["    ret"]


def xpass_asm(pl: dict, bl: dict) -> list[str]:
    """The x pass for symbol A (the embedding through XIN, block by block), as the story writer's."""
    d = pl["d"]
    init_x = (-BIAS * pl["Ge"]) & 0xFFFF
    L = ["xpass_x:", "    call build_xin", "    ld hl,ACC_XFILL", f"    ld bc,{3 * d}", f"    ld de,{init_x}",
         "    call fill16"]
    g0 = 0
    for k, b in enumerate(pl["xblocks"]):
        page = f"LUTPAGE+{P - b}"
        L += [f"    ld ix,XIN+{g0 * pl['g']}", f"    ld h,{page}", f"    ld a,{b}", "    call bl_group",
              f"    ld hl,WEIGHTS+{bl['xoffs'][k]}", "    ld ix,ACC_XROWS", f"    ld bc,{3 * d}",
              f"    ld de,mv_g{P - b}", f"    ld a,{page}", "    call matvec"]
        g0 += b
    return L + ["    ret", "xpass_finalize:", "    call xpass_x", "    jp finalize"]


MAIN = r"""
start:
    di
    ld sp,0
    ld iy,0x5C3A
    call zero_buffers
    ld a,2
    call 0x1601
new_conv:
    call 0x0DAF
    ld hl,msg_title
    call print_str
    call reset_state        ; the state, the slot table
    call nb_reset
    xor a
    ld (seeded),a
    call feed               ; '#'
turn:
    ld a,MARK_READ
    ld (0x5AFF),a
    ld hl,msg_you
    call print_str
    call read_line          ; Z: no line (the conversation is over)
    jp z,conv_end
    call convert_feed       ; your line, names as cast slots (on the screen as typed)
    ld a,YOU_END
    call print_sym          ; ends a name being spelled; prints nothing
    ld a,YOU_END
    call feed
    ld hl,msg_spec
    call print_str
    ld a,MARK_WRITE
    ld (0x5AFF),a
    call reply
    ld a,13
    call print_ch
    jr turn

feed:                       ; A = a symbol you typed (or '#'): the state moves on, the notebook notes it
    push af
    call hpass_feed
    pop af
    push af
    call nb_push
    pop af
    jp xpass_finalize

reply:                      ; the Spectrum writes until '<'
    xor a
    ld (rlen),a
rp_loop:
    call hpass
pf_hpass:
    call head_logits
pf_head:
    call nb_bonus
pf_bonus:
    call slot_mask
    ld hl,0x8000            ; only you end your turn
    ld (LOGITS + 2 * YOU_END),hl
rl_logits:
    ld a,(rlen)
    cp MAXREPLY
    ld a,SPEC_END
    jr nc,rp_have
    call sample
rp_have:
    ld (cur_sym),a
rl_sampled:
    or a
    jr nz,rp_show
    xor a                   ; '#': the slot table is cleared, nothing shown
    ld (nslots),a
    ld h,a
    ld l,a
    ld (spell_ptr),hl
    jr rp_fed
rp_show:
    call print_sym
rp_fed:
    ld a,(cur_sym)
    call nb_push
pf_push:
    ld a,(cur_sym)
    call xpass_finalize
pf_final:
    ld a,(cur_sym)
    cp SPEC_END
    ret z
    ld hl,rlen
    inc (hl)
    jr rp_loop
"""

AUTO_IO = r"""
read_line:                  ; the next built-in line into PROMPT, shown as if typed; Z if none is left
    ld hl,(auto_ptr)
    ld a,(hl)
    cp 0xFE
    ret z
    ld de,PROMPT
rl_copy:
    ld a,(hl)
    ld (de),a
    inc hl
    inc de
    cp 0xFF
    jr nz,rl_copy
    ld (auto_ptr),hl
    ld hl,PROMPT
rl_show:
    ld a,(hl)
    cp 0xFF
    jr z,rl_shown
    push hl
    call print_typed
    pop hl
    inc hl
    jr rl_show
rl_shown:
    ld a,13
    call print_ch
    or 1                    ; NZ: a line
    ret
conv_end:
    ld a,MARK_END
    ld (0x5AFF),a
    ld hl,msg_end
    call print_str
done:
    jr done
auto_ptr: dw LINES
"""

TYPED_IO = r"""
read_line:                  ; your line from the keyboard into PROMPT; Z if it was empty
    call read_prompt
    ld a,(PROMPT)
    cp 0xFF
    jr nz,rl_some
    xor a
    ret
rl_some:
    or 1
    ret
conv_end:                   ; an empty line: a new conversation
    jp new_conv
"""


def chat_io() -> list[str]:
    """The story writer's input, printing and cast slots (screen_keys.py, cast_slots.py), with the sampler seeded
    once a conversation, at your first ENTER (one sampler for the whole conversation)."""
    io = slot_io(IO)
    old = "rp_seed:\n    ld (seed),hl\n    ld (seed_used),hl\n"
    new = ("rp_seed:\n    ld a,(seeded)\n    or a\n    jr nz,rp_kept\n    inc a\n    ld (seeded),a\n    ld (seed),hl\n"
           "    ld (seed_used),hl\nrp_kept:\n")
    assert io.count(old) == 1
    return io.replace(old, new).splitlines()


def program(pl: dict, bl: dict, ex: dict, auto: list[str] | None, temp: float, minp: int, seed: int,
            nb_cut: int) -> str:
    d, V = pl["d"], pl["V"]
    VH = V * pl["hp"]
    tinv = 0 if temp == 0 else round(64 / temp)
    vocab_print = [13 if c == "\n" else ord(c) for c in CVOCAB]
    L = [f"V equ {V}", "NTYPE equ 35", f"NUNITS equ {d}", f"TINV equ {tinv}", f"MINP equ {minp}",
         f"MBOUND equ {pl['m']}", f"YOU_END equ {YOU_END}", f"SPEC_END equ {SPEC_END}", f"MAXREPLY equ {MAXREPLY}",
         f"ATT_M equ {int(ex['att_M'][0])}", f"ATT_SH equ {int(ex['att_sh'][0])}", f"NOTE equ {NOTE}",
         f"NB_CUT equ {nb_cut}",
         f"MAXW equ {MAXW}", f"NNOTES equ {NNOTES}", "SYM_STAR equ 37", "SYM_ONE equ 38", f"NSLOTS equ {NSLOTS}",
         f"MAXNAME equ {MAXNAME}", "MARK_READ equ 0x18", "MARK_WRITE equ 0x10", "MARK_END equ 0x20",
         f"    org {ORG}",
         "SIG_PAGE equ $ >> 8", '    incbin "sig.bin"', "EXP_PAGE equ $ >> 8", '    incbin "exp.bin"',
         "vocab_chars: db " + ",".join(map(str, vocab_print)),
         f"msg_title: db 16,0,'{TITLE}',13,0xFF", "msg_you: db 13,16,1,'you: ',0xFF", "msg_spec: db 16,0,0xFF",
         "msg_end: db 13,16,0,'(the end of the conversation)',13,0xFF", f"seed_used: dw {rng_start(seed)}",
         'PNB: incbin "pnb.bin"', 'SQ4: incbin "sq4.bin"']
    if auto is not None:  # the built-in lines sit in the gap before the next page boundary
        lines = [b for ln in auto for b in typed(ln) + [0xFF]] + [0xFE]
        L += ["LINES:"] + ["    db " + ",".join(map(str, lines[i: i + 64])) for i in range(0, len(lines), 64)]
    L += ["EXP_DATA_END:", "    align 256", "LOGSIG_PAGE equ $ >> 8", '    incbin "logsig.bin"',
          "SOFTPLUS_PAGE equ $ >> 8", '    incbin "softplus.bin"',
          'WEIGHTS: incbin "weights.bin"', 'NBW: incbin "nbw.bin"', 'PUNITS: incbin "units.bin"',
          'PHEAD: incbin "head.bin"', 'EMB: incbin "emb.bin"', "CODE_START:"]
    L += MAIN.splitlines() + (AUTO_IO if auto is not None else TYPED_IO).splitlines()
    L += hpass_asm(pl, bl, "hpass", True) + hpass_asm(pl, bl, "hpass_feed", False) + xpass_asm(pl, bl)
    L += E.finalize_src(pl) + build_lut_asm() + matvec_asm()
    L += [MUL, RNG.replace("seed: dw 1", f"seed: dw {rng_start(seed)}"), SAMPLE] + NOTEBOOK.splitlines()
    L += E.build_xin_asm(pl) + chat_io()
    L += [f"VH equ {VH}", "CODE_END:", "TAPE_END:", "BUFFERS:", "    align 256", "LUTPAGE equ $ >> 8",
          f"    ds {P * 256}"]
    L += [f"{nm}: ds {sz}" for nm, sz in buffers(pl)]
    L += ["NOTES_END equ NOTES + NNOTES * NOTE", "H16 equ HSTATE", f"HQ equ HSTATE + {2 * d}", "BUFFERS_END:",
          "RAM_END:", "    assert $ <= 0xFF00", f"ACC_XROWS equ ACC + {2 * (VH + 3 * d)}",
          f"ACC_XFILL equ ACC + {2 * (VH + 3 * d)}", "LOGITS equ ACC", "FITS equ LUTPAGE * 256",
          "QV equ LUTPAGE * 256 + 256"]
    pb = 0x5B00  # the printer buffer: the line you typed and the scratch variables, as the story writer
    for name, size in (("PROMPT", 130), ("WEIGHTS8", V), ("count", 2), ("fi_acc", 2), ("fi_h", 2), ("fi_hq", 2),
                       ("fi_n", 2), ("fi_r", 1), ("fi_z", 1), ("fi_xn", 2), ("fi_n14", 2), ("fi_t", 2), ("smax", 2),
                       ("sarg", 1), ("stotal", 2), ("sprod", 3)) + SLOT_VARS:
        L += [f"{name} equ {pb}"]
        pb += size
    assert pb <= 0x5C00, pb
    return "\n".join(L) + "\n"


def build(ex: dict, out: Path, auto: list[str] | None = None, seed: int = 1, temp: float = 0.6, minp: int = 13,
          margin: int = 0) -> dict:
    """Write the tape and its sources into out; returns the layout (addresses and sizes, and the memory map).
    margin 1 lets only the notes within a nat of the best take part (the notebook's cut-off: notes 17 sixteenths of
    a nat or more below the best get no weight, against 100 by default, where EXP itself reaches zero)."""
    assert ex.get("kind") == "notebook", "a notebook network's weights"
    nb_cut = MARGIN1_CUT if margin else int(np.asarray(ex.get("nb_cut", 100)).reshape(-1)[0])
    pl, bl = plan_and_blobs(ex)
    out.mkdir(parents=True, exist_ok=True)
    for k in ("weights", "units", "head", "emb", "sig", "exp", "nbw", "pnb", "logsig", "softplus", "sq4"):
        (out / f"{k}.bin").write_bytes(bl[k])
    (out / "chat.asm").write_text(program(pl, bl, ex, auto, temp, minp, seed, nb_cut), encoding="ascii")
    sym = assemble(out, "chat")
    code = (out / "chat.raw").read_bytes()[: sym["TAPE_END"] - ORG]
    (out / "chat.bin").write_bytes(code)
    (out / "chat.tap").write_bytes(tap(sym["start"], code, b"chat", b"notebook"))
    lay = {"org": ORG, "tape_bytes": len(code), "start": sym["start"], "auto": auto, "seed": seed, "temp": temp,
           "minp": minp, "nb_cut": nb_cut, "map": memory_map(sym, bl, pl), "sym": sym}
    (out / "layout.json").write_text(json.dumps(lay, indent=1))
    return lay


def memory_map(sym: dict, bl: dict, pl: dict) -> list[dict]:
    """Every region of the 48K's RAM, from the assembler's own addresses."""
    regions = [("screen (pixels and colour)", 0x4000, 0x5B00), ("printer buffer: your line, scratch", 0x5B00, 0x5C00),
               ("ROM system variables, channels, BASIC loader", 0x5C00, ORG),
               ("tables: sigmoid, exp; strings, notebook parameters, squares", ORG, sym["EXP_DATA_END"]),
               ("(alignment)", sym["EXP_DATA_END"], sym["LOGSIG_PAGE"] << 8),
               ("tables: log-sigmoid, softplus", sym["LOGSIG_PAGE"] << 8, sym["WEIGHTS"]),
               ("weights: head, GRU h and x rows (5 trits a byte)", sym["WEIGHTS"], sym["NBW"]),
               ("weights: notebook query, key, gate, trust", sym["NBW"], sym["PUNITS"]),
               ("row parameters (GRU)", sym["PUNITS"], sym["PHEAD"]), ("row parameters (head)", sym["PHEAD"], sym["EMB"]),
               ("embedding", sym["EMB"], sym["CODE_START"]), ("code", sym["CODE_START"], sym["CODE_END"]),
               ("(alignment)", sym["CODE_END"], sym["LUTPAGE"] << 8),
               ("LUT pages (and the notebook's scratch)", sym["LUTPAGE"] << 8, sym["ACC"]),
               ("accumulators, logits", sym["ACC"], sym["NBACC"]), ("notebook accumulators", sym["NBACC"], sym["HSTATE"]),
               ("state (h, quantised h)", sym["HSTATE"], sym["XIN"]), ("x input", sym["XIN"], sym["NOTES"]),
               ("notes: 64 x (key 16, gate, length, word 15)", sym["NOTES"], sym["NOTES_END"]),
               ("notebook and reply variables", sym["NOTES_END"], sym["BUFFERS_END"]),
               ("free (above the buffers, below the stack)", sym["BUFFERS_END"], 0x10000)]
    return [{"what": w, "from": a, "to": b, "bytes": b - a} for w, a, b in regions]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--auto", help="your lines, separated by |")
    ap.add_argument("--margin", type=int, choices=[0, 1], default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--minp", type=int, default=13)
    a = ap.parse_args()
    lay = build(load(a.weights), Path(a.out), a.auto.split("|") if a.auto else None, a.seed, a.temp, a.minp, a.margin)
    for r in lay["map"]:
        print(f"{r['from']:5d}-{r['to'] - 1:5d} {r['bytes']:6d}  {r['what']}")
    print(json.dumps({k: v for k, v in lay.items() if k not in ("sym", "map")}))


if __name__ == "__main__":
    main()

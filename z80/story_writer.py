"""The story writer for a 48K Spectrum: a 180-unit ternary GRU with cast slots, from its weights (weights/*.npz).

    python -m z80.story_writer weights/story-writer.npz build/story-writer [--auto "opening"] [--seed 1]
        [--temp 0.7] [--minp 13] [--n 280]

Writes engine.asm, the data blobs it includes, engine.bin (the CODE block from 0x5E00), story.tap (the BASIC loader
plus the CODE block) and layout.json (every address and size) into the folder. Without --auto the program asks you
for the opening of a story; with it, the opening is built in and it starts writing at once (the film's take).

Memory (from 0x5E00): the tables, then the weights (they stream once per character, so the slower contended RAM
costs little), the row parameters, the embedding and the code; then the page-aligned lookup pages and the zeroed
buffers, which are not on the tape; the stack at the top of RAM. reference/gru.py is the same network in Python.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reference.tables import EXP, SIG, TANH
from reference.text import MAXNAME, NSLOTS, SSTOI, SVOCAB, VOCAB
from reference.weights import load
from z80.gates import FINALIZE_BN
from z80.kernel import BIAS, G, P, build_lut_asm, encode_block, groups, matvec_asm, pad_cols, split
from z80.routines import MUL, RNG, SAMPLE
from z80.screen_keys import IO
from z80.cast_slots import VARS as SLOT_VARS
from z80.cast_slots import slots as slot_io
from z80.tape import ORG, assemble, tap


def plan(ex: dict) -> dict:
    """The sizes the code is generated for. Every network here has batch-normalised gates (a scale and a bias per
    row), one plane of input weights and a ternary layout of five weights a byte."""
    assert ex["quant"] == "ternary" and int(ex.get("x_planes", 1)) == 1, "a ternary GRU with one input plane"
    d, e = ex["d"], ex["e"]
    Gh, Ge = groups(d), groups(e)
    for name in ("x", "h", "head"):
        assert ex[f"{name}_sh"].min() >= 8, f"{name}: shift below 8"
    vocab = SVOCAB if ex.get("vocab") == "slots" else VOCAB
    xblocks = split(Ge)
    return {"d": d, "e": e, "g": G, "Gh": Gh, "Ge": Ge, "dpad": Gh * G, "epad": Ge * G,
            "vocab": vocab, "V": len(vocab), "slots": ex.get("vocab") == "slots",
            "hblocks": split(Gh), "m": ex["m"], "tanh": ex.get("tanh", "table"),
            "hp": int(ex.get("head_planes", 1)), "xblocks": xblocks, "xin": len(xblocks) > 1}


def blobs(ex: dict, pl: dict) -> dict:
    """Everything the tape carries besides code, as bytes."""
    d, dpad, epad = pl["d"], pl["dpad"], pl["epad"]
    heads = [pad_cols(ex["head_T"], dpad)] + [pad_cols(ex[f"head{k}_T"], dpad) for k in range(2, pl["hp"] + 1)]
    # rows: the head's planes, then h's r, z, n (the pass over the state); x's r, z, n (the pass over the input)
    h_rows = np.concatenate(heads + [pad_cols(ex["h_T"], dpad)])
    x_rows = pad_cols(ex["x_T"], epad)
    w, offs, g0 = bytearray(), [], 0
    for b in pl["hblocks"]:
        offs.append(len(w))
        w += encode_block(h_rows, g0, b)
        g0 += b
    xoffs, g0 = [], 0
    for b in pl["xblocks"]:
        xoffs.append(len(w))
        w += encode_block(x_rows, g0, b)
        g0 += b
    units = bytearray()
    for i in range(d):
        for k in (i, d + i, 2 * d + i):  # r, z, n: the x part, then the h part
            for part in ("x", "h"):
                units += bytes([int(ex[f"{part}_M"][k]), int(ex[f"{part}_sh"][k]), int(ex[f"{part}_B"][k]) & 0xFF])
    head = bytearray()
    for tag in ["head"] + [f"head{k}" for k in range(2, pl["hp"] + 1)]:
        for v in range(pl["V"]):
            b = int(ex["head_B"][v]) & 0xFFFF if tag == "head" else 0  # the bias rides on the first plane
            head += bytes([int(ex[f"{tag}_M"][v]), int(ex[f"{tag}_sh"][v]), b & 0xFF, b >> 8])
    if pl["xin"]:  # rows unpadded: build_xin copies them into XIN, whose padding stays zero
        emb = ex["emb"].astype(np.int8).tobytes()
    else:
        emb = np.pad(ex["emb"], ((0, 0), (0, epad - pl["e"]))).astype(np.int8).tobytes()
    assert all(int(v) == 0 for v in EXP[100:])
    return {"weights": bytes(w), "hoffs": offs, "xoff": xoffs[0], "xoffs": xoffs, "units": bytes(units),
            "head": bytes(head), "emb": emb, "sig": bytes(int(v) & 0xFF for v in SIG),
            "tanh": bytes(int(v) & 0xFF for v in TANH), "exp": bytes(int(v) for v in EXP[:100])}


def build_xin_asm(pl: dict) -> list[str]:
    """XIN = emb[A]: the input's embedding row, copied where the lookup pages read it."""
    e = pl["e"]
    return ["build_xin:", "    ld hl,EMB", f"    ld de,{e}", "    or a", "    jr z,bx_e", "    ld b,a", "bx_mul:",
            "    add hl,de", "    djnz bx_mul", "bx_e:", "    ld de,XIN", f"    ld bc,{e}", "    ldir", "    ret"]


def buffers(pl: dict) -> list[tuple[str, int]]:
    return ([("ACC", 2 * (pl["V"] * pl["hp"] + 6 * pl["d"])), ("H16", 2 * pl["d"]), ("HQ", pl["dpad"])]
            + ([("XIN", pl["epad"])] if pl["xin"] else []))


def finalize_src(pl: dict) -> list[str]:
    """The gates and the new state (z80/gates.py), its tanh and width filled in."""
    fin = FINALIZE_BN
    if pl["tanh"] == "sig":
        # n = SIG[idx] - 128 and n14 = n * 128: seven doublings, no "+ n"
        fin = fin.replace("    ld h,TANH_PAGE\n", "    ld h,SIG_PAGE\n").replace("TANH_FIX\n", "    sub 128\n")
        fin = fin.replace("    ld d,h\n    ld e,l\n    add hl,hl\n", "    add hl,hl\n", 1)
        fin = fin.replace("    add hl,hl\n    add hl,de               ; n14 = n * 129",
                          "    add hl,hl               ; n14 = n * 128")
    else:
        fin = fin.replace("TANH_FIX\n", "")
    for k in (16, 14, 12, 10, 8, 6, 4, 2):
        fin = fin.replace(f"{k}D", str(k * pl["d"]))
    return fin.splitlines()


def tables_and_data(pl: dict, auto: list[int] | None, seed: int) -> list[str]:
    """The page-aligned tables (0x5E00 is page-aligned); the exp table holds only its 100 non-zero entries, and the
    small initialised data fills the rest of its page."""
    L = ["SIG_PAGE equ $ >> 8", '    incbin "sig.bin"']
    if pl["tanh"] != "sig":
        L += ["TANH_PAGE equ $ >> 8", '    incbin "tanh.bin"']
    L += ["EXP_PAGE equ $ >> 8", '    incbin "exp.bin"']
    vocab_print = [13 if c == "\n" else ord(c) for c in pl["vocab"]]
    L += ["vocab_chars: db " + ",".join(map(str, vocab_print))]
    if auto is not None:
        L += ["PROMPT_TEXT: db " + ",".join(map(str, auto + [0xFF]))]
    L += ["msg_title: db 'a neural network writes a story',13,13,0xFF", "msg_prompt: db 'start the story"
          + (" (names with a capital)" if pl["slots"] else "") + ": ',0xFF",
          "msg_end: db 13,13,'(the end, seed ',0xFF", "msg_end2: db ')',13,0xFF", f"seed_used: dw {seed}"]
    return L


def program(pl: dict, bl: dict, auto: list[int] | None, temp: float, minp: int, nmax: int, seed: int,
            prefill: tuple = ()) -> str:
    """The whole program as sjasmplus source."""
    d, V_, m, Gh, Ge = pl["d"], pl["V"], pl["m"], pl["Gh"], pl["Ge"]
    tinv = 0 if temp == 0 else round(64 / temp)
    init_h = (-BIAS * Gh) & 0xFFFF
    init_x = (-BIAS * Ge) & 0xFFFF
    nrows_h = V_ * pl["hp"] + 3 * d
    VH = V_ * pl["hp"]
    # the bottom-right square's colour (0x5AFF) shows what a built-in run is doing: 0x10 writing, 0x20 finished
    mark = "    ld (0x5AFF),a" if auto is not None else "    nop"
    L = [f"V equ {V_}", "NTYPE equ 35", f"NUNITS equ {d}", f"TINV equ {tinv}", f"MINP equ {minp}", f"MBOUND equ {m}",
         f"    org {ORG}"] + tables_and_data(pl, auto, seed) + [
         "WEIGHTS: incbin \"weights.bin\"", "PUNITS: incbin \"units.bin\"", "PHEAD: incbin \"head.bin\"",
         "EMB: incbin \"emb.bin\"",
         "start:", "    di", "    ld sp,0", "    ld iy,0x5C3A", "    call zero_buffers", "    ld a,2", "    call 0x1601",
         "    call 0x0DAF", "    ld hl,msg_title", "    call print_str",
         "new_story:", "    call reset_state"]
    if auto is None:
        L += ["    ld hl,msg_prompt", "    call print_str", "    call read_prompt"]
    else:
        L += ["    ld hl,PROMPT_TEXT", "    call print_syms"]
    if pl["slots"]:  # typed or built in: capitalised words become cast slots as they are fed
        L += ["SYM_STAR equ 37", "SYM_ONE equ 38", f"NSLOTS equ {NSLOTS}", f"MAXNAME equ {MAXNAME}"]
        L += ["    xor a", "    call feed", "    call convert_feed", "    jr fp_done"]
    # without cast slots: '#', then the opening symbol by symbol (a slot build jumps over this)
    L += ["    xor a", "    call feed", "    ld hl,PROMPT", "fp_loop:", "    ld a,(hl)", "    cp 0xFF", "    jr z,fp_done",
          "    push hl", "    call feed", "    pop hl", "    inc hl", "    jr fp_loop",
          "fp_done:", f"    ld hl,{nmax}", "    ld (count),hl",
          "gen_loop:", "    ld a,0x10", mark,
          "    call hpass", "gl_hpass:", "    call head_logits", "gl_head:",
          *(["    call slot_mask"] if pl["slots"] else []),
          "    call sample", "gl_sample:", "    or a",
          "    jr z,story_end", "    push af", "    call print_sym", "    pop af", "gl_print:", "    call xpass_x",
          "gl_xpass:", "    call finalize", "gl_final:",
          "    ld hl,(count)", "    dec hl", "    ld (count),hl", "    ld a,h", "    or l", "    jr nz,gen_loop",
          "story_end:", "    ld a,0x20", mark,
          "after_story:", "    ld hl,msg_end", "    call print_str", "    ld hl,(seed_used)", "    ld b,h", "    ld c,l",
          "    call 0x2D2B", "    call 0x2DE3", "    ld hl,msg_end2", "    call print_str"]
    L += (["done:", "    jr done"] if auto is not None else
          ["    call wait_key", "    call 0x0DAF", "    jp new_story"])
    L += ["feed:", "    push af", "    call hpass", "    pop af", "    jp xpass_finalize"]
    # the pass over the state: the head, then the h part of r, z and n, block by block
    L += ["hpass:", "    ld hl,ACC", f"    ld bc,{VH + 3 * d}", f"    ld de,{init_h}", "    call fill16"]
    g0 = 0
    for k, b in enumerate(pl["hblocks"]):
        page = f"LUTPAGE+{P - b}"
        L += [f"    ld ix,HQ+{g0 * pl['g']}", f"    ld h,{page}", f"    ld a,{b}", "    call bl_group",
              f"    ld hl,WEIGHTS+{bl['hoffs'][k]}", "    ld ix,ACC", f"    ld bc,{nrows_h}", f"    ld de,mv_g{P - b}",
              f"    ld a,{page}", "    call matvec"]
        g0 += b
    L += ["    ret"]
    # the pass over the input symbol A, then the gates and the new state
    L += ["xpass_x:"]
    if pl["xin"]:
        L += ["    call build_xin", "    ld ix,XIN"]
    else:
        L += ["    ld hl,EMB", f"    ld de,{pl['epad']}", "    or a", "    jr z,xf_row", "    ld b,a",
              "xf_mul:", "    add hl,de", "    djnz xf_mul", "xf_row:", "    push hl", "    pop ix"]
    if len(pl["xblocks"]) > 1:  # wider than the lookup window: block by block, as the pass over the state
        L += ["    ld hl,ACC_XFILL", f"    ld bc,{3 * d}", f"    ld de,{init_x}", "    call fill16"]
        g0 = 0
        for k, b in enumerate(pl["xblocks"]):
            page = f"LUTPAGE+{P - b}"
            L += [f"    ld ix,XIN+{g0 * pl['g']}", f"    ld h,{page}", f"    ld a,{b}", "    call bl_group",
                  f"    ld hl,WEIGHTS+{bl['xoffs'][k]}", "    ld ix,ACC_XROWS", f"    ld bc,{3 * d}",
                  f"    ld de,mv_g{P - b}", f"    ld a,{page}", "    call matvec"]
            g0 += b
        L += ["    ret"]
    else:
        L += [f"    ld h,LUTPAGE+{P - Ge}", f"    ld a,{Ge}", "    call bl_group",
              "    ld hl,ACC_XFILL", f"    ld bc,{3 * d}", f"    ld de,{init_x}", "    call fill16",
              f"    ld hl,WEIGHTS+{bl['xoff']}", "    ld ix,ACC_XROWS", f"    ld bc,{3 * d}",
              f"    ld de,mv_g{P - Ge}", f"    ld a,LUTPAGE+{P - Ge}", "    jp matvec"]
    L += ["xpass_finalize:", "    call xpass_x", "    jp finalize"]
    L += finalize_src(pl)
    L += build_lut_asm()
    L += matvec_asm()
    L += [MUL, RNG.replace("seed: dw 1", f"seed: dw {seed}"), SAMPLE]
    if pl["xin"]:
        L += build_xin_asm(pl)
    L += (slot_io(IO) if pl["slots"] else IO).splitlines()
    L += [f"VH equ {VH}", "CODE_END:", "TAPE_END:"]
    # buffers (zeroed at start, not on the tape): those in `prefill` sit in the gap before the page-aligned pages
    bufs = dict(buffers(pl))
    L += ["BUFFERS:"] + [f"{nm}: ds {bufs[nm]}" for nm in prefill]
    L += ["    align 256", "LUTPAGE equ $ >> 8", f"    ds {P * 256}"]
    L += [f"{nm}: ds {sz}" for nm, sz in bufs.items() if nm not in prefill]
    L += ["BUFFERS_END:", "RAM_END:", "    assert $ <= 0xFF80",
          f"ACC_XROWS equ ACC + {2 * (VH + 3 * d)}", f"ACC_XFILL equ ACC + {2 * (VH + 3 * d)}",
          f"ACC_R equ ACC + {2 * (VH + d)}", "LOGITS equ ACC"]
    # the printer buffer (0x5B00-0x5BFF, unused without a ZX Printer) holds the opening and the scratch variables
    pb = 0x5B00
    for name, size in (("PROMPT", 130), ("WEIGHTS8", V_), ("count", 2), ("fi_acc", 2), ("fi_h", 2), ("fi_hq", 2),
                       ("fi_n", 2), ("fi_r", 1), ("fi_z", 1), ("fi_xn", 2), ("fi_n14", 2), ("fi_t", 2), ("smax", 2),
                       ("sarg", 1), ("stotal", 2), ("sprod", 3)) + (SLOT_VARS if pl["slots"] else ()):
        L += [f"{name} equ {pb}"]
        pb += size
    assert pb <= 0x5C00
    return "\n".join(L) + "\n"


def build(ex: dict, out: Path, auto: str | None = None, seed: int = 1, temp: float = 0.7, minp: int = 13,
          n: int = 280) -> dict:
    """Write the tape and its sources into out; returns the layout (addresses and sizes)."""
    pl = plan(ex)
    bl = blobs(ex, pl)
    out.mkdir(parents=True, exist_ok=True)
    for k in ("weights", "units", "head", "emb", "sig", "tanh", "exp"):
        (out / f"{k}.bin").write_bytes(bl[k])
    syms = None
    if auto is not None:  # a slot build holds the opening as typed (bit 7 = capital), as read_prompt stores it
        syms = ([SSTOI[c.lower()] | (0x80 if c.isupper() else 0) for c in auto] if pl["slots"]
                else [VOCAB.index(c) for c in auto.lower()])

    def assemble_with(prefill):
        (out / "engine.asm").write_text(program(pl, bl, syms, temp, minp, n, seed, prefill), encoding="ascii")
        return assemble(out, "engine")

    sym = assemble_with(())
    gap = (256 - (sym["TAPE_END"] & 255)) & 255
    prefill = []
    for nm, sz in sorted(buffers(pl), key=lambda b: -b[1]):  # the largest buffers that still fit the gap
        if sz <= gap:
            prefill.append(nm)
            gap -= sz
    if prefill:
        sym = assemble_with(tuple(prefill))
    code = (out / "engine.raw").read_bytes()[: sym["TAPE_END"] - ORG]
    (out / "engine.bin").write_bytes(code)
    (out / "story.tap").write_bytes(tap(sym["start"], code, b"story", b"gru"))
    lay = {"org": ORG, "tape_bytes": len(code), "tape_end": sym["TAPE_END"], "start": sym["start"],
           "ram_end": sym["RAM_END"], "free_above": 0x10000 - sym["RAM_END"],
           "padding": sym["LUTPAGE"] * 256 - sym["TAPE_END"] - sum(dict(buffers(pl))[nm] for nm in prefill),
           "buffers": [sym["BUFFERS"], sym["BUFFERS_END"]], "lutpage": sym["LUTPAGE"] << 8,
           "weights": len(bl["weights"]), "units": len(bl["units"]), "head": len(bl["head"]), "emb": len(bl["emb"]),
           "code": sym["CODE_END"] - sym["start"], "auto": auto, "seed": seed, "temp": temp, "minp": minp, "n": n,
           "sym": sym}
    (out / "layout.json").write_text(json.dumps(lay, indent=1))
    return lay


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--auto", help="an opening to build in (capitals for names)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--minp", type=int, default=13)
    ap.add_argument("--n", type=int, default=280, help="symbols to write at most")
    a = ap.parse_args()
    lay = build(load(a.weights), Path(a.out), a.auto, a.seed, a.temp, a.minp, a.n)
    print(json.dumps({k: v for k, v in lay.items() if k != "sym"}, indent=1))


if __name__ == "__main__":
    main()

"""ZX-LM for a 48K Spectrum: who owns what, one word at a time, from its weights (weights/zx-lm.npz).

    python -m z80.zx_lm weights/zx-lm.npz build/zx-lm [--auto "line|line|..."] [--seed 1]

Writes zxlm.asm, the data blobs it includes, zxlm.bin (the CODE block from 0x5E00), zxlm.tap (the BASIC loader plus
the CODE block) and layout.json (every address and size) into the folder. Without --auto you type: ENTER sends a
line, an empty line starts a new conversation. With --auto the lines are built in and typed for you.

Each word is one step. pre_step: the pass over the state (the lookup-table kernel) gives the GRU's h rows and the
108 projection rows; the projections are scaled; the previous word's deferred work is done (its mention's key part,
the turn's memory rewrite); attention over the memory. post_step: the read, the word's embedding, filing a mention,
the pass over the input and the GRU's gates (gates.py's FINALIZE_BN). Writing, choose() picks each word between the
network's own words and pointing at a remembered one. reference/zx_lm.py is the same network in Python.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reference.tables import EXP, SIG, TANH
from reference.weights import load
from reference.zx_lm import Vocab
from z80.gates import FINALIZE_BN
from z80.kernel import BIAS, P, build_lut_asm, encode, groups, matvec_asm, pad_cols, split
from z80.routines import MUL, RNG
from z80.tape import ORG, assemble, tap
from z80.zx_lm_memory import MEM, WRITE
from z80.zx_lm_text import TEXT

MAXLINE = 120  # characters a line holds (LINE is 128 bytes)


def blobs(ex: dict, v: Vocab) -> dict:
    d = int(ex["d"])
    gh, gx = groups(d), groups(int(ex["x_T"].shape[1]))
    h_rows = np.concatenate([pad_cols(ex["p_T"], 5 * gh), pad_cols(ex["h_T"], 5 * gh)])
    hw, hoffs = encode(h_rows, split(gh))
    xw, xoffs = encode(pad_cols(ex["x_T"], 5 * gx), split(gx))
    units = bytearray()
    for i in range(d):
        for k in (i, d + i, 2 * d + i):
            for part in ("x", "h"):
                units += bytes([int(ex[f"{part}_M"][k]), int(ex[f"{part}_sh"][k]), int(ex[f"{part}_B"][k]) & 0xFF])
    proj = bytearray()
    for i in range(len(ex["p_M"])):
        b = int(ex["p_B"][i]) & 0xFFFF
        proj += bytes([int(ex["p_M"][i]), int(ex["p_sh"][i]), b & 0xFF, b >> 8])
    vstr = bytearray()
    for w in v.words:
        vstr += bytes([len(w)]) + w.encode("ascii")
    vstr += b"\xff"
    assert all(int(x) == 0 for x in EXP[100:])
    return {"weights": hw + xw, "hoffs": hoffs, "xoffs": [len(hw) + o for o in xoffs], "units": bytes(units),
            "proj": bytes(proj), "emb": ex["emb"].astype(np.int8).tobytes(), "eref": ex["eref"].astype(np.int8).tobytes(),
            "nullk": ex["null_k"].astype(np.int8).tobytes(), "vocab": bytes(vstr),
            "outmap": bytes(v.index[w] for w in v.out_words),
            "sig": bytes(int(x) & 0xFF for x in SIG), "tanh": bytes(int(x) & 0xFF for x in TANH),
            "exp": bytes(int(x) for x in EXP[:100]), "gh": gh, "gx": gx}


def finalize_src(d: int) -> list[str]:
    fin = FINALIZE_BN.replace("TANH_FIX\n", "")
    for k in (16, 14, 12, 10, 8, 6, 4, 2):
        fin = fin.replace(f"{k}D", str(k * d))
    return fin.splitlines()


def pass_asm(name: str, src: str, acc: str, rows: int, gs: int, offs: list[int]) -> list[str]:
    """A matrix pass: fill the accumulators with the LUT bias, then each block: build its pages, multiply."""
    L = [f"{name}:", f"    ld hl,{acc}", f"    ld bc,{rows}", f"    ld de,{(-BIAS * gs) & 0xFFFF}", "    call fill16"]
    g0 = 0
    for b, off in zip(split(gs), offs):
        page = f"LUTPAGE+{P - b}"
        L += [f"    ld ix,{src}+{g0 * 5}", f"    ld h,{page}", f"    ld a,{b}", "    call bl_group",
              f"    ld hl,WEIGHTS+{off}", f"    ld ix,{acc}", f"    ld bc,{rows}", f"    ld de,mv_g{P - b}",
              f"    ld a,{page}", "    call matvec"]
        g0 += b
    return L + ["    ret"]


STEPS = r"""
; ---- one word: pre_step, then (writing: choose) post_step -----------------------------------------------------------
pre_step:
    call hpass
    call proj_scale
    call unpack_p
    ld a,(pend_mention)
    or a
    call nz,store_pre
    ld a,(pend_end)
    or a
    jr z,ps_att
    xor a
    ld (pend_end),a
    call mem_write
ps_att:
    jp attend

post_step:                  ; A = token, (tok_end) = 1 if it ends its turn
    ld (cur_tok),a
    call read_vec
    ld a,(cur_tok)          ; XIN[0..31] = its embedding
    ld l,a
    ld h,0
    add hl,hl
    add hl,hl
    add hl,hl
    add hl,hl
    add hl,hl
    ld de,EMB
    add hl,de
    ld de,XIN
    ld bc,32
    ldir
    ld a,(cur_tok)
    cp NWORDS
    jr c,pst_x
    sub NWORDS
    call file_mention
pst_x:
    call xpass
    call finalize
    ld a,(tok_end)
    ld (pend_end),a
    ret

feed:                       ; A = token, (tok_end) set
    push af
    call pre_step
    pop af
    jp post_step

proj_scale:                 ; PV = ((ACC * M) >> sh) + B for the 108 projection rows
    ld hl,ACC
    ld (ps_src),hl
    ld hl,PV
    ld (ps_dst),hl
    ld ix,PPROJ
    ld b,NPROJ
ps_loop:
    push bc
    ld hl,(ps_src)
    ld e,(hl)
    inc hl
    ld d,(hl)
    inc hl
    ld (ps_src),hl
    call scaled16
    ex de,hl
    ld hl,(ps_dst)
    ld (hl),e
    inc hl
    ld (hl),d
    inc hl
    ld (ps_dst),hl
    ld bc,4
    add ix,bc
    pop bc
    djnz ps_loop
    ret
ps_src: dw 0
ps_dst: dw 0

fill16:                     ; BC words of DE at HL
    ld (hl),e
    inc hl
    ld (hl),d
    inc hl
    dec bc
    ld a,b
    or c
    jr nz,fill16
    ret

; ---- a conversation --------------------------------------------------------------------------------------------
reset_conv:
    ld hl,H16               ; the state and its quantised copy (HQ follows H16)
    ld de,H16 + 1
    ld bc,H16_LEN - 1
    ld (hl),0
    ldir
    ld hl,CONV
    ld de,CONV + 1
    ld bc,CONV_END - CONV - 1
    ld (hl),0
    ldir
    ret

conversation:
    call reset_conv
turn:
    ld a,0x18               ; (auto) the bottom-right attribute marks reading your line
    call mark
    call get_line
    jp z,no_line
    call tokenize
    xor a
    ld (tk_i),a
    ld a,(NTOK)
    or a
    jr nz,tu_open
    inc a
    ld (tok_end),a
    ld a,TOK_U
    call feed
    jr spectrum
tu_open:
    xor a
    ld (tok_end),a
    ld a,TOK_U
    call feed
tu_loop:
    ld a,(tk_i)
    ld e,a
    ld d,0
    ld hl,TOKS
    add hl,de
    inc a
    ld (tk_i),a
    ld b,a
    ld a,(NTOK)
    cp b
    ld a,0
    jr nz,tu_end
    inc a
tu_end:
    ld (tok_end),a
    ld a,(hl)
    call feed
    ld a,(tok_end)
    or a
    jr z,tu_loop
spectrum:
    xor a
    ld (tok_end),a
    ld a,TOK_S
    call feed
    ld a,TOK_THINK
    call feed
    ld a,TOK_THINK
    call feed
    ld a,1
    ld (pt_first),a
    xor a
    ld (n_out),a
    ld a,0x10               ; (auto) ... writing the reply
    call mark
sp_word:
    call pre_step
    call choose
    cp NOUT
    jr nc,sp_ref
    ld e,a
    ld d,0
    ld hl,OUTMAP
    add hl,de
    ld a,(hl)
    jr sp_tok
sp_ref:
    sub NOUT
    add a,NWORDS
sp_tok:
    ld (sp_cur),a
    call print_tok
    ld hl,n_out
    inc (hl)
    ld a,(sp_cur)
    cp TOK_DOT
    ld a,1
    jr z,sp_last
    ld a,(n_out)
    cp MAXREPLY
    ld a,1
    jr nc,sp_last
    xor a
sp_last:
    ld (tok_end),a
    ld a,(sp_cur)
    call post_step
    ld a,(tok_end)
    or a
    jr z,sp_word
    ld a,13
    call print_ch
    ld a,13
    call print_ch
    jp turn
mark:                       ; built-in lines only: A to the bottom-right square's colour (what it is doing)
    ld e,a
    ld a,(auto_on)
    or a
    ret z
    ld a,e
    ld (0x5AFF),a
    ret
sp_cur: db 0
tk_i: db 0
n_out: db 0
"""


def program(ex: dict, bl: dict, v: Vocab, auto: list[str] | None, seed: int) -> str:
    d = int(ex["d"])
    nproj = len(ex["p_M"])
    rows_h = nproj + 3 * d
    L = [f"NUNITS equ {d}", f"VH equ {nproj}", f"NPROJ equ {nproj}", f"MBOUND equ {int(ex['m'])}",
         f"NOUT equ {v.n_out}", f"NWORDS equ {v.n_words}", f"TOK_U equ {v.index['<u>']}",
         f"TOK_S equ {v.index['<s>']}", f"TOK_THINK equ {v.index['<think>']}", f"TOK_DOT equ {v.index['.']}",
         "MAXREPLY equ 14", "V equ 1", "PHEAD equ PPROJ", f"H16_LEN equ {2 * d + 5 * groups(d)}", f"    org {ORG}",
         "SIG_PAGE equ $ >> 8", '    incbin "sig.bin"', "TANH_PAGE equ $ >> 8", '    incbin "tanh.bin"',
         "EXP_PAGE equ $ >> 8", '    incbin "exp.bin"',
         f"NULLB: dw {int(ex['null_b']) & 0xFFFF}", 'NULLK: incbin "nullk.bin"', 'OUTMAP: incbin "outmap.bin"',
         "msg_title: db 'zx-lm: tell me who owns what,',13,'then ask me. an empty line',13,'starts again.',13,13,0xFF",
         f"auto_on: db {1 if auto is not None else 0}", "auto_ptr: dw AUTO_TEXT", "seeded: db 0"]
    assert all(len(line) <= MAXLINE for line in (auto or [])), "a built-in line longer than LINE holds"
    text = [str(c) for line in (auto or []) for c in list(line.encode("ascii")) + [0]] + ["0xFF"]
    L += ["AUTO_TEXT:"] + ["    db " + ",".join(text[i:i + 32]) for i in range(0, len(text), 32)]
    L += ['WEIGHTS: incbin "weights.bin"', 'PUNITS: incbin "units.bin"', 'PPROJ: incbin "proj.bin"',
          'EMB: incbin "emb.bin"', 'EREF: incbin "eref.bin"', 'VOCAB_STR: incbin "vocab.bin"',
          "start:", "    di", "    ld sp,0", "    ld iy,0x5C3A", "    call zero_buffers", "    ld a,2",
          "    call 0x1601", "    call 0x0DAF", "    ld hl,msg_title", "    call print_str",
          f"    ld hl,{seed}", "    ld (seed),hl",
          "main:", "    jp conversation", "no_line:", "    ld a,(auto_on)", "    or a", "    jr nz,done",
          "    ld a,13", "    call print_ch", "    jp main",
          "done:", "    ld a,0x20", "    ld (0x5AFF),a", "done_loop:", "    jr done_loop",
          "zero_buffers:", "    ld hl,BUFFERS", "    ld de,BUFFERS + 1", "    ld bc,BUFFERS_END - BUFFERS - 1",
          "    ld (hl),0", "    ldir", "    ret"]
    L += STEPS.splitlines()
    L += pass_asm("hpass", "HQ", "ACC", rows_h, bl["gh"], bl["hoffs"])
    L += pass_asm("xpass", "XIN", "ACC_X", 3 * d, bl["gx"], bl["xoffs"])
    L += finalize_src(d)
    L += build_lut_asm()
    L += matvec_asm()
    L += [MUL, RNG.replace("seed: dw 1", f"seed: dw {seed}"), MEM, WRITE, TEXT]
    L += ["CODE_END:", "TAPE_END:"]
    bufs = [("ACC", 2 * (rows_h + 3 * d)), ("H16", 2 * d), ("HQ", 5 * bl["gh"]), ("XIN", 5 * bl["gx"]),
            ("PV", 2 * nproj), ("Q8", 16), ("Z8", 16), ("C8", 16), ("REC8", 1), ("SCORE", 130), ("WBUF", 65),
            ("ALPHA", 130), ("RACC", 64), ("REFP", 32), ("LINE", 128), ("TOKS", 64), ("WORD", 16), ("NTOK", 1),
            ("cur_tok", 1), ("tok_end", 1)]
    conv = [("KEYS", 1024), ("PRE", 2048), ("SREF", 64), ("BIRTH", 128), ("FLAGS", 64), ("REFS", 256),
            ("TURN", 2), ("NSL", 2), ("pend_mention", 1), ("pend_slot", 1), ("pend_end", 1)]
    L += ["BUFFERS:", "    align 256", "LUTPAGE equ $ >> 8", f"    ds {P * 256}"]
    L += [f"{n}: ds {s}" for n, s in bufs]
    L += ["CONV:"] + [f"{n}: ds {s}" for n, s in conv] + ["CONV_END:", "    ds 2"]
    L += ["BUFFERS_END:", "RAM_END:", "    assert $ <= 0xFF00", f"ACC_X equ ACC + {2 * rows_h}"]
    pb = 0x5B00  # the printer buffer: J's gate scratch
    for name, size in (("fi_acc", 2), ("fi_h", 2), ("fi_hq", 2), ("fi_n", 2), ("fi_r", 1), ("fi_z", 1),
                       ("fi_xn", 2), ("fi_n14", 2), ("fi_t", 2)):
        L += [f"{name} equ {pb}"]
        pb += size
    return "\n".join(L) + "\n"


def build(ex: dict, out: Path, auto: list[str] | None = None, seed: int = 1) -> dict:
    """Write the tape and its sources into out; returns the layout (addresses and sizes)."""
    v = Vocab(ex)
    bl = blobs(ex, v)
    out.mkdir(parents=True, exist_ok=True)
    for k in ("weights", "units", "proj", "emb", "eref", "nullk", "vocab", "outmap", "sig", "tanh", "exp"):
        (out / f"{k}.bin").write_bytes(bl[k])
    (out / "zxlm.asm").write_text(program(ex, bl, v, auto, seed), encoding="ascii")
    sym = assemble(out, "zxlm")
    code = (out / "zxlm.raw").read_bytes()[: sym["TAPE_END"] - ORG]
    (out / "zxlm.bin").write_bytes(code)
    (out / "zxlm.tap").write_bytes(tap(sym["start"], code, b"zx-lm", b"zxlm"))
    lay = {"org": ORG, "tape_bytes": len(code), "start": sym["start"], "ram_end": sym["RAM_END"],
           "free_above": 0x10000 - sym["RAM_END"], "weights": len(bl["weights"]), "code": sym["CODE_END"] - sym["start"],
           "buffers": sym["BUFFERS_END"] - sym["BUFFERS"], "auto": auto, "seed": seed, "sym": sym}
    (out / "layout.json").write_text(json.dumps(lay, indent=1))
    return lay


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("weights")
    ap.add_argument("out")
    ap.add_argument("--auto", help="your lines, separated by |")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    lay = build(load(a.weights), Path(a.out), a.auto.split("|") if a.auto else None, a.seed)
    print(json.dumps({k: v for k, v in lay.items() if k != "sym"}, indent=1))


if __name__ == "__main__":
    main()

"""The ternary lookup-table kernel: a matrix of weights -1, 0 and +1 times a vector of small integers, with no
multiplications, as Z80 source (sjasmplus).

1. Lookup tables. The inputs are taken five at a time. For each group of five, a 256-byte page of RAM gets every
   possible sum of those inputs times a pattern of five weights: 3^5 = 243 sums. Each sum is stored as one byte with
   a bias of 128, so it must stay within -128..127, which is why every input is held to |x| <= 25.
2. The product. Each weight byte is the number of its group's pattern (code = sum of d_i 3^i, digit 0, 1, 2 for weight
   0, +1, -1), so a row costs one lookup and one 8-bit add (with a carry into the high byte) per five weights. The
   weight bytes stream through the stack pointer, two at a time with POP, with interrupts off (an interrupt would push
   its return address over the weights).

The pages cost RAM, so the columns go in blocks of up to eight groups (eight pages): build the block's pages, add that
block's lookups into every row's 16-bit accumulator, then the next block. The weights are stored block by block (every
row's bytes for block 0, then block 1, ...), so the stack pointer still just streams.
"""
from __future__ import annotations

import numpy as np

BIAS = 128
G = 5  # weights per byte
P = 8  # lookup pages: the most groups a block holds
DIGIT = {0: 0, 1: 1, -1: 2}


def groups(n: int) -> int:
    """Groups of five for n inputs, rounded up to an even number (POP reads two weight bytes at a time)."""
    k = -(-n // G)
    return k + (k & 1)


def split(n: int) -> list[int]:
    """n groups as blocks of at most P."""
    out = []
    while n:
        out.append(min(P, n))
        n -= out[-1]
    return out


def pad_cols(t: np.ndarray, n: int) -> np.ndarray:
    return np.pad(t, ((0, 0), (0, n - t.shape[1])))


def encode_block(rows: np.ndarray, g0: int, ng: int) -> bytes:
    """The weight bytes for groups [g0, g0 + ng) of every row (rows already padded to whole groups)."""
    out = bytearray()
    for row in rows:
        for j in range(g0, g0 + ng):
            out.append(sum(DIGIT[int(v)] * 3 ** i for i, v in enumerate(row[j * G:(j + 1) * G])))
    return bytes(out)


def encode(rows: np.ndarray, blocks: list[int]) -> tuple[bytes, list[int]]:
    """Every block in turn: (the bytes, where each block starts)."""
    out, offs, g0 = bytearray(), [], 0
    for b in blocks:
        offs.append(len(out))
        out += encode_block(rows, g0, b)
        g0 += b
    return bytes(out), offs


def build_lut_asm() -> list[str]:
    """bl_group: fill A pages from page H, five inputs from IX each. Out: IX advanced past them."""
    out = ["bl_group:", "    ex af,af'", "    ld d,h"]
    # T[0] = BIAS; for digit i (stride s = 3^i): T[s + c] = T[c] + x_i, then T[2s + c] = T[c] - x_i, for c < s
    out += ["    ld l,0", f"    ld (hl),{BIAS}"]
    for i in range(G):
        s = 3 ** i
        out += [f"    ld c,(ix+{i})"]
        for sign, base in (("p", s), ("m", 2 * s)):
            if sign == "m":
                out += ["    ld a,c", "    neg", "    ld c,a"]
            if s < 3:
                out += ["    ld l,0", "    ld a,(hl)", "    add a,c", f"    ld l,{base}", "    ld (hl),a"]
                continue
            lbl = f"bl_0_{i}_{sign}"
            out += ["    ld l,0", f"    ld e,{base}", f"    ld b,{s // 3}", f"{lbl}:"]
            out += ["    ld a,(hl)", "    add a,c", "    ld (de),a", "    inc l", "    inc e"] * 3
            out += [f"    djnz {lbl}"]
    out += [f"    ld bc,{G}", "    add ix,bc", "    inc h"]
    out += ["    ex af,af'", "    dec a", "    jp nz,bl_group", "bl_done:", "    ret"]
    return out


def matvec_asm() -> list[str]:
    """matvec: HL = weights, IX = accumulators, BC = rows, DE = entry (which group pair to start at), A = first LUT
    page. Each row adds one lookup per group into its 16-bit accumulator."""
    out = ["matvec:", "    ld (mv_jp+1),de", "    ld (mv_hp+1),a", "    ld (mv_w),hl", "    ld a,c", "    or a",
           "    jr z,mv_full", "    inc b", "mv_full:", "    ld a,b", "    ld (mv_outer),a", "    ld b,c",
           "    ld (mv_sp+1),sp", "    ld sp,(mv_w)",
           "mv_row:", "    ld a,(ix+0)", "    ld c,(ix+1)", "mv_hp:", "    ld h,0", "mv_jp:", "    jp 0"]
    for j in range(P):
        out += [f"mv_g{j}:"]
        out += ["    pop de", "    ld l,e"] if j % 2 == 0 else ["    ld l,d"]
        out += ["    add a,(hl)", "    jr nc,$+3", "    inc c", "    inc h"]
    out += ["    ld (ix+0),a", "    ld (ix+1),c", "    inc ix", "    inc ix", "    dec b", "    jp nz,mv_row",
            "    ld a,(mv_outer)", "    dec a", "    ld (mv_outer),a", "    jp nz,mv_row",
            "mv_sp:", "    ld sp,0", "    ret", "mv_w: dw 0", "mv_outer: db 0"]
    return out

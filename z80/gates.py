"""Z80 source: the GRU's gates and new state, unit by unit (plain and batch-normalised layouts), for every engine here."""

FINALIZE = r"""
; ---- gates and the new state, unit by unit ------------------------------------------------------------------
finalize:
    ld hl,ACC + 2 * VH
    ld (fi_acc),hl          ; this unit's U_n accumulator; r, z, x_n are 2D, 4D, 6D bytes further
    ld hl,H16
    ld (fi_h),hl
    ld hl,HQ
    ld (fi_hq),hl
    ld ix,PUNITS
    ld hl,NUNITS
    ld (fi_n),hl
fi_loop:
    ld hl,(fi_acc)
    ld de,2D
    add hl,de
    ld e,(hl)
    inc hl
    ld d,(hl)
    call scaled8
    call clamp8
    xor 0x80
    ld l,a
    ld h,SIG_PAGE
    ld a,(hl)
    ld (fi_r),a
    ld bc,3
    add ix,bc
    ld hl,(fi_acc)
    ld de,4D
    add hl,de
    ld e,(hl)
    inc hl
    ld d,(hl)
    call scaled8
    call clamp8
    xor 0x80
    ld l,a
    ld h,SIG_PAGE
    ld a,(hl)
    ld (fi_z),a
    ld bc,3
    add ix,bc
    ld hl,(fi_acc)
    ld de,6D
    add hl,de
    ld e,(hl)
    inc hl
    ld d,(hl)
    call scaled8
    ld (fi_xn),hl
    ld bc,3
    add ix,bc
    ld hl,(fi_acc)
    ld e,(hl)
    inc hl
    ld d,(hl)
    call scaled8            ; HL = hn
    ld bc,3
    add ix,bc
    ex de,hl
    ld a,(fi_r)
    call mul_sh8            ; HL = (hn * r) >> 8
    ld de,(fi_xn)
    add hl,de
    call clamp8
    xor 0x80
    ld l,a
    ld h,TANH_PAGE
    ld a,(hl)               ; n
TANH_FIX
    ld l,a
    rla
    sbc a,a
    ld h,a
    ld d,h
    ld e,l
    add hl,hl
    add hl,hl
    add hl,hl
    add hl,hl
    add hl,hl
    add hl,hl
    add hl,hl
    add hl,de               ; n14 = n * 129
    ld (fi_n14),hl
    ex de,hl
    ld hl,(fi_h)
    ld a,(hl)
    inc hl
    ld h,(hl)
    ld l,a                  ; HL = h
    or a
    sbc hl,de               ; h - n14
    ex de,hl
    ld a,(fi_z)
    call mul_sh8            ; HL = (z * (h - n14)) >> 8
    ld de,(fi_n14)
    add hl,de               ; h'
    ex de,hl
    ld hl,(fi_h)
    ld (hl),e
    inc hl
    ld (hl),d
    inc hl
    ld (fi_h),hl
    ld a,MBOUND
    call mul16x8            ; C:HL = h' * m
    ld de,8192
    add hl,de
    jr nc,fi_nc
    inc c
fi_nc:
    add hl,hl
    rl c
    add hl,hl
    rl c
    ld a,c                  ; (h' * m + 8192) >> 14
    bit 7,a
    jr nz,fi_neg
    cp MBOUND + 1
    jr c,fi_q
    ld a,MBOUND
    jr fi_q
fi_neg:
    cp 256 - MBOUND
    jr nc,fi_q
    ld a,256 - MBOUND
fi_q:
    ld hl,(fi_hq)
    ld (hl),a
    inc hl
    ld (fi_hq),hl
    ld hl,(fi_acc)
    inc hl
    inc hl
    ld (fi_acc),hl
    ld hl,(fi_n)
    dec hl
    ld (fi_n),hl
    ld a,h
    or l
    jp nz,fi_loop
    ret

; ---- the output head: logits (1/16 nat) in place of the head accumulators -----------------------------------
head_logits:
    ld ix,PHEAD
    ld hl,ACC
    ld b,V
    ld a,VH / V
    ld (hl_planes),a
hl_loop:
    push bc
    push hl
    ld e,(hl)
    inc hl
    ld d,(hl)
    call scaled16
    ld a,(hl_planes)
    dec a
    jr z,hl_one
    push ix
    ld (hl_sum),hl
    pop hl
    ld de,4 * V
    add hl,de
    push hl
    pop ix                  ; plane 2's parameters
    pop hl
    push hl
    ld de,2 * V
    add hl,de               ; plane 2's accumulator
    ld e,(hl)
    inc hl
    ld d,(hl)
    call scaled16           ; (its bias is 0)
    ld de,(hl_sum)
    add hl,de
    ld de,-4 * V
    add ix,de
hl_one:
    ex de,hl
    pop hl
    ld (hl),e
    inc hl
    ld (hl),d
    inc hl
    ld bc,4
    add ix,bc
    pop bc
    djnz hl_loop
    ret
hl_planes: db 0
hl_sum: dw 0
"""

TAIL_MARK = "    ld de,(fi_xn)\n    add hl,de\n    call clamp8"
_TAIL = FINALIZE[FINALIZE.index(TAIL_MARK):]
_SC = """    ld hl,(fi_acc)
    ld de,{off}
    add hl,de
    ld e,(hl)
    inc hl
    ld d,(hl)
    call scaled8
    ld bc,3
    add ix,bc
"""
_GATE = """    ld de,(fi_t)
    add hl,de
    call clamp8
    xor 0x80
    ld l,a
    ld h,SIG_PAGE
    ld a,(hl)
    ld ({dst}),a
"""
_X2 = """    ld (fi_t),hl
""" + _SC.replace("{off}", "{off}") + """    ld de,(fi_t)
    add hl,de
"""
# a two-plane x matrix: the second plane's rows follow the first's (12D, 14D, 16D), scaled without a bias
FINALIZE_BN2 = (FINALIZE[: FINALIZE.index("fi_loop:")] + "fi_loop:\n"
                + _SC.format(off="6D") + _X2.format(off="12D") + "    ld (fi_t),hl\n" + _SC.format(off="0")
                + _GATE.format(dst="fi_r")
                + _SC.format(off="8D") + _X2.format(off="14D") + "    ld (fi_t),hl\n" + _SC.format(off="2D")
                + _GATE.format(dst="fi_z")
                + _SC.format(off="10D") + _X2.format(off="16D") + "    ld (fi_xn),hl\n" + _SC.format(off="4D")
                + "    ex de,hl\n    ld a,(fi_r)\n    call mul_sh8\n" + _TAIL)
FINALIZE_BN = (FINALIZE[: FINALIZE.index("fi_loop:")] + "fi_loop:\n"
               + _SC.format(off="6D") + "    ld (fi_t),hl\n" + _SC.format(off="0") + _GATE.format(dst="fi_r")
               + _SC.format(off="8D") + "    ld (fi_t),hl\n" + _SC.format(off="2D") + _GATE.format(dst="fi_z")
               + _SC.format(off="10D") + "    ld (fi_xn),hl\n" + _SC.format(off="4D")
               + "    ex de,hl\n    ld a,(fi_r)\n    call mul_sh8\n" + _TAIL)

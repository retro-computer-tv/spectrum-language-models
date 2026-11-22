"""Hand-written Z80 routines of the GRU engine (sjasmplus syntax), shared by every model size.

Conventions: IY stays 0x5C3A (the ROM's system variables) whenever the ROM is called or interrupts are on.
Signed values are two's complement. `>>` in comments is arithmetic (floor), as in reference/gru.py.
"""

MUL = r"""
; ---- C:HL = DE * A   (DE signed 16-bit, A unsigned 8-bit; signed 24-bit result) ----------------------------
mul16x8:
    ld (mul_a+1),a
    ld hl,0
    ld c,l
    ld b,8
mul_loop:
    add hl,hl
    rl c
    add a,a
    jr nc,mul_next
    add hl,de
    jr nc,mul_next
    inc c
mul_next:
    djnz mul_loop
    bit 7,d                 ; DE negative: signed(DE) = DE - 65536, so the product loses A * 65536
    ret z
    ld a,c
mul_a:
    sub 0
    ld c,a
    ret

; ---- HL = (DE * A) >> 8   (signed 16 x unsigned 8, floor) ------------------------------------------------
mul_sh8:
    call mul16x8
    ld l,h
    ld h,c
    ret

; ---- HL = ((DE * M) >> sh) + B, with M, sh, B read from (IX+0..2); B is a signed byte -----------------------
scaled8:
    ld a,(ix+0)
    call mul16x8
    ld l,h                  ; >> 8
    ld h,c
    ld a,(ix+1)
    sub 8
    jr z,sc_bias
    ld b,a
sc_shift:
    sra h
    rr l
    djnz sc_shift
sc_bias:
    ld a,(ix+2)
    ld e,a
    rla
    sbc a,a
    ld d,a
    add hl,de
    ret

; ---- the same with a 16-bit bias at (IX+2..3) (the output head) ---------------------------------------------
scaled16:
    ld a,(ix+0)
    call mul16x8
    ld l,h
    ld h,c
    ld a,(ix+1)
    sub 8
    jr z,sc16_bias
    ld b,a
sc16_shift:
    sra h
    rr l
    djnz sc16_shift
sc16_bias:
    ld e,(ix+2)
    ld d,(ix+3)
    add hl,de
    ret

; ---- A = HL clamped to -128..127 ---------------------------------------------------------------------------
clamp8:
    ld a,l
    rla
    ld a,h
    adc a,0
    jr nz,cl_over
    ld a,l
    ret
cl_over:
    bit 7,h
    ld a,0x7F
    ret z
    ld a,0x80
    ret
"""

RNG = r"""
; ---- xorshift16 (7, 9, 8), John Metcalf's Z80 form; HL = new state ----------------------------------------
xrnd:
    ld hl,(seed)
    ld a,h
    rra
    ld a,l
    rra
    xor h
    ld h,a
    ld a,l
    rra
    ld a,h
    rra
    xor l
    ld l,a
    xor h
    ld h,a
    ld (seed),hl
    ret
seed: dw 1
"""

SAMPLE = r"""
; ---- A = the next symbol, from the logits at LOGITS (V int16) ------------------------------------------------
; temp: Tinv (0 = greedy). w = EXP[((max - logit) * Tinv) >> 6] (0 past the table, 0 below minp).
sample:
    ld hl,LOGITS            ; max and its index
    ld e,(hl)
    inc hl
    ld d,(hl)
    inc hl
    ld (smax),de
    xor a
    ld (sarg),a
    ld b,V - 1
    ld c,1
smax_loop:
    push bc
    ld e,(hl)
    inc hl
    ld d,(hl)
    inc hl
    push hl
    ld hl,(smax)
    or a
    sbc hl,de               ; max - logit (signed compare)
    jp pe,smax_ovf
    jp p,smax_keep
    jr smax_new
smax_ovf:
    jp m,smax_keep
smax_new:
    ld (smax),de
    ld a,c
    ld (sarg),a
smax_keep:
    pop hl
    pop bc
    inc c
    djnz smax_loop
    ld a,(tinv)
    or a
    ld a,(sarg)
    ret z                   ; greedy
    ; weights
    ld hl,0
    ld (stotal),hl
    ld ix,LOGITS
    ld iy,WEIGHTS8
    ld b,V
sw_loop:
    push bc
    ld hl,(smax)
    ld e,(ix+0)
    ld d,(ix+1)
    or a
    sbc hl,de               ; delta >= 0
    ex de,hl
    ld a,(tinv)
    call mul16x8u           ; C:HL = delta * Tinv (both unsigned)
    ld a,c
    or a
    jr nz,sw_zero
    ld a,h
    cp 0x40                 ; (C:HL) >> 6 >= 256 ?
    jr nc,sw_zero
    add hl,hl               ; e = HL >> 6 = (HL << 2) >> 8
    add hl,hl
    ld a,h
    cp 100                  ; the table holds only the non-zero entries
    jr nc,sw_zero
    ld l,a
    ld h,EXP_PAGE
    ld a,(hl)
    ld hl,minp
    cp (hl)
    jr nc,sw_store
sw_zero:
    xor a
sw_store:
    ld (iy+0),a
    ld e,a
    ld d,0
    ld hl,(stotal)
    add hl,de
    ld (stotal),hl
    inc ix
    inc ix
    inc iy
    pop bc
    djnz sw_loop
    ld iy,0x5C3A
    ; r = (rand * total) >> 16
    call xrnd
    ex de,hl                ; DE = rand
    ld a,(stotal)
    call mul16x8u           ; C:HL = rand * total_lo
    ld (sprod),hl
    ld a,c
    ld (sprod+2),a
    ld a,(stotal+1)
    call mul16x8u           ; C:HL = rand * total_hi, to add << 8
    ld a,(sprod+1)
    add a,l
    ld l,a
    ld a,(sprod+2)
    adc a,h
    ld h,a
    ld a,0
    adc a,c
    ld c,a                  ; C:H is now the high 16 bits (bits 16..31), L is bits 8..15
    ld l,h
    ld h,c                  ; HL = r
    ; first symbol whose running total exceeds r
    ld de,WEIGHTS8
    ld bc,0                 ; running total
    xor a
    ld (sarg),a
spick:
    ld a,(de)
    add a,c
    ld c,a
    jr nc,spick_nc
    inc b
spick_nc:
    push hl
    or a
    sbc hl,bc               ; r - total < 0 ?
    pop hl
    jr c,spick_done
    inc de
    ld a,(sarg)
    inc a
    ld (sarg),a
    jr spick
spick_done:
    ld a,(sarg)
    ret

; unsigned C:HL = DE * A
mul16x8u:
    ld hl,0
    ld c,l
    ld b,8
mulu_loop:
    add hl,hl
    rl c
    add a,a
    jr nc,mulu_next
    add hl,de
    jr nc,mulu_next
    inc c
mulu_next:
    djnz mulu_loop
    ret

tinv: db TINV
minp: db MINP
"""

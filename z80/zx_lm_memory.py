"""Z80 source for ZX-LM's memory (reference/zx_lm.py is the reference): attention, the read, rewriting keys at a
turn's end, filing mentions, and choosing the answer word. sjasmplus syntax; `>>` is arithmetic (floor).

Data (zx_lm.py lays it out): KEYS 64 x 16 int8 (127ths); PRE 64 x 16 int16; SREF 64 (reference of each
slot); BIRTH 64 x 2 (turn filed); FLAGS 64 (bit 0 valid, bit 1 pending); SCORE 65 x 2; ALPHA 65 x 2 (256ths; entry 0
is "nothing"); PV 108 x 2 (the projections, scaled); Q8, Z8, C8 16 each; REC8; TURN 2; NSL 2.
"""

MEM = r"""
; ---- HL = A * C, both signed bytes ----------------------------------------------------------------------------
mulss:
    ld b,a
    xor c
    ld (ms_sign),a
    ld a,b
    or a
    jp p,ms_a
    neg
ms_a:
    ld b,a                  ; |A| (0x80 reads as 128)
    ld a,c
    or a
    jp p,ms_c
    neg
ms_c:
    ld e,a
    ld d,0
    ld hl,0
ms_loop:
    srl b
    jr nc,ms_skip
    add hl,de
ms_skip:
    sla e
    rl d
    ld a,b
    or a
    jr nz,ms_loop
    ld a,(ms_sign)
    or a
    ret p
    xor a                   ; HL = -HL
    sub l
    ld l,a
    sbc a,a
    sub h
    ld h,a
    ret
ms_sign: db 0

; ---- HL = sum over 16 of (IX+i) * (Q+i), signed bytes; DE' unused. In: IX = the key, (dot_q) = the other vector
dot16:
    ld hl,0
    ld (dot_acc),hl
    ld iy,(dot_q)
    ld b,16
dot_loop:
    push bc
    ld a,(ix+0)
    ld c,(iy+0)
    call mulss
    ld de,(dot_acc)
    add hl,de
    ld (dot_acc),hl
    inc ix
    inc iy
    pop bc
    djnz dot_loop
    ld iy,0x5C3A
    ld hl,(dot_acc)
    ret
dot_acc: dw 0
dot_q: dw 0

; ---- HL >>= 1 (arithmetic) --------------------------------------------------------------------------------
sra_hl:
    sra h
    rr l
    ret

; ---- the projections in their units: Q8, Z8 (clamped to 16), REC8, from PV ------------------------------------
P_Q equ 0
P_REC equ 16
P_KAT equ 17
P_KEND equ 33
P_C equ 49
P_Z equ 65
P_GEN equ 81
P_HEAD equ 82

clamp16:                    ; A = clamp(HL, -16, 16)
    bit 7,h
    jr nz,c16_neg
    ld a,h
    or a
    jr nz,c16_hi
    ld a,l
    cp 17
    ret c
c16_hi:
    ld a,16
    ret
c16_neg:
    ld a,h
    inc a
    jr nz,c16_lo
    ld a,l
    cp 256 - 16
    ret nc
c16_lo:
    ld a,256 - 16
    ret

unpack_p:
    ld hl,PV + 2 * P_Q
    ld de,Q8
    call up16
    ld hl,PV + 2 * P_Z
    ld de,Z8
    call up16
    ld hl,(PV + 2 * P_REC)
    call clamp8
    ld (REC8),a
    ret
up16:                       ; 16 int16 at HL -> clamped to +-16 at DE
    ld b,16
up_loop:
    push bc
    ld a,(hl)
    inc hl
    push hl
    ld h,(hl)
    ld l,a
    call clamp16
    ld (de),a
    inc de
    pop hl
    inc hl
    pop bc
    djnz up_loop
    ret

; ---- attention: ALPHA (256ths) for "nothing" and every valid slot ------------------------------------------------
attend:
    ld hl,Q8
    ld (dot_q),hl
    ld ix,NULLK
    call dot16
    call sra16_7            ; >> 7
    ld de,(NULLB)
    add hl,de
    ld (SCORE),hl
    ld (at_max),hl
    ld hl,0
    ld (at_m),hl            ; slot index, 16-bit for the address sums
at_slot:
    ld hl,(at_m)
    ld de,FLAGS
    add hl,de
    bit 0,(hl)
    jr z,at_next
    ld hl,(at_m)            ; IX = KEYS + 16 m
    add hl,hl
    add hl,hl
    add hl,hl
    add hl,hl
    ld de,KEYS
    add hl,de
    push hl
    pop ix
    call dot16
    call sra16_7
    push hl
    ld hl,(at_m)            ; age = min(TURN - BIRTH[m], 15)
    add hl,hl
    ld de,BIRTH
    add hl,de
    ld e,(hl)
    inc hl
    ld d,(hl)
    ld hl,(TURN)
    or a
    sbc hl,de
    ld a,h
    or a
    jr nz,at_age15
    ld a,l
    cp 16
    jr c,at_age
at_age15:
    ld a,15
at_age:
    ld c,a
    ld a,(REC8)
    call mulss              ; rec * age
    call sra_hl
    call sra_hl             ; >> 2
    pop de
    add hl,de
    ex de,hl                ; DE = score
    ld hl,(at_m)
    inc hl
    add hl,hl
    ld bc,SCORE
    add hl,bc
    ld (hl),e
    inc hl
    ld (hl),d
    ld hl,(at_max)          ; max = DE if DE > max (signed)
    ld a,h
    xor d
    jp m,at_sgn
    sbc hl,de               ; carry clear after xor
    jr nc,at_next
    ld (at_max),de
    jr at_next
at_sgn:
    bit 7,d
    jr nz,at_next
    ld (at_max),de
at_next:
    ld hl,(at_m)
    inc hl
    ld (at_m),hl
    ld a,l
    cp 64
    jp nz,at_slot
    ; weights: w = EXP[max - s] for s within 100, else 0; W = their sum
    ld hl,0
    ld (at_w),hl
    ld hl,0
    ld (at_m),hl
at_w_loop:
    ld a,(at_m)
    or a
    jr z,at_w_on            ; entry 0 ("nothing") always counts
    ld e,a
    ld d,0
    ld hl,FLAGS - 1
    add hl,de
    bit 0,(hl)
    jr z,at_w_zero
at_w_on:
    ld hl,(at_m)
    add hl,hl
    ld de,SCORE
    add hl,de
    ld e,(hl)
    inc hl
    ld d,(hl)
    ld hl,(at_max)
    or a
    sbc hl,de               ; e = max - s >= 0
    ld a,h
    or a
    jr nz,at_w_zero
    ld a,l
    cp 100
    jr nc,at_w_zero
    ld h,EXP_PAGE
    ld a,(hl)
    jr at_w_store
at_w_zero:
    xor a
at_w_store:
    ld hl,(at_m)
    ld de,WBUF
    add hl,de
    ld (hl),a
    ld e,a
    ld d,0
    ld hl,(at_w)
    add hl,de
    ld (at_w),hl
    ld hl,(at_m)
    inc hl
    ld (at_m),hl
    ld a,l
    cp 65
    jr nz,at_w_loop
    ; alpha = (w << 8) // W
    ld hl,0
    ld (at_m),hl
at_a_loop:
    ld hl,(at_m)
    ld de,WBUF
    add hl,de
    ld h,(hl)
    ld l,0
    ld a,h
    or a
    jr z,at_a_store         ; 0 / W = 0
    ld de,(at_w)
    call div16u             ; HL = HL // DE
at_a_store:
    ex de,hl
    ld hl,(at_m)
    add hl,hl
    ld bc,ALPHA
    add hl,bc
    ld (hl),e
    inc hl
    ld (hl),d
    ld hl,(at_m)
    inc hl
    ld (at_m),hl
    ld a,l
    cp 65
    jr nz,at_a_loop
    ret
at_m: dw 0
at_max: dw 0
at_w: dw 0

sra16_7:                    ; HL = HL >> 7 (|HL| <= 32512, so the result fits 9 bits)
    ld a,l
    rla                     ; carry = bit 7 of L
    ld a,h
    rla                     ; A = H << 1 | that bit; carry = the sign
    ld l,a
    sbc a,a
    ld h,a
    ret

; ---- HL = HL // DE, unsigned 16-bit -------------------------------------------------------------------------
div16u:
    ld a,h
    ld c,l
    ld hl,0
    ld b,16
dv_loop:
    sla c
    rla
    adc hl,hl
    sbc hl,de
    jr nc,dv_fit
    add hl,de
    dec c                   ; undo the quotient bit set below
dv_fit:
    inc c
    djnz dv_loop
    ld h,a
    ld l,c
    ret

; ---- the read: XIN[32..47] = ((sum alpha k >> 8) * 51 + 128) >> 8, XIN[48..63] = sum alpha e[ref] >> 8 --------
read_vec:
    ld hl,RACC
    ld de,RACC + 1
    ld bc,63
    ld (hl),0
    ldir                    ; 32 int16 accumulators
    ld hl,0
    ld (at_m),hl
rv_slot:
    ld hl,(at_m)
    ld de,FLAGS
    add hl,de
    bit 0,(hl)
    jp z,rv_next
    ld hl,(at_m)
    inc hl
    add hl,hl
    ld de,ALPHA
    add hl,de
    ld a,(hl)
    inc hl
    ld h,(hl)
    ld l,a
    or h
    jp z,rv_next
    ld (rv_alpha),hl
    ld hl,(at_m)            ; the key
    add hl,hl
    add hl,hl
    add hl,hl
    add hl,hl
    ld de,KEYS
    add hl,de
    ld (rv_src),hl
    ld hl,RACC
    ld (rv_dst),hl
    call rv_add16
    ld hl,(at_m)            ; the reference's embedding row
    ld de,SREF
    add hl,de
    ld l,(hl)
    ld h,0
    add hl,hl
    add hl,hl
    add hl,hl
    add hl,hl
    ld de,EREF
    add hl,de
    ld (rv_src),hl
    ld hl,RACC + 32
    ld (rv_dst),hl
    call rv_add16
rv_next:
    ld hl,(at_m)
    inc hl
    ld (at_m),hl
    ld a,l
    cp 64
    jp nz,rv_slot
    ; finish: key part to 25ths, reference part >> 8
    ld hl,XIN + 32
    ld (rv_out),hl
    ld hl,RACC
    ld b,16
rv_fin_k:
    push bc
    ld a,(hl)
    inc hl
    push hl
    ld h,(hl)               ; r127 = acc >> 8: the high byte
    ld e,h
    ld a,h
    rla
    sbc a,a
    ld d,a
    ld a,51
    push de
    call mul16x8            ; C:HL = r127 * 51
    pop de
    ld de,128
    add hl,de
    jr nc,rv_nc
    inc c
rv_nc:
    ld a,h                  ; (... + 128) >> 8: the low byte of C:H
    pop hl
    inc hl
    pop bc
    ex de,hl
    ld hl,(rv_out)
    ld (hl),a
    inc hl
    ld (rv_out),hl
    ex de,hl
    djnz rv_fin_k
    ld hl,RACC + 32
    ld b,16
rv_fin_e:
    inc hl
    ld a,(hl)               ; >> 8: the high byte
    inc hl
    ex de,hl
    ld hl,(rv_out)
    ld (hl),a
    inc hl
    ld (rv_out),hl
    ex de,hl
    djnz rv_fin_e
    ret
rv_alpha: dw 0
rv_src: dw 0
rv_dst: dw 0
rv_out: dw 0

rv_add16:                   ; 16 int16 at (rv_dst) += alpha * the signed bytes at (rv_src)
    ld b,16
ra_loop:
    push bc
    ld hl,(rv_src)
    ld a,(hl)
    inc hl
    ld (rv_src),hl
    ld e,a
    rla
    sbc a,a
    ld d,a                  ; DE = the byte, sign-extended
    ld hl,(rv_alpha)
    ld a,h
    or a
    jr z,ra_mul
    ld h,e                  ; alpha = 256: DE << 8
    ld l,0
    jr ra_add
ra_mul:
    ld a,l
    call mul16x8            ; HL = low 16 bits of DE * alpha
ra_add:
    ex de,hl
    ld hl,(rv_dst)
    ld a,(hl)
    add a,e
    ld (hl),a
    inc hl
    ld a,(hl)
    adc a,d
    ld (hl),a
    inc hl
    ld (rv_dst),hl
    pop bc
    djnz ra_loop
    ret
"""

WRITE = r"""
; ---- a turn's end: rewrite every valid key, then file the turn's mentions -------------------------------------------
mem_write:
    ld hl,PV + 2 * P_C      ; C8 = TANH[clamp8(upd_c)]
    ld de,C8
    ld b,16
mw_c:
    push bc
    ld a,(hl)
    inc hl
    push hl
    ld h,(hl)
    ld l,a
    call clamp8
    xor 0x80
    ld l,a
    ld h,TANH_PAGE
    ld a,(hl)
    ld (de),a
    inc de
    pop hl
    inc hl
    pop bc
    djnz mw_c
    ld hl,Z8
    ld (dot_q),hl
    ld hl,0
    ld (at_m),hl
mw_slot:
    ld hl,(at_m)
    ld de,FLAGS
    add hl,de
    bit 0,(hl)
    jp z,mw_next
    ld hl,(at_m)
    add hl,hl
    add hl,hl
    add hl,hl
    add hl,hl
    ld de,KEYS
    add hl,de
    ld (mw_key),hl
    push hl
    pop ix
    call dot16              ; k . z
    ld b,7
mw_sh:
    sra h
    rr l
    djnz mw_sh
    call clamp8
    xor 0x80
    ld l,a
    ld h,SIG_PAGE
    ld a,(hl)
    ld (mw_g),a
    ld hl,(mw_key)
    ld de,C8
    ld b,16
mw_dim:
    push bc
    push de
    push hl
    ld a,(de)               ; c
    ld e,a
    rla
    sbc a,a
    ld d,a
    ld a,(hl)               ; k
    ld (mw_k),a
    ld l,a
    rla
    sbc a,a
    ld h,a
    add hl,de               ; k + c
    call sra_hl
    call sra_hl             ; >> 2
    call clamp8
    xor 0x80
    ld l,a
    ld h,TANH_PAGE
    ld a,(hl)               ; t
    ld l,a
    rla
    sbc a,a
    ld h,a
    ld a,(mw_k)
    ld e,a
    rla
    sbc a,a
    ld d,a
    or a
    sbc hl,de               ; t - k
    ex de,hl
    ld a,(mw_g)
    call mul16x8            ; C:HL = g (t - k)
    ld l,h
    ld h,c                  ; >> 8
    ld a,(mw_k)
    add a,l                 ; k + ... (the result fits a byte)
    pop hl
    ld (hl),a
    inc hl
    pop de
    inc de
    pop bc
    djnz mw_dim
mw_next:
    ld hl,(at_m)
    inc hl
    ld (at_m),hl
    ld a,l
    cp 64
    jp nz,mw_slot
    ; the pending mentions: key = TANH[clamp8(pre + key_end)], valid, born this turn
    ld hl,0
    ld (at_m),hl
mw_new:
    ld hl,(at_m)
    ld de,FLAGS
    add hl,de
    bit 1,(hl)
    jp z,mw_new_next
    ld (hl),1               ; valid, no longer pending
    ld hl,(at_m)
    add hl,hl
    ld de,BIRTH
    add hl,de
    ld de,(TURN)
    ld (hl),e
    inc hl
    ld (hl),d
    ld hl,(at_m)            ; PRE + 32 m, KEYS + 16 m
    add hl,hl
    add hl,hl
    add hl,hl
    add hl,hl
    push hl
    ld de,KEYS
    add hl,de
    ld (mw_key),hl
    pop hl
    add hl,hl
    ld de,PRE
    add hl,de
    ld de,PV + 2 * P_KEND
    ld b,16
mw_nd:
    push bc
    ld c,(hl)
    inc hl
    ld b,(hl)
    inc hl
    push hl
    ex de,hl
    ld e,(hl)
    inc hl
    ld d,(hl)
    inc hl
    push hl
    ex de,hl
    add hl,bc               ; pre + key_end
    call clamp8
    xor 0x80
    ld l,a
    ld h,TANH_PAGE
    ld a,(hl)
    ld hl,(mw_key)
    ld (hl),a
    inc hl
    ld (mw_key),hl
    pop de
    pop hl
    pop bc
    djnz mw_nd
mw_new_next:
    ld hl,(at_m)
    inc hl
    ld (at_m),hl
    ld a,l
    cp 64
    jp nz,mw_new
    ld hl,(TURN)
    inc hl
    ld (TURN),hl
    ret
mw_key: dw 0
mw_g: db 0
mw_k: db 0

; ---- file a mention: SREF, pending (not valid); the key part waits for the next step's projections ------------------
; In: A = reference number
file_mention:
    push af
    ld a,(NSL)
    and 63
    ld (pend_slot),a
    ld e,a
    ld d,0
    ld hl,FLAGS
    add hl,de
    ld (hl),2               ; pending (a reused slot stops being valid)
    ld hl,SREF
    add hl,de
    pop af
    ld (hl),a
    ld hl,(NSL)
    inc hl
    ld (NSL),hl
    ld a,1
    ld (pend_mention),a
    ret

store_pre:                  ; PRE[pend_slot] = the key_at rows of PV
    ld a,(pend_slot)
    ld l,a
    ld h,0
    add hl,hl
    add hl,hl
    add hl,hl
    add hl,hl
    add hl,hl
    ld de,PRE
    add hl,de
    ex de,hl
    ld hl,PV + 2 * P_KAT
    ld bc,32
    ldir
    xor a
    ld (pend_mention),a
    ret

; ---- the answer word: A = output index (0..25 its own words, 26 + r a reference) -----------------------------------
choose:
    ld a,(PV + 2 * P_GEN + 1)
    bit 7,a
    jr nz,ch_ref
    ld ix,PV + 2 * P_HEAD   ; the first largest head logit
    ld hl,(PV + 2 * P_HEAD)
    ld (ch_best),hl
    xor a
    ld (ch_arg),a
    ld b,1
ch_w:
    inc ix
    inc ix
    ld e,(ix+0)
    ld d,(ix+1)
    ld hl,(ch_best)         ; DE > best (signed)?
    ld a,h
    xor d
    jp m,ch_w_sgn
    or a
    sbc hl,de
    jr nc,ch_w_next
    jr ch_w_new
ch_w_sgn:
    bit 7,d
    jr nz,ch_w_next
ch_w_new:
    ld (ch_best),de
    ld a,b
    ld (ch_arg),a
ch_w_next:
    inc b
    ld a,b
    cp NOUT
    jr nz,ch_w
    ld a,(ch_arg)
    ret
ch_ref:
    ld hl,REFP
    ld de,REFP + 1
    ld bc,31
    ld (hl),0
    ldir
    ld b,0                  ; slot
ch_r:
    ld e,b
    ld d,0
    ld hl,FLAGS
    add hl,de
    bit 0,(hl)
    jr z,ch_r_next
    ld hl,SREF
    add hl,de
    ld a,(hl)
    push af
    ld l,b                  ; ALPHA[slot + 1]
    ld h,0
    inc hl
    add hl,hl
    ld de,ALPHA
    add hl,de
    ld e,(hl)
    inc hl
    ld d,(hl)
    pop af
    ld l,a
    ld h,0
    add hl,hl
    push bc
    ld bc,REFP
    add hl,bc
    pop bc
    ld a,(hl)
    add a,e
    ld (hl),a
    inc hl
    ld a,(hl)
    adc a,d
    ld (hl),a
ch_r_next:
    inc b
    ld a,b
    cp 64
    jr nz,ch_r
    ld hl,(REFP)            ; the first largest (unsigned)
    ld (ch_best),hl
    xor a
    ld (ch_arg),a
    ld ix,REFP
    ld b,1
ch_rm:
    inc ix
    inc ix
    ld e,(ix+0)
    ld d,(ix+1)
    ld hl,(ch_best)
    or a
    sbc hl,de
    jr nc,ch_rm_next
    ld (ch_best),de
    ld a,b
    ld (ch_arg),a
ch_rm_next:
    inc b
    ld a,b
    cp 16
    jr nz,ch_rm
    ld a,(ch_arg)
    add a,NOUT
    ret
ch_best: dw 0
ch_arg: db 0
"""

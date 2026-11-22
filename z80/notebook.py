"""Z80 source for the conversation network's notebook (reference/notebook.py, the integer reference), for j.py.

A note is 33 bytes: its key (16 int8 in -m..m), its gate's log-sigmoid (int8, 1/16 nat), the length of its word
(1..15) and the word's symbols, held in the note itself so a note is exact however long ago it was written. 64 notes
in a ring, the oldest replaced.

nb_push (every symbol, after the h pass and before the x pass, so the key rows still hold the state before it) keeps
the reference's bookkeeping: a typed word opens a note, its letters fill it, anything else closes it; a name ('*')
notes its slot digit and is born closed; the reply's current word (`partial`, its last 15 symbols) for matching.
nb_bonus (after the head, before sampling): every closed note whose word continues the reply's current word scores
((Q . key) * A >> sh) + lsig; w = EXP[best - score]; each adds (SOFTPLUS[trust] * w) / sum(w) to the logit of the
symbol it proposes. Its scratch (the query and the fits) is the LUT pages, free between the h pass and the x pass.

Rows of the notebook pass (NBACC, int16, from the h pass): query 0..15, key 16..31, gate 32, trust 33; their
parameters (PNB) M, sh, B per row.
"""
from __future__ import annotations

NB_ROWS = 34
NOTE = 33
NNOTES = 64
MAXW = 15  # up to 15 symbols of a word
VARS = (("nb_count", 1), ("nb_open", 1), ("nb_prevw", 1), ("nb_stars", 1), ("nb_typing", 1), ("nb_plen", 1),
        ("nb_tw", 1), ("nb_wrp", 2), ("nb_last", 2), ("nb_nfit", 1), ("nb_fitp", 2), ("nb_best", 2),
        ("nb_total", 2), ("nb_beta", 1), ("nbb_ptr", 2), ("nbb_left", 1), ("nbb_sym", 1), ("PARTIAL", MAXW))

NOTEBOOK = r"""
; ---- the notebook (reference/notebook.py) ----------------------------------------------------------------------
nb_reset:                   ; a new conversation: no notes, you are typing
    xor a
    ld (nb_count),a
    ld (nb_open),a
    ld (nb_prevw),a
    ld (nb_stars),a
    ld (nb_plen),a
    inc a
    ld (nb_typing),a
    ld hl,NOTES
    ld (nb_wrp),hl
    ret

nb_push:                    ; A = the symbol fed or written
    or a
    jr z,nb_reset
    ld c,a
    sub 11
    cp 30
    ld b,0
    jr nc,np_nw
    inc b                   ; B = 1: a word symbol (letters, '*', slot digits)
np_nw:
    ld a,c
    cp 37
    jr nz,np_ns
    ld a,(nb_stars)
    cp 5                    ; at most symbol 42 (the reference has no symbol past it)
    jr nc,np_ns
    inc a
    ld (nb_stars),a
np_ns:
    ld a,(nb_typing)
    and b
    ld (nb_tw),a            ; a typed word symbol
    jr nz,np_tw
    ld (nb_open),a          ; anything else closes the open note
    jr np_after
np_tw:
    ld a,(nb_prevw)
    or a
    jr nz,np_cont
    push bc
    call nb_new
    pop bc
    jr np_after
np_cont:
    ld a,(nb_open)
    or a
    jr z,np_after
    ld hl,(nb_last)
    ld de,17
    add hl,de
    ld a,(hl)               ; the word's length so far
    cp MAXW
    jr nc,np_after
    inc (hl)
    inc hl
    ld e,a
    ld d,0
    add hl,de
    ld (hl),c
np_after:
    ld a,(nb_tw)
    ld (nb_prevw),a
    ld a,b
    or a
    jr nz,np_word
    ld (nb_plen),a          ; not a word symbol: the reply's current word is empty
    jr np_turn
np_word:
    ld a,(nb_plen)
    cp MAXW
    jr c,np_add
    push bc
    ld hl,PARTIAL + 1       ; full: keep the last 15
    ld de,PARTIAL
    ld bc,MAXW - 1
    ldir
    pop bc
    ld a,MAXW - 1
np_add:
    ld e,a
    ld d,0
    ld hl,PARTIAL
    add hl,de
    ld (hl),c
    inc a
    ld (nb_plen),a
np_turn:
    ld a,c
    cp YOU_END
    jr nz,np_t2
    xor a
    ld (nb_typing),a
    ret
np_t2:
    cp SPEC_END
    ret nz
    ld a,1
    ld (nb_typing),a
    ret

nb_new:                     ; C = the word's first symbol: a new note in the ring
    ld de,(nb_wrp)
    ld (nb_last),de
    ld hl,NBACC + 32        ; the key rows: the state before this symbol
    ld ix,PNB + 48
    ld b,16
nn_key:
    push bc
    push de
    ld e,(hl)
    inc hl
    ld d,(hl)
    inc hl
    push hl
    call scaled_round
    pop hl
    pop de
    ld (de),a
    inc de
    ld bc,3
    add ix,bc
    pop bc
    djnz nn_key
    push de
    push bc
    ld e,(hl)               ; the gate row
    inc hl
    ld d,(hl)
    call scaled8
    call clamp8
    xor 0x80
    ld l,a
    ld h,LOGSIG_PAGE
    ld a,(hl)
    pop bc
    pop de
    ld (de),a               ; its log-sigmoid
    inc de
    ld a,1
    ld (de),a               ; one symbol so far
    inc de
    ld a,c
    cp 37
    jr nz,nn_word
    ld a,(nb_stars)         ; a name: its slot digit, and the note is born closed
    add a,37
    ld (de),a
    xor a
    jr nn_open
nn_word:
    ld (de),a
    ld a,1
nn_open:
    ld (nb_open),a
    ld hl,(nb_wrp)
    ld de,NOTE
    add hl,de
    ld de,NOTES_END
    or a
    sbc hl,de
    add hl,de
    jr c,nn_wr
    ld hl,NOTES
nn_wr:
    ld (nb_wrp),hl
    ld a,(nb_count)
    cp NNOTES
    ret nc
    inc a
    ld (nb_count),a
    ret

scaled_round:               ; A = clamp((DE * M + 2^(sh-1)) >> sh, -m, m), M and sh at (IX+0), (IX+1); sh >= 9
    ld a,(ix+0)
    call mul16x8            ; C:HL = DE * M
    ld b,(ix+1)
    dec b
    ld de,1
sr_half:
    sla e
    rl d
    djnz sr_half
    add hl,de
    jr nc,sr_nc
    inc c
sr_nc:
    ld l,h
    ld h,c                  ; >> 8
    ld a,(ix+1)
    sub 8
    jr z,sr_clamp
    ld b,a
sr_sh:
    sra h
    rr l
    djnz sr_sh
sr_clamp:
    bit 7,h
    jr nz,sr_neg
    ld a,h
    or a
    jr nz,sr_pmax
    ld a,l
    cp MBOUND + 1
    ret c
sr_pmax:
    ld a,MBOUND
    ret
sr_neg:
    ld a,h
    inc a
    jr nz,sr_nmax
    ld a,l
    cp 256 - MBOUND
    ret nc
sr_nmax:
    ld a,256 - MBOUND
    ret

nb_query:                   ; QV = the query, 16 int8 from rows 0..15
    ld hl,NBACC
    ld de,QV
    ld ix,PNB
    ld b,16
nq_loop:
    push bc
    push de
    ld e,(hl)
    inc hl
    ld d,(hl)
    inc hl
    push hl
    call scaled_round
    pop hl
    pop de
    ld (de),a
    inc de
    ld bc,3
    add ix,bc
    pop bc
    djnz nq_loop
    ret

smul8:                      ; HL = A * E, both int8 with |.| <= 25: SQ4[|a+b|] - SQ4[|a-b|]
    ld d,a
    add a,e
    call sq4
    push hl
    ld a,d
    sub e
    call sq4
    ex de,hl
    pop hl
    or a
    sbc hl,de
    ret
sq4:                        ; HL = floor(A^2 / 4) for |A| <= 50 (DE kept)
    or a
    jp p,sq_pos
    neg
sq_pos:
    add a,a
    ld l,a
    ld h,0
    push de
    ld de,SQ4
    add hl,de
    pop de
    ld a,(hl)
    inc hl
    ld h,(hl)
    ld l,a
    ret

nb_score:                   ; HL = ((Q . key) * A >> sh) + lsig, for the note at (nbb_ptr)
    ld a,(nb_nfit)
    or a
    call z,nb_query         ; the query, at the step's first fit
    ld hl,(nbb_ptr)
    ld de,QV
    ld bc,0
    ld a,16
ns_loop:
    ex af,af'
    push hl
    push de
    push bc
    ld a,(de)
    ld e,(hl)
    call smul8
    pop bc
    add hl,bc
    ld b,h
    ld c,l
    pop de
    pop hl
    inc hl
    inc de
    ex af,af'
    dec a
    jr nz,ns_loop
    ld d,b
    ld e,c
    ld a,ATT_M
    call mul16x8            ; C:HL = dot * A
    ld l,h
    ld h,c
    ld b,ATT_SH - 8
ns_sh:
    sra h
    rr l
    djnz ns_sh
    ld a,(nbb_ptr)          ; + lsig (byte 16 of the note)
    ld e,a
    ld a,(nbb_ptr + 1)
    ld d,a
    ex de,hl
    ld bc,16
    add hl,bc
    ld a,(hl)
    ex de,hl
    ld e,a
    rla
    sbc a,a
    ld d,a
    add hl,de
    ret

nb_bonus:                   ; the notebook's bonus, added to LOGITS
    ld a,(nb_count)
    or a
    ret z
    ld (nbb_left),a
    xor a
    ld (nb_nfit),a
    ld hl,FITS
    ld (nb_fitp),hl
    ld hl,NOTES
nbb_cell:
    ld (nbb_ptr),hl
    ld a,(nb_open)          ; the open note proposes nothing
    or a
    jr z,nbb_closed
    ld de,(nb_last)
    ld a,l
    cp e
    jr nz,nbb_closed
    ld a,h
    cp d
    jr z,nbb_next
nbb_closed:
    ld de,17
    add hl,de
    ld a,(nb_plen)
    cp (hl)                 ; the reply's word must be shorter than the note's
    jr nc,nbb_next
    inc hl
    or a
    jr z,nbb_fit
    ld b,a
    ld de,PARTIAL
nbb_cmp:
    ld a,(de)
    cp (hl)
    jr nz,nbb_next
    inc de
    inc hl
    djnz nbb_cmp
nbb_fit:
    ld a,(hl)               ; the symbol this note proposes
    ld (nbb_sym),a
    call nb_score
    ex de,hl                ; DE = score
    ld hl,(nb_fitp)
    ld (hl),e
    inc hl
    ld (hl),d
    inc hl
    ld a,(nbb_sym)
    ld (hl),a
    inc hl
    inc hl
    ld (nb_fitp),hl
    ld a,(nb_nfit)
    or a
    jr z,nbb_best
    ld hl,(nb_best)
    or a
    sbc hl,de               ; best - score < 0: a new best (|scores| < 1000, no overflow)
    jp p,nbb_kept
nbb_best:
    ld (nb_best),de
nbb_kept:
    ld hl,nb_nfit
    inc (hl)
nbb_next:
    ld hl,(nbb_ptr)
    ld de,NOTE
    add hl,de
    ld a,(nbb_left)
    dec a
    ld (nbb_left),a
    jr nz,nbb_cell
    ld a,(nb_nfit)
    or a
    ret z
    ld hl,(NBACC + 66)      ; the trust: SOFTPLUS[clamp8(row 33) + 128]
    ex de,hl
    ld ix,PNB + 99
    call scaled8
    call clamp8
    xor 0x80
    ld l,a
    ld h,SOFTPLUS_PAGE
    ld a,(hl)
    ld (nb_beta),a
    ld hl,0
    ld (nb_total),hl
    ld hl,FITS
    ld a,(nb_nfit)
    ld b,a
nbw_loop:                   ; w = EXP[best - score] (0 from NB_CUT on: 100, EXP's own zero, unless the export sets it)
    push bc
    ld e,(hl)
    inc hl
    ld d,(hl)
    inc hl
    inc hl
    push hl
    ld hl,(nb_best)
    or a
    sbc hl,de
    ld a,h
    or a
    jr nz,nbw_zero
    ld a,l
    cp NB_CUT
    jr nc,nbw_zero
    ld h,EXP_PAGE
    ld a,(hl)
    jr nbw_put
nbw_zero:
    xor a
nbw_put:
    pop hl
    ld (hl),a
    inc hl
    ld e,a
    ld d,0
    push hl
    ld hl,(nb_total)
    add hl,de
    ld (nb_total),hl
    pop hl
    pop bc
    djnz nbw_loop
    ld hl,FITS
    ld a,(nb_nfit)
    ld b,a
nbo_loop:                   ; LOGITS[symbol] += (beta * w) / total
    push bc
    inc hl
    inc hl
    ld a,(hl)
    inc hl
    ld e,(hl)
    inc hl
    push hl
    push af
    ld d,0
    ld a,(nb_beta)
    call mul16x8u
    ex de,hl
    ld bc,(nb_total)
    call div16
    pop af
    ex de,hl
    add a,a
    ld l,a
    ld h,0
    ld bc,LOGITS
    add hl,bc
    ld a,(hl)
    add a,e
    ld (hl),a
    inc hl
    ld a,(hl)
    adc a,d
    ld (hl),a
    pop hl
    pop bc
    djnz nbo_loop
    ret

div16:                      ; HL = DE / BC, unsigned, BC > 0
    ld hl,0
    ld a,16
dv_loop:
    sla e
    rl d
    adc hl,hl
    or a
    sbc hl,bc
    jr nc,dv_set
    add hl,bc
    dec a
    jr nz,dv_loop
    ex de,hl
    ret
dv_set:
    inc e
    dec a
    jr nz,dv_loop
    ex de,hl
    ret
"""


def tables() -> dict[str, bytes]:
    """The notebook's tables: LOGSIG and SOFTPLUS (page-aligned, indexed by a clamped index + 128) and SQ4."""
    from reference.tables import LOGSIG, SOFTPLUS  # noqa: PLC0415

    assert LOGSIG.min() >= -128 and LOGSIG.max() <= 0 and SOFTPLUS.min() >= 0 and SOFTPLUS.max() <= 255
    sq4 = b"".join(int(x * x // 4).to_bytes(2, "little") for x in range(51))
    return {"logsig": bytes(int(v) & 0xFF for v in LOGSIG), "softplus": bytes(int(v) for v in SOFTPLUS), "sq4": sq4}

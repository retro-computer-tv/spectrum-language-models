"""Z80 source for cast slots (reference/text.py), for the story writer and J: the network spells a name once after a
'*', the Spectrum records it in a table (3 rows of 12 bytes, plus a spare row when the table is full) and prints the
row for a slot number, with a capital. The prompt is typed with capitals for names, as English writes them; convert_feed turns
it into the network's symbols by the same rule as text.encode_typed().

slots(io) rewrites the plain IO source: print_sym becomes print_plain, print_ch stays silent while names from the
prompt are recorded, read_prompt flags capitals (bit 7), and the new state is cleared for each story. A built-in
prompt is held as typed, capitals flagged, and goes through convert_feed like a typed one.
"""
from __future__ import annotations

SLOT_IO = r"""
; ---- cast slots ---------------------------------------------------------------------------------------------
print_sym:                  ; A = symbol, printed through the cast table (text.Decoder)
    ld c,a
    ld hl,(spell_ptr)
    ld a,h
    or l
    jr z,pz_free
    ld a,c
    sub 11
    cp 26
    jr nc,pz_end            ; not a letter: the name is complete
    ld a,(spell_room)
    or a
    jr z,pz_letter          ; past 11 letters: printed, not kept
    ld (hl),c
    inc hl
    ld (spell_ptr),hl
    dec a
    ld (spell_room),a
    cp MAXNAME - 1
    jr nz,pz_letter
    ld a,c                  ; the name's first letter: a capital
    call sym_ascii
    sub 32
    jp print_ch
pz_letter:
    ld a,c
    jp print_plain
pz_end:
    ld (hl),0xFF
    ld hl,0
    ld (spell_ptr),hl
pz_free:
    ld a,c
    cp SYM_STAR
    jr c,pz_letter
    jr z,pz_star
    sub SYM_ONE             ; a slot number: print the name it holds
    ld b,a
    ld a,(nslots)
    cp b
    ret z
    ret c
    ld a,b
    call slot_addr
    ld a,(hl)
    cp 0xFF
    ret z
    push hl
    call sym_ascii
    sub 32
    call print_ch
    pop hl
pz_rest:
    inc hl
    ld a,(hl)
    cp 0xFF
    ret z
    push hl
    call print_plain
    pop hl
    jr pz_rest
pz_star:
    ld a,(nslots)
    cp NSLOTS
    jr nc,pz_spare
    inc a
    ld (nslots),a
    dec a
    jr pz_open
pz_spare:
    ld a,NSLOTS             ; table full: spelled into the spare row, not kept
pz_open:
    call slot_addr
    ld (spell_ptr),hl
    ld a,MAXNAME
    ld (spell_room),a
    ret
slot_addr:                  ; HL = SLOTS + 12 * A
    ld l,a
    add a,a
    add a,l
    add a,a
    add a,a
    ld l,a
    ld h,0
    ld de,SLOTS
    add hl,de
    ret
sym_ascii:                  ; A = symbol -> A = its character
    ld hl,vocab_chars
    ld e,a
    ld d,0
    add hl,de
    ld a,(hl)
    ret
print_typed:                ; A = symbol, bit 7 = typed with a capital
    bit 7,a
    jp z,print_plain
    and 0x7F
    call sym_ascii
    sub 32
    jp print_ch

slot_mask:                  ; impossible symbols get the lowest logit (text.Decoder.banned): straight after a '*'
    ld hl,(spell_ptr)       ; anything but a letter; otherwise a slot number whose slot holds no name yet
    ld a,h
    or l
    jr z,sm_digits
    ld a,(spell_room)
    cp MAXNAME
    jr nz,sm_digits
    ld hl,LOGITS            ; symbols 0..10 and 37..40
    ld b,11
    call sm_low
    ld hl,LOGITS + 2 * SYM_STAR
    ld b,4
    jr sm_low
sm_digits:
    ld a,(nslots)
    ld b,a
    ld a,NSLOTS
    sub b
    ret z
    ld c,a                  ; slots still empty
    ld a,b
    add a,SYM_ONE
    add a,a
    ld e,a
    ld d,0
    ld hl,LOGITS
    add hl,de
    ld b,c
sm_low:                     ; B logits from HL: -32768
    ld (hl),0
    inc hl
    ld (hl),0x80
    inc hl
    djnz sm_low
    ret

convert_feed:               ; the typed prompt, capitalised words as cast slots (text.encode_typed), fed in order
    ld a,1
    ld (quiet),a            ; names go into the table unprinted: the prompt is on the screen as typed
    ld hl,PROMPT
cf_loop:
    ld a,(hl)
    cp 0xFF
    jr z,cf_done
    bit 7,a
    jr nz,cf_name
    inc hl
    call cf_emit
    jr cf_loop
cf_name:
    ld b,0
cf_try:
    ld a,(nslots)
    cp b
    jr z,cf_new
    push hl
    push bc
    ld a,b
    push hl
    call slot_addr
    ex de,hl                ; DE = the slot's name
    pop hl
cf_cmp:
    ld a,(hl)
    and 0x7F
    ld c,a
    sub 11
    cp 26
    jr nc,cf_wend
    ld a,(de)
    cp c
    jr nz,cf_no
    inc hl
    inc de
    jr cf_cmp
cf_wend:
    ld a,(de)
    cp 0xFF
    jr nz,cf_no
    pop bc
    pop af                  ; drop the word's start: HL is past the word
    ld a,SYM_ONE
    add a,b
    call cf_emit
    jr cf_loop
cf_no:
    pop bc
    pop hl
    inc b
    jr cf_try
cf_new:
    ld a,(nslots)
    cp NSLOTS
    jr nc,cf_letters
    ld a,SYM_STAR
    call cf_emit
cf_letters:
    ld a,(hl)
    and 0x7F
    ld c,a
    sub 11
    cp 26
    jr nc,cf_loop
    inc hl
    ld a,c
    call cf_emit
    jr cf_letters
cf_done:
    xor a
    ld (quiet),a
    ret
cf_emit:                    ; A = symbol: record it, then feed it to the network
    push hl
    push af
    call print_sym
    pop af
    call feed
    pop hl
    ret
"""

EDITS = [
    ("print_sym:                  ; A = symbol\n", "print_plain:                ; A = symbol\n"),
    ("print_ch:\n    push af\n    ld a,255\n", "print_ch:\n    push af\n    ld a,(quiet)\n    or a\n    jr nz,pc_quiet\n"
     "    ld a,255\n"),
    ("    rst 16\n    ret\nprint_str:", "    rst 16\n    ret\npc_quiet:\n    pop af\n    ret\nprint_str:"),
    ("    cp 'A'\n    jr c,rp_map\n    cp 'Z' + 1\n    jr nc,rp_map\n    add a,32\nrp_map:\n",
     "    ld c,0\n    cp 'A'\n    jr c,rp_map\n    cp 'Z' + 1\n    jr nc,rp_map\n    add a,32\n    ld c,0x80\nrp_map:\n"),
    ("rp_found:\n    ld a,c\n    pop bc\n    pop hl\n    ld e,a\n",
     "rp_found:\n    ld a,c\n    pop bc\n    pop hl\n    or c                    ; bit 7: typed with a capital\n    ld e,a\n"),
    ("    push de\n    call print_sym\n    pop de\n", "    push de\n    call print_typed\n    pop de\n"),  # print_syms
    ("    ld a,e\n    call print_sym\n    pop bc\n    pop hl\n    jr rp_key\n",
     "    ld a,e\n    call print_typed\n    pop bc\n    pop hl\n    jr rp_key\n"),
    ("    ldir                    ; H16 and HQ (HQ follows H16; its padding stays zero)\n    ret\n",
     "    ldir                    ; H16 and HQ (HQ follows H16; its padding stays zero)\n    xor a\n    ld (nslots),a\n"
     "    ld h,a\n    ld l,a\n    ld (spell_ptr),hl\n    ret\n"),
]

# printer-buffer variables (the engines place them after their own)
VARS = (("SLOTS", 48), ("nslots", 1), ("spell_ptr", 2), ("spell_room", 1), ("quiet", 1))


def slots(io: str) -> str:
    for old, new in EDITS:
        assert io.count(old) == 1, old
        io = io.replace(old, new)
    return io + SLOT_IO

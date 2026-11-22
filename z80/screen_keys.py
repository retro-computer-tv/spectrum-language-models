"""Z80 source: buffers, printing through the ROM, and the prompt typed on the Spectrum's keyboard, for the story writer and J."""

IO = r"""
; ---- buffers, state -----------------------------------------------------------------------------------------
zero_buffers:
    ld hl,0x5B00
    ld de,0x5B01
    ld bc,255
    ld (hl),0
    ldir
    ld hl,BUFFERS
    ld de,BUFFERS + 1
    ld bc,BUFFERS_END - BUFFERS - 1
    ld (hl),0
    ldir
    ret
reset_state:
    ld hl,H16
    ld de,H16 + 1
    ld bc,3 * NUNITS - 1
    ld (hl),0
    ldir                    ; H16 and HQ (HQ follows H16; its padding stays zero)
    ret
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

; ---- printing through the ROM (channel 2, the upper screen) ----------------------------------------------
print_sym:                  ; A = symbol
    ld hl,vocab_chars
    ld e,a
    ld d,0
    add hl,de
    ld a,(hl)
print_ch:
    push af
    ld a,255
    ld (0x5C8C),a           ; SCR_CT: never stop to ask 'scroll?'
    pop af
    rst 16
    ret
print_str:                  ; HL = text ending 0xFF
    ld a,(hl)
    cp 0xFF
    ret z
    push hl
    call print_ch
    pop hl
    inc hl
    jr print_str
print_syms:                 ; HL = symbols ending 0xFF, also copied to PROMPT
    ld de,PROMPT
ps_loop:
    ld a,(hl)
    ld (de),a
    cp 0xFF
    ret z
    push hl
    push de
    call print_sym
    pop de
    pop hl
    inc hl
    inc de
    jr ps_loop

; ---- the prompt, typed on the Spectrum's own keyboard (the ROM's interrupt scans it) -----------------------
read_prompt:
    ld hl,PROMPT
    ld b,0
rp_key:
    push hl
    push bc
    call wait_key
    pop bc
    pop hl
    cp 13
    jr z,rp_done
    cp 12
    jr z,rp_del
    cp 'A'
    jr c,rp_map
    cp 'Z' + 1
    jr nc,rp_map
    add a,32
rp_map:
    push hl
    push bc
    ld hl,vocab_chars + 2   ; symbols 2.. (no end-of-story, no newline)
    ld b,NTYPE
    ld c,2
rp_find:
    cp (hl)
    jr z,rp_found
    inc hl
    inc c
    djnz rp_find
    pop bc
    pop hl
    jr rp_key
rp_found:
    ld a,c
    pop bc
    pop hl
    ld e,a
    ld a,b
    cp 120
    jr nc,rp_key
    ld (hl),e
    inc hl
    inc b
    push hl
    push bc
    ld a,e
    call print_sym
    pop bc
    pop hl
    jr rp_key
rp_del:
    ld a,b
    or a
    jr z,rp_key
    dec hl
    dec b
    push hl
    push bc
    ld a,8
    call print_ch
    ld a,' '
    call print_ch
    ld a,8
    call print_ch
    pop bc
    pop hl
    jr rp_key
rp_done:
    ld (hl),0xFF
    ld hl,(0x5C78)          ; FRAMES at ENTER seeds the sampler (never 0)
    ld a,h
    or l
    jr nz,rp_seed
    inc hl
rp_seed:
    ld (seed),hl
    ld (seed_used),hl
    ld a,13
    call print_ch
    ret
wait_key:                   ; A = the next key's code (L mode), via LAST-K
    ei
    res 5,(iy+1)
wk_loop:
    set 3,(iy+1)            ; L mode: letters in lower case
    halt
    bit 5,(iy+1)
    jr z,wk_loop
    ld a,(0x5C08)
    di
    ret
"""

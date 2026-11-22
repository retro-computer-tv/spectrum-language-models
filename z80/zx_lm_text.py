"""Z80 source for ZX-LM's text (reference/zx_lm.py is the reference): the line typed or built in, words and
references, printing a reply. sjasmplus syntax.

Words follow vocab.words(): lower case; [a-z]+n't | [a-z]+ | 's | one of . , ? : !; anything else is skipped. A word
in the vocabulary is its token; any other word is a reference (REFS: 16 entries, a length byte and 15 letters).
A new word gets the reference xorshift16 & 15, or the next free one after it (reference/zx_lm.py's Z80Refs is the same rule).
"""

TEXT = r"""
; ---- the line: built in (AUTO_TEXT, lines ending 0, the list ending 0xFF) or typed --------------------------------
get_line:                   ; LINE = the next line (0-terminated); Z set when there is none
    ld a,(auto_on)
    or a
    jr z,read_line
    ld hl,(auto_ptr)
    ld a,(hl)
    cp 0xFF
    ret z
    push hl
    call print_z            ; shown as written
    pop hl
    ld de,LINE
gl_copy:                    ; kept in small letters, as typed lines are
    ld a,(hl)
    cp 'A'
    jr c,gl_put
    cp 'Z' + 1
    jr nc,gl_put
    add a,32
gl_put:
    ld (de),a
    inc hl
    inc de
    or a
    jr nz,gl_copy
    ld (auto_ptr),hl
    ld a,13
    call print_ch
    or 1                    ; NZ: a line
    ret

read_line:                  ; typed: letters (capitals made small), space, ' . , ? : ! ; DELETE; ENTER ends
    ld hl,LINE
    ld b,0
rl_key:
    push hl
    push bc
    call wait_key
    pop bc
    pop hl
    cp 13
    jr z,rl_done
    cp 12
    jr z,rl_del
    cp 'A'
    jr c,rl_chk
    cp 'Z' + 1
    jr nc,rl_chk
    add a,32
rl_chk:
    cp 'a'
    jr c,rl_sym
    cp 'z' + 1
    jr c,rl_ok
rl_sym:
    push hl
    push bc
    ld hl,typeable
    ld bc,7
    cpir
    pop bc
    pop hl
    jr nz,rl_key
rl_ok:
    ld e,a
    ld a,b
    cp 120
    jr nc,rl_key
    ld (hl),e
    inc hl
    inc b
    push hl
    push bc
    ld a,e
    call print_ch
    pop bc
    pop hl
    jr rl_key
rl_del:
    ld a,b
    or a
    jr z,rl_key
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
    jr rl_key
rl_done:
    ld (hl),0
    ld a,(seeded)
    or a
    jr nz,rl_nl
    ld hl,(0x5C78)          ; FRAMES at the first ENTER seeds the reference numbers (never 0)
    ld a,h
    or l
    jr nz,rl_seed
    inc hl
rl_seed:
    ld (seed),hl
    ld a,1
    ld (seeded),a
rl_nl:
    ld a,13
    call print_ch
    ld a,b
    or a                    ; Z: an empty line
    ret
typeable: db " '.,?:!"

wait_key:                   ; A = the next key's code (L mode), via LAST-K
    ei
    res 5,(iy+1)
wk_loop:
    set 3,(iy+1)
    halt
    bit 5,(iy+1)
    jr z,wk_loop
    ld a,(0x5C08)
    di
    ret

; ---- words to tokens: TOKS, NTOK ---------------------------------------------------------------------------
tokenize:
    ld hl,LINE
    ld de,TOKS
    xor a
    ld (NTOK),a
tk_loop:
    ld a,(hl)
    or a
    ret z
    cp 'a'
    jr c,tk_notl
    cp 'z' + 1
    jr c,tk_word
tk_notl:
    cp 39
    jr z,tk_apos
    push hl
    ld hl,punct
    ld bc,5
    cpir
    pop hl
    jr z,tk_punct
    inc hl
    jr tk_loop
tk_apos:
    inc hl
    ld a,(hl)
    cp 's'
    jr nz,tk_loop
    inc hl
    ld a,39
    ld (WORD + 1),a
    ld a,'s'
    ld (WORD + 2),a
    ld a,2
    ld (WORD),a
    jr tk_emit
tk_punct:
    ld (WORD + 1),a
    ld a,1
    ld (WORD),a
    inc hl
    jr tk_emit
tk_word:
    ld b,0                  ; letters kept (at most 15); C counts them all
    ld c,0
tk_letters:
    ld a,(hl)
    cp 'a'
    jr c,tk_wend
    cp 'z' + 1
    jr nc,tk_wend
    inc c
    ld (tk_last),a
    ld (tk_ch),a
    ld a,b
    cp 15
    jr nc,tk_skipch
    inc b
    push hl
    push de
    ld e,b
    ld d,0
    ld hl,WORD
    add hl,de
    ld a,(tk_ch)
    ld (hl),a
    pop de
    pop hl
tk_skipch:
    inc hl
    jr tk_letters
tk_wend:
    cp 39                   ; n't: the word's last letter is n, at least two letters, then ' t
    jr nz,tk_wdone
    ld a,c
    cp 2
    jr c,tk_wdone
    ld a,(tk_last)
    cp 'n'
    jr nz,tk_wdone
    inc hl
    ld a,(hl)
    dec hl
    cp 't'
    jr nz,tk_wdone
    inc hl
    inc hl
    ld a,b
    cp 14
    jr nc,tk_wdone
    push hl
    push de
    ld e,b
    ld d,0
    ld hl,WORD + 1
    add hl,de
    ld (hl),39
    inc hl
    ld (hl),'t'
    pop de
    pop hl
    inc b
    inc b
tk_wdone:
    ld a,b
    ld (WORD),a
tk_emit:                    ; WORD (length, letters) -> a token at DE
    push hl
    push de
    call vocab_find
    jr nc,tk_known
    call ref_get
    add a,NWORDS
tk_known:
    pop de
    ld (de),a
    inc de
    ld hl,NTOK
    inc (hl)
    pop hl
    ld a,(NTOK)
    cp 60
    ret nc
    jp tk_loop
punct: db ".,?:!"
tk_last: db 0
tk_ch: db 0

; ---- A = the vocabulary index of WORD (carry clear), or carry set -----------------------------------------------
vocab_find:
    ld hl,VOCAB_STR
    ld c,0
vf_word:
    ld a,(hl)
    cp 0xFF
    jr z,vf_none
    push hl
    ld de,WORD
    ld a,(de)
    ld b,(hl)
    cp b
    jr nz,vf_skip
    inc b                   ; compare the length byte and the letters
vf_cmp:
    ld a,(de)
    cp (hl)
    jr nz,vf_skip
    inc hl
    inc de
    djnz vf_cmp
    pop hl
    ld a,c
    or a                    ; carry clear
    ret
vf_skip:
    pop hl
    ld e,(hl)
    ld d,0
    inc hl
    add hl,de
    inc c
    jr vf_word
vf_none:
    scf
    ret

; ---- A = the reference holding WORD, a new one if none does --------------------------------------------------
ref_get:
    ld c,0
rg_find:
    ld a,c
    call ref_addr
    ld a,(hl)
    or a
    jr z,rg_next
    ld de,WORD
    ld a,(de)
    ld b,a
    inc b
rg_cmp:
    ld a,(de)
    cp (hl)
    jr nz,rg_next
    inc hl
    inc de
    djnz rg_cmp
    ld a,c
    ret
rg_next:
    inc c
    ld a,c
    cp 16
    jr nz,rg_find
    call xrnd               ; a new one: the next free from (xorshift16 & 15)
    ld a,l
    and 15
    ld c,a
    ld b,16
rg_probe:
    ld a,c
    call ref_addr
    ld a,(hl)
    or a
    jr z,rg_take
    ld a,c
    inc a
    and 15
    ld c,a
    djnz rg_probe
    ld a,c                  ; all 16 in use: replace this one
    call ref_addr
rg_take:
    ex de,hl
    ld hl,WORD
    ld a,(hl)
    inc a
    ld b,0
    push bc
    ld c,a
    ldir
    pop bc
    ld a,c
    ret
ref_addr:                   ; HL = REFS + 16 A
    ld l,a
    ld h,0
    add hl,hl
    add hl,hl
    add hl,hl
    add hl,hl
    ld de,REFS
    add hl,de
    ret

; ---- printing ------------------------------------------------------------------------------------------------
print_ch:
    push af
    ld a,255
    ld (0x5C8C),a           ; SCR_CT: never stop to ask 'scroll?'
    pop af
    rst 16
    ret
print_z:                    ; HL = 0-terminated text
    ld a,(hl)
    or a
    ret z
    push hl
    call print_ch
    pop hl
    inc hl
    jr print_z
print_str:                  ; HL = text ending 0xFF
    ld a,(hl)
    cp 0xFF
    ret z
    push hl
    call print_ch
    pop hl
    inc hl
    jr print_str
print_len:                  ; HL = a length byte and the letters
    ld b,(hl)
    inc hl
    ld a,b
    or a
    ret z
pl_loop:
    ld a,(hl)
    push hl
    push bc
    call print_ch
    pop bc
    pop hl
    inc hl
    djnz pl_loop
    ret

print_tok:                  ; A = token; a space first unless it opens the reply or starts with ' . , ? : !
    call tok_str
    ld (pt_str),hl
    ld a,(pt_first)
    or a
    jr nz,pt_nosp
    inc hl
    ld a,(hl)
    cp 39
    jr z,pt_nosp
    ld hl,punct
    ld bc,5
    cpir
    jr z,pt_nosp
    ld a,' '
    call print_ch
pt_nosp:
    xor a
    ld (pt_first),a
    ld hl,(pt_str)
    jp print_len
pt_str: dw 0
pt_first: db 0

tok_str:                    ; HL = the length-prefixed text of token A
    cp NWORDS
    jr c,tok_addr
    sub NWORDS
    jp ref_addr

tok_addr:                   ; HL = the vocabulary string of token A
    ld hl,VOCAB_STR
    or a
    ret z
    ld b,a
ta_loop:
    ld e,(hl)
    ld d,0
    inc hl
    add hl,de
    djnz ta_loop
    ret
"""

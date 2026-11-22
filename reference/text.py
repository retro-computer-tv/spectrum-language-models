"""The alphabets: what the networks read and write, symbol by symbol.

The story writer reads and writes 37 symbols: '#' (the end of a story), newline, space, ! " ' , - . : ? and the 26
lower-case letters. Cast slots add four: '*' and the digits 1 to 3. The first time a story names someone the network
writes '*' and spells the name; after that it writes the slot's digit, and the Spectrum keeps the spelling in a little
table and prints it, with a capital. So "Lily" and "Zara" look the same to the network once introduced: both are "1".
On the Spectrum you type a name with a capital, as English writes it, and encode_typed() is the rule the Z80 applies.

J (the conversation network) adds two turn markers to that:

    #hello! my name is *rosa.>hello 1! nice to meet you.<what is your name?>i'm a little computer.<

'#' starts a conversation, '>' ends what you typed (ENTER), '<' ends the Spectrum's reply.
ZX-LM reads whole words instead (reference/zx_lm.py).
"""
from __future__ import annotations

EOS = "#"
VOCAB = EOS + "\n !\"',-.:?" + "abcdefghijklmnopqrstuvwxyz"
STOI = {c: i for i, c in enumerate(VOCAB)}

NSLOTS = 3
MAXNAME = 11  # letters kept per slot (the Spectrum's table rows are 12 bytes with the end marker)
SVOCAB = VOCAB + "*123"
SSTOI = {c: i for i, c in enumerate(SVOCAB)}

CVOCAB = SVOCAB + "><"
CSTOI = {c: i for i, c in enumerate(CVOCAB)}
YOU_END, SPEC_END = CSTOI[">"], CSTOI["<"]


def encode_typed(prompt: str) -> str:
    """A prompt as typed on the Spectrum: a capital letter starts a name (the rest of its letters, any case, are the
    name). A name already in a slot becomes its digit, a new one "*" + the name while a slot is free, else plain."""
    out, table, i = [], [], 0
    while i < len(prompt):
        ch = prompt[i]
        if not ch.isupper():
            out.append(ch)
            i += 1
            continue
        j = i
        while j < len(prompt) and prompt[j].isalpha():
            j += 1
        w = prompt[i:j].lower()
        if w in table:
            out.append(str(table.index(w) + 1))
        elif len(table) < NSLOTS:
            table.append(w[:MAXNAME])
            out.append("*" + w)
        else:
            out.append(w)
        i = j
    return "".join(out)


class Decoder:
    """Streaming decode, symbol by symbol, as the Spectrum prints: returns the text each symbol adds. With caps, names
    are printed with a capital (the Spectrum's display)."""

    def __init__(self, caps: bool = False):
        self.caps = caps
        self.reset()

    def reset(self):
        self.slots: list[str] = []
        self.spelling: str | None = None

    def banned(self) -> list[int]:
        """Symbols that cannot come next (the sampling rule on the Spectrum): anything but a letter straight after a
        '*', and a slot number whose slot holds no name yet (a name being spelled counts, as on the Z80)."""
        if self.spelling == "":
            return [i for i, c in enumerate(SVOCAB) if not c.isalpha()]
        filled = len(self.slots) + (1 if self.spelling is not None and len(self.slots) < NSLOTS else 0)
        return [SSTOI[str(k + 1)] for k in range(filled, NSLOTS)]

    def push(self, ch: str) -> str:
        if ch == "#":
            self.reset()
            return "#"
        if self.spelling is not None:
            if ch.isalpha():
                self.spelling += ch
                return ch.upper() if self.caps and len(self.spelling) == 1 else ch
            if len(self.slots) < NSLOTS:
                self.slots.append(self.spelling[:MAXNAME])
            self.spelling = None
        if ch == "*":
            self.spelling = ""
            return ""
        if ch in "123":
            k = int(ch) - 1
            w = self.slots[k] if k < len(self.slots) else ""
            return w[:1].upper() + w[1:] if self.caps else w
        return ch


def decode(text: str) -> str:
    """Slot symbols back to plain text (names restored)."""
    d = Decoder()
    return "".join(d.push(c) for c in text)


class Stream:
    """A conversation as the Spectrum sees it, symbol by symbol: the slot table and the display. typed() turns a
    line as you type it into symbols (a capital starts a name, with the slot table of the whole conversation so far);
    push() records every symbol fed or written."""

    def __init__(self):
        self.dec = Decoder(caps=True)
        self.shown: list[str] = []

    def push(self, c: int) -> str:
        ch = CVOCAB[c]
        if ch == "#":
            self.dec.push("#")  # a new conversation: the slot table is cleared
            out = ""
        elif ch in "><":
            self.dec.push(" ")  # ends a name being spelled
            out = "\n"
        else:
            out = self.dec.push(ch)
        self.shown.append(out)
        return out

    def typed(self, line: str) -> list[int]:
        out, i = [], 0
        pending = list(self.dec.slots)
        while i < len(line):
            ch = line[i]
            if not ch.isupper():
                out.append(ch.lower())
                i += 1
                continue
            j = i
            while j < len(line) and line[j].isalpha():
                j += 1
            w = line[i:j].lower()
            if w in pending:
                out.append(str(pending.index(w) + 1))
            elif len(pending) < NSLOTS:
                pending.append(w[:MAXNAME])
                out.append("*" + w)
            else:
                out.append(w)
            i = j
        return [CSTOI[c] for c in "".join(out) + ">"]

    def banned(self) -> list[int]:
        """The slot sampling rule, plus: no '>' from the Spectrum (only you end your turn)."""
        return self.dec.banned() + [YOU_END]

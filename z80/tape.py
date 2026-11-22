"""The assembler call and the .tap file: a BASIC loader, then the machine code.

Each tape holds four blocks: a header for the BASIC program, the program, a header for the CODE block, and the code
itself, loaded at 24064 (0x5E00). The BASIC program is one line, which runs itself after loading:

    10 CLEAR VAL "24063": LOAD ""CODE : RANDOMIZE USR VAL "<start>"

VAL "..." keeps each number as its digits only (BASIC would otherwise store a hidden five-byte copy as well).

The assembler is sjasmplus; the tapes here were built with version 1.24.0. Its releases page,
https://github.com/z00m128/sjasmplus/releases, has a ready-made Windows download; on Linux and macOS it builds with
`make`. Put it on the PATH, or name it with the SJASMPLUS environment variable.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ORG = 0x5E00
CLEAR, VAL, LOAD, CODE, RANDOMIZE, USR, ENTER = 0xFD, 0xB0, 0xEF, 0xAF, 0xF9, 0xC0, 0x0D


def assembler() -> str:
    return os.environ.get("SJASMPLUS", "sjasmplus")


def assemble(folder: Path, name: str) -> dict:
    """Assemble <folder>/<name>.asm into <name>.raw and <name>.sym; returns the symbol table."""
    try:
        r = subprocess.run([assembler(), "--nologo", "--msg=war", f"--raw={name}.raw", f"--sym={name}.sym",
                            f"{name}.asm"], cwd=folder, capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit(f"sjasmplus not found ({assembler()}): put it on the PATH or set SJASMPLUS") from None
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    sym = {}
    for line in (folder / f"{name}.sym").read_text().splitlines():
        if ": EQU" in line:
            k, v = line.split(": EQU")
            sym[k.strip()] = int(v.strip(), 0)
    return sym


def basic_loader(start: int) -> tuple[bytes, int]:
    """The loader line as the ROM stores it (line number, length, tokens): (program bytes, length of the program)."""
    body = bytes([CLEAR, VAL]) + b'"%d":' % (ORG - 1) + bytes([LOAD]) + b'""' + bytes([CODE]) + b":" \
        + bytes([RANDOMIZE, USR, VAL]) + b'"%d"' % start + bytes([ENTER])
    prog = (10).to_bytes(2, "big") + len(body).to_bytes(2, "little") + body
    return prog, len(prog)


def tap_block(flag: int, data: bytes) -> bytes:
    body = bytes([flag]) + data
    x = 0
    for b in body:
        x ^= b
    body += bytes([x])
    return len(body).to_bytes(2, "little") + body


def tap(start: int, code: bytes, basic_name: bytes, code_name: bytes) -> bytes:
    """A whole tape: the BASIC loader (auto-run from line 10) and the CODE block at ORG."""
    prog, progvars = basic_loader(start)
    hdr = bytes([0]) + basic_name.ljust(10) + len(prog).to_bytes(2, "little") + (10).to_bytes(2, "little")
    hdr += progvars.to_bytes(2, "little")
    chdr = bytes([3]) + code_name.ljust(10) + len(code).to_bytes(2, "little") + ORG.to_bytes(2, "little") \
        + (32768).to_bytes(2, "little")
    return tap_block(0, hdr) + tap_block(0xFF, prog) + tap_block(0, chdr) + tap_block(0xFF, code)

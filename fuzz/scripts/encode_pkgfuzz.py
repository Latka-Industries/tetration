#!/usr/bin/env python3
"""Encode a length-prefixed package fuzz seed (THI-481).

Format (little-endian u32 lengths):
  PROTOCOL | CURRENT | MANIFEST.json | packed.tet

Usage:
  encode_pkgfuzz.py OUT.pkgfuzz --protocol FILE|--protocol-text STR \\
    [--current FILE|--current-text STR] [--manifest FILE|--manifest-text STR] \\
    [--packed FILE]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def field(data: bytes) -> bytes:
    return struct.pack("<I", len(data)) + data


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--protocol", type=Path)
    g.add_argument("--protocol-text", type=str)
    c = p.add_mutually_exclusive_group()
    c.add_argument("--current", type=Path)
    c.add_argument("--current-text", type=str, default=None)
    m = p.add_mutually_exclusive_group()
    m.add_argument("--manifest", type=Path)
    m.add_argument("--manifest-text", type=str)
    p.add_argument("--packed", type=Path)
    args = p.parse_args()

    protocol = (
        args.protocol.read_bytes()
        if args.protocol
        else args.protocol_text.encode("utf-8")
    )
    if args.current is not None:
        current = args.current.read_bytes()
    elif args.current_text is not None:
        current = args.current_text.encode("utf-8")
    else:
        current = b""
    if args.manifest is not None:
        manifest = args.manifest.read_bytes()
    elif args.manifest_text is not None:
        manifest = args.manifest_text.encode("utf-8")
    else:
        manifest = b""
    packed = args.packed.read_bytes() if args.packed else b""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(field(protocol) + field(current) + field(manifest) + field(packed))
    return 0


if __name__ == "__main__":
    sys.exit(main())

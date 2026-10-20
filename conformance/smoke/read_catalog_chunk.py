#!/usr/bin/env python3
"""SPEC smoke reader (THI-589) — Python stdlib only.

Opens a layout-v1 ``.tet``, prints the catalog, and decodes **one** raw
(``codec = 0``) chunk to little-endian ``f32`` values. Not a query engine.

Failure here usually means SPEC ambiguity — see ``SPEC/README.md`` §1.

Usage (from repo root)::

    python3 conformance/smoke/read_catalog_chunk.py fixtures/small/tet/sample.tet
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

DTYPE_F32 = 1
CODEC_RAW = 0
ENTRY_WIRE = 104


@dataclass
class Dataset:
    name: str
    dtype: int
    ndim: int
    shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]


@dataclass
class ChunkEntry:
    dataset_id: int
    chunk_index: tuple[int, ...]
    payload_offset: int
    raw_byte_len: int
    stored_byte_len: int
    codec: int


def align8(n: int) -> int:
    return (n + 7) & ~7


def read_superblock(data: bytes) -> tuple[int, int, int, int, int]:
    if len(data) < 32:
        raise ValueError("file shorter than 32-byte superblock")
    magic, layout, ds_count, flags, cio, cil = struct.unpack_from("<4sIIIQQ", data, 0)
    if magic != b"TETR":
        raise ValueError(f"bad magic {magic!r}")
    if layout != 1:
        raise ValueError(f"unsupported layout_version {layout}")
    return ds_count, flags, cio, cil, layout


def read_datasets(data: bytes, dataset_count: int) -> list[Dataset]:
    if dataset_count == 0:
        return []
    (blob_len,) = struct.unpack_from("<Q", data, 32)
    off = 40
    end = 40 + blob_len
    out: list[Dataset] = []
    for _ in range(dataset_count):
        if off + 16 > end:
            raise ValueError("dataset directory truncated")
        name_len, dtype, ndim, reserved = struct.unpack_from("<IIII", data, off)
        if reserved != 0:
            # v1 writers must write 0; smoke ignores nonzero (COMPATIBILITY §3.2)
            pass
        if not (1 <= ndim <= 8):
            raise ValueError(f"ndim out of range: {ndim}")
        name_start = off + 16
        name_end = name_start + name_len
        name = data[name_start:name_end].decode("utf-8")
        # pad so offset from record start to shape[0] is 8-aligned
        record_start = off
        after_name = name_end
        pad = align8(after_name - record_start) - (after_name - record_start)
        shape_off = after_name + pad
        shape = struct.unpack_from("<" + "Q" * ndim, data, shape_off)
        chunk_off = shape_off + 8 * ndim
        chunk_shape = struct.unpack_from("<" + "Q" * ndim, data, chunk_off)
        out.append(
            Dataset(
                name=name,
                dtype=dtype,
                ndim=ndim,
                shape=shape,
                chunk_shape=chunk_shape,
            )
        )
        off = chunk_off + 8 * ndim
    if off != end:
        raise ValueError(f"dataset blob length mismatch: consumed {off - 40} vs {blob_len}")
    return out


def read_chunk_index(data: bytes, offset: int, length: int) -> list[ChunkEntry]:
    if length == 0:
        return []
    if offset + length > len(data):
        raise ValueError("chunk index past EOF")
    magic, index_version, entry_count = struct.unpack_from("<4sIQ", data, offset)
    if magic != b"TIDX":
        raise ValueError(f"bad TIDX magic {magic!r}")
    if index_version != 1:
        raise ValueError(f"unsupported index_version {index_version}")
    expect = 32 + entry_count * ENTRY_WIRE
    if length != expect:
        raise ValueError(f"chunk_index_length {length} != {expect}")
    entries: list[ChunkEntry] = []
    pos = offset + 32
    for _ in range(entry_count):
        # dataset_id, chunk_index[8], payload_offset, raw, stored, codec, reserved
        fields = struct.unpack_from("<Q" + "Q" * 8 + "QQQII", data, pos)
        dataset_id = fields[0]
        chunk_index = fields[1:9]
        payload_offset = fields[9]
        raw_byte_len = fields[10]
        stored_byte_len = fields[11]
        codec = fields[12]
        reserved = fields[13]
        if reserved != 0:
            pass
        entries.append(
            ChunkEntry(
                dataset_id=dataset_id,
                chunk_index=chunk_index,
                payload_offset=payload_offset,
                raw_byte_len=raw_byte_len,
                stored_byte_len=stored_byte_len,
                codec=codec,
            )
        )
        pos += ENTRY_WIRE
    return entries


def decode_raw_f32(data: bytes, entry: ChunkEntry) -> list[float]:
    if entry.codec != CODEC_RAW:
        raise ValueError(f"smoke reader only supports codec=0 raw, got {entry.codec}")
    if entry.stored_byte_len != entry.raw_byte_len:
        raise ValueError("raw codec requires stored_byte_len == raw_byte_len")
    start = entry.payload_offset
    end = start + entry.stored_byte_len
    if end > len(data):
        raise ValueError("payload past EOF")
    blob = data[start:end]
    if len(blob) % 4 != 0:
        raise ValueError("f32 payload length not multiple of 4")
    n = len(blob) // 4
    return list(struct.unpack("<" + "f" * n, blob))


def smoke(path: Path) -> int:
    data = path.read_bytes()
    ds_count, _flags, cio, cil, layout = read_superblock(data)
    datasets = read_datasets(data, ds_count)
    if ds_count > 0:
        expect_cio = align8(40 + struct.unpack_from("<Q", data, 32)[0])
        if cio != expect_cio:
            raise ValueError(f"chunk_index_offset {cio} != align8(...) {expect_cio}")
    entries = read_chunk_index(data, cio, cil)

    print(f"file={path}")
    print(f"layout_version={layout} dataset_count={ds_count} chunk_entries={len(entries)}")
    for i, ds in enumerate(datasets):
        print(
            f"  dataset[{i}] name={ds.name!r} dtype={ds.dtype} "
            f"shape={list(ds.shape)} chunk_shape={list(ds.chunk_shape)}"
        )

    if not entries:
        print("no chunks; ok")
        return 0

    entry = entries[0]
    ds = datasets[entry.dataset_id]
    print(
        f"first_chunk dataset_id={entry.dataset_id} codec={entry.codec} "
        f"raw_byte_len={entry.raw_byte_len} offset={entry.payload_offset}"
    )
    if ds.dtype != DTYPE_F32:
        print(f"skip decode: smoke only decodes dtype=f32 (got {ds.dtype})")
        return 0
    values = decode_raw_f32(data, entry)
    preview = values[:8]
    print(f"decoded_f32_count={len(values)} preview={preview}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "tet",
        nargs="?",
        default="fixtures/small/tet/sample.tet",
        type=Path,
        help="path to a layout-v1 .tet (default: sample.tet)",
    )
    args = ap.parse_args(argv)
    try:
        return smoke(args.tet)
    except Exception as e:  # noqa: BLE001 — smoke: surface any SPEC mismatch
        print(f"smoke failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

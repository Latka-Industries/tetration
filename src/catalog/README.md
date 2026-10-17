# `catalog` — datasets, chunk index, and I/O

Maps the on-disk layout **after** the superblock: dataset directory, `TIDX` chunk index, raw/zstd payloads, and optional `THST` footer. This is the core read/write surface for `.tet` files.

Spec: [`docs/layout_v1.md`](../../docs/layout_v1.md).

## Embedder entrypoints

| Type / fn             | Use                                                              |
| --------------------- | ---------------------------------------------------------------- |
| `TetWriterSession`    | Buffer datasets, `commit()` or `commit_with_fill()` → one `.tet` |
| `TetDatasetWrite`     | In-memory row-major tiles: `f32_row_major`, `f64_row_major`, `row_major(dtype, …)` |
| `TetDatasetStreamSpec`| Streaming geometry: same builders, fill tiles at commit          |
| `TetFile`             | Open for mmap; `summary()`, path, backing mmap                   |
| `TetPackage`          | `.tet.d` skeleton: create/open, `CURRENT`, publish lock, seal `packed.tet` ([writer protocol v2](../../docs/writer_protocol_v2.md)) |
| `TetPackagePin`       | Reader pin: one generation for the life of a query handle (`pin_current` / `pin_generation`) |
| `TetShardWriter`      | Staging shard: `part-*.tet` + `PART_META.json` (no `CURRENT` flip) — multi-writer ingest |
| `publish_parts`       | Committer: frozen part ids → merge `packed.tet` → seal generation / flip `CURRENT` |
| `read_tet_summary_v1` | Parse catalog from bytes (no extra health checks)                |
| `TetFileSummaryV1`    | Superblock + datasets + chunks + history + metadata              |

Verify APIs are implemented in [`verify`](../verify/README.md) but re-exported here for compatibility.

## Submodules

| File              | Role                                                                           |
| ----------------- | ------------------------------------------------------------------------------ |
| `dataset.rs`      | Dataset directory record encoding (`name`, `dtype`, `shape`, `chunk_shape`, …) |
| `index.rs`        | `TIDX` header + `ChunkIndexEntryV1` parse/validate                             |
| `file_layout.rs`  | Shared chunk-grid math, index sizing, payload cursor (create/append/stream)    |
| `tile.rs`         | Chunk grid counts; which chunk coords intersect a global box                   |
| `write.rs`        | Create new files from in-memory tensors                                        |
| `stream_write.rs` | One-chunk-at-a-time writer (`StreamTileJob`)                                   |
| `append.rs`       | Add datasets to an existing file (rewrites catalog + index)                    |
| `history.rs`      | `THST` footer: convert history rows, metadata blob refs                        |
| `metadata.rs`     | Typed `metadata` JSON (axes, coord labels, file/dataset attrs)                 |
| `execution.rs`    | Per-file defaults in chunk index header (memory budget bps)                    |
| `session.rs`      | `TetWriterSession` / `TetFile` façade; `TetDatasetWrite` / `TetDatasetStreamSpec` |
| `package.rs`      | `.tet.d` package roles (`PROTOCOL`, `CURRENT`, staging/generations/publishing)   |
| `store/`          | `PackageStore` (`ReadAt` / `WritePart` / `PublishGeneration`): `LocalStore` + `MemoryStore` (THI-469) |
| `shard.rs`        | `TetShardWriter` / `PartMeta` — immutable staging parts for concurrent ingest    |

## Codecs and dtypes

- **LE wire + span helpers:** [`argus-chunk`](https://docs.rs/argus-chunk) (`LeReader` / `LeWriter`, align8, byte-span checks); payload encode/decode via `argus::encode` / `argus::decode`
- **Payload codecs:** `raw` (0), `zstd` (1) — `ChunkPayloadCodecV1::encode_tile_payload` / `decode_tile_payload`
- **Element dtypes:** wire tags 1–10 (`f32` … `u64`) — `DATASET_DTYPE_TAG_V1`, `ElementDtype` in `utils`

## Write paths (choose one)

```text
In-memory arrays      → write.rs / session::commit
Streaming tiles       → stream_write.rs / session::commit_with_fill
Append to sealed file → append.rs
Package multi-writer  → package::TetPackage + shard::TetShardWriter (staging only)
```

## Read path

```text
mmap → layout::read_superblock_v1 → dataset blob → index::parse_chunk_index → optional THST
```

Chunk payload bytes are **not** decoded here during summary read; decoding happens in [`query/decode`](../query/decode/README.md) during query execution.

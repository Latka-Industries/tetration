# Tetration on-disk layout — version 1

This document describes the **v1** `.tet` container: a fixed **32-byte superblock** at offset 0, followed (when `dataset_count > 0`) by a **dataset directory blob**, an **8-byte-aligned** **chunk index** region, and **raw chunk payloads**.

## File map

Byte offsets increase left → right. All integers are **little-endian**.  
`align8(n) = (n + 7) & !7` (round up to an 8-byte boundary).

### Regions at a glance

| #   | Region                | Starts at                                                | Size                                              | When present                                             |
| --- | --------------------- | -------------------------------------------------------- | ------------------------------------------------- | -------------------------------------------------------- |
| 1   | **Superblock**        | `0`                                                      | `32` B fixed                                      | always                                                   |
| 2   | **Dataset directory** | `32`                                                     | `8 + dataset_blob_len`                            | `dataset_count > 0`                                      |
| 3   | **Padding**           | `40 + dataset_blob_len`                                  | `0 … 7` B                                         | only bytes needed so the next region is 8-byte aligned   |
| 4   | **Chunk index**       | `chunk_index_offset` (= `align8(40 + dataset_blob_len)`) | `chunk_index_length` (= `32 + entry_count × 104`) | `dataset_count > 0`                                      |
| 5   | **Chunk payloads**    | each `payload_offset` from index                         | each `stored_byte_len`                            | one span per index row; may be non-contiguous in general |

**Empty file** (`dataset_count = 0`): regions 2–5 are absent; `chunk_index_offset = 32`, `chunk_index_length = 0`; the file may end at byte 32.

**Populated file**: the reference writer packs payloads after the index, but v1 only requires `payload_offset + stored_byte_len ≤ file_len` for every row.

**How pieces connect:** the superblock’s `chunk_index_offset` / `chunk_index_length` bound region ④. Each index row’s `payload_offset` points at one payload span in region ⑤. Dataset metadata in region ② supplies `shape`, `chunk_shape`, and `dtype` for interpreting those bytes (see [query engine](query_engine.md) for read planning).

## Endianness and alignment

- All multi-byte integers are **little-endian** (`u32`, `u64`).
- The superblock is **32 bytes**. Chunk index entries and payload offsets assume **8-byte alignment** for index base offsets computed from the dataset directory.

## Magic and `layout_version`

- Bytes `0..4` must be ASCII **`TETR`**.
- Bytes `4..8` are **`layout_version`** (`u32` LE). Only **`1`** is defined today.

This matches the optional `layout_version` field in JSON/TOML query documents (`tetration::query::QueryDocument`).

## Superblock v1 (32 bytes)

| Offset | Size | Type     | Field                | Notes                                                                           |
| ------ | ---- | -------- | -------------------- | ------------------------------------------------------------------------------- |
| 0      | 4    | `[u8;4]` | `magic`              | Must be `TETR`.                                                                 |
| 4      | 4    | `u32`    | `layout_version`     | Must be `1`.                                                                    |
| 8      | 4    | `u32`    | `dataset_count`      | Number of dataset records in the directory blob; **0** means no directory blob. |
| 12     | 4    | `u32`    | `flags`              | Bit **`1`**: optional **history footer** at EOF (see below). Otherwise **0**.   |
| 16     | 8    | `u64`    | `chunk_index_offset` | Byte offset to the **chunk index** region (see below).                          |
| 24     | 8    | `u64`    | `chunk_index_length` | Length in bytes of the chunk index region.                                      |

### Empty file (`dataset_count = 0`)

- `chunk_index_offset = 32`
- `chunk_index_length = 0`
- The file may end at byte 32 (no trailing bytes required).

Readers must ensure `chunk_index_offset + chunk_index_length` fits in the file and does not overflow.

**Typical populated file** (contrast with empty above): superblock with `dataset_count = N`, dataset records at offset 40, chunk index at `align8(40 + dataset_blob_len)`, payloads referenced by index rows.

## Dataset directory (only when `dataset_count > 0`)

When `dataset_count > 0`, bytes starting at offset **32** are:

| Offset | Size | Field              | Notes                                                                                   |
| ------ | ---- | ------------------ | --------------------------------------------------------------------------------------- |
| 32     | 8    | `dataset_blob_len` | Length in bytes of the concatenated **dataset records** that immediately follow.        |
| 40     | \*   | `dataset_records`  | Total size = `dataset_blob_len`. Must contain exactly `dataset_count` records in order. |

### Dataset record (concatenated, variable length per record)

Each record is:

| Field         | Type            | Notes                                                                                                                                                                                                  |
| ------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `name_len`    | `u32` LE        | Byte length of UTF-8 `name` that follows.                                                                                                                                                              |
| `dtype`       | `u32` LE        | `1` = **`f32`**, `2` = **`f64`**, `3` = **`i32`**, `4` = **`i64`**, `5` = **`u8`**, `6` = **`u16`**, `7` = **`i16`**, `8` = **`u32`**, `9` = **`f16`**, `10` = **`u64`**, row-major within each chunk. |
| `ndim`        | `u32` LE        | Rank in `1 … 8`.                                                                                                                                                                                       |
| `reserved`    | `u32` LE        | Write **0**.                                                                                                                                                                                           |
| `name`        | `[u8]`          | UTF-8 of length `name_len`.                                                                                                                                                                            |
| _padding_     | `0…7`           | Zero bytes so the **total byte length from the start of this record** to the first byte of `shape[0]` is a multiple of **8**.                                                                          |
| `shape`       | `ndim × u64` LE | Global array shape.                                                                                                                                                                                    |
| `chunk_shape` | `ndim × u64` LE | Chunk size along each axis (must tile `shape`; `write_one_chunk_raw_file` requires a single tile `chunk_shape == shape`; `write_raw_array_file` supports regular multi-chunk grids).                   |

Records are concatenated in catalog order; `dataset_id` in the chunk index is the **0-based index** into this list (`0` = first record).

**Booleans** are not a separate wire tag: import maps HDF5 `Boolean`, Zarr `bool`, and similar to **`u8`** (0/1). Layout v2 is reserved for changes that cannot extend v1.

### Axis metadata (Phase 7 baseline)

v1 **dataset records** carry **`shape`** and **`chunk_shape`** only. **Dimension names**, **coordinate labels**, and **CF-style attrs** live in the optional **`THST` footer** under `metadata.datasets[<name>]` (see [Optional history footer](#optional-history-footer-v1-extension)). Convert and [`TetWriterSession`](../src/catalog/session.rs) populate this on write; the query engine resolves `"mean": "time"` and `selection` `start_label` / `stop_label` from footer metadata at plan time.

Two **separate** layers:

| Layer                 | What it names                               | Count                 | Example (3D weather)                 | Example (2D table)                |
| --------------------- | ------------------------------------------- | --------------------- | ------------------------------------ | --------------------------------- |
| **Dimension names**   | Each **axis** (the “major” row/column role) | **`ndim` strings**    | `time`, `lat`, `lon`                 | `row`, `column`                   |
| **Coordinate labels** | Each **position** along one axis            | **`shape[d]` values** | timestamps, latitudes, station codes | row 0 = `Alice`, col 2 = `salary` |

Do **not** conflate them:

- **Dimension name** — “axis 0 is called **time**” → enables `"mean": "time"` (JSON/TOML) instead of `"mean": 0`.
- **Coordinate label** — “index 42 along **time** is **2024-03-15**” → enables slice/filter by value, alignment across datasets, and (with extra ops) group-by keys.

**Analogues:** NetCDF dimension name vs coordinate variable; pandas `Index.name` vs `Index` values; xarray `dims` vs `coords`.

**Storage (shipped):** footer JSON `metadata` (inline or `metadata_spill` when &gt; 64 KiB inline budget). Per-dataset shape:

```json
{
  "datasets": {
    "temperature": {
      "dim_names": ["time", "station"],
      "coords": {
        "time": { "labels": ["2024-01-01", "2024-01-02"] },
        "station": { "labels": ["A", "B"] }
      },
      "attrs": { "units": "K", "long_name": "surface temperature" }
    }
  }
}
```

- **`dim_names`** — O(`ndim`) strings; enables `"mean": "time"` in query JSON/TOML.
- **`coords`** — O(`shape[d]`) inline label vectors per axis (limits in [`MetadataLimitsV1`](../src/catalog/metadata.rs)); large coords may spill via `metadata_spill`.
- **`attrs`** — scalar metadata (`units`, `_FillValue`, `missing_value`, …); used by `null_count` and `tet info`.

**Coordinate labels ≠ a query index by default.** Storing labels enables **name → integer index** resolution at plan time. Fast **filter / group-by** on high-cardinality keys may additionally need an auxiliary lookup structure — a separate layout/query decision.

See [query engine — dimension names vs coordinates](query_engine.md#dimension-names-vs-coordinate-labels).

## Chunk index region

- Starts at `chunk_index_offset`, which **must** equal `align8(40 + dataset_blob_len)` when `dataset_count > 0`.
- `align8(n) = (n + 7) & !7`.

### Chunk index header (32 bytes)

| Offset | Size | Field                       | Notes                                                                     |
| ------ | ---- | --------------------------- | ------------------------------------------------------------------------- |
| 0      | 4    | magic                       | ASCII **`TIDX`**.                                                         |
| 4      | 4    | `index_version`             | `u32` LE; must be **1**.                                                  |
| 8      | 8    | `entry_count`               | Number of fixed-size entries.                                             |
| 16     | 2    | `memory_budget_percent_bps` | `u16` LE; basis points (10000 = 100%). **0** = engine default (25%).      |
| 18     | 2    | reserved                    | Write **0**.                                                              |
| 20     | 4    | `memory_budget_bytes`       | `u32` LE fixed RAM cap for dense decode; **0** = use percent of host RAM. |
| 24     | 8    | reserved                    | Write **0**.                                                              |

These fields map to [`FileExecutionSettingsV1`](../src/catalog/execution.rs). Writers set them via **`RawArrayWrite::file_execution`**; readers surface them on **`catalog.file_execution`** in query responses. Query JSON **`execution.memory_budget_*`** overrides file defaults at runtime (see [query engine — memory budget](query_engine.md#memory-budget-and-execution-strategies)). Basis points: **10000 = 100%** (engine default when both are zero: **2500 = 25%** of detected host RAM).

The total chunk index byte length must be exactly:

`32 + entry_count * entry_wire_len`

### Chunk index entry (fixed `entry_wire_len` bytes, LE)

Layout (see `ChunkIndexEntryV1::WIRE_LEN` in `src/catalog/index.rs`):

- `dataset_id`: `u64`
- `chunk_index[8]`: eight `u64` values — chunk grid coordinates `i0 … i7`; unused slots are **0** (rank comes from the dataset record’s `ndim`).
- `payload_offset`: `u64` — file offset to **stored** bytes for this chunk.
- `raw_byte_len`: `u64` — logical uncompressed size of the chunk tensor in bytes.
- `stored_byte_len`: `u64` — bytes on disk at `payload_offset`.
- `codec`: `u32` — **`0`** = raw copy (`stored_byte_len` **must** equal `raw_byte_len`). **`1`** = **zstd**–compressed payload at `payload_offset`; `stored_byte_len` is the compressed size on disk and **`raw_byte_len`** is the uncompressed tensor size in bytes after decode.
- `reserved`: `u32` — **0**.

Each entry is **104 bytes** on the wire (`ChunkIndexEntryV1::WIRE_LEN`).

Payload bytes must lie fully inside the file: `payload_offset + stored_byte_len ≤ file_len`.

**Index on disk:** `TIDX` header (32 B) then `entry_count` fixed-size rows (104 B each). See field list above.

**Reader resolution** (how a query reaches bytes; details in [query engine](query_engine.md)):

```mermaid
flowchart LR
    Q["dataset name"] --> D["dataset_id"]
    D --> G["chunk coords"]
    G --> L["index row"]
    L --> R["mmap payload"]
```

Each index row’s `payload_offset` selects the byte span in region ⑤.

### Contiguous raw payloads (query fold hint)

For a **full dense** selection over a dataset, the query engine may treat all touched **raw** (`codec = 0`) payloads as one logical byte stream when:

1. Each chunk’s `payload_offset + raw_byte_len` equals the next chunk’s `payload_offset` (in read-plan order).
2. The sum of `raw_byte_len` equals the logical selection size in bytes.

When that holds and the working set is **out-of-core** relative to host RAM, tier-A/B scalar folds can use **linear scan** — sequential 64 MiB windows over the hyperslab — instead of parallel per-chunk mmap. See [query engine — adaptive fold I/O](query_engine.md#adaptive-fold-io). **Zstd** chunks (`codec = 1`) or non-contiguous payload layout still use per-chunk decode/fold.

Reference writers (`write_raw_array_file`) append payloads sequentially, so converted multi-chunk grids typically satisfy (1)–(2) for full-file scans.

### Per-chunk payload codecs

| Codec      | `stored_byte_len` vs `raw_byte_len`         | On-disk bytes                                                |
| ---------- | ------------------------------------------- | ------------------------------------------------------------ |
| **0** raw  | must be equal                               | tensor payload (LE bytes for dataset `dtype`, tags `1`–`10`) |
| **1** zstd | `stored` = compressed, `raw` = decoded size | zstd frame; decode to `raw_byte_len` bytes                   |

See also [`query_engine.md`](query_engine.md) for how the query engine materializes planned chunks.

## Optional history footer (v1 extension)

When superblock **`flags & 1`**, the file may end with a self-describing footer **after** all chunk payloads:

| Region (suffix at EOF) | Size     | Notes                                                                                                                                                                                        |
| ---------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `history_json`         | variable | UTF-8 JSON object: `{"history":[…], "metadata":{…}}` (optional keys). See below.                                                                                                             |
| `metadata_spill`       | variable | Optional raw UTF-8 JSON for `metadata` when inline `history_json` would exceed **64 KiB**; referenced by `metadata_ref` in `history_json`. Lies after chunk payloads, before `history_json`. |
| `history_json_len`     | 8        | `u64` LE byte length of `history_json`.                                                                                                                                                      |
| `history_version`      | 4        | `u32` LE; must be **1**.                                                                                                                                                                     |
| magic                  | 4        | ASCII **`THST`**.                                                                                                                                                                            |

Chunk payload validation uses **`file_len − footer_suffix`** (footer JSON + tail + optional `metadata_spill`). Readers without history support ignore the flag and treat the full file length as payload bounds only when the flag is **0**. `tet info` / [`read_tet_summary_v1`](../src/catalog/mod.rs) surface parsed `history` and resolved `metadata`.

**`history` rows** — each entry is a JSON object `{ "op", "source", "at", "parents"?, "params"? }`. Legacy files may use a three-element array `[op, source, at]`; readers accept both. New writers emit objects only.

**`metadata`** — inline object when small (see [`MetadataLimitsV1`](../src/catalog/metadata.rs)). Shape: optional `file` attrs plus `datasets` map keyed by catalog name; each entry may include `dim_names`, `coords` (per-axis `labels`), and `attrs`. When too large, writers spill UTF-8 JSON to `metadata_spill` and set `"metadata_ref": { "offset", "len" }` in `history_json` (mutually exclusive with inline `metadata`).

## Reference subset (current Rust writer)

The `write_one_chunk_raw_file` helper in `tetration::catalog` writes exactly **one** dataset and **one** chunk: `chunk_shape` must equal `shape` so the chunk grid has a single tile; payloads are always **raw** (`codec = 0`). **`dtype`** may be any supported element tag (`1`–`10`).

`write_raw_array_file` / `RawArrayWrite` accept per-chunk **`chunk_codec`**: compare to **`CHUNK_PAYLOAD_CODEC_V1.raw`** (**0**, raw tiles) or **`CHUNK_PAYLOAD_CODEC_V1.zstd`** (**1**, zstd-compressed frames; `stored_byte_len` may differ from `raw_byte_len`). **`dtype`** may be any supported element tag (`1`–`10`). Optional **`file_execution`** writes TIDX execution settings (memory budget). The Rust API exposes this as the `ChunkPayloadCodecV1` struct plus the `CHUNK_PAYLOAD_CODEC_V1` constant in `tetration::catalog`; decode symmetry via **`ChunkPayloadCodecV1::decode_tile_payload`**.

## Concurrency (informative)

**Single-file layout v1** has **no** on-disk locking, lease, or live-append protocol on one `.tet`. Scale for a sealed file comes from **immutable bytes + mmap readers**.

| Pattern                               | Supported?     | Notes                                                                                                                                |
| ------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Sealed file, many readers**         | Yes (intended) | After `commit` / `tet convert` completes, any number of hosts may mmap read-only and query. Chunk index + payload offsets are fixed. |
| **One writer, then readers**          | Yes            | Reference writers (`write_raw_array_file`, [`TetWriterSession`](../src/catalog/session.rs)) produce a stable byte image.             |
| **Parallel writers on one file**      | No (undefined) | Exclusive create or a single coordinated writer; parallel payload append without a spec can corrupt the index.                       |
| **Read while another process writes** | No (undefined) | Readers may see torn index rows or short payloads.                                                                                   |
| **Many queries, one file**            | Yes            | Independent plans; per-query temp spills are unique; user export `spill` paths must not collide.                                     |

**Multi-writer (0.2+):** concurrent ingest is specified as **immutable shards → single publish → sealed generation**, not as parallel updates to one live TIDX. See [`writer_protocol_v2.md`](writer_protocol_v2.md). A published generation exposes a packed layout-v1 `.tet` for the same mmap reader contract as above.

### Package readers

Normative when the bytes live in a package (local `.tet.d` or a store key), not in a standalone `.tet` (THI-472).

| Event | What the open query does |
| ----- | ------------------------ |
| `CURRENT` moves from `G` to `G+1` | Keeps reading `generations/<G>/packed.tet`. The next open follows `G+1`. |
| Someone replaces the bytes at the pinned `packed.tet` key | Does **not** treat that as a new generation and does **not** re-read `CURRENT`. Local mmap stays on the inode from open. An object-store overwrite can show up on a later range GET; that is a verify failure on `G`. |

Writers publish a new generation under a new key. They do not rewrite a sealed `packed.tet`. Detail: [`query_engine.md` — Pin / consistency](query_engine.md#pin--consistency).

Optional generation sidecars are not regions of this file. `ZONE.json` (0.4 zone maps) and `INDEX.json` (value postings) live beside `packed.tet` under `generations/<G>/`. They are not a superblock flag, dataset-directory field, or chunk-index change. `layout_version` stays **1**. A reader that only opens `packed.tet`, including 0.4.0–0.4.4, still opens the package: `GenerationManifest` ignores unknown keys, and a standalone `.tet` has no index. `INDEX.json` exists only when the manifest says `secondary_indexes: "present"`. Verify checks that blob by hash and, with `--deep`, against a recompute from the sealed payloads. A missing or bad index is not a layout break; query falls back to zone maps. Detail: [`secondary_indexes.md`](secondary_indexes.md).

**Inside one query**, the engine already parallelizes chunk I/O on CPU when in-core (Rayon) or uses a sequential linear scan when out-of-core ([query engine — adaptive fold I/O](query_engine.md#adaptive-fold-io)).

Full deployment notes: [README — Concurrency and scale](../README.md#concurrency-and-scale).

## File health (verify / repair)

**`tet verify`** (library: [`verify_tet_file`](../src/verify/mod.rs)) checks superblock, dataset directory, chunk index, payload bounds, optional decode integrity (quick scan: first **128** chunks on large files; **`--deep`** / `VerifyOptions::deep_decode` for all chunks), dataset tensor-byte cross-check, and footer/metadata limits.

**`tet repair`** ([`repair_tet_file`](../src/repair/mod.rs)) applies safe in-place fixes from verify recommendations (today: **`footer_invalid`** — truncate after chunk payloads when the `THST` JSON is corrupt).

Tracked manual smoke files: [`fixtures/small/tet/README.md`](../fixtures/small/tet/README.md). Regenerate: `mise run fixtures:small-tet` or `cargo run --example gen_small_tet_fixtures`.

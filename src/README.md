# `src/` — crate layout

The **tetration** library and **`tet`** CLI live here. Public embedder API is re-exported from [`lib.rs`](lib.rs) and [`prelude`](lib.rs); most logic is split by concern below.

## Top-level modules

| Module                          | Role                                                     | CLI                      |
| ------------------------------- | -------------------------------------------------------- | ------------------------ |
| [`layout.rs`](layout.rs)        | 32-byte superblock, mmap helpers                         | —                        |
| [`catalog/`](catalog/README.md) | Dataset directory, chunk index, codecs, writers, readers | `tet info`               |
| [`query/`](query/README.md)     | JSON/TOML queries: plan, decode, fold, execute           | `tet query`, `tet qhist` |
| [`convert/`](convert/README.md) | HDF5 / NetCDF / Zarr / NumPy / CSV / Parquet → `.tet`    | `tet convert`            |
| [`export/`](export/README.md)   | `.tet` → Zarr v3 / HDF5 / NetCDF                         | `tet export`             |
| [`verify/`](verify/README.md)   | Read-only file health checks                             | `tet verify`             |
| [`repair/`](repair/README.md)   | In-place fixes from verify recommendations               | `tet repair`             |
| [`ffi/`](ffi/README.md)         | Optional C ABI (`tetration-ffi` feature)                 | —                        |
| [`utils/`](utils/README.md)     | Wire dtypes, endian helpers, host RAM probes             | —                        |
| [`bin/`](bin/README.md)         | `tet` binary entrypoint and subcommands                  | all `tet *`              |
| [`tests/`](tests/README.md)     | Integration tests (not public API)                       | —                        |

## `layout.rs` — on-disk superblock (v1)

Single-file module at [`layout.rs`](layout.rs) (not a directory). Defines the **first 32 bytes** of every `.tet` file: magic (`TETR`), layout version, dataset count, flags, and chunk-index region offset/length. Everything after the superblock is owned by [`catalog`](catalog/README.md) (see [`docs/layout_v1.md`](../docs/layout_v1.md)).

| Type / fn                                    | Role                                                |
| -------------------------------------------- | --------------------------------------------------- |
| `SuperblockV1`, `MAGIC`, `SUPERBLOCK_V1_LEN` | Parsed header; `TETR` magic, 32-byte length         |
| `SUPERBLOCK_FLAG_HISTORY_FOOTER`             | Optional `THST` JSON footer at EOF                  |
| `read_superblock_v1` / `write_superblock_v1` | Parse or serialize the header                       |
| `mmap_file_read`                             | Read-only mmap (used by `TetFile::open` and verify) |

Writers set superblock fields when sealing a file (`catalog::write`, `stream_write`, `append`). Readers always start here before the dataset directory (`catalog::read_tet_summary_v1`).

## Typical data flow

```text
Foreign file ──convert──► .tet file ◄──mmap── TetFile / verify / query
                              │
                              ├── catalog: superblock + datasets + TIDX + payloads (+ THST footer)
                              └── query: document → ReadPlan → decode → fold / transform → response
```

## External docs

- On-disk bytes: [`docs/layout_v1.md`](../docs/layout_v1.md)
- Query wire + execution: [`docs/query_engine.md`](../docs/query_engine.md)
- `tet` CLI: [`docs/cli.md`](../docs/cli.md)
- C ABI: [`docs/ffi.md`](../docs/ffi.md)
- User docs: [docs.tetration.sh](https://docs.tetration.sh)
- Published Rust API: [docs.rs/tetration](https://docs.rs/tetration)

## `prelude` quick path

```rust
use tetration::prelude::*;
// TetWriterSession, TetFile, parse_query_json, execute_query_json, verify_tet_file, …
```

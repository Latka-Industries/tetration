# Tetration

[![Crates.io](https://img.shields.io/crates/v/tetration.svg)](https://crates.io/crates/tetration)
[![docs.rs](https://img.shields.io/docsrs/tetration)](https://docs.rs/tetration)
![Build](https://github.com/Latka-Industries/tetration/workflows/Build/badge.svg)
![Rust](https://img.shields.io/badge/rust-1.95-orange.svg)

[_For those who are more cur..._](https://bookshop.org/p/books/book-of-numbers-a-novel-joshua-cohen/af5aa739b0fac506?ean=9780812986655&next=t)

**_STILL IN DEVELOPMENT — layout v1 and query JSON/TOML may change before 1.0._**

**HDF5-shaped** persistence (many large arrays in one durable file), **Zarr-shaped** chunking (regular grid, per-chunk compression, parallel I/O)—in a **single mmap-friendly `.tet` file`**, not a directory of shard blobs.

## What it does today (v1)

- **On-disk layout** — superblock, dataset directory, chunk index, raw or zstd payloads ([`docs/layout_v1.md`](docs/layout_v1.md)).
- **Mmap + read planning** — logical slices → chunk coordinates → [`ReadPlan`](https://docs.rs/tetration/latest/tetration/query/struct.ReadPlan.html).
- **JSON / TOML query + execute** — flat query documents (paired examples in [`fixtures/queries/`](fixtures/queries/)), streaming reductions, tier-C stats, spill export, **`transform`** (zscore, minmax, l1/l2, center, scale, log1p, sqrt, softmax) with **`write`** routing (`ram`, `spill`, `switch`, **`sidecar`** `.tet` publish), **`nan_mean`** / **`nan_std`**; **named axes**, **coord label** selection, QC counts (`nan_count`, `null_count`, `inf_count`, `any_inf`), **covariance** / **correlation** ([`docs/query_engine.md`](docs/query_engine.md)).
- **Import / export** — `tet convert` from HDF5, NetCDF, Zarr v3; **`tet export`** back to Zarr v3 (stored chunk bytes, nested groups).
- **File health** — `tet verify` (quick scan; **`--deep`** decodes every chunk), `tet repair` (plan / `--apply` safe fixes).
- **CLI** — `tet info`, `tet verify`, `tet repair`, `tet query`, `tet qhist`, `tet convert`, `tet export`.
- **Optional GPU (Phase 10, experimental)** — `execution.device` / `tet query --device` for tier-A/B **`f32`** (and **`f16`** on device); Metal (`tetration-metal`, macOS), CUDA (`tetration-gpu`), streaming fold + multi-GPU when host RAM does not fit a dense buffer. CPU streaming fold remains the default for large selections.
- **C ABI** — feature **`tetration-ffi`**, [`include/tetration.h`](include/tetration.h), [`docs/ffi.md`](docs/ffi.md).

**Wire dtypes** (tags `1`–`10`, row-major chunks): **`f32`**, **`f64`**, **`i32`**, **`i64`**, **`u8`**, **`u16`**, **`i16`**, **`u32`**, **`f16`**, **`u64`**. Booleans import as **`u8`**. See [`docs/layout_v1.md`](docs/layout_v1.md#dataset-record-concatenated-variable-length-per-record).

## Quick start

### macOS — Homebrew (recommended)

One-time tap (this repo ships `Formula/tetration.rb`; pulls in **HDF5** and **NetCDF** for `tet convert`):

```bash
brew tap Latka-Industries/tetration https://github.com/Latka-Industries/tetration
brew install tetration
tet --help
```

Upgrade later: `brew upgrade tetration`.

From a local clone (no tap): `brew install --build-from-source Formula/tetration.rb`

### `cargo install`

**Default features** need **system HDF5 and NetCDF** dev libraries (`.h5` / `.nc` convert; Zarr v3 is Rust + bundled **zstd**):

| Platform             | Typical packages                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Debian / Ubuntu**  | `libhdf5-dev`, `libnetcdf-dev`, `pkg-config`, `build-essential`                                                                 |
| **macOS (Homebrew)** | `brew install hdf5 netcdf pkg-config`                                                                                           |
| **Windows**          | OpenSSL + NetCDF/HDF5 (e.g. [vcpkg](https://vcpkg.io/) or conda-forge); see [`.github/scripts/`](.github/scripts/) for CI hints |

```bash
cargo install tetration
```

Without HDF5/NetCDF libs: **`cargo install tetration --no-default-features`** — `tet info` / `tet query` on `.tet` files and **Zarr** import still work.

### Build from source

```bash
git clone https://github.com/Latka-Industries/tetration.git
cd tetration
cargo build --release
export PATH="$PWD/target/release:$PATH"   # or: alias tet="$PWD/target/release/tet"
```

### First commands

```bash
tet convert volume.h5 volume.tet          # HDF5 / NetCDF / Zarr v3 → .tet
tet export volume.tet volume.zarr/      # .tet → Zarr v3 directory (empty or new dir)

tet info volume.tet
tet verify volume.tet
tet verify --deep volume.tet -q    # full chunk decode (large files sample 128 by default)
tet query fixtures/queries/mean_temperature.toml -t volume.tet -x -q   # after convert; <name> from info
tet query '{"dataset":"<name>","inf_count":[]}' -t volume.tet -x -q
```

**Daily driver:** plan + execute with readable stdout:

```bash
tet query fixtures/queries/mean_temperature.toml -t data.tet -x -q
tet query q.json -t data.tet -x --format stats              # slim JSON (no chunk list)
tet query q.toml -t data.tet -x --format table --preview 6  # ASCII tables + slice grid
tet query q.json -t data.tet --format plan                  # catalog + read_plan only
tet query '{"dataset":"a","transform":{"method":"zscore"},"write":{"target":"sidecar","timestamp":false}}' -t data.tet -x -q
```

Query documents are **flat** JSON or TOML (e.g. `"mean": []` / `mean = []`, `"spill": "slice.bin"`); nested `"operation"` objects are rejected. Details: [query document](docs/query_engine.md#query-document-json-and-toml).

## `tet` commands

[`docs/cli.md`](docs/cli.md) — `tet info`, `tet verify`, `tet repair`, `tet query`, `tet qhist`, `tet convert`, `tet export` (flags and subcommands). Live help: **`tet -h`** and **`tet <command> -h`**.

## Documentation map

| Doc                                                | Contents                                                                                                                                             |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`docs/cli.md`](docs/cli.md)                       | **`tet`** commands: flags and subcommands (`info`, `verify`, `repair`, `query`, `qhist`, `convert`, `export`)                                        |
| [`docs/layout_v1.md`](docs/layout_v1.md)           | On-disk **layout v1**: superblock, dataset catalog, chunk index, codecs, footer metadata/history, concurrency                                        |
| [`docs/query_engine.md`](docs/query_engine.md)     | **Query** JSON/TOML wire, planning, fold/spill execution, optional GPU, JSON security                                                                |
| [`docs/ffi.md`](docs/ffi.md)                       | **C ABI** (`tetration-ffi`): [`include/tetration.h`](include/tetration.h), linking, [`examples/ffi_query.c`](examples/ffi_query.c), release archives |
| [docs.rs / `tetration`](https://docs.rs/tetration) | Rust crate API (`prelude`, `TetFile`, `execute_query_json`, convert, verify, …)                                                                      |
| [**tet-py**](https://github.com/Latka-Industries/tet-py) | **Python** bindings (`pip install tet-py`, `import tet`) — NumPy read/write, query reductions, transform ram/spill/sidecar; pins **tetration 0.1.9** |

## Design stance (short)

**Partial I/O is the default case** — mmap payload regions, touch only chunks that intersect the selection, parallel decode across disjoint tiles. Full-array loads into RAM are not required for planning or tier-A/B aggregates.

**JSON/TOML is the control plane**, not the storage encoding: hosts validate input, cap size, and enforce spill path policy ([security notes](docs/query_engine.md#json-security-input-and-output)).

## Concurrency and scale

**Read-many / write-once** is the supported scale model for v1:

| Role          | Contract                                                                                                                                                                                                                                                                            |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Writer**    | One process (or coordinated [`TetWriterSession`](https://docs.rs/tetration/latest/tetration/catalog/struct.TetWriterSession.html) / `tet convert`) finishes the file before readers rely on it. v1 defines **no** file locking or live append protocol.                             |
| **Reader**    | Any number of processes or hosts may **mmap read-only** the same sealed `.tet` and run independent queries. The OS shares cold pages via the page cache; each query touches only chunks in its [`ReadPlan`](https://docs.rs/tetration/latest/tetration/query/struct.ReadPlan.html). |
| **Per query** | Tier-A/B folds merge **chunk-local** partials (parallel Rayon when in-core; linear scan when out-of-core). Temp spills use unique paths (`pid` + timestamp); export **`spill`** paths must differ per worker.                                                                       |

**Not supported without extra coordination:** multiple writers on one file, read-while-write, or two workers writing the same export spill path.

**CPU workers:** scale out with **N processes × independent queries** (or datasets), not by sharding one query inside the engine today. **Phase 10 GPU** uses the same chunk-parallel shape: dense materialize when RAM allows, else **streaming device fold** ([`gpu/streaming_fold.rs`](src/query/gpu/streaming_fold.rs)); **`cuda:multi` / `rocm:multi`** shard chunks across devices — see [query engine — scalability](docs/query_engine.md#scalability-read-many-and-phase-10).

Wire details: [`docs/layout_v1.md` — Concurrency](docs/layout_v1.md#concurrency-informative).

**Non-goals (v1):** SQL-on-files, arbitrary codec plugins, GPU codecs in the file format. Optional GPU: [`docs/query_engine.md`](docs/query_engine.md#phase-10--optional-gpu-experimental).

## Library use

```toml
[dependencies]
tetration = "0.1.9"
```

```rust
use tetration::prelude::*;
// TetWriterSession, TetFile, parse_query_json, parse_query_toml, execute_query_json, verify_tet_file, …
```

### Embedder flow

1. **Write** — `TetWriterSession::create` → `push_dataset` → `commit()` (or `commit_with_fill` for streaming).
2. **Read / aggregate** — `TetFile::open` → `execute_query_json` → [`QueryResponse`](https://docs.rs/tetration/latest/tetration/query/struct.QueryResponse.html).
3. **Dense tensors (no preview cap)** — mmap the `.tet`, then [`materialize_query_selection`](https://docs.rs/tetration/latest/tetration/query/fn.materialize_query_selection.html) (selection-only) or [`materialize_query_transform_ram`](https://docs.rs/tetration/latest/tetration/query/fn.materialize_query_transform_ram.html) (`transform` + `write: ram`) → [`DenseMaterializeOutcome`](https://docs.rs/tetration/latest/tetration/query/struct.DenseMaterializeOutcome.html). See [query engine — embedder dense export](docs/query_engine.md#embedder-dense-export).

```bash
cargo run --example create_and_query
cargo run --example session_write
```

### Query input: JSON or TOML

Flat JSON and TOML profiles compile to the same [`QueryDocument`](https://docs.rs/tetration/latest/tetration/query/struct.QueryDocument.html). `tet query` accepts `.json` / `.toml` paths, inline text, or stdin; leading `{` selects JSON, otherwise TOML (extension overrides).

```json
{ "dataset": "temperature", "mean": [] }
```

```toml
dataset = "temperature"
mean = [] # scalar reduction
```

Library: [`parse_query_json`](https://docs.rs/tetration/latest/tetration/query/fn.parse_query_json.html), [`parse_query_toml`](https://docs.rs/tetration/latest/tetration/query/fn.parse_query_toml.html), [`parse_query_text`](https://docs.rs/tetration/latest/tetration/query/fn.parse_query_text.html) (auto-detect).

See **Documentation map** above for layout, query engine, FFI, fixtures, and [docs.rs](https://docs.rs/tetration).

### Python (`tet-py`)

Official bindings live in [**tet-py**](https://github.com/Latka-Industries/tet-py) (PyPI **`tet-py`**, **`import tet`** — not the unrelated PyPI package `tetration`). Current release **0.1.1** links **tetration 0.1.9** from [crates.io](https://crates.io/crates/tetration).

```bash
pip install tet-py
python -c "import tet; print(tet.__version__, tet.core_version())"
```

```python
import tet

with tet.open("data.tet") as f:
    print(f.mean("temperature"))
    arr = f.read_numpy("temperature")  # ram
    z = f.transform.to_numpy.zscore("temperature")
```

API reference and ops: [tet-py `docs/operations.md`](https://github.com/Latka-Industries/tet-py/blob/main/docs/operations.md). Open work is tracked in the [tet-py README](https://github.com/Latka-Industries/tet-py/blob/main/README.md#to-do).

---

## To do

### Shipped (v0.1.9)

- [x] Layout v1, query engine, `tet convert` / `tet export`, verify/repair, optional GPU, C ABI
- [x] [`tetration` on crates.io](https://crates.io/crates/tetration) **0.1.9**
- [x] Official Python bindings — [**tet-py**](https://github.com/Latka-Industries/tet-py) on PyPI (`import tet`)

### Near-term

- [ ] **Selection-size preflight** ([#19](https://github.com/Latka-Industries/tetration/issues/19)) — estimate logical selection bytes before `materialize_query_selection`
- [ ] **`DenseBuffer` export gaps** ([#20](https://github.com/Latka-Industries/tetration/issues/20)) — `f16`, `u32`, `u64` in `embed_materialize` (transform ram stays f32/f64)
- [ ] **Catalog write API** — publish `TetDatasetWrite::f64_row_major` and generic `row_major(dtype, …)` on crates.io
- [ ] **Sidecar / writer unification** — route transform `write: sidecar` through `TetWriterSession` ([`transform/sidecar.rs`](src/query/transform/sidecar.rs))
- [ ] **Coordinate filter-by-value / group-by** — label *slicing* is shipped; value filters and lookup indexes remain deferred ([`query_engine.md` — intentional gaps](docs/query_engine.md#intentional-gaps-v1))

### Docs & sites

- [x] Link tet-py from [`ffi.md`](docs/ffi.md) and [`query_engine.md`](docs/query_engine.md)
- [ ] [**tetration-docs**](https://github.com/Latka-Industries/tetration-docs) — GitHub Pages deploy + content (format spec, CLI ref, crate usage)
- [ ] **docs.rs examples** — match on-disk layout/query guarantees when the format stabilizes toward **1.0**
- [ ] **Response schema versioning** — document breaking `QueryResponse` changes ([`query_engine.md` — hardening](docs/query_engine.md#hardening-roadmap))

### Long-term / discussion

- [ ] **Format conversion roadmap** ([#17](https://github.com/Latka-Industries/tetration/issues/17)) — importer roadblocks, additional sources beyond HDF5/NetCDF/Zarr v3
- [ ] **Architecture phases** ([#16](https://github.com/Latka-Industries/tetration/issues/16)) — append protocol, distributed read, plugin codecs, etc.
- [ ] **1.0 API freeze** — layout v1 + query JSON/TOML wire stability; breaking changes called out in release notes until then

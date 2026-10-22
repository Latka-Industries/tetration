# Tetration

[![Crates.io](https://img.shields.io/crates/v/tetration.svg)](https://crates.io/crates/tetration)
[![Docs](https://img.shields.io/badge/docs-docs.tetration.sh-blue)](https://docs.tetration.sh)
[![docs.rs](https://img.shields.io/docsrs/tetration)](https://docs.rs/tetration)
[![CI](https://github.com/Latka-Industries/tetration/actions/workflows/ci.yml/badge.svg)](https://github.com/Latka-Industries/tetration/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rustc-1.95%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE-MIT)

[_For those who are more cur..._](https://bookshop.org/p/books/book-of-numbers-a-novel-joshua-cohen/af5aa739b0fac506?ean=9780812986655&next=t)

**Pre-1.0** — layout v1 and query JSON/TOML may change before 1.0. Current crates.io / PyPI line: **0.6.6** (pack milestone **0.6.3**). See [`CHANGELOG.md`](CHANGELOG.md).

HDF5-shaped persistence (many arrays in one file) with Zarr-shaped chunking (regular grid, per-chunk codecs, parallel I/O) in a **single mmap-friendly `.tet` file** — plus multi-writer **`.tet.d` packages** (shards → publish → sealed generation).

| | |
| --- | --- |
| **Docs** | [docs.tetration.sh](https://docs.tetration.sh) |
| **Crate** | [`tetration`](https://crates.io/crates/tetration) on crates.io · [docs.rs](https://docs.rs/tetration) |
| **CLI** | `tet` — Homebrew / `cargo install tetration` |
| **Python** | [`tet-py`](https://github.com/Latka-Industries/tet-py) · [`pip install tet-py`](https://pypi.org/project/tet-py/) |

## Install

**macOS (Homebrew):**

```bash
brew tap Latka-Industries/tetration https://github.com/Latka-Industries/tetration
brew install tetration
```

**Cargo** (default features need system HDF5 + NetCDF for `tet convert` / `tet export`; omit with `--no-default-features`):

```bash
cargo install tetration
```

**Library:**

```bash
cargo add tetration
```

S3-compatible package stores are optional and off by default: `cargo add tetration --features tetration-s3`.

## First commands

```bash
tet convert volume.h5 volume.tet
tet info volume.tet
tet verify volume.tet
tet query fixtures/queries/mean_temperature.toml -t volume.tet -x -q
tet export volume.tet volume.zarr/   # or volume.h5 / volume.nc
```

Packages (multi-writer): `tet ingest` → `tet publish` → query the `.tet.d` (or `tet pack` to a single `.tet`). See [CLI — packages](https://docs.tetration.sh/cli/package).

## Documentation

Prefer the site for product docs. The **normative contract** for third-party readers is [`SPEC/`](SPEC/README.md); `docs/*.md` hold the detailed tables that SPEC incorporates by reference.

| Topic | Site | Repo |
| ----- | ---- | ---- |
| **Normative SPEC (0.6.3 pack / crate 0.6.6)** | — | [`SPEC/`](SPEC/README.md) · [`COMPATIBILITY`](SPEC/COMPATIBILITY.md) · [`conformance/`](conformance/README.md) |
| Getting started | [docs.tetration.sh](https://docs.tetration.sh/getting-started) | — |
| Format / layout | [/format/](https://docs.tetration.sh/format/) | [`docs/layout_v1.md`](docs/layout_v1.md) |
| CLI | [/cli/](https://docs.tetration.sh/cli/) | [`docs/cli.md`](docs/cli.md) |
| Query engine | [/guides/query-engine/](https://docs.tetration.sh/guides/query-engine/) | [`docs/query_engine.md`](docs/query_engine.md) |
| Writer protocol v2 | [/cli/package](https://docs.tetration.sh/cli/package) | [`docs/writer_protocol_v2.md`](docs/writer_protocol_v2.md) |
| Rust / FFI | [/rust/](https://docs.tetration.sh/rust/) | [`docs/ffi.md`](docs/ffi.md) · [docs.rs](https://docs.rs/tetration) |
| Python (tet-py) | [/python/](https://docs.tetration.sh/python/) | [tet-py](https://github.com/Latka-Industries/tet-py) |
| Conversions | — | [`docs/conversions.md`](docs/conversions.md) |
| Roadmap | [/roadmap](https://docs.tetration.sh/roadmap) | [`docs/roadmap.md`](docs/roadmap.md) |
| Optional secondary indexes | — | [`docs/secondary_indexes.md`](docs/secondary_indexes.md) |
| Declared semantics + caller policy (0.5.1) | [`/guides/semantics/`](https://docs.tetration.sh/guides/semantics/) | [`docs/semantics.md`](docs/semantics.md) |
| Changelog | — | [`CHANGELOG.md`](CHANGELOG.md) |

## Library sketch

```rust
use tetration::prelude::*;
// TetWriterSession, TetFile, parse_query_json, execute_query_json, verify_tet_file, …
```

```bash
cargo run --example create_and_query
cargo run --example session_write
```

Python: [docs.tetration.sh/python](https://docs.tetration.sh/python/) · `pip install tet-py` (import `tet`, not PyPI `tetration`).

## License

**MIT OR Apache-2.0**.

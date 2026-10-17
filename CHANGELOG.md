# Changelog

## [Unreleased]

## [0.3.0] - 2026-10-17

### Added

- `PackageStore` trait (`ReadAt` / `WritePart` / `PublishGeneration`) with `LocalStore` (`.tet.d`) and `MemoryStore` (THI-469)
- URI parse + range-read planner: `parse_tet_uri`, `pin_packed_tet`, catalog/chunk `read_at` plans, store-backed query execute (THI-470)
- `TetPackagePin` / `pin_current` / `pin_generation` so readers keep one sealed generation for the life of a handle (THI-498)

### Changed

- Pedantic clippy: `NetCDF` doc backticks; case-insensitive `.tet` / `.tet.d` URI suffixes (left `cast_*` alone)

## [0.2.3] - 2026-10-16

### Added

- `tet export` writes HDF5 and NetCDF-4 beside Zarr v3 (`export_tet`, feature-gated like convert; THI-491)

## [0.2.2] - 2026-10-16

### Added

- `tet convert` reports skipped datasets (`ConvertReport.skipped`, stderr lines) and `--strict` fails before writing when any array was skipped (THI-485)
- `tet convert` imports NumPy `.npy` and multi-member `.npz` (THI-486)
- `tet convert` imports CSV and TSV, with text columns stored as integer category codes (THI-488, THI-492)
- `tet convert` imports Zarr v2 stores beside v3 (THI-490)
- `tet convert` imports Parquet behind the `tetration-parquet` feature (THI-489)
- `tet convert --chunk-mib N` retile imports to about N MiB along the leading axis (THI-487)

### Changed

- Convert rustdoc now matches those importers: Zarr v2, multi-member `.npz`, CSV categories, and Parquet

## [0.2.1] - 2026-10-15

### Added

- Package verify rejects a missing `CURRENT` when sealed `generations/` exist; finding-code tables in CLI / writer-protocol docs (THI-480)
- Mid-publish crash tests: torn `publishing/` and rename-before-`set_current` keep the prior generation readable
- cargo-fuzz target `verify_tet_package` (`mise fuzz-package` / `fuzz-package-smoke`; THI-481)

### Changed

- Split long verify, recommendation, and CLI dispatch functions (`clippy::too_many_lines`)

## [0.2.0] - 2026-10-15

### Added

- **Writer protocol v2** (local packages): `.tet.d` with `PROTOCOL` / `CURRENT` / `staging` / `generations` / `publishing` ([`docs/writer_protocol_v2.md`](docs/writer_protocol_v2.md); THI-462 / THI-468)
- [`TetPackage`](src/catalog/package.rs) — create/open, publish locks, seal generation, `pack_generation`, `next_generation`
- [`TetShardWriter`](src/catalog/shard.rs) / `PartMeta` — concurrent staging parts (THI-463)
- [`publish_parts`](src/catalog/shard.rs) / `merge_tet_files` — freeze part ids → packed layout-v1 → flip `CURRENT` (THI-464)
- [`ingest_tet_file`](src/catalog/shard.rs) — stage an existing `.tet` as a sealed part
- [`verify_tet_package`](src/verify/package.rs) — PROTOCOL / CURRENT / MANIFEST / packed SHA + layout checks; stale publish quarantine (THI-466)
- CLI: `tet ingest` / `tet publish` / `tet compact` (publish alias) / `tet pack`; `tet info` / `tet query -t` / `tet verify` accept `.tet.d` (THI-467)
- Repair code `quarantine_publish` for stale `publishing/<G>/` locks
- Package reject / proptest coverage (`src/tests/package_verify.rs`); CLI smoke ingest → publish → query → pack

### Changed

- Pedantic clippy nits outside `cast_*` (duration units, heap SHA buffer, SIMD `pointer::cast` / wildcard allows)

## [0.1.12] - 2026-10-15

### Added

- Selection-size preflight on `materialize_query_selection` (same `ExecutionBudget` resolution as transform `write: ram`) plus `estimate_query_selection_bytes` (THI-16 / [#19](https://github.com/Latka-Industries/tetration/issues/19))
- `DenseBuffer` variants **`F16` / `U32` / `U64`** on embedder materialize (THI-15 / [#20](https://github.com/Latka-Industries/tetration/issues/20))
- Transform `write: sidecar` persists the draft through [`TetWriterSession`](src/catalog/session.rs) (THI-61); `TetWriterSession::push_history` for full history rows

### Changed

- Pedantic clippy cleanups outside `cast_*` (GPU streaming fold args, catalog summary split, CUDA docs/`is_ok_and`)

## [0.1.11] - 2026-10-15

### Added

- **cargo-fuzz** harness over `verify_tet_bytes` (`fuzz/`, `mise fuzz` / `mise fuzz-smoke`)
- Structured **write → plan/decode** offset invariants (`proptest` in `src/tests/write_invariants.rs`)
- Conformance **reject** corpus for TIDX / payload faults (`fixtures/small/tet/reject/`, `src/tests/reject_corpus.rs`)
- Manual CI fuzz smoke workflow (`.github/workflows/fuzz-smoke.yml`, `workflow_dispatch` only)

### Changed

- Dropped redundant release **Build** job from the PR CI matrix (Test Suite + MSRV only)
- CI `paths-ignore` for markdown / LICENSE / `.gitignore` so doc-only pushes skip the matrix
- Docs: fuzz README, verify module README, changelog / README version pins

## [0.1.10] - 2026-10-13

### Added

- Catalog session writers: `TetDatasetWrite::f64_row_major` / `row_major(dtype, …)` and matching `TetDatasetStreamSpec` helpers for any supported wire dtype
- Session tests for `f64`, `i32`, streaming `f64`, and bad-dtype rejection
- `CHANGELOG.md`; homepage / docs links → [docs.tetration.sh](https://docs.tetration.sh)

### Changed

- GitHub Release workflow publishes non-draft releases (`draft: false`)
- `hdf5-metno-sys` / lockfile refresh for Homebrew HDF5 **2.2**
- Clippy cleanups (`as_chunks_mut` in CF decode, pedantic nits in embed materialize / fold)
- README to-do list; tet-py references in `README.md`, `docs/ffi.md`, and `docs/query_engine.md`

## [0.1.9] - 2026-05-29

### Added

- Transform **`write: sidecar`** — publish a derived one-chunk `.tet` beside the source file
- Embedder dense export: `materialize_query_selection`, `materialize_query_transform_ram`, `DenseMaterializeOutcome` / `DenseBuffer` ([`embed_materialize.rs`](src/query/embed_materialize.rs))

### Changed

- Docs for sidecar transform and embedder dense export paths

## [0.1.8] - 2026-05-29

### Fixed

- `nan_mean` / `nan_std` on strided chunks now use the NaN-skipping fold path

## [0.1.7] - 2026-05-29

### Added

- Unified **`transform`** wire (`zscore`, `minmax`, `l1`/`l2`, `center`, `scale`, `log1p`, `sqrt`, `softmax`) with `write` routing (`ram`, `spill`, `switch`)
- `nan_mean`, `nan_std`, and div-by-zero index warnings on transforms
- `any_inf` tier-A/B reduction
- [`docs/cli.md`](docs/cli.md) — CLI reference split out of README
- Per-module READMEs under `src/`

### Fixed

- `nan_mean` / `nan_std` empty-selection fold edge case; Windows `extern` block

## [0.1.6] - 2026-05-27

### Added

- **C ABI** (`tetration-ffi`): `tet_open`, `tet_query_json`, `tet_summary_json`, `tet_verify_json` — [`include/tetration.h`](include/tetration.h), [`docs/ffi.md`](docs/ffi.md)
- Release archives: per-platform `tetration-ffi` tarballs on tag push

## [0.1.5] - 2026-05-27

### Added

- **Phase 10 GPU (experimental):** `execution.device` / `tet query --device` — Metal (`tetration-metal`), CUDA (`tetration-gpu`), streaming fold, multi-GPU sharding when host RAM is tight

## [0.1.4] - 2026-05-27

### Added

- **TOML query profiles** — `parse_query_toml`, CLI auto-detect, paired fixtures in [`fixtures/queries/`](fixtures/queries/)
- `tet query --format table` for ASCII table output

## [0.1.3] - 2026-05-26

### Added

- **Phase 9 ops:** `nan_count`, `inf_count`, `null_count`, `covariance`, `correlation`, histogram caller `min`/`max`
- Named axes (`dim_names`), coord label slicing (`start_label` / `stop_label`)
- **`tet export`** — `.tet` → Zarr v3 directory store

## [0.1.2] - 2026-05-26

### Added

- Wire dtypes **`u8`**, **`u16`**, **`i16`**, **`u32`**, **`f16`**, **`u64`** end-to-end (convert, fold, materialize)
- **`tet verify`** / **`tet repair`**; `tet verify --deep` full chunk decode
- THST footer metadata, dataset attrs, streaming `TetWriterSession::commit_with_fill`
- Bulk SIMD folds for all wire dtypes; tracked [`fixtures/small/tet/`](fixtures/small/tet/)

## [0.1.1] - 2026-05-25

### Added

- [crates.io](https://crates.io/crates/tetration) publish and Homebrew tap on `v*` tag
- Out-of-core linear scan + SIMD bulk fold for large dense raw selections
- Flat JSON query wire; `tet qhist`, plan/stats/table output modes

## [0.1.0] - 2026-04-02

### Added

- Initial public release: layout v1 `.tet`, mmap read planning, JSON query engine, `tet convert` (HDF5 / NetCDF / Zarr v3), `tet info` / `tet query`

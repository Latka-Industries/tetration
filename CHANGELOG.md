# Changelog

## [Unreleased]

## [0.5.4] - 2026-10-20

### Changed

- Hygiene (THI-604 / THI-599): drop several `clippy::allow`s (wildcard SSE imports, float_cmp, int_plus_one); centralize `usize→f64` fold casts; `ExportError::Io` is transparent `std::io::Error`; HDF5/NetCDF convert/export map library errors via `From`; optional **`tetration-cli`** feature gates `clap` / `indicatif` (still in default; required for the `tet` binary).
- Fuzz: new `parse_query_json` target + corpus seeds from fixtures / conformance reject queries (`mise fuzz-query`).

### Notes

- Semver patch: crate hygiene + fuzz coverage only. Layout / query ABI / conformance pack milestone unchanged (**0.5.3**). PyPI **tet-py 0.5.4** pins this crate.

## [0.5.3] - 2026-10-20

### Changed

- Conformance pack `spec_milestone` **0.5.3**: more accept cases (u32/f16 mean+var, sample strided mean, large verify, element_count), full legacy reject_verify set, and expanded closed-wire reject_query fixtures (THI-605).

### Notes

- Semver patch: conformance coverage grow only. Layout / query ABI unchanged. PyPI **tet-py 0.5.3** pins this crate.

## [0.5.2] - 2026-10-20

### Added

- Normative SPEC draft at [`SPEC/README.md`](SPEC/README.md): layout v1, writer protocol v2, query JSON/TOML wire (closed keys / limits / tiers), and C ABI versioning, with a one-paragraph “compatible reader” definition. Detailed tables remain in `docs/` (THI-586).
- Compatibility / capability-flag policy at [`SPEC/COMPATIBILITY.md`](SPEC/COMPATIBILITY.md): frozen vs evolvable bits, refuse vs may-ignore, independent version axes (layout / query / package / ABI), and 2046-reader examples (THI-587).
- Conformance pack [`conformance/`](conformance/README.md) (`manifest.v1.json` accept + reject_verify + reject_query) with CI gate `cargo test --lib conformance` (THI-592).
- Optional Python stdlib smoke reader [`conformance/smoke/read_catalog_chunk.py`](conformance/smoke/read_catalog_chunk.py) — catalog + one raw `f32` chunk without the Rust crate (THI-589).

### Changed

- `tet semantics scaffold` (empty attrs) emits a single docs URL — `https://docs.tetration.sh/guides/semantics/` — matching the site guide that now includes worked examples. Removed `SEMANTICS_EXAMPLES_URL` (THI-600).
- Scaffold policy stubs use bare keys (`[profiles.media_v1]`, JSON map keyed the same way) instead of `[profiles."dotted.key"]` or a spare `id` field. Recommended profile ids are `media_v1` / `timeseries_v1` / `climate_v1` (THI-601).

### Notes

- Semver patch: SPEC + conformance milestone. Query engine answers unchanged. PyPI **tet-py** stays on **0.5.1** until THI-591 (pin after crates.io shows 0.5.2).

## [0.5.1] - 2026-10-20

### Added

- Declared science/media attrs helpers (`tetration::catalog::{set_dataset_attrs, scaffold_policy, …}`), `tet semantics scaffold` / `set`, and recording fixture `fixtures/small/tet/recording.tet`. Attrs do not change fold/query answers by themselves; interpretation stays in caller policy (`docs/semantics.md`) (THI-583).
- Docs for declared vs caller policy: crate [`docs/semantics.md`](docs/semantics.md) examples, roadmap Phase 16 ship note, site guides at `docs.tetration.sh/guides/semantics/` (THI-582).

### Notes

- Semver patch: additive metadata/CLI; query engine unchanged when semantics are missing. PyPI **tet-py** stays on **0.5.0** until THI-585.

## [0.5.0] - 2026-10-20

### Added

- Distributed fold library path (`tetration::query::distributed`): split a `ReadPlan` into contiguous or interleaved chunk shards, JSON worker job/partial types (`WorkerJobV1`, protocol v1), merge associative tier-A/B `FoldPartialState` (same combine laws as parallel/GPU host merge), and an in-process multi-shard harness. Workers must share one pinned generation. Spill/sidecar/switch write targets are refused on this path (RAM partials only) (THI-576, THI-575, THI-577).

### Notes

- No RPC cluster runtime. Tier-C ops (median/quantile/histogram/…) and transforms stay single-process. Semver minor: additive API; crate version was 0.4.5.

## [0.4.5] - 2026-10-20

### Added

- Optional value-posting index at compact/publish (`generations/<G>/INDEX.json`) for an exact `filter.eq` on a dataset with at most 64 distinct non-null values. A missing or unusable index falls back to zone maps and does not change answers. Drops are `ReadPlan.index_filter_skipped_chunks`; `zone_filter_skipped_chunks` stays zone-map-only. Verify warns when the manifest omits the field, and errors when a claimed blob is missing, hashed wrong, or (`--deep`) stale (THI-568, THI-571, THI-570).

## [0.4.4] - 2026-10-19

### Added

- Element-wise `filter` on cells inside chunks the zone map kept. Chunk skip from 0.4.1 is unchanged (`ReadPlan.zone_filter_skipped_chunks`). Tier-A/B folds ignore non-matching cells. Materialize and transform keep the selection shape: non-matches are NaN for `f16`/`f32`/`f64`, or the dataset fill for integers. Integer materialize under `filter` without `_FillValue`, `missing_value`, or `fill_value` is refused (THI-563–THI-565).

## [0.4.3] - 2026-10-19

### Added

- `prune_store` / `CachingStore::prune`: dry-run by default. Apply drops leftover staging and sealed generations older than `CURRENT` / `CURRENT-1`. Open pins and a live `publishing/<G>` claim are refused. Does not write `CURRENT` (THI-557).
- `tet gc s3://bucket/prefix` (feature `tetration-s3`, credentials `TETRATION_S3_*`): same retention. `--protect-generation G` or `?generation=G` keeps a `packed.tet` an open pin in another process still needs. Local `.tet.d` `tet gc` is unchanged (THI-559).

## [0.4.2] - 2026-10-19

### Added

- Closed `group_by` on a query document: dimension names or indices that partition rank with one reduction. Tier-A/B folds stream per group (`memory_strategy = streaming_fold`); tier-C still materializes. `execution.group_count` is the product of the group axes' logical extents (THI-552).
- Optional `pipeline` array that lowers onto the same document: `select`, then `filter`, then one reduction, then `spill` or `write`. Unknown steps, repeats, and out-of-order stages are rejected. At most eight steps (THI-553).

## [0.4.1] - 2026-10-19

### Added

- Query value filter (`filter` wire key): closed compare / null predicates on cell values (`eq`/`ne`/`lt`/`le`/`gt`/`ge`, `is_null`/`is_not_null`). Planner conservatively skips chunks whose zone min/max/null cannot match; missing `ZONE.json` means no skip. `ReadPlan.zone_filter_skipped_chunks` reports drops on plan and execute (THI-543–THI-547).

## [0.4.0] - 2026-10-18

### Added

- Per-chunk min/max/null zone maps at compact/publish. Stats are written to `generations/<G>/ZONE.json` for the sealed image and recomputed for the next generation. Staging parts are left unchanged. `tet verify` treats `MANIFEST.zone_maps = "required"` as an error when the file is missing, hashed wrong, or (with `--deep`) stale, and warns when an older generation has no zone maps (THI-473).

### Changed

- **Breaking (0.4.0):** `ReadPlan.logical_f32_element_count` is `logical_element_count` (public field and JSON key). The count is every dtype. This release still accepts the old JSON key on input. `materialize_read_plan_f32_le` is unchanged (THI-540).
- **Breaking (0.4.0):** removed the `tetration::query` re-exports of CLI history, `tet info` text, and `format_query_response` / `format_query_stderr_hints` that 0.3.8 deprecated. The formatters remain on `tetration::query::cli` for the `tet` binary. Embedders use `prelude` and `execute_query_*` (THI-541).

## [0.3.8] - 2026-10-18

### Changed

- `f16` and `f64` materialize, spill, fold, and preview share one macro. Behavior is unchanged (THI-537).
- Matchable package failures (`CURRENT` missing, bad session or part id, unsealed part, ingest source missing or empty) are `CatalogError` variants. Display text is unchanged (THI-538).
- `tetration::query` history, info, and response formatters are deprecated. They still compile. `prelude` and `execute_query_*` are the embed path. Removal is 0.4.0 (THI-539).

## [0.3.7] - 2026-10-18

### Added

- Clearer publish overlap rejects: `CatalogError::ClaimOverlap` reports part ids, dataset name, and chunk ranges (THI-526)
- `tet publish --dry-run` / `dry_run_publish_parts`: claim and schema checks only; no seal or `CURRENT` flip (THI-527)
- Identical-byte claim overlaps allowed at publish (idempotent retry via `part_sha256` or matching chunk bytes); differing bytes still reject (THI-528)
- Opt-in `tet publish --replace DATASET` / `PublishPartsOptions::replace` for dtype/shape change vs sealed `CURRENT` on G+1; default remains reject; sealed G untouched (THI-529)
- Disjoint shard claims: `DatasetClaim::ranges`, `TetShardWriter::push_dataset_with_claim`; merge unions complementary same-name tiles (THI-532)

### Changed

- Writer protocol decision E: identical-overlap OK; opt-in replace for schema change (never mutate sealed G)
- Crate authors email → `tensor@tetration.sh`

## [0.3.6] - 2026-10-18

### Added

- `tet pack --remove-package` / `TetPackage::pack_generation_with`: after a verified copy to a standalone `.tet`, optionally delete the source `.tet.d` (THI-518)
- `tet gc` / `TetPackage::gc`: dry-run by default; `--apply` drops leftover staging and sealed generations older than `CURRENT` / `CURRENT-1` (THI-521)
- `tet publish --abort --generation G`: drop a stuck `publishing/<G>/` lock; refuses `CURRENT` (THI-522)

## [0.3.5] - 2026-10-17

### Added

- `tet convert` reads a Zarr v2 or v3 prefix on `s3://` one chunk at a time. The `.tet` matches a local-directory convert of the same store. A single-file `s3://bucket/file.ext` still downloads the whole object (`tetration-s3`, `TETRATION_S3_ENDPOINT`) (THI-516)

## [0.3.4] - 2026-10-17

### Added

- `open_s3_package` and `tet query -t s3://bucket/prefix` pin one generation and read it through the existing range planner (`tetration-s3`, `TETRATION_S3_ENDPOINT`, optional `?generation=N`). The query fetches the catalog and the chunks it touches (THI-517)

## [0.3.3] - 2026-10-17

### Added

- Optional `BlockCache` / `CachingStore`: remember packed.tet range reads by `(generation_id, offset)`; cache is off unless wrapped; a new pin does not reuse another generation’s blocks (THI-501)
- `tet convert s3://bucket/object.ext` downloads one foreign file and runs the existing importer (`tetration-s3`, `TETRATION_S3_ENDPOINT`). Zarr prefixes are rejected (THI-502)

## [0.3.2] - 2026-10-17

### Added

- Cross-host exclusive claim on `publishing/<G>` for `PackageStore`: heartbeat, stale window, and force-only takeover so a second host cannot flip `CURRENT` (THI-500)

## [0.3.1] - 2026-10-17

### Added

- Optional `tetration-s3` feature: `S3Store` / `S3Config` speak `PackageStore` against an S3-compatible bucket (path-style, ETag conditional put). Tests use local MinIO and skip unless `TETRATION_S3_ENDPOINT` is set (THI-499)
- `stage_part`, `seal_staged_generation`, and `publish_staged_parts`: N writers upload parts, one CAS publish. A lost precondition leaves the previous `CURRENT` and its `packed.tet` (THI-471)

### Changed

- A query keeps the generation it opened. A newer `CURRENT` is invisible until the next open. Replacing the pinned `packed.tet` is not a new generation (roadmap, layout, query docs, `tet query` help, and the pin rustdoc; THI-472)

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

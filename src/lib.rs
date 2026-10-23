//! **tetration** — Rust library for the Tetration mmap-oriented chunked tensor format.
//!
//! Public API: [`catalog`], [`convert`], [`export`], [`layout`], [`query`], [`verify`], [`repair`],
//! and [`prelude`] for common embedder imports. [`convert`] imports HDF5, `NetCDF`, Zarr v2/v3,
//! `NumPy`, CSV, and Parquet into one `.tet`. [`export`] writes Zarr v3, HDF5, and `NetCDF` from an
//! existing `.tet`. The companion CLI is **`tet`** (`src/bin/tet.rs`, `src/bin/tet/`).
//!
//! Package reads pin one sealed generation and do not follow a newer `CURRENT` mid-query.
//! `S3Store` (feature `tetration-s3`) is an optional S3-compatible backend for that same store.
//! Cross-host publish uses an exclusive `publishing/<G>` claim ([`PublishOptions`](catalog::PublishOptions)).
//! Optional [`CachingStore`](catalog::CachingStore) remembers packed ranges by generation + offset.
//! [`convert_s3_object_to_tet`](convert::convert_s3_object_to_tet) downloads one `s3://` file, or
//! reads a Zarr prefix one chunk at a time. `open_s3_package` pins an `s3://bucket/prefix` package
//! and queries it with range reads.
//!
//! **Current line (0.7.3):** query wire includes closed `filter`, `group_by`, `window`, `align`,
//! `pipeline`, and `label_filter`. Zone maps + secondary indexes (`value_postings`,
//! `sorted_axis_range`, `coord_label_postings`) live as **embedded footer spills**
//! (`history_version = 2`; convert / writer auto-embed) and/or generation sidecars.
//! Semantics profiles (`media_v1` / …), package `SEMANTICS.json`, and advisory lint are
//! declare-only (fold-inert). Distributed query stays a thin protocol
//! ([`docs/distributed_boundary.md`](../docs/distributed_boundary.md)): associative scalar folds
//! plus dense materialize / tier-C / transform (`write: ram`) with an optional localhost
//! N-process harness (`tet worker`). `layout_version` stays **1**; package / C ABI unchanged;
//! conformance pack milestone **0.7.0**. Per-release notes:
//! [`CHANGELOG.md`](../CHANGELOG.md). Embed contract:
//! [`docs/embedded_indexes.md`](../docs/embedded_indexes.md). Roadmap:
//! [`docs/roadmap.md`](../docs/roadmap.md).
//!
//! **Embedder walkthrough:** `cargo run --example session_write` (or `create_and_query`,
//! `inspect_catalog`; file health: `tet verify` / [`verify`](verify)).

pub(crate) mod utils;

pub mod catalog;
pub mod convert;
pub mod export;
pub mod layout;
pub mod query;
pub mod repair;
pub mod verify;

#[cfg(feature = "tetration-ffi")]
pub mod ffi;

/// Common embedder imports: query document types, parse/validate/plan, mmap open.
pub mod prelude {
    pub use crate::catalog::{
        FileMetadataDraft, StreamTileJob, TetDatasetStreamSpec, TetDatasetWrite, TetFile,
        TetWriterSession, read_tet_summary_v1,
    };
    pub use crate::layout::{MAGIC, mmap_file_read};
    pub use crate::query::{
        ExecuteQueryOptions, QueryDocument, QueryResponse, ReadPlan, execute_query_document,
        execute_query_json, parse_query_json, plan_query_with_tet_mmap_ex, validate_query,
    };
    pub use crate::verify::{TetVerifyReport, verify_tet_file};
}

#[cfg(test)]
mod tests;

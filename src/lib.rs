//! **tetration** — Rust library for the Tetration mmap-oriented chunked tensor format.
//!
//! Public API: [`catalog`], [`convert`], [`export`], [`layout`], [`query`], [`verify`], [`repair`], and [`prelude`]
//! for common embedder imports. [`convert`] imports HDF5, `NetCDF`, Zarr v2/v3, `NumPy`, CSV, and
//! Parquet into one `.tet`. [`export`] writes Zarr v3, HDF5, and `NetCDF` from an existing `.tet`.
//! The companion CLI is **`tet`** (`src/bin/tet.rs`, `src/bin/tet/`).
//!
//! Package reads pin one sealed generation and do not follow a newer `CURRENT` mid-query.
//! `S3Store` (feature `tetration-s3`) is an optional S3-compatible backend for that same store.
//! Cross-host publish uses an exclusive `publishing/<G>` claim ([`PublishOptions`](catalog::PublishOptions)).
//! Optional [`CachingStore`](catalog::CachingStore) remembers packed ranges by generation + offset.
//! [`convert_s3_object_to_tet`](convert::convert_s3_object_to_tet) downloads one `s3://` file, or reads a Zarr prefix one chunk at a time (`tetration-s3`).
//! `open_s3_package` (feature `tetration-s3`) pins an `s3://bucket/prefix` package and queries it with range reads.
//!
//! **0.4.0:** compact/publish attaches per-chunk min/max/null zone maps to the sealed generation
//! (`ZONE.json`). [`query::ReadPlan::logical_element_count`](query::ReadPlan::logical_element_count)
//! is the selection length for every dtype; JSON still accepts `logical_f32_element_count` for this
//! release. CLI history and response formatting live in [`query::cli`](query::cli), not on [`query`].
//!
//! **0.4.1:** query documents accept a closed top-level `filter` (compare / null predicates). The
//! planner skips chunks whose zone stats cannot match; see
//! [`query::ReadPlan::zone_filter_skipped_chunks`](query::ReadPlan::zone_filter_skipped_chunks).
//!
//! **0.4.2:** `group_by` partitions a reduction by dimension name. Tier-A/B stays on the streaming
//! fold; `execution.group_count` reports how many groups. A `pipeline` array lowers onto that same
//! document (`select`, `filter`, one reduction, then `spill` or `write`).
//!
//! **0.4.3:** `prune_store` drops leftover staging and sealed generations older than `CURRENT` /
//! `CURRENT-1` (dry-run unless apply). It does not write `CURRENT`. Open pins and a live publish
//! claim are refused. `tet gc s3://bucket/prefix` is that command; `--protect-generation` names a
//! pin this process cannot see.
//!
//! **0.4.4:** the same `filter` applies to cells inside a kept chunk. Tier-A/B folds ignore
//! non-matches. Materialize and transform keep the selection shape (NaN for floats, dataset fill
//! for integers). `zone_filter_skipped_chunks` remains the chunk-skip stat.
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

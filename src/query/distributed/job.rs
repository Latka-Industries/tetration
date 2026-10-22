//! Serializable worker job / partial message shapes (JSON protocol v1).
//!
//! Wire contract for the thin distributed protocol. Localhost harness (`tet worker`)
//! uses filesystem / `file://` pins only; other schemes stay for future hosts.

use serde::{Deserialize, Serialize};

use super::dense::DenseShardPartial;
use super::merge::FoldPartialState;
use crate::query::types::{QueryDocument, ReadPlan};

/// Protocol version stamped on [`WorkerJobV1`] / [`WorkerPartialV1`].
pub const WORKER_PROTOCOL_V1: u32 = 1;

/// Pin for a sealed package generation. Workers open this pin only — never re-read `CURRENT`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct WorkerPinV1 {
    pub generation: u64,
    /// Object key relative to the package root (e.g. `generations/00000003/packed.tet`).
    /// Empty for a standalone `.tet` path pin in the localhost harness.
    pub object_key: String,
    pub object_len: u64,
    /// Coordinator-defined package URI. Localhost harness: bare path or `file://…`.
    pub package_uri: String,
}

/// Coordinator → worker work unit (JSON).
///
/// `plan` carries the **subset** of chunk I/O rows for this shard. `cell_filter` is not on
/// the [`ReadPlan`] JSON wire — workers must rebuild it from [`Self::document`] (or receive
/// it out-of-band). Keep `document` so filter / op stay authoritative.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct WorkerJobV1 {
    pub protocol: u32,
    pub job_id: String,
    pub shard_id: usize,
    pub n_shards: usize,
    pub pin: WorkerPinV1,
    pub document: QueryDocument,
    /// Indices into the coordinator's full planned chunk list (audit / debug).
    pub chunk_indices: Vec<usize>,
    /// Shard read plan (geometry + subset payloads). Deserialized plans lose `cell_filter`.
    pub plan: ReadPlan,
    /// Usually `0` on workers (preview collected at the coordinator if needed).
    pub preview_max: usize,
}

/// Worker → coordinator result (JSON).
///
/// Fold jobs set [`Self::partial`] and leave [`Self::dense`] unset. Dense
/// materialize / transform jobs set [`Self::dense`] (and may leave `partial` empty).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct WorkerPartialV1 {
    pub protocol: u32,
    pub job_id: String,
    pub shard_id: usize,
    pub pin_generation: u64,
    pub partial: FoldPartialState,
    /// Sparse dense contribution (0.7.2 materialize / transform). Absent on fold jobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dense: Option<DenseShardPartial>,
}

//! Distributed fold: shard a [`ReadPlan`] across workers and merge associative partials.
//!
//! Protocol starts in-process (local harness). Workers must open the **same pinned
//! generation** — never re-resolve `CURRENT` mid-query. Tier-C ops and transforms are
//! refused; see [`operation_is_shard_mergeable`].
//!
//! Phase 14 / THI-576 + THI-575.

mod harness;
mod job;
mod merge;
mod shard;
mod spill;

pub use harness::execute_sharded_scalar_fold_local;
pub use job::{WORKER_PROTOCOL_V1, WorkerJobV1, WorkerPartialV1, WorkerPinV1};
pub use merge::{
    FoldPartialState, finish_fold_partial, merge_fold_partials, operation_is_shard_mergeable,
};
pub use shard::{
    ShardPartition, ShardSpec, interleaved_chunk_indices, plan_chunk_shards, split_read_plan,
};
pub use spill::{DistributedSpillPolicy, refuse_distributed_spill};

//! Distributed fold: shard a [`ReadPlan`] across workers and merge associative partials.
//!
//! Protocol starts in-process (local harness) and can spawn OS processes via
//! [`execute_sharded_scalar_fold_multiprocess`]. Workers must open the **same pinned
//! generation** — never re-resolve `CURRENT` mid-query. Tier-C ops and transforms are
//! refused; see [`operation_is_shard_mergeable`].
//!
//! Product boundary: [`docs/distributed_boundary.md`](../../../docs/distributed_boundary.md).
//! Phase 14 / THI-576 + THI-575; 0.7.1 lock THI-615 / THI-616; harness THI-618.

mod harness;
mod job;
mod merge;
mod shard;
mod spill;

pub use harness::{
    MultiprocessFoldOpts, execute_sharded_scalar_fold_local,
    execute_sharded_scalar_fold_multiprocess, execute_worker_job_v1, resolve_worker_pin_path,
    write_worker_partial_json,
};
pub use job::{WORKER_PROTOCOL_V1, WorkerJobV1, WorkerPartialV1, WorkerPinV1};
pub use merge::{
    FoldPartialState, finish_fold_partial, merge_fold_partials, operation_is_shard_mergeable,
};
pub use shard::{
    ShardPartition, ShardSpec, interleaved_chunk_indices, plan_chunk_shards, split_read_plan,
};
pub use spill::{DistributedSpillPolicy, refuse_distributed_spill};

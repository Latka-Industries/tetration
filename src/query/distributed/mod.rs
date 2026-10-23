//! Distributed fold / dense materialize: shard a [`ReadPlan`] across workers.
//!
//! Protocol starts in-process (local harness) and can spawn OS processes via
//! multiprocess helpers. Workers must open the **same pinned generation** — never
//! re-resolve `CURRENT` mid-query.
//!
//! - Tier-A/B folds: [`operation_is_shard_mergeable`] + [`FoldPartialState`] merge.
//! - Tier-C / transform / selection: [`operation_is_dense_distributable`] +
//!   [`DenseShardPartial`] assemble (shipped in **0.7.2** / THI-617).
//!
//! Product boundary: [`docs/distributed_boundary.md`](../../../docs/distributed_boundary.md).

mod dense;
mod harness;
mod job;
mod merge;
mod shard;
mod spill;

pub use dense::{
    DenseShardPartial, assemble_dense_f32, compact_buffer_to_dense_f32, materialize_shard_dense_f32,
};
pub use harness::{
    MultiprocessDenseOpts, MultiprocessFoldOpts, TransformRamLocalOpts,
    execute_sharded_materialize_local, execute_sharded_materialize_multiprocess,
    execute_sharded_scalar_fold_local, execute_sharded_scalar_fold_multiprocess,
    execute_sharded_tier_c_local, execute_sharded_tier_c_multiprocess,
    execute_sharded_transform_ram_local, execute_worker_job_v1, resolve_worker_pin_path,
    write_worker_partial_json,
};
pub use job::{WORKER_PROTOCOL_V1, WorkerJobV1, WorkerPartialV1, WorkerPinV1};
pub use merge::{
    FoldPartialState, finish_fold_partial, merge_fold_partials, operation_is_dense_distributable,
    operation_is_shard_mergeable,
};
pub use shard::{
    ShardPartition, ShardSpec, interleaved_chunk_indices, plan_chunk_shards, split_read_plan,
};
pub use spill::{DistributedSpillPolicy, refuse_distributed_spill};

//! JSON/TOML query documents: validated plans and execution.
//!
//! Embedders should use [`crate::prelude`] and [`execute_query_document`] / [`execute_query_json`].
//! History, `tet info` text, and response formatting live in [`cli`] for the `tet` binary.
//!
//! Closed wire highlights: cell [`filter`](types::ValueFilter), footer
//! [`label_filter`](types::LabelFilterSpec), `group_by` / `window` / `align` / `pipeline`.
//! Optional package sidecars (`ZONE.json` / `INDEX.json`) feed planner chunk skip; see
//! [`docs/secondary_indexes.md`](../../docs/secondary_indexes.md).

mod align;
pub mod cli;
pub(crate) mod decode;
mod device;
mod dispatch;
pub mod distributed;
mod document;
mod document_toml;
mod document_wire;
mod dtype_from_wire;
pub mod embed_materialize;
pub(crate) mod engine;
mod execute;
pub(crate) mod fold;
mod gpu;
mod label_group;
pub(crate) mod materialize;
mod plan;
pub(crate) mod resolve_axes;
pub(crate) mod resolve_selection;
pub(crate) mod transform;
pub(crate) mod types;

pub use crate::catalog::DEFAULT_MEMORY_BUDGET_PERCENT_BPS;
pub use device::{
    DeviceRoute, GPU_AUTO_MIN_LOGICAL_BYTES, GPU_HOST_MATERIALIZE_RAM_FRACTION,
    GPU_HOST_MATERIALIZE_UNKNOWN_HOST_CAP_BYTES, cuda_backend_available, gpu_backend_available,
    host_materialize_exceeds, metal_backend_available, resolve_device_route,
};
pub use distributed::{
    DistributedSpillPolicy, FoldPartialState, MultiprocessFoldOpts, ShardPartition, ShardSpec,
    WORKER_PROTOCOL_V1, WorkerJobV1, WorkerPartialV1, WorkerPinV1,
    execute_sharded_scalar_fold_local, execute_sharded_scalar_fold_multiprocess,
    execute_worker_job_v1, finish_fold_partial, interleaved_chunk_indices, merge_fold_partials,
    operation_is_shard_mergeable, plan_chunk_shards, refuse_distributed_spill,
    resolve_worker_pin_path, split_read_plan, write_worker_partial_json,
};
pub use document::{
    QueryInputFormat, QueryLimits, detect_query_input_format, parse_query_json, parse_query_text,
    validate_query,
};
pub use document_toml::parse_query_toml;
pub use embed_materialize::{
    DenseBuffer, DenseMaterializeOutcome, estimate_query_selection_bytes,
    materialize_query_selection, materialize_query_transform_ram,
};
#[doc(hidden)]
pub use engine::TempSpillFile;
#[cfg(feature = "tetration-s3")]
pub use engine::store::query_opened_s3_package;
pub use engine::store::{
    execute_query_on_store_pin, execute_query_with_store, plan_query_with_store,
};
pub use engine::{
    DEFAULT_MEMORY_BUDGET_BYTES, ExecutionBudget, MaterializeReadPlanF32IntoOutcome,
    MemoryStrategy, PlannedRead, SpillPathAllowlist, materialize_read_plan_f32_le,
    materialize_read_plan_f32_le_into, materialize_read_plan_f32_le_into_parallel,
    materialize_read_plan_f32_le_parallel, materialize_read_plan_f64_le,
    materialize_read_plan_i16_le, materialize_read_plan_i32_le, materialize_read_plan_i64_le,
    materialize_read_plan_u8_le, materialize_read_plan_u16_le, plan_query_empty,
    plan_query_with_tet_mmap, plan_query_with_tet_mmap_ex, plan_read_for_document,
    planned_chunk_mmap_slices, spill_read_plan_f32_le, spill_read_plan_i16_le,
    spill_read_plan_i32_le, spill_read_plan_i64_le, spill_read_plan_u8_le, spill_read_plan_u16_le,
};
pub use execute::{ExecuteQueryOptions, execute_query_document, execute_query_json};
pub use plan::fetch::{chunk_payload_ranges, plan_query_fetch};
pub use types::{
    AlignSpec, AlignmentAxisResult, AlignmentResult, AxisSlice, AxisToken, ChunkTouchPolicy,
    DatasetResolution, ExecutionDeviceHint, ExecutionHints, GroupBySpec, Operation, OutputHint,
    OutputHints, PlannedChunkIo, QueryDocument, QueryExecutionPreview, QueryResponse, ReadPlan,
    SimpleOp, TetError, TransformMethod, ValueCompareOp, ValueFilter, ValuePredicate, WindowAlign,
    WindowMode, WindowPad, WindowSpec, WriteHints, WriteTarget,
};

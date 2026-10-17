//! JSON query documents: validated plans, execution, and CLI formatters.
//!
//! Primary public surface for query/embedder use; see also [`crate::prelude`].

mod cli;
pub(crate) mod decode;
mod device;
mod dispatch;
mod document;
mod document_toml;
mod document_wire;
pub mod embed_materialize;
pub(crate) mod engine;
mod execute;
pub(crate) mod fold;
mod gpu;
pub(crate) mod materialize;
mod plan;
pub(crate) mod resolve_axes;
pub(crate) mod resolve_selection;
pub(crate) mod transform;
pub(crate) mod types;

pub use crate::catalog::DEFAULT_MEMORY_BUDGET_PERCENT_BPS;
pub use cli::{
    CliQueryHistoryEntry, DEFAULT_INFO_CHUNK_TABLE_LIMIT, HistoryExecuteFilter, HistoryListFilter,
    HistorySettings, InfoListFilter, InfoMetadataDisplay, InfoViewSections, QueryOutputFormat,
    append_cli_query_history, clear_cli_query_history, cli_query_history_enabled,
    cli_query_history_max, cli_query_history_path, format_history_list_json,
    format_history_list_text, format_info_json, format_info_quiet, format_info_text,
    format_query_response, format_query_stderr_hints, get_cli_query_history_entry,
    history_entry_mode, info_view_sections_from_flags, list_cli_query_history,
    parse_history_execute_filter,
};
pub use device::{
    DeviceRoute, GPU_AUTO_MIN_LOGICAL_BYTES, GPU_HOST_MATERIALIZE_RAM_FRACTION,
    GPU_HOST_MATERIALIZE_UNKNOWN_HOST_CAP_BYTES, cuda_backend_available, gpu_backend_available,
    host_materialize_exceeds, metal_backend_available, resolve_device_route,
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
    AxisSlice, CHUNK_TOUCH_POLICY, ChunkTouchPolicy, DatasetResolution, ExecutionDeviceHint,
    ExecutionHints, Operation, OutputHint, OutputHints, PlannedChunkIo, QueryDocument,
    QueryExecutionPreview, QueryResponse, ReadPlan, TetError, TransformMethod, WriteHints,
    WriteTarget,
};

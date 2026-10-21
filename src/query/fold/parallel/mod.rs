//! Parallel chunk-streaming scalar, partial-axis, and windowed folds (Rayon).
//!
//! Workers visit planned chunks concurrently. Preview / transform scatter buffers are
//! shared via [`disjoint_buffer_mut`] — safe only because each chunk owns a disjoint
//! set of logical indices.

mod merge;
mod partial;
mod preview;
mod scalar;
mod window;
mod workers;

pub(crate) use partial::{
    fold_read_plan_partial_operation_f64_parallel, fold_read_plan_partial_operation_parallel,
};
/// Alias of preview's disjoint raw-slice helper for non-preview buffers (e.g. transform pass-2).
pub(crate) use preview::disjoint_preview_mut as disjoint_buffer_mut;
pub(crate) use scalar::{
    fold_read_plan_scalar_operation_f16_parallel, fold_read_plan_scalar_operation_f64_parallel,
    fold_read_plan_scalar_operation_int_parallel, fold_read_plan_scalar_operation_parallel,
};
pub(crate) use window::{parallel_window_memory_ok, scan_window_cells_parallel};
pub(crate) use workers::{use_parallel_fold, with_fold_workers};

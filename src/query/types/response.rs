//! Query engine response types (catalog match, read plan, execution preview).

use serde::Serialize;

use super::document::Operation;
use crate::query::materialize::DecodePreviewBundle;

/// Result of matching `QueryDocument::dataset` against a mmap’d `.tet` catalog.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct DatasetResolution {
    pub matched: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_shape: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_index_rows: Option<usize>,
    /// `4 * product(shape)` when dtype is `f32` and the dataset matched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_f32_bytes: Option<u64>,
    /// `8 * product(shape)` when dtype is `f64` and the dataset matched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_f64_bytes: Option<u64>,
    /// `4 * product(shape)` when dtype is `i32` and the dataset matched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_i32_bytes: Option<u64>,
    /// `8 * product(shape)` when dtype is `i64` and the dataset matched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_i64_bytes: Option<u64>,
    /// `1 * product(shape)` when dtype is `u8` and the dataset matched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_u8_bytes: Option<u64>,
    /// `2 * product(shape)` when dtype is `u16` and the dataset matched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_u16_bytes: Option<u64>,
    /// `2 * product(shape)` when dtype is `i16` and the dataset matched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_i16_bytes: Option<u64>,
    /// Execution preferences from the `.tet` chunk index header.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_execution: Option<crate::catalog::FileExecutionSettingsV1>,
    /// Present when `matched` is false: names from the file’s dataset directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available_datasets: Option<Vec<String>>,
}

/// Operation outputs merged into [`QueryExecutionPreview`] via [`From<OperationPreviewFields>`].
#[derive(Debug, Clone, Default)]
pub(crate) struct OperationPreviewFields {
    pub element_count: Option<usize>,
    pub sum: Option<f64>,
    pub mean: Option<f64>,
    pub median: Option<f64>,
    pub quantile: Option<f64>,
    pub histogram_counts: Option<Vec<f64>>,
    pub histogram_edges: Option<Vec<f64>>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub var: Option<f64>,
    pub std: Option<f64>,
    pub product: Option<f64>,
    pub norm_l1: Option<f64>,
    pub norm_l2: Option<f64>,
    pub all_finite: Option<bool>,
    pub any_nan: Option<bool>,
    pub any_inf: Option<bool>,
    pub nan_count: Option<f64>,
    pub inf_count: Option<f64>,
    pub null_count: Option<f64>,
    pub argmin_index: Option<u64>,
    pub argmax_index: Option<u64>,
    pub reduced_shape: Option<Vec<u64>>,
    pub reduced_sum: Option<Vec<f64>>,
    pub reduced_mean: Option<Vec<f64>>,
    pub reduced_median: Option<Vec<f64>>,
    pub reduced_quantile: Option<Vec<f64>>,
    pub reduced_histogram_counts: Option<Vec<f64>>,
    pub reduced_min: Option<Vec<f64>>,
    pub reduced_max: Option<Vec<f64>>,
    pub reduced_count: Option<Vec<f64>>,
    pub reduced_var: Option<Vec<f64>>,
    pub reduced_std: Option<Vec<f64>>,
    pub reduced_product: Option<Vec<f64>>,
    pub reduced_norm_l1: Option<Vec<f64>>,
    pub reduced_norm_l2: Option<Vec<f64>>,
    pub reduced_all_finite: Option<Vec<bool>>,
    pub reduced_any_nan: Option<Vec<bool>>,
    pub reduced_any_inf: Option<Vec<bool>>,
    pub reduced_nan_count: Option<Vec<f64>>,
    pub reduced_inf_count: Option<Vec<f64>>,
    pub reduced_null_count: Option<Vec<f64>>,
    pub reduced_argmin: Option<Vec<u64>>,
    pub reduced_argmax: Option<Vec<u64>>,
    /// Row-major `order × order` population covariance (`ddof = 0`).
    pub covariance: Option<Vec<f64>>,
    pub covariance_order: Option<u64>,
    /// Row-major `order × order` Pearson correlation.
    pub correlation: Option<Vec<f64>>,
    pub correlation_order: Option<u64>,
    pub nan_mean: Option<f64>,
    pub nan_std: Option<f64>,
    pub reduced_nan_mean: Option<Vec<f64>>,
    pub reduced_nan_std: Option<Vec<f64>>,
    /// Pass-1 transform method token (e.g. `zscore`, `softmax`).
    pub transform_method: Option<String>,
    /// First [`crate::query::transform::warnings::MAX_LISTED_DIV_BY_ZERO_INDICES`] logical indices with div-by-zero (or similar) in pass 2.
    pub transform_div_by_zero_indices: Option<Vec<u64>>,
    /// Total pass-2 elements that became NaN due to div-by-zero or invalid `sqrt` shift.
    pub transform_div_by_zero_count: Option<u64>,
}

/// I/O and spill metadata when building a [`QueryExecutionPreview`].
#[derive(Debug, Clone)]
pub(crate) struct ExecutionPreviewIo {
    pub total_bytes_read_from_disk: u64,
    pub memory_strategy: Option<&'static str>,
    pub spill_f32_path: Option<String>,
    pub spill_f32_bytes: Option<u64>,
}

/// Decode previews, operation outputs, and execution I/O for [`QueryExecutionPreview::assemble`].
#[derive(Debug, Clone)]
pub(crate) struct QueryExecutionPreviewBuild {
    pub io: ExecutionPreviewIo,
    pub previews: DecodePreviewBundle,
    pub operation: OperationPreviewFields,
}

/// First `f32` values read from planned chunk payloads (little-endian), capped for JSON safety.
/// When an [`Operation`] is present and execution runs, **`operation_*`** fields summarize the
/// **full** decoded logical tensor (row-major over the strided selection); see
/// `plan_query_with_tet_mmap`. Scalar reductions (`axes: []`) use `operation_sum`,
/// `operation_mean`, `operation_min`, `operation_max`, `operation_var`, `operation_std`, or
/// `operation_element_count` (for `count`), or `operation_product`. Partial reductions use `operation_reduced_shape` with matching
/// `operation_reduced_*` vectors (axes are decimal dimension indices).
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "snake_case")]
#[allow(clippy::struct_excessive_bools)]
pub struct QueryExecutionPreview {
    /// Sum of `stored_byte_len` for all planned chunks (bytes touched on disk).
    pub total_bytes_read_from_disk: u64,
    pub f32_preview: Vec<f32>,
    pub f32_preview_truncated: bool,
    pub f64_preview: Vec<f64>,
    pub f64_preview_truncated: bool,
    pub i32_preview: Vec<i32>,
    pub i32_preview_truncated: bool,
    pub i64_preview: Vec<i64>,
    pub i64_preview_truncated: bool,
    pub u8_preview: Vec<u8>,
    pub u8_preview_truncated: bool,
    pub u16_preview: Vec<u16>,
    pub u16_preview_truncated: bool,
    pub i16_preview: Vec<i16>,
    pub i16_preview_truncated: bool,
    pub u32_preview: Vec<u32>,
    pub u32_preview_truncated: bool,
    pub u64_preview: Vec<u64>,
    pub u64_preview_truncated: bool,
    pub f16_preview: Vec<half::f16>,
    pub f16_preview_truncated: bool,
    /// Set when an operation ran: number of `f32` values aggregated (full planned decode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_element_count: Option<usize>,
    /// Scalar result when `operation.axes` is empty (reduce all elements).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_sum: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_mean: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_median: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_quantile: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_histogram_counts: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_histogram_edges: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_var: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_std: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_nan_mean: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_nan_std: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_product: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_norm_l1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_norm_l2: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_all_finite: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_any_nan: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_any_inf: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_nan_count: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_inf_count: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_null_count: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_argmin_index: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_argmax_index: Option<u64>,
    /// How execution respected memory limits (`streaming_fold`, `capped_in_memory`, `mmap_spill`,
    /// `in_memory_materialize`, `temp_spill_materialize`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_strategy: Option<&'static str>,
    /// When set, full logical selection was spilled as row-major `f32` LE to this path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spill_f32_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spill_f32_bytes: Option<u64>,
    /// Resolved RAM budget used for this execution (bytes).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_budget_bytes: Option<u64>,
    /// Host available RAM when the budget was resolved, if detectable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_available_ram_bytes: Option<u64>,
    /// Logical selection size in bytes (`4 * logical_element_count` for f32 datasets).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logical_selection_f32_bytes: Option<u64>,
    /// Logical selection size in bytes (`elem_size * logical_element_count`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logical_selection_bytes: Option<u64>,
    /// Whether streaming fold used parallel chunk workers for this run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fold_parallel: Option<bool>,
    /// Rayon worker cap for parallel chunk fold (`None` = global pool size).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fold_workers: Option<usize>,
    /// Page-cache regime used for fold I/O routing (`in_core`, `out_of_core`, `unknown`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub io_regime: Option<&'static str>,
    /// Sequential byte-stream fold over one contiguous raw payload span.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fold_linear_scan: Option<bool>,
    /// Requested `execution.device` / CLI `--device` token, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_requested: Option<String>,
    /// Device that ran tier-A/B reduction (`cpu` until GPU kernels ship).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_used: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_fallback_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_gpu_reduce: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_gpu_pipeline: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_gpu_multi: Option<bool>,
    /// Shape after reducing along `operation.axes` (decimal dimension indices); row-major flattened payloads follow.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_shape: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_sum: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_mean: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_median: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_quantile: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_histogram_counts: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_min: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_max: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_count: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_var: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_std: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_nan_mean: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_nan_std: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_product: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_norm_l1: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_norm_l2: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_all_finite: Option<Vec<bool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_any_nan: Option<Vec<bool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_any_inf: Option<Vec<bool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_nan_count: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_inf_count: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_null_count: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_argmin: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_argmax: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_covariance: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_covariance_order: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_correlation: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_correlation_order: Option<u64>,
    /// Pass-1 transform method when `operation` is `transform`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transform_method: Option<String>,
    /// Listed logical indices for transform div-by-zero warnings (capped at 256).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transform_div_by_zero_indices: Option<Vec<u64>>,
    /// Total transform elements written as NaN due to div-by-zero or invalid `sqrt` shift.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transform_div_by_zero_count: Option<u64>,
}

impl From<OperationPreviewFields> for QueryExecutionPreview {
    fn from(operation: OperationPreviewFields) -> Self {
        Self {
            operation_element_count: operation.element_count,
            operation_sum: operation.sum,
            operation_mean: operation.mean,
            operation_median: operation.median,
            operation_quantile: operation.quantile,
            operation_histogram_counts: operation.histogram_counts,
            operation_histogram_edges: operation.histogram_edges,
            operation_min: operation.min,
            operation_max: operation.max,
            operation_var: operation.var,
            operation_std: operation.std,
            operation_nan_mean: operation.nan_mean,
            operation_nan_std: operation.nan_std,
            operation_product: operation.product,
            operation_norm_l1: operation.norm_l1,
            operation_norm_l2: operation.norm_l2,
            operation_all_finite: operation.all_finite,
            operation_any_nan: operation.any_nan,
            operation_any_inf: operation.any_inf,
            operation_nan_count: operation.nan_count,
            operation_inf_count: operation.inf_count,
            operation_null_count: operation.null_count,
            operation_argmin_index: operation.argmin_index,
            operation_argmax_index: operation.argmax_index,
            operation_reduced_shape: operation.reduced_shape,
            operation_reduced_sum: operation.reduced_sum,
            operation_reduced_mean: operation.reduced_mean,
            operation_reduced_median: operation.reduced_median,
            operation_reduced_quantile: operation.reduced_quantile,
            operation_reduced_histogram_counts: operation.reduced_histogram_counts,
            operation_reduced_min: operation.reduced_min,
            operation_reduced_max: operation.reduced_max,
            operation_reduced_count: operation.reduced_count,
            operation_reduced_var: operation.reduced_var,
            operation_reduced_std: operation.reduced_std,
            operation_reduced_nan_mean: operation.reduced_nan_mean,
            operation_reduced_nan_std: operation.reduced_nan_std,
            operation_reduced_product: operation.reduced_product,
            operation_reduced_norm_l1: operation.reduced_norm_l1,
            operation_reduced_norm_l2: operation.reduced_norm_l2,
            operation_reduced_all_finite: operation.reduced_all_finite,
            operation_reduced_any_nan: operation.reduced_any_nan,
            operation_reduced_any_inf: operation.reduced_any_inf,
            operation_reduced_nan_count: operation.reduced_nan_count,
            operation_reduced_inf_count: operation.reduced_inf_count,
            operation_reduced_null_count: operation.reduced_null_count,
            operation_reduced_argmin: operation.reduced_argmin,
            operation_reduced_argmax: operation.reduced_argmax,
            operation_covariance: operation.covariance,
            operation_covariance_order: operation.covariance_order,
            operation_correlation: operation.correlation,
            operation_correlation_order: operation.correlation_order,
            transform_method: operation.transform_method,
            transform_div_by_zero_indices: operation.transform_div_by_zero_indices,
            transform_div_by_zero_count: operation.transform_div_by_zero_count,
            ..Self::default()
        }
    }
}

impl QueryExecutionPreview {
    /// Decode preview only; all `operation_*` fields stay [`None`].
    #[must_use]
    pub fn decode_preview(
        total_bytes_read_from_disk: u64,
        f32_preview: Vec<f32>,
        f32_preview_truncated: bool,
    ) -> Self {
        Self {
            total_bytes_read_from_disk,
            f32_preview,
            f32_preview_truncated,
            ..Self::default()
        }
    }

    #[must_use]
    pub(crate) fn assemble(build: QueryExecutionPreviewBuild) -> Self {
        let QueryExecutionPreviewBuild {
            io:
                ExecutionPreviewIo {
                    total_bytes_read_from_disk,
                    memory_strategy,
                    spill_f32_path,
                    spill_f32_bytes,
                },
            previews,
            operation,
        } = build;
        Self {
            total_bytes_read_from_disk,
            f32_preview: previews.f32,
            f32_preview_truncated: previews.f32_truncated,
            f64_preview: previews.f64,
            f64_preview_truncated: previews.f64_truncated,
            i32_preview: previews.i32,
            i32_preview_truncated: previews.i32_truncated,
            i64_preview: previews.i64,
            i64_preview_truncated: previews.i64_truncated,
            u8_preview: previews.u8,
            u8_preview_truncated: previews.u8_truncated,
            u16_preview: previews.u16,
            u16_preview_truncated: previews.u16_truncated,
            i16_preview: previews.i16,
            i16_preview_truncated: previews.i16_truncated,
            u32_preview: previews.u32,
            u32_preview_truncated: previews.u32_truncated,
            u64_preview: previews.u64,
            u64_preview_truncated: previews.u64_truncated,
            f16_preview: previews.f16,
            f16_preview_truncated: previews.f16_truncated,
            memory_strategy,
            spill_f32_path,
            spill_f32_bytes,
            ..operation.into()
        }
    }
}

/// JSON response from plan-only or execute query paths (`tet query`, [`crate::query::plan_query_empty`]).
#[derive(Debug, Clone, Serialize)]
pub struct QueryResponse {
    pub status: &'static str,
    pub accepted: bool,
    pub layout_version: Option<u32>,
    pub dataset: String,
    pub selection_axes: Option<usize>,
    pub operation: Option<Operation>,
    pub message: String,
    /// When planning with `--tet`, the path echoed back for logs and scripts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tet_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub catalog: Option<DatasetResolution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_plan: Option<super::plan::ReadPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution: Option<QueryExecutionPreview>,
}

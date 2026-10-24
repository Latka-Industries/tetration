//! Slim JSON `stats` format for `tet query` (no chunk rows or preview arrays).

use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::query::types::{
    DatasetResolution, Operation, QueryExecutionPreview, QueryResponse, ReadPlan,
};

#[derive(Serialize)]
struct StatsResponse<'a> {
    status: &'static str,
    accepted: bool,
    dataset: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    layout_version: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    selection_axes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    operation: Option<&'a Operation>,
    message: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    tet_file: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    catalog: Option<StatsCatalog<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    read_plan: Option<StatsReadPlan<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    execution: Option<Value>,
}

#[derive(Serialize)]
pub(super) struct StatsCatalog<'a> {
    matched: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    dataset_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dtype: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shape: Option<&'a Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    available_datasets: Option<&'a Vec<String>>,
}

#[derive(Serialize)]
pub(super) struct StatsReadPlan<'a> {
    chunk_touch_policy: &'a str,
    chunk_count: usize,
    #[serde(skip_serializing_if = "usize_is_zero")]
    zone_filter_skipped_chunks: usize,
    #[serde(skip_serializing_if = "usize_is_zero")]
    index_filter_skipped_chunks: usize,
    total_stored_bytes: u64,
    logical_selection_shape: &'a Vec<u64>,
    logical_element_count: usize,
}

// serde `skip_serializing_if` requires `fn(&T) -> bool`.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn usize_is_zero(n: &usize) -> bool {
    *n == 0
}

pub(super) fn format_stats_json(response: &QueryResponse) -> Result<String, String> {
    serde_json::to_string_pretty(&stats_view(response)).map_err(|e| e.to_string())
}

pub(super) fn stats_catalog(c: &DatasetResolution) -> StatsCatalog<'_> {
    StatsCatalog {
        matched: c.matched,
        dataset_index: c.dataset_index,
        dtype: c.dtype,
        shape: c.shape.as_ref(),
        available_datasets: if c.matched {
            None
        } else {
            c.available_datasets.as_ref()
        },
    }
}

pub(super) fn stats_read_plan(p: &ReadPlan) -> StatsReadPlan<'_> {
    StatsReadPlan {
        chunk_touch_policy: p.chunk_touch_policy,
        chunk_count: p.chunk_count,
        zone_filter_skipped_chunks: p.zone_filter_skipped_chunks,
        index_filter_skipped_chunks: p.index_filter_skipped_chunks,
        total_stored_bytes: p.total_stored_bytes,
        logical_selection_shape: &p.logical_selection_shape,
        logical_element_count: p.logical_element_count,
    }
}

fn stats_view(response: &QueryResponse) -> StatsResponse<'_> {
    let catalog = response.catalog.as_ref().map(stats_catalog);
    let read_plan = response.read_plan.as_ref().map(stats_read_plan);
    let execution = response.execution.as_ref().map(execution_stats_value);
    StatsResponse {
        status: response.status,
        accepted: response.accepted,
        dataset: &response.dataset,
        layout_version: response.layout_version,
        selection_axes: response.selection_axes,
        operation: response.operation.as_ref(),
        message: &response.message,
        tet_file: response.tet_file.as_deref(),
        catalog,
        read_plan,
        execution,
    }
}

fn execution_stats_value(ex: &QueryExecutionPreview) -> Value {
    let mut map = Map::new();
    push_execution_io_stats(&mut map, ex);
    push_execution_scalar_operation_stats(&mut map, ex);
    push_execution_reduced_operation_stats(&mut map, ex);
    Value::Object(map)
}

fn push_execution_io_stats(map: &mut Map<String, Value>, ex: &QueryExecutionPreview) {
    map.insert(
        "total_bytes_read_from_disk".into(),
        json!(ex.total_bytes_read_from_disk),
    );
    opt_str(map, "memory_strategy", ex.memory_strategy);
    opt_usize(map, "group_count", ex.group_count);
    opt_str(map, "spill_f32_path", ex.spill_f32_path.as_deref());
    opt_u64(map, "spill_f32_bytes", ex.spill_f32_bytes);
    opt_u64(map, "memory_budget_bytes", ex.memory_budget_bytes);
    opt_u64(map, "host_available_ram_bytes", ex.host_available_ram_bytes);
    opt_u64(map, "logical_selection_bytes", ex.logical_selection_bytes);
    opt_u64(
        map,
        "logical_selection_f32_bytes",
        ex.logical_selection_f32_bytes,
    );
    opt_bool(map, "fold_parallel", ex.fold_parallel);
    opt_usize(map, "fold_workers", ex.fold_workers);
    opt_str(map, "io_regime", ex.io_regime);
    opt_bool(map, "fold_linear_scan", ex.fold_linear_scan);
    opt_str(map, "device_requested", ex.device_requested.as_deref());
    opt_str(map, "device_used", ex.device_used);
    opt_str(
        map,
        "device_fallback_reason",
        ex.device_fallback_reason.as_deref(),
    );
    opt_bool(map, "device_gpu_reduce", ex.device_gpu_reduce);
    opt_bool(map, "device_gpu_pipeline", ex.device_gpu_pipeline);
    opt_bool(map, "device_gpu_multi", ex.device_gpu_multi);
}

fn push_execution_scalar_operation_stats(map: &mut Map<String, Value>, ex: &QueryExecutionPreview) {
    opt_f64(map, "operation_sum", ex.operation_sum);
    opt_f64(map, "operation_mean", ex.operation_mean);
    opt_f64(map, "operation_median", ex.operation_median);
    opt_f64(map, "operation_quantile", ex.operation_quantile);
    opt_f64(map, "operation_min", ex.operation_min);
    opt_f64(map, "operation_max", ex.operation_max);
    opt_f64(map, "operation_var", ex.operation_var);
    opt_f64(map, "operation_std", ex.operation_std);
    opt_f64(map, "operation_nan_mean", ex.operation_nan_mean);
    opt_f64(map, "operation_nan_std", ex.operation_nan_std);
    opt_f64(map, "operation_product", ex.operation_product);
    opt_f64(map, "operation_norm_l1", ex.operation_norm_l1);
    opt_f64(map, "operation_norm_l2", ex.operation_norm_l2);
    opt_usize(map, "operation_element_count", ex.operation_element_count);
    opt_bool(map, "operation_all_finite", ex.operation_all_finite);
    opt_bool(map, "operation_any_nan", ex.operation_any_nan);
    opt_bool(map, "operation_any_inf", ex.operation_any_inf);
    opt_f64(map, "operation_nan_count", ex.operation_nan_count);
    opt_f64(map, "operation_inf_count", ex.operation_inf_count);
    opt_f64(map, "operation_null_count", ex.operation_null_count);
    opt_u64(map, "operation_argmin_index", ex.operation_argmin_index);
    opt_u64(map, "operation_argmax_index", ex.operation_argmax_index);
    opt_vec_f64(
        map,
        "operation_histogram_counts",
        ex.operation_histogram_counts.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_histogram_edges",
        ex.operation_histogram_edges.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_covariance",
        ex.operation_covariance.as_ref(),
    );
    opt_u64(
        map,
        "operation_covariance_order",
        ex.operation_covariance_order,
    );
    opt_vec_f64(
        map,
        "operation_correlation",
        ex.operation_correlation.as_ref(),
    );
    opt_u64(
        map,
        "operation_correlation_order",
        ex.operation_correlation_order,
    );
    opt_str(map, "transform_method", ex.transform_method.as_deref());
    opt_u64(
        map,
        "transform_div_by_zero_count",
        ex.transform_div_by_zero_count,
    );
    opt_vec_u64(
        map,
        "transform_div_by_zero_indices",
        ex.transform_div_by_zero_indices.as_ref(),
    );
}

fn push_execution_reduced_operation_stats(
    map: &mut Map<String, Value>,
    ex: &QueryExecutionPreview,
) {
    push_execution_reduced_aggregate_stats(map, ex);
    push_execution_reduced_qc_stats(map, ex);
}

fn push_execution_reduced_aggregate_stats(
    map: &mut Map<String, Value>,
    ex: &QueryExecutionPreview,
) {
    opt_vec_u64(
        map,
        "operation_reduced_shape",
        ex.operation_reduced_shape.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_sum",
        ex.operation_reduced_sum.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_mean",
        ex.operation_reduced_mean.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_median",
        ex.operation_reduced_median.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_quantile",
        ex.operation_reduced_quantile.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_histogram_counts",
        ex.operation_reduced_histogram_counts.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_min",
        ex.operation_reduced_min.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_max",
        ex.operation_reduced_max.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_count",
        ex.operation_reduced_count.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_var",
        ex.operation_reduced_var.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_std",
        ex.operation_reduced_std.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_nan_mean",
        ex.operation_reduced_nan_mean.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_nan_std",
        ex.operation_reduced_nan_std.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_product",
        ex.operation_reduced_product.as_ref(),
    );
}

fn push_execution_reduced_qc_stats(map: &mut Map<String, Value>, ex: &QueryExecutionPreview) {
    opt_vec_f64(
        map,
        "operation_reduced_norm_l1",
        ex.operation_reduced_norm_l1.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_norm_l2",
        ex.operation_reduced_norm_l2.as_ref(),
    );
    opt_vec_bool(
        map,
        "operation_reduced_all_finite",
        ex.operation_reduced_all_finite.as_ref(),
    );
    opt_vec_bool(
        map,
        "operation_reduced_any_nan",
        ex.operation_reduced_any_nan.as_ref(),
    );
    opt_vec_bool(
        map,
        "operation_reduced_any_inf",
        ex.operation_reduced_any_inf.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_nan_count",
        ex.operation_reduced_nan_count.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_inf_count",
        ex.operation_reduced_inf_count.as_ref(),
    );
    opt_vec_f64(
        map,
        "operation_reduced_null_count",
        ex.operation_reduced_null_count.as_ref(),
    );
    opt_vec_u64(
        map,
        "operation_reduced_argmin",
        ex.operation_reduced_argmin.as_ref(),
    );
    opt_vec_u64(
        map,
        "operation_reduced_argmax",
        ex.operation_reduced_argmax.as_ref(),
    );
}

fn opt_f64(map: &mut Map<String, Value>, key: &str, v: Option<f64>) {
    if let Some(v) = v {
        map.insert(key.into(), json!(v));
    }
}

fn opt_usize(map: &mut Map<String, Value>, key: &str, v: Option<usize>) {
    if let Some(v) = v {
        map.insert(key.into(), json!(v));
    }
}

fn opt_u64(map: &mut Map<String, Value>, key: &str, v: Option<u64>) {
    if let Some(v) = v {
        map.insert(key.into(), json!(v));
    }
}

fn opt_bool(map: &mut Map<String, Value>, key: &str, v: Option<bool>) {
    if let Some(v) = v {
        map.insert(key.into(), json!(v));
    }
}

fn opt_str(map: &mut Map<String, Value>, key: &str, v: Option<&str>) {
    if let Some(v) = v {
        map.insert(key.into(), json!(v));
    }
}

fn opt_vec_f64(map: &mut Map<String, Value>, key: &str, v: Option<&Vec<f64>>) {
    if let Some(v) = v {
        map.insert(key.into(), json!(v));
    }
}

fn opt_vec_u64(map: &mut Map<String, Value>, key: &str, v: Option<&Vec<u64>>) {
    if let Some(v) = v {
        map.insert(key.into(), json!(v));
    }
}

fn opt_vec_bool(map: &mut Map<String, Value>, key: &str, v: Option<&Vec<bool>>) {
    if let Some(v) = v {
        map.insert(key.into(), json!(v));
    }
}

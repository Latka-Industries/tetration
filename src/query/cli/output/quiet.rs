//! Quiet one-line `tet query` stdout formatting.

use crate::query::types::{
    DatasetResolution, Operation, QueryExecutionPreview, QueryResponse, ReadPlan,
};

use super::format_num::{
    QUIET_VEC_INLINE_MAX, fmt_bool_list, fmt_f64, fmt_f64_list, fmt_i64_list, fmt_u64_list,
    missing_field,
};

pub(super) fn format_quiet_line(response: &QueryResponse) -> Result<String, String> {
    if let Some(catalog) = response.catalog.as_ref()
        && !catalog.matched
    {
        return Ok(unmatched_dataset_quiet_line(response, catalog));
    }

    if let Some(ex) = response.execution.as_ref() {
        if let Some(op) = response.operation.as_ref() {
            return operation_quiet_line(&response.dataset, op, ex);
        }
        return Ok(preview_quiet_line(&response.dataset, ex));
    }

    if let Some(plan) = response.read_plan.as_ref() {
        return Ok(plan_quiet_line(&response.dataset, plan));
    }

    Ok(plan_only_echo_quiet_line(response))
}

fn quiet_prefix(dataset: &str) -> String {
    format!("dataset={dataset}")
}

fn plan_only_echo_quiet_line(response: &QueryResponse) -> String {
    let mut parts = vec![quiet_prefix(&response.dataset), "validated".to_string()];
    if let Some(n) = response.selection_axes {
        parts.push(format!("selection_rank={n}"));
    }
    parts.push("hint=pass --tet PATH for catalog and read_plan".to_string());
    parts.join(" ")
}

fn unmatched_dataset_quiet_line(response: &QueryResponse, catalog: &DatasetResolution) -> String {
    let mut parts = vec![
        quiet_prefix(&response.dataset),
        "status=not_found".to_string(),
    ];
    if let Some(names) = catalog.available_datasets.as_ref() {
        parts.push(format!("available={}", fmt_name_list(names)));
    }
    parts.join(" ")
}

fn fmt_name_list(names: &[String]) -> String {
    const MAX: usize = 8;
    if names.len() <= MAX {
        return format!("[{}]", names.join(","));
    }
    let head = names[..MAX].join(",");
    format!("[{head},…+{}]", names.len() - MAX)
}

fn plan_quiet_line(dataset: &str, plan: &ReadPlan) -> String {
    let shape = fmt_shape(&plan.logical_selection_shape);
    let mut parts = vec![
        quiet_prefix(dataset),
        "status=planned".to_string(),
        format!("chunks={}", plan.chunk_count),
        format!("elements={}", plan.logical_element_count),
        format!("stored_bytes={}", plan.total_stored_bytes),
        format!("logical_shape={shape}"),
        format!("touch_policy={}", plan.chunk_touch_policy),
    ];
    if plan.zone_filter_skipped_chunks > 0 {
        parts.push(format!(
            "zone_filter_skipped={}",
            plan.zone_filter_skipped_chunks
        ));
    }
    if plan.index_filter_skipped_chunks > 0 {
        parts.push(format!(
            "index_filter_skipped={}",
            plan.index_filter_skipped_chunks
        ));
    }
    if plan.cell_filter.is_some() {
        parts.push("cell_filter=applied".to_string());
    }
    if plan.label_eq.is_some() {
        parts.push("label_filter=applied".to_string());
    }
    parts.join(" ")
}

fn preview_quiet_line(dataset: &str, ex: &QueryExecutionPreview) -> String {
    let mut parts = vec![
        quiet_prefix(dataset),
        "status=preview".to_string(),
        format!("read_bytes={}", ex.total_bytes_read_from_disk),
    ];
    if let Some(path) = ex.spill_f32_path.as_deref() {
        parts.push(format!("spill={path}"));
        if let Some(bytes) = ex.spill_f32_bytes {
            parts.push(format!("spill_bytes={bytes}"));
        }
    }
    if let Some(strategy) = ex.memory_strategy {
        parts.push(format!("memory_strategy={strategy}"));
    }

    if let Some((label, n, truncated)) = preview_dtype_summary(ex) {
        let tail = if truncated { " truncated=true" } else { "" };
        parts.push(format!("preview={n} {label}{tail}"));
        if let Some(sample) = preview_sample_values(ex, label) {
            parts.push(format!("sample={sample}"));
        }
    } else {
        parts.push("preview=empty".to_string());
    }
    parts.join(" ")
}

fn preview_dtype_summary(ex: &QueryExecutionPreview) -> Option<(&'static str, usize, bool)> {
    if !ex.f32_preview.is_empty() {
        return Some(("f32", ex.f32_preview.len(), ex.f32_preview_truncated));
    }
    if !ex.f64_preview.is_empty() {
        return Some(("f64", ex.f64_preview.len(), ex.f64_preview_truncated));
    }
    if !ex.i32_preview.is_empty() {
        return Some(("i32", ex.i32_preview.len(), ex.i32_preview_truncated));
    }
    if !ex.i64_preview.is_empty() {
        return Some(("i64", ex.i64_preview.len(), ex.i64_preview_truncated));
    }
    if !ex.u8_preview.is_empty() {
        return Some(("u8", ex.u8_preview.len(), ex.u8_preview_truncated));
    }
    if !ex.u16_preview.is_empty() {
        return Some(("u16", ex.u16_preview.len(), ex.u16_preview_truncated));
    }
    if !ex.i16_preview.is_empty() {
        return Some(("i16", ex.i16_preview.len(), ex.i16_preview_truncated));
    }
    if !ex.u32_preview.is_empty() {
        return Some(("u32", ex.u32_preview.len(), ex.u32_preview_truncated));
    }
    if !ex.u64_preview.is_empty() {
        return Some(("u64", ex.u64_preview.len(), ex.u64_preview_truncated));
    }
    if !ex.f16_preview.is_empty() {
        return Some(("f16", ex.f16_preview.len(), ex.f16_preview_truncated));
    }
    None
}

fn preview_sample_values(ex: &QueryExecutionPreview, label: &str) -> Option<String> {
    const SAMPLE: usize = 6;
    match label {
        "f32" => Some(fmt_f64_list(
            &ex.f32_preview
                .iter()
                .take(SAMPLE)
                .map(|v| f64::from(*v))
                .collect::<Vec<_>>(),
            QUIET_VEC_INLINE_MAX,
        )),
        "f64" => Some(fmt_f64_list(
            &ex.f64_preview
                .iter()
                .take(SAMPLE)
                .copied()
                .collect::<Vec<_>>(),
            QUIET_VEC_INLINE_MAX,
        )),
        "i32" => Some(fmt_i64_list(
            &ex.i32_preview
                .iter()
                .take(SAMPLE)
                .map(|v| i64::from(*v))
                .collect::<Vec<_>>(),
            QUIET_VEC_INLINE_MAX,
        )),
        "i64" => Some(fmt_i64_list(
            &ex.i64_preview
                .iter()
                .take(SAMPLE)
                .copied()
                .collect::<Vec<_>>(),
            QUIET_VEC_INLINE_MAX,
        )),
        "u8" => Some(fmt_u64_list(
            &ex.u8_preview
                .iter()
                .take(SAMPLE)
                .map(|v| u64::from(*v))
                .collect::<Vec<_>>(),
            QUIET_VEC_INLINE_MAX,
        )),
        "u16" => Some(fmt_u64_list(
            &ex.u16_preview
                .iter()
                .take(SAMPLE)
                .map(|v| u64::from(*v))
                .collect::<Vec<_>>(),
            QUIET_VEC_INLINE_MAX,
        )),
        "i16" => Some(fmt_i64_list(
            &ex.i16_preview
                .iter()
                .take(SAMPLE)
                .map(|v| i64::from(*v))
                .collect::<Vec<_>>(),
            QUIET_VEC_INLINE_MAX,
        )),
        "u32" => Some(fmt_u64_list(
            &ex.u32_preview
                .iter()
                .take(SAMPLE)
                .map(|v| u64::from(*v))
                .collect::<Vec<_>>(),
            QUIET_VEC_INLINE_MAX,
        )),
        "u64" => Some(fmt_u64_list(
            &ex.u64_preview
                .iter()
                .take(SAMPLE)
                .copied()
                .collect::<Vec<_>>(),
            QUIET_VEC_INLINE_MAX,
        )),
        "f16" => Some(fmt_f64_list(
            &ex.f16_preview
                .iter()
                .take(SAMPLE)
                .map(|v| f64::from(*v))
                .collect::<Vec<_>>(),
            QUIET_VEC_INLINE_MAX,
        )),
        _ => None,
    }
}

fn operation_quiet_line(
    dataset: &str,
    op: &Operation,
    ex: &QueryExecutionPreview,
) -> Result<String, String> {
    if op.axes().is_empty() {
        return scalar_operation_quiet_line(dataset, op, ex);
    }
    partial_operation_quiet_line(dataset, op, ex)
}

fn scalar_operation_quiet_line(
    dataset: &str,
    op: &Operation,
    ex: &QueryExecutionPreview,
) -> Result<String, String> {
    let mut parts = vec![
        quiet_prefix(dataset),
        "status=ok".to_string(),
        operation_op_tag(op),
    ];
    parts.extend(operation_extra_tags(op));
    let (name, value) = scalar_operation_display(op, ex)?;
    parts.push(format!("{name}={value}"));
    if let Some(n) = ex.operation_element_count {
        parts.push(format!("elements={n}"));
    }
    if let Some(n) = ex.group_count {
        parts.push(format!("group_count={n}"));
    }
    if let Some(strategy) = ex.memory_strategy {
        parts.push(format!("memory_strategy={strategy}"));
    }
    parts.extend(transform_warning_tags(ex));
    Ok(parts.join(" "))
}

fn partial_operation_quiet_line(
    dataset: &str,
    op: &Operation,
    ex: &QueryExecutionPreview,
) -> Result<String, String> {
    let mut parts = vec![
        quiet_prefix(dataset),
        "status=ok".to_string(),
        operation_op_tag(op),
        format!("axes={}", fmt_axes(op.axes())),
    ];
    parts.extend(operation_extra_tags(op));
    if let Some(shape) = ex.operation_reduced_shape.as_ref() {
        parts.push(format!("reduced_shape={}", fmt_shape(shape)));
    }
    let values = partial_operation_values(op, ex)?;
    parts.push(format!("values={values}"));
    if let Some(n) = ex.operation_element_count {
        parts.push(format!("elements={n}"));
    }
    if let Some(n) = ex.group_count {
        parts.push(format!("group_count={n}"));
    }
    if let Some(strategy) = ex.memory_strategy {
        parts.push(format!("memory_strategy={strategy}"));
    }
    parts.extend(transform_warning_tags(ex));
    Ok(parts.join(" "))
}

fn operation_op_tag(op: &Operation) -> String {
    format!("op={}", operation_name(op))
}

fn operation_name(op: &Operation) -> &'static str {
    op.wire_key()
}

fn operation_extra_tags(op: &Operation) -> Vec<String> {
    match op {
        Operation::Quantile { q, .. } => vec![format!("q={}", fmt_f64(*q))],
        Operation::Histogram { bins, min, max, .. } => {
            let mut tags = vec![format!("bins={bins}")];
            if let Some(v) = min {
                tags.push(format!("min={}", fmt_f64(*v)));
            }
            if let Some(v) = max {
                tags.push(format!("max={}", fmt_f64(*v)));
            }
            tags
        }
        Operation::NullCount { fill, .. } => fill
            .map(|v| vec![format!("fill={}", fmt_f64(v))])
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn fmt_axes(axes: &[String]) -> String {
    if axes.is_empty() {
        "[]".to_string()
    } else {
        format!("[{}]", axes.join(","))
    }
}

pub(super) fn fmt_shape(shape: &[u64]) -> String {
    crate::cli_pretty::fmt_shape_dims(shape, "0")
}

fn quiet_scalar_field<T>(
    label: &'static str,
    field: &'static str,
    value: Option<T>,
    format_value: impl FnOnce(T) -> String,
) -> Result<(&'static str, String), String> {
    value
        .map(|v| (label, format_value(v)))
        .ok_or_else(|| missing_field(field))
}

fn quiet_scalar_f64(
    label: &'static str,
    field: &'static str,
    value: Option<f64>,
) -> Result<(&'static str, String), String> {
    quiet_scalar_field(label, field, value, fmt_f64)
}

fn quiet_reduced_f64(field: &'static str, values: Option<&Vec<f64>>) -> Result<String, String> {
    values
        .map(|v| fmt_f64_list(v, QUIET_VEC_INLINE_MAX))
        .ok_or_else(|| missing_field(field))
}

fn quiet_reduced_bool(field: &'static str, values: Option<&Vec<bool>>) -> Result<String, String> {
    values
        .map(|v| fmt_bool_list(v, QUIET_VEC_INLINE_MAX))
        .ok_or_else(|| missing_field(field))
}

fn quiet_reduced_u64(field: &'static str, values: Option<&Vec<u64>>) -> Result<String, String> {
    values
        .map(|v| fmt_u64_list(v, QUIET_VEC_INLINE_MAX))
        .ok_or_else(|| missing_field(field))
}

pub(super) fn scalar_operation_display(
    op: &Operation,
    ex: &QueryExecutionPreview,
) -> Result<(&'static str, String), String> {
    match op {
        Operation::Sum { .. }
        | Operation::Mean { .. }
        | Operation::Min { .. }
        | Operation::Max { .. }
        | Operation::Var { .. }
        | Operation::Std { .. }
        | Operation::Product { .. }
        | Operation::NormL1 { .. }
        | Operation::NormL2 { .. }
        | Operation::NanCount { .. }
        | Operation::InfCount { .. }
        | Operation::NullCount { .. }
        | Operation::Median { .. } => scalar_operation_display_f64(op, ex),
        Operation::Count { .. }
        | Operation::AllFinite { .. }
        | Operation::AnyNan { .. }
        | Operation::AnyInf { .. }
        | Operation::ArgMin { .. }
        | Operation::ArgMax { .. }
        | Operation::Quantile { .. }
        | Operation::Histogram { .. }
        | Operation::Covariance { .. }
        | Operation::Correlation { .. } => scalar_operation_display_special(op, ex),
        Operation::NanMean { .. } => {
            quiet_scalar_f64("nan_mean", "operation_nan_mean", ex.operation_nan_mean)
        }
        Operation::NanStd { .. } => {
            quiet_scalar_f64("nan_std", "operation_nan_std", ex.operation_nan_std)
        }
        Operation::Transform { .. } => transform_operation_display(op, ex),
    }
}

/// Format one pass-1 stat from scalar or partial-axis preview fields.
fn transform_stat_value(
    partial: bool,
    scalar_field: &'static str,
    reduced_field: &'static str,
    scalar: Option<f64>,
    reduced: Option<&Vec<f64>>,
) -> Result<String, String> {
    if partial {
        quiet_reduced_f64(reduced_field, reduced)
    } else {
        scalar
            .map(fmt_f64)
            .ok_or_else(|| missing_field(scalar_field))
    }
}

/// Quiet-footer pass-1 stats for transform ops (`partial` selects `operation_reduced_*` fields).
fn transform_stats_line(ex: &QueryExecutionPreview, partial: bool) -> Result<String, String> {
    match ex.transform_method.as_deref() {
        Some("zscore") => {
            let mean = transform_stat_value(
                partial,
                "operation_mean",
                "operation_reduced_mean",
                ex.operation_mean,
                ex.operation_reduced_mean.as_ref(),
            )?;
            let std = transform_stat_value(
                partial,
                "operation_std",
                "operation_reduced_std",
                ex.operation_std,
                ex.operation_reduced_std.as_ref(),
            )?;
            Ok(format!("mean={mean} std={std}"))
        }
        Some("minmax") => {
            let min = transform_stat_value(
                partial,
                "operation_min",
                "operation_reduced_min",
                ex.operation_min,
                ex.operation_reduced_min.as_ref(),
            )?;
            let max = transform_stat_value(
                partial,
                "operation_max",
                "operation_reduced_max",
                ex.operation_max,
                ex.operation_reduced_max.as_ref(),
            )?;
            Ok(format!("min={min} max={max}"))
        }
        Some("center") => transform_stat_value(
            partial,
            "operation_mean",
            "operation_reduced_mean",
            ex.operation_mean,
            ex.operation_reduced_mean.as_ref(),
        )
        .map(|mean| format!("mean={mean}")),
        Some("scale") => transform_stat_value(
            partial,
            "operation_std",
            "operation_reduced_std",
            ex.operation_std,
            ex.operation_reduced_std.as_ref(),
        )
        .map(|std| format!("std={std}")),
        Some("l1") => transform_stat_value(
            partial,
            "operation_norm_l1",
            "operation_reduced_norm_l1",
            ex.operation_norm_l1,
            ex.operation_reduced_norm_l1.as_ref(),
        )
        .map(|n1| format!("norm_l1={n1}")),
        Some("l2") => transform_stat_value(
            partial,
            "operation_norm_l2",
            "operation_reduced_norm_l2",
            ex.operation_norm_l2,
            ex.operation_reduced_norm_l2.as_ref(),
        )
        .map(|n2| format!("norm_l2={n2}")),
        Some("log1p" | "sqrt") => transform_stat_value(
            partial,
            "operation_min",
            "operation_reduced_min",
            ex.operation_min,
            ex.operation_reduced_min.as_ref(),
        )
        .map(|min| format!("min={min}")),
        Some("softmax") => transform_stat_value(
            partial,
            "operation_max",
            "operation_reduced_max",
            ex.operation_max,
            ex.operation_reduced_max.as_ref(),
        )
        .map(|max| format!("max={max}")),
        Some(method) => Ok(format!("method={method}")),
        None => Ok("method=transform".to_string()),
    }
}

fn transform_scalar_stats_line(ex: &QueryExecutionPreview) -> Result<String, String> {
    transform_stats_line(ex, false)
}

/// Append `transform_div_by_zero_*` tags when pass 2 recorded NaN-producing divides.
fn transform_warning_tags(ex: &QueryExecutionPreview) -> Vec<String> {
    let mut tags = Vec::new();
    if let Some(count) = ex.transform_div_by_zero_count {
        tags.push(format!("transform_div_by_zero_count={count}"));
        if let Some(indices) = ex.transform_div_by_zero_indices.as_ref() {
            let listed = indices
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",");
            tags.push(format!("transform_div_by_zero_indices=[{listed}]"));
        }
    }
    tags
}

fn transform_operation_display(
    op: &Operation,
    ex: &QueryExecutionPreview,
) -> Result<(&'static str, String), String> {
    let Operation::Transform { axes, .. } = op else {
        unreachable!("transform_operation_display");
    };
    if axes.is_empty() {
        Ok((op.wire_key(), transform_scalar_stats_line(ex)?))
    } else {
        Ok((op.wire_key(), partial_operation_values(op, ex)?))
    }
}

fn scalar_operation_display_f64(
    op: &Operation,
    ex: &QueryExecutionPreview,
) -> Result<(&'static str, String), String> {
    match op {
        Operation::Sum { .. } => quiet_scalar_f64("sum", "operation_sum", ex.operation_sum),
        Operation::Mean { .. } => quiet_scalar_f64("mean", "operation_mean", ex.operation_mean),
        Operation::Min { .. } => quiet_scalar_f64("min", "operation_min", ex.operation_min),
        Operation::Max { .. } => quiet_scalar_f64("max", "operation_max", ex.operation_max),
        Operation::Var { .. } => quiet_scalar_f64("var", "operation_var", ex.operation_var),
        Operation::Std { .. } => quiet_scalar_f64("std", "operation_std", ex.operation_std),
        Operation::NanMean { .. } => {
            quiet_scalar_f64("nan_mean", "operation_nan_mean", ex.operation_nan_mean)
        }
        Operation::NanStd { .. } => {
            quiet_scalar_f64("nan_std", "operation_nan_std", ex.operation_nan_std)
        }
        Operation::Product { .. } => {
            quiet_scalar_f64("product", "operation_product", ex.operation_product)
        }
        Operation::NormL1 { .. } => {
            quiet_scalar_f64("norm_l1", "operation_norm_l1", ex.operation_norm_l1)
        }
        Operation::NormL2 { .. } => {
            quiet_scalar_f64("norm_l2", "operation_norm_l2", ex.operation_norm_l2)
        }
        Operation::NanCount { .. } => {
            quiet_scalar_f64("nan_count", "operation_nan_count", ex.operation_nan_count)
        }
        Operation::InfCount { .. } => {
            quiet_scalar_f64("inf_count", "operation_inf_count", ex.operation_inf_count)
        }
        Operation::NullCount { .. } => quiet_scalar_f64(
            "null_count",
            "operation_null_count",
            ex.operation_null_count,
        ),
        Operation::Median { .. } => {
            quiet_scalar_f64("median", "operation_median", ex.operation_median)
        }
        _ => unreachable!("scalar_operation_display_f64 called for non-f64 op"),
    }
}

fn scalar_operation_display_special(
    op: &Operation,
    ex: &QueryExecutionPreview,
) -> Result<(&'static str, String), String> {
    match op {
        Operation::Count { .. } => quiet_scalar_field(
            "count",
            "operation_element_count",
            ex.operation_element_count,
            |v| v.to_string(),
        ),
        Operation::AllFinite { .. } => quiet_scalar_field(
            "all_finite",
            "operation_all_finite",
            ex.operation_all_finite,
            |v| v.to_string(),
        ),
        Operation::AnyNan { .. } => {
            quiet_scalar_field("any_nan", "operation_any_nan", ex.operation_any_nan, |v| {
                v.to_string()
            })
        }
        Operation::AnyInf { .. } => {
            quiet_scalar_field("any_inf", "operation_any_inf", ex.operation_any_inf, |v| {
                v.to_string()
            })
        }
        Operation::ArgMin { .. } => quiet_scalar_field(
            "argmin_index",
            "operation_argmin_index",
            ex.operation_argmin_index,
            |v| v.to_string(),
        ),
        Operation::ArgMax { .. } => quiet_scalar_field(
            "argmax_index",
            "operation_argmax_index",
            ex.operation_argmax_index,
            |v| v.to_string(),
        ),
        Operation::Quantile { q, .. } => ex
            .operation_quantile
            .map(|v| ("quantile", fmt_f64(v)))
            .ok_or_else(|| missing_field(&format!("operation_quantile (q={})", fmt_f64(*q)))),
        Operation::Histogram { bins, .. } => {
            let counts = ex
                .operation_histogram_counts
                .as_ref()
                .ok_or_else(|| missing_field("operation_histogram_counts"))?;
            Ok((
                "histogram",
                format!(
                    "bins={bins} counts={}",
                    fmt_f64_list(counts, QUIET_VEC_INLINE_MAX)
                ),
            ))
        }
        Operation::Covariance { .. } | Operation::Correlation { .. } => {
            quiet_covariance_correlation_matrix(op, ex)
        }
        _ => unreachable!("scalar_operation_display_special called for f64 op"),
    }
}

fn quiet_covariance_correlation_matrix(
    op: &Operation,
    ex: &QueryExecutionPreview,
) -> Result<(&'static str, String), String> {
    let label = op.wire_key();
    let (order, matrix) = match op {
        Operation::Covariance { .. } => (
            ex.operation_covariance_order,
            ex.operation_covariance.as_ref(),
        ),
        Operation::Correlation { .. } => (
            ex.operation_correlation_order,
            ex.operation_correlation.as_ref(),
        ),
        _ => unreachable!("quiet_covariance_correlation_matrix called for non-matrix op"),
    };
    let order = order.ok_or_else(|| missing_field(&format!("operation_{label}_order")))?;
    let mat = matrix.ok_or_else(|| missing_field(&format!("operation_{label}")))?;
    Ok((
        label,
        format!(
            "order={order} matrix={}",
            fmt_f64_list(mat, QUIET_VEC_INLINE_MAX)
        ),
    ))
}

pub(super) fn partial_operation_values(
    op: &Operation,
    ex: &QueryExecutionPreview,
) -> Result<String, String> {
    match op {
        Operation::Sum { .. } => {
            quiet_reduced_f64("operation_reduced_sum", ex.operation_reduced_sum.as_ref())
        }
        Operation::Mean { .. } => {
            quiet_reduced_f64("operation_reduced_mean", ex.operation_reduced_mean.as_ref())
        }
        Operation::Min { .. } => {
            quiet_reduced_f64("operation_reduced_min", ex.operation_reduced_min.as_ref())
        }
        Operation::Max { .. } => {
            quiet_reduced_f64("operation_reduced_max", ex.operation_reduced_max.as_ref())
        }
        Operation::Count { .. } => quiet_reduced_f64(
            "operation_reduced_count",
            ex.operation_reduced_count.as_ref(),
        ),
        Operation::Var { .. } => {
            quiet_reduced_f64("operation_reduced_var", ex.operation_reduced_var.as_ref())
        }
        Operation::Std { .. } => {
            quiet_reduced_f64("operation_reduced_std", ex.operation_reduced_std.as_ref())
        }
        Operation::Product { .. } => quiet_reduced_f64(
            "operation_reduced_product",
            ex.operation_reduced_product.as_ref(),
        ),
        Operation::NormL1 { .. } => quiet_reduced_f64(
            "operation_reduced_norm_l1",
            ex.operation_reduced_norm_l1.as_ref(),
        ),
        Operation::NormL2 { .. } => quiet_reduced_f64(
            "operation_reduced_norm_l2",
            ex.operation_reduced_norm_l2.as_ref(),
        ),
        Operation::AllFinite { .. } => quiet_reduced_bool(
            "operation_reduced_all_finite",
            ex.operation_reduced_all_finite.as_ref(),
        ),
        Operation::AnyNan { .. } => quiet_reduced_bool(
            "operation_reduced_any_nan",
            ex.operation_reduced_any_nan.as_ref(),
        ),
        Operation::AnyInf { .. } => quiet_reduced_bool(
            "operation_reduced_any_inf",
            ex.operation_reduced_any_inf.as_ref(),
        ),
        Operation::NanCount { .. } => quiet_reduced_f64(
            "operation_reduced_nan_count",
            ex.operation_reduced_nan_count.as_ref(),
        ),
        Operation::InfCount { .. } => quiet_reduced_f64(
            "operation_reduced_inf_count",
            ex.operation_reduced_inf_count.as_ref(),
        ),
        Operation::NullCount { .. } => quiet_reduced_f64(
            "operation_reduced_null_count",
            ex.operation_reduced_null_count.as_ref(),
        ),
        Operation::ArgMin { .. } => quiet_reduced_u64(
            "operation_reduced_argmin",
            ex.operation_reduced_argmin.as_ref(),
        ),
        Operation::ArgMax { .. } => quiet_reduced_u64(
            "operation_reduced_argmax",
            ex.operation_reduced_argmax.as_ref(),
        ),
        Operation::Median { .. } => quiet_reduced_f64(
            "operation_reduced_median",
            ex.operation_reduced_median.as_ref(),
        ),
        Operation::Quantile { .. } => quiet_reduced_f64(
            "operation_reduced_quantile",
            ex.operation_reduced_quantile.as_ref(),
        ),
        Operation::Histogram { .. } => quiet_reduced_f64(
            "operation_reduced_histogram_counts",
            ex.operation_reduced_histogram_counts.as_ref(),
        ),
        Operation::Covariance { .. } | Operation::Correlation { .. } => Err(
            "covariance/correlation are not partial-axis reductions (use rank-2 selection)"
                .to_string(),
        ),
        Operation::NanMean { .. } => quiet_reduced_f64(
            "operation_reduced_nan_mean",
            ex.operation_reduced_nan_mean.as_ref(),
        ),
        Operation::NanStd { .. } => quiet_reduced_f64(
            "operation_reduced_nan_std",
            ex.operation_reduced_nan_std.as_ref(),
        ),
        Operation::Transform { .. } => transform_partial_stats_line(ex),
    }
}

fn transform_partial_stats_line(ex: &QueryExecutionPreview) -> Result<String, String> {
    transform_stats_line(ex, true)
}

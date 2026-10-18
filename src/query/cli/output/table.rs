//! Human-readable ASCII tables for `tet query --format table`.

use std::fmt::Write as _;

use crate::query::cli::text::truncate_field;
use crate::query::types::{
    DatasetResolution, Operation, QueryExecutionPreview, QueryResponse, ReadPlan,
};

use super::format_num::{fmt_f64, missing_field};
use super::quiet;

const PREVIEW_TABLE_MAX: usize = 16;

pub(super) fn format_table_text(response: &QueryResponse) -> Result<String, String> {
    let mut out = String::new();
    push_summary_table(&mut out, response);

    if let Some(catalog) = response.catalog.as_ref()
        && !catalog.matched
    {
        push_catalog_miss_table(&mut out, catalog);
        return Ok(out);
    }

    if let Some(plan) = response.read_plan.as_ref() {
        push_plan_table(&mut out, plan);
    }

    if let Some(ex) = response.execution.as_ref() {
        push_execution_io_table(&mut out, ex);
        let shape = response
            .read_plan
            .as_ref()
            .map(|p| p.logical_selection_shape.as_slice());
        if let Some(op) = response.operation.as_ref() {
            push_operation_table(&mut out, op, ex, shape)?;
        } else {
            push_preview_table(&mut out, ex, shape);
        }
    }

    Ok(out)
}

fn push_summary_table(out: &mut String, response: &QueryResponse) {
    let mut rows = vec![
        ("dataset".to_string(), response.dataset.clone()),
        ("status".to_string(), response.status.to_string()),
        ("accepted".to_string(), response.accepted.to_string()),
    ];
    if let Some(n) = response.selection_axes {
        rows.push(("selection_axes".to_string(), n.to_string()));
    }
    if let Some(op) = response.operation.as_ref() {
        rows.push(("operation".to_string(), operation_label(op)));
    }
    if let Some(path) = response.tet_file.as_deref() {
        rows.push(("tet_file".to_string(), path.to_string()));
    }
    if !response.message.is_empty() {
        rows.push(("message".to_string(), truncate_field(&response.message, 72)));
    }
    write_kv_section(out, "query", &rows);
}

fn operation_label(op: &Operation) -> String {
    let name = op.wire_key();
    let axes = op.axes();
    if axes.is_empty() {
        format!("{name} (scalar)")
    } else {
        format!("{name} axes=[{}]", axes.join(","))
    }
}

fn push_catalog_miss_table(out: &mut String, catalog: &DatasetResolution) {
    write_kv_section(
        out,
        "catalog",
        &[("matched".to_string(), "false".to_string())],
    );
    let Some(names) = catalog.available_datasets.as_ref() else {
        return;
    };
    if names.is_empty() {
        out.push_str("datasets:\n  (none)\n\n");
        return;
    }
    out.push_str("datasets:\n");
    let _ = writeln!(out, "  {:>3}  name", "#");
    for (i, name) in names.iter().enumerate() {
        let _ = writeln!(out, "  {:>3}  {}", i, truncate_field(name, 40));
    }
    out.push('\n');
}

fn push_plan_table(out: &mut String, plan: &ReadPlan) {
    let shape = quiet::fmt_shape(&plan.logical_selection_shape);
    write_kv_section(
        out,
        "read_plan",
        &[
            (
                "chunk_touch_policy".to_string(),
                plan.chunk_touch_policy.to_string(),
            ),
            ("chunk_count".to_string(), plan.chunk_count.to_string()),
            (
                "total_stored_bytes".to_string(),
                plan.total_stored_bytes.to_string(),
            ),
            ("logical_shape".to_string(), shape),
            (
                "logical_elements".to_string(),
                plan.logical_element_count.to_string(),
            ),
        ],
    );
}

fn push_execution_io_table(out: &mut String, ex: &QueryExecutionPreview) {
    let mut rows = vec![(
        "bytes_read".to_string(),
        ex.total_bytes_read_from_disk.to_string(),
    )];
    if let Some(s) = ex.memory_strategy {
        rows.push(("memory_strategy".to_string(), s.to_string()));
    }
    if let Some(b) = ex.memory_budget_bytes {
        rows.push(("memory_budget_bytes".to_string(), b.to_string()));
    }
    if let Some(path) = ex.spill_f32_path.as_deref() {
        rows.push(("spill_path".to_string(), path.to_string()));
    }
    if let Some(b) = ex.spill_f32_bytes {
        rows.push(("spill_bytes".to_string(), b.to_string()));
    }
    if let Some(regime) = ex.io_regime {
        rows.push(("io_regime".to_string(), regime.to_string()));
    }
    write_kv_section(out, "execution", &rows);
}

fn push_operation_table(
    out: &mut String,
    op: &Operation,
    ex: &QueryExecutionPreview,
    shape: Option<&[u64]>,
) -> Result<(), String> {
    if op.axes().is_empty() {
        let (name, value) = quiet::scalar_operation_display(op, ex)?;
        let mut rows = vec![(name.to_string(), value)];
        if let Some(n) = ex.operation_element_count {
            rows.push(("elements".to_string(), n.to_string()));
        }
        write_kv_section(out, "result", &rows);
    } else {
        let values = partial_f64_values(op, ex)?;
        let title = if let Some(shape) = ex.operation_reduced_shape.as_ref() {
            format!("result (reduced_shape={})", quiet::fmt_shape(shape))
        } else {
            "result".to_string()
        };
        out.push_str(&title);
        out.push_str(":\n");
        write_indexed_f64_table(out, &values);
        if let Some(n) = ex.operation_element_count {
            let _ = writeln!(out, "  {:>12}  {}", "elements", n);
            out.push('\n');
        }
    }
    push_preview_table(out, ex, shape);
    Ok(())
}

fn partial_f64_values(op: &Operation, ex: &QueryExecutionPreview) -> Result<Vec<f64>, String> {
    let values = match op {
        Operation::Sum { .. } => ex.operation_reduced_sum.as_ref(),
        Operation::Mean { .. } => ex.operation_reduced_mean.as_ref(),
        Operation::Min { .. } => ex.operation_reduced_min.as_ref(),
        Operation::Max { .. } => ex.operation_reduced_max.as_ref(),
        Operation::Count { .. } => ex.operation_reduced_count.as_ref(),
        Operation::Var { .. } => ex.operation_reduced_var.as_ref(),
        Operation::Std { .. } => ex.operation_reduced_std.as_ref(),
        Operation::NanMean { .. } => ex.operation_reduced_nan_mean.as_ref(),
        Operation::NanStd { .. } => ex.operation_reduced_nan_std.as_ref(),
        Operation::Product { .. } => ex.operation_reduced_product.as_ref(),
        Operation::NormL1 { .. } => ex.operation_reduced_norm_l1.as_ref(),
        Operation::NormL2 { .. } => ex.operation_reduced_norm_l2.as_ref(),
        Operation::NanCount { .. } => ex.operation_reduced_nan_count.as_ref(),
        Operation::InfCount { .. } => ex.operation_reduced_inf_count.as_ref(),
        Operation::NullCount { .. } => ex.operation_reduced_null_count.as_ref(),
        Operation::Median { .. } => ex.operation_reduced_median.as_ref(),
        Operation::Quantile { .. } => ex.operation_reduced_quantile.as_ref(),
        Operation::Histogram { .. } => ex.operation_reduced_histogram_counts.as_ref(),
        Operation::AllFinite { .. } | Operation::AnyNan { .. } | Operation::AnyInf { .. } => {
            return Err(
                "table format: boolean partial reductions use --format quiet or stats".into(),
            );
        }
        Operation::ArgMin { .. } => {
            return Err("table format: arg_min partial values use --format quiet or stats".into());
        }
        Operation::ArgMax { .. } => {
            return Err("table format: arg_max partial values use --format quiet or stats".into());
        }
        Operation::Covariance { .. } | Operation::Correlation { .. } => {
            return Err(
                "table format: covariance/correlation matrices use --format stats or full".into(),
            );
        }
        Operation::Transform { .. } => {
            return Err(
                "table format: transforms use the slice preview (see f32_preview / f64_preview)"
                    .into(),
            );
        }
    };
    values
        .cloned()
        .ok_or_else(|| missing_field("operation_reduced_*"))
}

fn write_indexed_f64_table(out: &mut String, values: &[f64]) {
    let _ = writeln!(out, "  {:>6}  value", "#");
    for (i, v) in values.iter().enumerate() {
        let _ = writeln!(out, "  {:>6}  {}", i, fmt_f64(*v));
    }
    out.push('\n');
}

fn push_preview_table(out: &mut String, ex: &QueryExecutionPreview, shape: Option<&[u64]>) {
    let Some((label, values, truncated)) = preview_values(ex) else {
        return;
    };
    let shape_label = shape.map(quiet::fmt_shape).unwrap_or_default();
    let title = if truncated {
        format!("slice ({label}, shape={shape_label}, truncated)")
    } else {
        format!("slice ({label}, shape={shape_label})")
    };
    out.push_str(&title);
    out.push_str(":\n");
    let show = values.len().min(PREVIEW_TABLE_MAX);
    let values = &values[..show];
    if let Some(shape) = shape.filter(|s| !s.is_empty()) {
        write_slice_value_table(out, shape, values, truncated);
    } else {
        write_indexed_f64_table(out, values);
    }
    if truncated {
        let _ = writeln!(out, "  … logical tensor continues beyond preview cap");
    }
    out.push('\n');
}

/// Row-major value grid for 1D/2D logical shapes; linear index table for higher rank.
fn write_slice_value_table(out: &mut String, shape: &[u64], values: &[f64], truncated: bool) {
    let _ = truncated;
    match shape.len() {
        1 => write_1d_slice_row(out, shape[0] as usize, values),
        2 => write_2d_slice_grid(out, shape[0] as usize, shape[1] as usize, values),
        _ => {
            let _ = writeln!(
                out,
                "  (rank {} — linear index; use --preview to raise cap)",
                shape.len()
            );
            write_indexed_f64_table(out, values);
        }
    }
}

fn write_1d_slice_row(out: &mut String, len: usize, values: &[f64]) {
    let cols = len.min(values.len());
    out.push_str("        ");
    for c in 0..cols {
        let _ = write!(out, "  {:>8}", format!("c{c}"));
    }
    out.push('\n');
    out.push_str("     r0 ");
    for v in values.iter().take(cols) {
        let _ = write!(out, "  {:>8}", fmt_f64(*v));
    }
    out.push('\n');
}

fn write_2d_slice_grid(out: &mut String, rows: usize, cols: usize, values: &[f64]) {
    out.push_str("        ");
    for c in 0..cols {
        let _ = write!(out, "  {:>8}", format!("c{c}"));
    }
    out.push('\n');
    for r in 0..rows {
        let _ = write!(out, "  {:>4} ", format!("r{r}"));
        for c in 0..cols {
            let i = r * cols + c;
            let cell = values
                .get(i)
                .map_or_else(|| "—".to_string(), |v| fmt_f64(*v));
            let _ = write!(out, "  {cell:>8}");
        }
        out.push('\n');
    }
}

fn preview_values(ex: &QueryExecutionPreview) -> Option<(&'static str, Vec<f64>, bool)> {
    if !ex.f32_preview.is_empty() {
        return Some((
            "f32",
            ex.f32_preview.iter().map(|v| f64::from(*v)).collect(),
            ex.f32_preview_truncated,
        ));
    }
    if !ex.f64_preview.is_empty() {
        return Some(("f64", ex.f64_preview.clone(), ex.f64_preview_truncated));
    }
    if !ex.i32_preview.is_empty() {
        return Some((
            "i32",
            ex.i32_preview.iter().map(|v| f64::from(*v)).collect(),
            ex.i32_preview_truncated,
        ));
    }
    if !ex.u8_preview.is_empty() {
        return Some((
            "u8",
            ex.u8_preview.iter().map(|v| f64::from(*v)).collect(),
            ex.u8_preview_truncated,
        ));
    }
    None
}

fn write_kv_section(out: &mut String, title: &str, rows: &[(String, String)]) {
    if rows.is_empty() {
        return;
    }
    let _ = writeln!(out, "{title}:");
    let key_width = rows.iter().map(|(k, _)| k.len()).max().unwrap_or(4).max(4);
    for (key, value) in rows {
        let _ = writeln!(out, "  {key:key_width$}  {value}");
    }
    out.push('\n');
}

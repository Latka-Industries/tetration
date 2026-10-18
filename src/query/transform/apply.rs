//! Pass-2 apply transforms while decoding the logical selection.
//!
//! Each element is rewritten using pass-1 stats ([`TransformStats`]). Non-finite inputs
//! are preserved; invalid divides and negative `sqrt` shifts become **NaN** and append
//! to [`TransformWarnings`].

use std::cell::RefCell;
use std::path::Path;

use crate::query::decode::chunk_decode::{visit_planned_chunk, visit_planned_chunk_f64};
use crate::query::fold::partial_geometry;
use crate::query::materialize::shared::{
    scatter_fill_chunks, spill_byte_len_from_elem_count, spill_read_plan_pod_le_impl,
};
use crate::query::types::{ReadPlan, TetError, TransformMethod};
use crate::utils::{f32_le, f64_le};

use super::stats::TransformStats;
use super::warnings::TransformWarnings;

/// Run scatter-fill with a shared [`RefCell`] so chunk visitors can record warnings.
fn with_warnings_cell<R>(
    warnings: &mut TransformWarnings,
    f: impl FnOnce(&RefCell<TransformWarnings>) -> Result<R, TetError>,
) -> Result<R, TetError> {
    let cell = RefCell::new(std::mem::take(warnings));
    let out = f(&cell)?;
    *warnings = cell.into_inner();
    Ok(out)
}

/// Divide when `denom` is finite and non-zero; otherwise record the logical index and return NaN.
fn div_or_nan(numer: f64, denom: f64, li: usize, warnings: &mut TransformWarnings) -> f64 {
    if denom == 0.0 || !denom.is_finite() {
        warnings.record_div_by_zero(li as u64);
        return f64::NAN;
    }
    numer / denom
}

/// Apply `stats.method` to one finite `f64` sample at logical index `li`.
fn transform_f64(
    x: f64,
    li: usize,
    shape: &[u64],
    stats: &TransformStats,
    warnings: &mut TransformWarnings,
) -> Result<f64, TetError> {
    if !x.is_finite() {
        return Ok(x);
    }
    let cell = partial_geometry::reduced_cell_index_or_global(li, shape, stats.layout.as_ref())?;
    Ok(match stats.method {
        TransformMethod::Center => x - stats.mean[cell],
        TransformMethod::Zscore => {
            let mean = stats.mean[cell];
            let std = stats.std[cell];
            div_or_nan(x - mean, std, li, warnings)
        }
        TransformMethod::Scale => div_or_nan(x, stats.std[cell], li, warnings),
        TransformMethod::Minmax => {
            let min = stats.min[cell];
            let range = stats.max[cell] - min;
            div_or_nan(x - min, range, li, warnings)
        }
        TransformMethod::L1 => div_or_nan(x, stats.norm_l1[cell], li, warnings),
        TransformMethod::L2 => div_or_nan(x, stats.norm_l2[cell], li, warnings),
        TransformMethod::Log1p => (x - stats.min[cell]).ln_1p(),
        TransformMethod::Sqrt => {
            let shifted = x - stats.min[cell];
            if shifted < 0.0 {
                warnings.record_div_by_zero(li as u64);
                f64::NAN
            } else {
                shifted.sqrt()
            }
        }
        TransformMethod::Softmax => {
            let max = stats.max[cell];
            let sum_exp = stats.sum_exp[cell];
            if sum_exp == 0.0 || !sum_exp.is_finite() {
                warnings.record_div_by_zero(li as u64);
                f64::NAN
            } else {
                (x - max).exp() / sum_exp
            }
        }
    })
}

fn transform_scatter_fill_f32(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [f32],
    stats: &TransformStats,
    warnings: &RefCell<TransformWarnings>,
) -> Result<u64, TetError> {
    let shape = plan.logical_selection_shape.clone();
    scatter_fill_chunks(mmap, plan, out, |mmap, plan, c, out| {
        visit_planned_chunk(mmap, plan, c, |li, v| {
            let y =
                transform_f64(f64::from(v), li, &shape, stats, &mut warnings.borrow_mut())? as f32;
            out[li] = y;
            Ok(())
        })
    })
}

fn transform_scatter_fill_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [f64],
    stats: &TransformStats,
    warnings: &RefCell<TransformWarnings>,
) -> Result<u64, TetError> {
    let shape = plan.logical_selection_shape.clone();
    scatter_fill_chunks(mmap, plan, out, |mmap, plan, c, out| {
        visit_planned_chunk_f64(mmap, plan, c, |li, v| {
            out[li] = transform_f64(v, li, &shape, stats, &mut warnings.borrow_mut())?;
            Ok(())
        })
    })
}

/// Materialize the transformed logical selection into a dense `f32` vector.
///
/// # Errors
///
/// Propagates materialize and geometry failures.
pub(crate) fn transform_read_plan_f32_le_ram(
    mmap: &[u8],
    plan: &ReadPlan,
    stats: &TransformStats,
    warnings: &mut TransformWarnings,
) -> Result<(Vec<f32>, u64), TetError> {
    with_warnings_cell(warnings, |cell| {
        let n = plan.logical_element_count;
        let mut out = vec![f32::NAN; n];
        let bytes = transform_scatter_fill_f32(mmap, plan, &mut out, stats, cell)?;
        Ok((out, bytes))
    })
}

/// Materialize the transformed logical selection into a dense `f64` vector.
///
/// # Errors
///
/// Propagates materialize and geometry failures.
pub(crate) fn transform_read_plan_f64_le_ram(
    mmap: &[u8],
    plan: &ReadPlan,
    stats: &TransformStats,
    warnings: &mut TransformWarnings,
) -> Result<(Vec<f64>, u64), TetError> {
    with_warnings_cell(warnings, |cell| {
        let n = plan.logical_element_count;
        let mut out = vec![f64::NAN; n];
        let bytes = transform_scatter_fill_f64(mmap, plan, &mut out, stats, cell)?;
        Ok((out, bytes))
    })
}

/// Spill the transformed logical selection as row-major little-endian POD to `path`.
///
/// # Errors
///
/// Propagates materialize and I/O failures.
pub(crate) fn transform_spill_f32_le(
    mmap: &[u8],
    plan: &ReadPlan,
    path: &Path,
    stats: &TransformStats,
    warnings: &mut TransformWarnings,
) -> Result<u64, TetError> {
    with_warnings_cell(warnings, |cell| {
        let byte_len = spill_byte_len_from_elem_count(
            plan.logical_element_count,
            f32_le::bytes_from_elem_count,
        )?;
        spill_read_plan_pod_le_impl(
            mmap,
            plan,
            path,
            byte_len,
            |mmap, plan, out| transform_scatter_fill_f32(mmap, plan, out, stats, cell),
            None, // intentional NaNs from div-by-zero; skip materialized completeness check
        )
    })
}

/// Spill the transformed logical selection as row-major `f64` LE to `path`.
///
/// # Errors
///
/// Propagates materialize and I/O failures.
pub(crate) fn transform_spill_f64_le(
    mmap: &[u8],
    plan: &ReadPlan,
    path: &Path,
    stats: &TransformStats,
    warnings: &mut TransformWarnings,
) -> Result<u64, TetError> {
    with_warnings_cell(warnings, |cell| {
        let byte_len = spill_byte_len_from_elem_count(
            plan.logical_element_count,
            f64_le::bytes_from_elem_count,
        )?;
        spill_read_plan_pod_le_impl(
            mmap,
            plan,
            path,
            byte_len,
            |mmap, plan, out| transform_scatter_fill_f64(mmap, plan, out, stats, cell),
            None, // intentional NaNs from div-by-zero; skip materialized completeness check
        )
    })
}

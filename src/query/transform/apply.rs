//! Pass-2 apply transforms while decoding the logical selection.
//!
//! Each element is rewritten using pass-1 stats ([`TransformStats`]). Non-finite inputs
//! are preserved; invalid divides and negative `sqrt` shifts become **NaN** and append
//! to [`TransformWarnings`]. Multi-chunk plans use Rayon scatter (disjoint logical
//! indices), matching parallel fold I/O.

use std::cell::RefCell;
use std::path::Path;

use rayon::prelude::*;

use crate::query::decode::chunk_decode::{visit_planned_chunk, visit_planned_chunk_f64};
use crate::query::dispatch::accumulate_chunk_read_bytes;
use crate::query::fold::parallel::disjoint_buffer_mut;
use crate::query::fold::partial_geometry;
use crate::query::materialize::shared::{
    scatter_fill_chunks, spill_byte_len_from_elem_count, spill_read_plan_pod_le_impl,
};
use crate::query::materialize::validate_read_plan_geometry;
use crate::query::types::{PlannedChunkIo, ReadPlan, TetError, TransformMethod, fold_keeps_cell};
use crate::utils::{f32_le, f64_le};

use super::stats::TransformStats;
use super::warnings::TransformWarnings;

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

/// Shared sequential / parallel scatter for transform pass-2.
fn transform_scatter_fill<T, V>(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [T],
    warnings: &mut TransformWarnings,
    visit_chunk: V,
) -> Result<u64, TetError>
where
    T: Copy + Send,
    V: Fn(
            &[u8],
            &ReadPlan,
            &PlannedChunkIo,
            &mut [T],
            &mut TransformWarnings,
        ) -> Result<u64, TetError>
        + Sync,
{
    if plan.chunks.len() <= 1 {
        return scatter_fill_chunks(mmap, plan, out, |mmap, plan, c, out| {
            visit_chunk(mmap, plan, c, out, warnings)
        });
    }
    validate_read_plan_geometry(plan, out.len())?;
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let parts: Vec<(u64, TransformWarnings)> = plan
        .chunks
        .par_iter()
        .map(|c| {
            let mut local = TransformWarnings::default();
            let slice = disjoint_buffer_mut::<T>(out_addr, out_len);
            let bytes = visit_chunk(mmap, plan, c, slice, &mut local)?;
            Ok((bytes, local))
        })
        .collect::<Result<Vec<_>, TetError>>()?;
    let mut total = 0u64;
    for (bytes, local) in parts {
        accumulate_chunk_read_bytes(&mut total, bytes)?;
        warnings.merge_from(&local);
    }
    Ok(total)
}

fn transform_scatter_fill_f32(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [f32],
    stats: &TransformStats,
    warnings: &mut TransformWarnings,
) -> Result<u64, TetError> {
    let shape = plan.logical_selection_shape.clone();
    transform_scatter_fill(mmap, plan, out, warnings, |mmap, plan, c, out, warnings| {
        visit_planned_chunk(mmap, plan, c, |li, v| {
            if !fold_keeps_cell(plan.cell_filter, f64::from(v)) {
                out[li] = f32::NAN;
                return Ok(());
            }
            out[li] = transform_f64(f64::from(v), li, &shape, stats, warnings)? as f32;
            Ok(())
        })
    })
}

fn transform_scatter_fill_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [f64],
    stats: &TransformStats,
    warnings: &mut TransformWarnings,
) -> Result<u64, TetError> {
    let shape = plan.logical_selection_shape.clone();
    transform_scatter_fill(mmap, plan, out, warnings, |mmap, plan, c, out, warnings| {
        visit_planned_chunk_f64(mmap, plan, c, |li, v| {
            if !fold_keeps_cell(plan.cell_filter, v) {
                out[li] = f64::NAN;
                return Ok(());
            }
            out[li] = transform_f64(v, li, &shape, stats, warnings)?;
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
    let n = plan.logical_element_count;
    let mut out = vec![f32::NAN; n];
    let bytes = transform_scatter_fill_f32(mmap, plan, &mut out, stats, warnings)?;
    Ok((out, bytes))
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
    let n = plan.logical_element_count;
    let mut out = vec![f64::NAN; n];
    let bytes = transform_scatter_fill_f64(mmap, plan, &mut out, stats, warnings)?;
    Ok((out, bytes))
}

/// Spill helper: `spill_read_plan_pod_le_impl` requires `Fn`, so warnings live in a [`RefCell`].
fn spill_transformed<T, F>(
    mmap: &[u8],
    plan: &ReadPlan,
    path: &Path,
    byte_len: u64,
    warnings: &mut TransformWarnings,
    scatter: F,
) -> Result<u64, TetError>
where
    T: bytemuck::Pod + Copy,
    F: Fn(&[u8], &ReadPlan, &mut [T], &mut TransformWarnings) -> Result<u64, TetError>,
{
    let bag = RefCell::new(std::mem::take(warnings));
    let bytes = spill_read_plan_pod_le_impl(
        mmap,
        plan,
        path,
        byte_len,
        |mmap, plan, out| scatter(mmap, plan, out, &mut bag.borrow_mut()),
        None, // intentional NaNs from div-by-zero; skip materialized completeness check
    )?;
    *warnings = bag.into_inner();
    Ok(bytes)
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
    let byte_len =
        spill_byte_len_from_elem_count(plan.logical_element_count, f32_le::bytes_from_elem_count)?;
    spill_transformed(
        mmap,
        plan,
        path,
        byte_len,
        warnings,
        |mmap, plan, out, warnings| transform_scatter_fill_f32(mmap, plan, out, stats, warnings),
    )
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
    let byte_len =
        spill_byte_len_from_elem_count(plan.logical_element_count, f64_le::bytes_from_elem_count)?;
    spill_transformed(
        mmap,
        plan,
        path,
        byte_len,
        warnings,
        |mmap, plan, out, warnings| transform_scatter_fill_f64(mmap, plan, out, stats, warnings),
    )
}

//! Per-element visitation and scalar fold over prepared / planned chunks.

use crate::catalog::{MAX_NDIM, tile};
use crate::query::decode::dense_visit::{dense_tile_logical_base, logical_index_unit_step};
use crate::query::decode::indexing::linear_rm_index;
use crate::query::fold::reduction::{ArgIndexAccum, ReductionKind, ValueAccum};
use crate::query::types::{PlannedChunkIo, ReadPlan, TetError, fold_keeps_cell};

use super::elem::{ChunkElem, FoldChunkElem};
use super::prepare::{PreparedChunk, prepare_chunk};

pub(super) fn global_matches_strided_selection(g: &[u64], plan: &ReadPlan) -> bool {
    (0..g.len()).all(|d| {
        let x = g[d];
        let st = plan.selection_box_start[d];
        let hi = plan.selection_box_stop_exclusive[d];
        let step = plan.selection_step[d];
        x >= st && x < hi && (x - st).is_multiple_of(step)
    })
}

pub(super) fn logical_index_for_global(
    plan: &ReadPlan,
    global: &[u64],
    ndim: usize,
) -> Result<usize, TetError> {
    if plan.selection_step.iter().all(|&s| s == 1) {
        return logical_index_unit_step(plan, global, ndim);
    }
    let mut lc = [0usize; MAX_NDIM];
    for (d, &x) in global.iter().enumerate().take(ndim) {
        let st = plan.selection_box_start[d];
        let q = (x - st) / plan.selection_step[d];
        lc[d] = usize::try_from(q).map_err(|_| {
            TetError::Validation(format!(
                "logical coordinate on axis {d} does not fit usize on this host"
            ))
        })?;
    }
    linear_rm_index(&lc[..ndim], &plan.logical_selection_shape)
}

fn visit_prepared_chunk_strided<E, F>(
    prep: &PreparedChunk<'_>,
    plan: &ReadPlan,
    mut visit: F,
) -> Result<(), TetError>
where
    E: ChunkElem,
    F: FnMut(usize, E) -> Result<(), TetError>,
{
    let ndim = prep.ndim;
    let mut global = [0u64; MAX_NDIM];
    for k in 0..prep.nelem {
        let local = tile::local_coords_from_linear(k as u64, &prep.tile, ndim);
        for (d, gv) in global.iter_mut().enumerate().take(ndim) {
            *gv = prep.chunk_coord[d]
                .saturating_mul(plan.chunk_shape[d])
                .saturating_add(local[d]);
        }
        if !global_matches_strided_selection(&global[..ndim], plan) {
            continue;
        }
        let li = logical_index_for_global(plan, &global[..ndim], ndim)?;
        visit(li, E::read_at(&prep.raw_bytes, k))?;
    }
    Ok(())
}

fn visit_prepared_chunk<E, F>(
    prep: &PreparedChunk<'_>,
    plan: &ReadPlan,
    mut visit: F,
) -> Result<(), TetError>
where
    E: ChunkElem,
    F: FnMut(usize, E) -> Result<(), TetError>,
{
    if prep.ndim == 1
        && let Some(li_base) =
            dense_tile_logical_base(plan, &prep.chunk_coord[..prep.ndim], &prep.tile)?
    {
        for k in 0..prep.nelem {
            visit(li_base + k, E::read_at(&prep.raw_bytes, k))?;
        }
        return Ok(());
    }
    visit_prepared_chunk_strided::<E, _>(prep, plan, visit)
}

pub(super) fn visit_planned_chunk_elem<E, F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    E: ChunkElem,
    F: FnMut(usize, E) -> Result<(), TetError>,
{
    let prep = prepare_chunk::<E>(mmap, plan, c)?;
    visit_prepared_chunk::<E, _>(&prep, plan, visit)?;
    Ok(prep.bytes_read)
}

fn fold_planned_chunk_elem<E>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    kind: ReductionKind,
    value: &mut ValueAccum,
    arg: &mut ArgIndexAccum,
    preview: &mut [E],
) -> Result<u64, TetError>
where
    E: FoldChunkElem,
{
    let prep = prepare_chunk::<E>(mmap, plan, c)?;
    let preview_len = preview.len();
    let filter_cells = plan.cell_filter.is_some();
    if prep.ndim == 1
        && let Some(li_base) =
            dense_tile_logical_base(plan, &prep.chunk_coord[..prep.ndim], &prep.tile)?
    {
        if filter_cells {
            for k in 0..prep.nelem {
                let v = E::read_at(&prep.raw_bytes, k);
                let li = li_base + k;
                record_fold_cell(plan, kind, value, arg, preview, li, v);
            }
        } else {
            match kind {
                ReductionKind::ArgMin | ReductionKind::ArgMax => {
                    for k in 0..prep.nelem {
                        let v = E::read_at(&prep.raw_bytes, k);
                        let li = li_base + k;
                        E::push_arg(arg, li as u64, v, kind);
                        if li < preview_len {
                            preview[li] = v;
                        }
                    }
                }
                _ => {
                    E::push_le_bytes(value, &prep.raw_bytes, kind);
                    if preview_len > 0 {
                        let cap = preview_len.saturating_sub(li_base).min(prep.nelem);
                        for k in 0..cap {
                            preview[li_base + k] = E::read_at(&prep.raw_bytes, k);
                        }
                    }
                }
            }
        }
        return Ok(prep.bytes_read);
    }
    visit_prepared_chunk::<E, _>(&prep, plan, |li, v| {
        record_fold_cell(plan, kind, value, arg, preview, li, v);
        Ok(())
    })?;
    Ok(prep.bytes_read)
}

pub(super) fn record_fold_cell<E: FoldChunkElem>(
    plan: &ReadPlan,
    kind: ReductionKind,
    value: &mut ValueAccum,
    arg: &mut ArgIndexAccum,
    preview: &mut [E],
    li: usize,
    v: E,
) {
    if li < preview.len() {
        preview[li] = v;
    }
    if !fold_keeps_cell(plan.cell_filter, E::as_f64(v)) {
        return;
    }
    match kind {
        ReductionKind::ArgMin | ReductionKind::ArgMax => E::push_arg(arg, li as u64, v, kind),
        ReductionKind::NanCount => value.push_nan_f64(E::as_f64(v)),
        ReductionKind::InfCount => value.push_inf_f64(E::as_f64(v)),
        ReductionKind::NullCount { fill } => value.push_null_f64(E::as_f64(v), fill),
        ReductionKind::NanMean => value.push_nan_mean_f64(E::as_f64(v)),
        ReductionKind::NanStd => value.push_nan_std_f64(E::as_f64(v)),
        _ => E::push_value(value, v),
    }
}

/// Scalar fold over one planned `f32` chunk without per-element visitor callbacks.
pub(crate) fn fold_planned_chunk_f32(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    kind: ReductionKind,
    value: &mut ValueAccum,
    arg: &mut ArgIndexAccum,
    preview: &mut [f32],
) -> Result<u64, TetError> {
    fold_planned_chunk_elem::<f32>(mmap, plan, c, kind, value, arg, preview)
}

/// Scalar fold over one planned `f64` chunk without per-element visitor callbacks.
pub(crate) fn fold_planned_chunk_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    kind: ReductionKind,
    value: &mut ValueAccum,
    arg: &mut ArgIndexAccum,
    preview: &mut [f64],
) -> Result<u64, TetError> {
    fold_planned_chunk_elem::<f64>(mmap, plan, c, kind, value, arg, preview)
}

/// Visit each selected `f32` in a planned chunk.
pub(crate) fn visit_planned_chunk<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f32) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<f32, _>(mmap, plan, c, visit)
}

/// Visit each selected `f64` in a planned chunk.
pub(crate) fn visit_planned_chunk_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<f64, _>(mmap, plan, c, visit)
}

/// Visit selected `f32` or `f64` samples in a planned chunk, promoting to `f64`.
///
/// Shared by transform softmax pass-1 and window folds that accumulate in `f64`.
///
/// # Errors
///
/// Returns [`TetError::Validation`] when `dtype` is not `f32`/`f64`, or propagates
/// decode / visit failures from the typed visitors.
pub(crate) fn visit_planned_chunk_float_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    dtype: crate::utils::dtype::ElementDtype,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    match dtype {
        crate::utils::dtype::ElementDtype::F32 => {
            visit_planned_chunk(mmap, plan, c, |li, v| visit(li, f64::from(v)))
        }
        crate::utils::dtype::ElementDtype::F64 => visit_planned_chunk_f64(mmap, plan, c, visit),
        _ => Err(TetError::Validation(
            "visit_planned_chunk_float_as_f64 requires f32 or f64".into(),
        )),
    }
}

/// Visit each selected `i32` in a planned chunk.
pub(crate) fn visit_planned_chunk_i32<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, i32) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<i32, _>(mmap, plan, c, visit)
}

/// Visit each selected `i64` in a planned chunk.
pub(crate) fn visit_planned_chunk_i64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, i64) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<i64, _>(mmap, plan, c, visit)
}

/// Visit each selected `i32`, promoting values to `f64` for accumulators.
pub(crate) fn visit_planned_chunk_i32_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_i32(mmap, plan, c, |li, v| visit(li, f64::from(v)))
}

/// Visit each selected `i64`, promoting values to `f64` for accumulators.
pub(crate) fn visit_planned_chunk_i64_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_i64(mmap, plan, c, |li, v| visit(li, v as f64))
}

/// Visit each selected `u8` in a planned chunk.
pub(crate) fn visit_planned_chunk_u8<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, u8) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<u8, _>(mmap, plan, c, visit)
}

/// Visit each selected `u8`, promoting values to `f64` for accumulators.
pub(crate) fn visit_planned_chunk_u8_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_u8(mmap, plan, c, |li, v| visit(li, f64::from(v)))
}

/// Visit each selected `u16` in a planned chunk.
pub(crate) fn visit_planned_chunk_u16<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, u16) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<u16, _>(mmap, plan, c, visit)
}

/// Visit each selected `u16`, promoting values to `f64` for accumulators.
pub(crate) fn visit_planned_chunk_u16_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_u16(mmap, plan, c, |li, v| visit(li, f64::from(v)))
}

/// Visit each selected `i16` in a planned chunk.
pub(crate) fn visit_planned_chunk_i16<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, i16) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<i16, _>(mmap, plan, c, visit)
}

/// Visit each selected `i16`, promoting values to `f64` for accumulators.
pub(crate) fn visit_planned_chunk_i16_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_i16(mmap, plan, c, |li, v| visit(li, f64::from(v)))
}

pub(crate) fn visit_planned_chunk_u32<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, u32) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<u32, _>(mmap, plan, c, visit)
}

pub(crate) fn visit_planned_chunk_u32_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_u32(mmap, plan, c, |li, v| visit(li, f64::from(v)))
}

pub(crate) fn visit_planned_chunk_u64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, u64) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<u64, _>(mmap, plan, c, visit)
}

pub(crate) fn visit_planned_chunk_u64_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_u64(mmap, plan, c, |li, v| visit(li, v as f64))
}

pub(crate) fn visit_planned_chunk_f16<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, half::f16) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<half::f16, _>(mmap, plan, c, visit)
}

#[allow(dead_code)]
pub(crate) fn visit_planned_chunk_f16_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_f16(mmap, plan, c, |li, v| visit(li, f64::from(v)))
}

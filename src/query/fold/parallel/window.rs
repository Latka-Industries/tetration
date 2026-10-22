//! Parallel chunk-streaming windowed fold (Rayon).
//!
//! Same worker / merge shape as partial-axis parallel folds: each chunk owns a local
//! accumulator replica + `real_counts`, then merges after the Rayon join. Gated by
//! [`parallel_window_memory_ok`] so huge `window_count` stays on the sequential path.

use rayon::prelude::*;

use super::merge::{
    PartialChunkArg, PartialChunkValue, merge_partial_arg_cells, merge_partial_value_cells,
};
use super::preview::{
    write_disjoint_preview_f16, write_disjoint_preview_f32, write_disjoint_preview_f64,
    write_disjoint_preview_i16, write_disjoint_preview_i32, write_disjoint_preview_i64,
    write_disjoint_preview_u8, write_disjoint_preview_u16, write_disjoint_preview_u32,
    write_disjoint_preview_u64,
};
use super::workers::with_fold_workers;
use crate::query::decode::{chunk_decode, indexing};
use crate::query::fold::window_geometry::{self, WindowAxisLayout};
use crate::query::fold::{reduction, shared};
use crate::query::materialize::int::IntVisit;
use crate::query::types::{ReadPlan, TetError, fold_keeps_element};
use crate::utils::dtype::ElementDtype;

/// Soft cap on per-worker accumulator replicas (~64 MiB) so huge `window_count`
/// stays on the single-buffer sequential path (memory-conscious).
const MAX_PARALLEL_REPLICA_BYTES: usize = 64 * 1024 * 1024;

/// Whether per-worker accumulator replicas fit under [`MAX_PARALLEL_REPLICA_BYTES`].
#[must_use]
pub(crate) fn parallel_window_memory_ok(out_len: usize, workers: Option<usize>) -> bool {
    let threads = workers.unwrap_or_else(rayon::current_num_threads).max(1);
    let cell = std::mem::size_of::<reduction::ValueAccum>()
        .max(std::mem::size_of::<reduction::ArgIndexAccum>());
    // One ValueAccum/ArgIndexAccum + one real_counts u64 per output cell, per worker.
    out_len
        .saturating_mul(threads)
        .saturating_mul(cell.saturating_add(std::mem::size_of::<u64>()))
        <= MAX_PARALLEL_REPLICA_BYTES
}

/// Merged parallel scan result — caller injects pad, applies `min_count`, finishes.
pub(crate) struct WindowParallelScan {
    pub value_cells: Option<Vec<reduction::ValueAccum>>,
    pub arg_cells: Option<Vec<reduction::ArgIndexAccum>>,
    pub real_counts: Vec<u64>,
    pub total_bytes: u64,
    pub preview: shared::FoldPreviewBuffer,
}

#[derive(Copy, Clone)]
enum WindowVisit {
    F32,
    F64,
    F16,
    Int(IntVisit),
}

impl WindowVisit {
    fn from_dtype(dtype: ElementDtype) -> Self {
        match dtype {
            ElementDtype::F32 => Self::F32,
            ElementDtype::F64 => Self::F64,
            ElementDtype::F16 => Self::F16,
            ElementDtype::I32 => Self::Int(IntVisit::I32),
            ElementDtype::I64 => Self::Int(IntVisit::I64),
            ElementDtype::U8 => Self::Int(IntVisit::U8),
            ElementDtype::U16 => Self::Int(IntVisit::U16),
            ElementDtype::I16 => Self::Int(IntVisit::I16),
            ElementDtype::U32 => Self::Int(IntVisit::U32),
            ElementDtype::U64 => Self::Int(IntVisit::U64),
        }
    }

    fn visit_chunk<F>(
        self,
        mmap: &[u8],
        plan: &ReadPlan,
        c: &crate::query::types::PlannedChunkIo,
        mut visit: F,
    ) -> Result<u64, TetError>
    where
        F: FnMut(usize, f64) -> Result<(), TetError>,
    {
        match self {
            Self::F32 => chunk_decode::visit_planned_chunk_float_as_f64(
                mmap,
                plan,
                c,
                ElementDtype::F32,
                visit,
            ),
            Self::F64 => chunk_decode::visit_planned_chunk_float_as_f64(
                mmap,
                plan,
                c,
                ElementDtype::F64,
                visit,
            ),
            Self::F16 => chunk_decode::visit_planned_chunk_f16(mmap, plan, c, |li, v| {
                visit(li, f64::from(v))
            }),
            Self::Int(v) => v.visit_chunk_as_f64(mmap, plan, c, visit),
        }
    }

    fn write_preview(self, preview_addr: usize, preview_len: usize, li: usize, v: f64) {
        match self {
            Self::F32 => write_disjoint_preview_f32(preview_addr, preview_len, li, v as f32),
            Self::F64 => write_disjoint_preview_f64(preview_addr, preview_len, li, v),
            Self::F16 => write_disjoint_preview_f16(preview_addr, preview_len, li, v),
            Self::Int(IntVisit::I32) => {
                write_disjoint_preview_i32(preview_addr, preview_len, li, v);
            }
            Self::Int(IntVisit::I64) => {
                write_disjoint_preview_i64(preview_addr, preview_len, li, v);
            }
            Self::Int(IntVisit::U8) => write_disjoint_preview_u8(preview_addr, preview_len, li, v),
            Self::Int(IntVisit::U16) => {
                write_disjoint_preview_u16(preview_addr, preview_len, li, v);
            }
            Self::Int(IntVisit::I16) => {
                write_disjoint_preview_i16(preview_addr, preview_len, li, v);
            }
            Self::Int(IntVisit::U32) => {
                write_disjoint_preview_u32(preview_addr, preview_len, li, v);
            }
            Self::Int(IntVisit::U64) => {
                write_disjoint_preview_u64(preview_addr, preview_len, li, v);
            }
        }
    }
}

struct WindowParallelCtx<'a> {
    mmap: &'a [u8],
    plan: &'a ReadPlan,
    visit: WindowVisit,
    kind: reduction::ReductionKind,
    layout: &'a WindowAxisLayout,
    shape: &'a [u64],
    preview_addr: usize,
    preview_len: usize,
    rank1_full: bool,
}

struct WindowChunkValue {
    inner: PartialChunkValue,
    real_counts: Vec<u64>,
}

struct WindowChunkArg {
    inner: PartialChunkArg,
    real_counts: Vec<u64>,
}

/// Push `v` into every covering window cell for logical index `li`.
fn scatter_value(
    ctx: &WindowParallelCtx<'_>,
    cells: &mut [reduction::ValueAccum],
    real_counts: &mut [u64],
    li: usize,
    v: f64,
) -> Result<(), TetError> {
    // Rank-1 full reduce: `li` is the window-axis coord and output index is `w`.
    if ctx.rank1_full {
        let axis_c =
            u64::try_from(li).map_err(|_| TetError::Validation("axis coord overflow".into()))?;
        let (w0, w1) = window_geometry::covering_window_bounds(axis_c, ctx.layout)?;
        for (cell, count) in cells[w0..w1].iter_mut().zip(&mut real_counts[w0..w1]) {
            cell.push_f64(v);
            *count = count.saturating_add(1);
        }
        return Ok(());
    }
    let coords = indexing::coords_from_linear_row_major(li, ctx.shape)?;
    let axis_c = u64::try_from(coords[ctx.layout.window_axis])
        .map_err(|_| TetError::Validation("axis coord overflow".into()))?;
    let (w0, w1) = window_geometry::covering_window_bounds(axis_c, ctx.layout)?;
    for w in w0..w1 {
        let oi = window_geometry::window_reduced_index(&coords, w, ctx.layout)?;
        cells[oi].push_f64(v);
        real_counts[oi] = real_counts[oi].saturating_add(1);
    }
    Ok(())
}

/// Push arg-index `(fiber, v)` into every covering window cell for logical index `li`.
fn scatter_arg(
    ctx: &WindowParallelCtx<'_>,
    cells: &mut [reduction::ArgIndexAccum],
    real_counts: &mut [u64],
    li: usize,
    v: f64,
) -> Result<(), TetError> {
    if ctx.rank1_full {
        let axis_c =
            u64::try_from(li).map_err(|_| TetError::Validation("axis coord overflow".into()))?;
        let (w0, w1) = window_geometry::covering_window_bounds(axis_c, ctx.layout)?;
        for (off, (cell, count)) in cells[w0..w1]
            .iter_mut()
            .zip(&mut real_counts[w0..w1])
            .enumerate()
        {
            let w = w0 + off;
            let fiber = window_geometry::window_fiber_index(axis_c, w, ctx.layout)?;
            cell.push_f64(fiber, v, ctx.kind);
            *count = count.saturating_add(1);
        }
        return Ok(());
    }
    let coords = indexing::coords_from_linear_row_major(li, ctx.shape)?;
    let axis_c = u64::try_from(coords[ctx.layout.window_axis])
        .map_err(|_| TetError::Validation("axis coord overflow".into()))?;
    let (w0, w1) = window_geometry::covering_window_bounds(axis_c, ctx.layout)?;
    for w in w0..w1 {
        let oi = window_geometry::window_reduced_index(&coords, w, ctx.layout)?;
        let fiber = window_geometry::window_fiber_index(axis_c, w, ctx.layout)?;
        cells[oi].push_f64(fiber, v, ctx.kind);
        real_counts[oi] = real_counts[oi].saturating_add(1);
    }
    Ok(())
}

fn parallel_window_value(
    ctx: &WindowParallelCtx<'_>,
    workers: Option<usize>,
) -> Result<(Vec<reduction::ValueAccum>, Vec<u64>, u64, bool), TetError> {
    let out_len = ctx.layout.out_len;
    let parts: Vec<WindowChunkValue> = with_fold_workers(workers, || {
        ctx.plan
            .chunks
            .par_iter()
            .map(|c| {
                let mut cells = vec![reduction::ValueAccum::default(); out_len];
                let mut real_counts = vec![0_u64; out_len];
                let mut saw_any = false;
                let bytes = ctx.visit.visit_chunk(ctx.mmap, ctx.plan, c, |li, v| {
                    saw_any = true;
                    ctx.visit
                        .write_preview(ctx.preview_addr, ctx.preview_len, li, v);
                    if !fold_keeps_element(ctx.plan, li, v) {
                        return Ok(());
                    }
                    scatter_value(ctx, &mut cells, &mut real_counts, li, v)
                })?;
                Ok(WindowChunkValue {
                    inner: PartialChunkValue {
                        bytes,
                        cells,
                        saw_any,
                    },
                    real_counts,
                })
            })
            .collect::<Result<Vec<_>, TetError>>()
    })?;

    let mut merged = vec![reduction::ValueAccum::default(); out_len];
    let mut real_counts = vec![0_u64; out_len];
    let mut saw_any = false;
    let mut total_bytes = 0u64;
    for p in &parts {
        saw_any |= p.inner.saw_any;
        merge_partial_value_cells(&mut merged, &p.inner.cells);
        for (d, s) in real_counts.iter_mut().zip(&p.real_counts) {
            *d = d.saturating_add(*s);
        }
        total_bytes = total_bytes
            .checked_add(p.inner.bytes)
            .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    }
    Ok((merged, real_counts, total_bytes, saw_any))
}

fn parallel_window_arg(
    ctx: &WindowParallelCtx<'_>,
    workers: Option<usize>,
) -> Result<(Vec<reduction::ArgIndexAccum>, Vec<u64>, u64, bool), TetError> {
    let out_len = ctx.layout.out_len;
    let parts: Vec<WindowChunkArg> = with_fold_workers(workers, || {
        ctx.plan
            .chunks
            .par_iter()
            .map(|c| {
                let mut cells = vec![reduction::ArgIndexAccum::default(); out_len];
                let mut real_counts = vec![0_u64; out_len];
                let mut saw_any = false;
                let bytes = ctx.visit.visit_chunk(ctx.mmap, ctx.plan, c, |li, v| {
                    saw_any = true;
                    ctx.visit
                        .write_preview(ctx.preview_addr, ctx.preview_len, li, v);
                    if !fold_keeps_element(ctx.plan, li, v) {
                        return Ok(());
                    }
                    scatter_arg(ctx, &mut cells, &mut real_counts, li, v)
                })?;
                Ok(WindowChunkArg {
                    inner: PartialChunkArg {
                        bytes,
                        cells,
                        saw_any,
                    },
                    real_counts,
                })
            })
            .collect::<Result<Vec<_>, TetError>>()
    })?;

    let mut merged = vec![reduction::ArgIndexAccum::default(); out_len];
    let mut real_counts = vec![0_u64; out_len];
    let mut saw_any = false;
    let mut total_bytes = 0u64;
    for p in &parts {
        saw_any |= p.inner.saw_any;
        merge_partial_arg_cells(&mut merged, &p.inner.cells, ctx.kind);
        for (d, s) in real_counts.iter_mut().zip(&p.real_counts) {
            *d = d.saturating_add(*s);
        }
        total_bytes = total_bytes
            .checked_add(p.inner.bytes)
            .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    }
    Ok((merged, real_counts, total_bytes, saw_any))
}

fn allocate_preview(dtype: ElementDtype, cap: usize) -> shared::FoldPreviewBuffer {
    match dtype {
        ElementDtype::F32 => shared::FoldPreviewBuffer::F32(vec![f32::NAN; cap]),
        ElementDtype::F64 => shared::FoldPreviewBuffer::F64(vec![f64::NAN; cap]),
        ElementDtype::F16 => shared::FoldPreviewBuffer::F16(vec![half::f16::NAN; cap]),
        ElementDtype::I32 => shared::FoldPreviewBuffer::I32(vec![0; cap]),
        ElementDtype::I64 => shared::FoldPreviewBuffer::I64(vec![0; cap]),
        ElementDtype::U8 => shared::FoldPreviewBuffer::U8(vec![0; cap]),
        ElementDtype::U16 => shared::FoldPreviewBuffer::U16(vec![0; cap]),
        ElementDtype::I16 => shared::FoldPreviewBuffer::I16(vec![0; cap]),
        ElementDtype::U32 => shared::FoldPreviewBuffer::U32(vec![0; cap]),
        ElementDtype::U64 => shared::FoldPreviewBuffer::U64(vec![0; cap]),
    }
}

fn preview_raw_parts(preview: &mut shared::FoldPreviewBuffer) -> (usize, usize) {
    match preview {
        shared::FoldPreviewBuffer::F32(v) => (v.as_mut_ptr() as usize, v.len()),
        shared::FoldPreviewBuffer::F64(v) => (v.as_mut_ptr() as usize, v.len()),
        shared::FoldPreviewBuffer::F16(v) => (v.as_mut_ptr() as usize, v.len()),
        shared::FoldPreviewBuffer::I32(v) => (v.as_mut_ptr() as usize, v.len()),
        shared::FoldPreviewBuffer::I64(v) => (v.as_mut_ptr() as usize, v.len()),
        shared::FoldPreviewBuffer::U8(v) => (v.as_mut_ptr() as usize, v.len()),
        shared::FoldPreviewBuffer::U16(v) => (v.as_mut_ptr() as usize, v.len()),
        shared::FoldPreviewBuffer::I16(v) => (v.as_mut_ptr() as usize, v.len()),
        shared::FoldPreviewBuffer::U32(v) => (v.as_mut_ptr() as usize, v.len()),
        shared::FoldPreviewBuffer::U64(v) => (v.as_mut_ptr() as usize, v.len()),
    }
}

fn validate_parallel_preview(
    visit: WindowVisit,
    saw_any: bool,
    preview: &shared::FoldPreviewBuffer,
    preview_cap: usize,
) -> Result<(), TetError> {
    match (visit, preview) {
        (WindowVisit::F32, shared::FoldPreviewBuffer::F32(buf)) => {
            shared::validate_fold_preview(saw_any, buf, preview_cap)
        }
        (WindowVisit::F64, shared::FoldPreviewBuffer::F64(buf)) => {
            shared::validate_fold_preview_f64(saw_any, buf, preview_cap)
        }
        (WindowVisit::F16, shared::FoldPreviewBuffer::F16(buf)) => {
            shared::validate_fold_preview_f16(saw_any, buf, preview_cap)
        }
        (WindowVisit::Int(_), _) => {
            if saw_any {
                Ok(())
            } else {
                Err(TetError::Validation(
                    "operation requires at least one decoded value from the read plan".into(),
                ))
            }
        }
        _ => Err(TetError::Validation(
            "internal: parallel window preview dtype mismatch".into(),
        )),
    }
}

/// Rayon windowed scan over planned chunks.
///
/// Returns merged accumulators + `real_counts`; the sequential window fold still
/// injects pad values, applies `min_count`, and builds the operation preview fields.
///
/// # Errors
///
/// Propagates decode, geometry, and preview validation failures.
pub(crate) fn scan_window_cells_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: reduction::ReductionKind,
    layout: &WindowAxisLayout,
    dtype: ElementDtype,
    workers: Option<usize>,
) -> Result<WindowParallelScan, TetError> {
    let shape = &plan.logical_selection_shape;
    let preview_cap = max_preview.min(plan.logical_element_count);
    let visit = WindowVisit::from_dtype(dtype);
    // Rank-1 + reduce window axis only: skip multi-dim coordinate decode.
    let rank1_full = shape.len() == 1 && layout.window_axis == 0 && layout.reduce_set.len() == 1;

    let mut preview = allocate_preview(dtype, preview_cap);
    let (preview_addr, preview_len) = preview_raw_parts(&mut preview);
    let ctx = WindowParallelCtx {
        mmap,
        plan,
        visit,
        kind,
        layout,
        shape,
        preview_addr,
        preview_len,
        rank1_full,
    };

    let (value_cells, arg_cells, real_counts, total_bytes, saw_any) = match kind {
        reduction::ReductionKind::ArgMin | reduction::ReductionKind::ArgMax => {
            let (cells, counts, bytes, saw) = parallel_window_arg(&ctx, workers)?;
            (None, Some(cells), counts, bytes, saw)
        }
        _ => {
            let (cells, counts, bytes, saw) = parallel_window_value(&ctx, workers)?;
            (Some(cells), None, counts, bytes, saw)
        }
    };

    validate_parallel_preview(visit, saw_any, &preview, preview_cap)?;
    Ok(WindowParallelScan {
        value_cells,
        arg_cells,
        real_counts,
        total_bytes,
        preview,
    })
}

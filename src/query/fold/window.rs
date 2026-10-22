//! Streaming windowed reductions along one axis (tier-A/B, all wire dtypes).
//!
//! I/O policy mirrors scalar/partial folds (memory-conscious), except windows
//! never take the out-of-core **`linear_scan`** byte-stream path (no scatter
//! implementation there). Callers pass `allows_linear_scan = false` so OOC
//! multi-chunk windows keep Rayon instead of falling back to single-threaded
//! sequential scatter:
//! - **Multi-chunk (in-core or OOC):** Rayon parallel chunk scan + merge (capped
//!   so huge `window_count` does not multiply accumulator replicas past ~64 MiB).
//! - **Single-chunk / oversize output / parallel refused:** one accumulator
//!   buffer, offset-ordered chunk visits when `FoldIoPolicy::prefer_ordered_chunk_io`.
//! - SIMD slab folds stay scalar-only for now (window scatter is multi-cell).

use super::partial::{partial_arg_fields, partial_fields};
use super::window_geometry::{self, WindowAxisLayout};
use crate::query::{
    decode::{chunk_decode, indexing},
    dispatch,
    fold::{fold_policy, parallel, reduction, shared},
    materialize::int::IntVisit,
    types::{ReadPlan, TetError, WindowPad, WindowSpec, fold_keeps_element},
};
use crate::utils::dtype::ElementDtype;

pub(crate) struct WindowFoldInput<'a> {
    pub mmap: &'a [u8],
    pub plan: &'a ReadPlan,
    pub max_preview: usize,
    pub kind: reduction::ReductionKind,
    pub window: &'a WindowSpec,
    pub axis_labels: &'a [String],
    pub dtype: ElementDtype,
    pub policy: &'a fold_policy::FoldIoPolicy,
}

/// Stream a windowed reduction without allocating the full logical tensor.
///
/// Supports all wire dtypes (promoted to `f64` accumulators). Pad / step / align /
/// `min_count` follow [`WindowSpec`].
pub(crate) fn fold_read_plan_window_operation(
    input: &WindowFoldInput<'_>,
) -> Result<shared::FoldPlanOutcome, TetError> {
    let layout = window_geometry::window_axis_layout(input.plan, input.window, input.axis_labels)?;
    let pad_value = resolve_pad_value(layout.pad, input.dtype, input.plan.dataset_fill)?;
    fold_window(&FoldWindowCtx {
        mmap: input.mmap,
        plan: input.plan,
        max_preview: input.max_preview,
        kind: input.kind,
        layout: &layout,
        dtype: input.dtype,
        pad_value,
        policy: input.policy,
    })
}

#[derive(Copy, Clone)]
struct FoldWindowCtx<'a> {
    mmap: &'a [u8],
    plan: &'a ReadPlan,
    max_preview: usize,
    kind: reduction::ReductionKind,
    layout: &'a WindowAxisLayout,
    dtype: ElementDtype,
    pad_value: Option<f64>,
    policy: &'a fold_policy::FoldIoPolicy,
}

fn resolve_pad_value(
    pad: WindowPad,
    dtype: ElementDtype,
    dataset_fill: Option<f64>,
) -> Result<Option<f64>, TetError> {
    match pad {
        WindowPad::None => Ok(None),
        WindowPad::Zero => Ok(Some(0.0)),
        WindowPad::Nan => {
            if dtype.is_integer() {
                return Err(TetError::Validation(
                    "`window.pad` \"nan\" is not valid for integer dtypes".into(),
                ));
            }
            Ok(Some(f64::NAN))
        }
        WindowPad::Fill => dataset_fill.map(Some).ok_or_else(|| {
            TetError::Validation(
                "`window.pad` \"fill\" requires a dataset fill (_FillValue, missing_value, or fill_value)"
                    .into(),
            )
        }),
    }
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
}

fn fold_window(ctx: &FoldWindowCtx<'_>) -> Result<shared::FoldPlanOutcome, TetError> {
    let n = ctx.plan.logical_element_count;
    if parallel::use_parallel_fold(ctx.plan, ctx.policy)
        && parallel::parallel_window_memory_ok(ctx.layout.out_len, ctx.policy.fold_workers)
    {
        return fold_window_parallel(ctx, n);
    }
    fold_window_sequential(ctx, n)
}

fn fold_window_parallel(
    ctx: &FoldWindowCtx<'_>,
    n: usize,
) -> Result<shared::FoldPlanOutcome, TetError> {
    let scan = parallel::scan_window_cells_parallel(
        ctx.mmap,
        ctx.plan,
        ctx.max_preview,
        ctx.kind,
        ctx.layout,
        ctx.dtype,
        ctx.policy.fold_workers,
    )?;
    let operation = match ctx.kind {
        reduction::ReductionKind::ArgMin | reduction::ReductionKind::ArgMax => {
            let mut cells = scan.arg_cells.ok_or_else(|| {
                TetError::Validation("internal: parallel window arg scan missing cells".into())
            })?;
            inject_pad_slots(ctx.layout, ctx.pad_value, cells.len(), |oi, fiber, pad| {
                cells[oi].push_f64(fiber, pad, ctx.kind);
                Ok(())
            })?;
            apply_min_count_arg(&mut cells, &scan.real_counts, ctx.layout);
            partial_arg_fields(ctx.kind, n, &ctx.layout.out_shape, &cells)
        }
        _ => {
            let mut cells = scan.value_cells.ok_or_else(|| {
                TetError::Validation("internal: parallel window value scan missing cells".into())
            })?;
            inject_pad_slots(ctx.layout, ctx.pad_value, cells.len(), |oi, _fiber, pad| {
                cells[oi].push_f64(pad);
                Ok(())
            })?;
            let reduced: Vec<f64> = cells
                .iter()
                .enumerate()
                .map(|(oi, c)| {
                    if scan.real_counts[oi] < ctx.layout.min_count {
                        f64::NAN
                    } else {
                        c.finish_f64(ctx.kind)
                    }
                })
                .collect();
            partial_fields(ctx.kind, n, &ctx.layout.out_shape, &reduced, &cells)
        }
    };
    Ok(shared::build_fold_plan_outcome_typed(
        scan.preview,
        ctx.max_preview,
        n,
        scan.total_bytes,
        operation,
    ))
}

fn fold_window_sequential(
    ctx: &FoldWindowCtx<'_>,
    n: usize,
) -> Result<shared::FoldPlanOutcome, TetError> {
    let mut preview = WindowPreview::new(ctx.dtype, ctx.max_preview.min(n));

    let (saw_any, total_bytes, operation) = match ctx.kind {
        reduction::ReductionKind::ArgMin | reduction::ReductionKind::ArgMax => {
            let mut cells = vec![reduction::ArgIndexAccum::default(); ctx.layout.out_len];
            let (saw_any, total_bytes, real_counts) =
                scan_window_cells(ctx, &mut preview, |oi, axis_c, w, v| {
                    let fiber = window_geometry::window_fiber_index(axis_c, w, ctx.layout)?;
                    cells[oi].push_f64(fiber, v, ctx.kind);
                    Ok(())
                })?;
            inject_pad_slots(ctx.layout, ctx.pad_value, cells.len(), |oi, fiber, pad| {
                cells[oi].push_f64(fiber, pad, ctx.kind);
                Ok(())
            })?;
            apply_min_count_arg(&mut cells, &real_counts, ctx.layout);
            (
                saw_any,
                total_bytes,
                partial_arg_fields(ctx.kind, n, &ctx.layout.out_shape, &cells),
            )
        }
        _ => {
            let mut cells = vec![reduction::ValueAccum::default(); ctx.layout.out_len];
            let (saw_any, total_bytes, real_counts) =
                scan_window_cells(ctx, &mut preview, |oi, _axis_c, _w, v| {
                    cells[oi].push_f64(v);
                    Ok(())
                })?;
            inject_pad_slots(ctx.layout, ctx.pad_value, cells.len(), |oi, _fiber, pad| {
                cells[oi].push_f64(pad);
                Ok(())
            })?;
            let reduced: Vec<f64> = cells
                .iter()
                .enumerate()
                .map(|(oi, c)| {
                    if real_counts[oi] < ctx.layout.min_count {
                        f64::NAN
                    } else {
                        c.finish_f64(ctx.kind)
                    }
                })
                .collect();
            (
                saw_any,
                total_bytes,
                partial_fields(ctx.kind, n, &ctx.layout.out_shape, &reduced, &cells),
            )
        }
    };
    preview.validate(saw_any)?;
    Ok(preview.into_outcome(ctx.max_preview, n, total_bytes, operation))
}

/// Shared chunk scan for windowed folds: decode → filter → covering windows → callback.
fn scan_window_cells(
    ctx: &FoldWindowCtx<'_>,
    preview: &mut WindowPreview,
    mut on_window: impl FnMut(usize, u64, usize, f64) -> Result<(), TetError>,
) -> Result<(bool, u64, Vec<u64>), TetError> {
    let shape = &ctx.plan.logical_selection_shape;
    let visit = WindowVisit::from_dtype(ctx.dtype);
    let mut saw_any = false;
    let mut total_bytes: u64 = 0;
    // Ordered I/O when policy asks (sequential_io or out-of-core linear_scan).
    let chunk_order =
        fold_policy::chunk_indices_for_fold(ctx.plan, ctx.policy.prefer_ordered_chunk_io());
    let mut real_counts = vec![0_u64; ctx.layout.out_len];
    // Rank-1 + reduce window axis only: `li` is the axis coord; output index is `w`.
    let rank1_full =
        shape.len() == 1 && ctx.layout.window_axis == 0 && ctx.layout.reduce_set.len() == 1;

    for i in chunk_order {
        let c = &ctx.plan.chunks[i];
        let chunk_bytes = visit.visit_chunk(ctx.mmap, ctx.plan, c, |li, v| {
            saw_any = true;
            preview.store(li, v);
            if !fold_keeps_element(ctx.plan, li, v) {
                return Ok(());
            }
            if rank1_full {
                let axis_c = u64::try_from(li)
                    .map_err(|_| TetError::Validation("axis coord overflow".into()))?;
                let (w0, w1) = window_geometry::covering_window_bounds(axis_c, ctx.layout)?;
                for (off, count) in real_counts[w0..w1].iter_mut().enumerate() {
                    let w = w0 + off;
                    on_window(w, axis_c, w, v)?;
                    *count = count.saturating_add(1);
                }
                return Ok(());
            }
            let coords = indexing::coords_from_linear_row_major(li, shape)?;
            let axis_c = u64::try_from(coords[ctx.layout.window_axis])
                .map_err(|_| TetError::Validation("axis coord overflow".into()))?;
            let (w0, w1) = window_geometry::covering_window_bounds(axis_c, ctx.layout)?;
            for w in w0..w1 {
                let oi = window_geometry::window_reduced_index(&coords, w, ctx.layout)?;
                on_window(oi, axis_c, w, v)?;
                real_counts[oi] = real_counts[oi].saturating_add(1);
            }
            Ok(())
        })?;
        dispatch::accumulate_chunk_read_bytes(&mut total_bytes, chunk_bytes)?;
    }
    Ok((saw_any, total_bytes, real_counts))
}

/// Inject pad cells for each output index (fiber used by arg folds; ignored by value folds).
fn inject_pad_slots(
    layout: &WindowAxisLayout,
    pad_value: Option<f64>,
    cell_count: usize,
    mut push: impl FnMut(usize, u64, f64) -> Result<(), TetError>,
) -> Result<(), TetError> {
    let Some(pad) = pad_value else {
        return Ok(());
    };
    for oi in 0..cell_count {
        let w = out_window_index(oi, layout)?;
        let axis_pad = window_geometry::window_pad_axis_slots(layout, w);
        let real_len = window_geometry::window_real_axis_len(layout, w);
        for k in 0..axis_pad {
            let fiber = real_len.saturating_add(k);
            for _ in 0..layout.other_reduce_product {
                push(oi, fiber, pad)?;
            }
        }
    }
    Ok(())
}

fn apply_min_count_arg(
    cells: &mut [reduction::ArgIndexAccum],
    real_counts: &[u64],
    layout: &WindowAxisLayout,
) {
    for (oi, cell) in cells.iter_mut().enumerate() {
        if real_counts[oi] < layout.min_count {
            *cell = reduction::ArgIndexAccum::default();
        }
    }
}

fn out_window_index(oi: usize, layout: &WindowAxisLayout) -> Result<usize, TetError> {
    if layout.out_shape.len() == 1 {
        return Ok(oi);
    }
    let coords = indexing::coords_from_linear_row_major(oi, &layout.out_shape)?;
    coords
        .get(layout.out_window_dim)
        .copied()
        .ok_or_else(|| TetError::Validation("window out index rank mismatch".into()))
}

enum WindowPreview {
    F32(Vec<f32>),
    F64(Vec<f64>),
    F16(Vec<half::f16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl WindowPreview {
    fn new(dtype: ElementDtype, preview_cap: usize) -> Self {
        match dtype {
            ElementDtype::F32 => Self::F32(vec![f32::NAN; preview_cap]),
            ElementDtype::F64 => Self::F64(vec![f64::NAN; preview_cap]),
            ElementDtype::F16 => Self::F16(vec![half::f16::NAN; preview_cap]),
            ElementDtype::I32 => Self::I32(vec![0; preview_cap]),
            ElementDtype::I64 => Self::I64(vec![0; preview_cap]),
            ElementDtype::U8 => Self::U8(vec![0; preview_cap]),
            ElementDtype::U16 => Self::U16(vec![0; preview_cap]),
            ElementDtype::I16 => Self::I16(vec![0; preview_cap]),
            ElementDtype::U32 => Self::U32(vec![0; preview_cap]),
            ElementDtype::U64 => Self::U64(vec![0; preview_cap]),
        }
    }

    /// Best-effort preview store; intentional truncating / sign casts from `f64`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn store(&mut self, li: usize, v: f64) {
        match self {
            Self::F32(buf) => {
                if li < buf.len() {
                    buf[li] = v as f32;
                }
            }
            Self::F64(buf) => {
                if li < buf.len() {
                    buf[li] = v;
                }
            }
            Self::F16(buf) => {
                if li < buf.len() {
                    buf[li] = half::f16::from_f64(v);
                }
            }
            Self::I32(buf) => {
                if li < buf.len() {
                    buf[li] = v as i32;
                }
            }
            Self::I64(buf) => {
                if li < buf.len() {
                    buf[li] = v as i64;
                }
            }
            Self::U8(buf) => {
                if li < buf.len() {
                    buf[li] = v as u8;
                }
            }
            Self::U16(buf) => {
                if li < buf.len() {
                    buf[li] = v as u16;
                }
            }
            Self::I16(buf) => {
                if li < buf.len() {
                    buf[li] = v as i16;
                }
            }
            Self::U32(buf) => {
                if li < buf.len() {
                    buf[li] = v as u32;
                }
            }
            Self::U64(buf) => {
                if li < buf.len() {
                    buf[li] = v as u64;
                }
            }
        }
    }

    fn validate(&self, saw_any: bool) -> Result<(), TetError> {
        match self {
            Self::F32(buf) => shared::validate_fold_preview(saw_any, buf, buf.len()),
            Self::F64(buf) => shared::validate_fold_preview_f64(saw_any, buf, buf.len()),
            Self::F16(buf) => shared::validate_fold_preview_f16(saw_any, buf, buf.len()),
            Self::I32(_)
            | Self::I64(_)
            | Self::U8(_)
            | Self::U16(_)
            | Self::I16(_)
            | Self::U32(_)
            | Self::U64(_) => {
                if saw_any {
                    Ok(())
                } else {
                    Err(TetError::Validation(
                        "operation requires at least one decoded value from the read plan".into(),
                    ))
                }
            }
        }
    }

    fn into_outcome(
        self,
        max_preview: usize,
        logical_len: usize,
        total_bytes: u64,
        operation: crate::query::types::OperationPreviewFields,
    ) -> shared::FoldPlanOutcome {
        let buf = match self {
            Self::F32(v) => shared::FoldPreviewBuffer::F32(v),
            Self::F64(v) => shared::FoldPreviewBuffer::F64(v),
            Self::F16(v) => shared::FoldPreviewBuffer::F16(v),
            Self::I32(v) => shared::FoldPreviewBuffer::I32(v),
            Self::I64(v) => shared::FoldPreviewBuffer::I64(v),
            Self::U8(v) => shared::FoldPreviewBuffer::U8(v),
            Self::U16(v) => shared::FoldPreviewBuffer::U16(v),
            Self::I16(v) => shared::FoldPreviewBuffer::I16(v),
            Self::U32(v) => shared::FoldPreviewBuffer::U32(v),
            Self::U64(v) => shared::FoldPreviewBuffer::U64(v),
        };
        shared::build_fold_plan_outcome_typed(buf, max_preview, logical_len, total_bytes, operation)
    }
}

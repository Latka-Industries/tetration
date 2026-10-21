//! Streaming windowed reductions along one axis (tier-A/B, all wire dtypes).

use super::partial::{partial_arg_fields, partial_fields};
use super::window_geometry::{self, WindowAxisLayout};
use crate::query::{
    decode::{chunk_decode, indexing},
    dispatch,
    fold::{fold_policy, reduction, shared},
    materialize::int::IntVisit,
    types::{ReadPlan, TetError, WindowPad, WindowSpec, fold_keeps_cell},
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
/// min_count follow [`WindowSpec`].
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
            Self::F32 => {
                chunk_decode::visit_planned_chunk(mmap, plan, c, |li, v| visit(li, f64::from(v)))
            }
            Self::F64 => chunk_decode::visit_planned_chunk_f64(mmap, plan, c, visit),
            Self::F16 => chunk_decode::visit_planned_chunk_f16(mmap, plan, c, |li, v| {
                visit(li, f64::from(v))
            }),
            Self::Int(v) => v.visit_chunk_as_f64(mmap, plan, c, visit),
        }
    }
}

fn fold_window(ctx: &FoldWindowCtx<'_>) -> Result<shared::FoldPlanOutcome, TetError> {
    let FoldWindowCtx {
        mmap,
        plan,
        max_preview,
        kind,
        layout,
        dtype,
        pad_value,
        policy,
    } = *ctx;
    let shape = &plan.logical_selection_shape;
    let n = plan.logical_element_count;
    let preview_cap = max_preview.min(n);
    let visit = WindowVisit::from_dtype(dtype);
    let mut preview = WindowPreview::new(dtype, preview_cap);
    let mut saw_any = false;
    let mut total_bytes: u64 = 0;
    let chunk_order = fold_policy::chunk_indices_for_fold(plan, policy.sequential_io);
    let mut real_counts = vec![0_u64; layout.out_len];

    let operation = match kind {
        reduction::ReductionKind::ArgMin | reduction::ReductionKind::ArgMax => {
            let mut cells = vec![reduction::ArgIndexAccum::default(); layout.out_len];
            for i in chunk_order {
                let c = &plan.chunks[i];
                let chunk_bytes = visit.visit_chunk(mmap, plan, c, |li, v| {
                    saw_any = true;
                    preview.store(li, v);
                    if !fold_keeps_cell(plan.cell_filter, v) {
                        return Ok(());
                    }
                    let coords = indexing::coords_from_linear_row_major(li, shape)?;
                    let axis_c = u64::try_from(coords[layout.window_axis])
                        .map_err(|_| TetError::Validation("axis coord overflow".into()))?;
                    for w in window_geometry::windows_covering(&coords, layout)? {
                        let oi = window_geometry::window_reduced_index(&coords, w, layout)?;
                        let fiber = window_geometry::window_fiber_index(axis_c, w, layout)?;
                        cells[oi].push_f64(fiber, v, kind);
                        real_counts[oi] = real_counts[oi].saturating_add(1);
                    }
                    Ok(())
                })?;
                dispatch::accumulate_chunk_read_bytes(&mut total_bytes, chunk_bytes)?;
            }
            inject_pad_arg(&mut cells, layout, pad_value, kind)?;
            apply_min_count_arg(&mut cells, &real_counts, layout);
            partial_arg_fields(kind, n, &layout.out_shape, &cells)
        }
        _ => {
            let mut cells = vec![reduction::ValueAccum::default(); layout.out_len];
            for i in chunk_order {
                let c = &plan.chunks[i];
                let chunk_bytes = visit.visit_chunk(mmap, plan, c, |li, v| {
                    saw_any = true;
                    preview.store(li, v);
                    if !fold_keeps_cell(plan.cell_filter, v) {
                        return Ok(());
                    }
                    let coords = indexing::coords_from_linear_row_major(li, shape)?;
                    for w in window_geometry::windows_covering(&coords, layout)? {
                        let oi = window_geometry::window_reduced_index(&coords, w, layout)?;
                        cells[oi].push_f64(v);
                        real_counts[oi] = real_counts[oi].saturating_add(1);
                    }
                    Ok(())
                })?;
                dispatch::accumulate_chunk_read_bytes(&mut total_bytes, chunk_bytes)?;
            }
            inject_pad_values(&mut cells, layout, pad_value)?;
            let reduced: Vec<f64> = cells
                .iter()
                .enumerate()
                .map(|(oi, c)| {
                    if real_counts[oi] < layout.min_count {
                        f64::NAN
                    } else {
                        c.finish_f64(kind)
                    }
                })
                .collect();
            partial_fields(kind, n, &layout.out_shape, &reduced, &cells)
        }
    };
    preview.validate(saw_any)?;
    Ok(preview.into_outcome(max_preview, n, total_bytes, operation))
}

fn inject_pad_values(
    cells: &mut [reduction::ValueAccum],
    layout: &WindowAxisLayout,
    pad_value: Option<f64>,
) -> Result<(), TetError> {
    let Some(pad) = pad_value else {
        return Ok(());
    };
    for (oi, cell) in cells.iter_mut().enumerate() {
        let w = out_window_index(oi, layout)?;
        let pad_n = window_geometry::window_pad_axis_slots(layout, w)
            .saturating_mul(layout.other_reduce_product);
        for _ in 0..pad_n {
            cell.push_f64(pad);
        }
    }
    Ok(())
}

fn inject_pad_arg(
    cells: &mut [reduction::ArgIndexAccum],
    layout: &WindowAxisLayout,
    pad_value: Option<f64>,
    kind: reduction::ReductionKind,
) -> Result<(), TetError> {
    let Some(pad) = pad_value else {
        return Ok(());
    };
    for (oi, cell) in cells.iter_mut().enumerate() {
        let w = out_window_index(oi, layout)?;
        let axis_pad = window_geometry::window_pad_axis_slots(layout, w);
        let real_len = window_geometry::window_real_axis_len(layout, w);
        for k in 0..axis_pad {
            let fiber = real_len.saturating_add(k);
            for _ in 0..layout.other_reduce_product {
                cell.push_f64(fiber, pad, kind);
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

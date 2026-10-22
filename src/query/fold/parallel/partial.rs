//! Parallel partial-axis fold over planned chunks.

use rayon::prelude::*;

use super::merge::{
    PartialChunkArg, PartialChunkValue, merge_partial_arg_cells, merge_partial_value_cells,
    reduced_cell_index,
};
use super::preview::{write_disjoint_preview_f32, write_disjoint_preview_f64};
use super::workers::with_fold_workers;
use crate::query::decode;
use crate::query::fold::{partial, partial_geometry, reduction, shared};
use crate::query::types::{OperationPreviewFields, ReadPlan, TetError, fold_keeps_element};

#[derive(Copy, Clone)]
enum PartialFoldVisit {
    F32,
    F64,
}

impl PartialFoldVisit {
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
            Self::F32 => decode::chunk_decode::visit_planned_chunk(mmap, plan, c, |li, v| {
                visit(li, f64::from(v))
            }),
            Self::F64 => decode::chunk_decode::visit_planned_chunk_f64(mmap, plan, c, visit),
        }
    }

    fn push_value(self, cell: &mut reduction::ValueAccum, v: f64) {
        match self {
            Self::F32 => cell.push(v as f32),
            Self::F64 => cell.push_f64(v),
        }
    }

    fn push_arg(
        self,
        cell: &mut reduction::ArgIndexAccum,
        fi: u64,
        v: f64,
        kind: reduction::ReductionKind,
    ) {
        match self {
            Self::F32 => cell.push(fi, v as f32, kind),
            Self::F64 => cell.push_f64(fi, v, kind),
        }
    }

    fn write_preview(self, preview_addr: usize, preview_len: usize, li: usize, v: f64) {
        match self {
            Self::F32 => write_disjoint_preview_f32(preview_addr, preview_len, li, v as f32),
            Self::F64 => write_disjoint_preview_f64(preview_addr, preview_len, li, v),
        }
    }

    fn validate_preview(
        self,
        saw_any: bool,
        f32_preview: &[f32],
        f64_preview: &[f64],
        preview_cap: usize,
    ) -> Result<(), TetError> {
        match self {
            Self::F32 => shared::validate_fold_preview(saw_any, f32_preview, preview_cap),
            Self::F64 => shared::validate_fold_preview_f64(saw_any, f64_preview, preview_cap),
        }
    }
}

struct PartialParallelWork {
    operation: OperationPreviewFields,
    total_bytes: u64,
    saw_any: bool,
}

struct PartialParallelCtx<'a> {
    mmap: &'a [u8],
    plan: &'a ReadPlan,
    visit: PartialFoldVisit,
    kind: reduction::ReductionKind,
    layout: &'a partial_geometry::PartialAxisLayout,
    shape: &'a [u64],
    preview_addr: usize,
    preview_len: usize,
    n: usize,
}

fn parallel_partial_arg(
    ctx: &PartialParallelCtx<'_>,
    workers: Option<usize>,
) -> Result<PartialParallelWork, TetError> {
    let out_len = ctx.layout.out_len;
    let parts: Vec<PartialChunkArg> = with_fold_workers(workers, || {
        ctx.plan
            .chunks
            .par_iter()
            .map(|c| {
                let mut cells = vec![reduction::ArgIndexAccum::default(); out_len];
                let mut saw_any = false;
                let bytes = ctx.visit.visit_chunk(ctx.mmap, ctx.plan, c, |li, v| {
                    saw_any = true;
                    ctx.visit
                        .write_preview(ctx.preview_addr, ctx.preview_len, li, v);
                    if !fold_keeps_element(ctx.plan, li, v) {
                        return Ok(());
                    }
                    let (oi, fi) = reduced_cell_index(li, ctx.shape, ctx.layout)?;
                    ctx.visit.push_arg(&mut cells[oi], fi, v, ctx.kind);
                    Ok(())
                })?;
                Ok(PartialChunkArg {
                    bytes,
                    cells,
                    saw_any,
                })
            })
            .collect::<Result<Vec<_>, TetError>>()
    })?;

    let mut merged = vec![reduction::ArgIndexAccum::default(); out_len];
    let mut saw_any = false;
    let mut total_bytes = 0u64;
    for p in &parts {
        saw_any |= p.saw_any;
        merge_partial_arg_cells(&mut merged, &p.cells, ctx.kind);
        total_bytes = total_bytes
            .checked_add(p.bytes)
            .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    }
    Ok(PartialParallelWork {
        operation: partial::partial_arg_fields(ctx.kind, ctx.n, &ctx.layout.out_shape, &merged),
        total_bytes,
        saw_any,
    })
}

fn parallel_partial_value(
    ctx: &PartialParallelCtx<'_>,
    workers: Option<usize>,
) -> Result<PartialParallelWork, TetError> {
    let out_len = ctx.layout.out_len;
    let parts: Vec<PartialChunkValue> = with_fold_workers(workers, || {
        ctx.plan
            .chunks
            .par_iter()
            .map(|c| {
                let mut cells = vec![reduction::ValueAccum::default(); out_len];
                let mut saw_any = false;
                let bytes = ctx.visit.visit_chunk(ctx.mmap, ctx.plan, c, |li, v| {
                    saw_any = true;
                    ctx.visit
                        .write_preview(ctx.preview_addr, ctx.preview_len, li, v);
                    if !fold_keeps_element(ctx.plan, li, v) {
                        return Ok(());
                    }
                    let (oi, _) = reduced_cell_index(li, ctx.shape, ctx.layout)?;
                    ctx.visit.push_value(&mut cells[oi], v);
                    Ok(())
                })?;
                Ok(PartialChunkValue {
                    bytes,
                    cells,
                    saw_any,
                })
            })
            .collect::<Result<Vec<_>, TetError>>()
    })?;

    let mut merged = vec![reduction::ValueAccum::default(); out_len];
    let mut saw_any = false;
    let mut total_bytes = 0u64;
    for p in &parts {
        saw_any |= p.saw_any;
        merge_partial_value_cells(&mut merged, &p.cells);
        total_bytes = total_bytes
            .checked_add(p.bytes)
            .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    }
    let reduced: Vec<f64> = merged.iter().map(|c| c.finish_f64(ctx.kind)).collect();
    Ok(PartialParallelWork {
        operation: partial::partial_fields(
            ctx.kind,
            ctx.n,
            &ctx.layout.out_shape,
            &reduced,
            &merged,
        ),
        total_bytes,
        saw_any,
    })
}

fn parallel_partial_axis_fold(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: reduction::ReductionKind,
    axis_labels: &[String],
    visit: PartialFoldVisit,
    workers: Option<usize>,
) -> Result<shared::FoldPlanOutcome, TetError> {
    let layout = partial_geometry::partial_axis_layout(plan, axis_labels)?;
    let shape = plan.logical_selection_shape.clone();
    let n = plan.logical_element_count;
    let preview_cap = max_preview.min(n);
    let mut f32_preview = vec![f32::NAN; preview_cap];
    let mut f64_preview = vec![f64::NAN; preview_cap];
    let (preview_addr, preview_len) = match visit {
        PartialFoldVisit::F32 => (f32_preview.as_mut_ptr() as usize, f32_preview.len()),
        PartialFoldVisit::F64 => (f64_preview.as_mut_ptr() as usize, f64_preview.len()),
    };

    let ctx = PartialParallelCtx {
        mmap,
        plan,
        visit,
        kind,
        layout: &layout,
        shape: &shape,
        preview_addr,
        preview_len,
        n,
    };
    let work = match kind {
        reduction::ReductionKind::ArgMin | reduction::ReductionKind::ArgMax => {
            parallel_partial_arg(&ctx, workers)?
        }
        _ => parallel_partial_value(&ctx, workers)?,
    };

    visit.validate_preview(work.saw_any, &f32_preview, &f64_preview, preview_cap)?;

    match visit {
        PartialFoldVisit::F32 => Ok(shared::build_fold_plan_outcome(
            f32_preview,
            max_preview,
            n,
            work.total_bytes,
            work.operation,
        )),
        PartialFoldVisit::F64 => Ok(shared::build_fold_plan_outcome_typed(
            shared::FoldPreviewBuffer::F64(f64_preview),
            max_preview,
            n,
            work.total_bytes,
            work.operation,
        )),
    }
}

/// Parallel partial-axis fold (`f32`).
pub(crate) fn fold_read_plan_partial_operation_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: reduction::ReductionKind,
    axis_labels: &[String],
    workers: Option<usize>,
) -> Result<shared::FoldPlanOutcome, TetError> {
    parallel_partial_axis_fold(
        mmap,
        plan,
        max_preview,
        kind,
        axis_labels,
        PartialFoldVisit::F32,
        workers,
    )
}

/// Parallel partial-axis fold (`f64`).
pub(crate) fn fold_read_plan_partial_operation_f64_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: reduction::ReductionKind,
    axis_labels: &[String],
    workers: Option<usize>,
) -> Result<shared::FoldPlanOutcome, TetError> {
    parallel_partial_axis_fold(
        mmap,
        plan,
        max_preview,
        kind,
        axis_labels,
        PartialFoldVisit::F64,
        workers,
    )
}

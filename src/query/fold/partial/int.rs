//! Sequential partial-axis fold for integer dtypes (promoted to `f64` accumulators).

use super::fields::{partial_arg_fields, partial_fields};
use crate::query::{
    decode::chunk_decode,
    dispatch,
    fold::{fold_policy, partial_geometry, reduction, shared},
    types::{OperationPreviewFields, ReadPlan, TetError, fold_keeps_element},
};
use crate::utils::dtype::ElementDtype;

#[derive(Copy, Clone)]
enum PromotedWidth {
    I32,
    I64,
}

enum PromotedPreview<'a> {
    I32(&'a mut [i32]),
    I64(&'a mut [i64]),
}

impl PromotedPreview<'_> {
    fn store(&mut self, li: usize, preview_cap: usize, v: f64) {
        if li >= preview_cap {
            return;
        }
        match self {
            Self::I32(buf) => buf[li] = v as i32,
            Self::I64(buf) => buf[li] = v as i64,
        }
    }
}

struct PromotedPartialRunCtx<'a> {
    mmap: &'a [u8],
    plan: &'a ReadPlan,
    kind: reduction::ReductionKind,
    layout: &'a partial_geometry::PartialAxisLayout,
    shape: &'a [u64],
    n: usize,
    preview_cap: usize,
    width: PromotedWidth,
    preview: PromotedPreview<'a>,
    saw_any: &'a mut bool,
    total_bytes: &'a mut u64,
    sequential_io: bool,
}

/// Stream a **partial-axis** reduction (`i32` / `i64` promoted to `f64` accumulators).
pub(crate) fn fold_read_plan_partial_operation_int(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: reduction::ReductionKind,
    axis_labels: &[String],
    dtype: ElementDtype,
    policy: &fold_policy::FoldIoPolicy,
) -> Result<shared::FoldPlanOutcome, TetError> {
    let width = match dtype {
        ElementDtype::I64 => PromotedWidth::I64,
        ElementDtype::U8
        | ElementDtype::I32
        | ElementDtype::U16
        | ElementDtype::I16
        | ElementDtype::U32
        | ElementDtype::U64 => PromotedWidth::I32,
        ElementDtype::F16 => {
            return Err(TetError::Validation(
                "partial-axis fold on f16 is not supported".into(),
            ));
        }
        _ => {
            return Err(TetError::Validation(
                "integer partial fold requires i32, i64, u8, u16, i16, u32, or u64 dtype".into(),
            ));
        }
    };
    fold_read_plan_partial_operation_promoted(
        mmap,
        plan,
        max_preview,
        kind,
        axis_labels,
        width,
        policy.sequential_io,
    )
}

fn fold_read_plan_partial_operation_promoted(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: reduction::ReductionKind,
    axis_labels: &[String],
    width: PromotedWidth,
    sequential_io: bool,
) -> Result<shared::FoldPlanOutcome, TetError> {
    let layout = partial_geometry::partial_axis_layout(plan, axis_labels)?;
    let shape = &plan.logical_selection_shape;
    let n = plan.logical_element_count;
    let preview_cap = max_preview.min(n);
    let mut saw_any = false;
    let mut total_bytes_read_from_disk: u64 = 0;

    match width {
        PromotedWidth::I32 => {
            let mut preview = vec![0i32; preview_cap];
            let operation = run_partial_promoted(PromotedPartialRunCtx {
                mmap,
                plan,
                kind,
                layout: &layout,
                shape,
                n,
                preview_cap,
                width,
                preview: PromotedPreview::I32(&mut preview),
                saw_any: &mut saw_any,
                total_bytes: &mut total_bytes_read_from_disk,
                sequential_io,
            })?;
            if !saw_any {
                return Err(TetError::Validation(
                    "operation requires at least one decoded value from the read plan".into(),
                ));
            }
            Ok(shared::build_fold_plan_outcome_typed(
                shared::FoldPreviewBuffer::I32(preview),
                max_preview,
                n,
                total_bytes_read_from_disk,
                operation,
            ))
        }
        PromotedWidth::I64 => {
            let mut preview = vec![0i64; preview_cap];
            let operation = run_partial_promoted(PromotedPartialRunCtx {
                mmap,
                plan,
                kind,
                layout: &layout,
                shape,
                n,
                preview_cap,
                width,
                preview: PromotedPreview::I64(&mut preview),
                saw_any: &mut saw_any,
                total_bytes: &mut total_bytes_read_from_disk,
                sequential_io,
            })?;
            if !saw_any {
                return Err(TetError::Validation(
                    "operation requires at least one decoded value from the read plan".into(),
                ));
            }
            Ok(shared::build_fold_plan_outcome_typed(
                shared::FoldPreviewBuffer::I64(preview),
                max_preview,
                n,
                total_bytes_read_from_disk,
                operation,
            ))
        }
    }
}

fn visit_promoted_chunk<F>(
    width: PromotedWidth,
    mmap: &[u8],
    plan: &ReadPlan,
    c: &crate::query::types::PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    match width {
        PromotedWidth::I32 => chunk_decode::visit_planned_chunk_i32_as_f64(mmap, plan, c, visit),
        PromotedWidth::I64 => chunk_decode::visit_planned_chunk_i64_as_f64(mmap, plan, c, visit),
    }
}

fn run_partial_promoted(
    ctx: PromotedPartialRunCtx<'_>,
) -> Result<OperationPreviewFields, TetError> {
    let PromotedPartialRunCtx {
        mmap,
        plan,
        kind,
        layout,
        shape,
        n,
        preview_cap,
        width,
        mut preview,
        saw_any,
        total_bytes,
        sequential_io,
    } = ctx;
    let chunk_order = fold_policy::chunk_indices_for_fold(plan, sequential_io);
    match kind {
        reduction::ReductionKind::ArgMin | reduction::ReductionKind::ArgMax => {
            let mut cells = vec![reduction::ArgIndexAccum::default(); layout.out_len];
            for i in chunk_order {
                let c = &plan.chunks[i];
                let chunk_bytes = visit_promoted_chunk(width, mmap, plan, c, |li, v| {
                    *saw_any = true;
                    preview.store(li, preview_cap, v);
                    if !fold_keeps_element(plan, li, v) {
                        return Ok(());
                    }
                    let (oi, fi) = partial_geometry::reduced_cell_index(li, shape, layout)?;
                    cells[oi].push_f64(fi, v, kind);
                    Ok(())
                })?;
                dispatch::accumulate_chunk_read_bytes(total_bytes, chunk_bytes)?;
            }
            Ok(partial_arg_fields(kind, n, &layout.out_shape, &cells))
        }
        _ => {
            let mut cells = vec![reduction::ValueAccum::default(); layout.out_len];
            for i in chunk_order {
                let c = &plan.chunks[i];
                let chunk_bytes = visit_promoted_chunk(width, mmap, plan, c, |li, v| {
                    *saw_any = true;
                    preview.store(li, preview_cap, v);
                    if !fold_keeps_element(plan, li, v) {
                        return Ok(());
                    }
                    let (oi, _) = partial_geometry::reduced_cell_index(li, shape, layout)?;
                    cells[oi].push_f64(v);
                    Ok(())
                })?;
                dispatch::accumulate_chunk_read_bytes(total_bytes, chunk_bytes)?;
            }
            let reduced: Vec<f64> = cells.iter().map(|c| c.finish_f64(kind)).collect();
            Ok(partial_fields(kind, n, &layout.out_shape, &reduced, &cells))
        }
    }
}

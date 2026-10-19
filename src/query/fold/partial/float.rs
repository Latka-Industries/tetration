//! Sequential partial-axis fold for `f32` and `f64`.

use super::fields::{partial_arg_fields, partial_fields};
use crate::query::{
    decode::{chunk_decode, indexing},
    dispatch,
    fold::{fold_policy, parallel, partial_geometry, reduction, shared},
    types::{OperationPreviewFields, ReadPlan, TetError, fold_keeps_cell},
};

struct PartialFoldCtx<'a> {
    mmap: &'a [u8],
    plan: &'a ReadPlan,
    kind: reduction::ReductionKind,
    layout: &'a partial_geometry::PartialAxisLayout,
    shape: &'a [u64],
    n: usize,
    preview_cap: usize,
    saw_any: &'a mut bool,
    total_bytes: &'a mut u64,
    sequential_io: bool,
}

/// Stream a **partial-axis** reduction without allocating the full logical tensor (f32).
pub(crate) fn fold_read_plan_partial_operation(
    mmap: &[u8],
    plan: &ReadPlan,
    max_f32: usize,
    kind: reduction::ReductionKind,
    axis_labels: &[String],
    policy: &fold_policy::FoldIoPolicy,
) -> Result<shared::FoldPlanOutcome, TetError> {
    if parallel::use_parallel_fold(plan, policy) {
        return parallel::fold_read_plan_partial_operation_parallel(
            mmap,
            plan,
            max_f32,
            kind,
            axis_labels,
            policy.fold_workers,
        );
    }
    fold_read_plan_partial_operation_impl(
        mmap,
        plan,
        max_f32,
        kind,
        axis_labels,
        false,
        policy.sequential_io,
    )
}

/// Stream a **partial-axis** reduction without allocating the full logical tensor (f64).
pub(crate) fn fold_read_plan_partial_operation_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: reduction::ReductionKind,
    axis_labels: &[String],
    policy: &fold_policy::FoldIoPolicy,
) -> Result<shared::FoldPlanOutcome, TetError> {
    if parallel::use_parallel_fold(plan, policy) {
        return parallel::fold_read_plan_partial_operation_f64_parallel(
            mmap,
            plan,
            max_preview,
            kind,
            axis_labels,
            policy.fold_workers,
        );
    }
    fold_read_plan_partial_operation_impl(
        mmap,
        plan,
        max_preview,
        kind,
        axis_labels,
        true,
        policy.sequential_io,
    )
}

fn fold_read_plan_partial_operation_impl(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: reduction::ReductionKind,
    axis_labels: &[String],
    f64_path: bool,
    sequential_io: bool,
) -> Result<shared::FoldPlanOutcome, TetError> {
    let layout = partial_geometry::partial_axis_layout(plan, axis_labels)?;
    let shape = &plan.logical_selection_shape;
    let n = plan.logical_element_count;
    let preview_cap = max_preview.min(n);
    let mut f32_preview = vec![f32::NAN; preview_cap];
    let mut f64_preview = vec![f64::NAN; preview_cap];
    let mut saw_any = false;
    let mut total_bytes_read_from_disk: u64 = 0;

    let mut ctx = PartialFoldCtx {
        mmap,
        plan,
        kind,
        layout: &layout,
        shape,
        n,
        preview_cap,
        saw_any: &mut saw_any,
        total_bytes: &mut total_bytes_read_from_disk,
        sequential_io,
    };
    let operation = if f64_path {
        run_partial_f64(&mut ctx, &mut f64_preview)?
    } else {
        run_partial_f32(&mut ctx, &mut f32_preview)?
    };

    if f64_path {
        shared::validate_fold_preview_f64(saw_any, &f64_preview, preview_cap)?;
        Ok(shared::build_fold_plan_outcome_typed(
            shared::FoldPreviewBuffer::F64(f64_preview),
            max_preview,
            n,
            total_bytes_read_from_disk,
            operation,
        ))
    } else {
        shared::validate_fold_preview(saw_any, &f32_preview, preview_cap)?;
        Ok(shared::build_fold_plan_outcome(
            f32_preview,
            max_preview,
            n,
            total_bytes_read_from_disk,
            operation,
        ))
    }
}

fn run_partial_f32(
    ctx: &mut PartialFoldCtx<'_>,
    preview: &mut [f32],
) -> Result<OperationPreviewFields, TetError> {
    let chunk_order = fold_policy::chunk_indices_for_fold(ctx.plan, ctx.sequential_io);
    match ctx.kind {
        reduction::ReductionKind::ArgMin | reduction::ReductionKind::ArgMax => {
            let mut cells = vec![reduction::ArgIndexAccum::default(); ctx.layout.out_len];
            for i in chunk_order {
                let c = &ctx.plan.chunks[i];
                let chunk_bytes =
                    chunk_decode::visit_planned_chunk(ctx.mmap, ctx.plan, c, |li, v| {
                        *ctx.saw_any = true;
                        if li < ctx.preview_cap {
                            preview[li] = v;
                        }
                        if !fold_keeps_cell(ctx.plan.cell_filter, f64::from(v)) {
                            return Ok(());
                        }
                        let coords = indexing::coords_from_linear_row_major(li, ctx.shape)?;
                        let oi = partial_geometry::reduced_index(
                            &coords,
                            &ctx.layout.axis_set,
                            &ctx.layout.out_shape,
                        )?;
                        let fi = partial_geometry::fiber_linear_index(
                            &coords,
                            &ctx.layout.axis_indices,
                            ctx.shape,
                        )? as u64;
                        cells[oi].push(fi, v, ctx.kind);
                        Ok(())
                    })?;
                dispatch::accumulate_chunk_read_bytes(ctx.total_bytes, chunk_bytes)?;
            }
            Ok(partial_arg_fields(
                ctx.kind,
                ctx.n,
                &ctx.layout.out_shape,
                &cells,
            ))
        }
        _ => {
            let mut cells = vec![reduction::ValueAccum::default(); ctx.layout.out_len];
            for i in chunk_order {
                let c = &ctx.plan.chunks[i];
                let chunk_bytes =
                    chunk_decode::visit_planned_chunk(ctx.mmap, ctx.plan, c, |li, v| {
                        *ctx.saw_any = true;
                        if li < ctx.preview_cap {
                            preview[li] = v;
                        }
                        if !fold_keeps_cell(ctx.plan.cell_filter, f64::from(v)) {
                            return Ok(());
                        }
                        let coords = indexing::coords_from_linear_row_major(li, ctx.shape)?;
                        let oi = partial_geometry::reduced_index(
                            &coords,
                            &ctx.layout.axis_set,
                            &ctx.layout.out_shape,
                        )?;
                        cells[oi].push(v);
                        Ok(())
                    })?;
                dispatch::accumulate_chunk_read_bytes(ctx.total_bytes, chunk_bytes)?;
            }
            let reduced: Vec<f64> = cells.iter().map(|c| c.finish_f64(ctx.kind)).collect();
            Ok(partial_fields(
                ctx.kind,
                ctx.n,
                &ctx.layout.out_shape,
                &reduced,
                &cells,
            ))
        }
    }
}

fn run_partial_f64(
    ctx: &mut PartialFoldCtx<'_>,
    preview: &mut [f64],
) -> Result<OperationPreviewFields, TetError> {
    let chunk_order = fold_policy::chunk_indices_for_fold(ctx.plan, ctx.sequential_io);
    match ctx.kind {
        reduction::ReductionKind::ArgMin | reduction::ReductionKind::ArgMax => {
            let mut cells = vec![reduction::ArgIndexAccum::default(); ctx.layout.out_len];
            for i in chunk_order {
                let c = &ctx.plan.chunks[i];
                let chunk_bytes =
                    chunk_decode::visit_planned_chunk_f64(ctx.mmap, ctx.plan, c, |li, v| {
                        *ctx.saw_any = true;
                        if li < ctx.preview_cap {
                            preview[li] = v;
                        }
                        if !fold_keeps_cell(ctx.plan.cell_filter, v) {
                            return Ok(());
                        }
                        let coords = indexing::coords_from_linear_row_major(li, ctx.shape)?;
                        let oi = partial_geometry::reduced_index(
                            &coords,
                            &ctx.layout.axis_set,
                            &ctx.layout.out_shape,
                        )?;
                        let fi = partial_geometry::fiber_linear_index(
                            &coords,
                            &ctx.layout.axis_indices,
                            ctx.shape,
                        )? as u64;
                        cells[oi].push_f64(fi, v, ctx.kind);
                        Ok(())
                    })?;
                dispatch::accumulate_chunk_read_bytes(ctx.total_bytes, chunk_bytes)?;
            }
            Ok(partial_arg_fields(
                ctx.kind,
                ctx.n,
                &ctx.layout.out_shape,
                &cells,
            ))
        }
        _ => {
            let mut cells = vec![reduction::ValueAccum::default(); ctx.layout.out_len];
            for i in chunk_order {
                let c = &ctx.plan.chunks[i];
                let chunk_bytes =
                    chunk_decode::visit_planned_chunk_f64(ctx.mmap, ctx.plan, c, |li, v| {
                        *ctx.saw_any = true;
                        if li < ctx.preview_cap {
                            preview[li] = v;
                        }
                        if !fold_keeps_cell(ctx.plan.cell_filter, v) {
                            return Ok(());
                        }
                        let coords = indexing::coords_from_linear_row_major(li, ctx.shape)?;
                        let oi = partial_geometry::reduced_index(
                            &coords,
                            &ctx.layout.axis_set,
                            &ctx.layout.out_shape,
                        )?;
                        cells[oi].push_f64(v);
                        Ok(())
                    })?;
                dispatch::accumulate_chunk_read_bytes(ctx.total_bytes, chunk_bytes)?;
            }
            let reduced: Vec<f64> = cells.iter().map(|c| c.finish_f64(ctx.kind)).collect();
            Ok(partial_fields(
                ctx.kind,
                ctx.n,
                &ctx.layout.out_shape,
                &reduced,
                &cells,
            ))
        }
    }
}

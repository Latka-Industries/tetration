//! Streaming windowed reductions along one axis (tier-A/B, `f32`).

use super::partial::{partial_arg_fields, partial_fields};
use super::window_geometry::{self, WindowAxisLayout};
use crate::query::{
    decode::{chunk_decode, indexing},
    dispatch,
    fold::{fold_policy, reduction, shared},
    types::{ReadPlan, TetError, WindowMode, WindowSpec, fold_keeps_cell},
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
/// **0.6.0:** `f32` only (same surface as the conformance fixture). Other dtypes refuse.
pub(crate) fn fold_read_plan_window_operation(
    input: &WindowFoldInput<'_>,
) -> Result<shared::FoldPlanOutcome, TetError> {
    if input.dtype != ElementDtype::F32 {
        return Err(TetError::Validation(
            "windowed fold requires f32 in 0.6.0 (other dtypes deferred)".into(),
        ));
    }
    let layout = window_geometry::window_axis_layout(input.plan, input.window, input.axis_labels)?;
    fold_window_f32(
        input.mmap,
        input.plan,
        input.max_preview,
        input.kind,
        &layout,
        input.policy,
    )
}

fn fold_window_f32(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: reduction::ReductionKind,
    layout: &WindowAxisLayout,
    policy: &fold_policy::FoldIoPolicy,
) -> Result<shared::FoldPlanOutcome, TetError> {
    let shape = &plan.logical_selection_shape;
    let n = plan.logical_element_count;
    let preview_cap = max_preview.min(n);
    let mut preview = vec![f32::NAN; preview_cap];
    let mut saw_any = false;
    let mut total_bytes: u64 = 0;
    let chunk_order = fold_policy::chunk_indices_for_fold(plan, policy.sequential_io);
    let operation = match kind {
        reduction::ReductionKind::ArgMin | reduction::ReductionKind::ArgMax => {
            let mut cells = vec![reduction::ArgIndexAccum::default(); layout.out_len];
            for i in chunk_order {
                let c = &plan.chunks[i];
                let chunk_bytes = chunk_decode::visit_planned_chunk(mmap, plan, c, |li, v| {
                    saw_any = true;
                    if li < preview_cap {
                        preview[li] = v;
                    }
                    if !fold_keeps_cell(plan.cell_filter, f64::from(v)) {
                        return Ok(());
                    }
                    let coords = indexing::coords_from_linear_row_major(li, shape)?;
                    let axis_c = coords[layout.window_axis] as u64;
                    for w in window_geometry::windows_covering(&coords, layout)? {
                        let oi = window_geometry::window_reduced_index(&coords, w, layout)?;
                        let fiber = match layout.mode {
                            WindowMode::Tumbling => axis_c % layout.size,
                            WindowMode::Sliding => {
                                axis_c.saturating_sub(u64::try_from(w).unwrap_or(0))
                            }
                        };
                        cells[oi].push(fiber, v, kind);
                    }
                    Ok(())
                })?;
                dispatch::accumulate_chunk_read_bytes(&mut total_bytes, chunk_bytes)?;
            }
            partial_arg_fields(kind, n, &layout.out_shape, &cells)
        }
        _ => {
            let mut cells = vec![reduction::ValueAccum::default(); layout.out_len];
            for i in chunk_order {
                let c = &plan.chunks[i];
                let chunk_bytes = chunk_decode::visit_planned_chunk(mmap, plan, c, |li, v| {
                    saw_any = true;
                    if li < preview_cap {
                        preview[li] = v;
                    }
                    if !fold_keeps_cell(plan.cell_filter, f64::from(v)) {
                        return Ok(());
                    }
                    let coords = indexing::coords_from_linear_row_major(li, shape)?;
                    for w in window_geometry::windows_covering(&coords, layout)? {
                        let oi = window_geometry::window_reduced_index(&coords, w, layout)?;
                        cells[oi].push(v);
                    }
                    Ok(())
                })?;
                dispatch::accumulate_chunk_read_bytes(&mut total_bytes, chunk_bytes)?;
            }
            let reduced: Vec<f64> = cells.iter().map(|c| c.finish_f64(kind)).collect();
            partial_fields(kind, n, &layout.out_shape, &reduced, &cells)
        }
    };
    shared::validate_fold_preview(saw_any, &preview, preview_cap)?;
    Ok(shared::build_fold_plan_outcome(
        preview,
        max_preview,
        n,
        total_bytes,
        operation,
    ))
}

//! `f32` logical materialize, spill, and scalar fold.

use std::path::Path;

use crate::query::decode::chunk_decode::{fold_planned_chunk_f32, scatter_chunk_into_plan};
use crate::query::{
    fold,
    types::{ReadPlan, TetError},
};

use super::parallel::materialize_read_plan_f32_le_parallel;
use super::shared::{
    check_materialized_nan_slice, materialize_into_vec_dispatch, materialize_read_plan_pod_le_core,
    scatter_fill_chunks, spill_byte_len_from_elem_count, spill_read_plan_pod_le_impl,
};

type ScatterFillFn = fn(&[u8], &ReadPlan, &mut [f32]) -> Result<u64, TetError>;

fn check_materialized_complete(out: &[f32]) -> Result<(), TetError> {
    check_materialized_nan_slice(out, f32::is_nan)
}

pub(crate) fn materialize_read_plan_f32_le_core(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    scatter_fill: ScatterFillFn,
) -> Result<(Vec<f32>, bool, u64), TetError> {
    materialize_read_plan_pod_le_core(
        mmap,
        plan,
        max_elements,
        scatter_fill,
        f32::NAN,
        check_materialized_complete,
    )
}

pub(crate) fn materialize_read_plan_f32_le_into_core(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    dst: &mut [f32],
    scatter_fill: ScatterFillFn,
) -> Result<MaterializeReadPlanF32IntoOutcome, TetError> {
    let n = plan.logical_element_count;
    if matches!(max_elements, Some(0)) {
        return Ok(MaterializeReadPlanF32IntoOutcome {
            logical_element_count: n,
            elements_written: 0,
            truncated: n > 0,
            total_bytes_read_from_disk: 0,
        });
    }
    let want_write = max_elements.map_or(n, |m| m.min(n));
    if dst.len() < want_write {
        return Err(TetError::Validation(format!(
            "destination buffer length {} < required {} (logical element count {})",
            dst.len(),
            want_write,
            n
        )));
    }
    let total_bytes_read_from_disk = if want_write < n {
        let mut tmp = vec![f32::NAN; want_write];
        let bytes = scatter_fill(mmap, plan, &mut tmp)?;
        if plan.cell_filter.is_none() {
            check_materialized_complete(&tmp)?;
        }
        dst[..want_write].copy_from_slice(&tmp);
        bytes
    } else {
        let bytes = scatter_fill(mmap, plan, &mut dst[..n])?;
        if plan.cell_filter.is_none() {
            check_materialized_complete(&dst[..n])?;
        }
        bytes
    };
    Ok(MaterializeReadPlanF32IntoOutcome {
        logical_element_count: n,
        elements_written: want_write,
        truncated: max_elements.is_some_and(|m| m < n),
        total_bytes_read_from_disk,
    })
}

/// Decode planned raw `f32` chunk payloads (little-endian) into **logical row-major** order for the
/// strided selection encoded on [`ReadPlan`].
///
/// `max_elements`: `None` decodes every float in the logical tensor. `Some(0)` returns an empty
/// vector and reads nothing from disk. `Some(n)` for `n > 0` returns the first `n` values in
/// logical row-major order and sets `truncated` when the logical tensor is longer.
///
/// # Errors
///
/// Returns [`TetError::Validation`] when chunk payloads disagree with tile geometry, the
/// strided selection is not fully covered by planned chunks, mmap bounds fail, or zstd decode
/// fails.
pub fn materialize_read_plan_f32_le(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<f32>, bool, u64), TetError> {
    materialize_read_plan_f32_le_core(mmap, plan, max_elements, materialize_scatter_fill)
}

/// Outcome of [`materialize_read_plan_f32_le_into`].
#[derive(Debug, Clone)]
pub struct MaterializeReadPlanF32IntoOutcome {
    /// Logical tensor element count (selection grid product).
    pub logical_element_count: usize,
    /// Values written to the start of the caller buffer (`min(max_elements.unwrap_or(logical), logical)`).
    pub elements_written: usize,
    pub truncated: bool,
    pub total_bytes_read_from_disk: u64,
}

/// Like [`materialize_read_plan_f32_le`], but writes decoded values into `dst` without allocating a `Vec`.
///
/// When `max_elements` is `None`, `dst.len()` must be at least [`ReadPlan::logical_element_count`].
/// When `max_elements` is `Some(m)` with `m > 0`, `dst.len()` must be at least `m.min(logical)`.
/// `Some(0)` writes nothing and does not touch `dst`.
///
/// # Errors
///
/// Same failure modes as [`materialize_read_plan_f32_le`], plus a short destination buffer.
pub fn materialize_read_plan_f32_le_into(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    dst: &mut [f32],
) -> Result<MaterializeReadPlanF32IntoOutcome, TetError> {
    materialize_read_plan_f32_le_into_core(mmap, plan, max_elements, dst, materialize_scatter_fill)
}

pub(crate) fn fold_read_plan_scalar_operation(
    mmap: &[u8],
    plan: &ReadPlan,
    max_f32: usize,
    kind: fold::reduction::ReductionKind,
    policy: &crate::query::fold::fold_policy::FoldIoPolicy,
) -> Result<fold::FoldPlanOutcome, TetError> {
    if crate::query::fold::parallel::use_parallel_fold(plan, policy) {
        return crate::query::fold::parallel::fold_read_plan_scalar_operation_parallel(
            mmap,
            plan,
            max_f32,
            kind,
            policy.fold_workers,
        );
    }
    let n = plan.logical_element_count;
    let preview_cap = max_f32.min(n);
    let mut preview = vec![f32::NAN; preview_cap];
    let mut total_bytes_read_from_disk: u64 = 0;
    let chunk_order =
        crate::query::fold::fold_policy::chunk_indices_for_fold(plan, policy.sequential_io);

    let operation = match kind {
        fold::reduction::ReductionKind::ArgMin | fold::reduction::ReductionKind::ArgMax => {
            let mut acc = fold::reduction::ArgIndexAccum::default();
            for i in chunk_order {
                let c = &plan.chunks[i];
                let chunk_bytes = fold_planned_chunk_f32(
                    mmap,
                    plan,
                    c,
                    kind,
                    &mut fold::reduction::ValueAccum::default(),
                    &mut acc,
                    &mut preview,
                )?;
                total_bytes_read_from_disk = total_bytes_read_from_disk
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            }
            fold::shared::validate_fold_preview(
                !acc.is_empty() || plan.cell_filter.is_some() || plan.label_eq.is_some(),
                &preview,
                preview_cap,
            )?;
            acc.finish_scalar(kind, n).into()
        }
        _ => {
            let mut acc = fold::reduction::ValueAccum::default();
            let mut arg = fold::reduction::ArgIndexAccum::default();
            for i in chunk_order {
                let c = &plan.chunks[i];
                let chunk_bytes =
                    fold_planned_chunk_f32(mmap, plan, c, kind, &mut acc, &mut arg, &mut preview)?;
                total_bytes_read_from_disk = total_bytes_read_from_disk
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            }
            fold::shared::validate_fold_preview(
                !acc.is_empty() || plan.cell_filter.is_some() || plan.label_eq.is_some(),
                &preview,
                preview_cap,
            )?;
            acc.finish_scalar(kind).into()
        }
    };

    Ok(fold::shared::build_fold_plan_outcome(
        preview,
        max_f32,
        n,
        total_bytes_read_from_disk,
        operation,
    ))
}

pub(crate) fn materialize_scatter_fill(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [f32],
) -> Result<u64, TetError> {
    scatter_fill_chunks(mmap, plan, out, scatter_chunk_into_plan)
}

/// Spill the full logical selection as row-major `f32` LE to `path` using a file-backed mmap
/// (disk-resident; does not allocate a dense `Vec` in RAM).
///
/// # Errors
///
/// Same validation failures as [`materialize_read_plan_f32_le`], plus I/O or mmap errors on `path`.
pub fn spill_read_plan_f32_le(mmap: &[u8], plan: &ReadPlan, path: &Path) -> Result<u64, TetError> {
    let byte_len = spill_byte_len_from_elem_count(
        plan.logical_element_count,
        crate::utils::f32_le::bytes_from_elem_count,
    )?;
    spill_read_plan_pod_le_impl(
        mmap,
        plan,
        path,
        byte_len,
        materialize_scatter_fill,
        Some(check_materialized_complete),
    )
}

pub(crate) fn materialize_into_vec(
    mmap: &[u8],
    plan: &ReadPlan,
) -> Result<(Vec<f32>, u64), TetError> {
    materialize_into_vec_dispatch(
        mmap,
        plan,
        materialize_read_plan_f32_le,
        materialize_read_plan_f32_le_parallel,
    )
}

pub(crate) fn preview_from_materialized_f32(
    backing: &super::types::LogicalF32Backing,
    logical_len: usize,
    max: usize,
) -> Result<(Vec<f32>, bool), TetError> {
    use super::shared::preview_from_backing_in_memory;

    match backing {
        super::types::LogicalF32Backing::InMemory(v) => {
            Ok(preview_from_backing_in_memory(v, logical_len, max))
        }
        super::types::LogicalF32Backing::TempSpill(temp) => {
            super::shared::preview_from_spill_file_pod(
                temp.path(),
                max.min(logical_len),
                logical_len,
            )
        }
    }
}

pub(crate) fn preview_from_spill_file_f32(
    path: &Path,
    cap: usize,
    logical_len: usize,
) -> Result<(Vec<f32>, bool), TetError> {
    super::shared::preview_from_spill_file_pod(path, cap, logical_len)
}

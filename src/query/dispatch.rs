//! Shared dtype-dispatch helpers for query execution.

use std::path::Path;

use crate::query::fold::{
    FoldPlanOutcome,
    fold_policy::FoldIoPolicy,
    partial::{
        fold_read_plan_partial_operation, fold_read_plan_partial_operation_f64,
        fold_read_plan_partial_operation_int,
    },
    reduction::ReductionKind,
};
use crate::query::materialize;
use crate::query::types::{ReadPlan, TetError};
use crate::utils::dtype::ElementDtype;

pub(crate) fn accumulate_chunk_read_bytes(
    total: &mut u64,
    chunk_bytes: u64,
) -> Result<(), TetError> {
    *total = total
        .checked_add(chunk_bytes)
        .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    Ok(())
}

pub(crate) fn sum_chunk_read_bytes(bytes: impl IntoIterator<Item = u64>) -> Result<u64, TetError> {
    bytes.into_iter().try_fold(0u64, |acc, b| {
        acc.checked_add(b)
            .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))
    })
}

macro_rules! materialize_read_plan_for_dtype {
    (
        $mmap:expr,
        $plan:expr,
        $max:expr,
        $parallel:expr,
        parallel $par:path,
        sequential $seq:path,
        preview $preview:ident
    ) => {{
        let (p, t, bytes) = if $parallel {
            $par($mmap, $plan, $max)?
        } else {
            $seq($mmap, $plan, $max)?
        };
        Ok((materialize::DecodePreviewBundle::$preview(p, t), bytes))
    }};
}

pub(crate) fn materialize_for_execution(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    dtype: ElementDtype,
) -> Result<(materialize::DecodePreviewBundle, u64), TetError> {
    let use_parallel = plan.chunks.len() > 1;
    match dtype {
        ElementDtype::F32 => materialize_read_plan_for_dtype!(
            mmap,
            plan,
            max_elements,
            use_parallel,
            parallel materialize::parallel::materialize_read_plan_f32_le_parallel,
            sequential materialize::materialize_read_plan_f32_le,
            preview f32_preview
        ),
        ElementDtype::F64 => materialize_read_plan_for_dtype!(
            mmap,
            plan,
            max_elements,
            use_parallel,
            parallel materialize::parallel::materialize_read_plan_f64_le_parallel,
            sequential materialize::materialize_read_plan_f64_le,
            preview f64_preview
        ),
        ElementDtype::I32 => materialize_read_plan_for_dtype!(
            mmap,
            plan,
            max_elements,
            use_parallel,
            parallel materialize::parallel::materialize_read_plan_i32_le_parallel,
            sequential materialize::int::materialize_read_plan_i32_le,
            preview i32_preview
        ),
        ElementDtype::I64 => materialize_read_plan_for_dtype!(
            mmap,
            plan,
            max_elements,
            use_parallel,
            parallel materialize::parallel::materialize_read_plan_i64_le_parallel,
            sequential materialize::int::materialize_read_plan_i64_le,
            preview i64_preview
        ),
        ElementDtype::U8 => materialize_read_plan_for_dtype!(
            mmap,
            plan,
            max_elements,
            use_parallel,
            parallel materialize::parallel::materialize_read_plan_u8_le_parallel,
            sequential materialize::int::materialize_read_plan_u8_le,
            preview u8_preview
        ),
        ElementDtype::U16 => materialize_read_plan_for_dtype!(
            mmap,
            plan,
            max_elements,
            use_parallel,
            parallel materialize::parallel::materialize_read_plan_u16_le_parallel,
            sequential materialize::int::materialize_read_plan_u16_le,
            preview u16_preview
        ),
        ElementDtype::I16 => materialize_read_plan_for_dtype!(
            mmap,
            plan,
            max_elements,
            use_parallel,
            parallel materialize::parallel::materialize_read_plan_i16_le_parallel,
            sequential materialize::int::materialize_read_plan_i16_le,
            preview i16_preview
        ),
        ElementDtype::U32 => materialize_read_plan_for_dtype!(
            mmap,
            plan,
            max_elements,
            use_parallel,
            parallel materialize::parallel::materialize_read_plan_u32_le_parallel,
            sequential materialize::int::materialize_read_plan_u32_le,
            preview u32_preview
        ),
        ElementDtype::U64 => materialize_read_plan_for_dtype!(
            mmap,
            plan,
            max_elements,
            use_parallel,
            parallel materialize::parallel::materialize_read_plan_u64_le_parallel,
            sequential materialize::int::materialize_read_plan_u64_le,
            preview u64_preview
        ),
        ElementDtype::F16 => materialize_read_plan_for_dtype!(
            mmap,
            plan,
            max_elements,
            use_parallel,
            parallel materialize::parallel::materialize_read_plan_f16_le_parallel,
            sequential materialize::materialize_read_plan_f16_le,
            preview f16_preview
        ),
    }
}

pub(crate) fn spill_full_selection(
    mmap: &[u8],
    plan: &ReadPlan,
    path: &Path,
    dtype: ElementDtype,
) -> Result<u64, TetError> {
    match dtype {
        ElementDtype::F32 => materialize::spill_read_plan_f32_le(mmap, plan, path),
        ElementDtype::F64 => materialize::spill_read_plan_f64_le(mmap, plan, path),
        ElementDtype::F16 => materialize::spill_read_plan_f16_le(mmap, plan, path),
        ElementDtype::I32
        | ElementDtype::I64
        | ElementDtype::U8
        | ElementDtype::U16
        | ElementDtype::I16
        | ElementDtype::U32
        | ElementDtype::U64 => materialize::int::spill_read_plan_int_le(mmap, plan, path, dtype),
    }
}

pub(crate) fn spill_export_preview(
    path: &Path,
    logical_len: usize,
    max_preview: usize,
    dtype: ElementDtype,
) -> Result<materialize::DecodePreviewBundle, TetError> {
    materialize::preview_from_spill_export_file(path, logical_len, max_preview, dtype)
}

pub(crate) fn scalar_fold(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    dtype: ElementDtype,
    policy: &FoldIoPolicy,
    tet_path: Option<&Path>,
) -> Result<FoldPlanOutcome, TetError> {
    if plan.cell_filter.is_none()
        && policy.linear_scan
        && crate::query::fold::linear_scan::supports_scalar_kind(kind)
    {
        return crate::query::fold::linear_scan::fold_read_plan_scalar_linear(
            mmap,
            plan,
            max_preview,
            kind,
            dtype,
            tet_path,
        );
    }
    match dtype {
        ElementDtype::F32 => crate::query::materialize::fold_read_plan_scalar_operation(
            mmap,
            plan,
            max_preview,
            kind,
            policy,
        ),
        ElementDtype::F64 => crate::query::materialize::fold_read_plan_scalar_operation_f64(
            mmap,
            plan,
            max_preview,
            kind,
            policy,
        ),
        ElementDtype::F16 => crate::query::materialize::fold_read_plan_scalar_operation_f16(
            mmap,
            plan,
            max_preview,
            kind,
            policy,
        ),
        ElementDtype::I32
        | ElementDtype::I64
        | ElementDtype::U8
        | ElementDtype::U16
        | ElementDtype::I16
        | ElementDtype::U32
        | ElementDtype::U64 => materialize::int::fold_read_plan_scalar_operation_int(
            mmap,
            plan,
            max_preview,
            kind,
            dtype,
            policy,
        ),
    }
}

pub(crate) fn partial_fold(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    axis_labels: &[String],
    dtype: ElementDtype,
    policy: &crate::query::fold::fold_policy::FoldIoPolicy,
) -> Result<FoldPlanOutcome, TetError> {
    match dtype {
        ElementDtype::F32 => {
            fold_read_plan_partial_operation(mmap, plan, max_preview, kind, axis_labels, policy)
        }
        ElementDtype::F64 => {
            fold_read_plan_partial_operation_f64(mmap, plan, max_preview, kind, axis_labels, policy)
        }
        ElementDtype::F16 => Err(TetError::Validation(
            "partial-axis fold on f16 is not supported; use f32/f64 or integer dtypes".into(),
        )),
        ElementDtype::I32
        | ElementDtype::I64
        | ElementDtype::U8
        | ElementDtype::U16
        | ElementDtype::I16
        | ElementDtype::U32
        | ElementDtype::U64 => fold_read_plan_partial_operation_int(
            mmap,
            plan,
            max_preview,
            kind,
            axis_labels,
            dtype,
            policy,
        ),
    }
}

//! Dense logical tensor export for embedders (Python, etc.) — no JSON preview cap.

use std::path::Path;

use crate::catalog::read_tet_summary_v1;
use crate::query::engine::{
    ExecutionBudget, PlannedRead, SpillPathAllowlist, materialize_read_plan_f32_le,
    materialize_read_plan_f64_le, materialize_read_plan_i16_le, materialize_read_plan_i32_le,
    materialize_read_plan_i64_le, materialize_read_plan_u8_le, materialize_read_plan_u16_le,
    plan_read_for_document,
};
use crate::query::fold::fold_policy::FoldIoPolicy;
use crate::query::materialize::int::{materialize_read_plan_u32_le, materialize_read_plan_u64_le};
use crate::query::materialize::materialize_read_plan_f16_le;
use crate::query::transform::{
    TransformDenseBuffer, TransformRunInput, materialize_transform_dense_ram,
};
use crate::query::types::{Operation, QueryDocument, TetError, WriteTarget};
use crate::utils::dtype::ElementDtype;

/// Row-major dense buffer for one logical selection (native element type).
#[derive(Debug, Clone)]
pub enum DenseBuffer {
    F32(Vec<f32>),
    F64(Vec<f64>),
    F16(Vec<half::f16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I16(Vec<i16>),
}

/// Full logical tensor decoded from a `.tet` selection.
#[derive(Debug, Clone)]
pub struct DenseMaterializeOutcome {
    pub dtype: u32,
    pub shape: Vec<u64>,
    pub buffer: DenseBuffer,
}

/// Estimate dense buffer size (bytes) for a selection-only document — planning only, no decode.
///
/// Uses the same logical element count as [`materialize_query_selection`]. Does **not** apply the
/// memory budget; callers that need a fail-fast check should call that materialize path (or compare
/// this estimate to [`ExecutionBudget::resolve`]).
///
/// # Errors
///
/// Validation when `doc` includes an `operation`, planning fails, or the byte product overflows.
pub fn estimate_query_selection_bytes(doc: &QueryDocument, mmap: &[u8]) -> Result<u64, TetError> {
    let planned = plan_selection_only(doc, mmap)?;
    let dtype = ElementDtype::from_wire(planned.dtype)?;
    ExecutionBudget::default()
        .logical_element_bytes(dtype, planned.read_plan.logical_f32_element_count)
}

/// Materialize a dataset selection (no `operation` key) into a dense buffer.
///
/// Resolves [`ExecutionBudget`] from the file TIDX header and query `execution` hints (same
/// precedence as transform `write: ram` / CLI). Rejects before decode when the logical selection
/// exceeds the budget.
///
/// # Errors
///
/// Validation when `doc` includes an `operation`, the selection exceeds the resolved memory budget,
/// or decode fails.
pub fn materialize_query_selection(
    doc: &QueryDocument,
    mmap: &[u8],
) -> Result<DenseMaterializeOutcome, TetError> {
    let planned = plan_selection_only(doc, mmap)?;
    let dtype = ElementDtype::from_wire(planned.dtype)?;
    let shape = planned.read_plan.logical_selection_shape.clone();
    let summary = read_tet_summary_v1(mmap)?;
    let budget = ExecutionBudget::resolve(&summary.file_execution, doc.execution.as_ref());
    if budget.full_tensor_exceeds_budget(&planned.read_plan, dtype)? {
        return Err(TetError::Validation(format!(
            "logical selection ({} elements, {} bytes) exceeds memory_budget_bytes ({}); \
             slice the selection, use spill, or raise execution.memory_budget_bytes",
            planned.read_plan.logical_f32_element_count,
            budget.logical_element_bytes(dtype, planned.read_plan.logical_f32_element_count)?,
            budget.memory_budget_bytes
        )));
    }
    let buffer = materialize_planned(mmap, &planned.read_plan, dtype)?;
    Ok(DenseMaterializeOutcome {
        dtype: planned.dtype,
        shape,
        buffer,
    })
}

fn plan_selection_only(doc: &QueryDocument, mmap: &[u8]) -> Result<PlannedRead, TetError> {
    if doc.operation.is_some() {
        return Err(TetError::Validation(
            "materialize_query_selection requires a selection-only document (no operation key)"
                .into(),
        ));
    }
    plan_read_for_document(doc, mmap)
}

/// Materialize a transform with `write: ram` into a dense buffer (full logical selection).
///
/// # Errors
///
/// Validation when the document is not a transform, `write` is not RAM, dtype is not float,
/// or the selection exceeds the resolved memory budget.
pub fn materialize_query_transform_ram(
    doc: &QueryDocument,
    mmap: &[u8],
    tet_path: &Path,
    spill_allowlist: Option<&SpillPathAllowlist>,
) -> Result<DenseMaterializeOutcome, TetError> {
    let Some(op @ Operation::Transform { .. }) = doc.operation.as_ref() else {
        return Err(TetError::Validation(
            "transform materialize requires transform operation".into(),
        ));
    };
    let write = doc.write.as_ref().ok_or_else(|| {
        TetError::Validation("transform materialize requires `write`: `ram`".into())
    })?;
    if write.target != WriteTarget::Ram {
        return Err(TetError::Validation(
            "materialize_query_transform_ram requires `write`: `ram`".into(),
        ));
    }
    let planned = plan_read_for_document(doc, mmap)?;
    let dtype = ElementDtype::from_wire(planned.dtype)?;
    let shape = planned.read_plan.logical_selection_shape.clone();
    let summary = read_tet_summary_v1(mmap)?;
    let budget = ExecutionBudget::resolve(&summary.file_execution, doc.execution.as_ref());
    let default_spill;
    let spill_ref = if let Some(p) = spill_allowlist {
        p
    } else {
        default_spill = SpillPathAllowlist::default_for_tet(
            tet_path,
            std::iter::empty::<std::path::PathBuf>(),
        )?;
        &default_spill
    };
    let fold_policy =
        FoldIoPolicy::resolve(&planned.read_plan, &budget, doc.execution.as_ref(), dtype)?;
    let transform_input = TransformRunInput {
        mmap,
        plan: &planned.read_plan,
        op,
        write: Some(write),
        source_dataset: &doc.dataset,
        max_preview: 0,
        budget,
        dtype,
        spill_allowlist: spill_ref,
        tet_path: Some(tet_path),
        fold_policy,
    };
    let buffer = match materialize_transform_dense_ram(&transform_input)? {
        TransformDenseBuffer::F32(v) => DenseBuffer::F32(v),
        TransformDenseBuffer::F64(v) => DenseBuffer::F64(v),
    };
    Ok(DenseMaterializeOutcome {
        dtype: planned.dtype,
        shape,
        buffer,
    })
}

fn materialize_planned(
    mmap: &[u8],
    plan: &crate::query::types::ReadPlan,
    dtype: ElementDtype,
) -> Result<DenseBuffer, TetError> {
    match dtype {
        ElementDtype::F32 => {
            let (v, _, _) = materialize_read_plan_f32_le(mmap, plan, None)?;
            Ok(DenseBuffer::F32(v))
        }
        ElementDtype::F64 => {
            let (v, _, _) = materialize_read_plan_f64_le(mmap, plan, None)?;
            Ok(DenseBuffer::F64(v))
        }
        ElementDtype::I32 => {
            let (v, _, _) = materialize_read_plan_i32_le(mmap, plan, None)?;
            Ok(DenseBuffer::I32(v))
        }
        ElementDtype::I64 => {
            let (v, _, _) = materialize_read_plan_i64_le(mmap, plan, None)?;
            Ok(DenseBuffer::I64(v))
        }
        ElementDtype::U8 => {
            let (v, _, _) = materialize_read_plan_u8_le(mmap, plan, None)?;
            Ok(DenseBuffer::U8(v))
        }
        ElementDtype::U16 => {
            let (v, _, _) = materialize_read_plan_u16_le(mmap, plan, None)?;
            Ok(DenseBuffer::U16(v))
        }
        ElementDtype::I16 => {
            let (v, _, _) = materialize_read_plan_i16_le(mmap, plan, None)?;
            Ok(DenseBuffer::I16(v))
        }
        ElementDtype::U32 => {
            let (v, _, _) = materialize_read_plan_u32_le(mmap, plan, None)?;
            Ok(DenseBuffer::U32(v))
        }
        ElementDtype::U64 => {
            let (v, _, _) = materialize_read_plan_u64_le(mmap, plan, None)?;
            Ok(DenseBuffer::U64(v))
        }
        ElementDtype::F16 => {
            let (v, _, _) = materialize_read_plan_f16_le(mmap, plan, None)?;
            Ok(DenseBuffer::F16(v))
        }
    }
}

//! Dense logical tensor export for embedders (Python, etc.) — no JSON preview cap.

use std::path::Path;

use crate::catalog::read_tet_summary_v1;
use crate::query::engine::{
    ExecutionBudget, SpillPathAllowlist, materialize_read_plan_f32_le,
    materialize_read_plan_f64_le, materialize_read_plan_i16_le, materialize_read_plan_i32_le,
    materialize_read_plan_i64_le, materialize_read_plan_u8_le, materialize_read_plan_u16_le,
    plan_read_for_document,
};
use crate::query::fold::fold_policy::FoldIoPolicy;
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
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
}

/// Full logical tensor decoded from a `.tet` selection.
#[derive(Debug, Clone)]
pub struct DenseMaterializeOutcome {
    pub dtype: u32,
    pub shape: Vec<u64>,
    pub buffer: DenseBuffer,
}

/// Materialize a dataset selection (no `operation` key) into a dense buffer.
///
/// # Errors
///
/// Validation when `doc` includes an `operation`, or decode / budget failures.
pub fn materialize_query_selection(
    doc: &QueryDocument,
    mmap: &[u8],
) -> Result<DenseMaterializeOutcome, TetError> {
    if doc.operation.is_some() {
        return Err(TetError::Validation(
            "materialize_query_selection requires a selection-only document (no operation key)"
                .into(),
        ));
    }
    let planned = plan_read_for_document(doc, mmap)?;
    let dtype = ElementDtype::from_wire(planned.dtype)?;
    let shape = planned.read_plan.logical_selection_shape.clone();
    let buffer = materialize_planned(mmap, &planned.read_plan, dtype)?;
    Ok(DenseMaterializeOutcome {
        dtype: planned.dtype,
        shape,
        buffer,
    })
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
        ElementDtype::U32 | ElementDtype::F16 | ElementDtype::U64 => Err(TetError::Validation(
            format!("dense numpy export does not support dtype wire tag yet ({dtype:?})"),
        )),
    }
}

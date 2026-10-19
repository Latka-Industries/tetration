//! Merge per-chunk parallel fold accumulators.

use crate::query::dispatch;
use crate::query::fold::{partial_geometry, reduction};
use crate::query::types::{OperationPreviewFields, TetError};

pub(crate) struct ScalarChunkWork {
    pub bytes: u64,
    pub value: reduction::ValueAccum,
    pub arg: reduction::ArgIndexAccum,
}

pub(crate) struct PartialChunkValue {
    pub bytes: u64,
    pub cells: Vec<reduction::ValueAccum>,
    pub saw_any: bool,
}

pub(crate) struct PartialChunkArg {
    pub bytes: u64,
    pub cells: Vec<reduction::ArgIndexAccum>,
    pub saw_any: bool,
}

pub(crate) fn sum_chunk_bytes(bytes: impl IntoIterator<Item = u64>) -> Result<u64, TetError> {
    dispatch::sum_chunk_read_bytes(bytes)
}

pub(crate) fn merge_scalar_chunks(
    parts: &[ScalarChunkWork],
    kind: reduction::ReductionKind,
    n: usize,
    allow_empty: bool,
) -> Result<OperationPreviewFields, TetError> {
    match kind {
        reduction::ReductionKind::ArgMin | reduction::ReductionKind::ArgMax => {
            let mut acc = reduction::ArgIndexAccum::default();
            for p in parts {
                acc.merge_from(&p.arg, kind);
            }
            if acc.is_empty() && !allow_empty {
                return Err(TetError::Validation(
                    "operation requires at least one decoded value from the read plan".into(),
                ));
            }
            Ok(acc.finish_scalar(kind, n).into())
        }
        _ => {
            let mut acc = reduction::ValueAccum::default();
            for p in parts {
                acc.merge_from(&p.value);
            }
            if acc.is_empty() && !allow_empty {
                return Err(TetError::Validation(
                    "operation requires at least one decoded value from the read plan".into(),
                ));
            }
            Ok(acc.finish_scalar(kind).into())
        }
    }
}

pub(crate) fn merge_partial_value_cells(
    dst: &mut [reduction::ValueAccum],
    src: &[reduction::ValueAccum],
) {
    for (d, s) in dst.iter_mut().zip(src) {
        d.merge_from(s);
    }
}

pub(crate) fn merge_partial_arg_cells(
    dst: &mut [reduction::ArgIndexAccum],
    src: &[reduction::ArgIndexAccum],
    kind: reduction::ReductionKind,
) {
    for (d, s) in dst.iter_mut().zip(src) {
        d.merge_from(s, kind);
    }
}

pub(crate) use partial_geometry::reduced_cell_index;

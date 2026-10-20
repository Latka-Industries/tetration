//! Merge associative tier-A/B fold partials (THI-575).

use serde::{Deserialize, Serialize};

use crate::query::fold::reduction::{
    ArgIndexAccum, ArgIndexAccumWire, ReductionKind, ValueAccum, ValueAccumWire,
};
use crate::query::types::{Operation, QueryExecutionPreview, TetError};

/// Serializable sufficient stats for one scalar (or arg) fold shard.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct FoldPartialState {
    value: ValueAccumWire,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    arg: Option<ArgIndexAccumWire>,
    bytes_read: u64,
}

impl FoldPartialState {
    /// Empty partial (no values visited).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            value: ValueAccum::default().to_wire(),
            arg: None,
            bytes_read: 0,
        }
    }

    /// Bytes read across contributing shards.
    #[must_use]
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    pub(crate) fn from_accums(
        value: ValueAccum,
        arg: ArgIndexAccum,
        bytes_read: u64,
        kind: ReductionKind,
    ) -> Self {
        let arg =
            matches!(kind, ReductionKind::ArgMin | ReductionKind::ArgMax).then(|| arg.to_wire());
        Self {
            value: value.to_wire(),
            arg,
            bytes_read,
        }
    }
}

/// True when `op` can be sharded and merged via [`FoldPartialState`] (tier A/B scalar folds).
#[must_use]
pub fn operation_is_shard_mergeable(op: &Operation) -> bool {
    !op.requires_materialize() && !op.requires_transform()
}

/// Merge `src` into `dst` (disjoint chunk visits). Kind selects argmin/argmax combine law.
///
/// # Errors
///
/// Returns [`TetError::Validation`] when `kind` is not a known fold kind for merge.
pub fn merge_fold_partials(
    dst: &mut FoldPartialState,
    src: &FoldPartialState,
    op: &Operation,
) -> Result<(), TetError> {
    if !operation_is_shard_mergeable(op) {
        return Err(TetError::Validation(
            "operation is not shard-mergeable (tier-C or transform)".into(),
        ));
    }
    let kind = ReductionKind::from(op);
    let mut dst_value = ValueAccum::from_wire(&dst.value);
    dst_value.merge_from(&ValueAccum::from_wire(&src.value));
    dst.value = dst_value.to_wire();
    dst.bytes_read = dst
        .bytes_read
        .checked_add(src.bytes_read)
        .ok_or_else(|| TetError::Validation("bytes_read overflow merging partials".into()))?;

    match kind {
        ReductionKind::ArgMin | ReductionKind::ArgMax => {
            let mut dst_arg = dst
                .arg
                .as_ref()
                .map(ArgIndexAccum::from_wire)
                .unwrap_or_default();
            let src_arg = src
                .arg
                .as_ref()
                .map(ArgIndexAccum::from_wire)
                .unwrap_or_default();
            dst_arg.merge_from(&src_arg, kind);
            dst.arg = Some(dst_arg.to_wire());
        }
        _ => {}
    }
    Ok(())
}

/// Finish merged sufficient stats into operation preview fields.
///
/// # Errors
///
/// Returns [`TetError::Validation`] when the op is not mergeable or the merged state is empty
/// when at least one value was required.
pub fn finish_fold_partial(
    state: FoldPartialState,
    op: &Operation,
    logical_element_count: usize,
    allow_empty: bool,
) -> Result<QueryExecutionPreview, TetError> {
    if !operation_is_shard_mergeable(op) {
        return Err(TetError::Validation(
            "operation is not shard-mergeable (tier-C or transform)".into(),
        ));
    }
    let kind = ReductionKind::from(op);
    let fields: crate::query::types::OperationPreviewFields = match kind {
        ReductionKind::ArgMin | ReductionKind::ArgMax => {
            let arg = state
                .arg
                .as_ref()
                .map(ArgIndexAccum::from_wire)
                .unwrap_or_default();
            if arg.is_empty() && !allow_empty {
                return Err(TetError::Validation(
                    "operation requires at least one decoded value from the read plan".into(),
                ));
            }
            arg.finish_scalar(kind, logical_element_count).into()
        }
        _ => {
            let value = ValueAccum::from_wire(&state.value);
            if value.is_empty() && !allow_empty {
                return Err(TetError::Validation(
                    "operation requires at least one decoded value from the read plan".into(),
                ));
            }
            value.finish_scalar(kind).into()
        }
    };
    Ok(QueryExecutionPreview::from(fields))
}

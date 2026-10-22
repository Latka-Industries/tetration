//! In-process multi-shard scalar fold harness (no RPC).

use crate::query::decode::chunk_decode::{fold_planned_chunk_f32, fold_planned_chunk_f64};
use crate::query::fold::reduction::{ArgIndexAccum, ReductionKind, ValueAccum};
use crate::query::types::{Operation, QueryExecutionPreview, ReadPlan, TetError};
use crate::utils::dtype::ElementDtype;

use super::merge::{
    FoldPartialState, finish_fold_partial, merge_fold_partials, operation_is_shard_mergeable,
};
use super::shard::{ShardPartition, plan_chunk_shards, split_read_plan};
use super::spill::refuse_distributed_spill;

/// Fold one shard plan into sufficient stats (f32 / f64 scalar paths).
fn fold_shard_to_partial(
    mmap: &[u8],
    plan: &ReadPlan,
    kind: ReductionKind,
    dtype: ElementDtype,
) -> Result<FoldPartialState, TetError> {
    let mut value = ValueAccum::default();
    let mut arg = ArgIndexAccum::default();
    let mut bytes_read = 0u64;
    match dtype {
        ElementDtype::F32 => {
            let mut preview = Vec::<f32>::new();
            for c in &plan.chunks {
                bytes_read = bytes_read
                    .checked_add(fold_planned_chunk_f32(
                        mmap,
                        plan,
                        c,
                        kind,
                        &mut value,
                        &mut arg,
                        &mut preview,
                    )?)
                    .ok_or_else(|| TetError::Validation("bytes_read overflow".into()))?;
            }
        }
        ElementDtype::F64 => {
            let mut preview = Vec::<f64>::new();
            for c in &plan.chunks {
                bytes_read = bytes_read
                    .checked_add(fold_planned_chunk_f64(
                        mmap,
                        plan,
                        c,
                        kind,
                        &mut value,
                        &mut arg,
                        &mut preview,
                    )?)
                    .ok_or_else(|| TetError::Validation("bytes_read overflow".into()))?;
            }
        }
        _ => {
            return Err(TetError::Validation(
                "sharded scalar harness currently supports f32 and f64 only".into(),
            ));
        }
    }
    Ok(FoldPartialState::from_accums(value, arg, bytes_read, kind))
}

/// Prove merge without RPC: split `plan` into `n_shards`, fold each, merge partials, finish.
///
/// # Errors
///
/// Validation when the op is not shard-mergeable, spill is requested, dtype unsupported,
/// or merge/finish fails.
pub fn execute_sharded_scalar_fold_local(
    mmap: &[u8],
    plan: &ReadPlan,
    op: &Operation,
    dtype: ElementDtype,
    n_shards: usize,
    partition: ShardPartition,
    write: Option<&crate::query::types::WriteHints>,
) -> Result<QueryExecutionPreview, TetError> {
    if !operation_is_shard_mergeable(op) {
        return Err(TetError::Validation(
            "operation is not shard-mergeable (tier-C or transform)".into(),
        ));
    }
    refuse_distributed_spill(write)?;
    let n_shards = n_shards.max(1);
    let kind = ReductionKind::from(op);
    let specs = plan_chunk_shards(plan.chunks.len(), n_shards, partition);
    let allow_empty = plan.cell_filter.is_some() || plan.label_eq.is_some();
    let mut merged = FoldPartialState::empty();
    for spec in &specs {
        let shard_plan = split_read_plan(plan, spec);
        let part = fold_shard_to_partial(mmap, &shard_plan, kind, dtype)?;
        merge_fold_partials(&mut merged, &part, op)?;
    }
    finish_fold_partial(merged, op, plan.logical_element_count, allow_empty)
}

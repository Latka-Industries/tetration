//! In-process and multi-process multi-shard scalar fold harnesses.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::layout::mmap_file_read;
use crate::query::decode::chunk_decode::{fold_planned_chunk_f32, fold_planned_chunk_f64};
use crate::query::engine::plan_read_for_document;
use crate::query::fold::reduction::{ArgIndexAccum, ReductionKind, ValueAccum};
use crate::query::types::{
    Operation, QueryDocument, QueryExecutionPreview, ReadPlan, TetError, WriteHints,
};
use crate::utils::dtype::ElementDtype;

use super::job::{WORKER_PROTOCOL_V1, WorkerJobV1, WorkerPartialV1, WorkerPinV1};
use super::merge::{
    FoldPartialState, finish_fold_partial, merge_fold_partials, operation_is_shard_mergeable,
};
use super::shard::{ShardPartition, plan_chunk_shards, split_read_plan};
use super::spill::refuse_distributed_spill;

/// Fold one shard plan into sufficient stats (f32 / f64 scalar paths).
///
/// # Errors
///
/// Unsupported dtype or fold decode failures.
pub(crate) fn fold_shard_to_partial(
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

/// Resolve a local filesystem path from [`WorkerPinV1::package_uri`].
///
/// Accepts a bare path or a `file://` URI. Other schemes are refused (0.7.1 harness).
///
/// # Errors
///
/// Non-file URI schemes.
pub fn resolve_worker_pin_path(pin: &WorkerPinV1) -> Result<PathBuf, TetError> {
    let uri = pin.package_uri.trim();
    if let Some(rest) = uri.strip_prefix("file://") {
        // `file:///abs` or `file://localhost/abs` — path after optional host.
        let owned = if rest.starts_with('/') {
            rest.to_string()
        } else if let Some((_, p)) = rest.split_once('/') {
            format!("/{p}")
        } else {
            rest.to_string()
        };
        return Ok(PathBuf::from(owned));
    }
    if uri.contains("://") {
        return Err(TetError::Validation(format!(
            "worker pin package_uri scheme not supported in localhost harness (got {uri:?}; use a path or file://)"
        )));
    }
    Ok(PathBuf::from(uri))
}

/// Execute one [`WorkerJobV1`]: open the pin, fold the shard plan, return a partial.
///
/// Rebuilds dtype from the sealed file + document. Uses `job.plan` chunk subset as-is
/// (cell filters are not on the JSON plan wire — prefer filter-free jobs for process
/// boundaries, or rebuild filters in a later cut).
///
/// # Errors
///
/// Spill refuse, non-mergeable op, pin open, dtype, or fold failures.
pub fn execute_worker_job_v1(job: &WorkerJobV1) -> Result<WorkerPartialV1, TetError> {
    if job.protocol != WORKER_PROTOCOL_V1 {
        return Err(TetError::Validation(format!(
            "unsupported worker protocol {} (want {WORKER_PROTOCOL_V1})",
            job.protocol
        )));
    }
    refuse_distributed_spill(job.document.write.as_ref())?;
    let op = job
        .document
        .operation
        .as_ref()
        .ok_or_else(|| TetError::Validation("worker job document missing operation".into()))?;
    if !operation_is_shard_mergeable(op) {
        return Err(TetError::Validation(
            "operation is not shard-mergeable (tier-C or transform)".into(),
        ));
    }
    let path = resolve_worker_pin_path(&job.pin)?;
    let mmap = mmap_file_read(&path).map_err(|e| {
        TetError::Validation(format!("worker failed to mmap pin {}: {e}", path.display()))
    })?;
    let object_len = u64::try_from(mmap.len())
        .map_err(|_| TetError::Validation("mmap length does not fit u64".into()))?;
    if object_len != job.pin.object_len {
        return Err(TetError::Validation(format!(
            "pin object_len mismatch: job has {} but file is {object_len}",
            job.pin.object_len
        )));
    }
    let planned = plan_read_for_document(&job.document, &mmap)?;
    let dtype = ElementDtype::from_wire(planned.dtype)?;
    let kind = ReductionKind::from(op);
    let partial = fold_shard_to_partial(&mmap, &job.plan, kind, dtype)?;
    Ok(WorkerPartialV1 {
        protocol: WORKER_PROTOCOL_V1,
        job_id: job.job_id.clone(),
        shard_id: job.shard_id,
        pin_generation: job.pin.generation,
        partial,
    })
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
    write: Option<&WriteHints>,
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

/// Inputs for [`execute_sharded_scalar_fold_multiprocess`].
#[derive(Debug, Clone, Copy)]
pub struct MultiprocessFoldOpts<'a> {
    pub tet_path: &'a Path,
    pub plan: &'a ReadPlan,
    pub doc: &'a QueryDocument,
    pub op: &'a Operation,
    pub n_shards: usize,
    pub partition: ShardPartition,
    pub write: Option<&'a WriteHints>,
    pub worker_bin: &'a Path,
}

/// Spawn `n_shards` worker processes (`worker_bin worker`), merge partials, finish.
///
/// Each worker opens `tet_path` via [`WorkerPinV1`]. Intended for localhost proof of the
/// JSON protocol — not a cluster scheduler.
///
/// # Errors
///
/// Spill refuse, non-mergeable op, process spawn/IO, JSON parse, or merge/finish failures.
pub fn execute_sharded_scalar_fold_multiprocess(
    opts: MultiprocessFoldOpts<'_>,
) -> Result<QueryExecutionPreview, TetError> {
    let MultiprocessFoldOpts {
        tet_path,
        plan,
        doc,
        op,
        n_shards,
        partition,
        write,
        worker_bin,
    } = opts;
    if !operation_is_shard_mergeable(op) {
        return Err(TetError::Validation(
            "operation is not shard-mergeable (tier-C or transform)".into(),
        ));
    }
    refuse_distributed_spill(write)?;
    let n_shards = n_shards.max(1);
    let meta = std::fs::metadata(tet_path)
        .map_err(|e| TetError::Validation(format!("stat {}: {e}", tet_path.display())))?;
    let object_len = meta.len();
    let pin = WorkerPinV1 {
        generation: 0,
        object_key: String::new(),
        object_len,
        package_uri: tet_path.display().to_string(),
    };
    let specs = plan_chunk_shards(plan.chunks.len(), n_shards, partition);
    let job_id = "local-mp".to_string();
    let mut merged = FoldPartialState::empty();

    for spec in &specs {
        let shard_plan = split_read_plan(plan, spec);
        let job = WorkerJobV1 {
            protocol: WORKER_PROTOCOL_V1,
            job_id: job_id.clone(),
            shard_id: spec.shard_id,
            n_shards,
            pin: pin.clone(),
            document: doc.clone(),
            chunk_indices: spec.chunk_indices.clone(),
            plan: shard_plan,
            preview_max: 0,
        };
        let job_json = serde_json::to_vec(&job)
            .map_err(|e| TetError::Validation(format!("serialize worker job: {e}")))?;

        let mut child = Command::new(worker_bin)
            .arg("worker")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                TetError::Validation(format!("spawn {} worker: {e}", worker_bin.display()))
            })?;
        {
            let mut stdin = child
                .stdin
                .take()
                .ok_or_else(|| TetError::Validation("worker stdin unavailable".into()))?;
            stdin
                .write_all(&job_json)
                .map_err(|e| TetError::Validation(format!("write worker stdin: {e}")))?;
        }
        let output = child.wait_with_output().map_err(|e| {
            TetError::Validation(format!("wait worker shard {}: {e}", spec.shard_id))
        })?;
        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            return Err(TetError::Validation(format!(
                "tet worker shard {} failed (status {:?}): {err}",
                spec.shard_id,
                output.status.code()
            )));
        }
        let partial: WorkerPartialV1 = serde_json::from_slice(&output.stdout).map_err(|e| {
            TetError::Validation(format!(
                "parse worker partial for shard {}: {e}; stdout={}",
                spec.shard_id,
                String::from_utf8_lossy(&output.stdout)
            ))
        })?;
        if partial.job_id != job_id || partial.shard_id != spec.shard_id {
            return Err(TetError::Validation(format!(
                "worker partial id mismatch: got job={} shard={}",
                partial.job_id, partial.shard_id
            )));
        }
        merge_fold_partials(&mut merged, &partial.partial, op)?;
    }

    let allow_empty = plan.cell_filter.is_some() || plan.label_eq.is_some();
    finish_fold_partial(merged, op, plan.logical_element_count, allow_empty)
}

/// Write a [`WorkerPartialV1`] as JSON to `out` (CLI helper).
///
/// # Errors
///
/// Serialization or write failures.
pub fn write_worker_partial_json(
    partial: &WorkerPartialV1,
    out: &mut dyn Write,
) -> Result<(), TetError> {
    serde_json::to_writer(out, partial)
        .map_err(|e| TetError::Validation(format!("write worker partial JSON: {e}")))?;
    Ok(())
}

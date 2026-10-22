//! In-process and localhost multi-process multi-shard harnesses.
//!
//! - [`execute_sharded_scalar_fold_local`] — same-process N shards (no OS spawn).
//! - [`execute_sharded_materialize_local`] / [`execute_sharded_tier_c_local`] — dense path (0.7.2).
//! - [`execute_sharded_transform_ram_local`] — transform `write: ram` (pass-1 local, pass-2 sharded).
//! - [`execute_worker_job_v1`] / `tet worker` — one job JSON → one partial JSON.
//! - [`execute_sharded_scalar_fold_multiprocess`] / [`execute_sharded_materialize_multiprocess`]
//!   — spawn N `tet worker` children.
//!
//! Product boundary (spill refuse, no cluster product):
//! [`docs/distributed_boundary.md`](../../../../docs/distributed_boundary.md).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::catalog::read_tet_summary_v1;
use crate::layout::mmap_file_read;
use crate::query::decode::chunk_decode::{fold_planned_chunk_f32, fold_planned_chunk_f64};
use crate::query::engine::budget::{ExecutionBudget, MemoryStrategy};
use crate::query::engine::plan_read_for_document;
use crate::query::fold::fold_policy::FoldIoPolicy;
use crate::query::fold::reduction::{ArgIndexAccum, ReductionKind, ValueAccum};
use crate::query::materialize::stats::run_tier_c_operation;
use crate::query::materialize::{LogicalF32Backing, MaterializedLogical};
use crate::query::transform::{
    TransformWarnings, collect_transform_stats, transform_read_plan_f32_le_ram,
};
use crate::query::types::{
    Operation, QueryDocument, QueryExecutionPreview, ReadPlan, TetError, WriteHints, WriteTarget,
};
use crate::utils::dtype::ElementDtype;

use super::dense::{
    DenseShardPartial, assemble_dense_f32, compact_buffer_to_dense_f32, materialize_shard_dense_f32,
};
use super::job::{WORKER_PROTOCOL_V1, WorkerJobV1, WorkerPartialV1, WorkerPinV1};
use super::merge::{
    FoldPartialState, finish_fold_partial, merge_fold_partials, operation_is_dense_distributable,
    operation_is_shard_mergeable,
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
    Ok(FoldPartialState::from_accums(&value, arg, bytes_read, kind))
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

fn open_worker_pin(job: &WorkerJobV1) -> Result<(PathBuf, memmap2::Mmap), TetError> {
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
    Ok((path, mmap))
}

fn worker_partial_fold(job: &WorkerJobV1, partial: FoldPartialState) -> WorkerPartialV1 {
    WorkerPartialV1 {
        protocol: WORKER_PROTOCOL_V1,
        job_id: job.job_id.clone(),
        shard_id: job.shard_id,
        pin_generation: job.pin.generation,
        partial,
        dense: None,
    }
}

fn worker_partial_dense(job: &WorkerJobV1, dense: DenseShardPartial) -> WorkerPartialV1 {
    WorkerPartialV1 {
        protocol: WORKER_PROTOCOL_V1,
        job_id: job.job_id.clone(),
        shard_id: job.shard_id,
        pin_generation: job.pin.generation,
        partial: FoldPartialState::empty(),
        dense: Some(dense),
    }
}

/// Execute one [`WorkerJobV1`]: open the pin, run the shard, return a partial.
///
/// Fold jobs return [`WorkerPartialV1::partial`]. Dense materialize jobs (tier-C op or
/// selection-only document) return [`WorkerPartialV1::dense`]. Transform jobs on the worker
/// are refused in 0.7.2 multiprocess (pass-1 stats are not on the wire yet); use the
/// in-process transform harness.
///
/// # Errors
///
/// Spill refuse, unsupported op, pin open, dtype, or fold/materialize failures.
pub fn execute_worker_job_v1(job: &WorkerJobV1) -> Result<WorkerPartialV1, TetError> {
    if job.protocol != WORKER_PROTOCOL_V1 {
        return Err(TetError::Validation(format!(
            "unsupported worker protocol {} (want {WORKER_PROTOCOL_V1})",
            job.protocol
        )));
    }
    refuse_distributed_spill(job.document.write.as_ref())?;
    let (_path, mmap) = open_worker_pin(job)?;
    let planned = plan_read_for_document(&job.document, &mmap)?;
    let dtype = ElementDtype::from_wire(planned.dtype)?;

    match job.document.operation.as_ref() {
        None => {
            if dtype != ElementDtype::F32 {
                return Err(TetError::Validation(
                    "distributed dense materialize currently supports f32 only".into(),
                ));
            }
            let dense = materialize_shard_dense_f32(&mmap, &job.plan)?;
            Ok(worker_partial_dense(job, dense))
        }
        Some(op) if operation_is_shard_mergeable(op) => {
            let kind = ReductionKind::from(op);
            let partial = fold_shard_to_partial(&mmap, &job.plan, kind, dtype)?;
            Ok(worker_partial_fold(job, partial))
        }
        Some(op) if op.requires_materialize() => {
            if dtype != ElementDtype::F32 {
                return Err(TetError::Validation(
                    "distributed dense materialize currently supports f32 only".into(),
                ));
            }
            let dense = materialize_shard_dense_f32(&mmap, &job.plan)?;
            Ok(worker_partial_dense(job, dense))
        }
        Some(op) if op.requires_transform() => Err(TetError::Validation(
            "distributed transform pass-2 over multiprocess requires stats on the job wire (use execute_sharded_transform_ram_local in 0.7.2)"
                .into(),
        )),
        Some(_) => Err(TetError::Validation(
            "operation is not supported on the distributed worker path".into(),
        )),
    }
}

/// Prove fold merge without RPC: split `plan` into `n_shards`, fold each, merge partials, finish.
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
    finish_fold_partial(&merged, op, plan.logical_element_count, allow_empty)
}

/// Shard-materialize a selection (or tier-C input) into a dense f32 buffer (in-process).
///
/// # Errors
///
/// Spill refuse, non-f32 dtype, shard decode, or assemble failures.
pub fn execute_sharded_materialize_local(
    mmap: &[u8],
    plan: &ReadPlan,
    dtype: ElementDtype,
    n_shards: usize,
    partition: ShardPartition,
    write: Option<&WriteHints>,
) -> Result<(Vec<f32>, u64), TetError> {
    refuse_distributed_spill(write)?;
    if dtype != ElementDtype::F32 {
        return Err(TetError::Validation(
            "sharded materialize harness currently supports f32 only".into(),
        ));
    }
    let n_shards = n_shards.max(1);
    let specs = plan_chunk_shards(plan.chunks.len(), n_shards, partition);
    let mut partials = Vec::with_capacity(specs.len());
    for spec in &specs {
        let shard_plan = split_read_plan(plan, spec);
        partials.push(materialize_shard_dense_f32(mmap, &shard_plan)?);
    }
    let allow_holes = plan.cell_filter.is_some() || plan.label_eq.is_some();
    assemble_dense_f32(&partials, plan.logical_element_count, allow_holes)
}

/// Distributed tier-C: shard-materialize, assemble, finish median/quantile/histogram locally.
///
/// # Errors
///
/// Non-dense op, spill refuse, materialize/assemble, or tier-C finish failures.
pub fn execute_sharded_tier_c_local(
    mmap: &[u8],
    plan: &ReadPlan,
    op: &Operation,
    dtype: ElementDtype,
    n_shards: usize,
    partition: ShardPartition,
    write: Option<&WriteHints>,
) -> Result<QueryExecutionPreview, TetError> {
    if !op.requires_materialize() {
        return Err(TetError::Validation(
            "execute_sharded_tier_c_local requires a materialize-required (tier-C) operation"
                .into(),
        ));
    }
    let (values, bytes_read) =
        execute_sharded_materialize_local(mmap, plan, dtype, n_shards, partition, write)?;
    let materialized = MaterializedLogical::F32 {
        backing: LogicalF32Backing::InMemory(values),
        total_bytes_read_from_disk: bytes_read,
        strategy: MemoryStrategy::InMemoryMaterialize,
    };
    let fields = run_tier_c_operation(&materialized, plan, op)?;
    let mut preview = QueryExecutionPreview::from(fields);
    preview.total_bytes_read_from_disk = bytes_read;
    preview.memory_strategy = Some(MemoryStrategy::InMemoryMaterialize.as_str());
    Ok(preview)
}

/// Inputs for [`execute_sharded_transform_ram_local`].
#[derive(Debug, Clone, Copy)]
pub struct TransformRamLocalOpts<'a> {
    pub mmap: &'a [u8],
    pub plan: &'a ReadPlan,
    pub op: &'a Operation,
    pub dtype: ElementDtype,
    pub n_shards: usize,
    pub partition: ShardPartition,
    pub write: Option<&'a WriteHints>,
    pub tet_path: Option<&'a Path>,
}

/// Distributed transform with `write: ram` (in-process).
///
/// Pass-1 stats run on the full plan (coordinator). Pass-2 apply is sharded; dense shards
/// are assembled into one buffer.
///
/// # Errors
///
/// Non-transform op, non-RAM write, spill refuse, stats/apply, or assemble failures.
pub fn execute_sharded_transform_ram_local(
    opts: TransformRamLocalOpts<'_>,
) -> Result<(Vec<f32>, u64), TetError> {
    let TransformRamLocalOpts {
        mmap,
        plan,
        op,
        dtype,
        n_shards,
        partition,
        write,
        tet_path,
    } = opts;
    if !op.requires_transform() {
        return Err(TetError::Validation(
            "execute_sharded_transform_ram_local requires a transform operation".into(),
        ));
    }
    refuse_distributed_spill(write)?;
    let write = write.ok_or_else(|| {
        TetError::Validation("distributed transform requires `write`: `ram`".into())
    })?;
    if write.target != WriteTarget::Ram {
        return Err(TetError::Validation(
            "distributed transform requires `write`: `ram`".into(),
        ));
    }
    if dtype != ElementDtype::F32 {
        return Err(TetError::Validation(
            "sharded transform harness currently supports f32 only".into(),
        ));
    }
    let summary = read_tet_summary_v1(mmap)?;
    let budget = ExecutionBudget::resolve(&summary.file_execution, None);
    let fold_policy = FoldIoPolicy::resolve(plan, &budget, None, dtype, true)?;
    let (stats, _fields, pass1_bytes) =
        collect_transform_stats(mmap, plan, op, dtype, &fold_policy, tet_path)?;

    let n_shards = n_shards.max(1);
    let specs = plan_chunk_shards(plan.chunks.len(), n_shards, partition);
    let mut partials = Vec::with_capacity(specs.len());
    let mut pass2_bytes = 0u64;
    let mut warnings = TransformWarnings::default();
    for spec in &specs {
        let shard_plan = split_read_plan(plan, spec);
        let (buf, bytes) =
            transform_read_plan_f32_le_ram(mmap, &shard_plan, &stats, &mut warnings)?;
        pass2_bytes = pass2_bytes
            .checked_add(bytes)
            .ok_or_else(|| TetError::Validation("bytes_read overflow".into()))?;
        partials.push(compact_buffer_to_dense_f32(mmap, &shard_plan, &buf, bytes)?);
    }
    let allow_holes = plan.cell_filter.is_some() || plan.label_eq.is_some();
    let (values, _) = assemble_dense_f32(&partials, plan.logical_element_count, allow_holes)?;
    let total = pass1_bytes
        .checked_add(pass2_bytes)
        .ok_or_else(|| TetError::Validation("bytes_read overflow".into()))?;
    Ok((values, total))
}

/// Inputs for [`execute_sharded_scalar_fold_multiprocess`].
///
/// `tet_path` must be a sealed local `.tet` (or path/`file://` URI the workers can mmap).
/// `worker_bin` is typically the `tet` CLI (`CARGO_BIN_EXE_tet` in integration tests).
#[derive(Debug, Clone, Copy)]
pub struct MultiprocessFoldOpts<'a> {
    /// Sealed `.tet` path shared by every worker pin.
    pub tet_path: &'a Path,
    /// Full coordinator [`ReadPlan`] (before shard split).
    pub plan: &'a ReadPlan,
    /// Query document (operation + dataset; must match `plan`).
    pub doc: &'a QueryDocument,
    /// Reduction op (must be shard-mergeable).
    pub op: &'a Operation,
    /// Number of worker processes (≥ 1 after clamp).
    pub n_shards: usize,
    /// How to partition `plan.chunks` across workers.
    pub partition: ShardPartition,
    /// Optional write hints; non-RAM targets are refused.
    pub write: Option<&'a WriteHints>,
    /// Path to the `tet` binary (invoked as `tet worker`).
    pub worker_bin: &'a Path,
}

/// Inputs for [`execute_sharded_materialize_multiprocess`] (dense / tier-C materialize).
#[derive(Debug, Clone, Copy)]
pub struct MultiprocessDenseOpts<'a> {
    /// Sealed `.tet` path shared by every worker pin.
    pub tet_path: &'a Path,
    /// Full coordinator [`ReadPlan`] (before shard split).
    pub plan: &'a ReadPlan,
    /// Query document (tier-C operation or selection-only).
    pub doc: &'a QueryDocument,
    /// Number of worker processes (≥ 1 after clamp).
    pub n_shards: usize,
    /// How to partition `plan.chunks` across workers.
    pub partition: ShardPartition,
    /// Optional write hints; non-RAM targets are refused.
    pub write: Option<&'a WriteHints>,
    /// Path to the `tet` binary (invoked as `tet worker`).
    pub worker_bin: &'a Path,
}

fn spawn_worker_partial(worker_bin: &Path, job: &WorkerJobV1) -> Result<WorkerPartialV1, TetError> {
    let job_json = serde_json::to_vec(job)
        .map_err(|e| TetError::Validation(format!("serialize worker job: {e}")))?;

    let mut child = Command::new(worker_bin)
        .arg("worker")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| TetError::Validation(format!("spawn {} worker: {e}", worker_bin.display())))?;
    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| TetError::Validation("worker stdin unavailable".into()))?;
        stdin
            .write_all(&job_json)
            .map_err(|e| TetError::Validation(format!("write worker stdin: {e}")))?;
    }
    let output = child
        .wait_with_output()
        .map_err(|e| TetError::Validation(format!("wait worker shard {}: {e}", job.shard_id)))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(TetError::Validation(format!(
            "tet worker shard {} failed (status {:?}): {err}",
            job.shard_id,
            output.status.code()
        )));
    }
    let partial: WorkerPartialV1 = serde_json::from_slice(&output.stdout).map_err(|e| {
        TetError::Validation(format!(
            "parse worker partial for shard {}: {e}; stdout={}",
            job.shard_id,
            String::from_utf8_lossy(&output.stdout)
        ))
    })?;
    if partial.job_id != job.job_id || partial.shard_id != job.shard_id {
        return Err(TetError::Validation(format!(
            "worker partial id mismatch: got job={} shard={}",
            partial.job_id, partial.shard_id
        )));
    }
    Ok(partial)
}

fn pin_for_tet(tet_path: &Path) -> Result<WorkerPinV1, TetError> {
    let meta = std::fs::metadata(tet_path)
        .map_err(|e| TetError::Validation(format!("stat {}: {e}", tet_path.display())))?;
    Ok(WorkerPinV1 {
        generation: 0,
        object_key: String::new(),
        object_len: meta.len(),
        package_uri: tet_path.display().to_string(),
    })
}

/// Spawn `n_shards` worker processes (`worker_bin worker`), merge fold partials, finish.
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
    let pin = pin_for_tet(tet_path)?;
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
        let partial = spawn_worker_partial(worker_bin, &job)?;
        merge_fold_partials(&mut merged, &partial.partial, op)?;
    }

    let allow_empty = plan.cell_filter.is_some() || plan.label_eq.is_some();
    finish_fold_partial(&merged, op, plan.logical_element_count, allow_empty)
}

/// Spawn N workers for dense materialize; assemble f32 buffer at the coordinator.
///
/// Document may be selection-only or carry a tier-C operation (workers only materialize).
///
/// # Errors
///
/// Spill refuse, transform docs, process/IO, missing dense payload, or assemble failures.
pub fn execute_sharded_materialize_multiprocess(
    opts: MultiprocessDenseOpts<'_>,
) -> Result<(Vec<f32>, u64), TetError> {
    let MultiprocessDenseOpts {
        tet_path,
        plan,
        doc,
        n_shards,
        partition,
        write,
        worker_bin,
    } = opts;
    refuse_distributed_spill(write)?;
    if let Some(op) = doc.operation.as_ref() {
        if op.requires_transform() {
            return Err(TetError::Validation(
                "multiprocess transform is not wired in 0.7.2; use execute_sharded_transform_ram_local"
                    .into(),
            ));
        }
        if operation_is_shard_mergeable(op) {
            return Err(TetError::Validation(
                "execute_sharded_materialize_multiprocess expects selection-only or tier-C docs"
                    .into(),
            ));
        }
        if !operation_is_dense_distributable(op) {
            return Err(TetError::Validation(
                "operation is not dense-distributable".into(),
            ));
        }
    }
    let n_shards = n_shards.max(1);
    let pin = pin_for_tet(tet_path)?;
    let specs = plan_chunk_shards(plan.chunks.len(), n_shards, partition);
    let job_id = "local-mp-dense".to_string();
    let mut partials = Vec::with_capacity(specs.len());

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
        let partial = spawn_worker_partial(worker_bin, &job)?;
        let dense = partial.dense.ok_or_else(|| {
            TetError::Validation(format!(
                "worker shard {} returned fold partial without dense payload",
                spec.shard_id
            ))
        })?;
        partials.push(dense);
    }

    let allow_holes = plan.cell_filter.is_some() || plan.label_eq.is_some();
    assemble_dense_f32(&partials, plan.logical_element_count, allow_holes)
}

/// Multiprocess tier-C: dense materialize across workers, finish op on the coordinator.
///
/// # Errors
///
/// Same as [`execute_sharded_materialize_multiprocess`], plus tier-C finish failures.
pub fn execute_sharded_tier_c_multiprocess(
    opts: MultiprocessDenseOpts<'_>,
    op: &Operation,
) -> Result<QueryExecutionPreview, TetError> {
    if !op.requires_materialize() {
        return Err(TetError::Validation(
            "execute_sharded_tier_c_multiprocess requires a materialize-required operation".into(),
        ));
    }
    let plan = opts.plan;
    let (values, bytes_read) = execute_sharded_materialize_multiprocess(opts)?;
    let materialized = MaterializedLogical::F32 {
        backing: LogicalF32Backing::InMemory(values),
        total_bytes_read_from_disk: bytes_read,
        strategy: MemoryStrategy::InMemoryMaterialize,
    };
    let fields = run_tier_c_operation(&materialized, plan, op)?;
    let mut preview = QueryExecutionPreview::from(fields);
    preview.total_bytes_read_from_disk = bytes_read;
    preview.memory_strategy = Some(MemoryStrategy::InMemoryMaterialize.as_str());
    Ok(preview)
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

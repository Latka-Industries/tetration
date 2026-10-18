//! Two-pass element-wise transforms (`transform` wire key).
//!
//! Pass 1 ([`stats`]) folds statistics over the logical selection. Pass 2 ([`apply`])
//! decodes chunks again and rewrites each element in place (RAM or spill). Division by
//! zero (or an invalid `sqrt` shift) yields **NaN** and is recorded in
//! [`warnings::TransformWarnings`] for the execution preview.

mod apply;
mod sidecar;
mod stats;
mod target;
mod warnings;

use std::path::Path;

use crate::query::dispatch;
use crate::query::engine::{
    budget::{ExecutionBudget, MemoryStrategy},
    spill_policy::SpillPathAllowlist,
};
use crate::query::fold::fold_policy::FoldIoPolicy;
use crate::query::materialize::DecodePreviewBundle;
use crate::query::types::{
    Operation, OperationPreviewFields, ReadPlan, TetError, TransformMethod, WriteHints, WriteTarget,
};
use crate::utils::dtype::ElementDtype;

/// Result of [`run_transform`]: decode preview, pass-1 stats, spill metadata, and warnings.
pub(crate) struct TransformOutcome {
    pub total_bytes_read_from_disk: u64,
    pub strategy: MemoryStrategy,
    pub spill_path: Option<String>,
    pub spill_bytes: Option<u64>,
    pub bundle: DecodePreviewBundle,
    pub operation: OperationPreviewFields,
}

use sidecar::SidecarContext;

/// Inputs shared by pass 1 and pass 2 of a transform operation.
pub(crate) struct TransformRunInput<'a> {
    pub mmap: &'a [u8],
    pub plan: &'a ReadPlan,
    pub op: &'a Operation,
    pub write: Option<&'a WriteHints>,
    pub source_dataset: &'a str,
    pub max_preview: usize,
    pub budget: ExecutionBudget,
    pub dtype: ElementDtype,
    pub spill_allowlist: &'a SpillPathAllowlist,
    pub tet_path: Option<&'a Path>,
    pub fold_policy: FoldIoPolicy,
}

struct Pass2Outcome {
    strategy: MemoryStrategy,
    spill_path: Option<String>,
    spill_bytes: Option<u64>,
    bundle: DecodePreviewBundle,
    pass2_bytes: u64,
    warnings: warnings::TransformWarnings,
}

const ERR_TRANSFORM_FLOAT: &str = "transform requires f32 or f64 datasets";

fn capped_preview<T: Copy>(values: &[T], max_preview: usize, logical_len: usize) -> (Vec<T>, bool) {
    let truncated = logical_len > max_preview;
    let preview = if max_preview == 0 {
        Vec::new()
    } else {
        values[..max_preview.min(logical_len)].to_vec()
    };
    (preview, truncated)
}

fn transform_ram_outcome(
    bundle: DecodePreviewBundle,
    bytes: u64,
    warnings: warnings::TransformWarnings,
) -> Pass2Outcome {
    Pass2Outcome {
        strategy: MemoryStrategy::TransformRam,
        spill_path: None,
        spill_bytes: None,
        bundle,
        pass2_bytes: bytes,
        warnings,
    }
}

fn run_pass2(
    input: &TransformRunInput<'_>,
    stats: &stats::TransformStats,
    output: target::ResolvedTransformOutput,
) -> Result<Pass2Outcome, TetError> {
    let logical_len = input.plan.logical_element_count;
    let mut warnings = warnings::TransformWarnings::default();
    match (input.dtype, output) {
        (ElementDtype::F32, target::ResolvedTransformOutput::Ram) => {
            let (values, bytes) = apply::transform_read_plan_f32_le_ram(
                input.mmap,
                input.plan,
                stats,
                &mut warnings,
            )?;
            let (preview, truncated) = capped_preview(&values, input.max_preview, logical_len);
            Ok(transform_ram_outcome(
                DecodePreviewBundle::f32_preview(preview, truncated),
                bytes,
                warnings,
            ))
        }
        (ElementDtype::F64, target::ResolvedTransformOutput::Ram) => {
            let (values, bytes) = apply::transform_read_plan_f64_le_ram(
                input.mmap,
                input.plan,
                stats,
                &mut warnings,
            )?;
            let (preview, truncated) = capped_preview(&values, input.max_preview, logical_len);
            Ok(transform_ram_outcome(
                DecodePreviewBundle::f64_preview(preview, truncated),
                bytes,
                warnings,
            ))
        }
        (ElementDtype::F32 | ElementDtype::F64, target::ResolvedTransformOutput::Spill(path)) => {
            spill_pass2(input, stats, &path)
        }
        (ElementDtype::F32, target::ResolvedTransformOutput::Sidecar(paths)) => {
            sidecar_pass2(input, stats, &paths, ElementDtype::F32)
        }
        (ElementDtype::F64, target::ResolvedTransformOutput::Sidecar(paths)) => {
            sidecar_pass2(input, stats, &paths, ElementDtype::F64)
        }
        _ => Err(TetError::Validation(ERR_TRANSFORM_FLOAT.into())),
    }
}

fn spill_pass2(
    input: &TransformRunInput<'_>,
    stats: &stats::TransformStats,
    path: &Path,
) -> Result<Pass2Outcome, TetError> {
    let mut warnings = warnings::TransformWarnings::default();
    let pass2_bytes = match input.dtype {
        ElementDtype::F32 => {
            apply::transform_spill_f32_le(input.mmap, input.plan, path, stats, &mut warnings)?
        }
        ElementDtype::F64 => {
            apply::transform_spill_f64_le(input.mmap, input.plan, path, stats, &mut warnings)?
        }
        _ => {
            return Err(TetError::Validation(ERR_TRANSFORM_FLOAT.into()));
        }
    };
    let spill_bytes = input
        .budget
        .logical_element_bytes(input.dtype, input.plan.logical_element_count)?;
    let bundle = dispatch::spill_export_preview(
        path,
        input.plan.logical_element_count,
        input.max_preview,
        input.dtype,
    )?;
    Ok(Pass2Outcome {
        strategy: MemoryStrategy::TransformSpill,
        spill_path: Some(path.display().to_string()),
        spill_bytes: Some(spill_bytes),
        bundle,
        pass2_bytes,
        warnings,
    })
}

fn sidecar_pass2(
    input: &TransformRunInput<'_>,
    stats: &stats::TransformStats,
    paths: &sidecar::SidecarPaths,
    dtype: ElementDtype,
) -> Result<Pass2Outcome, TetError> {
    let method = transform_method(input.op)?;
    let ctx = SidecarContext {
        tet_path: input.tet_path.ok_or_else(|| {
            TetError::Validation("sidecar write requires a source `.tet` path (`--tet`)".into())
        })?,
        source_dataset: input.source_dataset,
        method,
    };
    let mut warnings = warnings::TransformWarnings::default();
    let logical_len = input.plan.logical_element_count;
    let (payload, pass2_bytes, bundle) = match dtype {
        ElementDtype::F32 => {
            let (values, bytes) = apply::transform_read_plan_f32_le_ram(
                input.mmap,
                input.plan,
                stats,
                &mut warnings,
            )?;
            let (preview, truncated) = capped_preview(&values, input.max_preview, logical_len);
            (
                f32_vec_to_bytes(&values),
                bytes,
                DecodePreviewBundle::f32_preview(preview, truncated),
            )
        }
        ElementDtype::F64 => {
            let (values, bytes) = apply::transform_read_plan_f64_le_ram(
                input.mmap,
                input.plan,
                stats,
                &mut warnings,
            )?;
            let (preview, truncated) = capped_preview(&values, input.max_preview, logical_len);
            (
                f64_vec_to_bytes(&values),
                bytes,
                DecodePreviewBundle::f64_preview(preview, truncated),
            )
        }
        _ => return Err(TetError::Validation(ERR_TRANSFORM_FLOAT.into())),
    };
    sidecar::write_and_publish_sidecar(input.mmap, paths, ctx, input.plan, dtype, &payload)?;
    let spill_bytes = input
        .budget
        .logical_element_bytes(dtype, input.plan.logical_element_count)?;
    Ok(Pass2Outcome {
        strategy: MemoryStrategy::TransformSidecar,
        spill_path: Some(paths.dest.display().to_string()),
        spill_bytes: Some(spill_bytes),
        bundle,
        pass2_bytes,
        warnings,
    })
}

fn f32_vec_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn f64_vec_to_bytes(values: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn transform_method(op: &Operation) -> Result<TransformMethod, TetError> {
    match op {
        Operation::Transform { method, .. } => Ok(*method),
        _ => Err(TetError::Validation(
            "transform_method requires a transform operation".into(),
        )),
    }
}

fn sidecar_context<'a>(
    input: &'a TransformRunInput<'a>,
) -> Result<Option<SidecarContext<'a>>, TetError> {
    let Some(write) = input.write else {
        return Ok(None);
    };
    if write.target != WriteTarget::Sidecar {
        return Ok(None);
    }
    let tet_path = input.tet_path.ok_or_else(|| {
        TetError::Validation("sidecar write requires a source `.tet` path (`--tet`)".into())
    })?;
    Ok(Some(SidecarContext {
        tet_path,
        source_dataset: input.source_dataset,
        method: transform_method(input.op)?,
    }))
}

/// Full dense RAM buffer from transform pass-2 (no preview cap).
#[derive(Debug, Clone)]
pub enum TransformDenseBuffer {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// Decode and transform the full logical selection into an in-memory buffer (`write: ram` only).
///
/// # Errors
///
/// Same as [`run_transform`], plus validation when `write` is not RAM or the selection exceeds
/// [`ExecutionBudget`].
pub(crate) fn materialize_transform_dense_ram(
    input: &TransformRunInput<'_>,
) -> Result<TransformDenseBuffer, TetError> {
    let sidecar = sidecar_context(input)?;
    let (output, _) = target::resolve_transform_output(
        input.write,
        &input.budget,
        input.plan,
        input.dtype,
        input.spill_allowlist,
        sidecar,
    )?;
    if !matches!(output, target::ResolvedTransformOutput::Ram) {
        return Err(TetError::Validation(
            "materialize_transform_dense_ram requires `write`: `ram`".into(),
        ));
    }
    let (stats, _, _) = stats::collect_transform_stats(
        input.mmap,
        input.plan,
        input.op,
        input.dtype,
        &input.fold_policy,
        input.tet_path,
    )?;
    let mut warnings = warnings::TransformWarnings::default();
    match input.dtype {
        ElementDtype::F32 => {
            let (values, _) = apply::transform_read_plan_f32_le_ram(
                input.mmap,
                input.plan,
                &stats,
                &mut warnings,
            )?;
            Ok(TransformDenseBuffer::F32(values))
        }
        ElementDtype::F64 => {
            let (values, _) = apply::transform_read_plan_f64_le_ram(
                input.mmap,
                input.plan,
                &stats,
                &mut warnings,
            )?;
            Ok(TransformDenseBuffer::F64(values))
        }
        _ => Err(TetError::Validation(ERR_TRANSFORM_FLOAT.into())),
    }
}

/// Run pass-1 stats collection and pass-2 transform apply.
///
/// # Errors
///
/// Propagates fold, budget, spill-path, and materialize failures.
pub(crate) fn run_transform(input: &TransformRunInput<'_>) -> Result<TransformOutcome, TetError> {
    let sidecar = sidecar_context(input)?;
    let (output, _) = target::resolve_transform_output(
        input.write,
        &input.budget,
        input.plan,
        input.dtype,
        input.spill_allowlist,
        sidecar,
    )?;
    let (stats, mut operation, pass1_bytes) = stats::collect_transform_stats(
        input.mmap,
        input.plan,
        input.op,
        input.dtype,
        &input.fold_policy,
        input.tet_path,
    )?;
    let pass2 = run_pass2(input, &stats, output)?;
    let total_bytes_read_from_disk = pass1_bytes
        .checked_add(pass2.pass2_bytes)
        .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    operation.element_count = Some(input.plan.logical_element_count);
    if pass2.warnings.div_by_zero_count > 0 {
        operation.transform_div_by_zero_count = Some(pass2.warnings.div_by_zero_count);
        operation.transform_div_by_zero_indices = Some(pass2.warnings.div_by_zero_indices);
    }
    Ok(TransformOutcome {
        total_bytes_read_from_disk,
        strategy: pass2.strategy,
        spill_path: pass2.spill_path,
        spill_bytes: pass2.spill_bytes,
        bundle: pass2.bundle,
        operation,
    })
}

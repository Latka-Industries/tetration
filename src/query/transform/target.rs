//! Resolve transform `write` hints to an in-memory buffer or spill file path.

use std::path::{Path, PathBuf};

use crate::query::engine::{budget::ExecutionBudget, spill_policy::SpillPathAllowlist};
use crate::query::types::{ReadPlan, TetError, WriteHints, WriteTarget};
use crate::utils::dtype::ElementDtype;

use super::sidecar::{self, SidecarContext, SidecarPaths};

/// Where pass 2 should materialize the transformed logical selection.
pub(crate) enum ResolvedTransformOutput {
    Ram,
    Spill(PathBuf),
    Sidecar(SidecarPaths),
}

/// Map [`WriteHints`] and the memory budget to RAM or a validated spill path.
///
/// # Errors
///
/// Returns [`TetError::Validation`] for missing sidecar context, budget overflow with
/// `write: ram`, or spill path allowlist failures.
pub(crate) fn resolve_transform_output(
    write: Option<&WriteHints>,
    budget: &ExecutionBudget,
    plan: &ReadPlan,
    dtype: ElementDtype,
    allowlist: &SpillPathAllowlist,
    sidecar: Option<SidecarContext<'_>>,
) -> Result<(ResolvedTransformOutput, WriteTarget), TetError> {
    let target = write.map_or(WriteTarget::Switch, |w| w.target);
    let path_hint = write.and_then(|w| w.path.as_deref());

    if target == WriteTarget::Sidecar {
        let ctx = sidecar.ok_or_else(|| {
            TetError::Validation("sidecar write requires a source `.tet` path (`--tet`)".into())
        })?;
        let paths = sidecar::resolve_sidecar_paths(write, allowlist, ctx)?;
        return Ok((
            ResolvedTransformOutput::Sidecar(paths),
            WriteTarget::Sidecar,
        ));
    }

    let exceeds = budget.full_tensor_exceeds_budget(plan, dtype)?;

    match target {
        WriteTarget::Ram => {
            if exceeds {
                return Err(TetError::Validation(format!(
                    "logical selection ({} elements, {} bytes) exceeds memory_budget_bytes ({}); \
                     use `write`: `switch` or `spill`, or raise execution.memory_budget_bytes",
                    plan.logical_element_count,
                    budget.logical_element_bytes(dtype, plan.logical_element_count)?,
                    budget.memory_budget_bytes
                )));
            }
            Ok((ResolvedTransformOutput::Ram, WriteTarget::Ram))
        }
        WriteTarget::Spill => {
            let path = resolve_spill_path(path_hint, allowlist)?;
            Ok((ResolvedTransformOutput::Spill(path), WriteTarget::Spill))
        }
        WriteTarget::Switch => {
            if exceeds {
                let path = resolve_spill_path(path_hint, allowlist)?;
                Ok((ResolvedTransformOutput::Spill(path), WriteTarget::Switch))
            } else {
                Ok((ResolvedTransformOutput::Ram, WriteTarget::Switch))
            }
        }
        WriteTarget::Sidecar => unreachable!("handled above"),
    }
}

fn resolve_spill_path(
    path_hint: Option<&str>,
    allowlist: &SpillPathAllowlist,
) -> Result<PathBuf, TetError> {
    match path_hint {
        Some(handle) => allowlist.validate(Path::new(handle)),
        None => allowlist.allocate_temp_spill_path(),
    }
}

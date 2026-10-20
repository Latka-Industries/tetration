//! Minimal distributed spill / export policy (THI-577).

use crate::query::types::{TetError, WriteHints, WriteTarget};

/// Cluster spill policy for 0.5.0: keep it tiny — refuse distributed spill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistributedSpillPolicy {
    /// Workers keep fold state in RAM; coordinator finishes scalars.
    /// `write: spill|sidecar|switch` on a distributed job is refused.
    #[default]
    RamOnlyRefuseSpill,
}

/// Refuse distributed execute when the query asks for non-RAM write targets.
///
/// Scalar distributed folds do not materialize; spill/sidecar/switch are for transform
/// output routing and are not supported across workers in 0.5.0.
///
/// # Errors
///
/// Returns [`TetError::Validation`] with a stable message when spill/sidecar/switch is set.
pub fn refuse_distributed_spill(write: Option<&WriteHints>) -> Result<(), TetError> {
    let Some(w) = write else {
        return Ok(());
    };
    match w.target {
        WriteTarget::Ram => Ok(()),
        WriteTarget::Switch | WriteTarget::Spill | WriteTarget::Sidecar => Err(TetError::Validation(
            "distributed fold does not support write spill/sidecar/switch yet; use write ram (or omit write) and keep partials in memory (THI-577)"
                .into(),
        )),
    }
}

//! Distributed spill / export policy (THI-577; locked refuse in 0.7.1 — THI-616).

use crate::query::types::{TetError, WriteHints, WriteTarget};

/// Cluster spill policy: keep it tiny — refuse distributed spill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistributedSpillPolicy {
    /// Workers keep fold state in RAM; coordinator finishes scalars.
    /// `write: spill|sidecar|switch` on a distributed job is refused.
    ///
    /// Locked for 0.7.1–0.7.3 ([`docs/distributed_boundary.md`](../../../../docs/distributed_boundary.md)).
    #[default]
    RamOnlyRefuseSpill,
}

/// Refuse distributed execute when the query asks for non-RAM write targets.
///
/// Scalar distributed folds do not materialize; spill/sidecar/switch are for transform
/// output routing and are not supported across workers (boundary RFC / THI-616).
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
            "distributed fold refuses write spill/sidecar/switch; use write ram (or omit write) and keep partials in memory (THI-616 / distributed_boundary.md)"
                .into(),
        )),
    }
}

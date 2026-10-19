//! Store equivalent of local `tet gc` (decision D + J).
//!
//! [`prune_store`] lists leftover `staging/<session>` objects and sealed generations
//! older than `CURRENT` / `CURRENT-1`. Dry-run is the default. Apply deletes those
//! prefixes only. It never writes `CURRENT`.
//!
//! Open reader pins and a live `publishing/<G>` claim are withheld, not deleted.
//! A claim is live when `publishing/<G>/` exists and its heartbeat is not stale
//! ([`DEFAULT_PUBLISH_LOCK_STALE`](crate::catalog::DEFAULT_PUBLISH_LOCK_STALE)).
//! Any live claim withholds every staging session, because seal still reads staged
//! parts. Stale claims and `publishing/` itself are left for abort / repair.

use std::collections::BTreeSet;

use crate::catalog::CatalogError;
use crate::catalog::package::{
    self, DEFAULT_PUBLISH_LOCK_STALE, TetPackagePin, format_generation_id,
};

use super::pin::PinnedPackedTet;
use super::publish_lock::{publish_lock_exists, publish_lock_is_stale};
use super::{PackageStore, ReadAt};

/// Options for [`prune_store`].
///
/// `apply` defaults to `false` (report only), matching [`crate::catalog::GcOptions`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PruneOptions {
    /// When `false` (default), only report. When `true`, delete planned prefixes.
    pub apply: bool,
    /// Generations held by open [`TetPackagePin`] or [`PinnedPackedTet`] handles.
    ///
    /// Their `packed.tet` objects are never deleted, including when the id is older
    /// than `CURRENT-1`. [`crate::catalog::CachingStore::prune`] adds its own pin.
    pub pinned_generations: Vec<u64>,
}

impl PruneOptions {
    /// Delete planned prefixes. Dry-run is the default.
    #[must_use]
    pub fn apply(mut self) -> Self {
        self.apply = true;
        self
    }

    /// Protect these generation ids (open pins).
    #[must_use]
    pub fn with_pinned_generations(mut self, generations: impl IntoIterator<Item = u64>) -> Self {
        self.pinned_generations.extend(generations);
        self
    }

    /// Protect generations named by open [`TetPackagePin`] handles.
    #[must_use]
    pub fn with_package_pins(self, pins: &[TetPackagePin]) -> Self {
        self.with_pinned_generations(pins.iter().map(TetPackagePin::generation))
    }

    /// Protect generations named by open [`PinnedPackedTet`] handles.
    #[must_use]
    pub fn with_store_pins(self, pins: &[PinnedPackedTet]) -> Self {
        self.with_pinned_generations(pins.iter().map(|pin| pin.generation))
    }
}

/// What [`prune_store`] would remove, or did remove when [`PruneOptions::apply`] is set.
///
/// `keys` are the objects [`ReadAt::list_prefix`] saw under the planned prefixes.
/// Apply deletes each whole prefix (directory markers included). Refused targets are
/// not listed in `keys` and are not deleted.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PruneReport {
    /// Staging session prefixes (`staging/<session>`).
    pub staging_sessions: Vec<String>,
    /// Sealed generation ids older than `CURRENT` and `CURRENT-1`.
    pub generations: Vec<u64>,
    /// Object keys under [`Self::staging_sessions`] and [`Self::generations`].
    pub keys: Vec<String>,
    /// Candidates withheld because a pin or a live publish claim still needs them.
    pub refused: Vec<PruneRefusal>,
}

impl PruneReport {
    /// `true` when nothing would be removed.
    ///
    /// Refusals are not removals: a report can be empty and still list [`Self::refused`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.staging_sessions.is_empty() && self.generations.is_empty()
    }
}

/// One withheld prune candidate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PruneRefusal {
    /// Staging session or sealed generation that was not planned for deletion.
    pub target: PruneTarget,
    /// Why it was withheld.
    pub reason: PruneRefuseReason,
}

/// A staging session or sealed generation prune considered, then withheld.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PruneTarget {
    /// `generations/<G>` (zero-padded id).
    Generation(u64),
    /// `staging/<session>`.
    StagingSession(String),
}

/// Why [`prune_store`] left a candidate in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PruneRefuseReason {
    /// An open pin still names this generation's `packed.tet`.
    OpenPin,
    /// A live `publishing/<G>` claim still needs this prefix.
    ActivePublishClaim,
}

/// Plan store prune, and delete when [`PruneOptions::apply`] is set.
///
/// Retention matches local [`crate::catalog::TetPackage::gc`]: keep `CURRENT` and
/// `CURRENT-1`, and keep any sealed generation that is not older than `CURRENT`.
/// Does not write `CURRENT`.
///
/// Pass open pins via [`PruneOptions::pinned_generations`], [`PruneOptions::with_package_pins`],
/// or [`PruneOptions::with_store_pins`]. The store has no process-wide pin registry.
///
/// # Errors
///
/// [`CatalogError::NoCurrentGeneration`] when `CURRENT` is absent, or store I/O.
/// Apply also errors, without deleting, when a planned prefix is `CURRENT`,
/// `CURRENT-1`, pinned, or held by a live publish claim.
#[must_use = "the report lists what was kept and what would be deleted"]
pub fn prune_store(
    store: &impl PackageStore,
    opts: &PruneOptions,
) -> Result<PruneReport, CatalogError> {
    let report = plan_prune(store, opts)?;
    if !opts.apply || report.is_empty() {
        return Ok(report);
    }
    apply_prune(store, opts, &report)?;
    Ok(report)
}

fn plan_prune(store: &impl PackageStore, opts: &PruneOptions) -> Result<PruneReport, CatalogError> {
    let current = store
        .current_generation()?
        .ok_or(CatalogError::NoCurrentGeneration)?;
    let pins: BTreeSet<u64> = opts.pinned_generations.iter().copied().collect();
    let active = active_publish_generations(store)?;

    let (generations, mut refused) = classify_generations(store, current, &pins, &active)?;
    let staging_sessions = classify_staging(store, &active, &mut refused)?;
    refused.sort();

    let mut keys = Vec::new();
    for generation in &generations {
        keys.extend(objects_under(store, &generation_prefix(*generation))?);
    }
    for session in &staging_sessions {
        keys.extend(objects_under(store, session)?);
    }
    keys.sort();
    keys.dedup();

    Ok(PruneReport {
        staging_sessions,
        generations,
        keys,
        refused,
    })
}

fn classify_generations(
    store: &impl ReadAt,
    current: u64,
    pins: &BTreeSet<u64>,
    active: &BTreeSet<u64>,
) -> Result<(Vec<u64>, Vec<PruneRefusal>), CatalogError> {
    let mut generations = Vec::new();
    let mut refused = Vec::new();
    for generation in sealed_generations(store)? {
        if !is_below_retention(generation, current) {
            continue;
        }
        if pins.contains(&generation) {
            refused.push(PruneRefusal {
                target: PruneTarget::Generation(generation),
                reason: PruneRefuseReason::OpenPin,
            });
            continue;
        }
        if active.contains(&generation) {
            refused.push(PruneRefusal {
                target: PruneTarget::Generation(generation),
                reason: PruneRefuseReason::ActivePublishClaim,
            });
            continue;
        }
        generations.push(generation);
    }
    Ok((generations, refused))
}

fn classify_staging(
    store: &impl ReadAt,
    active: &BTreeSet<u64>,
    refused: &mut Vec<PruneRefusal>,
) -> Result<Vec<String>, CatalogError> {
    let mut staging_sessions = Vec::new();
    for session in staging_session_prefixes(store)? {
        if active.is_empty() {
            staging_sessions.push(session);
            continue;
        }
        refused.push(PruneRefusal {
            target: PruneTarget::StagingSession(session),
            reason: PruneRefuseReason::ActivePublishClaim,
        });
    }
    Ok(staging_sessions)
}

fn apply_prune(
    store: &impl PackageStore,
    opts: &PruneOptions,
    report: &PruneReport,
) -> Result<(), CatalogError> {
    // Re-check retention against a fresh `CURRENT` read. This function does not
    // call `publish_generation`.
    let current = store
        .current_generation()?
        .ok_or(CatalogError::NoCurrentGeneration)?;
    let pins: BTreeSet<u64> = opts.pinned_generations.iter().copied().collect();
    let active = active_publish_generations(store)?;
    if !report.staging_sessions.is_empty() && !active.is_empty() {
        return Err(CatalogError::Package(
            "refusing to delete staging held by an active publish claim",
        ));
    }
    for &generation in &report.generations {
        refuse_protected_generation(generation, current, &pins, &active)?;
    }
    for &generation in &report.generations {
        let prefix = generation_prefix(generation);
        store.remove_prefix(&prefix)?;
    }
    for session in &report.staging_sessions {
        ensure_staging_prefix(session)?;
        store.remove_prefix(session)?;
    }
    Ok(())
}

fn refuse_protected_generation(
    generation: u64,
    current: u64,
    pins: &BTreeSet<u64>,
    active: &BTreeSet<u64>,
) -> Result<(), CatalogError> {
    if generation == current || (current > 1 && generation == current - 1) {
        return Err(CatalogError::Package(
            "refusing to delete CURRENT or CURRENT-1",
        ));
    }
    if pins.contains(&generation) {
        return Err(CatalogError::Package(
            "refusing to delete a pinned generation",
        ));
    }
    if active.contains(&generation) {
        return Err(CatalogError::Package(
            "refusing to delete a generation with an active publish claim",
        ));
    }
    let prefix = generation_prefix(generation);
    if !prefix.starts_with("generations/") {
        return Err(CatalogError::Package(
            "refusing to delete an unsafe generation prefix",
        ));
    }
    Ok(())
}

fn is_below_retention(generation: u64, current: u64) -> bool {
    if generation == current {
        return false;
    }
    if current > 1 && generation == current - 1 {
        return false;
    }
    generation < current
}

fn generation_prefix(generation: u64) -> String {
    format!("generations/{}", format_generation_id(generation))
}

fn ensure_staging_prefix(prefix: &str) -> Result<(), CatalogError> {
    let Some(session) = prefix.strip_prefix("staging/") else {
        return Err(CatalogError::Package("refusing to delete staging root"));
    };
    if session.is_empty() || session.contains('/') || !package::is_safe_path_segment(session) {
        return Err(CatalogError::Package(
            "refusing to delete an unsafe staging prefix",
        ));
    }
    Ok(())
}

fn sealed_generations(store: &impl ReadAt) -> Result<Vec<u64>, CatalogError> {
    let keys = store.list_prefix("generations/")?;
    let mut out = Vec::new();
    for key in keys {
        let Some(generation) = generation_from_packed_key(&key) else {
            continue;
        };
        out.push(generation);
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

fn generation_from_packed_key(key: &str) -> Option<u64> {
    let rest = key.strip_prefix("generations/")?;
    let (segment, tail) = rest.split_once('/')?;
    if tail != "packed.tet" {
        return None;
    }
    parse_generation_segment(segment)
}

fn staging_session_prefixes(store: &impl ReadAt) -> Result<Vec<String>, CatalogError> {
    let keys = store.list_prefix("staging/")?;
    let mut sessions = BTreeSet::new();
    for key in keys {
        let Some(session) = staging_session_of(&key) else {
            continue;
        };
        sessions.insert(session);
    }
    Ok(sessions.into_iter().collect())
}

fn staging_session_of(key: &str) -> Option<String> {
    let rest = key.strip_prefix("staging/")?;
    let (segment, _) = rest.split_once('/')?;
    if !package::is_safe_path_segment(segment) {
        return None;
    }
    Some(format!("staging/{segment}"))
}

fn active_publish_generations(store: &impl ReadAt) -> Result<BTreeSet<u64>, CatalogError> {
    let keys = store.list_prefix("publishing/")?;
    let mut candidates = BTreeSet::new();
    for key in keys {
        if let Some(generation) = generation_from_publishing_key(&key) {
            candidates.insert(generation);
        }
    }
    let mut active = BTreeSet::new();
    for generation in candidates {
        if !publish_lock_exists(store, generation)? {
            continue;
        }
        if publish_lock_is_stale(store, generation, DEFAULT_PUBLISH_LOCK_STALE)? {
            continue;
        }
        active.insert(generation);
    }
    Ok(active)
}

fn generation_from_publishing_key(key: &str) -> Option<u64> {
    let rest = key.strip_prefix("publishing/")?;
    let segment = rest.split('/').next().unwrap_or("");
    parse_generation_segment(segment)
}

fn parse_generation_segment(segment: &str) -> Option<u64> {
    if segment.contains("quarantine") {
        return None;
    }
    let generation = segment.parse::<u64>().ok()?;
    if format_generation_id(generation) != segment {
        return None;
    }
    Some(generation)
}

fn objects_under(store: &impl ReadAt, prefix: &str) -> Result<Vec<String>, CatalogError> {
    let mut keys = store.list_prefix(prefix)?;
    let nested = format!("{prefix}/");
    keys.retain(|key| key == prefix || key.starts_with(&nested));
    Ok(keys)
}

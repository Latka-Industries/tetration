//! Cross-host publish claim on `publishing/<G>` for [`PackageStore`](super::PackageStore).
//!
//! Mirrors [`TetPackage`](crate::catalog::package::TetPackage) decision **G**: exclusive create,
//! `HEARTBEAT`, stale window, quarantine, and takeover only with explicit `force`.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::catalog::CatalogError;
use crate::catalog::package::{self, DEFAULT_PUBLISH_LOCK_STALE};

use super::{ReadAt, WritePart};

const HEARTBEAT_FILE: &str = "HEARTBEAT";

/// Options for store-backed publish (writer protocol decision **G**).
///
/// Default: no force takeover; stale age is [`DEFAULT_PUBLISH_LOCK_STALE`](crate::catalog::DEFAULT_PUBLISH_LOCK_STALE) (1h).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublishOptions {
    /// When `true`, quarantine a **stale** lock and retry the exclusive claim.
    /// A live lock still returns [`CatalogError::PublishLockHeld`].
    pub force: bool,
    /// Age past which a lock with a dead heartbeat may be quarantined under `force`.
    pub stale_after: Duration,
}

impl Default for PublishOptions {
    fn default() -> Self {
        Self {
            force: false,
            stale_after: DEFAULT_PUBLISH_LOCK_STALE,
        }
    }
}

/// Package-relative key for the publish lock directory (`publishing/<G>`).
#[must_use]
pub fn publishing_lock_key(generation: u64) -> String {
    format!("publishing/{}", package::format_generation_id(generation))
}

/// Package-relative key for `publishing/<G>/HEARTBEAT`.
#[must_use]
pub fn heartbeat_key(generation: u64) -> String {
    format!("{}/{HEARTBEAT_FILE}", publishing_lock_key(generation))
}

/// Write / refresh `publishing/<G>/HEARTBEAT` with the current unix time.
///
/// # Errors
///
/// Store I/O.
pub fn touch_publish_heartbeat(
    store: &impl WritePart,
    generation: u64,
) -> Result<(), CatalogError> {
    let body = format!("{}\n", unix_now());
    store.write_part(&heartbeat_key(generation), body.as_bytes())
}

/// `true` when any object exists under `publishing/<G>/` (including the dir marker).
///
/// # Errors
///
/// Store I/O.
pub fn publish_lock_exists(store: &impl ReadAt, generation: u64) -> Result<bool, CatalogError> {
    let lock = publishing_lock_key(generation);
    let keys = store.list_prefix(&lock)?;
    Ok(keys
        .iter()
        .any(|k| k == &lock || k.starts_with(&format!("{lock}/"))))
}

/// `true` when the lock exists and its heartbeat age exceeds `max_age`.
///
/// Age is taken from the unix timestamp in `HEARTBEAT`. Missing heartbeat with a live lock
/// prefix is treated as **not stale** (do not steal).
///
/// # Errors
///
/// Store I/O or unreadable heartbeat.
pub fn publish_lock_is_stale(
    store: &impl ReadAt,
    generation: u64,
    max_age: Duration,
) -> Result<bool, CatalogError> {
    if !publish_lock_exists(store, generation)? {
        return Ok(false);
    }
    let hb = heartbeat_key(generation);
    let Some(bytes) = store.read_all(&hb).ok() else {
        return Ok(false);
    };
    let Some(ts) = parse_heartbeat_unix(&bytes) else {
        return Ok(false);
    };
    let modified = UNIX_EPOCH + Duration::from_secs(ts);
    let age = SystemTime::now()
        .duration_since(modified)
        .unwrap_or(Duration::ZERO);
    Ok(age > max_age)
}

/// Quarantine a **stale** publish lock by renaming the prefix aside (decision G).
///
/// # Errors
///
/// [`CatalogError::PublishLockHeld`] when the lock is missing or not stale, or store I/O.
pub fn quarantine_publish_lock(
    store: &(impl ReadAt + WritePart),
    generation: u64,
    max_age: Duration,
) -> Result<String, CatalogError> {
    if !publish_lock_is_stale(store, generation, max_age)? {
        return Err(CatalogError::PublishLockHeld(generation));
    }
    let from = publishing_lock_key(generation);
    let to = format!(
        "publishing/{}.quarantine-{}",
        package::format_generation_id(generation),
        unix_now()
    );
    store.rename_prefix(&from, &to)?;
    Ok(to)
}

/// Exclusive-create `publishing/<G>/` and write a heartbeat.
///
/// With [`PublishOptions::force`], a stale lock is quarantined once and the claim retried.
///
/// # Errors
///
/// [`CatalogError::PublishLockHeld`] on collision with a live (or non-force) lock, or store I/O.
pub fn begin_publish_store(
    store: &(impl ReadAt + WritePart),
    generation: u64,
    options: &PublishOptions,
) -> Result<(), CatalogError> {
    match try_claim_lock(store, generation) {
        Ok(()) => {
            touch_publish_heartbeat(store, generation)?;
            return Ok(());
        }
        Err(CatalogError::PublishLockHeld(_)) if options.force => {}
        Err(err) => return Err(err),
    }

    quarantine_publish_lock(store, generation, options.stale_after)?;
    try_claim_lock(store, generation)?;
    touch_publish_heartbeat(store, generation)?;
    Ok(())
}

fn try_claim_lock(store: &impl WritePart, generation: u64) -> Result<(), CatalogError> {
    let lock = publishing_lock_key(generation);
    match store.create_dir_exclusive(&lock) {
        Ok(()) => Ok(()),
        Err(CatalogError::PackagePathExists(_)) => Err(CatalogError::PublishLockHeld(generation)),
        Err(err) => Err(err),
    }
}

fn parse_heartbeat_unix(bytes: &[u8]) -> Option<u64> {
    let text = std::str::from_utf8(bytes).ok()?;
    text.trim().parse::<u64>().ok()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

//! Package blob store (THI-469): filesystem vs object-store backends behind one protocol.
//!
//! Keys are package-relative POSIX paths (`CURRENT`, `staging/…`, `generations/00000001/packed.tet`).
//! Local `.tet.d` uses [`LocalStore`]; tests and future planners use [`MemoryStore`].
//! S3-compatible backends are 0.3.1 ([THI-499](https://linear.app/thicclatka/issue/THI-499)).

mod local;
mod memory;

use std::fmt;

use super::CatalogError;

pub use local::LocalStore;
pub use memory::MemoryStore;

/// Metadata for one store object (file-like key).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectMeta {
    /// Byte length of the object body.
    pub len: u64,
}

/// Range-read and list surface shared by every package backend.
pub trait ReadAt {
    /// Read `len` bytes starting at `offset` from `key`.
    ///
    /// # Errors
    ///
    /// Missing key, out-of-range offset, or backend I/O.
    fn read_at(&self, key: &str, offset: u64, len: u64) -> Result<Vec<u8>, CatalogError>;

    /// Read the entire object at `key`.
    ///
    /// # Errors
    ///
    /// Missing key or backend I/O.
    fn read_all(&self, key: &str) -> Result<Vec<u8>, CatalogError> {
        let Some(meta) = self.object_meta(key)? else {
            return Err(CatalogError::Package("store object missing"));
        };
        self.read_at(key, 0, meta.len)
    }

    /// Length of `key` when it exists.
    ///
    /// # Errors
    ///
    /// Backend I/O (not “missing”).
    fn object_meta(&self, key: &str) -> Result<Option<ObjectMeta>, CatalogError>;

    /// List object keys under `prefix` (inclusive of keys equal to `prefix` when present).
    ///
    /// Directory markers may or may not appear; callers should treat returned keys as objects.
    ///
    /// # Errors
    ///
    /// Backend I/O.
    fn list_prefix(&self, prefix: &str) -> Result<Vec<String>, CatalogError>;
}

/// Write sealed part / staging blobs (immutable after publish into the key).
pub trait WritePart {
    /// Create an empty directory key exclusively (staging session, publish lock).
    ///
    /// # Errors
    ///
    /// [`CatalogError::PackagePathExists`] when the directory already exists, or I/O.
    fn create_dir_exclusive(&self, key: &str) -> Result<(), CatalogError>;

    /// Atomically publish `bytes` at `key` (temp + rename / single map insert).
    ///
    /// # Errors
    ///
    /// Backend I/O.
    fn write_part(&self, key: &str, bytes: &[u8]) -> Result<(), CatalogError>;

    /// Remove a single object key. Missing key is OK.
    ///
    /// # Errors
    ///
    /// Backend I/O.
    fn remove_object(&self, key: &str) -> Result<(), CatalogError>;

    /// Remove every object (and directory marker) under `prefix`. Missing prefix is OK.
    ///
    /// # Errors
    ///
    /// Backend I/O.
    fn remove_prefix(&self, prefix: &str) -> Result<(), CatalogError>;

    /// Move every object under `from_prefix` to the same relative path under `to_prefix`.
    ///
    /// # Errors
    ///
    /// Missing source, destination collision, or I/O.
    fn rename_prefix(&self, from_prefix: &str, to_prefix: &str) -> Result<(), CatalogError>;
}

/// Flip `CURRENT` after a sealed generation is in place (decision K).
pub trait PublishGeneration {
    /// Set `CURRENT` to `generation` when the prior value matches `expected_current`.
    ///
    /// Local backends may treat the write as POSIX rename-atomic and still check `expected`.
    /// Object stores use conditional put (0.3.1).
    ///
    /// # Errors
    ///
    /// [`CatalogError::InvalidCurrent`] when the CAS precondition fails, or I/O.
    fn publish_generation(
        &self,
        generation: u64,
        expected_current: Option<u64>,
    ) -> Result<(), CatalogError>;

    /// Read the generation id named by `CURRENT`, if any.
    ///
    /// # Errors
    ///
    /// [`CatalogError::InvalidCurrent`] when the pointer cannot be parsed, or I/O.
    fn current_generation(&self) -> Result<Option<u64>, CatalogError>;
}

/// Full package store: read, write parts, publish generation.
pub trait PackageStore: ReadAt + WritePart + PublishGeneration {}

impl<T> PackageStore for T where T: ReadAt + WritePart + PublishGeneration {}

/// Reject empty keys, absolute paths, `..` segments, and Windows separators.
pub(crate) fn validate_store_key(key: &str) -> Result<(), CatalogError> {
    if key.is_empty() {
        return Err(CatalogError::Package("empty store key"));
    }
    if key.starts_with('/') || key.starts_with('\\') {
        return Err(CatalogError::Package("store key must be relative"));
    }
    if key.contains('\\') {
        return Err(CatalogError::Package("store key must use '/' separators"));
    }
    for segment in key.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(CatalogError::Package("invalid store key segment"));
        }
    }
    Ok(())
}

pub(crate) fn parse_current_bytes(bytes: &[u8]) -> Result<u64, CatalogError> {
    let text = std::str::from_utf8(bytes).map_err(|_| CatalogError::InvalidCurrent)?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(CatalogError::InvalidCurrent);
    }
    if trimmed.starts_with('{') {
        #[derive(serde::Deserialize)]
        struct CurrentJson {
            generation: u64,
        }
        let v: CurrentJson =
            serde_json::from_str(trimmed).map_err(|_| CatalogError::InvalidCurrent)?;
        return Ok(v.generation);
    }
    trimmed
        .parse::<u64>()
        .map_err(|_| CatalogError::InvalidCurrent)
}

impl fmt::Debug for dyn PackageStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PackageStore")
    }
}

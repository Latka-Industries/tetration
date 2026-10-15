//! `.tet.d` package skeleton (writer protocol v2).
//!
//! Directory roles: `PROTOCOL`, `CURRENT`, `staging/`, `generations/`, `publishing/`.
//! Sealed readers open only [`TetPackage::packed_tet_path`] (layout v1).
//!
//! Spec: [`docs/writer_protocol_v2.md`](../../docs/writer_protocol_v2.md).

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::CatalogError;
use super::session::TetFile;
use crate::utils::fs_device;

/// Package protocol version locked with writer protocol v2 (decision F).
pub const PACKAGE_PROTOCOL_VERSION: u32 = 2;

/// Default stale publish-lock age before verify/repair may quarantine (decision G).
pub const DEFAULT_PUBLISH_LOCK_STALE: Duration = Duration::from_hours(1);

const PROTOCOL_FILE: &str = "PROTOCOL";
const CURRENT_FILE: &str = "CURRENT";
const CURRENT_TMP: &str = "CURRENT.tmp";
const GENERATIONS_DIR: &str = "generations";
const STAGING_DIR: &str = "staging";
const PUBLISHING_DIR: &str = "publishing";
const PACKED_TET: &str = "packed.tet";
const MANIFEST_FILE: &str = "MANIFEST.json";
const HEARTBEAT_FILE: &str = "HEARTBEAT";

/// Root `PROTOCOL` file contents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageProtocol {
    /// Must equal [`PACKAGE_PROTOCOL_VERSION`] for this crate build.
    pub protocol_version: u32,
    /// Unix seconds when the package was created, if recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_unix: Option<u64>,
}

/// Sealed-generation `MANIFEST.json` (provenance + packed image identity).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationManifest {
    /// Generation id (`G`); never reused within a package.
    pub generation: u64,
    /// Echo of [`PACKAGE_PROTOCOL_VERSION`] at seal time.
    pub protocol_version: u32,
    /// Explicit part-id list frozen at publish (decision C).
    pub part_ids: Vec<String>,
    /// Byte length of `packed.tet` at seal time.
    pub packed_byte_len: u64,
    /// Hex SHA-256 of `packed.tet` at seal time (required for new seals; verified by package verify).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_sha256: Option<String>,
}

/// Open handle to a `.tet.d` package root.
///
/// Cheap to clone (path + parsed [`PackageProtocol`]); does not hold file locks.
#[derive(Debug, Clone)]
pub struct TetPackage {
    root: PathBuf,
    protocol: PackageProtocol,
}

impl TetPackage {
    /// Package root directory (the `.tet.d` path).
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Parsed `PROTOCOL` document from open/create.
    #[must_use]
    pub fn protocol(&self) -> &PackageProtocol {
        &self.protocol
    }

    /// Create a new empty package at `path` (must not already exist).
    ///
    /// Writes `PROTOCOL` and empty `staging/`, `generations/`, and `publishing/` dirs.
    /// `CURRENT` is absent until the first [`Self::seal_generation`].
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::PackagePathExists`] when `path` exists, or I/O errors while
    /// creating the tree / writing `PROTOCOL`.
    pub fn create(path: impl Into<PathBuf>) -> Result<Self, CatalogError> {
        let root = path.into();
        if root.exists() {
            return Err(CatalogError::PackagePathExists(root));
        }
        fs::create_dir_all(&root)?;
        for dir in [GENERATIONS_DIR, STAGING_DIR, PUBLISHING_DIR] {
            fs::create_dir_all(root.join(dir))?;
        }

        let protocol = PackageProtocol {
            protocol_version: PACKAGE_PROTOCOL_VERSION,
            created_unix: Some(unix_now()),
        };
        write_json_atomic(&root.join(PROTOCOL_FILE), &protocol)?;
        Ok(Self { root, protocol })
    }

    /// Open an existing package and validate `PROTOCOL`.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::NotAPackage`] when `PROTOCOL` is missing,
    /// [`CatalogError::UnsupportedProtocolVersion`] / [`CatalogError::InvalidPackageProtocol`]
    /// when the document is wrong, or I/O errors.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, CatalogError> {
        let root = path.into();
        let protocol_path = root.join(PROTOCOL_FILE);
        if !protocol_path.is_file() {
            return Err(CatalogError::NotAPackage(root));
        }
        let protocol: PackageProtocol = read_json(&protocol_path)?;
        if protocol.protocol_version != PACKAGE_PROTOCOL_VERSION {
            return Err(CatalogError::UnsupportedProtocolVersion(
                protocol.protocol_version,
            ));
        }
        Ok(Self { root, protocol })
    }

    /// Read `CURRENT` generation id, if any sealed generation has been published.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::InvalidCurrent`] when the pointer cannot be parsed, or I/O errors.
    pub fn current(&self) -> Result<Option<u64>, CatalogError> {
        let path = self.root.join(CURRENT_FILE);
        if !path.is_file() {
            return Ok(None);
        }
        let text = fs::read_to_string(&path)?;
        parse_current(&text).map(Some)
    }

    /// Atomically set `CURRENT` via temp file + same-filesystem rename (decision K).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::GenerationNotSealed`] when the generation directory exists but
    /// lacks `packed.tet`, or I/O / cross-volume publish failures.
    pub fn set_current(&self, generation: u64) -> Result<(), CatalogError> {
        if self.generation_dir(generation).exists() && !self.packed_tet_path(generation).is_file() {
            return Err(CatalogError::GenerationNotSealed(generation));
        }
        let tmp = self.root.join(CURRENT_TMP);
        let dest = self.root.join(CURRENT_FILE);
        write_text_atomic(&tmp, &format!("{generation}\n"))?;
        fs_device::publish_file(&tmp, &dest)?;
        Ok(())
    }

    /// `staging/` — shard writers only; never reader-visible.
    #[must_use]
    pub fn staging_dir(&self) -> PathBuf {
        self.root.join(STAGING_DIR)
    }

    /// `publishing/` — committer scratch / publish locks.
    #[must_use]
    pub fn publishing_dir(&self) -> PathBuf {
        self.root.join(PUBLISHING_DIR)
    }

    /// `generations/` — sealed generation trees.
    #[must_use]
    pub fn generations_dir(&self) -> PathBuf {
        self.root.join(GENERATIONS_DIR)
    }

    /// `generations/<G>/` for a generation id.
    #[must_use]
    pub fn generation_dir(&self, generation: u64) -> PathBuf {
        self.generations_dir()
            .join(format_generation_id(generation))
    }

    /// Path to the sealed layout-v1 image for `generation` (decision A).
    #[must_use]
    pub fn packed_tet_path(&self, generation: u64) -> PathBuf {
        self.generation_dir(generation).join(PACKED_TET)
    }

    /// Path to `MANIFEST.json` for a sealed generation.
    #[must_use]
    pub fn manifest_path(&self, generation: u64) -> PathBuf {
        self.generation_dir(generation).join(MANIFEST_FILE)
    }

    /// Exclusive-create `publishing/<G>/` and write a heartbeat (decision G).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::PublishLockHeld`] if the directory already exists, or I/O errors.
    pub fn begin_publish(&self, generation: u64) -> Result<PathBuf, CatalogError> {
        let dir = self.publishing_scratch(generation);
        match fs::create_dir(&dir) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                return Err(CatalogError::PublishLockHeld(generation));
            }
            Err(e) => return Err(e.into()),
        }
        touch_heartbeat(&dir)?;
        Ok(dir)
    }

    /// Remove `publishing/<G>/` after abandon or failed publish. Missing dir is OK.
    ///
    /// # Errors
    ///
    /// I/O while removing the directory tree.
    pub fn abort_publish(&self, generation: u64) -> Result<(), CatalogError> {
        let dir = self.publishing_scratch(generation);
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }

    /// `true` when `publishing/<G>/` exists and its heartbeat (or dir mtime) is older than `max_age`.
    ///
    /// # Errors
    ///
    /// I/O reading heartbeat / directory metadata.
    pub fn publish_lock_is_stale(
        &self,
        generation: u64,
        max_age: Duration,
    ) -> Result<bool, CatalogError> {
        let dir = self.publishing_scratch(generation);
        if !dir.is_dir() {
            return Ok(false);
        }
        let hb = dir.join(HEARTBEAT_FILE);
        let meta = if hb.is_file() {
            fs::metadata(&hb)?
        } else {
            fs::metadata(&dir)?
        };
        let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        let age = SystemTime::now()
            .duration_since(modified)
            .unwrap_or(Duration::ZERO);
        Ok(age > max_age)
    }

    /// Quarantine a **stale** publish lock by renaming aside (decision G). Does not steal a live lock.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::PublishLockHeld`] when the lock is missing or not stale, or I/O errors.
    pub fn quarantine_publish_lock(
        &self,
        generation: u64,
        max_age: Duration,
    ) -> Result<PathBuf, CatalogError> {
        if !self.publish_lock_is_stale(generation, max_age)? {
            return Err(CatalogError::PublishLockHeld(generation));
        }
        let dir = self.publishing_scratch(generation);
        let quarantine = self.publishing_dir().join(format!(
            "{}.quarantine-{}",
            format_generation_id(generation),
            unix_now()
        ));
        fs::rename(&dir, &quarantine)?;
        Ok(quarantine)
    }

    /// Seal generation `G`: copy `packed_tet` into the publish lock dir, write `MANIFEST`,
    /// rename into `generations/<G>/`, then flip `CURRENT` (decision I).
    ///
    /// Caller must hold the lock from [`Self::begin_publish`]. On success the lock directory
    /// becomes the sealed generation tree (no longer under `publishing/`).
    ///
    /// # Errors
    ///
    /// Missing lock, missing source file, destination already exists, or I/O errors.
    pub fn seal_generation(
        &self,
        generation: u64,
        packed_tet: &Path,
        part_ids: Vec<String>,
    ) -> Result<(), CatalogError> {
        let lock_dir = self.publishing_scratch(generation);
        if !lock_dir.is_dir() {
            return Err(CatalogError::Package(
                "seal_generation requires begin_publish lock",
            ));
        }
        let dest_dir = self.generation_dir(generation);
        if dest_dir.exists() {
            return Err(CatalogError::PackagePathExists(dest_dir));
        }
        if !packed_tet.is_file() {
            return Err(CatalogError::Package("packed.tet source missing"));
        }

        let packed_byte_len = fs::metadata(packed_tet)?.len();
        let packed_sha256 = sha256_file_hex(packed_tet)?;
        let manifest = GenerationManifest {
            generation,
            protocol_version: PACKAGE_PROTOCOL_VERSION,
            part_ids,
            packed_byte_len,
            packed_sha256: Some(packed_sha256),
        };

        let staged_packed = lock_dir.join(PACKED_TET);
        fs::copy(packed_tet, &staged_packed)?;
        write_json_atomic(&lock_dir.join(MANIFEST_FILE), &manifest)?;
        let _ = fs::remove_file(lock_dir.join(HEARTBEAT_FILE));

        fs::rename(&lock_dir, &dest_dir)?;
        self.set_current(generation)?;
        Ok(())
    }

    /// Memory-map the sealed `packed.tet` for `generation` (decision A — pin this path).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::GenerationNotSealed`] when `packed.tet` is missing, or I/O from
    /// [`TetFile::open`].
    pub fn open_generation(&self, generation: u64) -> Result<TetFile, CatalogError> {
        let packed = self.packed_tet_path(generation);
        if !packed.is_file() {
            return Err(CatalogError::GenerationNotSealed(generation));
        }
        Ok(TetFile::open(packed)?)
    }

    /// Pin the generation named by `CURRENT` and open its `packed.tet`.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::Package`] when `CURRENT` is absent, or errors from
    /// [`Self::open_generation`].
    pub fn open_current(&self) -> Result<TetFile, CatalogError> {
        let Some(g) = self.current()? else {
            return Err(CatalogError::Package("package has no CURRENT generation"));
        };
        self.open_generation(g)
    }

    /// Read `MANIFEST.json` for a sealed generation.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::GenerationNotSealed`] when the file is missing, or JSON / I/O errors.
    pub fn read_manifest(&self, generation: u64) -> Result<GenerationManifest, CatalogError> {
        let path = self.manifest_path(generation);
        if !path.is_file() {
            return Err(CatalogError::GenerationNotSealed(generation));
        }
        read_json(&path)
    }

    /// Create `staging/<session_id>/` for a shard writer session.
    ///
    /// `session_id` must be a single path segment (no separators or `..`).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::Package`] for an invalid id, [`CatalogError::PackagePathExists`]
    /// on collision, or I/O errors.
    pub fn create_staging_session(&self, session_id: &str) -> Result<PathBuf, CatalogError> {
        if !is_safe_session_id(session_id) {
            return Err(CatalogError::Package("invalid staging session id"));
        }
        let dir = self.staging_dir().join(session_id);
        match fs::create_dir(&dir) {
            Ok(()) => Ok(dir),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                Err(CatalogError::PackagePathExists(dir))
            }
            Err(e) => Err(e.into()),
        }
    }

    fn publishing_scratch(&self, generation: u64) -> PathBuf {
        self.publishing_dir().join(format_generation_id(generation))
    }
}

/// Zero-padded generation directory name (`00000001`).
#[must_use]
pub fn format_generation_id(generation: u64) -> String {
    format!("{generation:08}")
}

pub(crate) fn is_safe_path_segment(id: &str) -> bool {
    !id.is_empty() && !id.contains('/') && !id.contains('\\') && !id.contains("..")
}

fn is_safe_session_id(session_id: &str) -> bool {
    is_safe_path_segment(session_id)
}

fn parse_current(text: &str) -> Result<u64, CatalogError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(CatalogError::InvalidCurrent);
    }
    // Bare integer or `{"generation": N}`.
    if trimmed.starts_with('{') {
        #[derive(Deserialize)]
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

pub(crate) fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, CatalogError> {
    let raw = fs::read_to_string(path)?;
    serde_json::from_str(&raw).map_err(|e| CatalogError::InvalidPackageProtocol(e.to_string()))
}

pub(crate) fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), CatalogError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!(
        ".{}.tmp",
        path.file_name().and_then(|s| s.to_str()).unwrap_or("pkg")
    ));
    {
        let mut f = File::create(&tmp)?;
        serde_json::to_writer_pretty(&mut f, value)
            .map_err(|e| CatalogError::InvalidPackageProtocol(e.to_string()))?;
        f.write_all(b"\n")?;
        f.sync_all()?;
    }
    fs_device::publish_file(&tmp, path)?;
    Ok(())
}

fn write_text_atomic(path: &Path, text: &str) -> Result<(), CatalogError> {
    let mut f = File::create(path)?;
    f.write_all(text.as_bytes())?;
    f.sync_all()?;
    Ok(())
}

fn touch_heartbeat(dir: &Path) -> Result<(), CatalogError> {
    write_text_atomic(&dir.join(HEARTBEAT_FILE), &format!("{}\n", unix_now()))
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Hex-encoded SHA-256 of a file (streaming read).
pub(crate) fn sha256_file_hex(path: &Path) -> Result<String, CatalogError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024].into_boxed_slice();
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

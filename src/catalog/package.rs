//! `.tet.d` package skeleton (writer protocol v2) — PROTOCOL, CURRENT, dir roles.
//!
//! Spec: [`docs/writer_protocol_v2.md`](../../docs/writer_protocol_v2.md).
//! 0.2 readers open only `generations/<G>/packed.tet` (layout v1).

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use super::CatalogError;
use super::session::TetFile;
use crate::utils::fs_device;

/// Package protocol version locked with writer protocol v2 (decision F).
pub const PACKAGE_PROTOCOL_VERSION: u32 = 2;

/// Default stale publish-lock age before verify/repair may quarantine (decision G).
pub const DEFAULT_PUBLISH_LOCK_STALE: Duration = Duration::from_secs(60 * 60);

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
    pub protocol_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_unix: Option<u64>,
}

/// Sealed-generation `MANIFEST.json` (provenance + packed image identity).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationManifest {
    pub generation: u64,
    pub protocol_version: u32,
    /// Explicit part-id list frozen at publish (decision C).
    pub part_ids: Vec<String>,
    pub packed_byte_len: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_sha256: Option<String>,
}

/// Open handle to a `.tet.d` package root.
#[derive(Debug, Clone)]
pub struct TetPackage {
    root: PathBuf,
    protocol: PackageProtocol,
}

impl TetPackage {
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn protocol(&self) -> &PackageProtocol {
        &self.protocol
    }

    /// Create a new empty package at `path` (must not exist).
    ///
    /// # Errors
    ///
    /// I/O errors, or path already exists.
    pub fn create(path: impl Into<PathBuf>) -> Result<Self, CatalogError> {
        let root = path.into();
        if root.exists() {
            return Err(CatalogError::PackagePathExists(root));
        }
        fs::create_dir_all(&root)?;
        fs::create_dir_all(root.join(GENERATIONS_DIR))?;
        fs::create_dir_all(root.join(STAGING_DIR))?;
        fs::create_dir_all(root.join(PUBLISHING_DIR))?;

        let protocol = PackageProtocol {
            protocol_version: PACKAGE_PROTOCOL_VERSION,
            created_unix: Some(unix_now()),
        };
        write_json_atomic(&root.join(PROTOCOL_FILE), &protocol)?;

        Ok(Self { root, protocol })
    }

    /// Open an existing package; validates `PROTOCOL`.
    ///
    /// # Errors
    ///
    /// Missing/invalid protocol, unsupported version, or I/O.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, CatalogError> {
        let root = path.into();
        let protocol_path = root.join(PROTOCOL_FILE);
        if !protocol_path.is_file() {
            return Err(CatalogError::NotAPackage(root));
        }
        let raw = fs::read_to_string(&protocol_path)?;
        let protocol: PackageProtocol = serde_json::from_str(&raw)
            .map_err(|e| CatalogError::InvalidPackageProtocol(e.to_string()))?;
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
    /// Invalid pointer contents or I/O.
    pub fn current(&self) -> Result<Option<u64>, CatalogError> {
        let path = self.root.join(CURRENT_FILE);
        if !path.is_file() {
            return Ok(None);
        }
        let text = fs::read_to_string(&path)?;
        parse_current(&text).map(Some)
    }

    /// Atomically set `CURRENT` via temp + rename (decision K).
    ///
    /// # Errors
    ///
    /// I/O / cross-volume publish failures.
    pub fn set_current(&self, generation: u64) -> Result<(), CatalogError> {
        if self.generation_dir(generation).exists() && !self.packed_tet_path(generation).is_file() {
            return Err(CatalogError::GenerationNotSealed(generation));
        }
        let tmp = self.root.join(CURRENT_TMP);
        let dest = self.root.join(CURRENT_FILE);
        {
            let mut f = File::create(&tmp)?;
            writeln!(f, "{generation}")?;
            f.sync_all()?;
        }
        fs_device::publish_file(&tmp, &dest)?;
        Ok(())
    }

    #[must_use]
    pub fn staging_dir(&self) -> PathBuf {
        self.root.join(STAGING_DIR)
    }

    #[must_use]
    pub fn publishing_dir(&self) -> PathBuf {
        self.root.join(PUBLISHING_DIR)
    }

    #[must_use]
    pub fn generations_dir(&self) -> PathBuf {
        self.root.join(GENERATIONS_DIR)
    }

    #[must_use]
    pub fn generation_dir(&self, generation: u64) -> PathBuf {
        self.generations_dir()
            .join(format_generation_id(generation))
    }

    #[must_use]
    pub fn packed_tet_path(&self, generation: u64) -> PathBuf {
        self.generation_dir(generation).join(PACKED_TET)
    }

    #[must_use]
    pub fn manifest_path(&self, generation: u64) -> PathBuf {
        self.generation_dir(generation).join(MANIFEST_FILE)
    }

    /// Exclusive-create `publishing/<G>/` and write a heartbeat (decision G).
    ///
    /// # Errors
    ///
    /// [`CatalogError::PublishLockHeld`] if the directory already exists, or I/O.
    pub fn begin_publish(&self, generation: u64) -> Result<PathBuf, CatalogError> {
        let dir = self.publishing_dir().join(format_generation_id(generation));
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

    /// Drop a publish scratch dir (after success or abandon). Missing dir is OK.
    ///
    /// # Errors
    ///
    /// I/O while removing.
    pub fn abort_publish(&self, generation: u64) -> Result<(), CatalogError> {
        let dir = self.publishing_dir().join(format_generation_id(generation));
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }

    /// Whether `publishing/<G>/` looks stale (no recent heartbeat).
    ///
    /// # Errors
    ///
    /// I/O reading heartbeat metadata.
    pub fn publish_lock_is_stale(
        &self,
        generation: u64,
        max_age: Duration,
    ) -> Result<bool, CatalogError> {
        let dir = self.publishing_dir().join(format_generation_id(generation));
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

    /// Quarantine a stale publish lock by renaming aside (decision G). Does not steal.
    ///
    /// # Errors
    ///
    /// I/O, or lock is not stale under `max_age`.
    pub fn quarantine_publish_lock(
        &self,
        generation: u64,
        max_age: Duration,
    ) -> Result<PathBuf, CatalogError> {
        if !self.publish_lock_is_stale(generation, max_age)? {
            return Err(CatalogError::PublishLockHeld(generation));
        }
        let dir = self.publishing_dir().join(format_generation_id(generation));
        let quarantine = self.publishing_dir().join(format!(
            "{}.quarantine-{}",
            format_generation_id(generation),
            unix_now()
        ));
        fs::rename(&dir, &quarantine)?;
        Ok(quarantine)
    }

    /// Install a sealed generation: copy/move `packed_tet` + write `MANIFEST`, then set `CURRENT`.
    ///
    /// Caller must hold the publish lock (`begin_publish`). On success the lock dir is removed.
    ///
    /// # Errors
    ///
    /// Missing lock, I/O, or generation already sealed.
    pub fn seal_generation(
        &self,
        generation: u64,
        packed_tet: &Path,
        part_ids: Vec<String>,
    ) -> Result<(), CatalogError> {
        let lock_dir = self.publishing_dir().join(format_generation_id(generation));
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

        let meta = fs::metadata(packed_tet)?;
        let packed_byte_len = meta.len();
        let manifest = GenerationManifest {
            generation,
            protocol_version: PACKAGE_PROTOCOL_VERSION,
            part_ids,
            packed_byte_len,
            packed_sha256: None,
        };

        // Build under publishing/, then rename into generations/ (same FS).
        let staged_packed = lock_dir.join(PACKED_TET);
        let staged_manifest = lock_dir.join(MANIFEST_FILE);
        if packed_tet != staged_packed {
            fs::copy(packed_tet, &staged_packed)?;
        }
        write_json_atomic(&staged_manifest, &manifest)?;

        // Remove heartbeat so rename is clean-ish; ignore errors.
        let _ = fs::remove_file(lock_dir.join(HEARTBEAT_FILE));

        fs::rename(&lock_dir, &dest_dir)?;
        self.set_current(generation)?;
        Ok(())
    }

    /// Open mmap reader for a sealed generation's `packed.tet` (decision A).
    ///
    /// # Errors
    ///
    /// Generation not sealed, or [`TetFile::open`] failures.
    pub fn open_generation(&self, generation: u64) -> Result<TetFile, CatalogError> {
        let packed = self.packed_tet_path(generation);
        if !packed.is_file() {
            return Err(CatalogError::GenerationNotSealed(generation));
        }
        Ok(TetFile::open(packed)?)
    }

    /// Pin whatever `CURRENT` points at.
    ///
    /// # Errors
    ///
    /// No current generation, or open failures.
    pub fn open_current(&self) -> Result<TetFile, CatalogError> {
        let Some(g) = self.current()? else {
            return Err(CatalogError::Package("package has no CURRENT generation"));
        };
        self.open_generation(g)
    }

    /// Read sealed generation manifest.
    ///
    /// # Errors
    ///
    /// Missing file or JSON parse error.
    pub fn read_manifest(&self, generation: u64) -> Result<GenerationManifest, CatalogError> {
        let path = self.manifest_path(generation);
        if !path.is_file() {
            return Err(CatalogError::GenerationNotSealed(generation));
        }
        let raw = fs::read_to_string(path)?;
        serde_json::from_str(&raw).map_err(|e| CatalogError::InvalidPackageProtocol(e.to_string()))
    }

    /// Create a unique staging session directory.
    ///
    /// # Errors
    ///
    /// I/O.
    pub fn create_staging_session(&self, session_id: &str) -> Result<PathBuf, CatalogError> {
        if session_id.is_empty()
            || session_id.contains('/')
            || session_id.contains('\\')
            || session_id.contains("..")
        {
            return Err(CatalogError::Package("invalid staging session id"));
        }
        let dir = self.staging_dir().join(session_id);
        fs::create_dir(&dir).map_err(|e| {
            if e.kind() == io::ErrorKind::AlreadyExists {
                CatalogError::PackagePathExists(dir.clone())
            } else {
                CatalogError::Io(e)
            }
        })?;
        Ok(dir)
    }
}

/// Zero-padded generation directory name (`00000001`).
#[must_use]
pub fn format_generation_id(generation: u64) -> String {
    format!("{generation:08}")
}

fn parse_current(text: &str) -> Result<u64, CatalogError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(CatalogError::InvalidCurrent);
    }
    // Allow bare integer or `{"generation": N}`.
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

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), CatalogError> {
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

fn touch_heartbeat(dir: &Path) -> Result<(), CatalogError> {
    let path = dir.join(HEARTBEAT_FILE);
    let mut f = File::create(&path)?;
    writeln!(f, "{}", unix_now())?;
    f.sync_all()?;
    Ok(())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

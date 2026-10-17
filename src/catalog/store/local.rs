//! Local filesystem package store (`.tet.d` root).

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::utils::fs_device;

use super::{
    CatalogError, ObjectMeta, PublishGeneration, ReadAt, WritePart, parse_current_bytes,
    validate_store_key,
};

/// Package-relative keys mapped onto a directory root.
#[derive(Debug, Clone)]
pub struct LocalStore {
    root: PathBuf,
}

impl LocalStore {
    /// Store rooted at `root` (the `.tet.d` directory).
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Package root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path_for(&self, key: &str) -> Result<PathBuf, CatalogError> {
        validate_store_key(key)?;
        Ok(self.root.join(key))
    }
}

impl ReadAt for LocalStore {
    fn read_at(&self, key: &str, offset: u64, len: u64) -> Result<Vec<u8>, CatalogError> {
        let path = self.path_for(key)?;
        let mut file = File::open(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                CatalogError::Package("store object missing")
            } else {
                CatalogError::Io(e)
            }
        })?;
        let file_len = file.metadata()?.len();
        if offset > file_len {
            return Err(CatalogError::Package("read_at offset past end of object"));
        }
        let available = file_len - offset;
        let take = len.min(available);
        let take_usize = usize::try_from(take).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "read_at_len",
            value: take,
        })?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; take_usize];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn object_meta(&self, key: &str) -> Result<Option<ObjectMeta>, CatalogError> {
        let path = self.path_for(key)?;
        match fs::metadata(&path) {
            Ok(meta) if meta.is_file() => Ok(Some(ObjectMeta { len: meta.len() })),
            Ok(_) => Ok(None),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn list_prefix(&self, prefix: &str) -> Result<Vec<String>, CatalogError> {
        if prefix.is_empty() {
            return list_under(&self.root, "");
        }
        validate_store_key(prefix.trim_end_matches('/'))?;
        let base = self.root.join(prefix.trim_end_matches('/'));
        if !base.exists() {
            return Ok(Vec::new());
        }
        let mut out = list_under(&base, prefix.trim_end_matches('/'))?;
        out.sort();
        out.dedup();
        Ok(out)
    }
}

impl WritePart for LocalStore {
    fn create_dir_exclusive(&self, key: &str) -> Result<(), CatalogError> {
        let path = self.path_for(key)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        match fs::create_dir(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(CatalogError::PackagePathExists(path))
            }
            Err(e) => Err(e.into()),
        }
    }

    fn write_part(&self, key: &str, bytes: &[u8]) -> Result<(), CatalogError> {
        let dest = self.path_for(key)?;
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp_name = format!(
            ".{}.store.tmp",
            dest.file_name().and_then(|s| s.to_str()).unwrap_or("obj")
        );
        let tmp = dest
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(tmp_name);
        {
            let mut f = File::create(&tmp)?;
            f.write_all(bytes)?;
            f.sync_all()?;
        }
        fs_device::publish_file(&tmp, &dest)?;
        Ok(())
    }

    fn remove_object(&self, key: &str) -> Result<(), CatalogError> {
        let path = self.path_for(key)?;
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    fn remove_prefix(&self, prefix: &str) -> Result<(), CatalogError> {
        let path = if prefix.is_empty() {
            return Err(CatalogError::Package("refusing to remove store root"));
        } else {
            self.path_for(prefix.trim_end_matches('/'))?
        };
        match fs::remove_dir_all(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    fn rename_prefix(&self, from_prefix: &str, to_prefix: &str) -> Result<(), CatalogError> {
        let from = self.path_for(from_prefix.trim_end_matches('/'))?;
        let to = self.path_for(to_prefix.trim_end_matches('/'))?;
        if !from.exists() {
            return Err(CatalogError::Package("rename_prefix source missing"));
        }
        if to.exists() {
            return Err(CatalogError::PackagePathExists(to));
        }
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::rename(&from, &to)?;
        Ok(())
    }
}

impl PublishGeneration for LocalStore {
    fn publish_generation(
        &self,
        generation: u64,
        expected_current: Option<u64>,
    ) -> Result<(), CatalogError> {
        let actual = self.current_generation()?;
        if actual != expected_current {
            return Err(CatalogError::InvalidCurrent);
        }
        let body = format!("{generation}\n");
        self.write_part("CURRENT", body.as_bytes())
    }

    fn current_generation(&self) -> Result<Option<u64>, CatalogError> {
        match self.read_all("CURRENT") {
            Ok(bytes) => parse_current_bytes(&bytes).map(Some),
            Err(CatalogError::Package("store object missing")) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

fn list_under(dir: &Path, rel_prefix: &str) -> Result<Vec<String>, CatalogError> {
    let mut out = Vec::new();
    if !dir.is_dir() {
        if dir.is_file() && !rel_prefix.is_empty() {
            out.push(rel_prefix.to_string());
        }
        return Ok(out);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let child_rel = if rel_prefix.is_empty() {
            name.into_owned()
        } else {
            format!("{rel_prefix}/{name}")
        };
        let path = entry.path();
        if path.is_dir() {
            out.extend(list_under(&path, &child_rel)?);
        } else if path.is_file() {
            out.push(child_rel);
        }
    }
    Ok(out)
}

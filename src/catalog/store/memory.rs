//! In-memory object store for package protocol tests (no S3).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use super::{
    CatalogError, ObjectMeta, PublishGeneration, ReadAt, WritePart, parse_current_bytes,
    validate_store_key,
};

/// Thread-safe map of package-relative keys → object bytes.
#[derive(Debug, Default)]
pub struct MemoryStore {
    inner: Mutex<MemoryInner>,
}

#[derive(Debug, Default)]
struct MemoryInner {
    objects: BTreeMap<String, Vec<u8>>,
    dirs: BTreeSet<String>,
}

impl MemoryStore {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn with_mut<R>(
        &self,
        f: impl FnOnce(&mut MemoryInner) -> Result<R, CatalogError>,
    ) -> Result<R, CatalogError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| CatalogError::Package("memory store lock poisoned"))?;
        f(&mut guard)
    }

    fn with_ref<R>(
        &self,
        f: impl FnOnce(&MemoryInner) -> Result<R, CatalogError>,
    ) -> Result<R, CatalogError> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| CatalogError::Package("memory store lock poisoned"))?;
        f(&guard)
    }
}

impl ReadAt for MemoryStore {
    fn read_at(&self, key: &str, offset: u64, len: u64) -> Result<Vec<u8>, CatalogError> {
        validate_store_key(key)?;
        self.with_ref(|inner| {
            let Some(bytes) = inner.objects.get(key) else {
                return Err(CatalogError::Package("store object missing"));
            };
            let start = usize::try_from(offset).map_err(|_| CatalogError::TooLargeForPlatform {
                field: "read_at_offset",
                value: offset,
            })?;
            if start > bytes.len() {
                return Err(CatalogError::Package("read_at offset past end of object"));
            }
            let end = start.saturating_add(usize::try_from(len).unwrap_or(usize::MAX));
            let end = end.min(bytes.len());
            Ok(bytes[start..end].to_vec())
        })
    }

    fn object_meta(&self, key: &str) -> Result<Option<ObjectMeta>, CatalogError> {
        validate_store_key(key)?;
        self.with_ref(|inner| {
            Ok(inner.objects.get(key).map(|b| ObjectMeta {
                len: b.len() as u64,
            }))
        })
    }

    fn list_prefix(&self, prefix: &str) -> Result<Vec<String>, CatalogError> {
        if !prefix.is_empty() {
            validate_store_key(prefix.trim_end_matches('/'))?;
        }
        self.with_ref(|inner| {
            let p = prefix.trim_end_matches('/');
            let mut out: Vec<String> = inner
                .objects
                .keys()
                .filter(|k| {
                    if p.is_empty() {
                        true
                    } else {
                        *k == p || k.starts_with(&format!("{p}/"))
                    }
                })
                .cloned()
                .collect();
            out.sort();
            Ok(out)
        })
    }
}

impl WritePart for MemoryStore {
    fn create_dir_exclusive(&self, key: &str) -> Result<(), CatalogError> {
        validate_store_key(key)?;
        self.with_mut(|inner| {
            let marker = key.trim_end_matches('/').to_string();
            if inner.dirs.contains(&marker)
                || inner
                    .objects
                    .keys()
                    .any(|k| k.starts_with(&format!("{marker}/")))
            {
                return Err(CatalogError::PackagePathExists(std::path::PathBuf::from(
                    &marker,
                )));
            }
            ensure_parent_dirs(inner, &format!("{marker}/."));
            inner.dirs.insert(marker);
            Ok(())
        })
    }

    fn write_part(&self, key: &str, bytes: &[u8]) -> Result<(), CatalogError> {
        validate_store_key(key)?;
        self.with_mut(|inner| {
            ensure_parent_dirs(inner, key);
            inner.objects.insert(key.to_string(), bytes.to_vec());
            Ok(())
        })
    }

    fn remove_object(&self, key: &str) -> Result<(), CatalogError> {
        validate_store_key(key)?;
        self.with_mut(|inner| {
            inner.objects.remove(key);
            Ok(())
        })
    }

    fn remove_prefix(&self, prefix: &str) -> Result<(), CatalogError> {
        if prefix.is_empty() {
            return Err(CatalogError::Package("refusing to remove store root"));
        }
        validate_store_key(prefix.trim_end_matches('/'))?;
        self.with_mut(|inner| {
            let p = prefix.trim_end_matches('/');
            inner
                .objects
                .retain(|k, _| !(k == p || k.starts_with(&format!("{p}/"))));
            inner
                .dirs
                .retain(|d| !(d == p || d.starts_with(&format!("{p}/"))));
            Ok(())
        })
    }

    fn rename_prefix(&self, from_prefix: &str, to_prefix: &str) -> Result<(), CatalogError> {
        validate_store_key(from_prefix.trim_end_matches('/'))?;
        validate_store_key(to_prefix.trim_end_matches('/'))?;
        self.with_mut(|inner| {
            let from = from_prefix.trim_end_matches('/');
            let to = to_prefix.trim_end_matches('/');
            let from_slash = format!("{from}/");
            let keys: Vec<String> = inner
                .objects
                .keys()
                .filter(|k| *k == from || k.starts_with(&from_slash))
                .cloned()
                .collect();
            if keys.is_empty() && !inner.dirs.contains(from) {
                return Err(CatalogError::Package("rename_prefix source missing"));
            }
            for key in &keys {
                let new_key = if key == from {
                    to.to_string()
                } else {
                    format!("{to}{}", &key[from.len()..])
                };
                if inner.objects.contains_key(&new_key) {
                    return Err(CatalogError::PackagePathExists(std::path::PathBuf::from(
                        new_key,
                    )));
                }
            }
            for key in keys {
                let new_key = if key == from {
                    to.to_string()
                } else {
                    format!("{to}{}", &key[from.len()..])
                };
                let bytes = inner.objects.remove(&key).expect("key listed above");
                ensure_parent_dirs(inner, &new_key);
                inner.objects.insert(new_key, bytes);
            }
            let dirs: Vec<String> = inner
                .dirs
                .iter()
                .filter(|d| *d == from || d.starts_with(&from_slash))
                .cloned()
                .collect();
            for d in dirs {
                inner.dirs.remove(&d);
                let new_d = if d == from {
                    to.to_string()
                } else {
                    format!("{to}{}", &d[from.len()..])
                };
                inner.dirs.insert(new_d);
            }
            Ok(())
        })
    }
}

impl PublishGeneration for MemoryStore {
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

fn ensure_parent_dirs(inner: &mut MemoryInner, key: &str) {
    if let Some((parent, _)) = key.rsplit_once('/') {
        let mut acc = String::new();
        for segment in parent.split('/') {
            if !acc.is_empty() {
                acc.push('/');
            }
            acc.push_str(segment);
            inner.dirs.insert(acc.clone());
        }
    }
}

//! Byte source for a Zarr store: a local directory, or keys fetched one object at a time.
//!
//! Planning reads array metadata. Tile writes read the chunk keys that tile needs.
//! A directory listing discovers child groups; it does not download chunk bodies.

use std::fs;
use std::path::PathBuf;

use super::ConvertError;

/// Relative-key access to one Zarr v2 or v3 store.
pub(crate) trait ZarrObjects: Send + Sync {
    /// Read one object. `rel` is empty only for the store root (no body lives there).
    ///
    /// # Errors
    ///
    /// The object is missing or the backend read fails.
    fn read_bytes(&self, rel: &str) -> Result<Vec<u8>, ConvertError>;

    /// `true` when `rel` is an object (not a prefix).
    ///
    /// # Errors
    ///
    /// Backend metadata lookup fails. A missing object is `Ok(false)`.
    fn is_file(&self, rel: &str) -> Result<bool, ConvertError>;

    /// Immediate child nodes that contain Zarr metadata.
    ///
    /// `v3` selects `zarr.json`. Otherwise the child has `.zarray` or `.zgroup`.
    ///
    /// # Errors
    ///
    /// The directory or prefix listing fails.
    fn child_names(&self, rel: &str, v3: bool) -> Result<Vec<String>, ConvertError>;

    /// Display path or URI for errors.
    fn location(&self, rel: &str) -> String;
}

/// Local directory store. Keys are relative to the canonical root.
pub(crate) struct LocalZarrDir {
    root: PathBuf,
}

impl LocalZarrDir {
    pub(crate) fn new(root: PathBuf) -> Self {
        Self { root }
    }

    fn path(&self, rel: &str) -> PathBuf {
        if rel.is_empty() {
            self.root.clone()
        } else {
            self.root.join(rel)
        }
    }
}

impl ZarrObjects for LocalZarrDir {
    fn read_bytes(&self, rel: &str) -> Result<Vec<u8>, ConvertError> {
        let path = self.path(rel);
        fs::read(&path).map_err(|err| ConvertError::Zarr(format!("{}: {err}", path.display())))
    }

    fn is_file(&self, rel: &str) -> Result<bool, ConvertError> {
        Ok(self.path(rel).is_file())
    }

    fn child_names(&self, rel: &str, v3: bool) -> Result<Vec<String>, ConvertError> {
        let dir = self.path(rel);
        let mut out = Vec::new();
        for entry in fs::read_dir(&dir)
            .map_err(|err| ConvertError::Zarr(format!("{}: {err}", dir.display())))?
        {
            let entry = entry.map_err(|err| ConvertError::Zarr(err.to_string()))?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let marker = if v3 {
                path.join("zarr.json").is_file()
            } else {
                path.join(".zarray").is_file() || path.join(".zgroup").is_file()
            };
            if !marker {
                continue;
            }
            out.push(
                entry
                    .file_name()
                    .into_string()
                    .map_err(|_| ConvertError::Zarr("non-utf8 zarr child name".into()))?,
            );
        }
        out.sort();
        Ok(out)
    }

    fn location(&self, rel: &str) -> String {
        self.path(rel).display().to_string()
    }
}

/// `true` when the root object `zarr.json` is Zarr v3.
pub(crate) fn zarr_objects_v3(objects: &dyn ZarrObjects) -> bool {
    let Ok(raw) = objects.read_bytes("zarr.json") else {
        return false;
    };
    serde_json::from_slice::<ZarrFormatProbe>(&raw).is_ok_and(|meta| meta.zarr_format == 3)
}

/// `true` when the root has Zarr v2 metadata and is not v3.
pub(crate) fn zarr_objects_v2(objects: &dyn ZarrObjects) -> bool {
    if zarr_objects_v3(objects) {
        return false;
    }
    objects.is_file(".zgroup").unwrap_or(false) || objects.is_file(".zarray").unwrap_or(false)
}

#[derive(serde::Deserialize)]
struct ZarrFormatProbe {
    zarr_format: u32,
}

pub(crate) fn join_rel(prefix: &str, child: &str) -> String {
    if prefix.is_empty() {
        child.to_owned()
    } else {
        format!("{prefix}/{child}")
    }
}

/// Relative chunk key inside the store.
pub(crate) fn zarr_chunk_key(
    array_rel: &str,
    v2_sep: Option<u8>,
    chunk_coord: &[u64],
    ndim: usize,
) -> String {
    let coords: Vec<String> = chunk_coord
        .iter()
        .take(ndim)
        .map(ToString::to_string)
        .collect();
    match v2_sep {
        Some(b'/') => join_segments(array_rel, &coords),
        Some(sep) => {
            let key = coords.join(std::str::from_utf8(&[sep]).unwrap_or("."));
            join_rel(array_rel, &key)
        }
        None => {
            let mut parts = Vec::with_capacity(coords.len() + 1);
            parts.push("c".to_owned());
            parts.extend(coords);
            join_segments(array_rel, &parts)
        }
    }
}

fn join_segments(prefix: &str, parts: &[String]) -> String {
    let mut rel = prefix.to_owned();
    for part in parts {
        rel = join_rel(&rel, part);
    }
    rel
}

/// Immediate child node names whose metadata object is in `keys`.
#[must_use]
#[cfg_attr(not(any(test, feature = "tetration-s3")), allow(dead_code))]
pub(crate) fn zarr_child_names_from_keys(keys: &[String], rel: &str, v3: bool) -> Vec<String> {
    let markers: &[&str] = if v3 {
        &["zarr.json"]
    } else {
        &[".zarray", ".zgroup"]
    };
    let prefix = if rel.is_empty() {
        String::new()
    } else {
        format!("{rel}/")
    };
    let mut out = std::collections::BTreeSet::new();
    for key in keys {
        let rest = if rel.is_empty() {
            key.as_str()
        } else {
            let Some(rest) = key.strip_prefix(prefix.as_str()) else {
                continue;
            };
            rest
        };
        let Some((child, tail)) = rest.split_once('/') else {
            continue;
        };
        if child.is_empty() || !markers.contains(&tail) {
            continue;
        }
        out.insert(child.to_owned());
    }
    out.into_iter().collect()
}

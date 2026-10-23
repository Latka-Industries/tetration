//! Optional declare-only `SEMANTICS.json` beside a sealed generation (0.7.3 / THI-620).
//!
//! Facts for tools / exporters — **not** read by the query engine. Absence is valid.
//! Unknown JSON object keys are ignored on read (forward compatible). Old package readers
//! that do not know this file simply leave it alone.
//!
//! Spec: [`docs/semantics.md`](../../docs/semantics.md).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::history::read_footer_blob_from_path;
use super::metadata::DatasetMetadataV1;
use super::package::{GenerationManifest, TetPackage};
use super::semantics::{declared_attrs, is_declared_key};
use super::{CatalogError, read_tet_summary_v1};
use crate::layout::mmap_file_read;

/// File name beside `packed.tet` inside `generations/<G>/`.
pub const SEMANTICS_FILE: &str = "SEMANTICS.json";

/// Manifest `semantics` value when this seal recorded [`SEMANTICS_FILE`].
pub const SEMANTICS_PRESENT: &str = "present";

/// On-disk document version.
pub const PACKAGE_SEMANTICS_VERSION: u32 = 1;

/// Per-dataset declare-only overlay (string attrs + optional profile id).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DatasetSemanticsDecl {
    /// Doc profile id (`media_v1`, `cf_lite_v1`, …). Not a Rust enum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantics_profile: Option<String>,
    /// Declared string attrs (same vocabulary as footer `attrs`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attrs: BTreeMap<String, String>,
}

/// Optional package-level semantics document for one sealed generation.
///
/// Unknown top-level keys are ignored by serde (no `deny_unknown_fields`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PackageSemanticsV1 {
    pub version: u32,
    pub generation: u64,
    /// Dataset name → declare-only overlay.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub datasets: BTreeMap<String, DatasetSemanticsDecl>,
    /// Free-form note for humans / tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl PackageSemanticsV1 {
    /// Empty document for `generation`.
    #[must_use]
    pub fn empty(generation: u64) -> Self {
        Self {
            version: PACKAGE_SEMANTICS_VERSION,
            generation,
            datasets: BTreeMap::new(),
            notes: None,
        }
    }

    /// Build from footer metadata on a sealed `packed.tet` (declared keys only).
    ///
    /// # Errors
    ///
    /// Catalog / footer read failures.
    pub fn from_packed_tet(packed: &Path, generation: u64) -> Result<Self, CatalogError> {
        let mmap = mmap_file_read(packed)?;
        let summary = read_tet_summary_v1(&mmap)?;
        let mut datasets = BTreeMap::new();
        for (name, ds) in &summary.metadata.datasets {
            let decl = decl_from_dataset_meta(ds);
            if decl.semantics_profile.is_some() || !decl.attrs.is_empty() {
                datasets.insert(name.clone(), decl);
            }
        }
        if datasets.is_empty()
            && let Ok(blob) = read_footer_blob_from_path(packed)
            && let Some(meta) = blob.metadata
        {
            for (name, ds) in meta.datasets {
                let decl = decl_from_dataset_meta(&ds);
                if decl.semantics_profile.is_some() || !decl.attrs.is_empty() {
                    datasets.insert(name, decl);
                }
            }
        }
        Ok(Self {
            version: PACKAGE_SEMANTICS_VERSION,
            generation,
            datasets,
            notes: None,
        })
    }
}

fn decl_from_dataset_meta(ds: &DatasetMetadataV1) -> DatasetSemanticsDecl {
    let attrs = declared_attrs(&ds.attrs);
    let semantics_profile = attrs
        .get("semantics_profile")
        .filter(|s| !s.is_empty())
        .cloned();
    let attrs: BTreeMap<String, String> = attrs
        .into_iter()
        .filter(|(k, _)| is_declared_key(k))
        .collect();
    DatasetSemanticsDecl {
        semantics_profile,
        attrs,
    }
}

/// Pretty JSON bytes (trailing newline) and SHA-256 hex.
///
/// # Errors
///
/// JSON encode failures.
pub fn encode_package_semantics(
    doc: &PackageSemanticsV1,
) -> Result<(Vec<u8>, String), CatalogError> {
    let mut bytes = serde_json::to_vec_pretty(doc)
        .map_err(|e| CatalogError::InvalidPackageProtocol(format!("SEMANTICS.json encode: {e}")))?;
    if !bytes.ends_with(b"\n") {
        bytes.push(b'\n');
    }
    Ok((bytes.clone(), sha256_hex(&bytes)))
}

/// Parse `SEMANTICS.json` bytes (unknown object keys ignored).
///
/// # Errors
///
/// JSON / version failures.
pub fn decode_package_semantics(bytes: &[u8]) -> Result<PackageSemanticsV1, CatalogError> {
    let doc: PackageSemanticsV1 = serde_json::from_slice(bytes)
        .map_err(|e| CatalogError::InvalidPackageProtocol(format!("SEMANTICS.json parse: {e}")))?;
    if doc.version != PACKAGE_SEMANTICS_VERSION {
        return Err(CatalogError::InvalidPackageProtocol(format!(
            "SEMANTICS.json version {} unsupported (want {PACKAGE_SEMANTICS_VERSION})",
            doc.version
        )));
    }
    Ok(doc)
}

/// Read `generations/<G>/SEMANTICS.json` when present.
///
/// # Errors
///
/// I/O or parse failures. Returns `Ok(None)` when the file is absent.
pub fn read_package_semantics(
    package: &TetPackage,
    generation: u64,
) -> Result<Option<PackageSemanticsV1>, CatalogError> {
    let path = package.generation_dir(generation).join(SEMANTICS_FILE);
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&path)?;
    let doc = decode_package_semantics(&bytes)?;
    if doc.generation != generation {
        return Err(CatalogError::InvalidPackageProtocol(format!(
            "SEMANTICS.json generation {} != directory {generation}",
            doc.generation
        )));
    }
    Ok(Some(doc))
}

/// Write `SEMANTICS.json` for `generation` and optionally stamp the MANIFEST.
///
/// Does **not** rewrite `packed.tet`. Fold/query paths are unchanged.
///
/// # Errors
///
/// I/O, encode, or missing generation.
pub fn write_package_semantics(
    package: &TetPackage,
    generation: u64,
    doc: &PackageSemanticsV1,
    update_manifest: bool,
) -> Result<String, CatalogError> {
    if doc.generation != generation {
        return Err(CatalogError::InvalidWriteSpec(
            "SEMANTICS.json generation field must match target generation",
        ));
    }
    let dir = package.generation_dir(generation);
    if !dir.is_dir() {
        return Err(CatalogError::GenerationNotSealed(generation));
    }
    let (bytes, sha) = encode_package_semantics(doc)?;
    let path = dir.join(SEMANTICS_FILE);
    atomic_write(&path, &bytes)?;
    if update_manifest {
        let mut manifest = package.read_manifest(generation)?;
        stamp_manifest_semantics(&mut manifest, &sha);
        write_manifest_atomic(package, generation, &manifest)?;
    }
    Ok(sha)
}

/// Build from packed footer and write; stamps MANIFEST when `update_manifest`.
///
/// # Errors
///
/// Same as [`write_package_semantics`] / footer read.
pub fn write_package_semantics_from_packed(
    package: &TetPackage,
    generation: u64,
    update_manifest: bool,
) -> Result<(PackageSemanticsV1, String), CatalogError> {
    let packed = package.packed_tet_path(generation);
    if !packed.is_file() {
        return Err(CatalogError::GenerationNotSealed(generation));
    }
    let doc = PackageSemanticsV1::from_packed_tet(&packed, generation)?;
    let sha = write_package_semantics(package, generation, &doc, update_manifest)?;
    Ok((doc, sha))
}

fn stamp_manifest_semantics(manifest: &mut GenerationManifest, sha: &str) {
    manifest.semantics = Some(SEMANTICS_PRESENT.to_owned());
    manifest.semantics_sha256 = Some(sha.to_owned());
}

fn write_manifest_atomic(
    package: &TetPackage,
    generation: u64,
    manifest: &GenerationManifest,
) -> Result<(), CatalogError> {
    let path = package.generation_dir(generation).join("MANIFEST.json");
    let mut out = serde_json::to_vec_pretty(manifest)
        .map_err(|e| CatalogError::InvalidPackageProtocol(format!("MANIFEST.json encode: {e}")))?;
    if !out.ends_with(b"\n") {
        out.push(b'\n');
    }
    atomic_write(&path, &out)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), CatalogError> {
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip_ignores_unknown_keys() {
        let raw = br#"{
  "version": 1,
  "generation": 3,
  "datasets": {
    "frames": {
      "semantics_profile": "media_v1",
      "attrs": { "frame_rate": "30" },
      "future_tool_hint": true
    }
  },
  "extra_top": 1
}
"#;
        let doc = decode_package_semantics(raw).unwrap();
        assert_eq!(doc.generation, 3);
        assert_eq!(
            doc.datasets["frames"].semantics_profile.as_deref(),
            Some("media_v1")
        );
        assert_eq!(
            doc.datasets["frames"]
                .attrs
                .get("frame_rate")
                .map(String::as_str),
            Some("30")
        );
    }
}

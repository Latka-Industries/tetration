//! Pin a sealed generation's `packed.tet` object key (reader protocol).
//!
//! [`pin_packed_tet`] reads `CURRENT` at most once (or takes an explicit id) and returns
//! a [`PinnedPackedTet`] that query handles should keep for their lifetime (THI-498).
//! A newer `CURRENT` is invisible until the next pin. Replacing the bytes at the pinned
//! key is not a new generation: the handle does not switch keys. A later object-store
//! `read_at` can observe that overwrite; treat it as a verify failure on this generation.

use crate::catalog::package::{GenerationManifest, format_generation_id};
use crate::catalog::{CatalogError, PackageStore, ReadAt};

use super::uri::GenerationPin;

/// Fixed object key for a sealed generation image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedPackedTet {
    /// Generation id that was pinned (not re-read from `CURRENT` later).
    pub generation: u64,
    /// Package-relative key: `generations/{G:08}/packed.tet`.
    pub object_key: String,
    /// Object length at pin time.
    pub object_len: u64,
}

/// Store key for a sealed `packed.tet`.
#[must_use]
pub fn packed_tet_key(generation: u64) -> String {
    format!(
        "generations/{}/packed.tet",
        format_generation_id(generation)
    )
}

/// `ZONE.json` key for a sealed generation (beside `packed.tet`).
#[must_use]
pub fn zone_maps_key(generation: u64) -> String {
    format!(
        "generations/{}/{}",
        format_generation_id(generation),
        crate::catalog::ZONE_MAPS_FILE
    )
}

/// `INDEX.json` key for a sealed generation (beside `packed.tet`).
#[must_use]
pub fn secondary_index_key(generation: u64) -> String {
    format!(
        "generations/{}/{}",
        format_generation_id(generation),
        crate::catalog::SECONDARY_INDEX_FILE
    )
}

/// Manifest key for a sealed generation.
#[must_use]
pub fn generation_manifest_key(generation: u64) -> String {
    format!(
        "generations/{}/MANIFEST.json",
        format_generation_id(generation)
    )
}

/// Optional declare-only `SEMANTICS.json` key for a sealed generation.
#[must_use]
pub fn package_semantics_key(generation: u64) -> String {
    format!(
        "generations/{}/{}",
        format_generation_id(generation),
        crate::catalog::SEMANTICS_FILE
    )
}

/// Resolve [`GenerationPin`] against `store` and return a pinned packed object.
///
/// Reads `CURRENT` at most once when pin is [`GenerationPin::Current`]. Callers keep
/// [`PinnedPackedTet`] for the life of the query handle and do not re-resolve `CURRENT`
/// (THI-498 / THI-472). Replacing the object at [`PinnedPackedTet::object_key`] does not
/// move the pin to another generation.
///
/// # Errors
///
/// Missing `CURRENT`, missing `packed.tet`, length mismatch vs manifest, or store I/O.
pub fn pin_packed_tet(
    store: &dyn PackageStore,
    pin: GenerationPin,
) -> Result<PinnedPackedTet, CatalogError> {
    let generation = match pin {
        GenerationPin::Current => store
            .current_generation()?
            .ok_or(CatalogError::NoCurrentGeneration)?,
        GenerationPin::Explicit(g) => g,
    };
    let object_key = packed_tet_key(generation);
    let object_len = store
        .object_meta(&object_key)?
        .ok_or(CatalogError::GenerationNotSealed(generation))?
        .len;

    if let Ok(bytes) = store.read_all(&generation_manifest_key(generation)) {
        let manifest: GenerationManifest = serde_json::from_slice(&bytes)
            .map_err(|e| CatalogError::InvalidPackageProtocol(e.to_string()))?;
        if manifest.generation != generation {
            return Err(CatalogError::Package("manifest generation mismatch"));
        }
        if manifest.packed_byte_len != object_len {
            return Err(CatalogError::Package(
                "manifest packed_byte_len does not match object length",
            ));
        }
    }

    Ok(PinnedPackedTet {
        generation,
        object_key,
        object_len,
    })
}

/// Read `MANIFEST.json` for a generation when present.
///
/// # Errors
///
/// JSON / I/O errors. Missing manifest returns `Ok(None)`.
pub fn read_generation_manifest(
    store: &dyn ReadAt,
    generation: u64,
) -> Result<Option<GenerationManifest>, CatalogError> {
    match store.read_all(&generation_manifest_key(generation)) {
        Ok(bytes) => {
            let m = serde_json::from_slice(&bytes)
                .map_err(|e| CatalogError::InvalidPackageProtocol(e.to_string()))?;
            Ok(Some(m))
        }
        Err(CatalogError::Package("store object missing")) => Ok(None),
        Err(e) => Err(e),
    }
}

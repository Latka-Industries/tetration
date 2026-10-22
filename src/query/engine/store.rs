//! Plan and execute queries against a pinned store-backed `packed.tet` (THI-470).

use crate::catalog::store::{
    GenerationPin, PackageStore, PinnedPackedTet, assemble_object_from_ranges, pin_packed_tet,
    read_generation_manifest, read_superblock_at, read_tet_summary_at_store, secondary_index_key,
    zone_maps_key,
};
use crate::catalog::{
    CatalogError, GenerationSecondaryIndex, GenerationZoneMaps, SECONDARY_INDEX_PRESENT,
};
use crate::query::engine::run::{plan_query_with_zones, plan_read_for_document};
use crate::query::engine::spill_policy::SpillPathAllowlist;
use crate::query::execute::ExecuteQueryOptions;
use crate::query::plan::fetch::plan_query_fetch;
use crate::query::types::{QueryDocument, QueryResponse, TetError};

/// Plan a query against a pinned store object using range reads for the catalog.
///
/// # Errors
///
/// Store I/O, catalog parse, or planning failures.
pub fn plan_query_with_store(
    doc: &QueryDocument,
    store: &dyn PackageStore,
    pin: &PinnedPackedTet,
    tet_label: Option<&str>,
) -> Result<QueryResponse, TetError> {
    let buf = materialize_catalog_image(store, pin)?;
    let zones = load_store_zone_maps(store, pin.generation)?;
    let index = load_store_secondary_index(store, pin.generation);
    plan_query_with_zones(
        doc,
        tet_label,
        &buf,
        None,
        None,
        zones.as_ref(),
        index.as_ref(),
    )
}

/// Execute a query by fetching catalog + touched chunk ranges, then running the mmap engine.
///
/// # Errors
///
/// Store I/O, planning, or execution failures. Objects larger than the assemble limit fail.
pub fn execute_query_with_store(
    doc: &QueryDocument,
    store: &dyn PackageStore,
    pin: &PinnedPackedTet,
    options: ExecuteQueryOptions,
    spill_allowlist: Option<&SpillPathAllowlist>,
) -> Result<QueryResponse, TetError> {
    let sb = read_superblock_at(store, &pin.object_key, pin.object_len)
        .map_err(|e| TetError::Validation(e.to_string()))?;
    let catalog_image = materialize_catalog_image(store, pin)?;
    let planned = plan_read_for_document(doc, &catalog_image)?;
    let fetch = plan_query_fetch(&pin.object_key, pin.object_len, &sb, &planned.read_plan)
        .map_err(|e| TetError::Validation(e.to_string()))?;
    let buf = assemble_object_from_ranges(store, &pin.object_key, pin.object_len, &fetch.ranges)
        .map_err(|e| TetError::Validation(e.to_string()))?;
    let zones = load_store_zone_maps(store, pin.generation)?;
    let index = load_store_secondary_index(store, pin.generation);
    // No filesystem path: omit tet_path so spill defaults are not derived from a fake URI.
    plan_query_with_zones(
        doc,
        None,
        &buf,
        options.preview_max,
        spill_allowlist,
        zones.as_ref(),
        index.as_ref(),
    )
}

fn load_store_zone_maps(
    store: &dyn PackageStore,
    generation: u64,
) -> Result<Option<GenerationZoneMaps>, TetError> {
    match store.read_all(&zone_maps_key(generation)) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|e| TetError::Validation(e.to_string())),
        Err(CatalogError::Package("store object missing")) => Ok(None),
        Err(e) => Err(TetError::Catalog(e)),
    }
}

fn load_store_secondary_index(
    store: &dyn PackageStore,
    generation: u64,
) -> Option<GenerationSecondaryIndex> {
    let manifest = read_generation_manifest(store, generation).ok()??;
    if manifest.secondary_indexes.as_deref() != Some(SECONDARY_INDEX_PRESENT) {
        return None;
    }
    let sha = manifest.secondary_indexes_sha256.as_deref()?;
    let bytes = store.read_all(&secondary_index_key(generation)).ok()?;
    // The assembled image may be only the catalog. Trust the manifest hash of the
    // packed file instead of hashing a partial buffer.
    crate::query::plan::index_pushdown::usable_secondary_index(
        &bytes,
        sha,
        manifest.packed_sha256.as_deref(),
    )
}

/// Pin `CURRENT` (or an explicit generation) once and execute `doc` against that object.
///
/// Does not re-read `CURRENT` if a publish lands during the call. Does not parse an `s3://`
/// URI; pass a store you already connected. Feature `tetration-s3` adds `open_s3_package`
/// for `s3://bucket/prefix`.
///
/// # Errors
///
/// Pin, store, or query failures.
pub fn execute_query_on_store_pin(
    doc: &QueryDocument,
    store: &dyn PackageStore,
    generation: GenerationPin,
    options: ExecuteQueryOptions,
) -> Result<QueryResponse, TetError> {
    let pin = pin_packed_tet(store, generation).map_err(|e| TetError::Validation(e.to_string()))?;
    execute_query_with_store(doc, store, &pin, options, None)
}

fn materialize_catalog_image(
    store: &dyn PackageStore,
    pin: &PinnedPackedTet,
) -> Result<Vec<u8>, TetError> {
    // Ensure summary parses; also builds the sparse catalog image for planning.
    let _ = read_tet_summary_at_store(store, &pin.object_key, pin.object_len)
        .map_err(|e| TetError::Validation(e.to_string()))?;
    let sb = read_superblock_at(store, &pin.object_key, pin.object_len)
        .map_err(|e| TetError::Validation(e.to_string()))?;
    let plan = crate::catalog::store::plan_catalog_fetch(pin.object_len, &sb)
        .map_err(|e| TetError::Validation(e.to_string()))?;
    assemble_object_from_ranges(store, &pin.object_key, pin.object_len, &plan.ranges)
        .map_err(|e| TetError::Validation(e.to_string()))
}

/// Plan or execute `doc` against a package opened with [`crate::catalog::open_s3_package`].
///
/// Execute fetches the catalog and the chunks the plan touches. It does not download the rest
/// of `packed.tet`. `tet_label` is echoed as `tet_file` (the `s3://` URI).
///
/// # Errors
///
/// Store I/O, planning, or execution failures.
#[cfg(feature = "tetration-s3")]
pub fn query_opened_s3_package(
    doc: &QueryDocument,
    opened: &crate::catalog::OpenedS3Package,
    execute: bool,
    options: ExecuteQueryOptions,
    spill_allowlist: Option<&SpillPathAllowlist>,
    tet_label: Option<&str>,
) -> Result<QueryResponse, TetError> {
    let mut response = if execute {
        execute_query_with_store(doc, opened.store(), opened.pin(), options, spill_allowlist)?
    } else {
        plan_query_with_store(doc, opened.store(), opened.pin(), tet_label)?
    };
    if let Some(label) = tet_label {
        response.tet_file = Some(label.to_string());
    }
    Ok(response)
}

//! Plan and execute queries against a pinned store-backed `packed.tet` (THI-470).

use crate::catalog::store::{
    GenerationPin, PackageStore, PinnedPackedTet, assemble_object_from_ranges, pin_packed_tet,
    read_superblock_at, read_tet_summary_at_store,
};
use crate::query::engine::run::{plan_query_with_tet_mmap_ex, plan_read_for_document};
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
    plan_query_with_tet_mmap_ex(doc, tet_label, &buf, None, None)
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
    // No filesystem path: omit tet_path so spill defaults are not derived from a fake URI.
    plan_query_with_tet_mmap_ex(doc, None, &buf, options.preview_max, spill_allowlist)
}

/// Pin `CURRENT` (or an explicit generation) and execute `doc`.
///
/// # Errors
///
/// Pin, store, or query failures. `s3://` is not opened here.
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

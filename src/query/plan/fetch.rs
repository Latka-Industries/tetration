//! Query-side byte-range planning over store-backed `packed.tet` (THI-470).

use crate::catalog::CatalogError;
use crate::catalog::store::{ByteRange, ObjectFetchPlan, merge_byte_ranges, plan_catalog_fetch};
use crate::layout::SuperblockV1;
use crate::query::types::ReadPlan;

/// Payload ranges for a planned query (one range per touched chunk, then merged).
#[must_use]
pub fn chunk_payload_ranges(plan: &ReadPlan) -> Vec<ByteRange> {
    merge_byte_ranges(
        plan.chunks
            .iter()
            .map(|c| ByteRange::new(c.payload_offset, c.stored_byte_len)),
    )
}

/// Catalog + payload ranges for a pinned object and [`ReadPlan`].
///
/// # Errors
///
/// Catalog index bounds errors from [`plan_catalog_fetch`].
pub fn plan_query_fetch(
    object_key: &str,
    object_len: u64,
    sb: &SuperblockV1,
    read_plan: &ReadPlan,
) -> Result<ObjectFetchPlan, CatalogError> {
    let catalog = plan_catalog_fetch(object_len, sb)?;
    let mut ranges = catalog.ranges;
    ranges.extend(chunk_payload_ranges(read_plan));
    Ok(ObjectFetchPlan {
        object_key: object_key.to_string(),
        object_len,
        ranges: merge_byte_ranges(ranges),
    })
}

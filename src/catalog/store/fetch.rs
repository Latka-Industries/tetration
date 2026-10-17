//! Range-read planner for sealed `packed.tet` objects (THI-470).

use crate::catalog::{CatalogError, ReadAt, TetFileSummaryV1, read_tet_summary_v1};
use crate::layout::{SUPERBLOCK_FLAG_HISTORY_FOOTER, SUPERBLOCK_V1_LEN, SuperblockV1};

/// One half-open byte range `[offset, offset+len)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    pub offset: u64,
    pub len: u64,
}

impl ByteRange {
    #[must_use]
    pub const fn new(offset: u64, len: u64) -> Self {
        Self { offset, len }
    }
}

/// Planned ranges against one store object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectFetchPlan {
    pub object_key: String,
    pub object_len: u64,
    /// Sorted, merged, non-overlapping ranges.
    pub ranges: Vec<ByteRange>,
}

/// Merge overlapping / adjacent ranges (sort by offset).
#[must_use]
pub fn merge_byte_ranges(ranges: impl IntoIterator<Item = ByteRange>) -> Vec<ByteRange> {
    let mut v: Vec<ByteRange> = ranges.into_iter().filter(|r| r.len > 0).collect();
    if v.is_empty() {
        return v;
    }
    v.sort_by_key(|r| r.offset);
    let mut out = Vec::with_capacity(v.len());
    let mut cur = v[0];
    for r in v.into_iter().skip(1) {
        let cur_end = cur.offset.saturating_add(cur.len);
        if r.offset <= cur_end {
            let r_end = r.offset.saturating_add(r.len);
            cur.len = r_end.saturating_sub(cur.offset).max(cur.len);
        } else {
            out.push(cur);
            cur = r;
        }
    }
    out.push(cur);
    out
}

/// Catalog bootstrap ranges: superblock + dataset directory + full TIDX, then optional EOF footer.
///
/// # Errors
///
/// Index region past `object_len`.
pub fn plan_catalog_fetch(
    object_len: u64,
    sb: &SuperblockV1,
) -> Result<ObjectFetchPlan, CatalogError> {
    let header_end = sb
        .chunk_index_offset
        .checked_add(sb.chunk_index_length)
        .ok_or(CatalogError::Package("chunk index end overflow"))?;
    if header_end > object_len {
        return Err(CatalogError::TooShort {
            need: usize::try_from(header_end).unwrap_or(usize::MAX),
            got: usize::try_from(object_len).unwrap_or(usize::MAX),
        });
    }
    let mut ranges = vec![ByteRange::new(0, header_end)];
    if sb.flags & SUPERBLOCK_FLAG_HISTORY_FOOTER != 0 && object_len > 0 {
        let probe = object_len.min(64 * 1024);
        ranges.push(ByteRange::new(object_len - probe, probe));
    }
    Ok(ObjectFetchPlan {
        object_key: String::new(),
        object_len,
        ranges: merge_byte_ranges(ranges),
    })
}

/// Fetch `ranges` via [`ReadAt::read_at`] (never `read_all`).
///
/// # Errors
///
/// Store I/O or missing object.
pub fn fetch_ranges(
    store: &dyn ReadAt,
    key: &str,
    ranges: &[ByteRange],
) -> Result<Vec<(ByteRange, Vec<u8>)>, CatalogError> {
    let mut out = Vec::with_capacity(ranges.len());
    for r in ranges {
        let bytes = store.read_at(key, r.offset, r.len)?;
        out.push((*r, bytes));
    }
    Ok(out)
}

/// Read the superblock with a 32-byte range get.
///
/// # Errors
///
/// Layout or store errors.
pub fn read_superblock_at(
    store: &dyn ReadAt,
    key: &str,
    object_len: u64,
) -> Result<SuperblockV1, CatalogError> {
    if object_len < SUPERBLOCK_V1_LEN as u64 {
        return Err(CatalogError::TooShort {
            need: SUPERBLOCK_V1_LEN,
            got: usize::try_from(object_len).unwrap_or(0),
        });
    }
    let bytes = store.read_at(key, 0, SUPERBLOCK_V1_LEN as u64)?;
    let mut arr = [0u8; SUPERBLOCK_V1_LEN];
    arr.copy_from_slice(&bytes);
    // Do not use `read_superblock_v1` here: it validates the index against `data.len()`.
    Ok(SuperblockV1::from_bytes(&arr)?)
}

/// Maximum assembled object size for sparse materialize in this release (MiB).
pub const MAX_ASSEMBLE_OBJECT_MIB: u64 = 64;

/// Materialize `ranges` into a buffer of length `object_len` (unset bytes are zero).
///
/// Used so existing mmap query code can run against store-backed fixtures. Caps at
/// [`MAX_ASSEMBLE_OBJECT_MIB`] — larger objects need a non-assembling execute path later.
///
/// # Errors
///
/// Object too large, store I/O, or range past end.
pub fn assemble_object_from_ranges(
    store: &dyn ReadAt,
    key: &str,
    object_len: u64,
    ranges: &[ByteRange],
) -> Result<Vec<u8>, CatalogError> {
    let max_bytes = MAX_ASSEMBLE_OBJECT_MIB.saturating_mul(1024 * 1024);
    if object_len > max_bytes {
        return Err(CatalogError::Package(
            "packed.tet larger than assemble limit for store-backed query",
        ));
    }
    let len = usize::try_from(object_len).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "object_len",
        value: object_len,
    })?;
    let mut buf = vec![0u8; len];
    for (range, bytes) in fetch_ranges(store, key, ranges)? {
        let start =
            usize::try_from(range.offset).map_err(|_| CatalogError::TooLargeForPlatform {
                field: "range_offset",
                value: range.offset,
            })?;
        let end = start
            .checked_add(bytes.len())
            .filter(|&e| e <= buf.len())
            .ok_or(CatalogError::Package("assembled range past object_len"))?;
        buf[start..end].copy_from_slice(&bytes);
    }
    Ok(buf)
}

/// Fetch catalog ranges and parse [`TetFileSummaryV1`] without a full-object `read_all`.
///
/// # Errors
///
/// Layout, assemble limit, or store I/O.
pub fn read_tet_summary_at_store(
    store: &dyn ReadAt,
    key: &str,
    object_len: u64,
) -> Result<TetFileSummaryV1, CatalogError> {
    let sb = read_superblock_at(store, key, object_len)?;
    let plan = plan_catalog_fetch(object_len, &sb)?;
    let buf = assemble_object_from_ranges(store, key, object_len, &plan.ranges)?;
    read_tet_summary_v1(&buf)
}

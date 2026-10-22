//! Chunk allow-lists from a sealed secondary index (`value_postings`, `sorted_axis_range`,
//! `coord_label_postings`).
//!
//! Missing, unusable, or inapplicable indexes return `None`. The planner then keeps
//! the zone-map / geometric result. A `Some` allow-list may be empty: that predicate
//! matches no chunk, so every surviving candidate can be dropped.

use std::collections::BTreeSet;

use half::f16;

use crate::catalog::metadata::{CoordAxisV1, DatasetMetadataV1};
use crate::catalog::zone::{hex_le, sha256_hex};
use crate::catalog::{
    DATASET_DTYPE_TAG_V1, GenerationSecondaryIndex, SECONDARY_INDEX_VERSION, SecondaryIndexEntry,
};
use crate::query::types::{ValueCompareOp, ValueFilter, ValuePredicate};

/// Parse `INDEX.json` bytes when the manifest claim matches.
///
/// Checks: INDEX bytes hash = `manifest_sha256`, document `version` is 1, and (when
/// supplied) `manifest_packed_sha256` equals the document's `packed_sha256`. Does **not**
/// re-hash the sealed `packed.tet` — that is a multi-GiB cost on every query; deep verify
/// already recomputes the index from the image. Any failure returns `None` so the query
/// falls back instead of skipping the wrong chunks.
#[must_use]
pub(crate) fn usable_secondary_index(
    bytes: &[u8],
    manifest_sha256: &str,
    manifest_packed_sha256: Option<&str>,
) -> Option<GenerationSecondaryIndex> {
    if sha256_hex(bytes) != manifest_sha256 {
        return None;
    }
    usable_secondary_index_doc(bytes, manifest_packed_sha256)
}

/// Parse embedded `INDEX.json` spill bytes (no MANIFEST hash gate).
///
/// Same version / optional `packed_sha256` string checks as the sidecar path. Does not
/// re-hash the sealed image.
#[must_use]
pub(crate) fn usable_embedded_secondary_index(bytes: &[u8]) -> Option<GenerationSecondaryIndex> {
    usable_secondary_index_doc(bytes, None)
}

fn usable_secondary_index_doc(
    bytes: &[u8],
    packed_sha256: Option<&str>,
) -> Option<GenerationSecondaryIndex> {
    let doc: GenerationSecondaryIndex = serde_json::from_slice(bytes).ok()?;
    if doc.version != SECONDARY_INDEX_VERSION {
        return None;
    }
    if let Some(expected) = packed_sha256
        && expected != doc.packed_sha256
    {
        return None;
    }
    Some(doc)
}

/// Chunk coordinates that contain the filter's exact `eq` value, when the index applies.
///
/// `None` means do not consult the index (no `eq`, inexact literal, or no posting entry
/// for this dataset). `Some` is the complete allow-list, including an empty set.
#[must_use]
pub(crate) fn eq_chunk_allowlist(
    index: &GenerationSecondaryIndex,
    dataset_id: u64,
    dtype: u32,
    filter: &ValueFilter,
) -> Option<BTreeSet<Vec<u64>>> {
    let value = eq_literal(filter)?;
    let value_le = exact_eq_value_le(dtype, value)?;
    let postings = index.indexes.iter().find_map(|entry| match entry {
        SecondaryIndexEntry::ValuePostings {
            dataset_id: id,
            dtype: entry_dtype,
            postings,
        } if *id == dataset_id && *entry_dtype == dtype => Some(postings),
        _ => None,
    })?;
    let chunks = postings
        .iter()
        .find(|posting| posting.value_le == value_le)
        .map(|posting| posting.chunks.iter().cloned().collect())
        .unwrap_or_default();
    Some(chunks)
}

/// Chunk coordinates overlapping the selection's numeric coord span on sorted axes.
///
/// `None` means do not consult sorted-axis entries (none for this dataset, or labels at
/// the selection bounds are not finite numbers). `Some` is the intersected allow-list
/// across every `sorted_axis_range` entry for `dataset_id` (may be empty).
#[must_use]
pub(crate) fn sorted_axis_chunk_allowlist(
    index: &GenerationSecondaryIndex,
    dataset_id: u64,
    meta: &DatasetMetadataV1,
    dataset_shape: &[u64],
    g0: &[u64],
    g1_exclusive: &[u64],
) -> Option<BTreeSet<Vec<u64>>> {
    let mut combined: Option<BTreeSet<Vec<u64>>> = None;
    let mut saw_entry = false;
    for entry in &index.indexes {
        let SecondaryIndexEntry::SortedAxisRange {
            dataset_id: id,
            axis,
            segments,
            ..
        } = entry
        else {
            continue;
        };
        if *id != dataset_id {
            continue;
        }
        saw_entry = true;
        let axis = usize::try_from(*axis).ok()?;
        if axis >= g0.len() || axis >= g1_exclusive.len() || axis >= dataset_shape.len() {
            return None;
        }
        let (lo, hi) = selection_coord_span(
            meta,
            axis,
            dataset_shape[axis],
            g0[axis],
            g1_exclusive[axis],
        )?;
        let mut set = BTreeSet::new();
        for seg in segments {
            let seg_lo = f64_from_le_hex(&seg.lo_le)?;
            let seg_hi = f64_from_le_hex(&seg.hi_le)?;
            if seg_hi < lo || seg_lo > hi {
                continue;
            }
            for chunk in &seg.chunks {
                set.insert(chunk.clone());
            }
        }
        combined = Some(match combined.take() {
            None => set,
            Some(prev) => prev.intersection(&set).cloned().collect(),
        });
    }
    if saw_entry { combined } else { None }
}

/// Chunk coordinates that contain the footer label, when a `coord_label_postings` entry applies.
///
/// `None` means do not consult the index (no entry for this dataset/axis). `Some` is the
/// complete allow-list, including an empty set when the label is absent from the postings.
#[must_use]
pub(crate) fn label_eq_chunk_allowlist(
    index: &GenerationSecondaryIndex,
    dataset_id: u64,
    axis: u32,
    label: &str,
) -> Option<BTreeSet<Vec<u64>>> {
    let postings = index.indexes.iter().find_map(|entry| match entry {
        SecondaryIndexEntry::CoordLabelPostings {
            dataset_id: id,
            axis: entry_axis,
            postings,
        } if *id == dataset_id && *entry_axis == axis => Some(postings),
        _ => None,
    })?;
    let chunks = postings
        .iter()
        .find(|posting| posting.label == label)
        .map(|posting| posting.chunks.iter().cloned().collect())
        .unwrap_or_default();
    Some(chunks)
}

/// Intersect optional allow-lists. `None` means “no constraint from that kind”.
#[must_use]
pub(crate) fn intersect_allowlists(
    parts: [Option<BTreeSet<Vec<u64>>>; 3],
) -> Option<BTreeSet<Vec<u64>>> {
    let mut combined: Option<BTreeSet<Vec<u64>>> = None;
    for part in parts {
        let Some(set) = part else {
            continue;
        };
        combined = Some(match combined.take() {
            None => set,
            Some(prev) => prev.intersection(&set).cloned().collect(),
        });
    }
    combined
}

fn selection_coord_span(
    meta: &DatasetMetadataV1,
    axis: usize,
    axis_extent: u64,
    start: u64,
    stop_exclusive: u64,
) -> Option<(f64, f64)> {
    if start >= stop_exclusive {
        // Empty selection: no coordinate can match.
        return Some((f64::INFINITY, f64::NEG_INFINITY));
    }
    let start_i = usize::try_from(start).ok()?;
    let stop_i = usize::try_from(stop_exclusive).ok()?;
    let extent = usize::try_from(axis_extent).ok()?;
    let coord = coord_axis_for_dimension(meta, axis, Some(extent))?;
    if stop_i > coord.labels.len() {
        return None;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for label in &coord.labels[start_i..stop_i] {
        let v: f64 = label.parse().ok()?;
        if !v.is_finite() {
            return None;
        }
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    Some((lo, hi))
}

fn coord_axis_for_dimension(
    meta: &DatasetMetadataV1,
    d: usize,
    extent: Option<usize>,
) -> Option<&CoordAxisV1> {
    let coords = meta.coords.as_ref()?;
    if let Some(names) = &meta.dim_names
        && let Some(name) = names.get(d)
        && let Some(c) = coords.get(name)
    {
        return Some(c);
    }
    if let Some(c) = coords.get(d.to_string().as_str()) {
        return Some(c);
    }
    let extent = extent?;
    let mut matches = coords.values().filter(|c| c.labels.len() == extent);
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

fn f64_from_le_hex(hex: &str) -> Option<f64> {
    let bytes = decode_hex_bytes(hex)?;
    let arr: [u8; 8] = bytes.as_slice().try_into().ok()?;
    let v = f64::from_le_bytes(arr);
    v.is_finite().then_some(v)
}

fn decode_hex_bytes(hex: &str) -> Option<Vec<u8>> {
    if !hex.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    for pair in hex.as_bytes().chunks_exact(2) {
        let hi = hex_nibble(pair[0])?;
        let lo = hex_nibble(pair[1])?;
        out.push((hi << 4) | lo);
    }
    Some(out)
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn eq_literal(filter: &ValueFilter) -> Option<f64> {
    let mut found = None;
    for pred in &filter.predicates {
        if let ValuePredicate::Compare {
            op: ValueCompareOp::Eq,
            value,
        } = pred
        {
            if found.is_some() {
                return None;
            }
            found = Some(*value);
        }
    }
    found
}

fn exact_eq_value_le(dtype: u32, value: f64) -> Option<String> {
    if !value.is_finite() {
        return None;
    }
    let tags = DATASET_DTYPE_TAG_V1;
    let bytes = if tags.is_f64(dtype) {
        value.to_le_bytes().to_vec()
    } else if tags.is_f32(dtype) {
        exact_f32(value)?.to_le_bytes().to_vec()
    } else if tags.is_f16(dtype) {
        exact_f16(value)?.to_le_bytes().to_vec()
    } else if tags.is_i64(dtype) {
        exact_i64(value)?.to_le_bytes().to_vec()
    } else if tags.is_i32(dtype) {
        exact_i32(value)?.to_le_bytes().to_vec()
    } else if tags.is_i16(dtype) {
        exact_i16(value)?.to_le_bytes().to_vec()
    } else if tags.is_u64(dtype) {
        exact_u64(value)?.to_le_bytes().to_vec()
    } else if tags.is_u32(dtype) {
        exact_u32(value)?.to_le_bytes().to_vec()
    } else if tags.is_u16(dtype) {
        exact_u16(value)?.to_le_bytes().to_vec()
    } else if tags.is_u8(dtype) {
        vec![exact_u8(value)?]
    } else {
        return None;
    };
    Some(hex_le(&bytes))
}

fn round_trips(encoded: f64, value: f64) -> Option<f64> {
    (encoded.to_bits() == value.to_bits()).then_some(encoded)
}

fn exact_f32(value: f64) -> Option<f32> {
    #[allow(clippy::cast_possible_truncation)]
    let v = value as f32;
    round_trips(f64::from(v), value).map(|_| v)
}

fn exact_f16(value: f64) -> Option<f16> {
    let v = f16::from_f64(value);
    round_trips(f64::from(v), value).map(|_| v)
}

fn exact_i64(value: f64) -> Option<i64> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    let v = value as i64;
    #[allow(clippy::cast_precision_loss)]
    let back = v as f64;
    round_trips(back, value).map(|_| v)
}

fn exact_i32(value: f64) -> Option<i32> {
    #[allow(clippy::cast_possible_truncation)]
    let v = value as i32;
    round_trips(f64::from(v), value).map(|_| v)
}

fn exact_i16(value: f64) -> Option<i16> {
    #[allow(clippy::cast_possible_truncation)]
    let v = value as i16;
    round_trips(f64::from(v), value).map(|_| v)
}

fn exact_u64(value: f64) -> Option<u64> {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let v = value as u64;
    #[allow(clippy::cast_precision_loss)]
    let back = v as f64;
    round_trips(back, value).map(|_| v)
}

fn exact_u32(value: f64) -> Option<u32> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let v = value as u32;
    round_trips(f64::from(v), value).map(|_| v)
}

fn exact_u16(value: f64) -> Option<u16> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let v = value as u16;
    round_trips(f64::from(v), value).map(|_| v)
}

fn exact_u8(value: f64) -> Option<u8> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let v = value as u8;
    round_trips(f64::from(v), value).map(|_| v)
}

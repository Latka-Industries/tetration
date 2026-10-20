//! Equality chunk allow-list from a sealed value-posting index.
//!
//! Missing, unusable, or inapplicable indexes return `None`. The planner then keeps
//! the zone-map result. A `Some` allow-list may be empty: that value is absent, so
//! every surviving chunk can be dropped.

use std::collections::BTreeSet;

use half::f16;

use crate::catalog::zone::{hex_le, sha256_hex};
use crate::catalog::{
    DATASET_DTYPE_TAG_V1, GenerationSecondaryIndex, SECONDARY_INDEX_VERSION, SecondaryIndexEntry,
};
use crate::query::types::{ValueCompareOp, ValueFilter, ValuePredicate};

/// Parse `INDEX.json` bytes when the manifest claim and, when supplied, the packed image match.
///
/// Any failure returns `None` so the query falls back instead of skipping the wrong chunks.
#[must_use]
pub(crate) fn usable_secondary_index(
    bytes: &[u8],
    manifest_sha256: &str,
    manifest_packed_sha256: Option<&str>,
    packed: Option<&[u8]>,
) -> Option<GenerationSecondaryIndex> {
    if sha256_hex(bytes) != manifest_sha256 {
        return None;
    }
    let doc: GenerationSecondaryIndex = serde_json::from_slice(bytes).ok()?;
    if doc.version != SECONDARY_INDEX_VERSION {
        return None;
    }
    if let Some(expected) = manifest_packed_sha256
        && expected != doc.packed_sha256
    {
        return None;
    }
    if let Some(packed) = packed
        && sha256_hex(packed) != doc.packed_sha256
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

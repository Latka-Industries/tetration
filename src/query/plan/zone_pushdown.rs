//! Conservative chunk skip from zone-map stats vs query [`ValueFilter`].
//!
//! Missing zone stats → keep the chunk. Only drop when a predicate cannot match.

use half::f16;

use crate::catalog::{ChunkZoneStat, DATASET_DTYPE_TAG_V1};
use crate::query::types::{ValueCompareOp, ValueFilter, ValuePredicate};

/// True when `zone` might contain cells matching every predicate in `filter`.
///
/// Returns `true` (keep) when bounds cannot be decoded or the overlap is uncertain.
#[must_use]
pub(crate) fn zone_may_match_filter(
    dtype: u32,
    zone: &ChunkZoneStat,
    filter: &ValueFilter,
) -> bool {
    filter
        .predicates
        .iter()
        .all(|pred| predicate_may_match(dtype, zone, pred))
}

fn predicate_may_match(dtype: u32, zone: &ChunkZoneStat, pred: &ValuePredicate) -> bool {
    match pred {
        ValuePredicate::IsNull => zone.null_count > 0,
        ValuePredicate::IsNotNull => zone.null_count < zone.element_count,
        ValuePredicate::Compare { op, value } => compare_may_match(dtype, zone, *op, *value),
    }
}

fn compare_may_match(dtype: u32, zone: &ChunkZoneStat, op: ValueCompareOp, value: f64) -> bool {
    let Some(min) = zone
        .min_le
        .as_deref()
        .and_then(|h| decode_bound_f64(dtype, h))
    else {
        // All-null (or undecodable): no non-null cells for numeric compares.
        return matches!(op, ValueCompareOp::Ne);
    };
    let Some(max) = zone
        .max_le
        .as_deref()
        .and_then(|h| decode_bound_f64(dtype, h))
    else {
        return true;
    };
    match op {
        ValueCompareOp::Eq => value >= min && value <= max,
        ValueCompareOp::Ne => {
            // Skip only when every non-null cell equals `value` and there are no nulls
            // (null ≠ value is treated as a possible match).
            if zone.null_count > 0 {
                return true;
            }
            !(min == max && min == value)
        }
        ValueCompareOp::Lt => min < value,
        ValueCompareOp::Le => min <= value,
        ValueCompareOp::Gt => max > value,
        ValueCompareOp::Ge => max >= value,
    }
}

fn decode_bound_f64(dtype: u32, hex: &str) -> Option<f64> {
    let bytes = decode_hex(hex)?;
    let tags = DATASET_DTYPE_TAG_V1;
    if tags.is_f32(dtype) {
        let arr: [u8; 4] = bytes.as_slice().try_into().ok()?;
        return Some(f64::from(f32::from_le_bytes(arr)));
    }
    if tags.is_f64(dtype) {
        let arr: [u8; 8] = bytes.as_slice().try_into().ok()?;
        return Some(f64::from_le_bytes(arr));
    }
    if tags.is_f16(dtype) {
        let arr: [u8; 2] = bytes.as_slice().try_into().ok()?;
        return Some(f64::from(f16::from_le_bytes(arr)));
    }
    if tags.is_i64(dtype) {
        let arr: [u8; 8] = bytes.as_slice().try_into().ok()?;
        return Some(i64::from_le_bytes(arr) as f64);
    }
    if tags.is_i32(dtype) {
        let arr: [u8; 4] = bytes.as_slice().try_into().ok()?;
        return Some(f64::from(i32::from_le_bytes(arr)));
    }
    if tags.is_i16(dtype) {
        let arr: [u8; 2] = bytes.as_slice().try_into().ok()?;
        return Some(f64::from(i16::from_le_bytes(arr)));
    }
    if tags.is_u64(dtype) {
        let arr: [u8; 8] = bytes.as_slice().try_into().ok()?;
        return Some(u64::from_le_bytes(arr) as f64);
    }
    if tags.is_u32(dtype) {
        let arr: [u8; 4] = bytes.as_slice().try_into().ok()?;
        return Some(f64::from(u32::from_le_bytes(arr)));
    }
    if tags.is_u16(dtype) {
        let arr: [u8; 2] = bytes.as_slice().try_into().ok()?;
        return Some(f64::from(u16::from_le_bytes(arr)));
    }
    if tags.is_u8(dtype) {
        let b = *bytes.first()?;
        return Some(f64::from(b));
    }
    None
}

fn decode_hex(hex: &str) -> Option<Vec<u8>> {
    if !hex.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    let bytes = hex.as_bytes();
    for pair in bytes.as_chunks::<2>().0 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::types::ValuePredicate;

    fn zone(min: f32, max: f32, nulls: u64, elems: u64) -> ChunkZoneStat {
        ChunkZoneStat {
            dataset_id: 0,
            chunk_index: vec![0],
            element_count: elems,
            null_count: nulls,
            min_le: Some(hex_bytes(&min.to_le_bytes())),
            max_le: Some(hex_bytes(&max.to_le_bytes())),
        }
    }

    fn hex_bytes(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0xf) as usize] as char);
        }
        out
    }

    #[test]
    fn skips_chunk_below_gt_threshold() {
        let z = zone(1.0, 2.0, 0, 2);
        let filter = ValueFilter {
            predicates: vec![ValuePredicate::Compare {
                op: ValueCompareOp::Gt,
                value: 5.0,
            }],
        };
        assert!(!zone_may_match_filter(
            DATASET_DTYPE_TAG_V1.f32,
            &z,
            &filter
        ));
    }

    #[test]
    fn keeps_chunk_overlapping_range() {
        let z = zone(1.0, 10.0, 0, 2);
        let filter = ValueFilter {
            predicates: vec![
                ValuePredicate::Compare {
                    op: ValueCompareOp::Ge,
                    value: 5.0,
                },
                ValuePredicate::Compare {
                    op: ValueCompareOp::Lt,
                    value: 8.0,
                },
            ],
        };
        assert!(zone_may_match_filter(DATASET_DTYPE_TAG_V1.f32, &z, &filter));
    }

    #[test]
    fn all_null_skips_numeric_keeps_is_null() {
        let z = ChunkZoneStat {
            dataset_id: 0,
            chunk_index: vec![0],
            element_count: 4,
            null_count: 4,
            min_le: None,
            max_le: None,
        };
        let gt = ValueFilter {
            predicates: vec![ValuePredicate::Compare {
                op: ValueCompareOp::Gt,
                value: 0.0,
            }],
        };
        assert!(!zone_may_match_filter(DATASET_DTYPE_TAG_V1.f32, &z, &gt));
        let is_null = ValueFilter {
            predicates: vec![ValuePredicate::IsNull],
        };
        assert!(zone_may_match_filter(
            DATASET_DTYPE_TAG_V1.f32,
            &z,
            &is_null
        ));
    }
}

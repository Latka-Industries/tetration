//! Flat-wire key tables and membership checks.

use crate::query::types::Operation;

/// Top-level reduction / transform keys — sourced from [`Operation::WIRE_KEYS`].
pub(super) const OP_KEYS: &[&str] = Operation::WIRE_KEYS;

pub(super) const RESERVED_KEYS: &[&str] = &[
    "layout_version",
    "dataset",
    "selection",
    "filter",
    "label_filter",
    "group_by",
    "window",
    "align",
    "pipeline",
    "spill",
    "write",
    "execution",
];

pub(super) const FILTER_COMPARE_KEYS: &[&str] = &["eq", "ne", "lt", "le", "gt", "ge"];

pub(super) fn is_allowed_key(key: &str) -> bool {
    RESERVED_KEYS.contains(&key) || OP_KEYS.contains(&key)
}

pub(super) fn is_flat_analytic_key(key: &str) -> bool {
    matches!(
        key,
        "selection"
            | "filter"
            | "label_filter"
            | "group_by"
            | "window"
            | "align"
            | "spill"
            | "write"
    ) || OP_KEYS.contains(&key)
}

pub(super) fn is_filter_key(key: &str) -> bool {
    FILTER_COMPARE_KEYS.contains(&key) || matches!(key, "is_null" | "is_not_null")
}

//! Flat-wire key tables and membership checks.

pub(super) const OP_KEYS: &[&str] = &[
    "sum",
    "mean",
    "min",
    "max",
    "count",
    "var",
    "std",
    "product",
    "norm_l1",
    "norm_l2",
    "all_finite",
    "any_nan",
    "any_inf",
    "arg_min",
    "arg_max",
    "median",
    "quantile",
    "histogram",
    "nan_count",
    "inf_count",
    "null_count",
    "covariance",
    "correlation",
    "nan_mean",
    "nan_std",
    "transform",
];

pub(super) const RESERVED_KEYS: &[&str] = &[
    "layout_version",
    "dataset",
    "selection",
    "filter",
    "group_by",
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
    matches!(key, "selection" | "filter" | "group_by" | "spill" | "write") || OP_KEYS.contains(&key)
}

pub(super) fn is_filter_key(key: &str) -> bool {
    FILTER_COMPARE_KEYS.contains(&key) || matches!(key, "is_null" | "is_not_null")
}

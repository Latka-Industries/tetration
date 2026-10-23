//! Documented semantics profile registry tables (contracts-as-data).
//!
//! Human contracts: [`docs/semantics_profiles.md`](../../../docs/semantics_profiles.md).
//! Lint and scaffold consume these tables; they do **not** parse the markdown at runtime.
//! Adding a profile for a later cut = extend this file + the docs page.

/// Documented profile ids (canonical spellings). Open minting still allows any other string.
pub const DOCUMENTED_PROFILE_IDS: &[&str] =
    &["media_v1", "timeseries_v1", "cf_lite_v1", "prov_log_v1"];

/// Normalize legacy / alias profile ids to registry ids.
#[must_use]
pub fn normalize_profile_id(id: &str) -> &str {
    match id {
        "climate_v1" => "cf_lite_v1",
        other => other,
    }
}

/// `true` when `id` (after alias normalize) is in [`DOCUMENTED_PROFILE_IDS`].
#[must_use]
pub fn is_documented_profile(id: &str) -> bool {
    let canonical = normalize_profile_id(id);
    DOCUMENTED_PROFILE_IDS.contains(&canonical)
}

/// Recommended declared attr keys for a documented profile (empty → unknown / minted).
#[must_use]
pub fn recommended_keys_for_profile(profile: &str) -> &'static [&'static str] {
    match normalize_profile_id(profile) {
        "media_v1" => &[
            "semantics_profile",
            "media_kind",
            "frame_rate",
            "frame_rate_unit",
            "time_axis",
        ],
        "timeseries_v1" => &[
            "semantics_profile",
            "media_kind",
            "sample_rate",
            "sample_rate_unit",
            "time_axis",
            "units",
        ],
        "cf_lite_v1" => &["semantics_profile", "units", "standard_name", "crs"],
        "prov_log_v1" => &["semantics_profile"],
        _ => &[],
    }
}

/// Recommended `history[].params` / first-class keys when claiming `prov_log_v1`.
pub const PROV_LOG_RECOMMENDED_PARAMS: &[&str] = &["agent", "activity"];

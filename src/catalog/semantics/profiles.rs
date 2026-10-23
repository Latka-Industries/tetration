//! Documented semantics profile registry tables (contracts-as-data).
//!
//! Human contracts: [`docs/semantics_profiles.md`](../../../docs/semantics_profiles.md).
//! Lint and scaffold consume these tables; they do **not** parse the markdown at runtime.
//! Adding a profile for a later cut = extend this file + the docs page.
//!
//! Canonical ids are `name.vN` (`media.v1`). Legacy spellings (`media_v1`, `v1.media`,
//! `climate_v1`) normalize to those ids. Open minting still allows any other string.

/// Documented profile ids (canonical spellings). Open minting still allows any other string.
pub const DOCUMENTED_PROFILE_IDS: &[&str] =
    &["media.v1", "timeseries.v1", "cf_lite.v1", "prov_log.v1"];

/// Normalize legacy / alias profile ids to registry ids.
///
/// Unknown strings pass through unchanged (open minting).
#[must_use]
pub fn normalize_profile_id(id: &str) -> &str {
    match id {
        "media_v1" | "v1.media" => "media.v1",
        "timeseries_v1" | "v1.timeseries" => "timeseries.v1",
        "cf_lite_v1" | "v1.cf_lite" | "climate_v1" => "cf_lite.v1",
        "prov_log_v1" | "v1.prov_log" => "prov_log.v1",
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
        "media.v1" => &[
            "semantics_profile",
            "media_kind",
            "frame_rate",
            "frame_rate_unit",
            "time_axis",
        ],
        "timeseries.v1" => &[
            "semantics_profile",
            "media_kind",
            "sample_rate",
            "sample_rate_unit",
            "time_axis",
            "units",
        ],
        "cf_lite.v1" => &["semantics_profile", "units", "standard_name", "crs"],
        "prov_log.v1" => &["semantics_profile"],
        _ => &[],
    }
}

/// Recommended `history[].params` / first-class keys when claiming `prov_log.v1`.
pub const PROV_LOG_RECOMMENDED_PARAMS: &[&str] = &["agent", "activity"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_ids_are_documented_without_alias_rewrite() {
        for id in DOCUMENTED_PROFILE_IDS {
            assert_eq!(normalize_profile_id(id), *id);
            assert!(is_documented_profile(id));
            assert!(!recommended_keys_for_profile(id).is_empty());
        }
    }

    #[test]
    fn legacy_aliases_normalize_to_name_dot_version() {
        assert_eq!(normalize_profile_id("media_v1"), "media.v1");
        assert_eq!(normalize_profile_id("v1.media"), "media.v1");
        assert_eq!(normalize_profile_id("timeseries_v1"), "timeseries.v1");
        assert_eq!(normalize_profile_id("v1.timeseries"), "timeseries.v1");
        assert_eq!(normalize_profile_id("cf_lite_v1"), "cf_lite.v1");
        assert_eq!(normalize_profile_id("v1.cf_lite"), "cf_lite.v1");
        assert_eq!(normalize_profile_id("climate_v1"), "cf_lite.v1");
        assert_eq!(normalize_profile_id("prov_log_v1"), "prov_log.v1");
        assert_eq!(normalize_profile_id("v1.prov_log"), "prov_log.v1");
        assert!(is_documented_profile("v1.media"));
        assert_eq!(
            recommended_keys_for_profile("media_v1"),
            recommended_keys_for_profile("media.v1")
        );
    }

    #[test]
    fn unknown_profile_is_not_rewritten() {
        assert_eq!(normalize_profile_id("lab.custom"), "lab.custom");
        assert!(!is_documented_profile("lab.custom"));
        assert!(recommended_keys_for_profile("lab.custom").is_empty());
    }
}

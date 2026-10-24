//! Documented semantics profile registry (contracts-as-data).
//!
//! Human contracts: [`docs/semantics_profiles.md`](../../../docs/semantics_profiles.md).
//! Lint and scaffold consume these tables; they do **not** parse the markdown at runtime.
//! Adding a profile for a later cut = extend [`DOCUMENTED_PROFILES`] + the docs page.
//!
//! Canonical ids are `name.vN` (`media.v1`). Legacy spellings (`media_v1`, `v1.media`,
//! `climate_v1`) normalize to those ids. Open minting still allows any other string —
//! this module is **not** a closed Rust enum of all legal ids.

/// One documented profile contract (palette row).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfileContract {
    /// Canonical id (`media.v1`).
    pub id: &'static str,
    /// Legacy / alternate spellings that normalize to [`Self::id`].
    pub aliases: &'static [&'static str],
    /// Recommended footer attr keys when this profile is claimed.
    pub recommended_keys: &'static [&'static str],
    /// Attr used in scaffold index↔seconds formulas (`frame_rate` / `sample_rate`).
    pub rate_attr: Option<&'static str>,
    /// Footer attrs that imply this profile when `semantics_profile` is absent.
    pub infer_attr_keys: &'static [&'static str],
    /// `media_kind` values that imply this profile when `semantics_profile` is absent.
    pub infer_media_kinds: &'static [&'static str],
    /// Recommended `history` / PROV fields when claiming this profile.
    pub recommended_history_params: &'static [&'static str],
}

/// `media.v1` — video / frame stacks.
pub const MEDIA_V1: ProfileContract = ProfileContract {
    id: "media.v1",
    aliases: &["media_v1", "v1.media"],
    recommended_keys: &[
        "semantics_profile",
        "media_kind",
        "frame_rate",
        "frame_rate_unit",
        "time_axis",
    ],
    rate_attr: Some("frame_rate"),
    infer_attr_keys: &["frame_rate"],
    infer_media_kinds: &["video"],
    recommended_history_params: &[],
};

/// `timeseries.v1` — sampled signals.
pub const TIMESERIES_V1: ProfileContract = ProfileContract {
    id: "timeseries.v1",
    aliases: &["timeseries_v1", "v1.timeseries"],
    recommended_keys: &[
        "semantics_profile",
        "media_kind",
        "sample_rate",
        "sample_rate_unit",
        "time_axis",
        "units",
    ],
    rate_attr: Some("sample_rate"),
    infer_attr_keys: &["sample_rate"],
    infer_media_kinds: &["timeseries"],
    recommended_history_params: &[],
};

/// `cf_lite.v1` — CF-lite climate / grid labels.
pub const CF_LITE_V1: ProfileContract = ProfileContract {
    id: "cf_lite.v1",
    aliases: &["cf_lite_v1", "v1.cf_lite", "climate_v1"],
    recommended_keys: &["semantics_profile", "units", "standard_name", "crs"],
    rate_attr: None,
    infer_attr_keys: &["crs", "standard_name"],
    infer_media_kinds: &[],
    recommended_history_params: &[],
};

/// `prov_log.v1` — provenance via history rows.
pub const PROV_LOG_V1: ProfileContract = ProfileContract {
    id: "prov_log.v1",
    aliases: &["prov_log_v1", "v1.prov_log"],
    recommended_keys: &["semantics_profile"],
    rate_attr: None,
    infer_attr_keys: &[],
    infer_media_kinds: &[],
    recommended_history_params: &["agent", "activity"],
};

/// Documented profile contracts (palette). Order matches the docs registry.
pub const DOCUMENTED_PROFILES: &[ProfileContract] =
    &[MEDIA_V1, TIMESERIES_V1, CF_LITE_V1, PROV_LOG_V1];

/// Documented profile ids (canonical spellings). Open minting still allows any other string.
pub const DOCUMENTED_PROFILE_IDS: &[&str] =
    &[MEDIA_V1.id, TIMESERIES_V1.id, CF_LITE_V1.id, PROV_LOG_V1.id];

/// Recommended `history[].params` / first-class keys when claiming `prov_log.v1`.
pub const PROV_LOG_RECOMMENDED_PARAMS: &[&str] = PROV_LOG_V1.recommended_history_params;

/// Look up a documented contract by canonical id or alias.
#[must_use]
pub fn profile_contract(id: &str) -> Option<&'static ProfileContract> {
    let canonical = normalize_profile_id(id);
    DOCUMENTED_PROFILES.iter().find(|p| p.id == canonical)
}

/// Normalize legacy / alias profile ids to registry ids.
///
/// Unknown strings pass through unchanged (open minting).
#[must_use]
pub fn normalize_profile_id(id: &str) -> &str {
    for profile in DOCUMENTED_PROFILES {
        if profile.id == id || profile.aliases.contains(&id) {
            return profile.id;
        }
    }
    id
}

/// `true` when `id` (after alias normalize) is in [`DOCUMENTED_PROFILE_IDS`].
#[must_use]
pub fn is_documented_profile(id: &str) -> bool {
    profile_contract(id).is_some()
}

/// Recommended declared attr keys for a documented profile (empty → unknown / minted).
#[must_use]
pub fn recommended_keys_for_profile(profile: &str) -> &'static [&'static str] {
    profile_contract(profile).map_or(&[], |p| p.recommended_keys)
}

/// Attr used in scaffold index↔seconds formulas, when this profile defines one.
#[must_use]
pub fn rate_attr_for_profile(profile: &str) -> Option<&'static str> {
    profile_contract(profile).and_then(|p| p.rate_attr)
}

/// Default fold-inert index↔seconds formulas for a rate attr name (`frame_rate` / `sample_rate`).
#[must_use]
pub fn default_rate_formulas(rate_attr: &str) -> std::collections::BTreeMap<String, String> {
    let mut formulas = std::collections::BTreeMap::new();
    formulas.insert(
        "index_to_seconds".to_owned(),
        format!("index / {rate_attr}"),
    );
    formulas.insert(
        "seconds_to_index".to_owned(),
        format!("floor(seconds * {rate_attr})"),
    );
    formulas
}

/// Same formulas as [`default_rate_formulas`], as ordered pairs (wizard / display).
#[must_use]
pub fn default_rate_formula_pairs(rate_attr: &str) -> [(&'static str, String); 2] {
    [
        ("index_to_seconds", format!("index / {rate_attr}")),
        ("seconds_to_index", format!("floor(seconds * {rate_attr})")),
    ]
}

/// Infer a documented profile id from footer attrs when `semantics_profile` is absent.
#[must_use]
pub fn infer_profile_id(
    attrs: &std::collections::BTreeMap<String, String>,
) -> Option<&'static str> {
    let kind = attrs.get("media_kind").map(String::as_str);
    for profile in DOCUMENTED_PROFILES {
        if profile
            .infer_attr_keys
            .iter()
            .any(|key| attrs.contains_key(*key))
        {
            return Some(profile.id);
        }
        if let Some(kind) = kind
            && profile.infer_media_kinds.contains(&kind)
        {
            return Some(profile.id);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn canonical_ids_are_documented_without_alias_rewrite() {
        for id in DOCUMENTED_PROFILE_IDS {
            assert_eq!(normalize_profile_id(id), *id);
            assert!(is_documented_profile(id));
            assert!(!recommended_keys_for_profile(id).is_empty());
            assert_eq!(profile_contract(id).unwrap().id, *id);
        }
        assert_eq!(DOCUMENTED_PROFILES.len(), DOCUMENTED_PROFILE_IDS.len());
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
        assert_eq!(rate_attr_for_profile("media.v1"), Some("frame_rate"));
        assert_eq!(rate_attr_for_profile("timeseries.v1"), Some("sample_rate"));
        assert_eq!(rate_attr_for_profile("cf_lite.v1"), None);
    }

    #[test]
    fn unknown_profile_is_not_rewritten() {
        assert_eq!(normalize_profile_id("lab.custom"), "lab.custom");
        assert!(!is_documented_profile("lab.custom"));
        assert!(recommended_keys_for_profile("lab.custom").is_empty());
        assert!(profile_contract("lab.custom").is_none());
    }

    #[test]
    fn infer_profile_id_from_attrs() {
        let mut attrs = BTreeMap::new();
        attrs.insert("frame_rate".into(), "30".into());
        assert_eq!(infer_profile_id(&attrs), Some("media.v1"));
        attrs.clear();
        attrs.insert("media_kind".into(), "timeseries".into());
        assert_eq!(infer_profile_id(&attrs), Some("timeseries.v1"));
        attrs.clear();
        attrs.insert("crs".into(), "EPSG:4326".into());
        assert_eq!(infer_profile_id(&attrs), Some("cf_lite.v1"));
        attrs.clear();
        assert_eq!(infer_profile_id(&attrs), None);
    }
}

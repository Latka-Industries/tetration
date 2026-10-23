//! Declared science/media attrs, profile registry tables, and advisory lint.
//!
//! - [`declare`] — footer attrs / scaffold / set (0.5.1)
//! - [`profiles`] — documented profile ids + recommended-key tables (0.7.3)
//! - [`lint`] — advisory contract checks against [`profiles`]
//! - [`apply`] — one-file TOML declare stamp (`dim_names` + attrs); policy stays in the file
//!
//! Package-level `SEMANTICS.json` lives in [`super::package_semantics`].

pub mod apply;
pub mod declare;
pub mod lint;
pub mod profiles;

pub use apply::{
    SemanticsApplyReport, SemanticsDocument, StampedDataset, apply_semantics_toml,
    format_apply_report, parse_semantics_toml,
};
pub use declare::{
    DECLARED_ATTR_KEYS, SEMANTICS_GUIDE_URL, ScaffoldFormat, ScaffoldPolicy, declared_attrs,
    declared_attrs_for_dataset, get_dataset_attr, is_declared_key, metadata_has_declared_attrs,
    scaffold_annotated_toml, scaffold_policy, scaffold_policy_from_path, set_dataset_attrs,
    write_recording_fixture,
};
pub use lint::{
    SemanticsLintFinding, SemanticsLintReport, SemanticsLintSeverity, format_lint_report,
    lint_declared_semantics, lint_package_semantics, lint_summary, lint_tet_path,
};
pub use profiles::{
    DOCUMENTED_PROFILE_IDS, PROV_LOG_RECOMMENDED_PARAMS, is_documented_profile,
    normalize_profile_id, recommended_keys_for_profile,
};

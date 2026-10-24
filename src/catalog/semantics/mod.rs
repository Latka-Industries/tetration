//! Declared science/media attrs, profile registry tables, and advisory lint.
//!
//! - [`declare`] — footer attrs / scaffold / set (0.5.1)
//! - [`profiles`] — documented profile ids + recommended-key tables (0.7.3)
//! - [`lint`] — advisory contract checks against [`profiles`]
//! - [`apply`] — one-file TOML declare stamp (`dim_names` + attrs); policy stays in the file
//! - [`wizard`] — interactive declare prompts (`tet semantics wizard`)
//!
//! Package-level `SEMANTICS.json` lives in [`super::package_semantics`].

pub mod apply;
pub mod declare;
pub mod lint;
pub(crate) mod path_io;
pub mod profiles;
pub mod wizard;

pub use apply::{
    DatasetDeclare, SemanticsApplyReport, SemanticsDocument, StampedDataset, apply_semantics_toml,
    format_apply_report, parse_semantics_toml,
};
pub use declare::{
    ATTR_SEMANTICS_PROFILE, ATTR_TIME_AXIS, DECLARED_ATTR_KEYS, DeclareFacts, SEMANTICS_GUIDE_URL,
    ScaffoldFormat, ScaffoldPolicy, declared_attrs, declared_attrs_for_dataset, get_dataset_attr,
    is_declared_key, metadata_has_declared_attrs, scaffold_annotated_toml, scaffold_policy,
    scaffold_policy_from_path, set_dataset_attrs, write_recording_fixture,
    write_wizard_scratch_fixture,
};
pub use lint::{
    SemanticsLintCode, SemanticsLintFinding, SemanticsLintReport, SemanticsLintSeverity,
    format_lint_report, lint_declared_semantics, lint_package_semantics, lint_summary,
    lint_tet_path,
};
pub use profiles::{
    CF_LITE_V1, DOCUMENTED_PROFILE_IDS, DOCUMENTED_PROFILES, MEDIA_V1, PROV_LOG_RECOMMENDED_PARAMS,
    PROV_LOG_V1, ProfileContract, TIMESERIES_V1, default_rate_formula_pairs, default_rate_formulas,
    infer_profile_id, is_documented_profile, normalize_profile_id, profile_contract,
    rate_attr_for_profile, recommended_keys_for_profile,
};
#[cfg(test)]
pub(crate) use wizard::ScriptedWizardIo;
pub use wizard::{
    StdWizardIo, WizardChoice, WizardIo, WizardOptions, WizardResult, run_semantics_wizard,
};

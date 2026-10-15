//! `.tet` file health verification (layout, chunk integrity, footer).
//!
//! Distinct from [`crate::catalog`] parsing: this module runs **checks**, emits
//! [`VerifyFinding`]s, and suggests manual fixes via [`VerifyRecommendation`].
//! By default this module is read-only; use [`crate::repair`] or `tet repair` to apply fixes.
//!
//! # Example
//!
//! ```no_run
//! use tetration::verify::verify_tet_file;
//!
//! let report = verify_tet_file("data.tet".as_ref()).unwrap();
//! assert!(report.ok, "{:?}", report.recommendations);
//! ```

mod chunks;
mod datasets;
mod footer;
mod format;
mod options;
mod package;
mod recommend;
mod report;
mod run;

pub use chunks::DEEP_DECODE_MAX_CHUNKS;
pub use format::{format_verify_json, format_verify_quiet, format_verify_text};
pub use options::VerifyOptions;
pub use package::{looks_like_tet_package, verify_tet_package, verify_tet_package_with_options};
pub use report::{
    TetVerifyReport, VerifyFinding, VerifyFixHint, VerifyRecommendation, VerifySeverity,
    VerifySummary,
};
pub use run::{verify_tet_bytes, verify_tet_file, verify_tet_file_with_options};

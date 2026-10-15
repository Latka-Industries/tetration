//! Transform sidecar: draft `.tet` via [`TetWriterSession`], then publish beside the source file.
//!
//! Policy (paths, naming, lineage) lives here; payload persistence uses the catalog writer so
//! query sinks and embedder `TetWriterSession` share one on-disk write stack.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::catalog::{HistoryEvent, TetDatasetWrite, TetWriterSession};
use crate::query::engine::spill_policy::{self, SpillPathAllowlist};
use crate::query::types::{ReadPlan, TetError, TransformMethod, WriteHints, WriteTarget};
use crate::utils::dtype::ElementDtype;

/// Resolved draft + final paths for a sidecar transform.
#[derive(Debug, Clone)]
pub(crate) struct SidecarPaths {
    pub draft: PathBuf,
    pub dest: PathBuf,
    pub dataset_name: String,
}

/// Context required to resolve and write a transform sidecar.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SidecarContext<'a> {
    pub tet_path: &'a Path,
    pub source_dataset: &'a str,
    pub method: TransformMethod,
}

/// Catalog dataset name: `{source}-{method}` (e.g. `temperature-zscore`).
#[must_use]
pub fn sidecar_dataset_name(source_dataset: &str, method: TransformMethod) -> String {
    format!("{source_dataset}-{}", method.as_str())
}

/// Auto sidecar filename beside the parent `.tet` (`stem.method[.timestamp].tet`).
#[must_use]
pub fn auto_sidecar_filename(stem: &str, method: &str, include_timestamp: bool) -> String {
    if include_timestamp {
        format!("{stem}.{method}.{}.tet", utc_filename_timestamp())
    } else {
        format!("{stem}.{method}.tet")
    }
}

/// Resolve cache draft path and validated destination path for `write: sidecar`.
///
/// # Errors
///
/// Returns [`TetError::Validation`] when `tet_path` is missing, paths fail allowlist checks,
/// or no writable cache root exists for the draft.
pub(crate) fn resolve_sidecar_paths(
    write: Option<&WriteHints>,
    allowlist: &SpillPathAllowlist,
    ctx: SidecarContext<'_>,
) -> Result<SidecarPaths, TetError> {
    let hints = write.ok_or_else(|| {
        TetError::Validation("`write.target` sidecar requires a write hint".into())
    })?;
    if hints.target != WriteTarget::Sidecar {
        return Err(TetError::Validation(
            "resolve_sidecar_paths called without sidecar target".into(),
        ));
    }

    let stem = ctx
        .tet_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| TetError::Validation("source .tet path has no file stem".into()))?;

    let include_timestamp = hints.timestamp.unwrap_or(true);
    let dest_rel = hints.path.as_deref().map_or_else(
        || auto_sidecar_filename(stem, ctx.method.as_str(), include_timestamp),
        str::to_owned,
    );
    let dest = allowlist.validate(Path::new(&dest_rel))?;
    let draft = allowlist.allocate_temp_sidecar_draft()?;
    Ok(SidecarPaths {
        draft,
        dest,
        dataset_name: sidecar_dataset_name(ctx.source_dataset, ctx.method),
    })
}

/// Write transformed tensor bytes through [`TetWriterSession`] to `paths.draft`, then publish.
///
/// # Errors
///
/// Propagates catalog, I/O, and allowlist failures.
pub(crate) fn write_and_publish_sidecar(
    mmap: &[u8],
    paths: &SidecarPaths,
    ctx: SidecarContext<'_>,
    plan: &ReadPlan,
    dtype: ElementDtype,
    payload: &[u8],
) -> Result<(), TetError> {
    let shape = &plan.logical_selection_shape;
    let mut dataset = TetDatasetWrite::row_major(
        paths.dataset_name.clone(),
        dtype.wire_tag(),
        shape,
        shape,
        payload.to_vec(),
    )
    .map_err(TetError::Catalog)?;

    if let Some(parent) = parent_dataset_meta(mmap, ctx.source_dataset) {
        dataset.attrs = parent.attrs;
        dataset.dim_names = parent.dim_names;
        dataset.coords = parent.coords;
    }
    dataset
        .attrs
        .insert("derived_from".to_owned(), ctx.source_dataset.to_owned());

    let mut session = TetWriterSession::create(&paths.draft);
    session.metadata.tool = Some("tet query transform".to_owned());
    session.metadata.library_version = Some(env!("CARGO_PKG_VERSION").to_owned());
    session.push_history(build_transform_history(ctx, plan, &paths.dataset_name));
    session.push_dataset(dataset).map_err(TetError::Catalog)?;
    session.commit().map_err(TetError::Catalog)?;

    spill_policy::publish_output_file(&paths.draft, &paths.dest)?;
    Ok(())
}

fn build_transform_history(
    ctx: SidecarContext<'_>,
    plan: &ReadPlan,
    output_dataset: &str,
) -> HistoryEvent {
    let mut event = HistoryEvent::new("transform", ctx.tet_path.display().to_string());
    event.parents.push(ctx.tet_path.display().to_string());
    event
        .params
        .insert("method".to_owned(), ctx.method.as_str().to_owned());
    event
        .params
        .insert("dataset".to_owned(), ctx.source_dataset.to_owned());
    event
        .params
        .insert("output_dataset".to_owned(), output_dataset.to_owned());
    append_selection_params(&mut event.params, plan);
    event
}

fn append_selection_params(params: &mut BTreeMap<String, String>, plan: &ReadPlan) {
    params.insert(
        "logical_selection_shape".to_owned(),
        shape_wire(&plan.logical_selection_shape),
    );
    if plan.logical_selection_shape != plan.dataset_shape {
        params.insert(
            "selection_box_start".to_owned(),
            shape_wire(&plan.selection_box_start),
        );
        params.insert(
            "selection_box_stop_exclusive".to_owned(),
            shape_wire(&plan.selection_box_stop_exclusive),
        );
        if plan.selection_step.iter().any(|&s| s != 1) {
            params.insert(
                "selection_step".to_owned(),
                shape_wire(&plan.selection_step),
            );
        }
    }
}

fn shape_wire(shape: &[u64]) -> String {
    shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

fn parent_dataset_meta(
    mmap: &[u8],
    source_dataset: &str,
) -> Option<crate::catalog::DatasetMetadataV1> {
    let summary = crate::catalog::read_tet_summary_v1(mmap).ok()?;
    summary.metadata.datasets.get(source_dataset).cloned()
}

fn utc_filename_timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format_unix_utc_compact(secs)
}

fn format_unix_utc_compact(secs: u64) -> String {
    const SECS_PER_DAY: u64 = 86_400;
    let days = secs / SECS_PER_DAY;
    let time_of_day = secs % SECS_PER_DAY;
    let (y, m, d) = civil_from_days(days);
    let hh = time_of_day / 3600;
    let mm = (time_of_day % 3600) / 60;
    let ss = time_of_day % 60;
    format!("{y:04}{m:02}{d:02}T{hh:02}{mm:02}{ss:02}Z")
}

/// Days since 1970-01-01 → (year, month, day) in UTC (proleptic Gregorian).
fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = y + u64::from(m <= 2);
    (year, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_dataset_name_joins_source_and_method() {
        assert_eq!(
            sidecar_dataset_name("temperature", TransformMethod::Zscore),
            "temperature-zscore"
        );
    }

    #[test]
    fn auto_filename_without_timestamp() {
        assert_eq!(
            auto_sidecar_filename("volume", "zscore", false),
            "volume.zscore.tet"
        );
    }

    #[test]
    fn utc_compact_format_is_fixed_width() {
        let s = format_unix_utc_compact(1_704_067_200);
        assert_eq!(s, "20240101T000000Z");
    }
}

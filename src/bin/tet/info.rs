//! `tet info` — catalog summary for a `.tet` file.

use std::path::PathBuf;

use tetration::catalog::read_tet_summary_v1;
use tetration::layout::mmap_file_read;
use tetration::query::cli::{
    DEFAULT_INFO_CHUNK_TABLE_LIMIT, InfoListFilter, InfoMetadataDisplay, InfoSection,
    InfoSectionRequest, format_info_json, format_info_quiet, format_info_text,
    info_view_sections_from_flags,
};

use crate::package_ops::resolve_tet_read_path;
use crate::util::cli_error;

/// Output mode for `tet info`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InfoOutputMode {
    Text,
    Json,
    Quiet,
}

pub(crate) struct InfoRunOpts {
    pub path: PathBuf,
    pub output: InfoOutputMode,
    pub sections: InfoSectionRequest,
    pub limit: Option<usize>,
    pub dataset: Option<String>,
    pub grep: Option<String>,
    pub metadata: bool,
}

impl InfoRunOpts {
    /// Build section request from clap bools.
    #[must_use]
    pub(crate) fn sections_from_flags(
        all: bool,
        layout: bool,
        execution: bool,
        datasets: bool,
        chunks: bool,
        show_footer_history: bool,
    ) -> InfoSectionRequest {
        let mut req = InfoSectionRequest::default();
        if all {
            req = req.with_all();
        }
        if layout {
            req = req.enable(InfoSection::Layout);
        }
        if execution {
            req = req.enable(InfoSection::Execution);
        }
        if datasets {
            req = req.enable(InfoSection::Datasets);
        }
        if chunks {
            req = req.enable(InfoSection::Chunks);
        }
        if show_footer_history {
            req = req.enable(InfoSection::History);
        }
        req
    }
}

pub(crate) fn run_info(opts: InfoRunOpts) -> Result<(), String> {
    let resolved = resolve_tet_read_path(&opts.path)?;
    let mmap = mmap_file_read(&resolved).map_err(cli_error)?;
    let file_len = u64::try_from(mmap.len())
        .map_err(|_| format!("file size {} exceeds u64::MAX", mmap.len()))?;
    let summary = read_tet_summary_v1(&mmap).map_err(cli_error)?;
    let filter = build_info_filter(opts.dataset, opts.grep);
    let path_ref = Some(opts.path.as_path());

    let out = match opts.output {
        InfoOutputMode::Json => {
            format_info_json(path_ref, file_len, &summary, filter.as_ref()).map_err(cli_error)?
        }
        InfoOutputMode::Quiet => format_info_quiet(path_ref, file_len, &summary, filter.as_ref()),
        InfoOutputMode::Text => {
            let sections = info_view_sections_from_flags(opts.sections);
            let chunk_limit = opts.limit.unwrap_or(DEFAULT_INFO_CHUNK_TABLE_LIMIT);
            let metadata_display = if opts.metadata {
                InfoMetadataDisplay::Verbose
            } else {
                InfoMetadataDisplay::WhenPresent
            };
            format_info_text(
                path_ref,
                file_len,
                &summary,
                filter.as_ref(),
                sections,
                chunk_limit,
                metadata_display,
            )
        }
    };
    print!("{out}");
    Ok(())
}

fn build_info_filter(
    dataset: Option<String>,
    grep: Option<String>,
) -> Option<tetration::query::cli::InfoListFilter> {
    let filter = InfoListFilter { dataset, grep };
    if filter.is_empty() {
        None
    } else {
        Some(filter)
    }
}

//! CLI formatting for `tet info` (`.tet` catalog summary).

use std::fmt::Write as _;
use std::path::Path;

use crate::catalog::{
    CHUNK_PAYLOAD_CODEC_V1, ChunkIndexEntryV1, CoordAxisV1, DATASET_DTYPE_TAG_V1,
    DatasetMetadataV1, DatasetRecordV1, FileExecutionSettingsV1, HistoryEventV1, TetFileSummaryV1,
    TetMetadataV1,
};
use crate::layout::SuperblockV1;

use super::pretty;
use super::text::{contains_ascii_case_insensitive, truncate_field};

/// Default max chunk rows in the text table (`tet info --chunks`).
pub const DEFAULT_INFO_CHUNK_TABLE_LIMIT: usize = 32;

/// Max coordinate labels printed per axis in `tet info --metadata` text mode.
pub const INFO_METADATA_VERBOSE_LABELS: usize = 12;

/// How dataset footer metadata is rendered in the default dataset table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InfoMetadataDisplay {
    /// Under each dataset row when footer metadata exists: `dim_names`, compact `coords`, attrs.
    #[default]
    WhenPresent,
    /// Like [`WhenPresent`] but prints coordinate label previews (`--metadata`).
    Verbose,
}

/// One optional text section for `tet info`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum InfoSection {
    Layout = 1 << 0,
    Execution = 1 << 1,
    Datasets = 1 << 2,
    Chunks = 1 << 3,
    History = 1 << 4,
}

/// Which sections to print in text mode (`tet info` without `--json`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InfoViewSections {
    bits: u8,
}

impl InfoViewSections {
    const fn with(bits: u8) -> Self {
        Self { bits }
    }

    /// Whether `section` is enabled.
    #[must_use]
    pub const fn contains(self, section: InfoSection) -> bool {
        self.bits & (section as u8) != 0
    }

    #[must_use]
    pub const fn layout(self) -> bool {
        self.contains(InfoSection::Layout)
    }

    #[must_use]
    pub const fn execution(self) -> bool {
        self.contains(InfoSection::Execution)
    }

    #[must_use]
    pub const fn datasets(self) -> bool {
        self.contains(InfoSection::Datasets)
    }

    #[must_use]
    pub const fn chunks(self) -> bool {
        self.contains(InfoSection::Chunks)
    }

    #[must_use]
    pub const fn history(self) -> bool {
        self.contains(InfoSection::History)
    }

    /// Default text view: dataset catalog table only (plus file header).
    #[must_use]
    pub const fn default_table() -> Self {
        Self::with(InfoSection::Datasets as u8)
    }

    /// All text sections (chunk table still obeys `chunk_limit`).
    #[must_use]
    pub const fn all() -> Self {
        Self::with(
            (InfoSection::Layout as u8)
                | (InfoSection::Execution as u8)
                | (InfoSection::Datasets as u8)
                | (InfoSection::Chunks as u8)
                | (InfoSection::History as u8),
        )
    }
}

/// CLI section toggles for [`info_view_sections_from_flags`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InfoSectionRequest {
    all: bool,
    bits: u8,
}

impl InfoSectionRequest {
    /// Enable `--all` (every section).
    #[must_use]
    pub const fn with_all(mut self) -> Self {
        self.all = true;
        self
    }

    /// Enable one section flag (`--layout`, `--chunks`, …).
    #[must_use]
    pub const fn enable(mut self, section: InfoSection) -> Self {
        self.bits |= section as u8;
        self
    }

    /// True when any individual section flag is set.
    #[must_use]
    pub const fn any_section(self) -> bool {
        self.bits != 0
    }
}

/// Filters for dataset / chunk listing (case-insensitive substrings).
#[derive(Debug, Clone, Default)]
pub struct InfoListFilter {
    /// Substring on dataset name.
    pub dataset: Option<String>,
    /// Substring on dataset name or dtype label.
    pub grep: Option<String>,
}

impl InfoListFilter {
    /// True when no predicates are set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dataset.is_none() && self.grep.is_none()
    }

    /// Human summary for headers (empty when no predicates).
    #[must_use]
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(d) = &self.dataset {
            parts.push(format!("dataset~{d}"));
        }
        if let Some(g) = &self.grep {
            parts.push(format!("grep~{g}"));
        }
        parts.join(" ")
    }

    /// Whether `rec` matches all set predicates (`grep` also searches footer metadata when set).
    #[must_use]
    pub fn matches_dataset(
        &self,
        rec: &DatasetRecordV1,
        ds_meta: Option<&DatasetMetadataV1>,
    ) -> bool {
        if let Some(needle) = self.dataset.as_deref()
            && !contains_ascii_case_insensitive(&rec.name, needle)
        {
            return false;
        }
        if let Some(needle) = self.grep.as_deref() {
            let hay = dataset_grep_haystack(rec, ds_meta);
            if !contains_ascii_case_insensitive(&hay, needle) {
                return false;
            }
        }
        true
    }
}

/// Resolve text sections from CLI flags (`--all`, `--layout`, …).
#[must_use]
pub fn info_view_sections_from_flags(req: InfoSectionRequest) -> InfoViewSections {
    if req.all {
        return InfoViewSections::all();
    }
    if req.any_section() {
        InfoViewSections::with(req.bits)
    } else {
        InfoViewSections::default_table()
    }
}

/// Pretty JSON envelope (`tet info --json`).
///
/// # Errors
///
/// JSON serialization error.
pub fn format_info_json(
    path: Option<&Path>,
    file_len: u64,
    summary: &TetFileSummaryV1,
    filter: Option<&InfoListFilter>,
) -> Result<String, String> {
    let filtered = filtered_summary(summary, filter);
    let out = serde_json::json!({
        "path": path.map(|p| p.display().to_string()),
        "file_len": file_len,
        "filter": filter.filter(|f| !f.is_empty()).map(InfoListFilter::summary),
        "summary": filtered,
    });
    serde_json::to_string_pretty(&out).map_err(|e| e.to_string())
}

/// One-line summary (`tet info -q`).
#[must_use]
pub fn format_info_quiet(
    path: Option<&Path>,
    file_len: u64,
    summary: &TetFileSummaryV1,
    filter: Option<&InfoListFilter>,
) -> String {
    let filtered = filtered_summary(summary, filter);
    let path_s = path.map_or("-".to_owned(), |p| p.display().to_string());
    format!(
        "path={path_s} file_len={file_len} layout={} datasets={} chunks={} history={}",
        summary.superblock.layout_version,
        filtered.datasets.len(),
        filtered.chunks.len(),
        filtered.history.len()
    )
}

/// Multi-section text report (default `tet info`).
#[must_use]
pub fn format_info_text(
    path: Option<&Path>,
    file_len: u64,
    summary: &TetFileSummaryV1,
    filter: Option<&InfoListFilter>,
    sections: InfoViewSections,
    chunk_limit: usize,
    metadata_display: InfoMetadataDisplay,
) -> String {
    let filtered = filtered_summary(summary, filter);
    let mut out = String::new();
    let mut header_rows = Vec::new();
    if let Some(p) = path {
        header_rows.push(("file".into(), p.display().to_string()));
    }
    header_rows.push(("size_bytes".into(), file_len.to_string()));
    header_rows.push((
        "layout_version".into(),
        summary.superblock.layout_version.to_string(),
    ));
    header_rows.push(("datasets".into(), summary.datasets.len().to_string()));
    header_rows.push(("chunks".into(), summary.chunks.len().to_string()));
    if let Some(f) = filter.filter(|f| !f.is_empty()) {
        header_rows.push(("filter".into(), f.summary()));
    }
    out.push_str(&pretty::render_kv_section("file:", &header_rows));
    if sections.layout() {
        out.push('\n');
        out.push_str(&format_layout_section(&summary.superblock));
    }
    if sections.execution() {
        out.push('\n');
        out.push_str(&format_execution_section(summary.file_execution));
    }
    if sections.datasets() {
        out.push('\n');
        if let Some(file_meta) = &filtered.metadata.file {
            out.push_str(&format_file_metadata_section(file_meta));
            out.push('\n');
        }
        out.push_str(&format_datasets_table(
            &filtered.datasets,
            &filtered.chunks,
            &filtered.metadata,
            filter,
            metadata_display,
        ));
    }
    if sections.chunks() {
        out.push('\n');
        out.push_str(&format_chunks_table(
            &filtered.datasets,
            &filtered.chunks,
            chunk_limit,
        ));
    }
    if sections.history() {
        out.push('\n');
        out.push_str(&format_history_section(&filtered.history));
    }
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

fn filtered_summary(
    summary: &TetFileSummaryV1,
    filter: Option<&InfoListFilter>,
) -> TetFileSummaryV1 {
    let Some(f) = filter.filter(|x| !x.is_empty()) else {
        return summary.clone();
    };
    let mut datasets = Vec::new();
    let mut id_map = std::collections::HashMap::new();
    for (old_id, rec) in summary.datasets.iter().enumerate() {
        let ds_meta = summary.metadata.datasets.get(&rec.name);
        if f.matches_dataset(rec, ds_meta) {
            id_map.insert(old_id as u64, datasets.len() as u64);
            datasets.push(rec.clone());
        }
    }
    let chunks = summary
        .chunks
        .iter()
        .filter_map(|c| {
            let new_id = *id_map.get(&c.dataset_id)?;
            let mut ch = c.clone();
            ch.dataset_id = new_id;
            Some(ch)
        })
        .collect();
    let metadata = filter_metadata_for_datasets(&summary.metadata, &datasets);
    TetFileSummaryV1 {
        superblock: summary.superblock.clone(),
        datasets,
        chunks,
        file_execution: summary.file_execution,
        history: summary.history.clone(),
        metadata,
    }
}

fn filter_metadata_for_datasets(
    meta: &TetMetadataV1,
    datasets: &[DatasetRecordV1],
) -> TetMetadataV1 {
    let names: std::collections::HashSet<&str> = datasets.iter().map(|d| d.name.as_str()).collect();
    let datasets: std::collections::BTreeMap<_, _> = meta
        .datasets
        .iter()
        .filter(|(k, _)| names.contains(k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if meta.file.is_none() && datasets.is_empty() {
        return TetMetadataV1::default();
    }
    TetMetadataV1 {
        file: meta.file.clone(),
        datasets,
    }
}

fn format_layout_section(sb: &SuperblockV1) -> String {
    let mut out = String::new();
    out.push_str("layout:\n");
    let _ = writeln!(out, "  layout_version:    {}", sb.layout_version);
    let _ = writeln!(out, "  dataset_count:     {}", sb.dataset_count);
    let _ = writeln!(
        out,
        "  flags:             {} (history_footer={})",
        sb.flags,
        sb.flags & crate::layout::SUPERBLOCK_FLAG_HISTORY_FOOTER != 0
    );
    let _ = writeln!(out, "  chunk_index_offset: {}", sb.chunk_index_offset);
    let _ = writeln!(out, "  chunk_index_length: {}", sb.chunk_index_length);
    out
}

fn format_execution_section(ex: FileExecutionSettingsV1) -> String {
    let mut out = String::new();
    out.push_str("execution (file defaults):\n");
    if ex.memory_budget_bytes == 0 && ex.memory_budget_percent_bps == 0 {
        out.push_str("  (engine defaults — 25% host RAM when unset in query JSON)\n");
        return out;
    }
    if ex.memory_budget_percent_bps != 0 {
        let pct = f64::from(ex.memory_budget_percent_bps) / 100.0;
        let _ = writeln!(out, "  memory_budget_percent: {pct}");
    }
    if ex.memory_budget_bytes != 0 {
        let _ = writeln!(out, "  memory_budget_bytes: {}", ex.memory_budget_bytes);
    }
    out
}

fn format_file_metadata_section(file: &crate::catalog::FileMetadataV1) -> String {
    let mut out = String::new();
    out.push_str("file metadata:\n");
    if let Some(tool) = &file.tool {
        let _ = writeln!(out, "  tool: {tool}");
    }
    if let Some(ver) = &file.library_version {
        let _ = writeln!(out, "  library_version: {ver}");
    }
    if let Some(at) = &file.created_at {
        let _ = writeln!(out, "  created_at: {at}");
    }
    out
}

fn dataset_grep_haystack(rec: &DatasetRecordV1, ds_meta: Option<&DatasetMetadataV1>) -> String {
    let mut hay = format!("{} {}", rec.name, dtype_label(rec.dtype));
    let Some(m) = ds_meta else {
        return hay;
    };
    if let Some(dim_names) = &m.dim_names {
        hay.push(' ');
        hay.push_str(&dim_names.join(" "));
    }
    for (k, v) in &m.attrs {
        hay.push(' ');
        hay.push_str(k);
        hay.push(' ');
        hay.push_str(v);
    }
    if let Some(coords) = &m.coords {
        for (axis, c) in coords {
            hay.push(' ');
            hay.push_str(axis);
            for label in &c.labels {
                hay.push(' ');
                hay.push_str(label);
            }
        }
    }
    hay
}

fn format_datasets_table(
    datasets: &[DatasetRecordV1],
    chunks: &[ChunkIndexEntryV1],
    metadata: &TetMetadataV1,
    filter: Option<&InfoListFilter>,
    metadata_display: InfoMetadataDisplay,
) -> String {
    let mut out = String::new();
    if datasets.is_empty() {
        if filter.is_some_and(|f| !f.is_empty()) {
            out.push_str("datasets: (no rows match filter)\n");
        } else {
            out.push_str("datasets: (empty file)\n");
        }
        return out;
    }
    let _ = writeln!(out, "{}", pretty::section_title("datasets:"));
    let rows: Vec<Vec<String>> = datasets
        .iter()
        .enumerate()
        .map(|(id, ds)| {
            let n_chunks = chunks.iter().filter(|c| c.dataset_id == id as u64).count();
            vec![
                id.to_string(),
                truncate_field(&ds.name, 40),
                dtype_label(ds.dtype).to_owned(),
                shape_label(&ds.shape),
                shape_label(&ds.chunk_shape),
                n_chunks.to_string(),
            ]
        })
        .collect();
    out.push_str(&pretty::render_table(
        &["id", "name", "dtype", "shape", "chunk_shape", "chunks"],
        &rows,
    ));
    out.push('\n');
    for ds in datasets {
        if let Some(ds_meta) = metadata.datasets.get(&ds.name) {
            append_dataset_metadata_lines(&mut out, ds.name.as_str(), ds_meta, metadata_display);
        }
    }
    out
}

fn append_dataset_metadata_lines(
    out: &mut String,
    dataset: &str,
    ds_meta: &DatasetMetadataV1,
    display: InfoMetadataDisplay,
) {
    let mut rows = Vec::new();
    if let Some(dim_names) = &ds_meta.dim_names {
        rows.push(("dim_names".to_owned(), dim_names.join(", ")));
    }
    if let Some(coords) = &ds_meta.coords {
        match display {
            InfoMetadataDisplay::WhenPresent => {
                let summary: Vec<String> = coords
                    .iter()
                    .map(|(axis, c)| format!("{axis}({} labels)", c.labels.len()))
                    .collect();
                rows.push(("coords".to_owned(), summary.join(", ")));
            }
            InfoMetadataDisplay::Verbose => {
                for (axis, c) in coords {
                    rows.push(("coords".to_owned(), format_coord_axis_verbose(axis, c)));
                }
            }
        }
    }
    for (k, v) in &ds_meta.attrs {
        rows.push((k.clone(), v.clone()));
    }
    if rows.is_empty() {
        return;
    }
    out.push_str(&pretty::render_kv_section(
        &format!("{dataset} metadata:"),
        &rows,
    ));
}

fn format_coord_axis_verbose(axis: &str, c: &CoordAxisV1) -> String {
    let labels = &c.labels;
    if labels.is_empty() {
        return format!("{axis}: (empty)");
    }
    let show = INFO_METADATA_VERBOSE_LABELS.min(labels.len());
    let head = labels[..show].join(", ");
    if labels.len() > show {
        format!("{axis}: {head} … (+{} more)", labels.len() - show)
    } else {
        format!("{axis}: {head}")
    }
}

fn format_chunks_table(
    datasets: &[DatasetRecordV1],
    chunks: &[ChunkIndexEntryV1],
    limit: usize,
) -> String {
    let mut out = String::new();
    if chunks.is_empty() {
        out.push_str("chunks: (none)\n");
        return out;
    }
    let show = if limit == 0 {
        chunks.len()
    } else {
        limit.min(chunks.len())
    };
    let _ = writeln!(out, "{}", pretty::section_title("chunks:"));
    if show < chunks.len() {
        let _ = writeln!(
            out,
            "{}",
            pretty::dim_line(&format!(
                "(showing {show} of {}; use -n 0 for all)",
                chunks.len()
            ))
        );
    }
    let rows: Vec<Vec<String>> = chunks
        .iter()
        .take(show)
        .enumerate()
        .map(|(i, ch)| {
            let ds_name = datasets
                .get(ch.dataset_id as usize)
                .map_or("?", |d| d.name.as_str());
            vec![
                i.to_string(),
                truncate_field(ds_name, 24),
                chunk_coords_label(ch, datasets.get(ch.dataset_id as usize)),
                ch.raw_byte_len.to_string(),
                ch.stored_byte_len.to_string(),
                codec_label(ch.codec).to_owned(),
                ch.payload_offset.to_string(),
            ]
        })
        .collect();
    out.push_str(&pretty::render_table(
        &["#", "dataset", "coords", "raw", "stored", "codec", "offset"],
        &rows,
    ));
    out
}

fn format_history_section(history: &[HistoryEventV1]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "{}", pretty::section_title("history:"));
    if history.is_empty() {
        out.push_str("  (none)\n");
        return out;
    }
    let rows: Vec<Vec<String>> = history
        .iter()
        .enumerate()
        .map(|(i, ev)| {
            vec![
                (i + 1).to_string(),
                ev.op.clone(),
                ev.source.clone(),
                ev.at.clone(),
            ]
        })
        .collect();
    out.push_str(&pretty::render_table(&["#", "op", "source", "at"], &rows));
    for (i, ev) in history.iter().enumerate() {
        let mut detail = Vec::new();
        if !ev.parents.is_empty() {
            detail.push(("parents".to_owned(), ev.parents.join(", ")));
        }
        for (k, v) in &ev.params {
            detail.push((k.clone(), v.clone()));
        }
        if !detail.is_empty() {
            out.push_str(&pretty::render_kv_section(
                &format!("history[{}] params:", i + 1),
                &detail,
            ));
        }
    }
    out
}

fn dtype_label(tag: u32) -> &'static str {
    let t = DATASET_DTYPE_TAG_V1;
    if t.is_f32(tag) {
        "f32"
    } else if t.is_f64(tag) {
        "f64"
    } else if t.is_i32(tag) {
        "i32"
    } else if t.is_i64(tag) {
        "i64"
    } else {
        "?"
    }
}

fn codec_label(codec: u32) -> &'static str {
    let c = CHUNK_PAYLOAD_CODEC_V1;
    if c.is_raw(codec) {
        "raw"
    } else if c.is_zstd(codec) {
        "zstd"
    } else {
        "?"
    }
}

fn shape_label(shape: &[u64]) -> String {
    if shape.is_empty() {
        return "-".to_owned();
    }
    shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("×")
}

fn chunk_coords_label(ch: &ChunkIndexEntryV1, ds: Option<&DatasetRecordV1>) -> String {
    let rank = ds.map_or(0, |d| d.shape.len());
    if rank == 0 {
        return "-".to_owned();
    }
    ch.chunk_index[..rank]
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

//! Interactive metadata wizard (`tet semantics wizard`) — THI-638.
//!
//! Reads a mutable `.tet`, prompts for dim names + recommended attrs, confirms,
//! writes `tet-semantics.toml`, and optionally [`apply_semantics_toml`].

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
#[cfg(not(feature = "tetration-cli"))]
use std::io::{self, BufRead, Write as IoWrite};
use std::path::{Path, PathBuf};

use serde::Serialize;

use super::apply::{
    SemanticsApplyReport, apply_semantics_toml, format_apply_report, parse_semantics_toml,
};
use super::declare::{
    ATTR_SEMANTICS_PROFILE, ATTR_TIME_AXIS, DeclareFacts, dotted_table, dtype_label,
    render_dataset_declare_toml, toml_string,
};
use super::path_io::{ensure_mutable_tet, read_summary_from_path};
use super::profiles::{
    CF_LITE_V1, MEDIA_V1, PROV_LOG_V1, ProfileContract, TIMESERIES_V1, default_rate_formula_pairs,
    normalize_profile_id, rate_attr_for_profile,
};
use crate::catalog::{CatalogError, DatasetRecordV1, TetFileSummaryV1};
use crate::cli_pretty::{dim_line, fmt_shape_dims, render_table, section_title};

/// Documented profiles the wizard walks — keeps UI copy next to the id, not scattered string matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WizardProfile {
    Media,
    Timeseries,
    CfLite,
    ProvLog,
}

impl WizardProfile {
    const ALL: [Self; 4] = [Self::Media, Self::Timeseries, Self::CfLite, Self::ProvLog];

    fn from_id(id: &str) -> Option<Self> {
        match normalize_profile_id(id) {
            "media.v1" => Some(Self::Media),
            "timeseries.v1" => Some(Self::Timeseries),
            "cf_lite.v1" => Some(Self::CfLite),
            "prov_log.v1" => Some(Self::ProvLog),
            _ => None,
        }
    }

    fn contract(self) -> &'static ProfileContract {
        match self {
            Self::Media => &MEDIA_V1,
            Self::Timeseries => &TIMESERIES_V1,
            Self::CfLite => &CF_LITE_V1,
            Self::ProvLog => &PROV_LOG_V1,
        }
    }

    fn id(self) -> &'static str {
        self.contract().id
    }

    fn blurb(self) -> &'static str {
        match self {
            Self::Media => "video / frame stacks",
            Self::Timeseries => "sampled signals",
            Self::CfLite => "CF-lite grids",
            Self::ProvLog => "provenance via history",
        }
    }

    fn attr_help(self) -> &'static str {
        match self {
            Self::Media => {
                "Facts for a frame stack: kind, rate along time_axis, optional frame_rate_unit (not fixed to Hz)."
            }
            Self::Timeseries => {
                "Facts for a continuous signal: sample_rate (+ unit), time_axis, physical units."
            }
            Self::CfLite => {
                "CF-style labels only — units / standard_name / crs; no climate engine in the crate."
            }
            Self::ProvLog => {
                "Profile id alone is enough here; prefer history rows for agent/activity."
            }
        }
    }

    fn extras_help(self) -> &'static str {
        match self {
            Self::Media => {
                "Optional KEY=VAL beyond the list (e.g. pixel_unit=px, color_space=sRGB)."
            }
            Self::Timeseries => "Optional KEY=VAL beyond the list (e.g. start_time=ISO-8601).",
            Self::CfLite => "Optional KEY=VAL beyond the list (e.g. uncertainty_kind=1sigma).",
            Self::ProvLog => {
                "Optional KEY=VAL; provenance prefers history rows over inventing attrs."
            }
        }
    }

    fn attr_hint(self, key: &str) -> Option<&'static str> {
        match (self, key) {
            (Self::Media, "media_kind") => Some("video"),
            (Self::Media, "frame_rate") => Some("30"),
            (Self::Media, "frame_rate_unit") | (Self::Timeseries, "sample_rate_unit") => Some("Hz"),
            (Self::Media | Self::Timeseries, ATTR_TIME_AXIS) => Some("0"),
            (Self::Timeseries, "media_kind") => Some("timeseries"),
            (Self::Timeseries, "sample_rate") => Some("30000"),
            (Self::Timeseries, "units") => Some("1"),
            (Self::CfLite, "units") => Some("K"),
            (Self::CfLite, "standard_name") => Some("air_temperature"),
            (Self::CfLite, "crs") => Some("EPSG:4326"),
            _ => None,
        }
    }

    fn select_label(self) -> String {
        format!("{} — {}", self.id(), self.blurb())
    }

    fn recommended_attr_keys(self) -> Vec<&'static str> {
        self.contract()
            .recommended_keys
            .iter()
            .copied()
            .filter(|k| *k != ATTR_SEMANTICS_PROFILE)
            .collect()
    }
}

/// Wizard section banners — one place for the `── … ──` titles.
#[derive(Debug, Clone, Copy)]
enum WizardSection {
    Axes,
    Profile,
    Recommended,
    IncludedPolicy,
    Confirm,
    Extras,
    Redo,
}

impl WizardSection {
    fn title(self) -> &'static str {
        match self {
            Self::Axes => "── axes ──",
            Self::Profile => "── profile ──",
            Self::Recommended => "── recommended attrs ──",
            Self::IncludedPolicy => "── included policy ──",
            Self::Confirm => "── confirm ──",
            Self::Extras => "── extras ──",
            Self::Redo => {
                "── redo ──  last selection shown in [brackets]; empty keeps it, - clears"
            }
        }
    }

    fn titled(self, suffix: &str) -> String {
        match self {
            Self::Recommended => format!("── recommended attrs ({suffix}) ──"),
            Self::IncludedPolicy => format!("── included policy ({suffix}) ──"),
            other => other.title().to_owned(),
        }
    }
}

/// CLI / library options for one wizard run.
#[derive(Debug, Clone)]
pub struct WizardOptions {
    /// Restrict to one dataset (default: prompt if several).
    pub dataset: Option<String>,
    /// Where to write the combined TOML (declare + generated policy stubs).
    pub toml_out: PathBuf,
    /// After writing TOML, stamp the `.tet` via apply.
    pub apply: bool,
}

impl WizardOptions {
    /// Default TOML path beside the `.tet` (`tet-semantics.toml` in the same directory).
    #[must_use]
    pub fn beside_tet(tet: &Path) -> Self {
        let toml_out = tet
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("tet-semantics.toml");
        Self {
            dataset: None,
            toml_out,
            apply: true,
        }
    }
}

/// What the wizard wrote / stamped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WizardResult {
    pub dataset: String,
    pub toml_path: PathBuf,
    pub apply: Option<SemanticsApplyReport>,
}

/// One selectable row: stable `id` for soft-defaults / scripts, `label` for display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WizardChoice {
    /// Stable key (profile id, axis index, dataset name).
    pub id: String,
    /// Human-facing label shown in the menu.
    pub label: String,
}

impl WizardChoice {
    #[must_use]
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
        }
    }
}

/// Line-oriented IO for the prompt loop (stdin in CLI; scripted in tests).
pub trait WizardIo {
    /// Print a line to the user (no trailing newline required in `msg`).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when the underlying writer or terminal fails.
    fn println(&mut self, msg: &str) -> Result<(), CatalogError>;

    /// Free-text prompt. Empty line → `Ok(None)` (caller decides keep/skip).
    ///
    /// Last selection is shown as a soft default when present (Enter keeps it).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] on I/O failure or exhausted scripted answers.
    fn prompt(&mut self, label: &str, last: Option<&str>) -> Result<Option<String>, CatalogError>;

    /// Arrow-key (or scripted) pick among `choices`. `None` = skip.
    ///
    /// `last_id` is matched against [`WizardChoice::id`] (then label) for the initial cursor.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] on I/O failure, bad index, or exhausted script.
    fn select(
        &mut self,
        label: &str,
        choices: &[WizardChoice],
        last_id: Option<&str>,
    ) -> Result<Option<usize>, CatalogError>;
}

/// Interactive CLI IO (`dialoguer` when `tetration-cli` is enabled).
#[derive(Debug, Default)]
pub struct StdWizardIo;

impl StdWizardIo {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl WizardIo for StdWizardIo {
    fn println(&mut self, msg: &str) -> Result<(), CatalogError> {
        eprintln!("{msg}");
        Ok(())
    }

    fn prompt(&mut self, label: &str, last: Option<&str>) -> Result<Option<String>, CatalogError> {
        #[cfg(feature = "tetration-cli")]
        {
            use dialoguer::{Input, theme::ColorfulTheme};
            let theme = ColorfulTheme::default();
            let mut input = Input::<String>::with_theme(&theme).with_prompt(label);
            if let Some(d) = last.filter(|s| !s.is_empty()) {
                input = input.default(d.to_owned());
            }
            let value = input
                .allow_empty(true)
                .interact_text()
                .map_err(|e| dialoguer_err(&e))?;
            Ok((!value.is_empty()).then_some(value))
        }
        #[cfg(not(feature = "tetration-cli"))]
        {
            let mut writer = io::stderr().lock();
            match last {
                Some(d) if !d.is_empty() => {
                    write!(writer, "{label} (last: {d}; Enter keeps): ").map_err(io_err)?
                }
                _ => write!(writer, "{label}: ").map_err(io_err)?,
            }
            writer.flush().map_err(io_err)?;
            let mut line = String::new();
            io::stdin().lock().read_line(&mut line).map_err(io_err)?;
            let trimmed = line.trim_end_matches(['\r', '\n']).trim();
            if trimmed.is_empty() {
                Ok(last.filter(|s| !s.is_empty()).map(str::to_owned))
            } else {
                Ok(Some(trimmed.to_owned()))
            }
        }
    }

    fn select(
        &mut self,
        label: &str,
        choices: &[WizardChoice],
        last_id: Option<&str>,
    ) -> Result<Option<usize>, CatalogError> {
        if choices.is_empty() {
            return Ok(None);
        }
        #[cfg(feature = "tetration-cli")]
        {
            use dialoguer::{Select, theme::ColorfulTheme};
            let mut opts: Vec<String> = choices.iter().map(|c| c.label.clone()).collect();
            let skip_idx = opts.len();
            opts.push("(skip)".to_owned());
            let default = last_id
                .and_then(|v| find_choice_index(choices, v))
                .unwrap_or(0);
            match Select::with_theme(&ColorfulTheme::default())
                .with_prompt(label)
                .items(&opts)
                .default(default)
                .interact_opt()
                .map_err(|e| dialoguer_err(&e))?
            {
                None => Ok(None),
                Some(i) if i == skip_idx => Ok(None),
                Some(i) => Ok(Some(i)),
            }
        }
        #[cfg(not(feature = "tetration-cli"))]
        {
            self.println(label)?;
            for (i, c) in choices.iter().enumerate() {
                self.println(&format!("  {i}. {}", c.label))?;
            }
            match self.prompt("  pick index", last_id)?.as_deref() {
                None => Ok(last_id.and_then(|v| find_choice_index(choices, v))),
                Some("-" | "skip") => Ok(None),
                Some(s) => parse_select_index(s, choices.len()),
            }
        }
    }
}

/// Queued answers for unit tests (`None` in the queue = empty line).
#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct ScriptedWizardIo {
    answers: Vec<Option<String>>,
    cursor: usize,
    pub log: Vec<String>,
}

#[cfg(test)]
impl ScriptedWizardIo {
    #[must_use]
    pub(crate) fn new(answers: Vec<Option<String>>) -> Self {
        Self {
            answers,
            cursor: 0,
            log: Vec::new(),
        }
    }

    fn next_answer(&mut self, shown: &str) -> Result<Option<String>, CatalogError> {
        let ans = self.answers.get(self.cursor).cloned().ok_or_else(|| {
            CatalogError::InvalidSemantics(format!("wizard script exhausted at {shown:?}"))
        })?;
        self.cursor += 1;
        Ok(ans)
    }
}

#[cfg(test)]
impl WizardIo for ScriptedWizardIo {
    fn println(&mut self, msg: &str) -> Result<(), CatalogError> {
        self.log.push(msg.to_owned());
        Ok(())
    }

    fn prompt(&mut self, label: &str, last: Option<&str>) -> Result<Option<String>, CatalogError> {
        let shown = match last {
            Some(d) if !d.is_empty() => format!("{label} (last: {d})"),
            _ => label.to_owned(),
        };
        self.log.push(format!("PROMPT {shown}"));
        self.next_answer(&shown)
    }

    fn select(
        &mut self,
        label: &str,
        choices: &[WizardChoice],
        last_id: Option<&str>,
    ) -> Result<Option<usize>, CatalogError> {
        self.log.push(format!(
            "SELECT {label} :: {}",
            choices
                .iter()
                .map(|c| c.label.as_str())
                .collect::<Vec<_>>()
                .join(" | ")
        ));
        match self.next_answer(label)?.as_deref() {
            None => Ok(last_id.and_then(|v| find_choice_index(choices, v))),
            Some("-" | "skip") => Ok(None),
            Some(s) => {
                if s.parse::<usize>().is_ok() {
                    return parse_select_index(s, choices.len());
                }
                find_choice_index(choices, s).map(Some).ok_or_else(|| {
                    CatalogError::InvalidSemantics(format!("scripted select miss {s:?}"))
                })
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
struct DatasetDraft {
    dim_names: Vec<String>,
    attrs: BTreeMap<String, String>,
}

/// Run the interactive wizard on a mutable `.tet`.
///
/// # Errors
///
/// Package targets, missing datasets, I/O, or apply validation.
pub fn run_semantics_wizard(
    tet_path: &Path,
    opts: &WizardOptions,
    io: &mut dyn WizardIo,
) -> Result<WizardResult, CatalogError> {
    ensure_mutable_tet(tet_path, "semantics wizard")?;
    let summary = read_summary_from_path(tet_path)?;
    if summary.datasets.is_empty() {
        return Err(CatalogError::InvalidSemantics(
            "file has no datasets".to_owned(),
        ));
    }

    let dataset = resolve_dataset(&summary, opts.dataset.as_deref(), io)?;
    let ds = summary
        .datasets
        .iter()
        .find(|d| d.name == dataset)
        .ok_or_else(|| {
            CatalogError::InvalidSemantics(format!(
                "resolved dataset {dataset} missing from catalog"
            ))
        })?;

    print_banner(io, tet_path, &dataset, ds)?;

    let mut draft = seed_draft(&summary, &dataset, ds.shape.len());
    loop {
        draft = collect_draft(ds, &draft, io)?;
        io.println("")?;
        io.println(&format_confirm(&dataset, ds, &draft))?;
        if confirm_write(io)? {
            break;
        }
        io.println("")?;
        io.println(WizardSection::Redo.title())?;
    }

    let toml_text = render_wizard_toml(&dataset, &draft);
    if let Some(parent) = opts.toml_out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(&opts.toml_out, &toml_text)?;
    io.println(&format!("wrote {}", opts.toml_out.display()))?;
    let _ = parse_semantics_toml(&toml_text)?;

    let apply = if opts.apply {
        let report = apply_semantics_toml(tet_path, &opts.toml_out)?;
        io.println(format_apply_report(tet_path, &report).trim_end())?;
        Some(report)
    } else {
        io.println("skipped apply (--no-apply)")?;
        None
    };

    Ok(WizardResult {
        dataset,
        toml_path: opts.toml_out.clone(),
        apply,
    })
}

fn print_banner(
    io: &mut dyn WizardIo,
    tet_path: &Path,
    dataset: &str,
    ds: &DatasetRecordV1,
) -> Result<(), CatalogError> {
    io.println("── tet semantics wizard ──")?;
    io.println(&format!("file     {}", tet_path.display()))?;
    io.println(&format!(
        "dataset  {dataset} · {} · {} · rank {}",
        dtype_label(ds.dtype),
        shape_label(&ds.shape),
        ds.shape.len()
    ))?;
    io.println("empty keeps [last] · - clears/skips · fold-inert (attrs do not change query)")
}

fn confirm_write(io: &mut dyn WizardIo) -> Result<bool, CatalogError> {
    match io
        .prompt(
            "Confirm and write? [Y/n]  (n = redo with last selection)",
            None,
        )?
        .as_deref()
    {
        None | Some("y" | "Y" | "yes" | "YES") => Ok(true),
        Some("n" | "N" | "no" | "NO") => Ok(false),
        Some(other) => Err(CatalogError::InvalidSemantics(format!(
            "expected y/n, got {other:?}"
        ))),
    }
}

fn resolve_dataset(
    summary: &TetFileSummaryV1,
    forced: Option<&str>,
    io: &mut dyn WizardIo,
) -> Result<String, CatalogError> {
    if let Some(name) = forced {
        if !summary.datasets.iter().any(|d| d.name == name) {
            return Err(CatalogError::InvalidSemantics(format!(
                "dataset {name} not found in catalog"
            )));
        }
        return Ok(name.to_owned());
    }
    if summary.datasets.len() == 1 {
        return Ok(summary.datasets[0].name.clone());
    }
    let choices: Vec<WizardChoice> = summary
        .datasets
        .iter()
        .map(|d| {
            WizardChoice::new(
                d.name.clone(),
                format!("{}  shape={}", d.name, shape_label(&d.shape)),
            )
        })
        .collect();
    let i = io
        .select("dataset", &choices, Some(&summary.datasets[0].name))?
        .unwrap_or(0);
    Ok(summary.datasets[i].name.clone())
}

fn seed_draft(summary: &TetFileSummaryV1, dataset: &str, rank: usize) -> DatasetDraft {
    let meta = summary.metadata.datasets.get(dataset);
    let mut dim_names = vec![String::new(); rank];
    if let Some(names) = meta.and_then(|m| m.dim_names.as_ref()) {
        for (i, name) in names.iter().enumerate().take(rank) {
            dim_names[i].clone_from(name);
        }
    }
    DatasetDraft {
        dim_names,
        attrs: meta.map(|m| m.attrs.clone()).unwrap_or_default(),
    }
}

fn collect_draft(
    ds: &DatasetRecordV1,
    previous: &DatasetDraft,
    io: &mut dyn WizardIo,
) -> Result<DatasetDraft, CatalogError> {
    let dim_names = prompt_dim_names(ds, previous, io)?;
    let (mut attrs, profile) = prompt_profile_and_attrs(ds, &dim_names, previous, io)?;
    prompt_extras(profile, &mut attrs, io)?;
    Ok(DatasetDraft { dim_names, attrs })
}

fn print_block(io: &mut dyn WizardIo, block: &str) -> Result<(), CatalogError> {
    for line in block.lines() {
        io.println(line)?;
    }
    Ok(())
}

fn prompt_dim_names(
    ds: &DatasetRecordV1,
    previous: &DatasetDraft,
    io: &mut dyn WizardIo,
) -> Result<Vec<String>, CatalogError> {
    let rank = ds.shape.len();
    let mut dim_names = Vec::with_capacity(rank);
    io.println("")?;
    print_block(io, &format_axes_header(ds))?;
    for i in 0..rank {
        let size = ds.shape[i];
        let def = previous.dim_names.get(i).filter(|s| !s.is_empty());
        loop {
            let ans = io.prompt(&format!("axis {i}  size {size}"), def.map(String::as_str))?;
            let name = match ans {
                None => def.cloned().unwrap_or_default(),
                Some(s) if s == "-" => String::new(),
                Some(s) => s,
            };
            if name.is_empty() {
                io.println("  need a name (not empty)")?;
                continue;
            }
            dim_names.push(name);
            break;
        }
    }
    Ok(dim_names)
}

fn prompt_profile_and_attrs(
    ds: &DatasetRecordV1,
    dim_names: &[String],
    previous: &DatasetDraft,
    io: &mut dyn WizardIo,
) -> Result<(BTreeMap<String, String>, Option<WizardProfile>), CatalogError> {
    let mut attrs = BTreeMap::new();
    io.println("")?;
    print_block(io, &format_profile_menu())?;

    let profile_choices: Vec<WizardChoice> = WizardProfile::ALL
        .iter()
        .map(|p| WizardChoice::new(p.id(), p.select_label()))
        .collect();
    let profile_last = previous
        .attrs
        .get(ATTR_SEMANTICS_PROFILE)
        .map(String::as_str);
    let profile =
        if let Some(i) = io.select(ATTR_SEMANTICS_PROFILE, &profile_choices, profile_last)? {
            let p = WizardProfile::ALL[i];
            attrs.insert(ATTR_SEMANTICS_PROFILE.to_owned(), p.id().to_owned());
            Some(p)
        } else {
            None
        };
    let keys = profile
        .map(WizardProfile::recommended_attr_keys)
        .unwrap_or_default();

    if let Some(profile) = profile {
        io.println("")?;
        io.println(&format_included_policy(profile))?;
        io.println("")?;
        print_block(io, &format_recommended_header(profile))?;
        for key in &keys {
            if *key == ATTR_TIME_AXIS {
                prompt_time_axis(ds, dim_names, previous, &mut attrs, io)?;
                continue;
            }
            let prev = previous.attrs.get(*key).map(String::as_str);
            let label = attr_prompt_label(profile, key, prev.is_none());
            keep_or_set_attr(&mut attrs, key, io.prompt(&label, prev)?.as_deref(), prev);
        }
    }

    for (k, v) in &previous.attrs {
        if k == ATTR_SEMANTICS_PROFILE || attrs.contains_key(k) || keys.contains(&k.as_str()) {
            continue;
        }
        attrs.insert(k.clone(), v.clone());
    }
    Ok((attrs, profile))
}

fn keep_or_set_attr(
    attrs: &mut BTreeMap<String, String>,
    key: &str,
    ans: Option<&str>,
    prev: Option<&str>,
) {
    match ans {
        None => {
            if let Some(v) = prev {
                attrs.insert(key.to_owned(), v.to_owned());
            }
        }
        Some("-") => {}
        Some(s) => {
            attrs.insert(key.to_owned(), s.to_owned());
        }
    }
}

fn attr_prompt_label(profile: WizardProfile, key: &str, show_hint: bool) -> String {
    match (show_hint, profile.attr_hint(key)) {
        (true, Some(hint)) => format!("{key}  (e.g. {hint})"),
        _ => key.to_owned(),
    }
}

fn prompt_extras(
    profile: Option<WizardProfile>,
    attrs: &mut BTreeMap<String, String>,
    io: &mut dyn WizardIo,
) -> Result<(), CatalogError> {
    io.println("")?;
    io.println(WizardSection::Extras.title())?;
    io.println(extras_help(profile))?;
    io.println("Empty line finishes.")?;
    while let Some(raw) = io.prompt("  extra", None)? {
        if raw == "-" {
            continue;
        }
        let Some((k, v)) = raw.split_once('=') else {
            io.println("    expected KEY=VAL")?;
            continue;
        };
        let k = k.trim();
        let v = v.trim();
        if k.is_empty() || v.is_empty() {
            io.println("    KEY and VAL must be non-empty")?;
            continue;
        }
        attrs.insert(k.to_owned(), v.to_owned());
    }
    Ok(())
}

fn extras_help(profile: Option<WizardProfile>) -> &'static str {
    match profile {
        Some(p) => p.extras_help(),
        None => "Optional KEY=VAL declare attrs (no profile — open minting).",
    }
}

fn format_axes_header(ds: &DatasetRecordV1) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "{}", section_title(WizardSection::Axes.title()));
    let _ = writeln!(
        out,
        "{}",
        dim_line("Name each axis in rank order (all required).")
    );
    let rows: Vec<Vec<String>> = ds
        .shape
        .iter()
        .enumerate()
        .map(|(i, size)| vec![i.to_string(), size.to_string()])
        .collect();
    out.push_str(&render_table(&["axis", "size"], &rows));
    out
}

fn profile_includes_list(profile: WizardProfile) -> String {
    profile.recommended_attr_keys().join(", ")
}

fn format_profile_menu() -> String {
    let mut out = String::new();
    let _ = writeln!(out, "{}", section_title(WizardSection::Profile.title()));
    let _ = writeln!(
        out,
        "{}",
        dim_line("Soft contract for callers — not a crate runtime mode.")
    );
    let _ = writeln!(
        out,
        "{}",
        dim_line("Pick stamps semantics_profile; rate formulas go in the TOML only.")
    );
    let rows: Vec<Vec<String>> = WizardProfile::ALL
        .iter()
        .enumerate()
        .map(|(i, p)| {
            vec![
                (i + 1).to_string(),
                p.id().to_owned(),
                p.blurb().to_owned(),
                profile_includes_list(*p),
            ]
        })
        .collect();
    out.push_str(&render_table(
        &["#", "id", "about", "includes (attrs)"],
        &rows,
    ));
    let _ = writeln!(out, "{}", dim_line("↑↓ pick · Enter · Esc/(skip) = none"));
    out
}

fn format_recommended_header(profile: WizardProfile) -> String {
    let title = WizardSection::Recommended.titled(profile.id());
    let mut out = String::new();
    let _ = writeln!(out, "{}", section_title(&title));
    let _ = writeln!(out, "{}", dim_line(profile.attr_help()));
    let _ = writeln!(
        out,
        "{}",
        dim_line(&format!("walk keys: {}", profile_includes_list(profile)))
    );
    let _ = writeln!(
        out,
        "{}",
        dim_line("Enter keeps last · - clears · hints are examples only")
    );
    out
}

fn prompt_time_axis(
    ds: &DatasetRecordV1,
    dim_names: &[String],
    previous: &DatasetDraft,
    attrs: &mut BTreeMap<String, String>,
    io: &mut dyn WizardIo,
) -> Result<(), CatalogError> {
    let rank = ds.shape.len();
    if rank == 0 {
        return Ok(());
    }
    let choices: Vec<WizardChoice> = (0..rank)
        .map(|i| {
            let name = dim_names
                .get(i)
                .filter(|s| !s.is_empty())
                .map_or("?", String::as_str);
            WizardChoice::new(i.to_string(), format!("{i}  {name}  size {}", ds.shape[i]))
        })
        .collect();
    let last = previous.attrs.get(ATTR_TIME_AXIS).map(String::as_str);
    if let Some(i) = io.select(ATTR_TIME_AXIS, &choices, last)? {
        attrs.insert(ATTR_TIME_AXIS.to_owned(), i.to_string());
    }
    Ok(())
}

fn format_confirm(dataset: &str, ds: &DatasetRecordV1, draft: &DatasetDraft) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "{}", section_title(WizardSection::Confirm.title()));
    out.push_str(&render_table(
        &["field", "value"],
        &[
            vec!["dataset".into(), dataset.to_owned()],
            vec!["dtype".into(), dtype_label(ds.dtype).to_owned()],
            vec!["shape".into(), shape_label(&ds.shape)],
            vec![
                "dim_names".into(),
                format!("[{}]", draft.dim_names.join(", ")),
            ],
        ],
    ));
    if draft.attrs.is_empty() {
        let _ = writeln!(out, "{}", dim_line("attrs  (none)"));
    } else {
        let rows: Vec<Vec<String>> = draft
            .attrs
            .iter()
            .map(|(k, v)| vec![k.clone(), v.clone()])
            .collect();
        out.push_str(&render_table(&["attr", "value"], &rows));
    }
    append_included_policy(&mut out, draft);
    out
}

fn append_included_policy(out: &mut String, draft: &DatasetDraft) {
    if let Some(profile) = draft
        .attrs
        .get(ATTR_SEMANTICS_PROFILE)
        .and_then(|s| WizardProfile::from_id(s))
    {
        out.push_str(&format_included_policy(profile));
    }
}

fn format_included_policy(profile: WizardProfile) -> String {
    let id = profile.id();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{}",
        section_title(&WizardSection::IncludedPolicy.titled(id))
    );
    let _ = writeln!(
        out,
        "{}",
        dim_line("TOML only · not stamped · not editable here")
    );
    let formulas = profile_policy_formulas(id);
    if formulas.is_empty() {
        let _ = writeln!(
            out,
            "{}",
            dim_line("(no rate formulas — profile table still recorded in the TOML)")
        );
    } else {
        let rows: Vec<Vec<String>> = formulas
            .into_iter()
            .map(|(k, v)| vec![k.to_owned(), v])
            .collect();
        out.push_str(&render_table(&["formula", "expression"], &rows));
    }
    out
}

fn profile_policy_formulas(profile_id: &str) -> Vec<(&'static str, String)> {
    let Some(rate) = rate_attr_for_profile(profile_id) else {
        return Vec::new();
    };
    default_rate_formula_pairs(rate).into_iter().collect()
}

fn render_wizard_toml(dataset: &str, draft: &DatasetDraft) -> String {
    let mut out = String::new();
    out.push_str("# Generated by `tet semantics wizard`.\n");
    out.push_str(
        "# `tet semantics apply` stamps dim_names and attrs. Profiles and units stay here.\n\n",
    );
    out.push_str("schema_version = 1\n\n");
    let facts = DeclareFacts {
        dim_names: Some(draft.dim_names.clone()),
        attrs: draft.attrs.clone(),
    };
    render_dataset_declare_toml(&mut out, dataset, &facts, &[]);

    if let Some(id) = draft.attrs.get(ATTR_SEMANTICS_PROFILE) {
        let canonical = normalize_profile_id(id);
        out.push_str(&dotted_table("profiles", canonical));
        out.push('\n');
        let formulas = profile_policy_formulas(canonical);
        if formulas.is_empty() {
            out.push_str("# Policy formulas stay in this file (fold-inert).\n");
        } else {
            for (k, v) in formulas {
                let _ = writeln!(out, "{k} = {}", toml_string(&v));
            }
        }
        out.push('\n');
    }
    out
}

fn shape_label(shape: &[u64]) -> String {
    fmt_shape_dims(shape, "")
}

fn find_choice_index(choices: &[WizardChoice], needle: &str) -> Option<usize> {
    choices
        .iter()
        .position(|c| c.id == needle)
        .or_else(|| choices.iter().position(|c| c.label == needle))
}

#[cfg(any(test, not(feature = "tetration-cli")))]
fn parse_select_index(raw: &str, len: usize) -> Result<Option<usize>, CatalogError> {
    let i: usize = raw
        .parse()
        .map_err(|_| CatalogError::InvalidSemantics(format!("bad select index {raw:?}")))?;
    if i < len {
        Ok(Some(i))
    } else {
        Err(CatalogError::InvalidSemantics(format!(
            "select index {i} out of range"
        )))
    }
}

#[cfg(not(feature = "tetration-cli"))]
fn io_err(e: io::Error) -> CatalogError {
    CatalogError::Io(e)
}

#[cfg(feature = "tetration-cli")]
fn dialoguer_err(e: &dialoguer::Error) -> CatalogError {
    CatalogError::InvalidSemantics(format!("wizard input: {e}"))
}

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

use super::apply::{
    SemanticsApplyReport, apply_semantics_toml, format_apply_report, parse_semantics_toml,
};
use super::declare::{bare_or_quoted, dotted_table, dtype_label, toml_string};
use super::profiles::{
    DOCUMENTED_PROFILES, normalize_profile_id, rate_attr_for_profile, recommended_keys_for_profile,
};
use crate::catalog::{CatalogError, DatasetRecordV1, TetFileSummaryV1, read_tet_summary_v1};

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WizardResult {
    pub dataset: String,
    pub toml_path: PathBuf,
    pub apply: Option<SemanticsApplyReport>,
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

    /// Arrow-key (or scripted) pick among `items`. `None` = skip.
    ///
    /// `last` is matched against `items` to set the initial cursor when possible.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] on I/O failure, bad index, or exhausted script.
    fn select(
        &mut self,
        label: &str,
        items: &[String],
        last: Option<&str>,
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
        items: &[String],
        last: Option<&str>,
    ) -> Result<Option<usize>, CatalogError> {
        if items.is_empty() {
            return Ok(None);
        }
        #[cfg(feature = "tetration-cli")]
        {
            use dialoguer::{Select, theme::ColorfulTheme};
            let mut opts = items.to_vec();
            let skip_idx = opts.len();
            opts.push("(skip)".to_owned());
            let default = last.and_then(|v| find_select_index(items, v)).unwrap_or(0);
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
            for (i, it) in items.iter().enumerate() {
                self.println(&format!("  {i}. {it}"))?;
            }
            match self.prompt("  pick index", last)?.as_deref() {
                None => Ok(last.and_then(|v| find_select_index(items, v))),
                Some("-" | "skip") => Ok(None),
                Some(s) => parse_select_index(s, items.len()),
            }
        }
    }
}

/// Queued answers for unit tests (`None` in the queue = empty line).
#[derive(Debug, Default)]
pub struct ScriptedWizardIo {
    answers: Vec<Option<String>>,
    cursor: usize,
    pub log: Vec<String>,
}

impl ScriptedWizardIo {
    #[must_use]
    pub fn new(answers: Vec<Option<String>>) -> Self {
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
        items: &[String],
        last: Option<&str>,
    ) -> Result<Option<usize>, CatalogError> {
        self.log
            .push(format!("SELECT {label} :: {}", items.join(" | ")));
        match self.next_answer(label)?.as_deref() {
            None => Ok(last.and_then(|v| find_select_index(items, v))),
            Some("-" | "skip") => Ok(None),
            Some(s) => {
                if s.parse::<usize>().is_ok() {
                    return parse_select_index(s, items.len());
                }
                find_select_index(items, s).map(Some).ok_or_else(|| {
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
    ensure_mutable_tet(tet_path)?;
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
        io.println("── redo ──  last selection shown in [brackets]; empty keeps it, - clears")?;
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
    let items: Vec<String> = summary
        .datasets
        .iter()
        .map(|d| format!("{}  shape={}", d.name, shape_label(&d.shape)))
        .collect();
    let i = io
        .select("dataset", &items, Some(&summary.datasets[0].name))?
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
    let mut attrs = prompt_profile_and_attrs(ds, &dim_names, previous, io)?;
    let profile_for_extras = attrs.get("semantics_profile").cloned();
    prompt_extras(profile_for_extras.as_deref(), &mut attrs, io)?;
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
) -> Result<BTreeMap<String, String>, CatalogError> {
    let mut attrs = BTreeMap::new();
    io.println("")?;
    print_block(io, &format_profile_menu())?;

    let profile_items: Vec<String> = DOCUMENTED_PROFILES
        .iter()
        .map(|p| format!("{} — {}", p.id, profile_blurb(p.id)))
        .collect();
    let profile_last = previous.attrs.get("semantics_profile").map(String::as_str);
    if let Some(i) = io.select("semantics_profile", &profile_items, profile_last)? {
        attrs.insert(
            "semantics_profile".to_owned(),
            DOCUMENTED_PROFILES[i].id.to_owned(),
        );
    }

    let profile_id = attrs
        .get("semantics_profile")
        .map(|s| normalize_profile_id(s).to_owned());
    let keys = profile_id
        .as_deref()
        .map(recommended_attr_keys)
        .unwrap_or_default();

    if let Some(id) = profile_id.as_deref() {
        io.println("")?;
        io.println(&format_included_policy(id))?;
        io.println("")?;
        print_block(io, &format_recommended_header(id))?;
        for key in &keys {
            if *key == "time_axis" {
                prompt_time_axis(ds, dim_names, previous, &mut attrs, io)?;
                continue;
            }
            let prev = previous.attrs.get(*key).map(String::as_str);
            let label = attr_prompt_label(id, key, prev.is_none());
            keep_or_set_attr(&mut attrs, key, io.prompt(&label, prev)?.as_deref(), prev);
        }
    }

    for (k, v) in &previous.attrs {
        if k == "semantics_profile" || attrs.contains_key(k) || keys.contains(&k.as_str()) {
            continue;
        }
        attrs.insert(k.clone(), v.clone());
    }
    Ok(attrs)
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

fn attr_prompt_label(profile: &str, key: &str, show_hint: bool) -> String {
    match (show_hint, attr_hint(profile, key)) {
        (true, Some(hint)) => format!("{key}  (e.g. {hint})"),
        _ => key.to_owned(),
    }
}

fn prompt_extras(
    profile_id: Option<&str>,
    attrs: &mut BTreeMap<String, String>,
    io: &mut dyn WizardIo,
) -> Result<(), CatalogError> {
    let profile_id = profile_id.map(normalize_profile_id);
    io.println("")?;
    io.println("── extras ──")?;
    io.println(extras_help(profile_id))?;
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

fn extras_help(profile_id: Option<&str>) -> &'static str {
    match profile_id {
        Some("media.v1") => {
            "Optional KEY=VAL beyond the list (e.g. pixel_unit=px, color_space=sRGB)."
        }
        Some("timeseries.v1") => "Optional KEY=VAL beyond the list (e.g. start_time=ISO-8601).",
        Some("cf_lite.v1") => "Optional KEY=VAL beyond the list (e.g. uncertainty_kind=1sigma).",
        Some("prov_log.v1") => {
            "Optional KEY=VAL; provenance prefers history rows over inventing attrs."
        }
        Some(_) => "Optional KEY=VAL for any other declare attrs you want stamped.",
        None => "Optional KEY=VAL declare attrs (no profile — open minting).",
    }
}

fn format_axes_header(ds: &DatasetRecordV1) -> String {
    #[cfg(feature = "tetration-cli")]
    {
        let mut out = section_open("── axes ──");
        push_dim(&mut out, "Name each axis in rank order (all required).");
        let rows: Vec<Vec<String>> = ds
            .shape
            .iter()
            .enumerate()
            .map(|(i, size)| vec![i.to_string(), size.to_string()])
            .collect();
        push_table(&mut out, &["axis", "size"], &rows);
        out
    }
    #[cfg(not(feature = "tetration-cli"))]
    {
        let mut out = String::new();
        let _ = writeln!(out, "── axes ──");
        let _ = writeln!(out, "Name each axis in rank order (all required).");
        for (i, size) in ds.shape.iter().enumerate() {
            let _ = writeln!(out, "  axis {i}  size {size}");
        }
        out
    }
}

fn profile_blurb(id: &str) -> &'static str {
    match id {
        "media.v1" => "video / frame stacks",
        "timeseries.v1" => "sampled signals",
        "cf_lite.v1" => "CF-lite grids",
        "prov_log.v1" => "provenance via history",
        _ => "",
    }
}

fn recommended_attr_keys(id: &str) -> Vec<&'static str> {
    recommended_keys_for_profile(id)
        .iter()
        .copied()
        .filter(|k| *k != "semantics_profile")
        .collect()
}

fn profile_includes_list(id: &str) -> String {
    recommended_attr_keys(id).join(", ")
}

fn format_profile_menu() -> String {
    #[cfg(feature = "tetration-cli")]
    {
        let mut out = section_open("── profile ──");
        push_dim(
            &mut out,
            "Soft contract for callers — not a crate runtime mode.",
        );
        push_dim(
            &mut out,
            "Pick stamps semantics_profile; rate formulas go in the TOML only.",
        );
        let rows: Vec<Vec<String>> = DOCUMENTED_PROFILES
            .iter()
            .enumerate()
            .map(|(i, p)| {
                vec![
                    (i + 1).to_string(),
                    p.id.to_owned(),
                    profile_blurb(p.id).to_owned(),
                    profile_includes_list(p.id),
                ]
            })
            .collect();
        push_table(&mut out, &["#", "id", "about", "includes (attrs)"], &rows);
        push_dim(&mut out, "↑↓ pick · Enter · Esc/(skip) = none");
        out
    }
    #[cfg(not(feature = "tetration-cli"))]
    {
        let mut out = String::new();
        let _ = writeln!(out, "── profile ──");
        let _ = writeln!(
            out,
            "Doc profile id — soft contract for callers (not a crate runtime mode)."
        );
        let _ = writeln!(
            out,
            "Picking one stamps semantics_profile; rate formulas stay in the TOML."
        );
        for (i, p) in DOCUMENTED_PROFILES.iter().enumerate() {
            let _ = writeln!(out, "  {}. {}", i, p.id);
            let _ = writeln!(out, "     {}", profile_blurb(p.id));
            let _ = writeln!(out, "     includes: {}", profile_includes_list(p.id));
        }
        let _ = writeln!(out, "  (index · id · empty/skip = none)");
        out
    }
}

fn profile_attr_help(id: &str) -> &'static str {
    match id {
        "media.v1" => {
            "Facts for a frame stack: kind, rate along time_axis, optional frame_rate_unit (not fixed to Hz)."
        }
        "timeseries.v1" => {
            "Facts for a continuous signal: sample_rate (+ unit), time_axis, physical units."
        }
        "cf_lite.v1" => {
            "CF-style labels only — units / standard_name / crs; no climate engine in the crate."
        }
        "prov_log.v1" => "Profile id alone is enough here; prefer history rows for agent/activity.",
        _ => "Recommended keys for this profile id (advisory; open minting allows others).",
    }
}

fn format_recommended_header(id: &str) -> String {
    #[cfg(feature = "tetration-cli")]
    {
        let mut out = section_open(&format!("── recommended attrs ({id}) ──"));
        push_dim(&mut out, profile_attr_help(id));
        push_dim(
            &mut out,
            &format!("walk keys: {}", profile_includes_list(id)),
        );
        push_dim(
            &mut out,
            "Enter keeps last · - clears · hints are examples only",
        );
        out
    }
    #[cfg(not(feature = "tetration-cli"))]
    {
        let mut out = String::new();
        let _ = writeln!(out, "── recommended attrs ({id}) ──");
        let _ = writeln!(out, "{}", profile_attr_help(id));
        let _ = writeln!(out, "walk keys: {}", profile_includes_list(id));
        let _ = writeln!(out, "Enter keeps last · - clears · hints are examples only");
        out
    }
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
    let items: Vec<String> = (0..rank)
        .map(|i| {
            let name = dim_names
                .get(i)
                .filter(|s| !s.is_empty())
                .map_or("?", String::as_str);
            format!("{i}  {name}  size {}", ds.shape[i])
        })
        .collect();
    let last = previous.attrs.get("time_axis").map(String::as_str);
    if let Some(i) = io.select("time_axis", &items, last)? {
        attrs.insert("time_axis".to_owned(), i.to_string());
    }
    Ok(())
}

fn attr_hint(profile: &str, key: &str) -> Option<&'static str> {
    match (profile, key) {
        ("media.v1", "media_kind") => Some("video"),
        ("media.v1", "frame_rate") => Some("30"),
        ("media.v1", "frame_rate_unit") | ("timeseries.v1", "sample_rate_unit") => Some("Hz"),
        ("media.v1" | "timeseries.v1", "time_axis") => Some("0"),
        ("timeseries.v1", "media_kind") => Some("timeseries"),
        ("timeseries.v1", "sample_rate") => Some("30000"),
        ("timeseries.v1", "units") => Some("1"),
        ("cf_lite.v1", "units") => Some("K"),
        ("cf_lite.v1", "standard_name") => Some("air_temperature"),
        ("cf_lite.v1", "crs") => Some("EPSG:4326"),
        _ => None,
    }
}

fn format_confirm(dataset: &str, ds: &DatasetRecordV1, draft: &DatasetDraft) -> String {
    #[cfg(feature = "tetration-cli")]
    {
        use console::style;
        let mut out = section_open("── confirm ──");
        push_table(
            &mut out,
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
        );
        if draft.attrs.is_empty() {
            let _ = writeln!(out, "{}", style("attrs  (none)").dim());
        } else {
            let rows: Vec<Vec<String>> = draft
                .attrs
                .iter()
                .map(|(k, v)| vec![k.clone(), v.clone()])
                .collect();
            push_table(&mut out, &["attr", "value"], &rows);
        }
        append_included_policy(&mut out, draft);
        out
    }
    #[cfg(not(feature = "tetration-cli"))]
    {
        let mut out = String::new();
        let _ = writeln!(out, "── confirm ──");
        let _ = writeln!(out, "dataset    {dataset}");
        let _ = writeln!(out, "shape      {}", shape_label(&ds.shape));
        let _ = writeln!(out, "dim_names  [{}]", draft.dim_names.join(", "));
        if draft.attrs.is_empty() {
            let _ = writeln!(out, "attrs      (none)");
        } else {
            let _ = writeln!(out, "attrs");
            for (k, v) in &draft.attrs {
                let _ = writeln!(out, "  {k} = {v}");
            }
        }
        append_included_policy(&mut out, draft);
        out
    }
}

fn append_included_policy(out: &mut String, draft: &DatasetDraft) {
    if let Some(id) = draft
        .attrs
        .get("semantics_profile")
        .map(|s| normalize_profile_id(s))
    {
        out.push_str(&format_included_policy(id));
    }
}

fn format_included_policy(profile_id: &str) -> String {
    #[cfg(feature = "tetration-cli")]
    {
        let mut out = section_open(&format!("── included policy ({profile_id}) ──"));
        push_dim(&mut out, "TOML only · not stamped · not editable here");
        let formulas = profile_policy_formulas(profile_id);
        if formulas.is_empty() {
            push_dim(
                &mut out,
                "(no rate formulas — profile table still recorded in the TOML)",
            );
        } else {
            let rows: Vec<Vec<String>> = formulas
                .into_iter()
                .map(|(k, v)| vec![k.to_owned(), v])
                .collect();
            push_table(&mut out, &["formula", "expression"], &rows);
        }
        out
    }
    #[cfg(not(feature = "tetration-cli"))]
    {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "── included policy ({profile_id}) ──  TOML only · not stamped · not editable here"
        );
        let formulas = profile_policy_formulas(profile_id);
        if formulas.is_empty() {
            let _ = writeln!(
                out,
                "  (no rate formulas for this profile — table still recorded in the TOML)"
            );
        } else {
            for (k, v) in formulas {
                let _ = writeln!(out, "  {k} = {v}");
            }
        }
        out
    }
}

fn profile_policy_formulas(profile_id: &str) -> Vec<(&'static str, String)> {
    let Some(rate) = rate_attr_for_profile(profile_id) else {
        return Vec::new();
    };
    vec![
        ("index_to_seconds", format!("index / {rate}")),
        ("seconds_to_index", format!("floor(seconds * {rate})")),
    ]
}

fn render_wizard_toml(dataset: &str, draft: &DatasetDraft) -> String {
    let mut out = String::new();
    out.push_str("# Generated by `tet semantics wizard`.\n");
    out.push_str(
        "# `tet semantics apply` stamps dim_names and attrs. Profiles and units stay here.\n\n",
    );
    out.push_str("schema_version = 1\n\n");
    let _ = writeln!(out, "[datasets.{}]", bare_or_quoted(dataset));
    out.push_str("dim_names = [");
    for (i, name) in draft.dim_names.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&toml_string(name));
    }
    out.push_str("]\n");
    if !draft.attrs.is_empty() {
        let _ = writeln!(out, "[datasets.{}.attrs]", bare_or_quoted(dataset));
        for (k, v) in &draft.attrs {
            out.push_str(&bare_or_quoted(k));
            out.push_str(" = ");
            out.push_str(&toml_string(v));
            out.push('\n');
        }
    }
    out.push('\n');

    if let Some(id) = draft.attrs.get("semantics_profile") {
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
    shape
        .iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join("×")
}

fn find_select_index(items: &[String], needle: &str) -> Option<usize> {
    items
        .iter()
        .position(|it| it == needle || it.starts_with(&format!("{needle} ")))
}

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

fn ensure_mutable_tet(path: &Path) -> Result<(), CatalogError> {
    if path.is_dir() || path.join("PROTOCOL").is_file() || ancestor_is_package(path) {
        return Err(CatalogError::InvalidSemantics(
            "semantics wizard targets a mutable .tet file; a sealed .tet.d generation is immutable"
                .to_owned(),
        ));
    }
    Ok(())
}

fn ancestor_is_package(path: &Path) -> bool {
    let mut cur = path.parent();
    while let Some(dir) = cur {
        if dir.join("PROTOCOL").is_file() {
            return true;
        }
        cur = dir.parent();
    }
    false
}

fn read_summary_from_path(path: &Path) -> Result<TetFileSummaryV1, CatalogError> {
    let data = fs::read(path)?;
    read_tet_summary_v1(&data)
}

#[cfg(feature = "tetration-cli")]
fn section_open(title: &str) -> String {
    use console::style;
    let mut out = String::new();
    let _ = writeln!(out, "{}", style(title).bold());
    out
}

#[cfg(feature = "tetration-cli")]
fn push_dim(out: &mut String, msg: &str) {
    use console::style;
    let _ = writeln!(out, "{}", style(msg).dim());
}

#[cfg(feature = "tetration-cli")]
fn push_table(out: &mut String, headers: &[&str], rows: &[Vec<String>]) {
    use comfy_table::{Attribute, Cell, ContentArrangement, Table, presets::UTF8_FULL};
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(
            headers
                .iter()
                .map(|h| Cell::new(h).add_attribute(Attribute::Bold))
                .collect::<Vec<_>>(),
        );
    for row in rows {
        table.add_row(row.iter().map(Cell::new).collect::<Vec<_>>());
    }
    let _ = writeln!(out, "{table}");
}

#[cfg(not(feature = "tetration-cli"))]
fn io_err(e: io::Error) -> CatalogError {
    CatalogError::Io(e)
}

#[cfg(feature = "tetration-cli")]
fn dialoguer_err(e: &dialoguer::Error) -> CatalogError {
    CatalogError::InvalidSemantics(format!("wizard input: {e}"))
}

//! Shared pretty-print helpers for `tet` human text (`comfy-table` / `console` / `indicatif`).
//!
//! Used by `query::cli` (info / query tables), `catalog::semantics::wizard`, and the `tet`
//! binary (`tet run` step lines, convert progress bar). Gated on `tetration-cli` so library
//! builds without the feature keep plain ASCII. Only the `pub` items below are for the binary;
//! the module is `#[doc(hidden)]` and not part of the supported library API.

use std::fmt::Write as _;

/// Bold section title (or plain when CLI pretty is off).
pub(crate) fn section_title(title: &str) -> String {
    #[cfg(feature = "tetration-cli")]
    {
        use console::style;
        style(title).bold().to_string()
    }
    #[cfg(not(feature = "tetration-cli"))]
    {
        title.to_owned()
    }
}

/// Dim helper line.
pub(crate) fn dim_line(msg: &str) -> String {
    #[cfg(feature = "tetration-cli")]
    {
        use console::style;
        style(msg).dim().to_string()
    }
    #[cfg(not(feature = "tetration-cli"))]
    {
        msg.to_owned()
    }
}

/// Data / kv table (`UTF8_FULL` — same style as info / query / wizard).
pub(crate) fn render_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    #[cfg(feature = "tetration-cli")]
    {
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
        format!("{table}\n")
    }
    #[cfg(not(feature = "tetration-cli"))]
    {
        let mut out = String::new();
        let _ = writeln!(out, "  {}", headers.join("  "));
        for row in rows {
            let _ = writeln!(out, "  {}", row.join("  "));
        }
        out
    }
}

/// Key/value section as a real table (same preset as [`render_table`]).
pub(crate) fn render_kv_section(title: &str, rows: &[(String, String)]) -> String {
    if rows.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    let _ = writeln!(out, "{}", section_title(title));
    let table_rows: Vec<Vec<String>> = rows
        .iter()
        .map(|(k, v)| vec![k.clone(), v.clone()])
        .collect();
    out.push_str(&render_table(&["field", "value"], &table_rows));
    out.push('\n');
    out
}

/// Join shape dims with `×`; use `empty` when `shape` is empty.
///
/// Callers pick the empty token intentionally (do not unify):
/// - wizard banners: `""`
/// - `tet info` tables: `"-"`
/// - query quiet / stats: `"0"`
pub(crate) fn fmt_shape_dims(shape: &[u64], empty: &str) -> String {
    if shape.is_empty() {
        return empty.to_owned();
    }
    shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("×")
}

// ---------------------------------------------------------------------------------------------
// `tet run` step lines (stderr chrome; the `job` library itself never prints)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum StepTone {
    Planned,
    Ok,
}

fn style_status(tone: StepTone) -> String {
    let text = match tone {
        StepTone::Planned => "plan",
        StepTone::Ok => "ok",
    };
    #[cfg(feature = "tetration-cli")]
    {
        use console::style;
        let styled = style(text).for_stderr();
        match tone {
            StepTone::Planned => styled.dim(),
            StepTone::Ok => styled.green().bold(),
        }
        .to_string()
    }
    #[cfg(not(feature = "tetration-cli"))]
    {
        text.to_owned()
    }
}

/// `[1/3]`, right-aligned to the width of `total` so columns line up.
fn step_counter(index_1based: usize, total: usize) -> String {
    let width = total.to_string().len();
    format!("[{index_1based:>width$}/{total}]")
}

/// Dim stderr text (plain when CLI pretty is off).
fn stderr_dim(text: &str) -> String {
    #[cfg(feature = "tetration-cli")]
    {
        console::style(text).for_stderr().dim().to_string()
    }
    #[cfg(not(feature = "tetration-cli"))]
    {
        text.to_owned()
    }
}

/// Indent every line and dim for nesting under a step header on stderr.
fn indent_stderr_dim(text: &str) -> String {
    text.lines()
        .map(|line| stderr_dim(&format!("    {line}")))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Dry-run line: `[1/3] convert a.csv → a.tet  plan`.
#[must_use]
pub fn step_planned(index_1based: usize, total: usize, description: &str) -> String {
    format!(
        "{} {description}  {}",
        step_counter(index_1based, total),
        style_status(StepTone::Planned)
    )
}

/// Finished-step line: `[1/3] convert a.csv → a.tet  ok  0.12s`.
#[must_use]
pub fn step_ok(index_1based: usize, total: usize, description: &str, elapsed_secs: f64) -> String {
    format!(
        "{} {description}  {}  {}",
        step_counter(index_1based, total),
        style_status(StepTone::Ok),
        stderr_dim(&format!("{elapsed_secs:.2}s"))
    )
}

/// Dim, indented continuation line under a step (notes / non-kv quiet lines).
#[must_use]
pub fn step_note_line(line: &str) -> String {
    indent_stderr_dim(line)
}

/// Parse a quiet `key=value key2=value2 …` line into ordered pairs.
///
/// Values may contain `=` (everything after the first `=`). Tokens without `=` are skipped.
#[must_use]
pub fn parse_kv_line(line: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for token in line.split_whitespace() {
        let Some((k, v)) = token.split_once('=') else {
            continue;
        };
        if k.is_empty() {
            continue;
        }
        out.push((k.to_owned(), v.to_owned()));
    }
    out
}

/// Light kv table for quiet job results (indented under a step).
///
/// [`UTF8_NO_BORDERS`] — not the full box preset. Drops `memory_strategy`.
#[must_use]
pub fn step_kv_table(pairs: &[(String, String)]) -> String {
    let pairs: Vec<_> = pairs
        .iter()
        .filter(|(k, _)| k.as_str() != "memory_strategy")
        .cloned()
        .collect();
    if pairs.is_empty() {
        return String::new();
    }

    #[cfg(feature = "tetration-cli")]
    {
        use comfy_table::{Attribute, Cell, ContentArrangement, Table, presets::UTF8_NO_BORDERS};
        let mut table = Table::new();
        table
            .load_preset(UTF8_NO_BORDERS)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec![
                Cell::new("field").add_attribute(Attribute::Bold),
                Cell::new("value").add_attribute(Attribute::Bold),
            ]);
        for (k, v) in &pairs {
            table.add_row(vec![Cell::new(k), Cell::new(v)]);
        }
        let mut out = indent_stderr_dim(&format!("{table}"));
        out.push('\n');
        out
    }
    #[cfg(not(feature = "tetration-cli"))]
    {
        let mut body = String::new();
        let _ = writeln!(body, "field  value");
        for (k, v) in &pairs {
            let _ = writeln!(body, "{k}  {v}");
        }
        indent_stderr_dim(body.trim_end()) + "\n"
    }
}

/// Chunk progress bar template shared by `tet convert` (and any future job-side bar).
///
/// # Errors
///
/// Template parse failure (cannot happen for the built-in template).
#[cfg(feature = "tetration-cli")]
pub fn chunk_progress_style() -> Result<indicatif::ProgressStyle, indicatif::style::TemplateError> {
    Ok(indicatif::ProgressStyle::with_template(
        "{msg} [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})",
    )?
    .progress_chars("=>-"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_lines_are_ascii_structured() {
        let plain = |s: String| console_strip(&s);
        assert_eq!(
            plain(step_planned(1, 3, "convert a → b")),
            "[1/3] convert a → b  plan"
        );
        assert_eq!(
            plain(step_ok(2, 12, "verify x", 0.5)),
            "[ 2/12] verify x  ok  0.50s"
        );
        assert_eq!(plain(step_note_line("hi")), "    hi");
    }

    #[test]
    fn parse_kv_line_splits_on_first_equals() {
        let pairs = parse_kv_line("path=/a/b.tet status=ok mean=2.5 hint=x=y");
        assert_eq!(
            pairs,
            vec![
                ("path".into(), "/a/b.tet".into()),
                ("status".into(), "ok".into()),
                ("mean".into(), "2.5".into()),
                ("hint".into(), "x=y".into()),
            ]
        );
    }

    #[test]
    fn step_kv_table_is_light_vertical_and_drops_memory_strategy() {
        let plain = console_strip(&step_kv_table(&[
            ("path".into(), "work/a.tet".into()),
            ("status".into(), "ok".into()),
            ("memory_strategy".into(), "streaming_fold".into()),
        ]));
        assert!(plain.contains("field"), "{plain}");
        assert!(plain.contains("path"), "{plain}");
        assert!(plain.contains("work/a.tet"), "{plain}");
        assert!(plain.contains("status"), "{plain}");
        assert!(!plain.contains("memory_strategy"), "{plain}");
        assert!(!plain.contains('┌'), "must not use UTF8_FULL box: {plain}");
    }

    fn console_strip(s: &str) -> String {
        #[cfg(feature = "tetration-cli")]
        {
            console::strip_ansi_codes(s).into_owned()
        }
        #[cfg(not(feature = "tetration-cli"))]
        {
            s.to_owned()
        }
    }
}

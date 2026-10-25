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

/// Fixed-width (4) status word, coloured on stderr when pretty output is on.
fn step_tag(tone: StepTone) -> String {
    let text = match tone {
        StepTone::Planned => "plan",
        StepTone::Ok => "ok  ",
    };
    #[cfg(feature = "tetration-cli")]
    {
        use console::style;
        let styled = style(text).for_stderr();
        match tone {
            StepTone::Planned => styled.dim(),
            StepTone::Ok => styled.green(),
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

/// Dry-run line: `[1/3] plan convert a.csv → a.tet`.
#[must_use]
pub fn step_planned(index_1based: usize, total: usize, description: &str) -> String {
    format!(
        "{} {} {description}",
        step_counter(index_1based, total),
        step_tag(StepTone::Planned)
    )
}

/// Finished-step line: `[1/3] ok   convert a.csv → a.tet (0.12s)`.
#[must_use]
pub fn step_ok(index_1based: usize, total: usize, description: &str, elapsed_secs: f64) -> String {
    format!(
        "{} {} {description} ({elapsed_secs:.2}s)",
        step_counter(index_1based, total),
        step_tag(StepTone::Ok)
    )
}

/// Dim, indented continuation line under a step (notes / skipped datasets).
#[must_use]
pub fn step_note_line(line: &str) -> String {
    let text = format!("       {line}");
    #[cfg(feature = "tetration-cli")]
    {
        console::style(text).for_stderr().dim().to_string()
    }
    #[cfg(not(feature = "tetration-cli"))]
    {
        text
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
            "[1/3] plan convert a → b"
        );
        assert_eq!(
            plain(step_ok(2, 12, "verify x", 0.5)),
            "[ 2/12] ok   verify x (0.50s)"
        );
        assert_eq!(plain(step_note_line("hi")), "       hi");
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

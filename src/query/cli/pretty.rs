//! Shared pretty-print helpers for `tet` human text (`comfy-table` / `console`).
//!
//! Gated on `tetration-cli` so library builds without the feature keep plain ASCII.

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

/// Data / kv table (`UTF8_FULL` — same style as the original query tables).
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

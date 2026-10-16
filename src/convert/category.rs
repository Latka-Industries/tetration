//! First-seen category codes shared by CSV and Parquet.
//!
//! Text cells become `i64` codes starting at 0. Blank or null is [`CATEGORY_MISSING`]
//! (`-1`). The label map is stored as the footer attr [`CATEGORY_ATTR`]. A map that
//! does not fit in [`MetadataLimitsV1::attr_string_bytes`] is a skip, not a truncated note.

use std::collections::BTreeMap;

use crate::catalog::MetadataLimitsV1;

/// Footer attribute holding `{"codes":{...},"missing":-1}`.
pub(crate) const CATEGORY_ATTR: &str = "tetration_categories";
/// Code written for a blank CSV cell or a Parquet null.
pub(crate) const CATEGORY_MISSING: i64 = -1;

/// One text column accumulated in first-seen order.
pub(crate) struct CategoryColumn {
    codes: BTreeMap<String, i64>,
    next: i64,
    values: Vec<i64>,
    overflow: bool,
}

/// Little-endian `i64` payload plus the footer attribute map.
pub(crate) struct CategoryImport {
    pub bytes: Vec<u8>,
    pub attrs: BTreeMap<String, String>,
}

impl CategoryColumn {
    pub(crate) fn new() -> Self {
        Self {
            codes: BTreeMap::new(),
            next: 0,
            values: Vec::new(),
            overflow: false,
        }
    }

    pub(crate) fn is_overflow(&self) -> bool {
        self.overflow
    }

    pub(crate) fn push_missing(&mut self) {
        if !self.overflow {
            self.values.push(CATEGORY_MISSING);
        }
    }

    pub(crate) fn push_label(&mut self, label: &str) {
        if self.overflow {
            return;
        }
        let code = if let Some(code) = self.codes.get(label) {
            *code
        } else {
            let code = self.next;
            self.codes.insert(label.to_owned(), code);
            self.next += 1;
            if category_note_json(&self.codes)
                .is_none_or(|json| json.len() > MetadataLimitsV1::DEFAULT.attr_string_bytes)
            {
                self.overflow = true;
                self.codes.clear();
                self.values.clear();
                return;
            }
            code
        };
        self.values.push(code);
    }

    pub(crate) fn finish(self) -> Result<CategoryImport, String> {
        if self.overflow {
            return Err(format!(
                "category map exceeds the {}-byte footer attr limit",
                MetadataLimitsV1::DEFAULT.attr_string_bytes
            ));
        }
        if self.codes.is_empty() {
            return Err("empty".to_owned());
        }
        let mut attrs = BTreeMap::new();
        if let Some(json) = category_note_json(&self.codes) {
            attrs.insert(CATEGORY_ATTR.to_owned(), json);
        }
        let mut bytes = Vec::with_capacity(self.values.len() * 8);
        for value in self.values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(CategoryImport { bytes, attrs })
    }
}

pub(crate) fn category_note_json(codes: &BTreeMap<String, i64>) -> Option<String> {
    serde_json::to_string(&serde_json::json!({
        "codes": codes,
        "missing": CATEGORY_MISSING,
    }))
    .ok()
}

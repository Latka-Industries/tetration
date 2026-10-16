//! CSV / TSV → `.tet`.
//!
//! Each numeric column becomes one 1-D dataset. A text column whose label map fits
//! in one footer attribute becomes an `i64` category dataset ([THI-492](https://linear.app/thicclatka/issue/THI-492)).
//! Mixed columns, and category maps that do not fit, are skips.
//! The file is parsed once into column buffers (CSV rows are not hyperslabs);
//! those buffers are then written in the same ~64 MiB tiles as `NumPy`.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::catalog::MetadataLimitsV1;
use crate::utils::dtype::ElementDtype;

use super::numpy::{NUMPY_DEFAULT_CHUNK_BYTES, numpy_chunk_shape};
use super::shared::{ImportPlan, ensure_non_empty, ensure_strict, write_plans_streaming};
use super::{ConvertError, ConvertProgress, ConvertReport, report};

/// Import a `.csv` (comma) or `.tsv` (tab) table into one `.tet`.
///
/// # Errors
///
/// Returns [`ConvertError`] when the file cannot be read, a row is ragged, or two
/// columns share a name.
pub fn convert_csv_to_tet(input: &Path, output: &Path) -> Result<ConvertReport, ConvertError> {
    convert_csv_to_tet_with_progress(input, output, 0, None::<fn(ConvertProgress)>)
}

/// Like [`convert_csv_to_tet`], invoking `progress` after each chunk payload is written.
///
/// # Errors
///
/// Returns [`ConvertError`] when import fails.
pub fn convert_csv_to_tet_with_progress(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    convert_csv_inner(input, output, parallel_jobs, false, progress)
}

pub(crate) fn convert_csv_inner(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    strict: bool,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    convert_csv_with_chunk_bytes(
        input,
        output,
        parallel_jobs,
        strict,
        NUMPY_DEFAULT_CHUNK_BYTES,
        progress,
    )
}

pub(crate) fn convert_csv_with_chunk_bytes(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    strict: bool,
    chunk_bytes: u64,
    mut progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    let started = Instant::now();
    let table = read_table(input)?;
    let mut plans = Vec::new();
    let mut payloads = Vec::new();
    let mut skipped = table.skipped;
    for column in table.columns {
        let elem_size = column.dtype.elem_size();
        let nrows = u64::try_from(column.bytes.len() / elem_size).unwrap_or(0);
        if nrows == 0 {
            super::shared::record_skip(&mut skipped, column.name, "empty");
            continue;
        }
        let chunk_shape = numpy_chunk_shape(&[nrows], elem_size, chunk_bytes);
        payloads.push(ColumnPayload {
            name: column.name.clone(),
            elem_size,
            chunk_elems: chunk_shape.first().copied().unwrap_or(1),
            bytes: column.bytes,
        });
        let mut plan = ImportPlan::new(column.name, column.dtype, vec![nrows], chunk_shape, None);
        if !column.attrs.is_empty() {
            plan = plan.with_import(column.attrs, None, None);
        }
        plans.push(plan);
    }

    ensure_strict(input, &skipped, strict)?;
    ensure_non_empty(
        input,
        &plans
            .iter()
            .map(|plan| plan.name.clone())
            .collect::<Vec<_>>(),
    )?;

    let mut progress_bridge = |done: u64, total: u64, dataset: &str| {
        if let Some(ref mut cb) = progress {
            cb(ConvertProgress {
                chunks_done: done,
                chunks_total: total,
                dataset: dataset.to_owned(),
            });
        }
    };
    let parallel_jobs = super::parallel::resolve_parallel_jobs(parallel_jobs);
    let source = CsvSource {
        payloads: payloads.into(),
    };
    write_plans_streaming(
        output,
        &plans,
        parallel_jobs,
        |job, buf| {
            let column = &source.payloads[job.dataset_id];
            let start = usize::try_from(job.chunk_coord[0] * column.chunk_elems)
                .map_err(|_| ConvertError::Csv("chunk offset overflow".to_owned()))?
                .saturating_mul(column.elem_size);
            let end = start + buf.len();
            if end > column.bytes.len() {
                return Err(ConvertError::Csv(format!(
                    "column `{}` tile exceeds parsed rows",
                    column.name
                )));
            }
            buf.copy_from_slice(&column.bytes[start..end]);
            Ok(())
        },
        Some(&mut progress_bridge as &mut dyn FnMut(u64, u64, &str)),
    )?;
    let history = super::import_metadata::finish_convert_footer(output, "csv", &plans)?;

    Ok(report(
        input,
        output,
        &plans,
        skipped,
        history,
        started.elapsed().as_secs_f64(),
    ))
}

struct ParsedColumn {
    name: String,
    dtype: ElementDtype,
    bytes: Vec<u8>,
    attrs: BTreeMap<String, String>,
}

struct ColumnPayload {
    name: String,
    elem_size: usize,
    chunk_elems: u64,
    bytes: Vec<u8>,
}

struct ParsedTable {
    columns: Vec<ParsedColumn>,
    skipped: Vec<super::ConvertSkip>,
}

struct CsvSource {
    payloads: Arc<[ColumnPayload]>,
}

fn read_table(input: &Path) -> Result<ParsedTable, ConvertError> {
    let delimiter = delimiter_for(input)?;
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(false)
        .flexible(true)
        .from_path(input)
        .map_err(|err| csv_io(input, &err))?;
    let mut records = reader.records();
    let Some(first) = records.next() else {
        return Ok(ParsedTable {
            columns: Vec::new(),
            skipped: Vec::new(),
        });
    };
    let first = first.map_err(|err| csv_io(input, &err))?;
    let width = first.len();
    if width == 0 {
        return Ok(ParsedTable {
            columns: Vec::new(),
            skipped: Vec::new(),
        });
    }
    let first_is_header = first.iter().any(|field| classify(field) == Cell::Text);
    let names = if first_is_header {
        header_names(&first)
    } else {
        (0..width).map(|index| format!("col{index}")).collect()
    };
    ensure_unique_names(input, &names)?;

    let mut columns: Vec<Building> = (0..width).map(|_| Building::Empty).collect();
    let mut row_index: u64 = 1;
    if !first_is_header {
        push_row(input, &first, &mut columns, row_index, width)?;
        row_index += 1;
    }
    for record in records {
        let record = record.map_err(|err| csv_io(input, &err))?;
        push_row(input, &record, &mut columns, row_index, width)?;
        row_index += 1;
    }

    let mut parsed = Vec::new();
    let mut skipped = Vec::new();
    for (name, column) in names.into_iter().zip(columns) {
        match column.finish() {
            Finished::Import {
                dtype,
                bytes,
                attrs,
            } => {
                parsed.push(ParsedColumn {
                    name,
                    dtype,
                    bytes,
                    attrs,
                });
            }
            Finished::Skip { reason } => {
                super::shared::record_skip(&mut skipped, name, &reason);
            }
        }
    }
    Ok(ParsedTable {
        columns: parsed,
        skipped,
    })
}

fn delimiter_for(path: &Path) -> Result<u8, ConvertError> {
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "csv" => Ok(b','),
        "tsv" => Ok(b'\t'),
        other => Err(ConvertError::Csv(format!(
            "expected a .csv or .tsv path, got extension `{other}`"
        ))),
    }
}

fn header_names(record: &csv::StringRecord) -> Vec<String> {
    record
        .iter()
        .enumerate()
        .map(|(index, field)| {
            let trimmed = field.trim().trim_start_matches('\u{feff}');
            if trimmed.is_empty() {
                format!("col{index}")
            } else {
                trimmed.to_owned()
            }
        })
        .collect()
}

fn ensure_unique_names(input: &Path, names: &[String]) -> Result<(), ConvertError> {
    let mut seen = std::collections::BTreeSet::new();
    for name in names {
        if !seen.insert(name.clone()) {
            return Err(ConvertError::Csv(format!(
                "csv {} has more than one column named `{name}`",
                input.display()
            )));
        }
    }
    Ok(())
}

fn push_row(
    input: &Path,
    record: &csv::StringRecord,
    columns: &mut [Building],
    row_index: u64,
    width: usize,
) -> Result<(), ConvertError> {
    if record.len() != width {
        return Err(ConvertError::Csv(format!(
            "csv {} row {row_index} has {} fields, expected {width}",
            input.display(),
            record.len()
        )));
    }
    for (column, field) in columns.iter_mut().zip(record.iter()) {
        column.push_field(field);
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum Cell {
    Blank,
    Int(i64),
    Float(f64),
    Text,
}

fn classify(field: &str) -> Cell {
    let token = field.trim().trim_start_matches('\u{feff}');
    if token.is_empty() {
        return Cell::Blank;
    }
    if is_plain_integer(token)
        && let Ok(value) = token.parse::<i64>()
    {
        return Cell::Int(value);
    }
    if let Ok(value) = token.parse::<f64>()
        && (value.is_finite() || is_non_finite_token(token))
    {
        return Cell::Float(value);
    }
    Cell::Text
}

fn is_plain_integer(token: &str) -> bool {
    let digits = token.strip_prefix(['+', '-']).unwrap_or(token);
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

fn is_non_finite_token(token: &str) -> bool {
    matches!(
        token.to_ascii_lowercase().as_str(),
        "nan" | "inf" | "+inf" | "-inf" | "infinity" | "+infinity" | "-infinity"
    )
}

const CATEGORY_ATTR: &str = "tetration_categories";
const CATEGORY_MISSING: i64 = -1;
const SKIP_NON_NUMERIC: &str = "non-numeric";
const SKIP_EMPTY: &str = "empty";
fn skip_category_too_big() -> String {
    format!(
        "category map exceeds the {}-byte footer attr limit",
        MetadataLimitsV1::DEFAULT.attr_string_bytes
    )
}

struct CategoryBuild {
    codes: BTreeMap<String, i64>,
    next: i64,
    values: Vec<i64>,
    overflow: bool,
}

enum Building {
    Empty,
    Blanks(u64),
    Int(Vec<i64>),
    Float(Vec<f64>),
    Category(CategoryBuild),
    Skip(String),
}

enum Finished {
    Import {
        dtype: ElementDtype,
        bytes: Vec<u8>,
        attrs: BTreeMap<String, String>,
    },
    Skip {
        reason: String,
    },
}

impl Building {
    fn push_field(&mut self, field: &str) {
        if matches!(self, Self::Skip(_)) {
            return;
        }
        let token = field.trim().trim_start_matches('\u{feff}');
        match classify(field) {
            Cell::Text => self.push_text(token),
            Cell::Blank => self.push_blank(),
            Cell::Int(value) => self.push_int(value),
            Cell::Float(value) => self.push_float(value),
        }
    }

    fn push_text(&mut self, label: &str) {
        match self {
            Self::Empty => {
                let mut column = CategoryBuild::new();
                column.push_label(label);
                *self = Self::Category(column);
            }
            Self::Blanks(count) => {
                let count = *count;
                let mut column = CategoryBuild::new();
                for _ in 0..count {
                    column.values.push(CATEGORY_MISSING);
                }
                column.push_label(label);
                *self = Self::Category(column);
            }
            Self::Int(_) | Self::Float(_) => *self = Self::Skip(SKIP_NON_NUMERIC.to_owned()),
            Self::Category(column) => {
                if column.overflow {
                    return;
                }
                column.push_label(label);
            }
            Self::Skip(_) => {}
        }
    }

    fn push_blank(&mut self) {
        match self {
            Self::Empty => *self = Self::Blanks(1),
            Self::Blanks(count) => *count += 1,
            Self::Int(values) => {
                let mut floats: Vec<f64> = values.iter().map(|value| *value as f64).collect();
                floats.push(f64::NAN);
                *self = Self::Float(floats);
            }
            Self::Float(values) => values.push(f64::NAN),
            Self::Category(column) => {
                if !column.overflow {
                    column.values.push(CATEGORY_MISSING);
                }
            }
            Self::Skip(_) => {}
        }
    }

    fn push_int(&mut self, value: i64) {
        match self {
            Self::Empty => *self = Self::Int(vec![value]),
            Self::Blanks(count) => {
                let mut floats = vec![f64::NAN; usize::try_from(*count).unwrap_or(0)];
                floats.push(value as f64);
                *self = Self::Float(floats);
            }
            Self::Int(values) => values.push(value),
            Self::Float(values) => values.push(value as f64),
            Self::Category(_) => *self = Self::Skip(SKIP_NON_NUMERIC.to_owned()),
            Self::Skip(_) => {}
        }
    }

    fn push_float(&mut self, value: f64) {
        match self {
            Self::Empty => *self = Self::Float(vec![value]),
            Self::Blanks(count) => {
                let mut floats = vec![f64::NAN; usize::try_from(*count).unwrap_or(0)];
                floats.push(value);
                *self = Self::Float(floats);
            }
            Self::Int(values) => {
                let mut floats: Vec<f64> = values.iter().map(|item| *item as f64).collect();
                floats.push(value);
                *self = Self::Float(floats);
            }
            Self::Float(values) => values.push(value),
            Self::Category(_) => *self = Self::Skip(SKIP_NON_NUMERIC.to_owned()),
            Self::Skip(_) => {}
        }
    }

    fn finish(self) -> Finished {
        match self {
            Self::Skip(reason) => Finished::Skip { reason },
            Self::Empty | Self::Blanks(_) => Finished::Skip {
                reason: SKIP_EMPTY.to_owned(),
            },
            Self::Int(values) if values.is_empty() => Finished::Skip {
                reason: SKIP_EMPTY.to_owned(),
            },
            Self::Float(values)
                if values.is_empty() || values.iter().all(|value| value.is_nan()) =>
            {
                Finished::Skip {
                    reason: SKIP_EMPTY.to_owned(),
                }
            }
            Self::Category(column) if column.overflow || column.codes.is_empty() => {
                Finished::Skip {
                    reason: if column.overflow {
                        skip_category_too_big()
                    } else {
                        SKIP_EMPTY.to_owned()
                    },
                }
            }
            Self::Int(values) => Finished::Import {
                dtype: ElementDtype::I64,
                bytes: i64s_to_le_bytes(&values),
                attrs: BTreeMap::new(),
            },
            Self::Float(values) => {
                let mut bytes = Vec::with_capacity(values.len() * 8);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                Finished::Import {
                    dtype: ElementDtype::F64,
                    bytes,
                    attrs: BTreeMap::new(),
                }
            }
            Self::Category(column) => {
                let mut attrs = BTreeMap::new();
                if let Some(json) = category_note_json(&column.codes) {
                    attrs.insert(CATEGORY_ATTR.to_owned(), json);
                }
                Finished::Import {
                    dtype: ElementDtype::I64,
                    bytes: i64s_to_le_bytes(&column.values),
                    attrs,
                }
            }
        }
    }
}

impl CategoryBuild {
    fn new() -> Self {
        Self {
            codes: BTreeMap::new(),
            next: 0,
            values: Vec::new(),
            overflow: false,
        }
    }

    fn push_label(&mut self, label: &str) {
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
}

fn category_note_json(codes: &BTreeMap<String, i64>) -> Option<String> {
    serde_json::to_string(&serde_json::json!({
        "codes": codes,
        "missing": CATEGORY_MISSING,
    }))
    .ok()
}

fn i64s_to_le_bytes(values: &[i64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn csv_io(path: &Path, err: &csv::Error) -> ConvertError {
    ConvertError::Csv(format!("{}: {err}", path.display()))
}

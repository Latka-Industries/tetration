//! CSV / TSV → `.tet`.
//!
//! Each numeric column becomes one 1-D dataset. A text column whose label map fits
//! in one footer attribute becomes an `i64` category dataset ([THI-492](https://linear.app/thicclatka/issue/THI-492)).
//! Mixed columns, and category maps that do not fit, are skips.
//! The file is parsed once into column buffers (CSV rows are not hyperslabs);
//! those buffers are then written in the same ~64 MiB tiles as `NumPy`
//! (`--chunk-mib` overrides that target).

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use crate::utils::dtype::ElementDtype;

use super::category::CategoryColumn;
use super::column_buf::{BufferedColumnSpec, accept_buffered_column, stream_buffered_columns};
use super::numpy::NUMPY_DEFAULT_CHUNK_BYTES;
use super::shared::{ensure_non_empty, ensure_strict};
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
    convert_csv_inner(input, output, parallel_jobs, false, None, progress)
}

pub(crate) fn convert_csv_inner(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    strict: bool,
    chunk_bytes: Option<u64>,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    convert_csv_with_chunk_bytes(
        input,
        output,
        parallel_jobs,
        strict,
        chunk_bytes.unwrap_or(NUMPY_DEFAULT_CHUNK_BYTES),
        progress,
    )
}

pub(crate) fn convert_csv_with_chunk_bytes(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    strict: bool,
    chunk_bytes: u64,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    let started = Instant::now();
    let table = read_table(input)?;
    let mut plans = Vec::new();
    let mut payloads = Vec::new();
    let mut skipped = table.skipped;
    for column in table.columns {
        accept_buffered_column(
            &mut plans,
            &mut payloads,
            &mut skipped,
            BufferedColumnSpec {
                name: column.name,
                dtype: column.dtype,
                bytes: column.bytes,
                attrs: column.attrs,
            },
            chunk_bytes,
        );
    }

    ensure_strict(input, &skipped, strict)?;
    ensure_non_empty(
        input,
        &plans
            .iter()
            .map(|plan| plan.name.clone())
            .collect::<Vec<_>>(),
    )?;
    let history = stream_buffered_columns(
        output,
        &plans,
        payloads,
        parallel_jobs,
        "csv",
        progress,
        ConvertError::Csv,
    )?;

    Ok(report(
        input,
        output,
        &plans,
        skipped,
        history,
        started.elapsed().as_secs_f64(),
    ))
}

/// One column that will become a 1-D dataset.
struct ParsedColumn {
    name: String,
    dtype: ElementDtype,
    bytes: Vec<u8>,
    attrs: BTreeMap<String, String>,
}

/// Columns kept for import, plus columns recorded as skips.
struct ParsedTable {
    columns: Vec<ParsedColumn>,
    skipped: Vec<super::ConvertSkip>,
}

/// Parse the file into one buffer per column.
///
/// `.csv` is comma-separated and `.tsv` is tab-separated. Cells are trimmed and a
/// leading BOM is stripped. The first row is a header when any cell is text;
/// otherwise names are `col0`, `col1`, … and that row is data. A blank in an
/// integer column promotes the column to `f64` with NaN for the blank. A column
/// whose non-blank cells are all text becomes category codes. Mixed text and
/// numbers, and columns that are empty or all NaN, are skips. Ragged rows and
/// duplicate names are errors. The csv crate drops lines with no fields; a
/// whitespace-only field is a blank.
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

/// Comma for `.csv`, tab for `.tsv`. Any other extension is an error.
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

/// Header labels, trimmed. An empty cell becomes `col{index}`.
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

/// Reject two columns that would share a catalog name.
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

/// Append one record. `row_index` is 1-based in the error text.
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
/// A trimmed cell after the header decision.
enum Cell {
    Blank,
    Int(i64),
    Float(f64),
    Text,
}

/// Plain integers (optional sign, digits only) stay integers. Decimals, exponents,
/// and `nan` / `inf` spellings are floats. Anything else is text.
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

/// Optional `+` or `-`, then one or more ASCII digits.
fn is_plain_integer(token: &str) -> bool {
    let digits = token.strip_prefix(['+', '-']).unwrap_or(token);
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

/// `nan`, `inf`, and `infinity`, with an optional sign, ignoring case.
fn is_non_finite_token(token: &str) -> bool {
    matches!(
        token.to_ascii_lowercase().as_str(),
        "nan" | "inf" | "+inf" | "-inf" | "infinity" | "+infinity" | "-infinity"
    )
}

const SKIP_NON_NUMERIC: &str = "non-numeric";
const SKIP_EMPTY: &str = "empty";

/// Column state while rows are scanned. The first non-blank cell chooses the kind.
enum Building {
    Empty,
    Blanks(u64),
    Int(Vec<i64>),
    Float(Vec<f64>),
    Category(CategoryColumn),
    Skip(String),
}

/// Little-endian payload ready to tile, or a skip reason.
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
    /// Route one cell into the column state machine.
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

    /// Start or extend a category column. A number already stored makes the column a skip.
    fn push_text(&mut self, label: &str) {
        match self {
            Self::Empty => {
                let mut column = CategoryColumn::new();
                column.push_label(label);
                *self = Self::Category(column);
            }
            Self::Blanks(count) => {
                let count = *count;
                let mut column = CategoryColumn::new();
                for _ in 0..count {
                    column.push_missing();
                }
                column.push_label(label);
                *self = Self::Category(column);
            }
            Self::Int(_) | Self::Float(_) => *self = Self::Skip(SKIP_NON_NUMERIC.to_owned()),
            Self::Category(column) => {
                if column.is_overflow() {
                    return;
                }
                column.push_label(label);
            }
            Self::Skip(_) => {}
        }
    }

    /// A blank is a category missing code, or NaN after an integer column promotes to `f64`.
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
            Self::Category(column) => column.push_missing(),
            Self::Skip(_) => {}
        }
    }

    /// Keep an all-integer column as `i64`. A blank seen earlier promotes it to `f64`.
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

    /// Store a decimal or non-finite value, promoting an integer column to `f64`.
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

    /// Emit `i64`, `f64`, or category bytes. Empty, all-NaN, and mixed columns are skips.
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
            Self::Category(column) => match column.finish() {
                Ok(imported) => Finished::Import {
                    dtype: ElementDtype::I64,
                    bytes: imported.bytes,
                    attrs: imported.attrs,
                },
                Err(reason) => Finished::Skip { reason },
            },
        }
    }
}

/// Little-endian `i64` payload.
fn i64s_to_le_bytes(values: &[i64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Attach the input path to a csv crate error.
fn csv_io(path: &Path, err: &csv::Error) -> ConvertError {
    ConvertError::Csv(format!("{}: {err}", path.display()))
}

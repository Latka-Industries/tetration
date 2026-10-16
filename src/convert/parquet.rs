//! Parquet → `.tet` (feature `tetration-parquet`).
//!
//! One 1-D dataset per primitive numeric column. UTF-8 columns use the same
//! category codes as CSV. Nested lists and structs are skips.

use std::fs::File;
use std::path::Path;
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::utils::dtype::ElementDtype;

use super::category::CategoryColumn;
use super::column_buf::{BufferedColumnSpec, accept_buffered_column, stream_buffered_columns};
use super::numpy::NUMPY_DEFAULT_CHUNK_BYTES;
use super::shared::{ensure_non_empty, ensure_strict};
use super::{ConvertError, ConvertProgress, ConvertReport, report};

use std::collections::BTreeMap;

/// Import numeric and category columns from a Parquet file into one `.tet`.
///
/// # Errors
///
/// Returns [`ConvertError`] when the file cannot be read or no supported column remains.
pub fn convert_parquet_to_tet(input: &Path, output: &Path) -> Result<ConvertReport, ConvertError> {
    convert_parquet_to_tet_with_progress(input, output, 0, None::<fn(ConvertProgress)>)
}

/// Like [`convert_parquet_to_tet`], invoking `progress` after each chunk payload is written.
///
/// # Errors
///
/// Returns [`ConvertError`] when import fails.
pub fn convert_parquet_to_tet_with_progress(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    convert_parquet_inner(input, output, parallel_jobs, false, progress)
}

pub(crate) fn convert_parquet_inner(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    strict: bool,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    convert_parquet_with_chunk_bytes(
        input,
        output,
        parallel_jobs,
        strict,
        NUMPY_DEFAULT_CHUNK_BYTES,
        progress,
    )
}

pub(crate) fn convert_parquet_with_chunk_bytes(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    strict: bool,
    chunk_bytes: u64,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    let started = Instant::now();
    let columns = read_columns(input)?;
    let mut plans = Vec::new();
    let mut payloads = Vec::new();
    let mut skipped = Vec::new();
    for column in columns {
        let name = column.name.clone();
        match column.finish() {
            Finished::Skip { reason } => {
                super::shared::record_skip(&mut skipped, name, &reason);
            }
            Finished::Import {
                dtype,
                bytes,
                attrs,
            } => accept_buffered_column(
                &mut plans,
                &mut payloads,
                &mut skipped,
                BufferedColumnSpec {
                    name,
                    dtype,
                    bytes,
                    attrs,
                },
                chunk_bytes,
            ),
        }
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
        "parquet",
        progress,
        ConvertError::Parquet,
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

struct ColumnBuild {
    name: String,
    kind: Kind,
}

enum Kind {
    Num {
        dtype: ElementDtype,
        bytes: Vec<u8>,
        signed: bool,
    },
    Category(CategoryColumn),
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

fn read_columns(input: &Path) -> Result<Vec<ColumnBuild>, ConvertError> {
    let file = File::open(input)
        .map_err(|err| ConvertError::Parquet(format!("{}: {err}", input.display())))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|err| ConvertError::Parquet(err.to_string()))?;
    let schema = builder.schema().clone();
    let mut columns: Vec<ColumnBuild> = schema
        .fields()
        .iter()
        .map(|field| ColumnBuild {
            name: field.name().clone(),
            kind: kind_for(field.data_type()),
        })
        .collect();
    let reader = builder
        .build()
        .map_err(|err| ConvertError::Parquet(err.to_string()))?;
    for batch in reader {
        let batch = batch.map_err(|err| ConvertError::Parquet(err.to_string()))?;
        push_batch(&mut columns, &batch)?;
    }
    Ok(columns)
}

fn kind_for(dtype: &DataType) -> Kind {
    let numeric = |dtype, signed| Kind::Num {
        dtype,
        bytes: Vec::new(),
        signed,
    };
    match dtype {
        DataType::Int16 => numeric(ElementDtype::I16, true),
        DataType::Int32 => numeric(ElementDtype::I32, true),
        DataType::Int64 => numeric(ElementDtype::I64, true),
        DataType::UInt8 => numeric(ElementDtype::U8, false),
        DataType::UInt16 => numeric(ElementDtype::U16, false),
        DataType::UInt32 => numeric(ElementDtype::U32, false),
        DataType::UInt64 => numeric(ElementDtype::U64, false),
        DataType::Float32 => numeric(ElementDtype::F32, true),
        DataType::Float64 => numeric(ElementDtype::F64, true),
        DataType::Utf8 | DataType::LargeUtf8 => Kind::Category(CategoryColumn::new()),
        other => Kind::Skip(format!("arrow type `{other}`")),
    }
}

fn push_batch(columns: &mut [ColumnBuild], batch: &RecordBatch) -> Result<(), ConvertError> {
    if batch.num_columns() != columns.len() {
        return Err(ConvertError::Parquet(
            "record batch column count does not match the schema".to_owned(),
        ));
    }
    for (column, array) in columns.iter_mut().zip(batch.columns()) {
        column.push(array.as_ref());
    }
    Ok(())
}

impl ColumnBuild {
    fn push(&mut self, array: &dyn Array) {
        if matches!(self.kind, Kind::Skip(_)) {
            return;
        }
        match array.data_type() {
            DataType::Int16 => {
                self.push_int(array.as_primitive::<arrow_array::types::Int16Type>(), 2);
            }
            DataType::Int32 => {
                self.push_int(array.as_primitive::<arrow_array::types::Int32Type>(), 4);
            }
            DataType::Int64 => {
                self.push_int(array.as_primitive::<arrow_array::types::Int64Type>(), 8);
            }
            DataType::UInt8 => {
                self.push_uint(array.as_primitive::<arrow_array::types::UInt8Type>(), 1);
            }
            DataType::UInt16 => {
                self.push_uint(array.as_primitive::<arrow_array::types::UInt16Type>(), 2);
            }
            DataType::UInt32 => {
                self.push_uint(array.as_primitive::<arrow_array::types::UInt32Type>(), 4);
            }
            DataType::UInt64 => {
                self.push_uint(array.as_primitive::<arrow_array::types::UInt64Type>(), 8);
            }
            DataType::Float32 => {
                self.push_float32(array.as_primitive::<arrow_array::types::Float32Type>());
            }
            DataType::Float64 => {
                self.push_float64(array.as_primitive::<arrow_array::types::Float64Type>());
            }
            DataType::Utf8 => self.push_text_utf8(array),
            DataType::LargeUtf8 => self.push_text_large(array),
            other => self.kind = Kind::Skip(format!("arrow type `{other}`")),
        }
    }

    fn push_int<T>(&mut self, array: &arrow_array::PrimitiveArray<T>, width: usize)
    where
        T: arrow_array::types::ArrowPrimitiveType,
        T::Native: Into<i128> + Copy,
    {
        for value in array {
            match value {
                None => self.push_numeric_null(),
                Some(value) => self.push_numeric_bits(&signed_le_bytes(value.into(), width), width),
            }
        }
    }

    fn push_uint<T>(&mut self, array: &arrow_array::PrimitiveArray<T>, width: usize)
    where
        T: arrow_array::types::ArrowPrimitiveType,
        T::Native: Into<u128> + Copy,
    {
        for value in array {
            match value {
                None => self.push_numeric_null(),
                Some(value) => {
                    self.push_numeric_bits(&unsigned_le_bytes(value.into(), width), width);
                }
            }
        }
    }

    fn push_float32(
        &mut self,
        array: &arrow_array::PrimitiveArray<arrow_array::types::Float32Type>,
    ) {
        for value in array {
            let bits = value.unwrap_or(f32::NAN).to_le_bytes();
            self.push_numeric_bits(&bits, 4);
        }
    }

    fn push_float64(
        &mut self,
        array: &arrow_array::PrimitiveArray<arrow_array::types::Float64Type>,
    ) {
        for value in array {
            let bits = value.unwrap_or(f64::NAN).to_le_bytes();
            self.push_numeric_bits(&bits, 8);
        }
    }

    fn push_numeric_bits(&mut self, bits: &[u8], width: usize) {
        let Kind::Num {
            dtype,
            bytes,
            signed,
        } = &mut self.kind
        else {
            return;
        };
        if *dtype == ElementDtype::F64 && width != 8 {
            let value = integer_bits_as_f64(bits, *signed);
            bytes.extend_from_slice(&value.to_le_bytes());
            return;
        }
        bytes.extend_from_slice(bits);
    }

    fn push_numeric_null(&mut self) {
        let Kind::Num {
            dtype,
            bytes,
            signed,
        } = &mut self.kind
        else {
            return;
        };
        if *dtype != ElementDtype::F64 {
            *bytes = promote_to_f64(bytes, dtype.elem_size(), *signed);
            *dtype = ElementDtype::F64;
        }
        bytes.extend_from_slice(&f64::NAN.to_le_bytes());
    }

    fn push_text_utf8(&mut self, array: &dyn Array) {
        let values = array.as_string::<i32>();
        for index in 0..values.len() {
            if values.is_null(index) {
                self.push_category_missing();
            } else {
                self.push_category_label(values.value(index));
            }
        }
    }

    fn push_text_large(&mut self, array: &dyn Array) {
        let values = array.as_string::<i64>();
        for index in 0..values.len() {
            if values.is_null(index) {
                self.push_category_missing();
            } else {
                self.push_category_label(values.value(index));
            }
        }
    }

    fn push_category_missing(&mut self) {
        if let Kind::Category(column) = &mut self.kind {
            column.push_missing();
        }
    }

    fn push_category_label(&mut self, label: &str) {
        if let Kind::Category(column) = &mut self.kind {
            column.push_label(label);
        }
    }

    fn finish(self) -> Finished {
        match self.kind {
            Kind::Skip(reason) => Finished::Skip { reason },
            Kind::Num { bytes, .. } if bytes.is_empty() => Finished::Skip {
                reason: "empty".to_owned(),
            },
            Kind::Num { dtype, bytes, .. } => Finished::Import {
                dtype,
                bytes,
                attrs: BTreeMap::new(),
            },
            Kind::Category(column) => match column.finish() {
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

fn signed_le_bytes(value: i128, width: usize) -> Vec<u8> {
    value.to_le_bytes()[..width].to_vec()
}

fn unsigned_le_bytes(value: u128, width: usize) -> Vec<u8> {
    value.to_le_bytes()[..width].to_vec()
}

fn integer_bits_as_f64(bits: &[u8], signed: bool) -> f64 {
    if signed {
        let mut wide = if bits.last().is_some_and(|byte| byte & 0x80 != 0) {
            [0xff; 16]
        } else {
            [0; 16]
        };
        wide[..bits.len()].copy_from_slice(bits);
        i128::from_le_bytes(wide) as f64
    } else {
        let mut wide = [0u8; 16];
        wide[..bits.len()].copy_from_slice(bits);
        u128::from_le_bytes(wide) as f64
    }
}

fn promote_to_f64(bytes: &[u8], width: usize, signed: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() / width * 8);
    for chunk in bytes.chunks(width) {
        out.extend_from_slice(&integer_bits_as_f64(chunk, signed).to_le_bytes());
    }
    out
}

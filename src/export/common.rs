//! Shared types and densify helpers for `.tet` → foreign export.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use memmap2::Mmap;
use serde::Serialize;

use crate::catalog::{
    CHUNK_PAYLOAD_CODEC_V1, CatalogError, ChunkIndexEntryV1, DatasetRecordV1, MAX_NDIM,
    TetFileSummaryV1, read_tet_summary_v1, tile,
};
use crate::layout::{self, mmap_file_read};
use crate::utils::dtype::ElementDtype;

/// Progress event emitted while writing datasets / chunks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ExportProgress {
    pub chunks_done: u64,
    pub chunks_total: u64,
    pub dataset: String,
}

/// Summary returned after a successful export.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ExportReport {
    pub input: String,
    pub output: String,
    pub dataset_count: usize,
    pub dataset_names: Vec<String>,
    pub chunks_written: u64,
    pub elapsed_secs: f64,
}

/// Export pipeline failures.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error(transparent)]
    Layout(#[from] layout::LayoutError),
    #[error(transparent)]
    Catalog(#[from] CatalogError),
    #[error("no datasets in {path}")]
    NoDatasets { path: String },
    #[error("export output `{path}` exists and is not empty")]
    OutputNotEmpty { path: String },
    #[error("export output `{path}` already exists")]
    OutputExists { path: String },
    #[error(
        "unsupported export output for `{path}` (use a Zarr directory, .h5/.hdf5, or .nc/.netcdf)"
    )]
    UnsupportedOutput { path: String },
    #[error("export I/O failed: {0}")]
    Io(String),
    #[error("Zarr export failed: {0}")]
    Zarr(String),
    #[error("HDF5 export failed: {0}")]
    Hdf5(String),
    #[error("NetCDF export failed: {0}")]
    Netcdf(String),
    #[error("unsupported dataset `{name}` dtype for export: {detail}")]
    UnsupportedDtype { name: String, detail: String },
    #[error("{format} export requires feature `{feature}` (rebuild with that feature enabled)")]
    ExportFeatureDisabled {
        format: &'static str,
        feature: &'static str,
    },
}

/// Destination format for [`super::export_tet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportOutputFormat {
    Zarr,
    H5,
    Netcdf,
}

impl ExportOutputFormat {
    /// Detect export target from the output path (extension, or directory → Zarr).
    #[must_use]
    pub fn detect(output: &Path) -> Option<Self> {
        if let Some(ext) = output.extension().and_then(|e| e.to_str()) {
            let ext = ext.to_ascii_lowercase();
            if matches!(ext.as_str(), "h5" | "hdf5" | "hdf" | "he2" | "he5") {
                return Some(Self::H5);
            }
            if matches!(
                ext.as_str(),
                "nc" | "netcdf" | "nc4" | "nc3" | "cdf" | "ncdf"
            ) {
                return Some(Self::Netcdf);
            }
            if ext == "zarr" {
                return Some(Self::Zarr);
            }
        }
        // Bare directory path (existing dir, or missing path without a file extension).
        if output.is_dir() || output.extension().is_none() {
            return Some(Self::Zarr);
        }
        None
    }
}

pub(crate) fn open_summary(input: &Path) -> Result<(Mmap, TetFileSummaryV1), ExportError> {
    let mmap = mmap_file_read(input).map_err(|e| ExportError::Io(e.to_string()))?;
    let summary = read_tet_summary_v1(&mmap)?;
    if summary.datasets.is_empty() {
        return Err(ExportError::NoDatasets {
            path: input.display().to_string(),
        });
    }
    Ok((mmap, summary))
}

pub(crate) fn finish_report(
    input: &Path,
    output: &Path,
    summary: &TetFileSummaryV1,
    chunks_written: u64,
    started: Instant,
) -> ExportReport {
    ExportReport {
        input: input.display().to_string(),
        output: output.display().to_string(),
        dataset_count: summary.datasets.len(),
        dataset_names: summary.datasets.iter().map(|d| d.name.clone()).collect(),
        chunks_written,
        elapsed_secs: started.elapsed().as_secs_f64(),
    }
}

pub(crate) fn ensure_missing_file(path: &Path) -> Result<(), ExportError> {
    if path.exists() {
        return Err(ExportError::OutputExists {
            path: path.display().to_string(),
        });
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| ExportError::Io(e.to_string()))?;
    }
    Ok(())
}

pub(crate) fn element_dtype(ds: &DatasetRecordV1) -> Result<ElementDtype, ExportError> {
    ElementDtype::try_from_wire_tag(ds.dtype).ok_or_else(|| ExportError::UnsupportedDtype {
        name: ds.name.clone(),
        detail: format!("unsupported wire dtype tag {:#x}", ds.dtype),
    })
}

pub(crate) fn element_dtype_wire_label(elem: ElementDtype) -> &'static str {
    match elem {
        ElementDtype::F32 => "f32",
        ElementDtype::F64 => "f64",
        ElementDtype::I32 => "i32",
        ElementDtype::I64 => "i64",
        ElementDtype::U8 => "u8",
        ElementDtype::U16 => "u16",
        ElementDtype::I16 => "i16",
        ElementDtype::U32 => "u32",
        ElementDtype::F16 => "f16",
        ElementDtype::U64 => "u64",
    }
}

/// Attr names safe to rewrite as foreign string attributes (skip typed CF packing fields).
const BASIC_EXPORT_ATTRS: &[&str] = &[
    "units",
    "long_name",
    "standard_name",
    "description",
    "title",
];

pub(crate) fn dataset_footer_attrs(
    summary: &TetFileSummaryV1,
    name: &str,
) -> BTreeMap<String, String> {
    summary
        .metadata
        .datasets
        .get(name)
        .map(|m| m.attrs.clone())
        .unwrap_or_default()
}

/// Footer attrs suitable for HDF5 / `NetCDF` string attributes.
pub(crate) fn dataset_basic_export_attrs(
    summary: &TetFileSummaryV1,
    name: &str,
) -> BTreeMap<String, String> {
    dataset_footer_attrs(summary, name)
        .into_iter()
        .filter(|(k, _)| BASIC_EXPORT_ATTRS.contains(&k.as_str()))
        .collect()
}

pub(crate) fn dataset_dim_names(summary: &TetFileSummaryV1, name: &str) -> Option<Vec<String>> {
    summary
        .metadata
        .datasets
        .get(name)
        .and_then(|m| m.dim_names.clone())
}

pub(crate) fn chunks_for_dataset(
    summary: &TetFileSummaryV1,
    ds_id: u64,
) -> Vec<&ChunkIndexEntryV1> {
    summary
        .chunks
        .iter()
        .filter(|c| c.dataset_id == ds_id)
        .collect()
}

/// Decode every chunk and scatter into a contiguous row-major buffer.
pub(crate) fn densify_dataset(
    mmap: &[u8],
    ds: &DatasetRecordV1,
    chunks: &[&ChunkIndexEntryV1],
) -> Result<Vec<u8>, ExportError> {
    let elem = element_dtype(ds)?;
    let nbytes =
        elem.tensor_bytes_for_shape(&ds.shape)
            .ok_or_else(|| ExportError::UnsupportedDtype {
                name: ds.name.clone(),
                detail: "shape overflow computing tensor byte length".into(),
            })?;
    let nbytes_u64 = nbytes;
    let nbytes = usize::try_from(nbytes).map_err(|_| {
        ExportError::Catalog(CatalogError::TooLargeForPlatform {
            field: "tensor_bytes",
            value: nbytes_u64,
        })
    })?;
    let mut full = vec![0u8; nbytes];
    let ndim = ds.shape.len();
    let elem_size = elem.elem_size();

    for entry in chunks {
        let start = usize_from_u64("payload_offset", entry.payload_offset)?;
        let end = start
            .checked_add(usize_from_u64("stored_byte_len", entry.stored_byte_len)?)
            .ok_or(ExportError::Catalog(CatalogError::InvalidWriteSpec(
                "chunk payload span overflow",
            )))?;
        if end > mmap.len() {
            return Err(ExportError::Catalog(CatalogError::PayloadOutOfBounds {
                file_len: mmap.len() as u64,
                start: entry.payload_offset,
                end: end as u64,
            }));
        }
        let stored = &mmap[start..end];
        let raw = CHUNK_PAYLOAD_CODEC_V1.decode_tile_payload(
            stored,
            entry.raw_byte_len,
            entry.stored_byte_len,
            entry.codec,
        )?;
        scatter_tile_into_full(
            &mut full,
            raw.as_ref(),
            &ds.shape,
            &ds.chunk_shape,
            &entry.chunk_index[..ndim],
            ndim,
            elem_size,
        )?;
    }
    Ok(full)
}

fn scatter_tile_into_full(
    full: &mut [u8],
    tile_bytes: &[u8],
    shape: &[u64],
    chunk_shape: &[u64],
    chunk_coord: &[u64],
    ndim: usize,
    elem_size: usize,
) -> Result<(), ExportError> {
    let tile_ext = tile::tile_extent(shape, chunk_shape, chunk_coord, ndim);
    let nelem: u64 = tile_ext
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(b))
        .ok_or(CatalogError::InvalidWriteSpec(
            "tile element count overflow",
        ))?;
    let expect = nelem
        .checked_mul(elem_size as u64)
        .ok_or(CatalogError::InvalidWriteSpec("tile byte length overflow"))?;
    let expect_usize = usize::try_from(expect).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "tile_byte_length",
        value: expect,
    })?;
    if tile_bytes.len() != expect_usize {
        return Err(ExportError::Catalog(CatalogError::DecodedLengthMismatch {
            decoded: tile_bytes.len(),
            raw: expect,
        }));
    }
    for k in 0..nelem {
        let local = tile::local_coords_from_linear(k, &tile_ext, ndim);
        let mut global = [0u64; MAX_NDIM];
        for d in 0..ndim {
            global[d] = chunk_coord[d]
                .checked_mul(chunk_shape[d])
                .and_then(|s| s.checked_add(local[d]))
                .ok_or(CatalogError::InvalidWriteSpec(
                    "tile global coordinate overflow",
                ))?;
        }
        let li = tile::linear_elem_row_major(&global[..ndim], shape, ndim)?;
        let dst = usize::try_from(li)
            .map_err(|_| CatalogError::TooLargeForPlatform {
                field: "linear_element_index",
                value: li,
            })?
            .checked_mul(elem_size)
            .ok_or(CatalogError::InvalidWriteSpec("byte offset overflow"))?;
        let src = usize::try_from(k)
            .map_err(|_| CatalogError::TooLargeForPlatform {
                field: "tile_linear_index",
                value: k,
            })?
            .checked_mul(elem_size)
            .ok_or(CatalogError::InvalidWriteSpec("byte offset overflow"))?;
        if dst + elem_size > full.len() || src + elem_size > tile_bytes.len() {
            return Err(ExportError::Catalog(CatalogError::InvalidWriteSpec(
                "scatter tile out of bounds",
            )));
        }
        full[dst..dst + elem_size].copy_from_slice(&tile_bytes[src..src + elem_size]);
    }
    Ok(())
}

fn usize_from_u64(field: &'static str, v: u64) -> Result<usize, ExportError> {
    usize::try_from(v)
        .map_err(|_| ExportError::Catalog(CatalogError::TooLargeForPlatform { field, value: v }))
}

pub(crate) fn shape_as_usize(shape: &[u64]) -> Result<Vec<usize>, ExportError> {
    shape
        .iter()
        .map(|&d| {
            usize::try_from(d).map_err(|_| {
                ExportError::Catalog(CatalogError::TooLargeForPlatform {
                    field: "shape_dim",
                    value: d,
                })
            })
        })
        .collect()
}

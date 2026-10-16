//! Zarr v3 directory store → `.tet` conversion.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;

use crate::utils::dtype::ElementDtype;

use super::import_metadata::{finish_convert_footer, zarr_array_attrs};
use super::parallel::ZarrParallelSource;
use super::shared::{
    ImportPlan, ImportTileRead, chunk_shape_for_import, ensure_non_empty, ensure_strict,
    join_catalog_path, note_plan, write_plans_streaming,
};
use super::tile_io::tile_axis_ranges;
use super::{ConvertError, ConvertProgress, ConvertReport, report};

/// Import supported numeric arrays from a Zarr v3 directory store into one `.tet`.
///
/// Supported element types: `float32`, `float64`, `int32`, `int64`. Nested groups map to
/// slash-separated catalog names (`primary/f32`). Chunk payloads may be raw `bytes` or `bytes` + `zstd`.
///
/// # Errors
///
/// Returns [`ConvertError`] when metadata cannot be read or no supported arrays are present.
pub fn convert_zarr_to_tet(input: &Path, output: &Path) -> Result<ConvertReport, ConvertError> {
    convert_zarr_to_tet_with_progress(input, output, 0, None::<fn(ConvertProgress)>)
}

/// Like [`convert_zarr_to_tet`], invoking `progress` after each chunk payload is written.
///
/// # Errors
///
/// Returns [`ConvertError`] when metadata cannot be read or no supported arrays are present.
pub fn convert_zarr_to_tet_with_progress(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    convert_zarr_inner(input, output, parallel_jobs, false, progress)
}

pub(crate) fn convert_zarr_inner(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    strict: bool,
    mut progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    let started = Instant::now();
    let store = input
        .canonicalize()
        .map_err(|e| ConvertError::Zarr(e.to_string()))?;
    let mut plans = Vec::new();
    let mut skipped = Vec::new();
    collect_zarr_plans(&store, "", "", &mut plans, &mut skipped)?;

    ensure_non_empty(
        input,
        &plans.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
    )?;
    ensure_strict(input, &skipped, strict)?;

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
    let source = ZarrParallelSource::new(store.clone(), plans.clone());
    write_plans_streaming(
        output,
        &plans,
        parallel_jobs,
        |job, buf| source.fill_tile(job, buf),
        Some(&mut progress_bridge as &mut dyn FnMut(u64, u64, &str)),
    )?;
    let history = finish_convert_footer(output, "zarr", &plans)?;

    Ok(report(
        input,
        output,
        &plans,
        skipped,
        history,
        started.elapsed().as_secs_f64(),
    ))
}

pub(crate) fn read_zarr_tile_le_into(
    store: &Path,
    array_rel: &str,
    zstd: bool,
    spec: ImportTileRead<'_>,
    buf: &mut [u8],
) -> Result<(), ConvertError> {
    let elem_size = spec.dtype.elem_size();
    let ranges = tile_axis_ranges(spec);
    let tile_shape = tile_shape_from_ranges(&ranges)?;
    let chunk_origin: Vec<u64> = (0..spec.ndim)
        .map(|d| spec.chunk_coord[d] * spec.chunk_shape[d])
        .collect();
    let chunk_bytes = read_zarr_chunk_le(
        store,
        array_rel,
        zstd,
        spec.chunk_coord,
        spec.ndim,
        spec.chunk_shape,
        elem_size,
    )?;
    ChunkTileCopy {
        chunk: &chunk_bytes,
        chunk_shape: spec.chunk_shape,
        tile_ranges: &ranges,
        chunk_origin: &chunk_origin,
        elem_size,
        ndim: spec.ndim,
        tile_shape: &tile_shape,
    }
    .copy_into(buf)
}

fn collect_zarr_plans(
    store: &Path,
    rel: &str,
    catalog_prefix: &str,
    plans: &mut Vec<ImportPlan>,
    skipped: &mut Vec<super::ConvertSkip>,
) -> Result<(), ConvertError> {
    let dir = zarr_node_dir(store, rel);
    let meta = read_zarr_meta(&dir)?;
    if meta.zarr_format != 3 {
        return Err(ConvertError::Zarr(format!(
            "unsupported zarr_format {} at {}",
            meta.zarr_format,
            dir.display()
        )));
    }
    match meta.node_type.as_str() {
        "array" => note_plan(plan_zarr_array(catalog_prefix, rel, &meta), plans, skipped)?,
        "group" => {
            for child in zarr_child_names(&dir)? {
                let child_rel = join_rel(rel, &child);
                let child_catalog = join_catalog_path(catalog_prefix, &child);
                collect_zarr_plans(store, &child_rel, &child_catalog, plans, skipped)?;
            }
        }
        other => {
            return Err(ConvertError::Zarr(format!(
                "unsupported node_type `{other}` at {}",
                dir.display()
            )));
        }
    }
    Ok(())
}

fn plan_zarr_array(
    name: &str,
    array_rel: &str,
    meta: &ZarrNodeMeta,
) -> Result<ImportPlan, ConvertError> {
    let shape = meta
        .shape
        .clone()
        .ok_or_else(|| ConvertError::Zarr(format!("array `{name}` missing shape")))?;
    if shape.is_empty() || shape.contains(&0) {
        return Err(ConvertError::UnsupportedDtype {
            name: name.to_owned(),
            detail: "empty zarr array".into(),
        });
    }
    let dtype = map_zarr_dtype(name, meta.data_type.as_ref())?;
    let zarr_chunks = meta
        .chunk_grid
        .as_ref()
        .and_then(|grid| (grid.name == "regular").then(|| grid.configuration.chunk_shape.clone()))
        .map(|v| {
            v.into_iter()
                .map(|c| usize::try_from(c).unwrap_or(0))
                .collect()
        });
    let chunk_shape = chunk_shape_for_import(&shape, zarr_chunks);
    Ok(
        ImportPlan::new(name.to_owned(), dtype, shape, chunk_shape, None)
            .with_zarr(array_rel.to_owned(), zarr_chunk_payload_zstd(meta))
            .with_import(zarr_array_attrs(&meta.attributes), None, None),
    )
}

fn map_zarr_dtype(
    name: &str,
    data_type: Option<&serde_json::Value>,
) -> Result<ElementDtype, ConvertError> {
    let tag = data_type
        .and_then(|v| v.as_str().map(str::to_owned))
        .or_else(|| {
            data_type
                .and_then(|v| v.get("name"))
                .and_then(|v| v.as_str().map(str::to_owned))
        })
        .ok_or_else(|| ConvertError::Zarr(format!("array `{name}` missing data_type")))?;
    match tag.as_str() {
        "float16" => Ok(ElementDtype::F16),
        "float32" => Ok(ElementDtype::F32),
        "float64" => Ok(ElementDtype::F64),
        "uint8" | "bool" | "int8" => Ok(ElementDtype::U8),
        "uint16" => Ok(ElementDtype::U16),
        "int16" => Ok(ElementDtype::I16),
        "int32" => Ok(ElementDtype::I32),
        "int64" => Ok(ElementDtype::I64),
        "uint32" => Ok(ElementDtype::U32),
        "uint64" => Ok(ElementDtype::U64),
        other => Err(ConvertError::UnsupportedDtype {
            name: name.to_owned(),
            detail: format!("zarr data_type `{other}`"),
        }),
    }
}

fn zarr_chunk_payload_zstd(meta: &ZarrNodeMeta) -> bool {
    meta.codecs.as_ref().is_some_and(|codecs| {
        codecs.iter().any(|codec| {
            codec
                .get("name")
                .and_then(|v| v.as_str())
                .is_some_and(|name| name == "zstd")
        })
    })
}

fn read_zarr_chunk_le(
    store: &Path,
    array_rel: &str,
    zstd: bool,
    chunk_coord: &[u64],
    ndim: usize,
    chunk_shape: &[u64],
    elem_size: usize,
) -> Result<Vec<u8>, ConvertError> {
    let mut path = store.join(array_rel).join("c");
    for &coord in chunk_coord.iter().take(ndim) {
        path.push(coord.to_string());
    }
    let on_disk = fs::read(&path).map_err(|e| ConvertError::Zarr(e.to_string()))?;
    let decoded = if zstd {
        zstd::decode_all(on_disk.as_slice()).map_err(|e| ConvertError::Zarr(e.to_string()))?
    } else {
        on_disk
    };
    let expected = chunk_shape.iter().take(ndim).fold(elem_size, |acc, &d| {
        acc.saturating_mul(usize::try_from(d).unwrap_or(0))
    });
    if decoded.len() != expected {
        return Err(ConvertError::Zarr(format!(
            "chunk {} byte length mismatch (expected {expected}, got {})",
            path.display(),
            decoded.len()
        )));
    }
    Ok(decoded)
}

fn tile_shape_from_ranges(ranges: &[(u64, u64)]) -> Result<Vec<usize>, ConvertError> {
    ranges
        .iter()
        .map(|&(start, end)| {
            usize::try_from(end - start)
                .map_err(|_| ConvertError::Zarr("tile axis too large".into()))
        })
        .collect()
}

struct ChunkTileCopy<'a> {
    chunk: &'a [u8],
    chunk_shape: &'a [u64],
    tile_ranges: &'a [(u64, u64)],
    chunk_origin: &'a [u64],
    elem_size: usize,
    ndim: usize,
    tile_shape: &'a [usize],
}

impl ChunkTileCopy<'_> {
    fn copy_into(self, out: &mut [u8]) -> Result<(), ConvertError> {
        let expected = self.tile_shape.iter().product::<usize>() * self.elem_size;
        if out.len() != expected {
            return Err(ConvertError::Zarr(format!(
                "tile output length mismatch (expected {expected}, got {})",
                out.len()
            )));
        }
        if self.ndim == 1 {
            let axis_start = usize::try_from(self.tile_ranges[0].0 - self.chunk_origin[0])
                .map_err(map_usize_err)?;
            let axis_len = self.tile_shape[0];
            let row_bytes = axis_len * self.elem_size;
            let start = axis_start * self.elem_size;
            out.copy_from_slice(&self.chunk[start..start + row_bytes]);
            return Ok(());
        }
        let mut idx = vec![0; self.ndim];
        self.copy_level(0, &mut idx, 0, out)
    }

    fn copy_level(
        &self,
        level: usize,
        idx: &mut [usize],
        out_offset: usize,
        out: &mut [u8],
    ) -> Result<(), ConvertError> {
        if level == self.ndim {
            let offset = linear_offset(idx, self.chunk_shape, self.ndim, self.elem_size)?;
            out[out_offset..out_offset + self.elem_size]
                .copy_from_slice(&self.chunk[offset..offset + self.elem_size]);
            return Ok(());
        }
        let axis_start = usize::try_from(self.tile_ranges[level].0 - self.chunk_origin[level])
            .map_err(map_usize_err)?;
        let axis_len = self.tile_shape[level];
        let inner = self.tile_shape[level + 1..].iter().product::<usize>() * self.elem_size;
        for i in 0..axis_len {
            idx[level] = axis_start + i;
            self.copy_level(level + 1, idx, out_offset + i * inner, out)?;
        }
        Ok(())
    }
}

fn linear_offset(
    idx: &[usize],
    chunk_shape: &[u64],
    ndim: usize,
    elem_size: usize,
) -> Result<usize, ConvertError> {
    let mut stride = elem_size;
    let mut offset = 0usize;
    for d in (0..ndim).rev() {
        offset = offset.saturating_add(idx[d].saturating_mul(stride));
        stride = stride.saturating_mul(usize::try_from(chunk_shape[d]).map_err(map_usize_err)?);
    }
    Ok(offset)
}

fn map_usize_err(_: TryFromIntError) -> ConvertError {
    ConvertError::Zarr("index exceeds platform usize".into())
}

use std::num::TryFromIntError;

fn read_zarr_meta(dir: &Path) -> Result<ZarrNodeMeta, ConvertError> {
    let path = dir.join("zarr.json");
    let raw = fs::read_to_string(&path).map_err(|e| ConvertError::Zarr(e.to_string()))?;
    serde_json::from_str(&raw).map_err(|e| ConvertError::Zarr(e.to_string()))
}

fn zarr_node_dir(store: &Path, rel: &str) -> PathBuf {
    if rel.is_empty() {
        store.to_path_buf()
    } else {
        store.join(rel)
    }
}

fn join_rel(prefix: &str, child: &str) -> String {
    if prefix.is_empty() {
        child.to_owned()
    } else {
        format!("{prefix}/{child}")
    }
}

fn zarr_child_names(dir: &Path) -> Result<Vec<String>, ConvertError> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| ConvertError::Zarr(e.to_string()))? {
        let entry = entry.map_err(|e| ConvertError::Zarr(e.to_string()))?;
        let path = entry.path();
        if path.is_dir() && path.join("zarr.json").is_file() {
            out.push(
                entry
                    .file_name()
                    .into_string()
                    .map_err(|_| ConvertError::Zarr("non-utf8 zarr child name".into()))?,
            );
        }
    }
    out.sort();
    Ok(out)
}

/// Returns true when `path` is a directory containing Zarr v3 root metadata.
#[must_use]
pub fn is_zarr_v3_directory(path: &Path) -> bool {
    let meta_path = path.join("zarr.json");
    if !meta_path.is_file() {
        return false;
    }
    let Ok(raw) = fs::read_to_string(meta_path) else {
        return false;
    };
    let Ok(meta) = serde_json::from_str::<ZarrNodeMeta>(&raw) else {
        return false;
    };
    meta.zarr_format == 3
}

#[derive(Debug, Deserialize)]
struct ZarrNodeMeta {
    zarr_format: u32,
    node_type: String,
    #[serde(default)]
    attributes: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    shape: Option<Vec<u64>>,
    #[serde(default)]
    data_type: Option<serde_json::Value>,
    #[serde(default)]
    chunk_grid: Option<ZarrChunkGrid>,
    #[serde(default)]
    codecs: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct ZarrChunkGrid {
    name: String,
    configuration: ZarrChunkGridConfig,
}

#[derive(Debug, Deserialize)]
struct ZarrChunkGridConfig {
    chunk_shape: Vec<u64>,
}

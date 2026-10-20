//! Zarr v2 and v3 directory stores → `.tet`.
//!
//! v3 (`zarr.json`) wins when both metadata files are present. v2 (`.zgroup` / `.zarray`)
//! uses history source `zarr2`. Chunk files are raw or zstd; other codecs are skips.

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use serde::Deserialize;

use super::zarr_io::{LocalZarrDir, ZarrObjects, join_rel, zarr_chunk_key, zarr_objects_v2};

use crate::utils::dtype::ElementDtype;
use crate::utils::zarr_dtype::{ZarrV2DtypeParse, element_dtype_from_zarr_v3, parse_zarr_v2_dtype};

use super::import_metadata::{finish_convert_footer, zarr_array_attrs};
use super::numpy::numpy_chunk_shape;
use super::parallel::ZarrParallelSource;
use super::shared::{
    ImportPlan, ImportTileRead, chunk_shape_for_import, ensure_non_empty, ensure_strict,
    join_catalog_path, note_plan, write_plans_streaming,
};
use super::tile_io::tile_axis_ranges;
use super::{ConvertError, ConvertProgress, ConvertReport, report};

/// Import supported numeric arrays from a Zarr v2 or v3 directory store into one `.tet`.
///
/// Nested groups map to slash-separated catalog names (`primary/f32`). Chunk payloads may be
/// raw bytes or zstd.
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
    convert_zarr_inner(input, output, parallel_jobs, false, None, progress)
}

pub(crate) fn convert_zarr_inner(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    strict: bool,
    chunk_bytes: Option<u64>,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    let store = input
        .canonicalize()
        .map_err(|e| ConvertError::Zarr(e.to_string()))?;
    convert_zarr_from_objects(
        input,
        Arc::new(LocalZarrDir::new(store)),
        output,
        parallel_jobs,
        strict,
        chunk_bytes,
        progress,
    )
}

pub(crate) fn convert_zarr_from_objects(
    input: &Path,
    objects: Arc<dyn ZarrObjects>,
    output: &Path,
    parallel_jobs: usize,
    strict: bool,
    chunk_bytes: Option<u64>,
    mut progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    let started = Instant::now();
    let v2 = zarr_objects_v2(objects.as_ref());
    let mut plans = Vec::new();
    let mut skipped = Vec::new();
    if v2 {
        collect_zarr_v2_plans(
            objects.as_ref(),
            "",
            "",
            chunk_bytes,
            &mut plans,
            &mut skipped,
        )?;
    } else {
        collect_zarr_plans(
            objects.as_ref(),
            "",
            "",
            chunk_bytes,
            &mut plans,
            &mut skipped,
        )?;
    }

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
    let source = ZarrParallelSource::new(objects, plans.clone());
    write_plans_streaming(
        output,
        &plans,
        parallel_jobs,
        |job, buf| source.fill_tile(job, buf),
        Some(&mut progress_bridge as &mut dyn FnMut(u64, u64, &str)),
    )?;
    let history = finish_convert_footer(output, if v2 { "zarr2" } else { "zarr" }, &plans)?;

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
    objects: &dyn ZarrObjects,
    array_rel: &str,
    zstd: bool,
    v2_sep: Option<u8>,
    spec: ImportTileRead<'_>,
    buf: &mut [u8],
) -> Result<(), ConvertError> {
    if let Some(source_chunks) = spec.zarr_source_chunks {
        return read_zarr_rechunked_tile(
            objects,
            array_rel,
            zstd,
            v2_sep,
            spec,
            source_chunks,
            buf,
        );
    }
    let elem_size = spec.dtype.elem_size();
    let ranges = tile_axis_ranges(spec);
    let tile_shape = tile_shape_from_ranges(&ranges)?;
    let chunk_origin: Vec<u64> = (0..spec.ndim)
        .map(|d| spec.chunk_coord[d] * spec.chunk_shape[d])
        .collect();
    let chunk_bytes = read_zarr_chunk_le(
        objects,
        &ZarrChunkRead {
            array_rel,
            zstd,
            v2_sep,
            chunk_coord: spec.chunk_coord,
            ndim: spec.ndim,
            chunk_shape: spec.chunk_shape,
            elem_size,
        },
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

/// Assemble one `.tet` tile from Zarr chunk files whose grid is not the output grid.
fn read_zarr_rechunked_tile(
    objects: &dyn ZarrObjects,
    array_rel: &str,
    zstd: bool,
    v2_sep: Option<u8>,
    spec: ImportTileRead<'_>,
    source_chunks: &[u64],
    buf: &mut [u8],
) -> Result<(), ConvertError> {
    let ndim = spec.ndim;
    let ranges = tile_axis_ranges(spec);
    let tile_extent: Vec<u64> = ranges.iter().map(|(start, end)| end - start).collect();
    let mut lo = vec![0u64; ndim];
    let mut hi = vec![0u64; ndim];
    for axis in 0..ndim {
        let cs = source_chunks[axis].max(1);
        lo[axis] = ranges[axis].0 / cs;
        let last = ranges[axis].1.saturating_sub(1);
        hi[axis] = last / cs + 1;
    }
    let mut coord = vec![0u64; ndim];
    visit_source_chunks(
        0,
        &mut coord,
        &lo,
        &hi,
        &SourceChunkVisit {
            objects,
            array_rel,
            zstd,
            v2_sep,
            shape: spec.shape,
            source_chunks,
            ranges: &ranges,
            tile_extent: &tile_extent,
            elem_size: spec.dtype.elem_size(),
            ndim,
        },
        buf,
    )
}

struct SourceChunkVisit<'a> {
    objects: &'a dyn ZarrObjects,
    array_rel: &'a str,
    zstd: bool,
    v2_sep: Option<u8>,
    shape: &'a [u64],
    source_chunks: &'a [u64],
    ranges: &'a [(u64, u64)],
    tile_extent: &'a [u64],
    elem_size: usize,
    ndim: usize,
}

fn visit_source_chunks(
    axis: usize,
    coord: &mut [u64],
    lo: &[u64],
    hi: &[u64],
    visit: &SourceChunkVisit<'_>,
    buf: &mut [u8],
) -> Result<(), ConvertError> {
    if axis == visit.ndim {
        return copy_source_chunk_into_tile(coord, visit, buf);
    }
    for index in lo[axis]..hi[axis] {
        coord[axis] = index;
        visit_source_chunks(axis + 1, coord, lo, hi, visit, buf)?;
    }
    Ok(())
}

fn copy_source_chunk_into_tile(
    coord: &[u64],
    visit: &SourceChunkVisit<'_>,
    buf: &mut [u8],
) -> Result<(), ConvertError> {
    let (bytes, layout) = load_zarr_source_chunk(coord, visit)?;
    let mut overlap = Vec::with_capacity(visit.ndim);
    let mut src_origin = Vec::with_capacity(visit.ndim);
    let mut tile_origin = Vec::with_capacity(visit.ndim);
    for axis in 0..visit.ndim {
        let origin = coord[axis].saturating_mul(visit.source_chunks[axis]);
        let src_end = origin.saturating_add(layout[axis]);
        let start = visit.ranges[axis].0.max(origin);
        let end = visit.ranges[axis].1.min(src_end);
        if start >= end {
            return Ok(());
        }
        overlap.push((start, end));
        src_origin.push(origin);
        tile_origin.push(visit.ranges[axis].0);
    }
    let mut src_local = vec![0u64; visit.ndim];
    let mut dst_local = vec![0u64; visit.ndim];
    blit_overlap(
        &Blit {
            src: &bytes,
            layout: &layout,
            src_origin: &src_origin,
            tile_origin: &tile_origin,
            overlap: &overlap,
            tile_extent: visit.tile_extent,
            elem_size: visit.elem_size,
            ndim: visit.ndim,
        },
        0,
        &mut src_local,
        &mut dst_local,
        buf,
    )
}

fn load_zarr_source_chunk(
    coord: &[u64],
    visit: &SourceChunkVisit<'_>,
) -> Result<(Vec<u8>, Vec<u64>), ConvertError> {
    let key = zarr_chunk_key(visit.array_rel, visit.v2_sep, coord, visit.ndim);
    let on_disk = visit.objects.read_bytes(&key)?;
    let decoded = if visit.zstd {
        zstd::decode_all(on_disk.as_slice()).map_err(|err| ConvertError::Zarr(err.to_string()))?
    } else {
        on_disk
    };
    let partial: Vec<u64> = (0..visit.ndim)
        .map(|axis| {
            let start = coord[axis].saturating_mul(visit.source_chunks[axis]);
            let end = start
                .saturating_add(visit.source_chunks[axis])
                .min(visit.shape[axis]);
            end.saturating_sub(start).max(1)
        })
        .collect();
    let partial_bytes = axis_product_bytes(&partial, visit.elem_size);
    let full_bytes = axis_product_bytes(visit.source_chunks, visit.elem_size);
    if Some(decoded.len()) == partial_bytes {
        return Ok((decoded, partial));
    }
    if Some(decoded.len()) == full_bytes {
        return Ok((decoded, visit.source_chunks.to_vec()));
    }
    Err(ConvertError::Zarr(format!(
        "chunk {} byte length mismatch (got {})",
        visit.objects.location(&key),
        decoded.len()
    )))
}

fn axis_product_bytes(dims: &[u64], elem_size: usize) -> Option<usize> {
    let mut acc = elem_size;
    for dim in dims {
        acc = acc.checked_mul(usize::try_from(*dim).ok()?)?;
    }
    Some(acc)
}

struct Blit<'a> {
    src: &'a [u8],
    layout: &'a [u64],
    src_origin: &'a [u64],
    tile_origin: &'a [u64],
    overlap: &'a [(u64, u64)],
    tile_extent: &'a [u64],
    elem_size: usize,
    ndim: usize,
}

fn blit_overlap(
    blit: &Blit<'_>,
    axis: usize,
    src_local: &mut [u64],
    dst_local: &mut [u64],
    out: &mut [u8],
) -> Result<(), ConvertError> {
    let (start, end) = blit.overlap[axis];
    if axis + 1 == blit.ndim {
        let len = usize::try_from(end - start).map_err(map_usize_err)?;
        src_local[axis] = start - blit.src_origin[axis];
        dst_local[axis] = start - blit.tile_origin[axis];
        let src_off = linear_u64(src_local, blit.layout, blit.ndim, blit.elem_size)?;
        let dst_off = linear_u64(dst_local, blit.tile_extent, blit.ndim, blit.elem_size)?;
        let nbytes = len.saturating_mul(blit.elem_size);
        out[dst_off..dst_off + nbytes].copy_from_slice(&blit.src[src_off..src_off + nbytes]);
        return Ok(());
    }
    for index in start..end {
        src_local[axis] = index - blit.src_origin[axis];
        dst_local[axis] = index - blit.tile_origin[axis];
        blit_overlap(blit, axis + 1, src_local, dst_local, out)?;
    }
    Ok(())
}

fn linear_u64(
    idx: &[u64],
    shape: &[u64],
    ndim: usize,
    elem_size: usize,
) -> Result<usize, ConvertError> {
    let mut stride = elem_size;
    let mut offset = 0usize;
    for axis in (0..ndim).rev() {
        let index = usize::try_from(idx[axis]).map_err(map_usize_err)?;
        offset = offset.saturating_add(index.saturating_mul(stride));
        let dim = usize::try_from(shape[axis]).map_err(map_usize_err)?;
        stride = stride.saturating_mul(dim);
    }
    Ok(offset)
}

fn collect_zarr_plans(
    objects: &dyn ZarrObjects,
    rel: &str,
    catalog_prefix: &str,
    chunk_bytes: Option<u64>,
    plans: &mut Vec<ImportPlan>,
    skipped: &mut Vec<super::ConvertSkip>,
) -> Result<(), ConvertError> {
    let meta = read_zarr_meta(objects, rel)?;
    if meta.zarr_format != 3 {
        return Err(ConvertError::Zarr(format!(
            "unsupported zarr_format {} at {}",
            meta.zarr_format,
            objects.location(rel)
        )));
    }
    match meta.node_type.as_str() {
        "array" => note_plan(
            plan_zarr_array(catalog_prefix, rel, &meta, chunk_bytes),
            plans,
            skipped,
        )?,
        "group" => {
            for child in objects.child_names(rel, true)? {
                let child_rel = join_rel(rel, &child);
                let child_catalog = join_catalog_path(catalog_prefix, &child);
                collect_zarr_plans(
                    objects,
                    &child_rel,
                    &child_catalog,
                    chunk_bytes,
                    plans,
                    skipped,
                )?;
            }
        }
        other => {
            return Err(ConvertError::Zarr(format!(
                "unsupported node_type `{other}` at {}",
                objects.location(rel)
            )));
        }
    }
    Ok(())
}

fn plan_zarr_array(
    name: &str,
    array_rel: &str,
    meta: &ZarrNodeMeta,
    chunk_bytes: Option<u64>,
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
    let source = chunk_shape_for_import(&shape, zarr_chunks);
    Ok(finish_zarr_plan(ZarrPlanBuild {
        name,
        array_rel,
        dtype,
        shape,
        source,
        zstd: zarr_chunk_payload_zstd(meta),
        v2_sep: None,
        attrs: zarr_array_attrs(&meta.attributes),
        chunk_bytes,
    }))
}

fn map_zarr_dtype(
    name: &str,
    data_type: Option<&serde_json::Value>,
) -> Result<ElementDtype, ConvertError> {
    if data_type.is_none() {
        return Err(ConvertError::Zarr(format!(
            "array `{name}` missing data_type"
        )));
    }
    element_dtype_from_zarr_v3(data_type).ok_or_else(|| {
        let tag = data_type
            .and_then(|v| v.as_str())
            .or_else(|| {
                data_type
                    .and_then(|v| v.get("name"))
                    .and_then(|v| v.as_str())
            })
            .unwrap_or("<missing>");
        ConvertError::UnsupportedDtype {
            name: name.to_owned(),
            detail: format!("zarr data_type `{tag}`"),
        }
    })
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

struct ZarrChunkRead<'a> {
    array_rel: &'a str,
    zstd: bool,
    v2_sep: Option<u8>,
    chunk_coord: &'a [u64],
    ndim: usize,
    chunk_shape: &'a [u64],
    elem_size: usize,
}

fn read_zarr_chunk_le(
    objects: &dyn ZarrObjects,
    read: &ZarrChunkRead<'_>,
) -> Result<Vec<u8>, ConvertError> {
    let key = zarr_chunk_key(read.array_rel, read.v2_sep, read.chunk_coord, read.ndim);
    let on_disk = objects.read_bytes(&key)?;
    let decoded = if read.zstd {
        zstd::decode_all(on_disk.as_slice()).map_err(|e| ConvertError::Zarr(e.to_string()))?
    } else {
        on_disk
    };
    let expected = read
        .chunk_shape
        .iter()
        .take(read.ndim)
        .fold(read.elem_size, |acc, &dim| {
            acc.saturating_mul(usize::try_from(dim).unwrap_or(0))
        });
    if decoded.len() != expected {
        return Err(ConvertError::Zarr(format!(
            "chunk {} byte length mismatch (expected {expected}, got {})",
            objects.location(&key),
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

fn read_zarr_meta(objects: &dyn ZarrObjects, rel: &str) -> Result<ZarrNodeMeta, ConvertError> {
    let key = join_rel(rel, "zarr.json");
    let raw = objects.read_bytes(&key)?;
    serde_json::from_slice(&raw)
        .map_err(|err| ConvertError::Zarr(format!("{}: {err}", objects.location(&key))))
}

fn collect_zarr_v2_plans(
    objects: &dyn ZarrObjects,
    rel: &str,
    catalog_prefix: &str,
    chunk_bytes: Option<u64>,
    plans: &mut Vec<ImportPlan>,
    skipped: &mut Vec<super::ConvertSkip>,
) -> Result<(), ConvertError> {
    if objects.is_file(&join_rel(rel, ".zarray"))? {
        let name = if catalog_prefix.is_empty() {
            "data".to_owned()
        } else {
            catalog_prefix.to_owned()
        };
        return note_plan(
            plan_zarr_v2_array(&name, rel, objects, chunk_bytes),
            plans,
            skipped,
        );
    }
    if !objects.is_file(&join_rel(rel, ".zgroup"))? {
        return Err(ConvertError::Zarr(format!(
            "zarr v2 node {} has neither .zarray nor .zgroup",
            objects.location(rel)
        )));
    }
    for child in objects.child_names(rel, false)? {
        let child_rel = join_rel(rel, &child);
        let child_catalog = join_catalog_path(catalog_prefix, &child);
        collect_zarr_v2_plans(
            objects,
            &child_rel,
            &child_catalog,
            chunk_bytes,
            plans,
            skipped,
        )?;
    }
    Ok(())
}

fn plan_zarr_v2_array(
    name: &str,
    array_rel: &str,
    objects: &dyn ZarrObjects,
    chunk_bytes: Option<u64>,
) -> Result<ImportPlan, ConvertError> {
    let meta_key = join_rel(array_rel, ".zarray");
    let raw = objects.read_bytes(&meta_key)?;
    let meta: ZarrV2Array = serde_json::from_slice(&raw)
        .map_err(|err| ConvertError::Zarr(format!("{}: {err}", objects.location(&meta_key))))?;
    if meta.zarr_format != 2 {
        return Err(ConvertError::Zarr(format!(
            "array `{name}` has zarr_format {}, expected 2",
            meta.zarr_format
        )));
    }
    if meta.shape.is_empty() || meta.shape.contains(&0) {
        return Err(ConvertError::UnsupportedDtype {
            name: name.to_owned(),
            detail: "empty zarr array".into(),
        });
    }
    if meta.order != "C" {
        return Err(ConvertError::UnsupportedDtype {
            name: name.to_owned(),
            detail: format!("zarr order `{}` (only C-order is supported)", meta.order),
        });
    }
    let dtype = map_zarr_v2_dtype(name, &meta.dtype)?;
    let zstd = zarr_v2_zstd(name, &meta)?;
    let sep = match meta.dimension_separator.as_deref() {
        Some("/") => b'/',
        Some(".") | None => b'.',
        Some(other) => {
            return Err(ConvertError::UnsupportedDtype {
                name: name.to_owned(),
                detail: format!("zarr dimension_separator `{other}`"),
            });
        }
    };
    let chunks = meta
        .chunks
        .iter()
        .map(|dim| usize::try_from(*dim).unwrap_or(0))
        .collect();
    let source = chunk_shape_for_import(&meta.shape, Some(chunks));
    let attrs = read_zarr_v2_attrs(objects, array_rel);
    Ok(finish_zarr_plan(ZarrPlanBuild {
        name,
        array_rel,
        dtype,
        shape: meta.shape,
        source,
        zstd,
        v2_sep: Some(sep),
        attrs,
        chunk_bytes,
    }))
}

struct ZarrPlanBuild<'a> {
    name: &'a str,
    array_rel: &'a str,
    dtype: ElementDtype,
    shape: Vec<u64>,
    source: Vec<u64>,
    zstd: bool,
    v2_sep: Option<u8>,
    attrs: std::collections::BTreeMap<String, String>,
    chunk_bytes: Option<u64>,
}

/// Apply `--chunk-mib` on top of the Zarr chunk grid, keeping the source grid for reads.
fn finish_zarr_plan(build: ZarrPlanBuild<'_>) -> ImportPlan {
    let chunk_shape = match build.chunk_bytes {
        Some(bytes) => numpy_chunk_shape(&build.shape, build.dtype.elem_size(), bytes),
        None => build.source.clone(),
    };
    let mut plan = ImportPlan::new(
        build.name.to_owned(),
        build.dtype,
        build.shape,
        chunk_shape.clone(),
        None,
    )
    .with_zarr(build.array_rel.to_owned(), build.zstd)
    .with_import(build.attrs, None, None);
    if let Some(sep) = build.v2_sep {
        plan = plan.with_zarr_v2_keys(sep);
    }
    if chunk_shape != build.source {
        plan = plan.with_zarr_source_chunks(build.source);
    }
    plan
}

fn map_zarr_v2_dtype(name: &str, dtype: &str) -> Result<ElementDtype, ConvertError> {
    match parse_zarr_v2_dtype(dtype) {
        ZarrV2DtypeParse::Ok(elem) => Ok(elem),
        ZarrV2DtypeParse::BigEndian => Err(ConvertError::UnsupportedDtype {
            name: name.to_owned(),
            detail: format!("big-endian zarr dtype `{dtype}`"),
        }),
        ZarrV2DtypeParse::Unsupported => Err(ConvertError::UnsupportedDtype {
            name: name.to_owned(),
            detail: format!("zarr dtype `{dtype}`"),
        }),
    }
}

fn zarr_v2_zstd(name: &str, meta: &ZarrV2Array) -> Result<bool, ConvertError> {
    if let Some(filters) = &meta.filters
        && !(filters.is_null() || filters.as_array().is_some_and(Vec::is_empty))
    {
        return Err(ConvertError::UnsupportedDtype {
            name: name.to_owned(),
            detail: "zarr v2 filters are not supported".into(),
        });
    }
    match &meta.compressor {
        None => Ok(false),
        Some(value) if value.is_null() => Ok(false),
        Some(value) => {
            let id = value.get("id").and_then(|id| id.as_str()).unwrap_or("");
            if id == "zstd" {
                Ok(true)
            } else {
                Err(ConvertError::UnsupportedDtype {
                    name: name.to_owned(),
                    detail: format!("zarr v2 compressor `{id}`"),
                })
            }
        }
    }
}

fn read_zarr_v2_attrs(
    objects: &dyn ZarrObjects,
    array_rel: &str,
) -> std::collections::BTreeMap<String, String> {
    let key = join_rel(array_rel, ".zattrs");
    if !objects.is_file(&key).unwrap_or(false) {
        return std::collections::BTreeMap::new();
    }
    let Ok(raw) = objects.read_bytes(&key) else {
        return std::collections::BTreeMap::new();
    };
    let Ok(text) = String::from_utf8(raw) else {
        return std::collections::BTreeMap::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&text)
    else {
        return std::collections::BTreeMap::new();
    };
    zarr_array_attrs(&value)
}

/// Returns true when `path` is a directory containing Zarr v2 root metadata.
#[must_use]
pub fn is_zarr_v2_directory(path: &Path) -> bool {
    path.is_dir() && (path.join(".zgroup").is_file() || path.join(".zarray").is_file())
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

#[derive(Debug, Deserialize)]
struct ZarrV2Array {
    zarr_format: u32,
    shape: Vec<u64>,
    chunks: Vec<u64>,
    dtype: String,
    order: String,
    #[serde(default)]
    compressor: Option<serde_json::Value>,
    #[serde(default)]
    filters: Option<serde_json::Value>,
    #[serde(default)]
    dimension_separator: Option<String>,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    use crate::convert::zarr_io::{ZarrObjects, zarr_child_names_from_keys};
    use crate::convert::{ConvertError, convert_to_tet};
    use crate::query::{ExecuteQueryOptions, execute_query_json};

    struct MemZarr {
        files: BTreeMap<String, Vec<u8>>,
        reads: Mutex<Vec<String>>,
    }

    impl ZarrObjects for MemZarr {
        fn read_bytes(&self, rel: &str) -> Result<Vec<u8>, ConvertError> {
            self.reads.lock().expect("reads").push(rel.to_owned());
            self.files
                .get(rel)
                .cloned()
                .ok_or_else(|| ConvertError::Zarr(format!("missing {rel}")))
        }

        fn is_file(&self, rel: &str) -> Result<bool, ConvertError> {
            Ok(self.files.contains_key(rel))
        }

        fn child_names(&self, rel: &str, v3: bool) -> Result<Vec<String>, ConvertError> {
            let keys: Vec<String> = self.files.keys().cloned().collect();
            Ok(zarr_child_names_from_keys(&keys, rel, v3))
        }

        fn location(&self, rel: &str) -> String {
            rel.to_owned()
        }
    }

    fn sum_of(path: &std::path::Path) -> f64 {
        let bytes = std::fs::read(path).unwrap();
        let summary = crate::catalog::read_tet_summary_v1(&bytes).unwrap();
        let name = &summary.datasets[0].name;
        let json = format!(r#"{{"dataset":"{name}","sum":[]}}"#);
        let resp = execute_query_json(
            &json,
            path,
            &bytes,
            ExecuteQueryOptions::execute_no_preview(),
            None,
        )
        .unwrap();
        resp.execution.unwrap().operation_sum.unwrap()
    }

    #[test]
    fn object_convert_reads_chunks_and_skips_unrelated_keys() {
        let mut files = BTreeMap::new();
        files.insert(
            "zarr.json".into(),
            br#"{"zarr_format":3,"node_type":"group"}"#.to_vec(),
        );
        files.insert(
            "data/zarr.json".into(),
            br#"{"zarr_format":3,"node_type":"array","shape":[4],"data_type":"float32","chunk_grid":{"name":"regular","configuration":{"chunk_shape":[2]}},"codecs":[{"name":"bytes","configuration":{"endian":"little"}}]}"#.to_vec(),
        );
        let mut chunk0 = Vec::new();
        chunk0.extend(1.0f32.to_le_bytes());
        chunk0.extend(2.0f32.to_le_bytes());
        let mut chunk1 = Vec::new();
        chunk1.extend(3.0f32.to_le_bytes());
        chunk1.extend(4.0f32.to_le_bytes());
        files.insert("data/c/0".into(), chunk0);
        files.insert("data/c/1".into(), chunk1);
        files.insert("noise.bin".into(), b"do-not-download".to_vec());

        let dir = tempfile::tempdir().unwrap();
        let local = dir.path().join("store");
        for (rel, bytes) in &files {
            if rel == "noise.bin" {
                continue;
            }
            let path = local.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, bytes).unwrap();
        }
        let local_tet = dir.path().join("local.tet");
        let remote_tet = dir.path().join("remote.tet");
        let local_report = convert_to_tet(&local, &local_tet).unwrap();

        let objects = Arc::new(MemZarr {
            files,
            reads: Mutex::new(Vec::new()),
        });
        let as_trait: Arc<dyn ZarrObjects> = objects.clone();
        let report = super::convert_zarr_from_objects(
            std::path::Path::new("mem://store"),
            as_trait,
            &remote_tet,
            2,
            false,
            None,
            None::<fn(_)>,
        )
        .unwrap();
        assert_eq!(report.dataset_names, local_report.dataset_names);
        assert_eq!(report.history[0].source, local_report.history[0].source);
        assert!((sum_of(&local_tet) - sum_of(&remote_tet)).abs() < 1e-6);
        assert!((sum_of(&remote_tet) - 10.0).abs() < 1e-6);

        let reads = objects.reads.lock().expect("reads");
        assert!(reads.iter().any(|key| key == "data/c/0"));
        assert!(reads.iter().any(|key| key == "data/c/1"));
        assert!(!reads.iter().any(|key| key == "noise.bin"));
    }
}

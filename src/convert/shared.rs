//! Shared helpers for foreign-format import.

use std::collections::BTreeMap;

use super::cf::CfTransform;

use crate::catalog::{
    ArrayWriteMeta, CatalogError, CoordAxisV1, StreamTileJob, StreamWriteProgress,
    write_multi_raw_array_streaming,
};
use crate::utils::dtype::ElementDtype;

use super::{ConvertError, ConvertSkip};

/// Dataset geometry collected before a streaming write.
#[derive(Clone)]
pub(crate) struct ImportPlan {
    pub name: String,
    pub dtype: ElementDtype,
    pub shape: Vec<u64>,
    pub chunk_shape: Vec<u64>,
    pub cf: Option<CfTransform>,
    /// Zarr array path relative to store root (`primary/f32`); `None` for HDF5/NetCDF.
    pub zarr_array_rel: Option<String>,
    /// When importing Zarr: chunk files on disk are zstd-compressed (`bytes` + `zstd` codecs).
    pub zarr_zstd: bool,
    /// Zarr v2 chunk-key separator (`b'.'` or `b'/'`). `None` is a v3 `c/0/0` key.
    pub zarr_v2_sep: Option<u8>,
    /// Zarr on-disk chunk grid when [`Self::chunk_shape`] was retiled by `--chunk-mib`.
    /// `None` means chunk keys match `chunk_shape`.
    pub zarr_source_chunks: Option<Vec<u64>>,
    /// Dataset attributes copied into footer `metadata.datasets[name].attrs`.
    pub import_attrs: BTreeMap<String, String>,
    /// `NetCDF` dimension names → `metadata.datasets[name].dim_names`.
    pub import_dim_names: Option<Vec<String>>,
    /// Inline coordinate labels → `metadata.datasets[name].coords`.
    pub import_coords: Option<BTreeMap<String, CoordAxisV1>>,
}

pub(crate) fn join_catalog_path(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}/{name}")
    }
}

/// One tile read during import: geometry from the plan plus the chunk coordinate from the stream job.
#[derive(Clone, Copy)]
pub(crate) struct ImportTileRead<'a> {
    pub dtype: ElementDtype,
    pub shape: &'a [u64],
    pub chunk_shape: &'a [u64],
    pub chunk_coord: &'a [u64],
    pub ndim: usize,
    pub cf: Option<CfTransform>,
    /// Zarr source chunk grid when it differs from `chunk_shape`.
    pub zarr_source_chunks: Option<&'a [u64]>,
}

impl ImportPlan {
    /// Numeric dataset geometry with empty import metadata and no Zarr path.
    pub(crate) fn new(
        name: String,
        dtype: ElementDtype,
        shape: Vec<u64>,
        chunk_shape: Vec<u64>,
        cf: Option<CfTransform>,
    ) -> Self {
        Self {
            name,
            dtype,
            shape,
            chunk_shape,
            cf,
            zarr_array_rel: None,
            zarr_zstd: false,
            zarr_v2_sep: None,
            zarr_source_chunks: None,
            import_attrs: BTreeMap::new(),
            import_dim_names: None,
            import_coords: None,
        }
    }

    /// Attach Zarr v3 store path and whether on-disk chunk files are zstd-wrapped.
    pub(crate) fn with_zarr(mut self, array_rel: String, zstd: bool) -> Self {
        self.zarr_array_rel = Some(array_rel);
        self.zarr_zstd = zstd;
        self
    }

    /// Mark this plan as a Zarr v2 array whose chunk keys use `sep` (`.` or `/`).
    pub(crate) fn with_zarr_v2_keys(mut self, sep: u8) -> Self {
        self.zarr_v2_sep = Some(sep);
        self
    }

    /// Remember the Zarr chunk grid so a retiled `.tet` grid can be assembled from those files.
    pub(crate) fn with_zarr_source_chunks(mut self, chunks: Vec<u64>) -> Self {
        self.zarr_source_chunks = Some(chunks);
        self
    }

    /// Footer metadata fields copied into `THST` on convert finish.
    pub(crate) fn with_import(
        mut self,
        attrs: BTreeMap<String, String>,
        dim_names: Option<Vec<String>>,
        coords: Option<BTreeMap<String, CoordAxisV1>>,
    ) -> Self {
        self.import_attrs = attrs;
        self.import_dim_names = dim_names;
        self.import_coords = coords;
        self
    }

    pub(crate) fn tile_read<'a>(&'a self, job: &'a StreamTileJob<'_>) -> ImportTileRead<'a> {
        ImportTileRead {
            dtype: self.dtype,
            shape: &self.shape,
            chunk_shape: &self.chunk_shape,
            chunk_coord: &job.chunk_coord[..job.ndim],
            ndim: job.ndim,
            cf: self.cf,
            zarr_source_chunks: self.zarr_source_chunks.as_deref(),
        }
    }
}

/// Pick a v1 chunk grid: reuse source chunking when it matches rank, else one tile per array.
pub(crate) fn chunk_shape_for_import(shape: &[u64], source_chunks: Option<Vec<usize>>) -> Vec<u64> {
    if let Some(chunks) = source_chunks
        && chunks.len() == shape.len()
        && chunks.iter().all(|&c| c > 0)
    {
        let mut out = Vec::with_capacity(chunks.len());
        for (&dim, &chunk) in shape.iter().zip(chunks.iter()) {
            let c = u64::try_from(chunk).unwrap_or(dim);
            out.push(c.min(dim).max(1));
        }
        return out;
    }
    shape.to_vec()
}

/// `.tet` chunk grid for one dataset.
///
/// `target_bytes` (`--chunk-mib`) replaces the source grid with a leading-axis tile of about
/// that many bytes. `None` reuses `source_chunks` when the rank matches, otherwise one tile
/// for the whole array.
pub(crate) fn import_chunk_shape(
    shape: &[u64],
    elem_size: usize,
    source_chunks: Option<Vec<usize>>,
    target_bytes: Option<u64>,
) -> Vec<u64> {
    if let Some(bytes) = target_bytes {
        return super::numpy::numpy_chunk_shape(shape, elem_size, bytes);
    }
    chunk_shape_for_import(shape, source_chunks)
}

/// Turn `--chunk-mib` into a byte target. `None` leaves each importer on its default grid.
///
/// # Errors
///
/// Returns [`ConvertError::ChunkMibZero`] when `chunk_mib` is `Some(0)`.
pub(crate) fn chunk_target_bytes(chunk_mib: Option<u32>) -> Result<Option<u64>, ConvertError> {
    match chunk_mib {
        None => Ok(None),
        Some(0) => Err(ConvertError::ChunkMibZero),
        Some(mib) => Ok(Some(u64::from(mib).saturating_mul(1024 * 1024))),
    }
}

pub(crate) fn write_plans_streaming(
    output: &std::path::Path,
    plans: &[ImportPlan],
    parallel_jobs: usize,
    fill_tile: impl Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), ConvertError> + Sync + Send,
    progress_hook: Option<&mut StreamWriteProgress<'_>>,
) -> Result<Vec<crate::catalog::ChunkZoneStat>, ConvertError> {
    let mut metas: Vec<ArrayWriteMeta<'_>> = Vec::with_capacity(plans.len());
    let mut fills: Vec<Option<Vec<u8>>> = Vec::with_capacity(plans.len());
    for plan in plans {
        metas.push(ArrayWriteMeta::row_major(
            &plan.name,
            plan.dtype.wire_tag(),
            &plan.shape,
            &plan.chunk_shape,
            None,
        ));
        fills.push(crate::catalog::zone::fill_le_bytes(
            plan.dtype.wire_tag(),
            Some(&plan.import_attrs),
        ));
    }
    write_multi_raw_array_streaming(
        output,
        &metas,
        parallel_jobs,
        |job, buf| match fill_tile(job, buf) {
            Ok(()) => Ok(()),
            Err(ConvertError::Catalog(c)) => Err(c),
            Err(e) => Err(CatalogError::Io(std::io::Error::other(e.to_string()))),
        },
        progress_hook,
        Some(fills.as_slice()),
    )
    .map_err(ConvertError::from)
}

pub(crate) fn ensure_non_empty(
    path: &std::path::Path,
    names: &[String],
) -> Result<(), ConvertError> {
    if names.is_empty() {
        return Err(ConvertError::NoDatasets {
            path: path.display().to_string(),
        });
    }
    Ok(())
}

/// Keep a planned dataset, or record an unsupported dtype as a skip.
pub(crate) fn note_plan(
    result: Result<ImportPlan, ConvertError>,
    plans: &mut Vec<ImportPlan>,
    skipped: &mut Vec<ConvertSkip>,
) -> Result<(), ConvertError> {
    match result {
        Ok(plan) => {
            plans.push(plan);
            Ok(())
        }
        Err(ConvertError::UnsupportedDtype { name, detail }) => {
            skipped.push(ConvertSkip {
                path: name,
                reason: detail,
            });
            Ok(())
        }
        Err(err) => Err(err),
    }
}

pub(crate) fn record_skip(skipped: &mut Vec<ConvertSkip>, path: impl Into<String>, reason: &str) {
    skipped.push(ConvertSkip {
        path: path.into(),
        reason: reason.to_owned(),
    });
}

/// `--strict`: refuse to write when the walk skipped any dataset.
pub(crate) fn ensure_strict(
    path: &std::path::Path,
    skipped: &[ConvertSkip],
    strict: bool,
) -> Result<(), ConvertError> {
    if !strict || skipped.is_empty() {
        return Ok(());
    }
    let detail = skipped
        .iter()
        .map(|skip| format!("{} ({})", skip.path, skip.reason))
        .collect::<Vec<_>>()
        .join("; ");
    Err(ConvertError::StrictSkips {
        path: path.display().to_string(),
        count: skipped.len(),
        detail,
    })
}

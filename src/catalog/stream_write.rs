//! Memory-efficient `.tet` writer: one chunk payload at a time (raw codec).

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use rayon::prelude::*;

use super::dataset::encode_dataset_blob;
use super::execution::FileExecutionSettingsV1;
use super::index::ChunkIndexEntryV1;
use super::tile;
use super::zone::{ChunkZoneStat, chunk_zone_stat_from_raw};
use super::{
    CHUNK_PAYLOAD_CODEC_V1, CatalogError, MAX_NDIM, elem_size_for_wire_tag, usize_from_u64,
};

/// Zone stats accumulated while streaming payloads (for embed without a second payload scan).
#[derive(Debug)]
pub struct StreamingWriteAcc {
    pub zones: Vec<ChunkZoneStat>,
}

impl StreamingWriteAcc {
    /// Consume the accumulator and return write-time zone stats.
    #[must_use]
    pub fn into_zones(self) -> Vec<ChunkZoneStat> {
        self.zones
    }
}

/// Dataset metadata for streaming writes (no in-memory tensor buffer).
#[derive(Debug, Clone)]
pub struct ArrayWriteMeta<'a> {
    pub name: &'a str,
    pub dtype: u32,
    pub shape: &'a [u64],
    pub chunk_shape: &'a [u64],
    pub chunk_codec: u32,
    pub file_execution: Option<FileExecutionSettingsV1>,
}

impl<'a> ArrayWriteMeta<'a> {
    /// Streaming meta with raw chunk codec (**0**); used by convert and [`TetWriterSession`].
    #[must_use]
    pub fn row_major(
        name: &'a str,
        dtype: u32,
        shape: &'a [u64],
        chunk_shape: &'a [u64],
        file_execution: Option<FileExecutionSettingsV1>,
    ) -> Self {
        Self {
            name,
            dtype,
            shape,
            chunk_shape,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            file_execution,
        }
    }
}

/// Progress hook: `(chunks_done, chunks_total, current_dataset_name)`.
pub type StreamWriteProgress<'a> = dyn FnMut(u64, u64, &str) + 'a;

/// Identifies one chunk tile the caller must fill during a streaming write.
#[derive(Debug, Clone)]
pub struct StreamTileJob<'a> {
    pub dataset_id: usize,
    pub dataset_name: &'a str,
    pub chunk_k: u64,
    pub chunk_coord: [u64; MAX_NDIM],
    pub ndim: usize,
    pub raw_byte_len: u64,
}

/// Validate [`ArrayWriteMeta`] (shape/chunk grid/dtype/codec) without requiring tensor bytes.
///
/// # Errors
///
/// Returns [`CatalogError`] when arguments are inconsistent.
pub fn validate_array_write_meta(spec: &ArrayWriteMeta<'_>) -> Result<(), CatalogError> {
    super::dataset::validate_tensor_geometry(spec.shape, spec.chunk_shape, spec.dtype)?;
    if !CHUNK_PAYLOAD_CODEC_V1.is_raw(spec.chunk_codec) {
        return Err(CatalogError::InvalidWriteSpec(
            "streaming write supports raw chunk codec (0) only",
        ));
    }
    Ok(())
}

/// Total chunk count across all datasets in `specs` (delegates to [`file_layout::sum_chunk_counts`]).
///
/// # Errors
///
/// Returns [`CatalogError::InvalidWriteSpec`] when a per-dataset chunk grid count overflows or
/// the summed total exceeds `u64::MAX`.
pub fn total_chunk_count_for_meta(specs: &[ArrayWriteMeta<'_>]) -> Result<u64, CatalogError> {
    super::file_layout::sum_chunk_counts(specs.iter().map(|s| (s.shape, s.chunk_shape)))
}

/// Write a `.tet` by filling one tile at a time via `fill_tile` (peak RAM ≈ one chunk).
///
/// When `parallel_jobs` is greater than 1, reads up to that many chunks concurrently via
/// Rayon, then writes payloads in order (peak RAM ≈ `parallel_jobs` × largest tile).
///
/// `on_progress`, when set, is invoked after each payload is flushed (`chunks_done`, `chunks_total`,
/// current dataset name).
///
/// Returns write-time zone stats for standalone embed (no full-file SHA-256).
/// `dataset_fills[dataset_id]` is optional fill-value LE bytes.
///
/// # Errors
///
/// Returns [`CatalogError`] when layout rules are violated, `fill_tile` fails, or I/O fails.
pub fn write_multi_raw_array_streaming(
    path: &Path,
    specs: &[ArrayWriteMeta<'_>],
    parallel_jobs: usize,
    fill_tile: impl Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), CatalogError> + Sync + Send,
    on_progress: Option<&mut StreamWriteProgress<'_>>,
    dataset_fills: Option<&[Option<Vec<u8>>]>,
) -> Result<StreamingWriteAcc, CatalogError> {
    if specs.is_empty() {
        return Err(CatalogError::InvalidWriteSpec(
            "at least one dataset is required",
        ));
    }
    for spec in specs {
        validate_array_write_meta(spec)?;
    }

    let blob = encode_all_dataset_blobs(specs)?;
    let dataset_blob_len = blob.len() as u64;
    let n_chunks_total = total_chunk_count_for_meta(specs)?;
    let (index_base, chunk_index_length, payload_start) =
        super::file_layout::chunk_index_layout(dataset_blob_len, n_chunks_total)?;
    let (entries, jobs) = build_stream_index_and_jobs(specs, payload_start, n_chunks_total)?;
    let sb = super::file_layout::layout_superblock(specs.len(), index_base, chunk_index_length)?;
    let file_execution = specs[0]
        .file_execution
        .unwrap_or_else(FileExecutionSettingsV1::default_engine);
    let index_bytes = super::file_layout::build_chunk_index_bytes(
        &entries,
        n_chunks_total,
        chunk_index_length,
        file_execution,
    )?;

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    super::file_layout::write_file_preamble(&mut f, &sb, &blob, index_base, &index_bytes)?;
    let parallel_jobs = parallel_jobs.max(1);
    let zones = if parallel_jobs == 1 {
        write_filled_tiles_sequential(
            &mut f,
            specs,
            &entries,
            &jobs,
            n_chunks_total,
            &fill_tile,
            on_progress,
            dataset_fills,
        )?
    } else {
        write_filled_tiles_parallel(
            &mut f,
            specs,
            &entries,
            &jobs,
            n_chunks_total,
            parallel_jobs,
            &fill_tile,
            on_progress,
            dataset_fills,
        )?
    };
    f.sync_all()?;
    Ok(StreamingWriteAcc { zones })
}

fn tile_zone_stat(
    specs: &[ArrayWriteMeta<'_>],
    job: &StreamTileJob<'_>,
    tile_buf: &[u8],
    dataset_fills: Option<&[Option<Vec<u8>>]>,
) -> Result<ChunkZoneStat, CatalogError> {
    let spec = specs
        .get(job.dataset_id)
        .ok_or(CatalogError::InvalidWriteSpec(
            "stream job dataset_id out of range",
        ))?;
    let elem_size = elem_size_for_wire_tag(spec.dtype)?;
    let fill = dataset_fills
        .and_then(|fills| fills.get(job.dataset_id))
        .and_then(|f| f.as_deref());
    chunk_zone_stat_from_raw(
        u64::try_from(job.dataset_id).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "dataset_id",
            value: u64::MAX,
        })?,
        &job.chunk_coord[..job.ndim],
        spec.dtype,
        elem_size,
        tile_buf,
        fill,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_filled_tiles_sequential(
    f: &mut File,
    specs: &[ArrayWriteMeta<'_>],
    entries: &[ChunkIndexEntryV1],
    jobs: &[StreamTileJob<'_>],
    n_chunks_total: u64,
    fill_tile: &(impl Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), CatalogError> + Sync + Send),
    mut on_progress: Option<&mut StreamWriteProgress<'_>>,
    dataset_fills: Option<&[Option<Vec<u8>>]>,
) -> Result<Vec<ChunkZoneStat>, CatalogError> {
    let sequential = entries_are_sequential(entries);
    if sequential && !entries.is_empty() {
        let pos = f.stream_position()?;
        if pos != entries[0].payload_offset {
            f.seek(SeekFrom::Start(entries[0].payload_offset))?;
        }
    }

    let mut zones = Vec::with_capacity(jobs.len());
    let mut tile_buf = Vec::new();
    for (i, job) in jobs.iter().enumerate() {
        let entry = &entries[i];
        prepare_tile_buffer(&mut tile_buf, job.raw_byte_len)?;
        fill_tile(job, &mut tile_buf)?;
        if tile_buf.len() as u64 != entry.raw_byte_len {
            return Err(CatalogError::InvalidWriteSpec(
                "fill_tile returned wrong byte length for chunk",
            ));
        }
        zones.push(tile_zone_stat(specs, job, &tile_buf, dataset_fills)?);
        write_tile_payload(f, entry, &tile_buf, sequential)?;
        if let Some(ref mut progress) = on_progress {
            progress(
                u64::try_from(i + 1).unwrap_or(u64::MAX),
                n_chunks_total,
                job.dataset_name,
            );
        }
    }
    Ok(zones)
}

#[allow(clippy::too_many_arguments)]
fn write_filled_tiles_parallel(
    f: &mut File,
    specs: &[ArrayWriteMeta<'_>],
    entries: &[ChunkIndexEntryV1],
    jobs: &[StreamTileJob<'_>],
    n_chunks_total: u64,
    parallel_jobs: usize,
    fill_tile: &(impl Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), CatalogError> + Sync + Send),
    mut on_progress: Option<&mut StreamWriteProgress<'_>>,
    dataset_fills: Option<&[Option<Vec<u8>>]>,
) -> Result<Vec<ChunkZoneStat>, CatalogError> {
    let sequential = entries_are_sequential(entries);
    if sequential && !entries.is_empty() {
        let pos = f.stream_position()?;
        if pos != entries[0].payload_offset {
            f.seek(SeekFrom::Start(entries[0].payload_offset))?;
        }
    }

    let mut zones = Vec::with_capacity(jobs.len());
    let mut i = 0usize;
    while i < jobs.len() {
        let end = (i + parallel_jobs).min(jobs.len());
        let mut batch: Vec<(usize, Vec<u8>)> = (i..end)
            .into_par_iter()
            .map(|idx| {
                let job = &jobs[idx];
                let mut buf = Vec::new();
                prepare_tile_buffer(&mut buf, job.raw_byte_len)?;
                fill_tile(job, &mut buf)?;
                Ok((idx, buf))
            })
            .collect::<Result<Vec<_>, CatalogError>>()?;
        batch.sort_by_key(|(idx, _)| *idx);
        for (idx, tile_buf) in batch {
            let entry = &entries[idx];
            if tile_buf.len() as u64 != entry.raw_byte_len {
                return Err(CatalogError::InvalidWriteSpec(
                    "fill_tile returned wrong byte length for chunk",
                ));
            }
            zones.push(tile_zone_stat(specs, &jobs[idx], &tile_buf, dataset_fills)?);
            write_tile_payload(f, entry, &tile_buf, sequential)?;
            if let Some(ref mut progress) = on_progress {
                progress(
                    u64::try_from(idx + 1).unwrap_or(u64::MAX),
                    n_chunks_total,
                    jobs[idx].dataset_name,
                );
            }
        }
        i = end;
    }
    Ok(zones)
}

fn write_tile_payload(
    f: &mut File,
    entry: &ChunkIndexEntryV1,
    tile_buf: &[u8],
    sequential: bool,
) -> Result<(), CatalogError> {
    if sequential {
        f.write_all(tile_buf)?;
    } else {
        f.seek(SeekFrom::Start(entry.payload_offset))?;
        f.write_all(tile_buf)?;
    }
    Ok(())
}

fn encode_all_dataset_blobs(specs: &[ArrayWriteMeta<'_>]) -> Result<Vec<u8>, CatalogError> {
    let mut blob = Vec::new();
    for spec in specs {
        blob.extend_from_slice(&encode_dataset_blob(
            spec.name,
            spec.dtype,
            spec.shape,
            spec.chunk_shape,
        )?);
    }
    Ok(blob)
}

fn build_stream_index_and_jobs<'a>(
    specs: &'a [ArrayWriteMeta<'_>],
    payload_start: u64,
    n_chunks_total: u64,
) -> Result<(Vec<ChunkIndexEntryV1>, Vec<StreamTileJob<'a>>), CatalogError> {
    let mut entries: Vec<ChunkIndexEntryV1> =
        Vec::with_capacity(usize_from_u64("chunk_entry_count", n_chunks_total)?);
    let mut jobs: Vec<StreamTileJob<'_>> = Vec::with_capacity(entries.capacity());
    let mut cursor = payload_start;
    for (dataset_id, spec) in specs.iter().enumerate() {
        let grid = super::chunk_grid_plan(spec.shape, spec.chunk_shape, spec.dtype)?;

        for k in 0..grid.n_chunks {
            let coord = tile::chunk_coord_from_linear(k, &grid.counts, grid.ndim);
            let raw_len = tile::tile_raw_byte_len(
                spec.shape,
                spec.chunk_shape,
                &coord[..grid.ndim],
                grid.ndim,
                grid.elem_size,
            )?;
            let chunk_index = ChunkIndexEntryV1::padded_chunk_index(&coord[..grid.ndim], grid.ndim);
            entries.push(ChunkIndexEntryV1 {
                dataset_id: super::file_layout::wire_dataset_id(dataset_id)?,
                chunk_index,
                payload_offset: cursor,
                raw_byte_len: raw_len,
                stored_byte_len: raw_len,
                codec: spec.chunk_codec,
            });
            jobs.push(StreamTileJob {
                dataset_id,
                dataset_name: spec.name,
                chunk_k: k,
                chunk_coord: chunk_index,
                ndim: grid.ndim,
                raw_byte_len: raw_len,
            });
            cursor = cursor
                .checked_add(raw_len)
                .ok_or(CatalogError::InvalidWriteSpec("payload cursor overflow"))?;
        }
    }
    Ok((entries, jobs))
}

fn entries_are_sequential(entries: &[ChunkIndexEntryV1]) -> bool {
    entries
        .windows(2)
        .all(|pair| pair[0].payload_offset + pair[0].raw_byte_len == pair[1].payload_offset)
}

fn prepare_tile_buffer(tile_buf: &mut Vec<u8>, raw_byte_len: u64) -> Result<(), CatalogError> {
    let len = usize_from_u64("tile_buffer", raw_byte_len)?;
    tile_buf.clear();
    tile_buf.try_reserve(len).map_err(std::io::Error::from)?;
    // SAFETY: `fill_tile` must write every byte before we flush the buffer.
    unsafe {
        tile_buf.set_len(len);
    }
    Ok(())
}

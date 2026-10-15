//! Append datasets to an existing layout v1 `.tet` (rewrites catalog, index, and payloads).
//!
//! Reuses [`crate::catalog::file_layout`] for chunk grid math, index sizing, and
//! [`EncodedChunkPush`] when copying or appending tile payloads.

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use super::dataset::{self, RawArrayWrite};
use super::execution::FileExecutionSettingsV1;
use super::index::ChunkIndexEntryV1;
use super::stream_write::{ArrayWriteMeta, StreamTileJob};
use super::tile;
use super::{
    CHUNK_PAYLOAD_CODEC_V1, CatalogError, DatasetRecordV1, read_tet_summary_v1, usize_from_u64,
};

/// Append in-memory datasets to an existing file (rewrites the file; strips an EOF footer until
/// the caller rewrites it).
///
/// Existing chunk bytes are copied verbatim. New dataset names must be unique.
///
/// # Errors
///
/// Returns [`CatalogError`] when the file is invalid, a name collides, or I/O fails.
pub fn append_multi_raw_array_file(
    path: &Path,
    new_specs: &[RawArrayWrite<'_>],
) -> Result<(), CatalogError> {
    if new_specs.is_empty() {
        return Ok(());
    }
    for spec in new_specs {
        dataset::validate_raw_array_write(spec)?;
    }
    let data = std::fs::read(path)?;
    let summary = read_tet_summary_v1(&data)?;
    ensure_new_names(&summary.datasets, new_specs)?;

    let file_execution = new_specs[0]
        .file_execution
        .unwrap_or(summary.file_execution);
    let (entries, payloads) = copy_existing_chunks(&data, &summary.chunks)?;
    let (entries, payloads) =
        append_in_memory_chunks(entries, payloads, summary.datasets.len(), new_specs)?;
    rewrite_file(
        path,
        &summary.datasets,
        new_specs,
        &[],
        entries,
        &payloads,
        file_execution,
    )
}

/// Append in-memory and streaming datasets in one rewrite (raw codec for streaming tiles).
///
/// # Errors
///
/// Returns [`CatalogError`] when the file is invalid, `fill` fails, or I/O fails.
pub fn append_multi_mixed(
    path: &Path,
    new_specs: &[RawArrayWrite<'_>],
    new_streaming: &[ArrayWriteMeta<'_>],
    fill: impl Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), CatalogError> + Sync + Send,
) -> Result<(), CatalogError> {
    if new_specs.is_empty() && new_streaming.is_empty() {
        return Ok(());
    }
    for spec in new_specs {
        dataset::validate_raw_array_write(spec)?;
    }
    for spec in new_streaming {
        super::validate_array_write_meta(spec)?;
    }
    let data = std::fs::read(path)?;
    let summary = read_tet_summary_v1(&data)?;
    ensure_new_names(&summary.datasets, new_specs)?;
    ensure_new_meta_names(&summary.datasets, new_streaming)?;
    ensure_no_cross_name_duplicates(new_specs, new_streaming)?;

    let file_execution = new_specs
        .first()
        .and_then(|s| s.file_execution)
        .unwrap_or(summary.file_execution);

    let (entries, payloads) = copy_existing_chunks(&data, &summary.chunks)?;
    let base = summary.datasets.len();
    let (entries, payloads) = append_in_memory_chunks(entries, payloads, base, new_specs)?;
    let (entries, payloads) = append_streaming_chunks(
        entries,
        payloads,
        base + new_specs.len(),
        new_streaming,
        &fill,
    )?;
    rewrite_file(
        path,
        &summary.datasets,
        new_specs,
        new_streaming,
        entries,
        &payloads,
        file_execution,
    )
}

/// Append streaming datasets (raw codec only) to an existing file.
///
/// # Errors
///
/// Returns [`CatalogError`] when the file is invalid, `fill` fails, or I/O fails.
pub fn append_multi_raw_array_streaming(
    path: &Path,
    new_specs: &[ArrayWriteMeta<'_>],
    fill: impl Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), CatalogError> + Sync + Send,
) -> Result<(), CatalogError> {
    if new_specs.is_empty() {
        return Ok(());
    }
    for spec in new_specs {
        super::validate_array_write_meta(spec)?;
    }
    let data = std::fs::read(path)?;
    let summary = read_tet_summary_v1(&data)?;
    ensure_new_meta_names(&summary.datasets, new_specs)?;

    let (entries, payloads) = copy_existing_chunks(&data, &summary.chunks)?;
    let (entries, payloads) =
        append_streaming_chunks(entries, payloads, summary.datasets.len(), new_specs, &fill)?;
    rewrite_file(
        path,
        &summary.datasets,
        &[],
        new_specs,
        entries,
        &payloads,
        summary.file_execution,
    )
}

fn ensure_new_names(
    existing: &[DatasetRecordV1],
    new_specs: &[RawArrayWrite<'_>],
) -> Result<(), CatalogError> {
    let mut seen: HashSet<&str> = existing.iter().map(|d| d.name.as_str()).collect();
    for spec in new_specs {
        if !seen.insert(spec.name) {
            return Err(CatalogError::InvalidWriteSpec(
                "dataset name already exists in file",
            ));
        }
    }
    Ok(())
}

fn ensure_no_cross_name_duplicates(
    new_specs: &[RawArrayWrite<'_>],
    new_streaming: &[ArrayWriteMeta<'_>],
) -> Result<(), CatalogError> {
    let mut seen: HashSet<&str> = new_specs.iter().map(|s| s.name).collect();
    for spec in new_streaming {
        if !seen.insert(spec.name) {
            return Err(CatalogError::InvalidWriteSpec(
                "duplicate dataset name in append batch",
            ));
        }
    }
    Ok(())
}

fn ensure_new_meta_names(
    existing: &[DatasetRecordV1],
    new_specs: &[ArrayWriteMeta<'_>],
) -> Result<(), CatalogError> {
    let mut seen: HashSet<&str> = existing.iter().map(|d| d.name.as_str()).collect();
    for spec in new_specs {
        if !seen.insert(spec.name) {
            return Err(CatalogError::InvalidWriteSpec(
                "dataset name already exists in file",
            ));
        }
    }
    Ok(())
}

fn copy_stored_payload(data: &[u8], entry: &ChunkIndexEntryV1) -> Result<Vec<u8>, CatalogError> {
    let start = usize_from_u64("payload_offset", entry.payload_offset)?;
    let len = usize_from_u64("stored_byte_len", entry.stored_byte_len)?;
    let end = start
        .checked_add(len)
        .ok_or(CatalogError::InvalidWriteSpec("payload span overflow"))?;
    if end > data.len() {
        return Err(CatalogError::PayloadOutOfBounds {
            file_len: data.len() as u64,
            start: entry.payload_offset,
            end: entry.payload_offset.saturating_add(entry.stored_byte_len),
        });
    }
    Ok(data[start..end].to_vec())
}

fn copy_existing_chunks(
    data: &[u8],
    existing_chunks: &[ChunkIndexEntryV1],
) -> Result<(Vec<ChunkIndexEntryV1>, Vec<Vec<u8>>), CatalogError> {
    let mut entries = Vec::with_capacity(existing_chunks.len());
    let mut payloads = Vec::with_capacity(existing_chunks.len());
    let mut cursor = 0u64;
    for entry in existing_chunks {
        let stored = copy_stored_payload(data, entry)?;
        cursor = super::file_layout::EncodedChunkPush {
            dataset_id: entry.dataset_id,
            chunk_index: entry.chunk_index,
            raw_byte_len: entry.raw_byte_len,
            chunk_codec: entry.codec,
            stored,
        }
        .push(&mut entries, &mut payloads, cursor)?;
    }
    Ok((entries, payloads))
}

fn payload_cursor(entries: &[ChunkIndexEntryV1]) -> u64 {
    entries
        .last()
        .map_or(0, |e| e.payload_offset.saturating_add(e.stored_byte_len))
}

fn append_in_memory_chunks(
    mut entries: Vec<ChunkIndexEntryV1>,
    mut payloads: Vec<Vec<u8>>,
    base_dataset_id: usize,
    new_specs: &[RawArrayWrite<'_>],
) -> Result<(Vec<ChunkIndexEntryV1>, Vec<Vec<u8>>), CatalogError> {
    let mut cursor = payload_cursor(&entries);

    for (offset, spec) in new_specs.iter().enumerate() {
        let dataset_id = super::file_layout::wire_dataset_id(base_dataset_id + offset)?;
        let grid = super::chunk_grid_plan(spec.shape, spec.chunk_shape, spec.dtype)?;
        cursor = super::file_layout::push_raw_tiles_from_tensor(
            spec,
            &grid,
            dataset_id,
            cursor,
            &mut entries,
            &mut payloads,
        )?;
    }
    Ok((entries, payloads))
}

fn append_streaming_chunks<F>(
    mut entries: Vec<ChunkIndexEntryV1>,
    mut payloads: Vec<Vec<u8>>,
    base_dataset_id: usize,
    new_specs: &[ArrayWriteMeta<'_>],
    fill: &F,
) -> Result<(Vec<ChunkIndexEntryV1>, Vec<Vec<u8>>), CatalogError>
where
    F: Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), CatalogError>,
{
    let mut cursor = payload_cursor(&entries);

    for (offset, spec) in new_specs.iter().enumerate() {
        let dataset_id = base_dataset_id + offset;
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
            let mut tile_buf = vec![0u8; usize_from_u64("tile_raw_byte_len", raw_len)?];
            let chunk_index = ChunkIndexEntryV1::padded_chunk_index(&coord[..grid.ndim], grid.ndim);
            let job = StreamTileJob {
                dataset_id,
                dataset_name: spec.name,
                chunk_k: k,
                chunk_coord: chunk_index,
                ndim: grid.ndim,
                raw_byte_len: raw_len,
            };
            fill(&job, &mut tile_buf)?;
            let stored_vec =
                CHUNK_PAYLOAD_CODEC_V1.encode_tile_payload(spec.chunk_codec, tile_buf)?;
            cursor = super::file_layout::EncodedChunkPush {
                dataset_id: super::file_layout::wire_dataset_id(dataset_id)?,
                chunk_index,
                raw_byte_len: raw_len,
                chunk_codec: spec.chunk_codec,
                stored: stored_vec,
            }
            .push(&mut entries, &mut payloads, cursor)?;
        }
    }
    Ok((entries, payloads))
}

/// Merge one or more sealed layout-v1 `.tet` files into `dest` (byte-copy chunks, remapped ids).
///
/// Dataset names must be unique across `sources`. Footers are not preserved (0.2 publish packs a
/// fresh catalog). A single source is copied with [`std::fs::copy`].
///
/// # Errors
///
/// Empty `sources`, duplicate dataset names, invalid inputs, or I/O failures.
pub fn merge_tet_files(dest: &Path, sources: &[&Path]) -> Result<(), CatalogError> {
    if sources.is_empty() {
        return Err(CatalogError::InvalidWriteSpec(
            "merge_tet_files requires at least one source",
        ));
    }
    if sources.len() == 1 {
        std::fs::copy(sources[0], dest)?;
        return Ok(());
    }

    let mut all_datasets: Vec<DatasetRecordV1> = Vec::new();
    let mut entries: Vec<ChunkIndexEntryV1> = Vec::new();
    let mut payloads: Vec<Vec<u8>> = Vec::new();
    let mut seen_names: HashSet<String> = HashSet::new();
    let mut file_execution = FileExecutionSettingsV1::default_engine();
    let mut cursor = 0u64;

    for (source_i, source) in sources.iter().enumerate() {
        let data = std::fs::read(source)?;
        let summary = read_tet_summary_v1(&data)?;
        if source_i == 0 {
            file_execution = summary.file_execution;
        }
        let base_id =
            u64::try_from(all_datasets.len()).map_err(|_| CatalogError::TooLargeForPlatform {
                field: "dataset_count",
                value: u64::MAX,
            })?;
        for ds in &summary.datasets {
            if !seen_names.insert(ds.name.clone()) {
                return Err(CatalogError::InvalidWriteSpec(
                    "duplicate dataset name across merge sources",
                ));
            }
            all_datasets.push(ds.clone());
        }
        for entry in &summary.chunks {
            let stored = copy_stored_payload(&data, entry)?;
            let dataset_id =
                entry
                    .dataset_id
                    .checked_add(base_id)
                    .ok_or(CatalogError::InvalidWriteSpec(
                        "dataset_id overflow while merging",
                    ))?;
            cursor = super::file_layout::EncodedChunkPush {
                dataset_id,
                chunk_index: entry.chunk_index,
                raw_byte_len: entry.raw_byte_len,
                chunk_codec: entry.codec,
                stored,
            }
            .push(&mut entries, &mut payloads, cursor)?;
        }
    }

    rewrite_file(
        dest,
        &all_datasets,
        &[],
        &[],
        entries,
        &payloads,
        file_execution,
    )
}

fn rewrite_file(
    path: &Path,
    existing_datasets: &[DatasetRecordV1],
    new_specs: &[RawArrayWrite<'_>],
    new_streaming: &[ArrayWriteMeta<'_>],
    mut entries: Vec<ChunkIndexEntryV1>,
    payloads: &[Vec<u8>],
    file_execution: FileExecutionSettingsV1,
) -> Result<(), CatalogError> {
    let mut blob = Vec::new();
    for ds in existing_datasets {
        blob.extend_from_slice(&dataset::encode_dataset_blob(
            &ds.name,
            ds.dtype,
            &ds.shape,
            &ds.chunk_shape,
        )?);
    }
    for spec in new_specs {
        blob.extend_from_slice(&dataset::encode_dataset_blob(
            spec.name,
            spec.dtype,
            spec.shape,
            spec.chunk_shape,
        )?);
    }
    for spec in new_streaming {
        blob.extend_from_slice(&dataset::encode_dataset_blob(
            spec.name,
            spec.dtype,
            spec.shape,
            spec.chunk_shape,
        )?);
    }

    let dataset_count = existing_datasets
        .len()
        .checked_add(new_specs.len())
        .and_then(|n| n.checked_add(new_streaming.len()))
        .ok_or(CatalogError::InvalidWriteSpec("dataset count overflow"))?;
    let dataset_blob_len = blob.len() as u64;
    let n_chunks_total =
        u64::try_from(entries.len()).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "chunk_entry_count",
            value: u64::MAX,
        })?;
    let (index_base, chunk_index_length, payload_start) =
        super::file_layout::chunk_index_layout(dataset_blob_len, n_chunks_total)?;
    let mut cursor = payload_start;
    for entry in &mut entries {
        entry.payload_offset = cursor;
        cursor = cursor
            .checked_add(entry.stored_byte_len)
            .ok_or(CatalogError::InvalidWriteSpec("payload cursor overflow"))?;
    }

    let sb = super::file_layout::layout_superblock(dataset_count, index_base, chunk_index_length)?;
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
    for p in payloads {
        f.write_all(p)?;
    }
    f.sync_all()?;
    Ok(())
}

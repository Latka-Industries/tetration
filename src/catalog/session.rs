//! Embedder-oriented create / open helpers (Phase 7).
//!
//! [`TetWriterSession`] buffers datasets and optional history rows, then writes one `.tet` on
//! [`TetWriterSession::commit`] (in-memory tensors) or [`TetWriterSession::commit_with_fill`]
//! (streaming tiles). [`TetFile`] keeps the backing file open for mmap query execution.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use super::dataset::RawArrayWrite;
use super::execution::FileExecutionSettingsV1;
use super::history::{self, FooterBlobV1, HistoryEvent, write_footer_blob};
use super::metadata::{CoordAxisV1, DatasetMetadataV1, FileMetadataV1, TetMetadataV1};
use super::stream_write::{ArrayWriteMeta, StreamTileJob, write_multi_raw_array_streaming};
use super::tile;
use super::{
    CHUNK_PAYLOAD_CODEC_V1, CatalogError, DATASET_DTYPE_TAG_V1, DatasetRecordV1, TetFileSummaryV1,
    append_multi_mixed, append_multi_raw_array_file, read_tet_summary_v1,
    validate_array_write_meta, write_multi_raw_array_file,
};
use crate::layout::{self, SuperblockV1, mmap_file_read};

fn empty_dataset_import_fields() -> (
    BTreeMap<String, String>,
    Option<Vec<String>>,
    Option<BTreeMap<String, CoordAxisV1>>,
) {
    (BTreeMap::new(), None, None)
}

/// In-memory dataset queued for [`TetWriterSession::commit`].
#[derive(Debug, Clone)]
pub struct TetDatasetWrite {
    pub name: String,
    pub dtype: u32,
    pub shape: Vec<u64>,
    pub chunk_shape: Vec<u64>,
    pub chunk_codec: u32,
    pub data: Vec<u8>,
    /// CF-style string attributes persisted in the footer `metadata` object.
    pub attrs: BTreeMap<String, String>,
    /// Optional dimension names (`ndim` strings).
    pub dim_names: Option<Vec<String>>,
    /// Optional per-axis index labels (axis name → label list).
    pub coords: Option<BTreeMap<String, CoordAxisV1>>,
}

impl TetDatasetWrite {
    /// Row-major `f32` tensor with raw chunk codec (**0**).
    pub fn f32_row_major(
        name: impl Into<String>,
        shape: &[u64],
        chunk_shape: &[u64],
        data: Vec<u8>,
    ) -> Result<Self, CatalogError> {
        Self::row_major(name, DATASET_DTYPE_TAG_V1.f32, shape, chunk_shape, data)
    }

    /// Row-major `f64` tensor with raw chunk codec (**0**).
    pub fn f64_row_major(
        name: impl Into<String>,
        shape: &[u64],
        chunk_shape: &[u64],
        data: Vec<u8>,
    ) -> Result<Self, CatalogError> {
        Self::row_major(name, DATASET_DTYPE_TAG_V1.f64, shape, chunk_shape, data)
    }

    /// Row-major tensor with a supported wire dtype tag and raw chunk codec (**0**).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when `dtype`, `shape`, `chunk_shape`, or `data` are inconsistent.
    pub fn row_major(
        name: impl Into<String>,
        dtype: u32,
        shape: &[u64],
        chunk_shape: &[u64],
        data: Vec<u8>,
    ) -> Result<Self, CatalogError> {
        let name = name.into();
        let spec = RawArrayWrite::from_tensor(
            &name,
            dtype,
            shape,
            chunk_shape,
            CHUNK_PAYLOAD_CODEC_V1.raw,
            &data,
            None,
        );
        super::dataset::validate_raw_array_write(&spec)?;
        let (attrs, dim_names, coords) = empty_dataset_import_fields();
        Ok(Self {
            name,
            dtype,
            shape: shape.to_vec(),
            chunk_shape: chunk_shape.to_vec(),
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data,
            attrs,
            dim_names,
            coords,
        })
    }
}

/// Streaming dataset spec: geometry + footer metadata only (tile bytes supplied at commit).
#[derive(Debug, Clone)]
pub struct TetDatasetStreamSpec {
    pub name: String,
    pub dtype: u32,
    pub shape: Vec<u64>,
    pub chunk_shape: Vec<u64>,
    pub attrs: BTreeMap<String, String>,
    pub dim_names: Option<Vec<String>>,
    pub coords: Option<BTreeMap<String, CoordAxisV1>>,
}

impl TetDatasetStreamSpec {
    /// Row-major `f32` grid with raw chunk codec (**0**).
    pub fn f32_row_major(
        name: impl Into<String>,
        shape: &[u64],
        chunk_shape: &[u64],
    ) -> Result<Self, CatalogError> {
        Self::row_major(name, DATASET_DTYPE_TAG_V1.f32, shape, chunk_shape)
    }

    /// Row-major `f64` grid with raw chunk codec (**0**).
    pub fn f64_row_major(
        name: impl Into<String>,
        shape: &[u64],
        chunk_shape: &[u64],
    ) -> Result<Self, CatalogError> {
        Self::row_major(name, DATASET_DTYPE_TAG_V1.f64, shape, chunk_shape)
    }

    /// Row-major grid with a supported wire dtype tag and raw chunk codec (**0**).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when `dtype`, `shape`, or `chunk_shape` are inconsistent.
    pub fn row_major(
        name: impl Into<String>,
        dtype: u32,
        shape: &[u64],
        chunk_shape: &[u64],
    ) -> Result<Self, CatalogError> {
        let name = name.into();
        let meta = ArrayWriteMeta::row_major(&name, dtype, shape, chunk_shape, None);
        validate_array_write_meta(&meta)?;
        let (attrs, dim_names, coords) = empty_dataset_import_fields();
        Ok(Self {
            name,
            dtype,
            shape: shape.to_vec(),
            chunk_shape: chunk_shape.to_vec(),
            attrs,
            dim_names,
            coords,
        })
    }
}

/// Draft file-level metadata; mapped into [`FileMetadataV1`] on commit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileMetadataDraft {
    pub tool: Option<String>,
    pub library_version: Option<String>,
}

#[derive(Debug, Clone)]
enum SessionDataset {
    InMemory(TetDatasetWrite),
    Streaming(TetDatasetStreamSpec),
}

impl SessionDataset {
    fn name(&self) -> &str {
        match self {
            Self::InMemory(d) => &d.name,
            Self::Streaming(d) => &d.name,
        }
    }

    fn attrs(&self) -> &BTreeMap<String, String> {
        match self {
            Self::InMemory(d) => &d.attrs,
            Self::Streaming(d) => &d.attrs,
        }
    }

    fn dim_names(&self) -> Option<&Vec<String>> {
        match self {
            Self::InMemory(d) => d.dim_names.as_ref(),
            Self::Streaming(d) => d.dim_names.as_ref(),
        }
    }

    fn coords(&self) -> Option<&BTreeMap<String, CoordAxisV1>> {
        match self {
            Self::InMemory(d) => d.coords.as_ref(),
            Self::Streaming(d) => d.coords.as_ref(),
        }
    }

    fn is_streaming(&self) -> bool {
        matches!(self, Self::Streaming(_))
    }
}

/// Normalized dataset row for [`TetWriterSession::commit_with_fill`].
struct CommitPrepared {
    name: String,
    dtype: u32,
    shape: Vec<u64>,
    chunk_shape: Vec<u64>,
    data: Option<Vec<u8>>,
}

/// Split queued session datasets into a uniform row for mixed commit.
fn take_commit_prepared(datasets: Vec<SessionDataset>) -> Vec<CommitPrepared> {
    datasets
        .into_iter()
        .map(|ds| match ds {
            SessionDataset::InMemory(w) => CommitPrepared {
                name: w.name,
                dtype: w.dtype,
                shape: w.shape,
                chunk_shape: w.chunk_shape,
                data: Some(w.data),
            },
            SessionDataset::Streaming(s) => CommitPrepared {
                name: s.name,
                dtype: s.dtype,
                shape: s.shape,
                chunk_shape: s.chunk_shape,
                data: None,
            },
        })
        .collect()
}

/// In-memory rows from [`CommitPrepared`] as [`RawArrayWrite`] refs (raw codec).
fn memory_raw_writes(
    prepared: &[CommitPrepared],
    file_execution: Option<FileExecutionSettingsV1>,
) -> Vec<RawArrayWrite<'_>> {
    prepared
        .iter()
        .filter_map(|d| {
            Some(RawArrayWrite::from_tensor(
                &d.name,
                d.dtype,
                &d.shape,
                &d.chunk_shape,
                CHUNK_PAYLOAD_CODEC_V1.raw,
                d.data.as_ref()?,
                file_execution,
            ))
        })
        .collect()
}

/// [`ArrayWriteMeta`] rows; `streaming_only` skips in-memory datasets.
fn commit_array_metas(
    prepared: &[CommitPrepared],
    file_execution: Option<FileExecutionSettingsV1>,
    streaming_only: bool,
) -> Vec<ArrayWriteMeta<'_>> {
    prepared
        .iter()
        .filter(|d| !streaming_only || d.data.is_none())
        .map(|d| {
            ArrayWriteMeta::row_major(&d.name, d.dtype, &d.shape, &d.chunk_shape, file_execution)
        })
        .collect()
}

/// Fill one tile: slice in-memory tensor bytes or delegate to streaming `fill`.
fn fill_commit_prepared_tile<F>(
    prepared: &[CommitPrepared],
    job: &StreamTileJob<'_>,
    buf: &mut [u8],
    fill: &F,
) -> Result<(), CatalogError>
where
    F: Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), CatalogError>,
{
    let ds = prepared.iter().find(|p| p.name == job.dataset_name).ok_or(
        CatalogError::InvalidWriteSpec("unknown dataset in tile job"),
    )?;
    if let Some(data) = &ds.data {
        let elem_size = super::elem_size_for_wire_tag(ds.dtype)?;
        tile::write_tile_row_major_into(
            data,
            &ds.shape,
            &ds.chunk_shape,
            &job.chunk_coord[..job.ndim],
            job.ndim,
            elem_size,
            buf,
        )
    } else {
        fill(job, buf)
    }
}

#[derive(Debug, Clone)]
enum SessionMode {
    Create,
    Append { existing_names: HashSet<String> },
}

/// Buffered writer: queue datasets and history, flush on commit.
#[derive(Debug, Clone)]
pub struct TetWriterSession {
    path: PathBuf,
    mode: SessionMode,
    base_metadata: TetMetadataV1,
    datasets: Vec<SessionDataset>,
    history: Vec<HistoryEvent>,
    file_execution: Option<FileExecutionSettingsV1>,
    pub metadata: FileMetadataDraft,
}

impl TetWriterSession {
    /// New session targeting `path` (file is created on commit, truncating any prior file).
    #[must_use]
    pub fn create(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            mode: SessionMode::Create,
            base_metadata: TetMetadataV1::default(),
            datasets: Vec::new(),
            history: Vec::new(),
            file_execution: None,
            metadata: FileMetadataDraft::default(),
        }
    }

    /// Open an existing `.tet` and queue additional datasets (merged footer metadata/history on commit).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when the file cannot be read or parsed.
    pub fn open_append(path: impl Into<PathBuf>) -> Result<Self, CatalogError> {
        let path = path.into();
        let data = std::fs::read(&path)?;
        let summary = read_tet_summary_v1(&data)?;
        let existing_names: HashSet<String> =
            summary.datasets.iter().map(|d| d.name.clone()).collect();
        let tool = summary.metadata.file.as_ref().and_then(|f| f.tool.clone());
        let library_version = summary
            .metadata
            .file
            .as_ref()
            .and_then(|f| f.library_version.clone());
        Ok(Self {
            path,
            mode: SessionMode::Append { existing_names },
            base_metadata: summary.metadata,
            datasets: Vec::new(),
            history: summary.history,
            file_execution: Some(summary.file_execution),
            metadata: FileMetadataDraft {
                tool,
                library_version,
            },
        })
    }

    /// Whether this session appends to an existing file ([`Self::open_append`]).
    #[must_use]
    pub fn is_append(&self) -> bool {
        matches!(self.mode, SessionMode::Append { .. })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn dataset_count(&self) -> usize {
        self.datasets.len()
    }

    /// Chunk-index execution settings applied when the file has at least one dataset.
    #[must_use]
    pub fn file_execution(mut self, settings: FileExecutionSettingsV1) -> Self {
        self.file_execution = Some(settings);
        self
    }

    /// Queue an in-memory dataset (validated immediately).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when the write spec is invalid.
    pub fn push_dataset(&mut self, dataset: TetDatasetWrite) -> Result<(), CatalogError> {
        let name = dataset.name.as_str();
        let spec = RawArrayWrite {
            name,
            dtype: dataset.dtype,
            shape: &dataset.shape,
            chunk_shape: &dataset.chunk_shape,
            chunk_codec: dataset.chunk_codec,
            data: &dataset.data,
            file_execution: None,
        };
        super::dataset::validate_raw_array_write(&spec)?;
        self.ensure_unique_dataset_name(&dataset.name)?;
        self.datasets.push(SessionDataset::InMemory(dataset));
        Ok(())
    }

    fn ensure_unique_dataset_name(&self, name: &str) -> Result<(), CatalogError> {
        if let SessionMode::Append { existing_names } = &self.mode
            && existing_names.contains(name)
        {
            return Err(CatalogError::InvalidWriteSpec(
                "dataset name already exists in file",
            ));
        }
        if self.datasets.iter().any(|d| d.name() == name) {
            return Err(CatalogError::InvalidWriteSpec(
                "dataset name already queued in session",
            ));
        }
        Ok(())
    }

    /// Queue a streaming dataset (geometry validated; tile bytes supplied in [`Self::commit_with_fill`]).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when the write spec is invalid.
    pub fn push_dataset_streaming(
        &mut self,
        spec: TetDatasetStreamSpec,
    ) -> Result<(), CatalogError> {
        let meta =
            ArrayWriteMeta::row_major(&spec.name, spec.dtype, &spec.shape, &spec.chunk_shape, None);
        validate_array_write_meta(&meta)?;
        self.ensure_unique_dataset_name(&spec.name)?;
        self.datasets.push(SessionDataset::Streaming(spec));
        Ok(())
    }

    /// Append a history row flushed on commit (`op`, `source`, Unix seconds as decimal string).
    pub fn push_history_event(&mut self, op: impl Into<String>, source: impl Into<String>) {
        self.history.push(HistoryEvent::new(op, source));
    }

    fn has_streaming(&self) -> bool {
        self.datasets.iter().any(SessionDataset::is_streaming)
    }

    fn build_footer_metadata(&self) -> Result<Option<TetMetadataV1>, CatalogError> {
        let mut meta = self.base_metadata.clone();
        let has_file = self.metadata.tool.is_some() || self.metadata.library_version.is_some();
        if has_file {
            let prior = meta.file.take();
            meta.file = Some(FileMetadataV1 {
                tool: self
                    .metadata
                    .tool
                    .clone()
                    .or_else(|| prior.as_ref().and_then(|f| f.tool.clone())),
                library_version: self
                    .metadata
                    .library_version
                    .clone()
                    .or_else(|| prior.as_ref().and_then(|f| f.library_version.clone()))
                    .or_else(|| Some(env!("CARGO_PKG_VERSION").to_owned())),
                created_at: prior
                    .as_ref()
                    .and_then(|f| f.created_at.clone())
                    .or_else(|| Some(history::unix_timestamp_now())),
            });
        }
        for ds in &self.datasets {
            if DatasetMetadataV1::import_is_empty(ds.attrs(), ds.dim_names(), ds.coords()) {
                continue;
            }
            meta.dataset_mut(ds.name())
                .apply_import(ds.attrs(), ds.dim_names(), ds.coords());
        }
        if meta.file.is_none() && meta.datasets.is_empty() {
            return Ok(None);
        }
        meta.validate()?;
        Ok(Some(meta))
    }

    fn ensure_default_history(&mut self) {
        if self.history.is_empty() {
            let source = self.path.display().to_string();
            self.push_history_event("write", source);
        }
    }

    fn flush_footer(&self, metadata: Option<TetMetadataV1>) -> Result<(), CatalogError> {
        if !self.history.is_empty() || metadata.is_some() {
            write_footer_blob(
                &self.path,
                &FooterBlobV1 {
                    history: self.history.clone(),
                    metadata,
                    metadata_ref: None,
                },
            )?;
        }
        Ok(())
    }

    /// Write the `.tet` when every queued dataset is in-memory.
    ///
    /// Streaming datasets require [`Self::commit_with_fill`].
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when no datasets were queued, a streaming dataset is present,
    /// layout validation fails, or I/O fails.
    pub fn commit(mut self) -> Result<PathBuf, CatalogError> {
        self.ensure_default_history();
        if self.datasets.is_empty() {
            return Err(CatalogError::InvalidWriteSpec(
                "TetWriterSession: at least one dataset is required",
            ));
        }
        if self.has_streaming() {
            return Err(CatalogError::InvalidWriteSpec(
                "TetWriterSession: streaming datasets require commit_with_fill",
            ));
        }
        let specs: Vec<RawArrayWrite<'_>> = self
            .datasets
            .iter()
            .map(|d| {
                let d = match d {
                    SessionDataset::InMemory(w) => w,
                    SessionDataset::Streaming(_) => unreachable!(),
                };
                RawArrayWrite::from_tensor(
                    &d.name,
                    d.dtype,
                    &d.shape,
                    &d.chunk_shape,
                    d.chunk_codec,
                    &d.data,
                    self.file_execution,
                )
            })
            .collect();
        let footer_metadata = self.build_footer_metadata()?;
        match self.mode {
            SessionMode::Create => write_multi_raw_array_file(&self.path, &specs)?,
            SessionMode::Append { .. } => {
                append_multi_raw_array_file(&self.path, &specs)?;
            }
        }
        self.flush_footer(footer_metadata)?;
        Ok(self.path)
    }

    /// Write the `.tet`, filling streaming tiles via `fill` (raw codec only).
    ///
    /// In-memory datasets are sliced automatically; `fill` is invoked only for
    /// [`TetDatasetStreamSpec`] entries. When every dataset is in-memory, behaves like
    /// [`Self::commit`] and ignores `fill`.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when no datasets were queued, layout validation fails, `fill`
    /// fails, or I/O fails.
    pub fn commit_with_fill<F>(
        mut self,
        parallel_jobs: usize,
        fill: F,
    ) -> Result<PathBuf, CatalogError>
    where
        F: Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), CatalogError> + Sync + Send,
    {
        if self.datasets.is_empty() {
            return Err(CatalogError::InvalidWriteSpec(
                "TetWriterSession: at least one dataset is required",
            ));
        }
        if !self.has_streaming() {
            return self.commit();
        }

        self.ensure_default_history();
        let footer_metadata = self.build_footer_metadata()?;
        let file_execution = self.file_execution;
        let prepared = take_commit_prepared(std::mem::take(&mut self.datasets));
        let memory_specs = memory_raw_writes(&prepared, file_execution);
        let stream_metas = commit_array_metas(&prepared, file_execution, true);
        let fill_tile = |job: &StreamTileJob<'_>, buf: &mut [u8]| {
            fill_commit_prepared_tile(&prepared, job, buf, &fill)
        };

        match self.mode {
            SessionMode::Create => write_multi_raw_array_streaming(
                &self.path,
                &commit_array_metas(&prepared, file_execution, false),
                parallel_jobs,
                fill_tile,
                None,
            )?,
            SessionMode::Append { .. } => {
                append_multi_mixed(&self.path, &memory_specs, &stream_metas, fill_tile)?;
            }
        }
        self.flush_footer(footer_metadata)?;
        Ok(self.path)
    }
}

/// Open `.tet` for mmap reads and query execution.
pub struct TetFile {
    path: PathBuf,
    _file: File,
    mmap: Mmap,
}

impl TetFile {
    /// Memory-map an existing file read-only.
    ///
    /// # Errors
    ///
    /// Propagates I/O errors from open or mmap.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, std::io::Error> {
        let path = path.into();
        let file = File::open(&path)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        Ok(Self {
            path,
            _file: file,
            mmap,
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn mmap(&self) -> &[u8] {
        &self.mmap
    }

    /// Parsed superblock + catalog (+ footer history/metadata when present).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when layout rules are violated.
    pub fn summary(&self) -> Result<TetFileSummaryV1, CatalogError> {
        read_tet_summary_v1(self.mmap())
    }

    /// # Errors
    ///
    /// Returns [`layout::LayoutError`] when bytes at offset 0 are not a valid v1 superblock.
    pub fn superblock(&self) -> Result<SuperblockV1, layout::LayoutError> {
        layout::read_superblock_v1(self.mmap())
    }

    /// Dataset catalog records.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when the catalog cannot be read.
    pub fn datasets(&self) -> Result<Vec<DatasetRecordV1>, CatalogError> {
        Ok(self.summary()?.datasets)
    }

    /// Metadata for a dataset by catalog name.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when the catalog cannot be read.
    pub fn dataset_metadata(&self, name: &str) -> Result<Option<DatasetMetadataV1>, CatalogError> {
        Ok(self.summary()?.metadata.datasets.get(name).cloned())
    }
}

impl TetFile {
    /// Convenience: mmap from `path` without keeping [`TetFile`] alive.
    ///
    /// Prefer [`Self::open`] when running multiple queries on the same file.
    ///
    /// # Errors
    ///
    /// Propagates I/O errors from [`mmap_file_read`].
    pub fn mmap_bytes(path: impl AsRef<Path>) -> Result<Vec<u8>, std::io::Error> {
        let mmap = mmap_file_read(path.as_ref())?;
        Ok(mmap.to_vec())
    }
}

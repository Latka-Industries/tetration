//! Query planning and execution entrypoints against mmap’d `.tet` files.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use crate::catalog::{
    CatalogError, DATASET_DTYPE_TAG_V1, DatasetRecordV1, GenerationManifest,
    GenerationSecondaryIndex, GenerationZoneMaps, SECONDARY_INDEX_FILE, SECONDARY_INDEX_PRESENT,
    TetFileSummaryV1, ZONE_MAPS_FILE, ZONE_MAPS_VERSION, chunk_coords_intersecting_strided,
    read_embedded_secondary_index_bytes, read_embedded_zone_maps_bytes, read_tet_summary_v1,
    read_zone_maps_file, tensor_bytes_from_shape,
};
use crate::layout::{SUPERBLOCK_FLAG_HISTORY_FOOTER, SUPERBLOCK_V1_LEN, SuperblockV1};
use crate::query::resolve_axes::resolve_query_document_axes;
use crate::query::resolve_selection::resolve_query_document_selection;
use crate::query::types::{
    ChunkTouchPolicy, DatasetResolution, OutputHint, QueryDocument, QueryExecutionPreview,
    QueryResponse, ReadPlan, TetError,
};

fn query_requests_spill(doc: &QueryDocument) -> bool {
    matches!(
        doc.output.as_ref().and_then(|o| o.preferred.as_ref()),
        Some(OutputHint::SpillArray { .. })
    )
}

use crate::query::engine::budget::ExecutionBudget;
use crate::query::engine::operations::build_execution_preview;
use crate::query::engine::spill_policy::SpillPathAllowlist;
use crate::query::plan::read_plan::{ReadPlanGeometry, ZonePushdown, build_read_plan};
use crate::query::plan::selection::resolved_dense_global_box;

fn dataset_tensor_bytes(dtype: u32, shape: &[u64], expected: u32) -> Option<u64> {
    (dtype == expected)
        .then(|| tensor_bytes_from_shape(shape, expected))
        .flatten()
}

fn group_count_for(doc: &QueryDocument, plan: &ReadPlan) -> Option<usize> {
    let group_by = doc.group_by.as_ref()?;
    if let Some(remap) = &plan.label_group_remap {
        let mut count = 1usize;
        for &extent in &remap.extents {
            let extent = usize::try_from(extent).ok()?;
            count = count.saturating_mul(extent);
        }
        return Some(count);
    }
    let mut count = 1usize;
    for key in group_by.axes() {
        let idx = key.parse::<usize>().ok()?;
        let extent = *plan.logical_selection_shape.get(idx)?;
        let extent = usize::try_from(extent).ok()?;
        count = count.saturating_mul(extent);
    }
    Some(count)
}

fn map_geometry_catalog_error(e: CatalogError) -> TetError {
    match e {
        CatalogError::InvalidWriteSpec(msg) => TetError::Validation(format!(
            "selection does not form a valid global box for this dataset: {msg}"
        )),
        other => TetError::Catalog(other),
    }
}

fn query_response(
    doc: &QueryDocument,
    message: impl Into<String>,
    tet_path: Option<&str>,
    catalog: Option<DatasetResolution>,
    read_plan: Option<ReadPlan>,
    execution: Option<QueryExecutionPreview>,
) -> QueryResponse {
    QueryResponse {
        status: "planned",
        accepted: true,
        layout_version: doc.layout_version,
        dataset: doc.dataset.clone(),
        selection_axes: doc.selection.as_ref().map(Vec::len),
        operation: doc.operation.clone(),
        message: message.into(),
        tet_file: tet_path.map(str::to_string),
        catalog,
        read_plan,
        execution,
    }
}

fn base_response(doc: &QueryDocument, message: impl Into<String>) -> QueryResponse {
    query_response(doc, message, None, None, None, None)
}

/// Build a response echoing the plan without consulting a `.tet` file (no mmap).
#[must_use]
pub fn plan_query_empty(doc: &QueryDocument) -> QueryResponse {
    base_response(
        doc,
        "query accepted and validated; pass `--tet PATH` for catalog + read_plan, or add `--execute` with `--tet` for a capped raw f32 mmap preview",
    )
}

/// Like [`plan_query_empty`], but read catalog metadata from `mmap` (full `.tet` bytes).
///
/// `tet_path` is echoed in the JSON as `tet_file` when `Some`.
///
/// When the dataset matches, [`QueryResponse::read_plan`] lists chunk payloads that intersect the
/// per-axis selection (default full tensor). Non-unit JSON `step` uses strided chunk selection
/// (see [`crate::query::ChunkTouchPolicy`]).
///
/// `raw_f32_preview_max`: when `Some(n)` with `n > 0`, read planned chunk payloads from `mmap`
/// (raw or zstd `f32` tiles, codecs **0** / **1**) and attach up to `n` decoded little-endian `f32`
/// values under [`QueryResponse::execution`] (requires dataset
/// [`DATASET_DTYPE_TAG_V1`](crate::catalog::DATASET_DTYPE_TAG_V1) `.f32`). When `Some(0)`, only attach execution when
/// [`QueryDocument::operation`] is set: the full logical tensor is still decoded for aggregation,
/// but `f32_preview` is empty. When [`QueryDocument::operation`] is set, a limit must be passed
/// (use `0` to skip preview floats). Partial reductions populate `operation_reduced_*` fields;
/// scalar reductions (`sum`, `mean`, `min`, `max`, `count`, `var`, `std` with `axes: []`) use single-pass fold fields such as `operation_sum` / `operation_var`.
///
/// # Errors
///
/// Returns [`TetError::Catalog`] when the file is not a valid v1 catalog view, or
/// [`TetError::Validation`] when `selection` is inconsistent with dataset rank or shape, when an
/// `operation` is set but `raw_f32_preview_max` is `None`, when `Some(0)` is used without an
/// `operation`, when preview or aggregation requires a non-`f32` dataset, or when payload bytes
/// cannot be read or decoded from `mmap`.
pub fn plan_query_with_tet_mmap(
    doc: &QueryDocument,
    tet_path: Option<&str>,
    mmap: &[u8],
    raw_f32_preview_max: Option<usize>,
) -> Result<QueryResponse, TetError> {
    plan_query_with_tet_mmap_ex(doc, tet_path, mmap, raw_f32_preview_max, None)
}

/// Like [`plan_query_with_tet_mmap`], with an optional spill path allowlist.
///
/// When `spill_allowlist` is `None` and `tet_path` is set, [`SpillPathAllowlist::default_for_tet`]
/// applies (`.tet` parent + platform cache dirs). Pass an explicit allowlist to override defaults
/// entirely, or build [`SpillPathAllowlist::default_for_tet`] with extra `--spill-allow` roots.
///
/// # Errors
///
/// Same as [`plan_query_with_tet_mmap`], plus [`TetError::Validation`] when the default spill
/// allowlist cannot be built from `tet_path`, when export spill or materialize-required operations
/// need an allowlist but none is available, or when a spill path fails allowlist validation.
pub fn plan_query_with_tet_mmap_ex(
    doc: &QueryDocument,
    tet_path: Option<&str>,
    mmap: &[u8],
    raw_f32_preview_max: Option<usize>,
    spill_allowlist: Option<&SpillPathAllowlist>,
) -> Result<QueryResponse, TetError> {
    // Discovery: embedded footer spills first, then generation sidecars (0.7.0).
    let zones = match zone_maps_embedded(mmap)? {
        Some(z) => Some(z),
        None => zone_maps_beside(tet_path)?,
    };
    let index = secondary_index_embedded(mmap).or_else(|| secondary_index_beside(tet_path));
    plan_query_with_zones(
        doc,
        tet_path,
        mmap,
        raw_f32_preview_max,
        spill_allowlist,
        zones.as_ref(),
        index.as_ref(),
    )
}

pub(crate) fn plan_query_with_zones(
    doc: &QueryDocument,
    tet_path: Option<&str>,
    mmap: &[u8],
    raw_f32_preview_max: Option<usize>,
    spill_allowlist: Option<&SpillPathAllowlist>,
    zones: Option<&GenerationZoneMaps>,
    index: Option<&GenerationSecondaryIndex>,
) -> Result<QueryResponse, TetError> {
    let summary = read_tet_summary_v1(mmap)?;
    match summary.datasets.iter().position(|d| d.name == doc.dataset) {
        Some(dataset_idx) => plan_query_matched_dataset(
            &summary,
            dataset_idx,
            doc,
            mmap,
            &PlanAttach {
                tet_path,
                raw_f32_preview_max,
                spill_allowlist,
                zones,
                index,
            },
        ),
        None => Ok(plan_query_unmatched_dataset(doc, tet_path, &summary)),
    }
}

/// Directory that holds `ZONE.json` / `INDEX.json` / `MANIFEST.json` for a query path.
///
/// Accepts either a sealed `packed.tet` (sidecars beside it) or a `.tet.d` package root
/// (sidecars under the pinned generation). Standalone `.tet` files fall back to embeds.
fn sidecar_dir(tet_path: &Path) -> Option<PathBuf> {
    if crate::verify::looks_like_tet_package(tet_path) {
        let pkg = crate::catalog::TetPackage::open(tet_path).ok()?;
        let pin = pkg.pin_current().ok()?;
        return pin.packed_tet_path().parent().map(Path::to_path_buf);
    }
    tet_path.parent().map(Path::to_path_buf)
}

fn zone_maps_embedded(mmap: &[u8]) -> Result<Option<GenerationZoneMaps>, TetError> {
    let Some(sb) = superblock_if_present(mmap)? else {
        return Ok(None);
    };
    let Some(bytes) = read_embedded_zone_maps_bytes(mmap, sb.flags).map_err(TetError::Catalog)?
    else {
        return Ok(None);
    };
    let Ok(doc) = serde_json::from_slice::<GenerationZoneMaps>(&bytes) else {
        return Ok(None);
    };
    if doc.version != ZONE_MAPS_VERSION {
        return Ok(None);
    }
    Ok(Some(doc))
}

fn secondary_index_embedded(mmap: &[u8]) -> Option<GenerationSecondaryIndex> {
    let sb = superblock_if_present(mmap).ok()??;
    let bytes = read_embedded_secondary_index_bytes(mmap, sb.flags).ok()??;
    crate::query::plan::index_pushdown::usable_embedded_secondary_index(&bytes)
}

fn superblock_if_present(mmap: &[u8]) -> Result<Option<SuperblockV1>, TetError> {
    if mmap.len() < SUPERBLOCK_V1_LEN {
        return Ok(None);
    }
    let sb = SuperblockV1::from_bytes(
        mmap[..SUPERBLOCK_V1_LEN]
            .try_into()
            .map_err(|_| TetError::Catalog(CatalogError::InvalidWriteSpec("superblock length")))?,
    )
    .map_err(|e| TetError::Catalog(e.into()))?;
    if sb.flags & SUPERBLOCK_FLAG_HISTORY_FOOTER == 0 {
        return Ok(None);
    }
    Ok(Some(sb))
}

fn zone_maps_beside(tet_path: Option<&str>) -> Result<Option<GenerationZoneMaps>, TetError> {
    let Some(tet_path) = tet_path else {
        return Ok(None);
    };
    let Some(dir) = sidecar_dir(Path::new(tet_path)) else {
        return Ok(None);
    };
    let zone = dir.join(ZONE_MAPS_FILE);
    if !zone.is_file() {
        return Ok(None);
    }
    read_zone_maps_file(&zone)
        .map(Some)
        .map_err(TetError::Catalog)
}

/// Usable `INDEX.json` beside `packed.tet`, or `None` when the claim is missing or bad.
///
/// A bad blob is ignored. The query keeps the zone-map plan. Also resolves `.tet.d`
/// package roots to the pinned generation (same sidecars as `packed.tet`).
fn secondary_index_beside(tet_path: Option<&str>) -> Option<GenerationSecondaryIndex> {
    let dir = sidecar_dir(Path::new(tet_path?))?;
    let manifest_path = dir.join("MANIFEST.json");
    if !manifest_path.is_file() {
        return None;
    }
    let manifest: GenerationManifest =
        serde_json::from_slice(&fs::read(&manifest_path).ok()?).ok()?;
    if manifest.secondary_indexes.as_deref() != Some(SECONDARY_INDEX_PRESENT) {
        return None;
    }
    let sha = manifest.secondary_indexes_sha256.as_deref()?;
    let bytes = fs::read(dir.join(SECONDARY_INDEX_FILE)).ok()?;
    crate::query::plan::index_pushdown::usable_secondary_index(
        &bytes,
        sha,
        manifest.packed_sha256.as_deref(),
    )
}

fn chunk_index_rows_for_dataset(summary: &TetFileSummaryV1, dataset_idx: usize) -> usize {
    summary
        .chunks
        .iter()
        .filter(|c| c.dataset_id == dataset_idx as u64)
        .count()
}

/// Planned logical read for a matched dataset (no execution / preview).
#[derive(Debug, Clone)]
pub struct PlannedRead {
    pub read_plan: ReadPlan,
    pub dtype: u32,
}

/// Resolve axes, selection, and chunk touch list for `doc` against `mmap`.
///
/// # Errors
///
/// Catalog read failures, unknown dataset, or invalid selection geometry.
pub fn plan_read_for_document(doc: &QueryDocument, mmap: &[u8]) -> Result<PlannedRead, TetError> {
    let summary = read_tet_summary_v1(mmap)?;
    let dataset_idx = summary
        .datasets
        .iter()
        .position(|d| d.name == doc.dataset)
        .ok_or_else(|| {
            TetError::Validation(format!("dataset {:?} not found in catalog", doc.dataset))
        })?;
    let rec = &summary.datasets[dataset_idx];
    let mut doc = doc.clone();
    let dataset_meta = summary.metadata.datasets.get(&doc.dataset);
    resolve_query_document_axes(&mut doc, dataset_meta, rec.shape.len())?;
    resolve_query_document_selection(&mut doc, dataset_meta, &rec.shape)?;
    let read_plan = build_matched_read_plan(&summary, dataset_idx, &doc, None, None)?;
    Ok(PlannedRead {
        read_plan,
        dtype: rec.dtype,
    })
}

fn build_matched_read_plan(
    summary: &TetFileSummaryV1,
    dataset_idx: usize,
    doc: &QueryDocument,
    zones: Option<&GenerationZoneMaps>,
    index: Option<&GenerationSecondaryIndex>,
) -> Result<ReadPlan, TetError> {
    let rec = &summary.datasets[dataset_idx];
    let (g0, g1, steps) = resolved_dense_global_box(doc, &rec.shape)?;
    let strided = steps.iter().any(|&t| t != 1);
    let touch_policy = if strided {
        ChunkTouchPolicy::StridedHalfOpen
    } else {
        ChunkTouchPolicy::DenseHalfOpenUnitStep
    };
    let coords = chunk_coords_intersecting_strided(&rec.shape, &rec.chunk_shape, &g0, &g1, &steps)
        .map_err(map_geometry_catalog_error)?;
    build_read_plan(
        summary,
        dataset_idx,
        rec.shape.len(),
        &coords,
        touch_policy,
        &ReadPlanGeometry::new(&rec.shape, &rec.chunk_shape, &g0, &g1, &steps),
        &ZonePushdown {
            zones,
            dtype: rec.dtype,
            filter: doc.filter.as_ref(),
            label_filter: doc.label_filter.as_ref(),
            fill: summary
                .metadata
                .datasets
                .get(&doc.dataset)
                .and_then(|meta| crate::query::resolve_axes::fill_from_attrs(&meta.attrs)),
            index,
            dataset_meta: summary.metadata.datasets.get(&doc.dataset),
        },
    )
}

fn matched_dataset_resolution(
    summary: &TetFileSummaryV1,
    dataset_idx: usize,
    rec: &DatasetRecordV1,
    chunk_rows: usize,
) -> DatasetResolution {
    DatasetResolution {
        matched: true,
        dataset_index: Some(dataset_idx),
        dtype: Some(rec.dtype),
        shape: Some(rec.shape.clone()),
        chunk_shape: Some(rec.chunk_shape.clone()),
        chunk_index_rows: Some(chunk_rows),
        dataset_f32_bytes: dataset_tensor_bytes(rec.dtype, &rec.shape, DATASET_DTYPE_TAG_V1.f32),
        dataset_f64_bytes: dataset_tensor_bytes(rec.dtype, &rec.shape, DATASET_DTYPE_TAG_V1.f64),
        dataset_i32_bytes: dataset_tensor_bytes(rec.dtype, &rec.shape, DATASET_DTYPE_TAG_V1.i32),
        dataset_i64_bytes: dataset_tensor_bytes(rec.dtype, &rec.shape, DATASET_DTYPE_TAG_V1.i64),
        dataset_u8_bytes: dataset_tensor_bytes(rec.dtype, &rec.shape, DATASET_DTYPE_TAG_V1.u8),
        dataset_u16_bytes: dataset_tensor_bytes(rec.dtype, &rec.shape, DATASET_DTYPE_TAG_V1.u16),
        dataset_i16_bytes: dataset_tensor_bytes(rec.dtype, &rec.shape, DATASET_DTYPE_TAG_V1.i16),
        file_execution: Some(summary.file_execution),
        available_datasets: None,
    }
}

struct MatchedExecutionCtx<'a> {
    doc: &'a QueryDocument,
    mmap: &'a [u8],
    read_plan: &'a ReadPlan,
    rec: &'a DatasetRecordV1,
    summary: &'a TetFileSummaryV1,
    tet_path: Option<&'a str>,
    raw_f32_preview_max: Option<usize>,
    spill_allowlist: Option<&'a SpillPathAllowlist>,
}

fn matched_dataset_execution(
    ctx: &MatchedExecutionCtx<'_>,
    message: &mut String,
) -> Result<Option<QueryExecutionPreview>, TetError> {
    let Some(limit) = ctx.raw_f32_preview_max else {
        return Ok(None);
    };
    if limit == 0 && ctx.doc.operation.is_none() && !query_requests_spill(ctx.doc) {
        return Err(TetError::Validation(
            "preview limit 0 without `operation` or `spill` would attach an empty execution block; omit `--execute`, set `spill`, or use a positive `--preview`".into(),
        ));
    }
    let default_spill;
    let spill_ref = match ctx.spill_allowlist {
        Some(p) => Some(p),
        None => {
            if let Some(tp) = ctx.tet_path {
                default_spill = SpillPathAllowlist::default_for_tet(
                    Path::new(tp),
                    std::iter::empty::<PathBuf>(),
                )?;
                Some(&default_spill)
            } else {
                None
            }
        }
    };
    let preview =
        build_execution_preview(&crate::query::engine::operations::ExecutionPreviewInput {
            mmap: ctx.mmap,
            plan: ctx.read_plan,
            dtype: ctx.rec.dtype,
            operation: ctx.doc.operation.as_ref(),
            window: ctx.doc.window.as_ref(),
            output: ctx.doc.output.as_ref(),
            write: ctx.doc.write.as_ref(),
            source_dataset: &ctx.doc.dataset,
            max_f32: limit,
            budget: ExecutionBudget::resolve(
                &ctx.summary.file_execution,
                ctx.doc.execution.as_ref(),
            ),
            execution: ctx.doc.execution.as_ref(),
            spill_allowlist: spill_ref,
            tet_path: ctx.tet_path.map(Path::new),
        })?;
    if ctx.doc.operation.is_some() {
        message.push_str("; operation executed (see execution.memory_strategy and operation_*)");
    } else {
        message.push_str("; mmap f32 preview attached (see execution)");
    }
    Ok(Some(preview))
}

#[derive(Clone, Copy)]
struct PlanAttach<'a> {
    tet_path: Option<&'a str>,
    raw_f32_preview_max: Option<usize>,
    spill_allowlist: Option<&'a SpillPathAllowlist>,
    zones: Option<&'a GenerationZoneMaps>,
    index: Option<&'a GenerationSecondaryIndex>,
}

#[allow(clippy::too_many_lines)] // matched-path attach + execution stays together
fn plan_query_matched_dataset(
    summary: &TetFileSummaryV1,
    dataset_idx: usize,
    doc: &QueryDocument,
    mmap: &[u8],
    attach: &PlanAttach<'_>,
) -> Result<QueryResponse, TetError> {
    let rec = &summary.datasets[dataset_idx];
    let mut doc = doc.clone();
    let dataset_meta = summary.metadata.datasets.get(&doc.dataset);
    if let Some(align) = &doc.align {
        if attach.raw_f32_preview_max.is_none() {
            return Err(TetError::Validation(
                "`align` requires mmap execution with an explicit preview limit (e.g. `--execute --preview 0`)".into(),
            ));
        }
        let alignment =
            crate::query::align::check_alignment(summary, &doc.dataset, rec, dataset_meta, align)?;
        let rows = chunk_index_rows_for_dataset(summary, dataset_idx);
        let message = if alignment.aligned {
            format!(
                "query accepted; dataset matched; alignment check ok with {:?}",
                alignment.with
            )
        } else {
            format!(
                "query accepted; dataset matched; alignment check reported mismatch with {:?}",
                alignment.with
            )
        };
        let preview = QueryExecutionPreview {
            memory_strategy: Some("alignment_check"),
            alignment: Some(alignment),
            ..Default::default()
        };
        return Ok(query_response(
            &doc,
            message,
            attach.tet_path,
            Some(matched_dataset_resolution(summary, dataset_idx, rec, rows)),
            None,
            Some(preview),
        ));
    }
    resolve_query_document_axes(&mut doc, dataset_meta, rec.shape.len())?;
    resolve_query_document_selection(&mut doc, dataset_meta, &rec.shape)?;
    let rows = chunk_index_rows_for_dataset(summary, dataset_idx);
    let mut read_plan =
        build_matched_read_plan(summary, dataset_idx, &doc, attach.zones, attach.index)?;
    if let Some(group_by) = &doc.group_by {
        read_plan.label_group_remap =
            crate::query::label_group::build_label_group_remap(group_by, dataset_meta, &read_plan)?;
    }
    if doc.operation.is_some() && attach.raw_f32_preview_max.is_none() {
        return Err(TetError::Validation(
            "`operation` requires mmap execution with an explicit preview limit (e.g. `--execute --preview-f32 64`, or `--preview-f32 0` to omit preview floats)".into(),
        ));
    }
    let mut message = if read_plan.zone_filter_skipped_chunks > 0 {
        format!(
            "query accepted; dataset matched; read_plan lists mmap payload regions for touched chunks (zone filter skipped {})",
            read_plan.zone_filter_skipped_chunks
        )
    } else {
        String::from(
            "query accepted; dataset matched; read_plan lists mmap payload regions for touched chunks",
        )
    };
    if read_plan.cell_filter.is_some() {
        message.push_str("; cell filter applied to kept chunks");
    }
    if read_plan.label_eq.is_some() {
        message.push_str("; label filter applied to kept chunks");
    }
    if read_plan.index_filter_skipped_chunks > 0 {
        use std::fmt::Write as _;
        let _ = write!(
            message,
            "; index filter skipped {}",
            read_plan.index_filter_skipped_chunks
        );
    }
    let mut execution = matched_dataset_execution(
        &MatchedExecutionCtx {
            doc: &doc,
            mmap,
            read_plan: &read_plan,
            rec,
            summary,
            tet_path: attach.tet_path,
            raw_f32_preview_max: attach.raw_f32_preview_max,
            spill_allowlist: attach.spill_allowlist,
        },
        &mut message,
    )?;
    if let Some(preview) = execution.as_mut()
        && let Some(count) = group_count_for(&doc, &read_plan)
    {
        preview.group_count = Some(count);
        let _ = write!(message, "; group_count={count}");
    }
    Ok(query_response(
        &doc,
        message,
        attach.tet_path,
        Some(matched_dataset_resolution(summary, dataset_idx, rec, rows)),
        Some(read_plan),
        execution,
    ))
}

fn plan_query_unmatched_dataset(
    doc: &QueryDocument,
    tet_path: Option<&str>,
    summary: &TetFileSummaryV1,
) -> QueryResponse {
    let names: Vec<String> = summary.datasets.iter().map(|d| d.name.clone()).collect();
    query_response(
        doc,
        "query accepted; dataset name not found in this file (see catalog.available_datasets)",
        tet_path,
        Some(DatasetResolution {
            matched: false,
            dataset_index: None,
            dtype: None,
            shape: None,
            chunk_shape: None,
            chunk_index_rows: None,
            dataset_f32_bytes: None,
            dataset_f64_bytes: None,
            dataset_i32_bytes: None,
            dataset_i64_bytes: None,
            dataset_u8_bytes: None,
            dataset_u16_bytes: None,
            dataset_i16_bytes: None,
            file_execution: Some(summary.file_execution),
            available_datasets: Some(names),
        }),
        None,
        None,
    )
}

//! Build [`ReadPlan`] from catalog summary and chunk coordinate list.

use crate::catalog::{
    ChunkIndexEntryV1, ChunkZoneStat, DatasetMetadataV1, GenerationSecondaryIndex,
    GenerationZoneMaps, MAX_NDIM, TetFileSummaryV1,
};
use crate::query::plan::index_pushdown::{
    eq_chunk_allowlist, intersect_allowlists, label_eq_chunk_allowlist, sorted_axis_chunk_allowlist,
};
use crate::query::plan::zone_pushdown::zone_may_match_filter;
use crate::query::types::{
    CellPredicate, LabelEqPred, LabelFilterSpec, PlannedChunkIo, ReadPlan, TetError, ValueFilter,
};

pub(crate) struct ReadPlanGeometry<'a> {
    dataset_shape: &'a [u64],
    chunk_shape: &'a [u64],
    g0: &'a [u64],
    g1_exclusive: &'a [u64],
    step: &'a [u64],
}

impl<'a> ReadPlanGeometry<'a> {
    pub(crate) const fn new(
        dataset_shape: &'a [u64],
        chunk_shape: &'a [u64],
        g0: &'a [u64],
        g1_exclusive: &'a [u64],
        step: &'a [u64],
    ) -> Self {
        Self {
            dataset_shape,
            chunk_shape,
            g0,
            g1_exclusive,
            step,
        }
    }
}

/// Optional zone-map / secondary-index pushdown inputs for [`build_read_plan`].
pub(crate) struct ZonePushdown<'a> {
    pub zones: Option<&'a GenerationZoneMaps>,
    pub dtype: u32,
    pub filter: Option<&'a ValueFilter>,
    /// Footer label equality (`label_filter`), axis already resolved to a decimal index.
    pub label_filter: Option<&'a LabelFilterSpec>,
    /// Dataset fill (`_FillValue` / `missing_value` / `fill_value`) for cell-null tests.
    pub fill: Option<f64>,
    /// Sealed secondary index, when the open generation has a usable one.
    pub index: Option<&'a GenerationSecondaryIndex>,
    /// Footer metadata for sorted-axis / coord-label lookups.
    pub dataset_meta: Option<&'a DatasetMetadataV1>,
}

pub(crate) fn selection_logical_shape_u64(
    g0: &[u64],
    g1_exclusive: &[u64],
    step: &[u64],
) -> Result<Vec<u64>, TetError> {
    let nd = g0.len();
    if g1_exclusive.len() != nd || step.len() != nd {
        return Err(TetError::Validation(
            "internal: selection box/step length mismatch".into(),
        ));
    }
    let mut out = Vec::with_capacity(nd);
    for d in 0..nd {
        let span = g1_exclusive[d].checked_sub(g0[d]).ok_or_else(|| {
            TetError::Validation(format!(
                "selection span underflow on axis {d} (start={} stop={})",
                g0[d], g1_exclusive[d]
            ))
        })?;
        let n = span.div_ceil(step[d]).max(1);
        out.push(n);
    }
    Ok(out)
}

pub(crate) fn shape_product_usize(shape: &[u64]) -> Result<usize, TetError> {
    let mut p: usize = 1;
    for &s in shape {
        let su = usize::try_from(s).map_err(|_| {
            TetError::Validation(format!(
                "logical selection extent {s} is too large for this host"
            ))
        })?;
        p = p.checked_mul(su).ok_or_else(|| {
            TetError::Validation("logical selection element count overflow".into())
        })?;
    }
    Ok(p)
}

fn planned_chunk_io(
    ndim: usize,
    coord: &[u64; MAX_NDIM],
    entry: &ChunkIndexEntryV1,
    zone: Option<ChunkZoneStat>,
) -> PlannedChunkIo {
    PlannedChunkIo {
        chunk_index: coord[..ndim].to_vec(),
        payload_offset: entry.payload_offset,
        stored_byte_len: entry.stored_byte_len,
        raw_byte_len: entry.raw_byte_len,
        codec: entry.codec,
        zone,
    }
}

fn zone_for_chunk(
    zones: Option<&GenerationZoneMaps>,
    dataset_idx: usize,
    ndim: usize,
    coord: &[u64; MAX_NDIM],
) -> Option<ChunkZoneStat> {
    let zones = zones?;
    let index = &coord[..ndim];
    zones.chunks.iter().find_map(|z| {
        (z.dataset_id == dataset_idx as u64 && z.chunk_index == index).then(|| z.clone())
    })
}

fn find_chunk_entry<'a>(
    summary: &'a TetFileSummaryV1,
    dataset_idx: usize,
    ndim: usize,
    coord: &[u64; MAX_NDIM],
) -> Option<&'a ChunkIndexEntryV1> {
    summary.chunks.iter().find(|c| {
        c.dataset_id == dataset_idx as u64 && (0..ndim).all(|d| c.chunk_index[d] == coord[d])
    })
}

pub(crate) fn build_read_plan(
    summary: &TetFileSummaryV1,
    dataset_idx: usize,
    ndim: usize,
    coords: &[[u64; MAX_NDIM]],
    chunk_touch_policy: crate::query::types::ChunkTouchPolicy,
    geom: &ReadPlanGeometry<'_>,
    pushdown: &ZonePushdown<'_>,
) -> Result<ReadPlan, TetError> {
    let logical_selection_shape =
        selection_logical_shape_u64(geom.g0, geom.g1_exclusive, geom.step)?;
    let logical_element_count = shape_product_usize(&logical_selection_shape)?;
    let eq_allow = pushdown.filter.and_then(|filter| {
        pushdown
            .index
            .and_then(|index| eq_chunk_allowlist(index, dataset_idx as u64, pushdown.dtype, filter))
    });
    let sorted_allow = pushdown.index.and_then(|index| {
        pushdown.dataset_meta.and_then(|meta| {
            sorted_axis_chunk_allowlist(
                index,
                dataset_idx as u64,
                meta,
                geom.dataset_shape,
                geom.g0,
                geom.g1_exclusive,
            )
        })
    });
    let label_eq_pred = pushdown
        .label_filter
        .map(|lf| build_label_eq_pred(lf, pushdown.dataset_meta, geom.dataset_shape))
        .transpose()?;
    let label_allow = pushdown.label_filter.and_then(|lf| {
        let axis: u32 = lf.axis.parse().ok()?;
        pushdown
            .index
            .and_then(|index| label_eq_chunk_allowlist(index, dataset_idx as u64, axis, &lf.eq))
    });
    let allowlist = intersect_allowlists([eq_allow, sorted_allow, label_allow]);
    let mut chunks = Vec::with_capacity(coords.len());
    let mut total_stored: u64 = 0;
    let mut zone_filter_skipped_chunks = 0usize;
    let mut index_filter_skipped_chunks = 0usize;
    for coord in coords {
        let entry = find_chunk_entry(summary, dataset_idx, ndim, coord).ok_or_else(|| {
            TetError::Validation(format!(
                "chunk index has no row for dataset_id={dataset_idx} chunk_index={:?}",
                &coord[..ndim]
            ))
        })?;
        let zone = zone_for_chunk(pushdown.zones, dataset_idx, ndim, coord);
        if let (Some(filter), Some(zone_stat)) = (pushdown.filter, zone.as_ref())
            && !zone_may_match_filter(pushdown.dtype, zone_stat, filter)
        {
            // Conservative pushdown: drop only when zone stats prove a miss.
            zone_filter_skipped_chunks = zone_filter_skipped_chunks.saturating_add(1);
            continue;
        }
        if let Some(allowed) = allowlist.as_ref()
            && !allowed.contains(&coord[..ndim].to_vec())
        {
            index_filter_skipped_chunks = index_filter_skipped_chunks.saturating_add(1);
            continue;
        }
        total_stored = total_stored
            .checked_add(entry.stored_byte_len)
            .ok_or_else(|| {
                TetError::Validation("total stored bytes overflow when summing read plan".into())
            })?;
        chunks.push(planned_chunk_io(ndim, coord, entry, zone));
    }
    Ok(ReadPlan {
        chunk_touch_policy,
        chunk_count: chunks.len(),
        zone_filter_skipped_chunks,
        index_filter_skipped_chunks,
        total_stored_bytes: total_stored,
        chunks,
        dataset_shape: geom.dataset_shape.to_vec(),
        chunk_shape: geom.chunk_shape.to_vec(),
        selection_box_start: geom.g0.to_vec(),
        selection_box_stop_exclusive: geom.g1_exclusive.to_vec(),
        selection_step: geom.step.to_vec(),
        logical_selection_shape,
        logical_element_count,
        cell_filter: pushdown
            .filter
            .map(|filter| CellPredicate::from_filter(filter, pushdown.fill)),
        label_eq: label_eq_pred,
        label_group_remap: None,
        dataset_fill: pushdown.fill,
    })
}

fn build_label_eq_pred(
    label_filter: &LabelFilterSpec,
    meta: Option<&DatasetMetadataV1>,
    dataset_shape: &[u64],
) -> Result<LabelEqPred, TetError> {
    let axis: usize = label_filter.axis.parse().map_err(|_| {
        TetError::Validation(format!(
            "invalid label_filter axis {:?} after name resolution",
            label_filter.axis
        ))
    })?;
    if axis >= dataset_shape.len() {
        return Err(TetError::Validation(format!(
            "label_filter axis {axis} out of range for rank {}",
            dataset_shape.len()
        )));
    }
    let meta = meta.ok_or_else(|| {
        TetError::Validation(
            "`label_filter` requires footer coordinate labels for this dataset".into(),
        )
    })?;
    let extent = dataset_shape[axis];
    let extent_usize = usize::try_from(extent)
        .map_err(|_| TetError::Validation("dataset axis extent does not fit usize".into()))?;
    let coord = coord_axis_for_label_filter(meta, axis, extent_usize).ok_or_else(|| {
        TetError::Validation(format!(
            "`label_filter` requires coordinate labels on axis {axis}"
        ))
    })?;
    if coord.labels.len() != extent_usize {
        return Err(TetError::Validation(format!(
            "coordinate labels on axis {axis} have length {}, expected {extent_usize}",
            coord.labels.len()
        )));
    }
    if !coord.labels.iter().any(|l| l == &label_filter.eq) {
        return Err(TetError::Validation(format!(
            "unknown coordinate label {:?} on axis {axis} (axis has {} labels)",
            label_filter.eq,
            coord.labels.len()
        )));
    }
    let matching: Vec<bool> = coord.labels.iter().map(|l| l == &label_filter.eq).collect();
    Ok(LabelEqPred { axis, matching })
}

fn coord_axis_for_label_filter(
    meta: &DatasetMetadataV1,
    d: usize,
    extent: usize,
) -> Option<&crate::catalog::CoordAxisV1> {
    let coords = meta.coords.as_ref()?;
    if let Some(names) = &meta.dim_names
        && let Some(name) = names.get(d)
        && let Some(c) = coords.get(name)
    {
        return Some(c);
    }
    if let Some(c) = coords.get(d.to_string().as_str()) {
        return Some(c);
    }
    let mut matches = coords.values().filter(|c| c.labels.len() == extent);
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

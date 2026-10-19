//! Chunk payload I/O: mmap slices, codec decode, and per-element visitation.

use std::borrow::Cow;

use crate::catalog::{CHUNK_PAYLOAD_CODEC_V1, MAX_NDIM, tile};
use crate::query::decode::dense_visit::{dense_tile_logical_base, logical_index_unit_step};
use crate::query::decode::indexing::linear_rm_index;
use crate::query::fold::reduction::{ArgIndexAccum, ReductionKind, ValueAccum};
use crate::query::types::{
    CellPredicate, INTEGER_FILTER_FILL_MSG, PlannedChunkIo, ReadPlan, TetError, fold_keeps_cell,
};
use crate::utils::{
    f16_le, f32_le, f64_le, i16_le, i32_le, i64_le, u8_le, u16_le, u32_le, u64_le, wire,
};

struct PreparedChunk<'a> {
    bytes_read: u64,
    raw_bytes: Cow<'a, [u8]>,
    nelem: usize,
    tile: Vec<u64>,
    chunk_coord: [u64; MAX_NDIM],
    ndim: usize,
}

/// Wire element type for chunk decode / visit / scatter.
trait ChunkElem: Copy {
    const ELEM_SIZE: usize;
    fn read_at(raw: &[u8], index: usize) -> Self;
}

impl ChunkElem for f32 {
    const ELEM_SIZE: usize = 4;
    fn read_at(raw: &[u8], index: usize) -> Self {
        f32_le::read_f32_le_at(raw, index)
    }
}

impl ChunkElem for f64 {
    const ELEM_SIZE: usize = 8;
    fn read_at(raw: &[u8], index: usize) -> Self {
        f64_le::read_f64_le_at(raw, index)
    }
}

impl ChunkElem for i32 {
    const ELEM_SIZE: usize = 4;
    fn read_at(raw: &[u8], index: usize) -> Self {
        i32_le::read_i32_le_at(raw, index)
    }
}

impl ChunkElem for i64 {
    const ELEM_SIZE: usize = 8;
    fn read_at(raw: &[u8], index: usize) -> Self {
        i64_le::read_i64_le_at(raw, index)
    }
}

impl ChunkElem for u8 {
    const ELEM_SIZE: usize = 1;
    fn read_at(raw: &[u8], index: usize) -> Self {
        u8_le::read_u8_le_at(raw, index)
    }
}

impl ChunkElem for u16 {
    const ELEM_SIZE: usize = 2;
    fn read_at(raw: &[u8], index: usize) -> Self {
        u16_le::read_u16_le_at(raw, index)
    }
}

impl ChunkElem for i16 {
    const ELEM_SIZE: usize = 2;
    fn read_at(raw: &[u8], index: usize) -> Self {
        i16_le::read_i16_le_at(raw, index)
    }
}

impl ChunkElem for u32 {
    const ELEM_SIZE: usize = 4;
    fn read_at(raw: &[u8], index: usize) -> Self {
        u32_le::read_u32_le_at(raw, index)
    }
}

impl ChunkElem for u64 {
    const ELEM_SIZE: usize = 8;
    fn read_at(raw: &[u8], index: usize) -> Self {
        u64_le::read_u64_le_at(raw, index)
    }
}

impl ChunkElem for half::f16 {
    const ELEM_SIZE: usize = 2;
    fn read_at(raw: &[u8], index: usize) -> Self {
        f16_le::read_f16_le_at(raw, index)
    }
}

/// Scalar fold hooks for floating chunk payloads (`f32` / `f64`).
trait FoldChunkElem: ChunkElem {
    fn as_f64(v: Self) -> f64;
    fn push_le_bytes(value: &mut ValueAccum, raw: &[u8], kind: ReductionKind);
    fn push_value(value: &mut ValueAccum, v: Self);
    fn push_arg(arg: &mut ArgIndexAccum, li: u64, v: Self, kind: ReductionKind);
}

impl FoldChunkElem for f32 {
    fn as_f64(v: Self) -> f64 {
        f64::from(v)
    }
    fn push_le_bytes(value: &mut ValueAccum, raw: &[u8], kind: ReductionKind) {
        value.push_f32_le_bytes(raw, kind);
    }
    fn push_value(value: &mut ValueAccum, v: Self) {
        value.push(v);
    }
    fn push_arg(arg: &mut ArgIndexAccum, li: u64, v: Self, kind: ReductionKind) {
        arg.push(li, v, kind);
    }
}

impl FoldChunkElem for f64 {
    fn as_f64(v: Self) -> f64 {
        v
    }
    fn push_le_bytes(value: &mut ValueAccum, raw: &[u8], kind: ReductionKind) {
        value.push_f64_le_bytes(raw, kind);
    }
    fn push_value(value: &mut ValueAccum, v: Self) {
        value.push_f64(v);
    }
    fn push_arg(arg: &mut ArgIndexAccum, li: u64, v: Self, kind: ReductionKind) {
        arg.push_f64(li, v, kind);
    }
}

fn u64_to_usize(field: &'static str, v: u64) -> Result<usize, TetError> {
    usize::try_from(v)
        .map_err(|_| TetError::Validation(format!("{field}={v} is too large for this platform")))
}

fn map_span_err(e: wire::SpanError, chunk_index: &[u64], mmap_len: usize) -> TetError {
    match e {
        wire::SpanError::AddOverflow => TetError::Validation("payload byte range overflow".into()),
        wire::SpanError::OutOfBounds { end } => TetError::Validation(format!(
            "chunk_index={chunk_index:?}: payload byte range extends past mmap length {mmap_len} (end={end})"
        )),
    }
}

fn map_codec_err(e: &crate::catalog::CatalogError, chunk_index: &[u64]) -> TetError {
    TetError::Validation(format!("chunk_index={chunk_index:?}: {e}"))
}

pub(crate) fn planned_chunk_stored_slice<'a>(
    mmap: &'a [u8],
    c: &PlannedChunkIo,
) -> Result<&'a [u8], TetError> {
    let off = u64_to_usize("payload_offset", c.payload_offset)?;
    let len = u64_to_usize("stored_byte_len", c.stored_byte_len)?;
    let range = wire::checked_usize_subslice(off, len, mmap.len())
        .map_err(|e| map_span_err(e, &c.chunk_index, mmap.len()))?;
    Ok(&mmap[range])
}

pub(crate) fn decode_planned_chunk_bytes<'a>(
    stored: &'a [u8],
    c: &PlannedChunkIo,
) -> Result<Cow<'a, [u8]>, TetError> {
    CHUNK_PAYLOAD_CODEC_V1
        .decode_tile_payload(stored, c.raw_byte_len, c.stored_byte_len, c.codec)
        .map_err(|e| map_codec_err(&e, &c.chunk_index))
}

/// Map each planned chunk to a subslice of `mmap` (zero-copy).
///
/// # Errors
///
/// Returns [`TetError::Validation`] if any planned chunk uses a non-raw codec, if payload
/// offsets or lengths are invalid or extend past `mmap`, or if stored payload decode fails.
pub fn planned_chunk_mmap_slices<'a>(
    mmap: &'a [u8],
    plan: &ReadPlan,
) -> Result<Vec<&'a [u8]>, TetError> {
    let mut out = Vec::with_capacity(plan.chunks.len());
    for c in &plan.chunks {
        if !CHUNK_PAYLOAD_CODEC_V1.is_raw(c.codec) {
            return Err(TetError::Validation(format!(
                "planned_chunk_mmap_slices requires codec {} (raw); got codec={} for chunk_index={:?}",
                CHUNK_PAYLOAD_CODEC_V1.raw, c.codec, c.chunk_index
            )));
        }
        let stored = planned_chunk_stored_slice(mmap, c)?;
        decode_planned_chunk_bytes(stored, c)?;
        out.push(stored);
    }
    Ok(out)
}

fn global_matches_strided_selection(g: &[u64], plan: &ReadPlan) -> bool {
    (0..g.len()).all(|d| {
        let x = g[d];
        let st = plan.selection_box_start[d];
        let hi = plan.selection_box_stop_exclusive[d];
        let step = plan.selection_step[d];
        x >= st && x < hi && (x - st).is_multiple_of(step)
    })
}

fn prepare_chunk<'a, E: ChunkElem>(
    mmap: &'a [u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
) -> Result<PreparedChunk<'a>, TetError> {
    let ndim = plan.dataset_shape.len();
    if c.chunk_index.len() != ndim {
        return Err(TetError::Validation(format!(
            "chunk_index={:?} rank {} != dataset rank {}",
            c.chunk_index,
            c.chunk_index.len(),
            ndim
        )));
    }
    let stored = planned_chunk_stored_slice(mmap, c)?;
    let bytes_read = stored.len() as u64;
    let raw_bytes = decode_planned_chunk_bytes(stored, c)?;
    if !raw_bytes.len().is_multiple_of(E::ELEM_SIZE) {
        return Err(TetError::Validation(format!(
            "chunk raw length {} is not a multiple of {} (chunk_index={:?})",
            raw_bytes.len(),
            E::ELEM_SIZE,
            c.chunk_index
        )));
    }
    let nelem = raw_bytes.len() / E::ELEM_SIZE;
    let mut chunk_coord = [0u64; MAX_NDIM];
    chunk_coord[..ndim].copy_from_slice(&c.chunk_index[..ndim]);
    let tile = tile::tile_extent(&plan.dataset_shape, &plan.chunk_shape, &chunk_coord, ndim);
    let tile_elems: u64 = tile
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(b))
        .ok_or_else(|| TetError::Validation("tile element count overflow".into()))?;
    let tile_elems_us = usize::try_from(tile_elems)
        .map_err(|_| TetError::Validation("tile element count too large for this host".into()))?;
    if tile_elems_us != nelem {
        return Err(TetError::Validation(format!(
            "chunk_index={:?}: tile has {tile_elems_us} values but raw_byte_len implies {nelem}",
            c.chunk_index
        )));
    }
    Ok(PreparedChunk {
        bytes_read,
        raw_bytes,
        nelem,
        tile,
        chunk_coord,
        ndim,
    })
}

fn logical_index_for_global(
    plan: &ReadPlan,
    global: &[u64],
    ndim: usize,
) -> Result<usize, TetError> {
    if plan.selection_step.iter().all(|&s| s == 1) {
        return logical_index_unit_step(plan, global, ndim);
    }
    let mut lc = [0usize; MAX_NDIM];
    for (d, &x) in global.iter().enumerate().take(ndim) {
        let st = plan.selection_box_start[d];
        let q = (x - st) / plan.selection_step[d];
        lc[d] = usize::try_from(q).map_err(|_| {
            TetError::Validation(format!(
                "logical coordinate on axis {d} does not fit usize on this host"
            ))
        })?;
    }
    linear_rm_index(&lc[..ndim], &plan.logical_selection_shape)
}

fn visit_prepared_chunk_strided<E, F>(
    prep: &PreparedChunk<'_>,
    plan: &ReadPlan,
    mut visit: F,
) -> Result<(), TetError>
where
    E: ChunkElem,
    F: FnMut(usize, E) -> Result<(), TetError>,
{
    let ndim = prep.ndim;
    let mut global = [0u64; MAX_NDIM];
    for k in 0..prep.nelem {
        let local = tile::local_coords_from_linear(k as u64, &prep.tile, ndim);
        for (d, gv) in global.iter_mut().enumerate().take(ndim) {
            *gv = prep.chunk_coord[d]
                .saturating_mul(plan.chunk_shape[d])
                .saturating_add(local[d]);
        }
        if !global_matches_strided_selection(&global[..ndim], plan) {
            continue;
        }
        let li = logical_index_for_global(plan, &global[..ndim], ndim)?;
        visit(li, E::read_at(&prep.raw_bytes, k))?;
    }
    Ok(())
}

fn visit_prepared_chunk<E, F>(
    prep: &PreparedChunk<'_>,
    plan: &ReadPlan,
    mut visit: F,
) -> Result<(), TetError>
where
    E: ChunkElem,
    F: FnMut(usize, E) -> Result<(), TetError>,
{
    if prep.ndim == 1
        && let Some(li_base) =
            dense_tile_logical_base(plan, &prep.chunk_coord[..prep.ndim], &prep.tile)?
    {
        for k in 0..prep.nelem {
            visit(li_base + k, E::read_at(&prep.raw_bytes, k))?;
        }
        return Ok(());
    }
    visit_prepared_chunk_strided::<E, _>(prep, plan, visit)
}

fn visit_planned_chunk_elem<E, F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    E: ChunkElem,
    F: FnMut(usize, E) -> Result<(), TetError>,
{
    let prep = prepare_chunk::<E>(mmap, plan, c)?;
    visit_prepared_chunk::<E, _>(&prep, plan, visit)?;
    Ok(prep.bytes_read)
}

fn fold_planned_chunk_elem<E>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    kind: ReductionKind,
    value: &mut ValueAccum,
    arg: &mut ArgIndexAccum,
    preview: &mut [E],
) -> Result<u64, TetError>
where
    E: FoldChunkElem,
{
    let prep = prepare_chunk::<E>(mmap, plan, c)?;
    let preview_len = preview.len();
    let filter_cells = plan.cell_filter.is_some();
    if prep.ndim == 1
        && let Some(li_base) =
            dense_tile_logical_base(plan, &prep.chunk_coord[..prep.ndim], &prep.tile)?
    {
        if filter_cells {
            for k in 0..prep.nelem {
                let v = E::read_at(&prep.raw_bytes, k);
                let li = li_base + k;
                record_fold_cell(plan, kind, value, arg, preview, li, v);
            }
        } else {
            match kind {
                ReductionKind::ArgMin | ReductionKind::ArgMax => {
                    for k in 0..prep.nelem {
                        let v = E::read_at(&prep.raw_bytes, k);
                        let li = li_base + k;
                        E::push_arg(arg, li as u64, v, kind);
                        if li < preview_len {
                            preview[li] = v;
                        }
                    }
                }
                _ => {
                    E::push_le_bytes(value, &prep.raw_bytes, kind);
                    if preview_len > 0 {
                        let cap = preview_len.saturating_sub(li_base).min(prep.nelem);
                        for k in 0..cap {
                            preview[li_base + k] = E::read_at(&prep.raw_bytes, k);
                        }
                    }
                }
            }
        }
        return Ok(prep.bytes_read);
    }
    visit_prepared_chunk::<E, _>(&prep, plan, |li, v| {
        record_fold_cell(plan, kind, value, arg, preview, li, v);
        Ok(())
    })?;
    Ok(prep.bytes_read)
}

fn record_fold_cell<E: FoldChunkElem>(
    plan: &ReadPlan,
    kind: ReductionKind,
    value: &mut ValueAccum,
    arg: &mut ArgIndexAccum,
    preview: &mut [E],
    li: usize,
    v: E,
) {
    if li < preview.len() {
        preview[li] = v;
    }
    if !fold_keeps_cell(plan.cell_filter, E::as_f64(v)) {
        return;
    }
    match kind {
        ReductionKind::ArgMin | ReductionKind::ArgMax => E::push_arg(arg, li as u64, v, kind),
        ReductionKind::NanCount => value.push_nan_f64(E::as_f64(v)),
        ReductionKind::InfCount => value.push_inf_f64(E::as_f64(v)),
        ReductionKind::NullCount { fill } => value.push_null_f64(E::as_f64(v), fill),
        ReductionKind::NanMean => value.push_nan_mean_f64(E::as_f64(v)),
        ReductionKind::NanStd => value.push_nan_std_f64(E::as_f64(v)),
        _ => E::push_value(value, v),
    }
}

trait FloatScatter: ChunkElem {
    fn to_f64(self) -> f64;
    fn filtered_mask() -> Self;
}

impl FloatScatter for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn filtered_mask() -> Self {
        f32::NAN
    }
}

impl FloatScatter for f64 {
    fn to_f64(self) -> f64 {
        self
    }
    fn filtered_mask() -> Self {
        f64::NAN
    }
}

impl FloatScatter for half::f16 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn filtered_mask() -> Self {
        half::f16::NAN
    }
}

trait IntScatter: ChunkElem {
    fn to_f64(self) -> f64;
    fn from_fill(fill: f64) -> Result<Self, TetError>;
}

fn int_from_fill<T, F>(fill: f64, cast: F, roundtrip: impl Fn(T) -> f64) -> Result<T, TetError>
where
    T: Copy,
    F: FnOnce(f64) -> T,
{
    if !fill.is_finite() {
        return Err(TetError::Validation(
            "integer materialize fill must be a finite integer".into(),
        ));
    }
    let v = cast(fill);
    if roundtrip(v).to_bits() != fill.to_bits() {
        return Err(TetError::Validation(
            "integer materialize fill is outside this dtype".into(),
        ));
    }
    Ok(v)
}

impl IntScatter for i32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_fill(fill: f64) -> Result<Self, TetError> {
        int_from_fill(fill, |v| v as i32, f64::from)
    }
}

impl IntScatter for i64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_fill(fill: f64) -> Result<Self, TetError> {
        int_from_fill(fill, |v| v as i64, |v| v as f64)
    }
}

impl IntScatter for i16 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_fill(fill: f64) -> Result<Self, TetError> {
        int_from_fill(fill, |v| v as i16, f64::from)
    }
}

impl IntScatter for u8 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_fill(fill: f64) -> Result<Self, TetError> {
        int_from_fill(fill, |v| v as u8, f64::from)
    }
}

impl IntScatter for u16 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_fill(fill: f64) -> Result<Self, TetError> {
        int_from_fill(fill, |v| v as u16, f64::from)
    }
}

impl IntScatter for u32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_fill(fill: f64) -> Result<Self, TetError> {
        int_from_fill(fill, |v| v as u32, f64::from)
    }
}

impl IntScatter for u64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_fill(fill: f64) -> Result<Self, TetError> {
        int_from_fill(fill, |v| v as u64, |v| v as f64)
    }
}

fn integer_filter_mask<E: IntScatter>(pred: CellPredicate) -> Result<E, TetError> {
    let Some(fill) = pred.fill_f64() else {
        return Err(TetError::Validation(INTEGER_FILTER_FILL_MSG.into()));
    };
    E::from_fill(fill)
}

fn scatter_chunk_into<E>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [E],
) -> Result<u64, TetError>
where
    E: FloatScatter,
{
    visit_planned_chunk_elem::<E, _>(mmap, plan, c, |li, v| {
        if li < out.len() {
            out[li] = if fold_keeps_cell(plan.cell_filter, v.to_f64()) {
                v
            } else {
                E::filtered_mask()
            };
        }
        Ok(())
    })
}

fn scatter_chunk_into_option<E>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<E>],
) -> Result<u64, TetError>
where
    E: IntScatter,
{
    let mask = plan.cell_filter.map(integer_filter_mask::<E>).transpose()?;
    visit_planned_chunk_elem::<E, _>(mmap, plan, c, |li, v| {
        if li < out.len() {
            out[li] = Some(match mask {
                Some(masked) if !fold_keeps_cell(plan.cell_filter, v.to_f64()) => masked,
                _ => v,
            });
        }
        Ok(())
    })
}

/// Scalar fold over one planned `f32` chunk without per-element visitor callbacks.
pub(crate) fn fold_planned_chunk_f32(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    kind: ReductionKind,
    value: &mut ValueAccum,
    arg: &mut ArgIndexAccum,
    preview: &mut [f32],
) -> Result<u64, TetError> {
    fold_planned_chunk_elem::<f32>(mmap, plan, c, kind, value, arg, preview)
}

/// Scalar fold over one planned `f64` chunk without per-element visitor callbacks.
pub(crate) fn fold_planned_chunk_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    kind: ReductionKind,
    value: &mut ValueAccum,
    arg: &mut ArgIndexAccum,
    preview: &mut [f64],
) -> Result<u64, TetError> {
    fold_planned_chunk_elem::<f64>(mmap, plan, c, kind, value, arg, preview)
}

/// Visit each selected `f32` in a planned chunk.
pub(crate) fn visit_planned_chunk<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f32) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<f32, _>(mmap, plan, c, visit)
}

/// Visit each selected `f64` in a planned chunk.
pub(crate) fn visit_planned_chunk_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<f64, _>(mmap, plan, c, visit)
}

/// Visit each selected `i32` in a planned chunk.
pub(crate) fn visit_planned_chunk_i32<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, i32) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<i32, _>(mmap, plan, c, visit)
}

/// Visit each selected `i64` in a planned chunk.
pub(crate) fn visit_planned_chunk_i64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, i64) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<i64, _>(mmap, plan, c, visit)
}

/// Visit each selected `i32`, promoting values to `f64` for accumulators.
pub(crate) fn visit_planned_chunk_i32_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_i32(mmap, plan, c, |li, v| visit(li, f64::from(v)))
}

/// Visit each selected `i64`, promoting values to `f64` for accumulators.
pub(crate) fn visit_planned_chunk_i64_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_i64(mmap, plan, c, |li, v| visit(li, v as f64))
}

pub(crate) fn scatter_chunk_into_plan(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [f32],
) -> Result<u64, TetError> {
    scatter_chunk_into::<f32>(mmap, plan, c, out)
}

pub(crate) fn scatter_chunk_into_plan_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [f64],
) -> Result<u64, TetError> {
    scatter_chunk_into::<f64>(mmap, plan, c, out)
}

pub(crate) fn scatter_chunk_into_plan_i32(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<i32>],
) -> Result<u64, TetError> {
    scatter_chunk_into_option::<i32>(mmap, plan, c, out)
}

pub(crate) fn scatter_chunk_into_plan_i64(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<i64>],
) -> Result<u64, TetError> {
    scatter_chunk_into_option::<i64>(mmap, plan, c, out)
}

/// Visit each selected `u8` in a planned chunk.
pub(crate) fn visit_planned_chunk_u8<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, u8) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<u8, _>(mmap, plan, c, visit)
}

/// Visit each selected `u8`, promoting values to `f64` for accumulators.
pub(crate) fn visit_planned_chunk_u8_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_u8(mmap, plan, c, |li, v| visit(li, f64::from(v)))
}

pub(crate) fn scatter_chunk_into_plan_u8(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<u8>],
) -> Result<u64, TetError> {
    scatter_chunk_into_option::<u8>(mmap, plan, c, out)
}

/// Visit each selected `u16` in a planned chunk.
pub(crate) fn visit_planned_chunk_u16<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, u16) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<u16, _>(mmap, plan, c, visit)
}

/// Visit each selected `u16`, promoting values to `f64` for accumulators.
pub(crate) fn visit_planned_chunk_u16_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_u16(mmap, plan, c, |li, v| visit(li, f64::from(v)))
}

pub(crate) fn scatter_chunk_into_plan_u16(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<u16>],
) -> Result<u64, TetError> {
    scatter_chunk_into_option::<u16>(mmap, plan, c, out)
}

/// Visit each selected `i16` in a planned chunk.
pub(crate) fn visit_planned_chunk_i16<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, i16) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<i16, _>(mmap, plan, c, visit)
}

/// Visit each selected `i16`, promoting values to `f64` for accumulators.
pub(crate) fn visit_planned_chunk_i16_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_i16(mmap, plan, c, |li, v| visit(li, f64::from(v)))
}

pub(crate) fn scatter_chunk_into_plan_i16(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<i16>],
) -> Result<u64, TetError> {
    scatter_chunk_into_option::<i16>(mmap, plan, c, out)
}

pub(crate) fn visit_planned_chunk_u32<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, u32) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<u32, _>(mmap, plan, c, visit)
}

pub(crate) fn visit_planned_chunk_u32_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_u32(mmap, plan, c, |li, v| visit(li, f64::from(v)))
}

pub(crate) fn scatter_chunk_into_plan_u32(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<u32>],
) -> Result<u64, TetError> {
    scatter_chunk_into_option::<u32>(mmap, plan, c, out)
}

pub(crate) fn visit_planned_chunk_u64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, u64) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<u64, _>(mmap, plan, c, visit)
}

pub(crate) fn visit_planned_chunk_u64_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_u64(mmap, plan, c, |li, v| visit(li, v as f64))
}

pub(crate) fn scatter_chunk_into_plan_u64(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<u64>],
) -> Result<u64, TetError> {
    scatter_chunk_into_option::<u64>(mmap, plan, c, out)
}

pub(crate) fn visit_planned_chunk_f16<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, half::f16) -> Result<(), TetError>,
{
    visit_planned_chunk_elem::<half::f16, _>(mmap, plan, c, visit)
}

#[allow(dead_code)]
pub(crate) fn visit_planned_chunk_f16_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_f16(mmap, plan, c, |li, v| visit(li, f64::from(v)))
}

pub(crate) fn scatter_chunk_into_plan_f16(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [half::f16],
) -> Result<u64, TetError> {
    scatter_chunk_into::<half::f16>(mmap, plan, c, out)
}

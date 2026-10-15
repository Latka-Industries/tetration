//! Chunk grid dimensions and row-major `f32` tile extraction.

use super::{CatalogError, MAX_NDIM};

/// Number of chunks along each axis: `ceil(shape[d] / chunk_shape[d])`.
///
/// # Errors
///
/// Returns [`CatalogError::InvalidWriteSpec`] when any `chunk_shape[d] == 0` (avoids
/// `div_ceil` panic on untrusted catalog records — THI-376).
pub(crate) fn chunk_grid_counts(
    shape: &[u64],
    chunk_shape: &[u64],
) -> Result<Vec<u64>, CatalogError> {
    shape
        .iter()
        .zip(chunk_shape.iter())
        .map(|(&s, &cs)| {
            if cs == 0 {
                return Err(CatalogError::InvalidWriteSpec(
                    "chunk_shape entries must be non-zero",
                ));
            }
            Ok(s.div_ceil(cs))
        })
        .collect()
}

pub(crate) fn total_chunk_count(counts: &[u64]) -> Result<u64, CatalogError> {
    counts.iter().try_fold(1u64, |a, &b| {
        a.checked_mul(b).ok_or(CatalogError::InvalidWriteSpec(
            "chunk grid element count overflow",
        ))
    })
}

/// Uncompressed byte length of one tile (product of per-axis extent × `elem_size`).
pub(crate) fn tile_raw_byte_len(
    shape: &[u64],
    chunk_shape: &[u64],
    chunk_coord: &[u64],
    ndim: usize,
    elem_size: usize,
) -> Result<u64, CatalogError> {
    let tile = tile_extent(shape, chunk_shape, chunk_coord, ndim);
    let nelem: u64 = tile.iter().try_fold(1u64, |a, &b| a.checked_mul(b)).ok_or(
        CatalogError::InvalidWriteSpec("tile element count overflow"),
    )?;
    nelem
        .checked_mul(
            u64::try_from(elem_size)
                .map_err(|_| CatalogError::InvalidWriteSpec("element size overflow"))?,
        )
        .ok_or(CatalogError::InvalidWriteSpec("tile byte length overflow"))
}

/// Per-axis extent of the tile for this chunk (may be smaller than `chunk_shape` at edges).
///
/// Uses saturating arithmetic so malformed / out-of-grid `chunk_coord` values never panic
/// (verify and decode paths must stay panic-free on untrusted `.tet` bytes — THI-376).
pub(crate) fn tile_extent(
    shape: &[u64],
    chunk_shape: &[u64],
    chunk_coord: &[u64],
    ndim: usize,
) -> Vec<u64> {
    (0..ndim)
        .map(|d| {
            let start = chunk_coord[d].saturating_mul(chunk_shape[d]);
            let end = start.saturating_add(chunk_shape[d]).min(shape[d]);
            end.saturating_sub(start)
        })
        .collect()
}

/// Row-major stride in **elements** for axis `d` (last index is contiguous in memory).
pub(crate) fn row_major_stride_elems(shape: &[u64], d: usize) -> u64 {
    shape[d + 1..].iter().product()
}

/// Linear element index (row-major) for `global[..ndim]`.
pub(crate) fn linear_elem_row_major(
    global: &[u64],
    shape: &[u64],
    ndim: usize,
) -> Result<u64, CatalogError> {
    let mut idx: u64 = 0;
    for d in 0..ndim {
        let g = global[d];
        if g >= shape[d] {
            return Err(CatalogError::InvalidWriteSpec(
                "tile extraction produced out-of-bounds index",
            ));
        }
        let stride = row_major_stride_elems(shape, d);
        idx = idx
            .checked_add(
                g.checked_mul(stride)
                    .ok_or(CatalogError::InvalidWriteSpec("linear index overflow"))?,
            )
            .ok_or(CatalogError::InvalidWriteSpec("linear index overflow"))?;
    }
    Ok(idx)
}

/// Decode `k` into local tile coordinates (last axis varies fastest).
///
/// Empty axes (`tile[d] == 0`) yield `0` for that axis and leave `rem` unchanged (no `% 0`).
pub(crate) fn local_coords_from_linear(k: u64, tile: &[u64], ndim: usize) -> [u64; MAX_NDIM] {
    let mut rem = k;
    let mut local = [0u64; MAX_NDIM];
    for d in (0..ndim).rev() {
        let td = tile[d];
        if td == 0 {
            local[d] = 0;
            continue;
        }
        local[d] = rem % td;
        rem /= td;
    }
    local
}

/// Linear chunk coordinate `k` in `0..total` → chunk index tuple (last axis varies fastest).
pub(crate) fn chunk_coord_from_linear(k: u64, counts: &[u64], ndim: usize) -> [u64; MAX_NDIM] {
    let mut rem = k;
    let mut coord = [0u64; MAX_NDIM];
    for d in (0..ndim).rev() {
        let c = counts[d];
        coord[d] = rem % c;
        rem /= c;
    }
    coord
}

/// Half-open tile `[interval_lo, interval_hi)` intersects the arithmetic progression
/// `{s + k·step | k ≥ 0, s + k·step < e}`.
pub(crate) fn ap_intersects_half_open(
    s: u64,
    e: u64,
    step: u64,
    interval_lo: u64,
    interval_hi: u64,
) -> bool {
    if step == 0 || s >= e || interval_lo >= interval_hi {
        return false;
    }
    let end = interval_hi.min(e);
    if interval_lo >= end {
        return false;
    }
    let k_lo = if s >= interval_lo {
        0
    } else {
        (interval_lo - s).div_ceil(step)
    };
    let max_valid = end.saturating_sub(1);
    if s > max_valid {
        return false;
    }
    let k_hi = (max_valid - s) / step;
    k_lo <= k_hi
}

/// Chunk grid coordinates whose tiles intersect the strided per-axis selection
/// `s[d] + k·step[d] < g1[d]` (half-open stop per axis), with `step[d] ≥ 1`.
///
/// When every `step[d] == 1`, this matches the dense half-open box `∏ [g0[d], g1[d])`.
///
/// Results follow linear chunk index order (**last axis fastest**), matching the reference writer.
///
/// # Errors
///
/// Returns [`CatalogError::InvalidWriteSpec`] when slice lengths disagree, the global box is
/// empty or out of bounds, or chunk-grid arithmetic overflows.
pub fn chunk_coords_intersecting_strided(
    shape: &[u64],
    chunk_shape: &[u64],
    g0: &[u64],
    g1_exclusive: &[u64],
    step: &[u64],
) -> Result<Vec<[u64; MAX_NDIM]>, CatalogError> {
    let ndim = shape.len();
    if chunk_shape.len() != ndim
        || g0.len() != ndim
        || g1_exclusive.len() != ndim
        || step.len() != ndim
    {
        return Err(CatalogError::InvalidWriteSpec(
            "shape, chunk_shape, global box, and step must have the same rank",
        ));
    }
    for d in 0..ndim {
        if step[d] == 0 {
            return Err(CatalogError::InvalidWriteSpec(
                "step must be >= 1 on every axis",
            ));
        }
        if g1_exclusive[d] > shape[d] || g0[d] >= g1_exclusive[d] {
            return Err(CatalogError::InvalidWriteSpec(
                "global selection box must satisfy 0 <= start < stop <= shape[d] on every axis",
            ));
        }
    }
    let counts = chunk_grid_counts(shape, chunk_shape)?;
    let n = total_chunk_count(&counts)?;
    let mut out = Vec::new();
    for k in 0..n {
        let c = chunk_coord_from_linear(k, &counts, ndim);
        let mut touch = true;
        for d in 0..ndim {
            let cs = chunk_shape[d];
            let tile_start = c[d].saturating_mul(cs);
            let tile_end_exclusive = (tile_start + cs).min(shape[d]);
            if !ap_intersects_half_open(
                g0[d],
                g1_exclusive[d],
                step[d],
                tile_start,
                tile_end_exclusive,
            ) {
                touch = false;
                break;
            }
        }
        if touch {
            out.push(c);
        }
    }
    Ok(out)
}

/// Chunk grid coordinates whose logical tile intersects the half-open global element box
/// formed by the axis-wise intervals `[g0[d], g1_exclusive[d])` for `d = 0..ndim-1`.
///
/// Equivalent to [`chunk_coords_intersecting_strided`] with `step[d] = 1` everywhere.
///
/// Results are ordered by linear chunk index with **last axis varying fastest**, matching the
/// reference writer iteration order.
///
/// # Errors
///
/// Same failure modes as [`chunk_coords_intersecting_strided`].
pub fn chunk_coords_intersecting_global_box(
    shape: &[u64],
    chunk_shape: &[u64],
    g0: &[u64],
    g1_exclusive: &[u64],
) -> Result<Vec<[u64; MAX_NDIM]>, CatalogError> {
    let ndim = shape.len();
    if chunk_shape.len() != ndim || g0.len() != ndim || g1_exclusive.len() != ndim {
        return Err(CatalogError::InvalidWriteSpec(
            "shape, chunk_shape, and global box must have the same rank",
        ));
    }
    for d in 0..ndim {
        if g1_exclusive[d] > shape[d] || g0[d] >= g1_exclusive[d] {
            return Err(CatalogError::InvalidWriteSpec(
                "global selection box must satisfy 0 <= start < stop <= shape[d] on every axis",
            ));
        }
    }
    let steps = [1u64; MAX_NDIM];
    chunk_coords_intersecting_strided(shape, chunk_shape, g0, g1_exclusive, &steps[..ndim])
}

/// Copy one tile (row-major) from a full row-major tensor buffer into a new byte vec.
pub(crate) fn extract_tile_row_major(
    full: &[u8],
    shape: &[u64],
    chunk_shape: &[u64],
    chunk_coord: &[u64],
    ndim: usize,
    elem_size: usize,
) -> Result<Vec<u8>, CatalogError> {
    let tile = tile_extent(shape, chunk_shape, chunk_coord, ndim);
    let nelem: u64 = tile.iter().try_fold(1u64, |a, &b| a.checked_mul(b)).ok_or(
        CatalogError::InvalidWriteSpec("tile element count overflow"),
    )?;
    let nbytes_u64 = nelem
        .checked_mul(
            u64::try_from(elem_size)
                .map_err(|_| CatalogError::InvalidWriteSpec("element size overflow"))?,
        )
        .ok_or(CatalogError::InvalidWriteSpec("tile byte length overflow"))?;
    let nbytes = usize::try_from(nbytes_u64).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "tile_byte_length",
        value: nbytes_u64,
    })?;
    let mut out = vec![0u8; nbytes];
    let mut o = 0usize;
    for k in 0..nelem {
        let local = local_coords_from_linear(k, &tile, ndim);
        let mut global = [0u64; MAX_NDIM];
        for d in 0..ndim {
            global[d] = chunk_coord[d]
                .checked_mul(chunk_shape[d])
                .and_then(|s| s.checked_add(local[d]))
                .ok_or(CatalogError::InvalidWriteSpec(
                    "tile global coordinate overflow",
                ))?;
        }
        let li = linear_elem_row_major(&global[..ndim], shape, ndim)?;
        let src = usize::try_from(li)
            .map_err(|_| CatalogError::TooLargeForPlatform {
                field: "linear_element_index",
                value: li,
            })?
            .checked_mul(elem_size)
            .ok_or(CatalogError::InvalidWriteSpec("byte offset overflow"))?;
        if src + elem_size > full.len() {
            return Err(CatalogError::InvalidWriteSpec(
                "full tensor buffer shorter than implied by shape",
            ));
        }
        out[o..o + elem_size].copy_from_slice(&full[src..src + elem_size]);
        o += elem_size;
    }
    Ok(out)
}

/// Extract one row-major tile and copy it into `buf` (length must match the tile).
///
/// # Errors
///
/// Same as [`extract_tile_row_major`], plus [`CatalogError::InvalidWriteSpec`] when `buf.len()`
/// does not match the extracted tile.
pub(crate) fn write_tile_row_major_into(
    full: &[u8],
    shape: &[u64],
    chunk_shape: &[u64],
    chunk_coord: &[u64],
    ndim: usize,
    elem_size: usize,
    buf: &mut [u8],
) -> Result<(), CatalogError> {
    let tile = extract_tile_row_major(full, shape, chunk_shape, chunk_coord, ndim, elem_size)?;
    if tile.len() != buf.len() {
        return Err(CatalogError::InvalidWriteSpec(
            "tile buffer length mismatch for chunk",
        ));
    }
    buf.copy_from_slice(&tile);
    Ok(())
}

/// Copy one `f32` tile (row-major) from a full row-major tensor buffer into a new byte vec.
#[allow(dead_code)]
pub(crate) fn extract_f32_tile_row_major(
    full: &[u8],
    shape: &[u64],
    chunk_shape: &[u64],
    chunk_coord: &[u64],
    ndim: usize,
) -> Result<Vec<u8>, CatalogError> {
    extract_tile_row_major(full, shape, chunk_shape, chunk_coord, ndim, 4)
}

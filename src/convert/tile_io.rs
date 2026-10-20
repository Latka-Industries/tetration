//! Read one logical `.tet` tile from foreign array files (hyperslab / slice I/O).

use crate::catalog::tile;

use crate::utils::dtype::ElementDtype;

use super::ConvertError;
use super::shared::ImportTileRead;

/// Half-open `[start, end)` index ranges per axis for one tile.
pub(crate) fn tile_axis_ranges(spec: ImportTileRead<'_>) -> Vec<(u64, u64)> {
    let extent = tile::tile_extent(spec.shape, spec.chunk_shape, spec.chunk_coord, spec.ndim);
    (0..spec.ndim)
        .map(|d| {
            let start = spec.chunk_coord[d] * spec.chunk_shape[d];
            let end = start + extent[d];
            (start, end)
        })
        .collect()
}

#[cfg(feature = "tetration-hdf5")]
pub(crate) fn read_hdf5_tile_le_into(
    ds: &hdf5_metno::Dataset,
    spec: ImportTileRead<'_>,
    buf: &mut [u8],
) -> Result<(), ConvertError> {
    let ranges = tile_axis_ranges(spec);
    match spec.dtype {
        ElementDtype::F32 => read_hdf5_tile_typed_into::<f32>(ds, &ranges, spec.ndim, buf),
        ElementDtype::F64 => read_hdf5_tile_typed_into::<f64>(ds, &ranges, spec.ndim, buf),
        ElementDtype::U8 => read_hdf5_tile_typed_into::<u8>(ds, &ranges, spec.ndim, buf),
        ElementDtype::U16 => read_hdf5_tile_typed_into::<u16>(ds, &ranges, spec.ndim, buf),
        ElementDtype::I16 => read_hdf5_tile_typed_into::<i16>(ds, &ranges, spec.ndim, buf),
        ElementDtype::I32 => read_hdf5_tile_typed_into::<i32>(ds, &ranges, spec.ndim, buf),
        ElementDtype::I64 => read_hdf5_tile_typed_into::<i64>(ds, &ranges, spec.ndim, buf),
        ElementDtype::U32 => read_hdf5_tile_typed_into::<u32>(ds, &ranges, spec.ndim, buf),
        ElementDtype::U64 => read_hdf5_tile_typed_into::<u64>(ds, &ranges, spec.ndim, buf),
        ElementDtype::F16 => read_hdf5_tile_f16_into(ds, &ranges, spec.ndim, buf),
    }?;
    if let Some(cf) = spec.cf {
        cf.apply_tile_le(spec.dtype, buf)?;
    }
    Ok(())
}

#[cfg(feature = "tetration-hdf5")]
fn copy_pod_tile_to_buf<T: bytemuck::Pod>(
    values: &[T],
    buf: &mut [u8],
) -> Result<(), ConvertError> {
    let bytes = bytemuck::cast_slice(values);
    if bytes.len() != buf.len() {
        return Err(ConvertError::Hdf5(format!(
            "tile byte length mismatch (expected {}, got {})",
            buf.len(),
            bytes.len()
        )));
    }
    buf.copy_from_slice(bytes);
    Ok(())
}

#[cfg(feature = "tetration-hdf5")]
fn read_hdf5_tile_f16_into(
    ds: &hdf5_metno::Dataset,
    ranges: &[(u64, u64)],
    ndim: usize,
    buf: &mut [u8],
) -> Result<(), ConvertError> {
    let mut tmp = vec![0u16; buf.len() / 2];
    read_hdf5_tile_typed_into::<u16>(ds, ranges, ndim, bytemuck::cast_slice_mut(&mut tmp))?;
    let out: &mut [half::f16] = bytemuck::cast_slice_mut(buf);
    for (o, bits) in out.iter_mut().zip(tmp) {
        *o = half::f16::from_bits(bits);
    }
    Ok(())
}

#[cfg(feature = "tetration-hdf5")]
fn read_hdf5_tile_typed_into<T>(
    ds: &hdf5_metno::Dataset,
    ranges: &[(u64, u64)],
    ndim: usize,
    buf: &mut [u8],
) -> Result<(), ConvertError>
where
    T: hdf5_metno::H5Type + bytemuck::Pod,
{
    let err = ConvertError::from;
    match ndim {
        1 => {
            let sel = hdf5_range_selection(&ranges[..1])?;
            let vals = ds.read_slice_1d::<T, _>(sel).map_err(err)?;
            copy_pod_tile_to_buf(&vals.into_raw_vec_and_offset().0, buf)
        }
        2 => {
            let sel = hdf5_range_selection(ranges)?;
            let vals = ds.read_slice_2d::<T, _>(sel).map_err(err)?;
            copy_pod_tile_to_buf(&vals.into_raw_vec_and_offset().0, buf)
        }
        3..=8 => read_hdf5_tile_peeled_into::<T>(ds, ranges, ndim, buf, err),
        _ => Err(ConvertError::Catalog(
            crate::catalog::CatalogError::BadNdim { ndim },
        )),
    }
}

#[cfg(feature = "tetration-hdf5")]
fn read_hdf5_tile_peeled_into<T>(
    ds: &hdf5_metno::Dataset,
    ranges: &[(u64, u64)],
    ndim: usize,
    buf: &mut [u8],
    err: impl Fn(hdf5_metno::Error) -> ConvertError + Copy,
) -> Result<(), ConvertError>
where
    T: hdf5_metno::H5Type + bytemuck::Pod,
{
    let lead = ndim - 2;
    let mut out = Vec::new();
    let mut prefix = vec![0u64; lead];
    peel_hdf5_leading_axes_into::<T>(ds, ranges, lead, 0, &mut prefix, &mut out, err)?;
    if out.len() != buf.len() {
        return Err(ConvertError::Hdf5(format!(
            "tile byte length mismatch (expected {}, got {})",
            buf.len(),
            out.len()
        )));
    }
    buf.copy_from_slice(&out);
    Ok(())
}

#[cfg(feature = "tetration-hdf5")]
fn peel_hdf5_leading_axes_into<T>(
    ds: &hdf5_metno::Dataset,
    ranges: &[(u64, u64)],
    lead: usize,
    level: usize,
    prefix: &mut [u64],
    out: &mut Vec<u8>,
    err: impl Fn(hdf5_metno::Error) -> ConvertError + Copy,
) -> Result<(), ConvertError>
where
    T: hdf5_metno::H5Type + bytemuck::Pod,
{
    use hdf5_metno::{Hyperslab, Selection};

    if level == lead {
        let mut dims = Vec::with_capacity(ranges.len());
        for &idx in prefix.iter() {
            dims.push(hdf5_index_slice(idx)?);
        }
        for &(start, end) in &ranges[lead..] {
            dims.push(hdf5_range_slice(start, end)?);
        }
        let sel = Selection::Hyperslab(Hyperslab::new(dims));
        let slab = ds.read_slice_2d::<T, _>(sel).map_err(err)?;
        out.extend_from_slice(bytemuck::cast_slice(&slab.into_raw_vec_and_offset().0));
        return Ok(());
    }
    for idx in ranges[level].0..ranges[level].1 {
        prefix[level] = idx;
        peel_hdf5_leading_axes_into::<T>(ds, ranges, lead, level + 1, prefix, out, err)?;
    }
    Ok(())
}

#[cfg(feature = "tetration-hdf5")]
fn hdf5_range_selection(ranges: &[(u64, u64)]) -> Result<hdf5_metno::Selection, ConvertError> {
    use hdf5_metno::{Hyperslab, Selection};

    let mut dims = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        dims.push(hdf5_range_slice(start, end)?);
    }
    Ok(Selection::Hyperslab(Hyperslab::new(dims)))
}

#[cfg(feature = "tetration-hdf5")]
fn hdf5_range_slice(start: u64, end: u64) -> Result<hdf5_metno::SliceOrIndex, ConvertError> {
    Ok(hdf5_metno::SliceOrIndex::SliceTo {
        start: usize_from_u64(start, "hdf5 slice start")?,
        end: usize_from_u64(end, "hdf5 slice end")?,
        step: 1,
        block: 1,
    })
}

#[cfg(feature = "tetration-hdf5")]
fn hdf5_index_slice(index: u64) -> Result<hdf5_metno::SliceOrIndex, ConvertError> {
    Ok(hdf5_metno::SliceOrIndex::Index(usize_from_u64(
        index,
        "hdf5 index",
    )?))
}

#[cfg(feature = "tetration-netcdf")]
pub(crate) fn read_netcdf_tile_le_into(
    var: &netcdf::Variable<'_>,
    spec: ImportTileRead<'_>,
    buf: &mut [u8],
) -> Result<(), ConvertError> {
    if !(1..=8).contains(&spec.ndim) {
        return Err(ConvertError::Catalog(
            crate::catalog::CatalogError::BadNdim { ndim: spec.ndim },
        ));
    }
    let ranges = tile_axis_ranges(spec);
    let slices = netcdf_axis_ranges(&ranges)?;
    read_netcdf_raw_values_into(var, &slices, buf)?;
    if let Some(cf) = spec.cf {
        cf.apply_tile_le(spec.dtype, buf)?;
    }
    Ok(())
}

#[cfg(feature = "tetration-netcdf")]
fn netcdf_axis_ranges(ranges: &[(u64, u64)]) -> Result<Vec<std::ops::Range<usize>>, ConvertError> {
    ranges
        .iter()
        .map(|&(start, end)| Ok(usize_from_u64(start, "nc start")?..usize_from_u64(end, "nc end")?))
        .collect()
}

#[cfg(feature = "tetration-netcdf")]
fn read_netcdf_raw_values_into(
    var: &netcdf::Variable<'_>,
    slices: &[std::ops::Range<usize>],
    buf: &mut [u8],
) -> Result<(), netcdf::Error> {
    match slices {
        [s0] => var.get_raw_values_into(buf, s0.clone()),
        [s0, s1] => var.get_raw_values_into(buf, [s0.clone(), s1.clone()]),
        [s0, s1, s2] => var.get_raw_values_into(buf, [s0.clone(), s1.clone(), s2.clone()]),
        [s0, s1, s2, s3] => {
            var.get_raw_values_into(buf, [s0.clone(), s1.clone(), s2.clone(), s3.clone()])
        }
        [s0, s1, s2, s3, s4] => var.get_raw_values_into(
            buf,
            [s0.clone(), s1.clone(), s2.clone(), s3.clone(), s4.clone()],
        ),
        [s0, s1, s2, s3, s4, s5] => var.get_raw_values_into(
            buf,
            [
                s0.clone(),
                s1.clone(),
                s2.clone(),
                s3.clone(),
                s4.clone(),
                s5.clone(),
            ],
        ),
        [s0, s1, s2, s3, s4, s5, s6] => var.get_raw_values_into(
            buf,
            [
                s0.clone(),
                s1.clone(),
                s2.clone(),
                s3.clone(),
                s4.clone(),
                s5.clone(),
                s6.clone(),
            ],
        ),
        [s0, s1, s2, s3, s4, s5, s6, s7] => var.get_raw_values_into(
            buf,
            [
                s0.clone(),
                s1.clone(),
                s2.clone(),
                s3.clone(),
                s4.clone(),
                s5.clone(),
                s6.clone(),
                s7.clone(),
            ],
        ),
        _ => unreachable!("ndim validated by caller"),
    }
}

fn usize_from_u64(v: u64, field: &'static str) -> Result<usize, ConvertError> {
    usize::try_from(v).map_err(|_| {
        ConvertError::Catalog(crate::catalog::CatalogError::TooLargeForPlatform { field, value: v })
    })
}

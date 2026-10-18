//! Tier-C statistics over a fully materialized logical selection.

use std::cmp::Ordering;
use std::path::Path;

use memmap2::MmapMut;

use crate::query::{
    decode::indexing::coords_from_linear_row_major,
    engine::budget::MemoryStrategy,
    fold::partial_geometry::{partial_axis_layout, reduced_index},
    types::{Operation, OperationPreviewFields, ReadPlan, TetError},
};

use super::{
    LogicalF32Backing, LogicalF64Backing, MaterializedLogical,
    covariance::{observation_axis_index, run_covariance_correlation},
    materialized_logical_as_f64,
};

struct HistogramTierCParams<'a> {
    plan: &'a ReadPlan,
    axes: &'a [String],
    n: usize,
    shape: &'a [u64],
    bins: u32,
    range_min: Option<f64>,
    range_max: Option<f64>,
}

fn median_f64(values: &mut [f64]) -> Result<f64, TetError> {
    if values.is_empty() {
        return Err(TetError::Validation(
            "median requires at least one element".into(),
        ));
    }
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(Ordering::Equal);
    let n = values.len();
    let mid = n / 2;
    if n.is_multiple_of(2) {
        values.select_nth_unstable_by(mid, cmp);
        let hi = values[mid];
        values.select_nth_unstable_by(mid - 1, cmp);
        Ok(f64::midpoint(values[mid - 1], hi))
    } else {
        values.select_nth_unstable_by(mid, cmp);
        Ok(values[mid])
    }
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn quantile_f64(values: &mut [f64], q: f64) -> Result<f64, TetError> {
    if values.is_empty() {
        return Err(TetError::Validation(
            "quantile requires at least one element".into(),
        ));
    }
    if values.len() == 1 {
        return Ok(values[0]);
    }
    let pos = q * (values.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(Ordering::Equal);
    values.select_nth_unstable_by(lo, cmp);
    if lo == hi {
        return Ok(values[lo]);
    }
    values.select_nth_unstable_by(hi, cmp);
    let lower = values[lo];
    let upper = values[hi];
    Ok(lower + (upper - lower) * (pos - lo as f64))
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn histogram_f64(
    values: &[f64],
    bins: u32,
    range_min: Option<f64>,
    range_max: Option<f64>,
) -> Result<(Vec<f64>, Vec<f64>), TetError> {
    if values.is_empty() {
        return Err(TetError::Validation(
            "histogram requires at least one element".into(),
        ));
    }
    let bins = usize::try_from(bins)
        .map_err(|_| TetError::Validation("histogram bins overflow".into()))?;
    let min = range_min.unwrap_or_else(|| values.iter().copied().fold(f64::INFINITY, f64::min));
    let max = range_max.unwrap_or_else(|| values.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let mut counts = vec![0.0; bins];
    let mut edges = vec![0.0; bins + 1];
    if (max - min).abs() < f64::EPSILON {
        counts[0] = values.len() as f64;
        for (i, e) in edges.iter_mut().enumerate() {
            *e = if i == 0 { min } else { max };
        }
        return Ok((counts, edges));
    }
    let width = (max - min) / bins as f64;
    for (i, e) in edges.iter_mut().enumerate() {
        *e = min + width * i as f64;
    }
    if let Some(last) = edges.last_mut() {
        *last = max;
    }
    for &v in values {
        let mut idx = ((v - min) / width).floor() as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        counts[idx] += 1.0;
    }
    Ok((counts, edges))
}

fn median_f32(values: &mut [f32]) -> Result<f64, TetError> {
    let mut tmp: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
    median_f64(&mut tmp)
}

fn gather_partial_in_memory<V: Copy>(
    values: &[V],
    shape: &[u64],
    layout: &crate::query::fold::partial_geometry::PartialAxisLayout,
) -> Result<Vec<Vec<V>>, TetError> {
    let mut cells = vec![Vec::new(); layout.out_len];
    for (li, &value) in values.iter().enumerate() {
        let coords = coords_from_linear_row_major(li, shape)?;
        let oi = reduced_index(&coords, &layout.axis_set, &layout.out_shape)?;
        cells[oi].push(value);
    }
    Ok(cells)
}

fn gather_partial_in_memory_f64_from_f32(
    values: &[f32],
    shape: &[u64],
    layout: &crate::query::fold::partial_geometry::PartialAxisLayout,
) -> Result<Vec<Vec<f64>>, TetError> {
    let mut cells = vec![Vec::new(); layout.out_len];
    for (li, &value) in values.iter().enumerate() {
        let coords = coords_from_linear_row_major(li, shape)?;
        let oi = reduced_index(&coords, &layout.axis_set, &layout.out_shape)?;
        cells[oi].push(f64::from(value));
    }
    Ok(cells)
}

fn gather_partial_f64<F>(
    n: usize,
    shape: &[u64],
    layout: &crate::query::fold::partial_geometry::PartialAxisLayout,
    mut read_at: F,
) -> Result<Vec<Vec<f64>>, TetError>
where
    F: FnMut(usize) -> Result<f64, TetError>,
{
    let mut cells = vec![Vec::new(); layout.out_len];
    for li in 0..n {
        let coords = coords_from_linear_row_major(li, shape)?;
        let oi = reduced_index(&coords, &layout.axis_set, &layout.out_shape)?;
        cells[oi].push(read_at(li)?);
    }
    Ok(cells)
}

fn read_f64_at_spill_f32(path: &Path, li: usize) -> Result<f64, TetError> {
    let file = std::fs::File::open(path)
        .map_err(|e| TetError::Validation(format!("temp spill read failed: {e}")))?;
    let mmap = unsafe {
        memmap2::Mmap::map(&file)
            .map_err(|e| TetError::Validation(format!("temp spill mmap failed: {e}")))?
    };
    Ok(f64::from(bytemuck::cast_slice::<_, f32>(&mmap)[li]))
}

fn read_f64_at_spill_f64(path: &Path, li: usize) -> Result<f64, TetError> {
    let file = std::fs::File::open(path)
        .map_err(|e| TetError::Validation(format!("temp spill read failed: {e}")))?;
    let mmap = unsafe {
        memmap2::Mmap::map(&file)
            .map_err(|e| TetError::Validation(format!("temp spill mmap failed: {e}")))?
    };
    Ok(bytemuck::cast_slice::<_, f64>(&mmap)[li])
}

fn median_f32_spill_file(path: &Path, n: usize) -> Result<f64, TetError> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| TetError::Validation(format!("temp spill open failed: {e}")))?;
    let mut mmap = unsafe {
        MmapMut::map_mut(&file)
            .map_err(|e| TetError::Validation(format!("temp spill mmap mut failed: {e}")))?
    };
    let slice = bytemuck::cast_slice_mut(mmap.as_mut());
    median_f32(&mut slice[..n])
}

fn median_f64_spill_file(path: &Path, n: usize) -> Result<f64, TetError> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| TetError::Validation(format!("temp spill open failed: {e}")))?;
    let mut mmap = unsafe {
        MmapMut::map_mut(&file)
            .map_err(|e| TetError::Validation(format!("temp spill mmap mut failed: {e}")))?
    };
    let slice = bytemuck::cast_slice_mut(mmap.as_mut());
    median_f64(&mut slice[..n])
}

fn run_tier_c_median_f32(
    backing: &LogicalF32Backing,
    plan: &ReadPlan,
    axes: &[String],
    n: usize,
    shape: &[u64],
) -> Result<OperationPreviewFields, TetError> {
    if axes.is_empty() {
        let median = match backing {
            LogicalF32Backing::InMemory(v) => median_f32(&mut v.clone())?,
            LogicalF32Backing::TempSpill(temp) => median_f32_spill_file(temp.path(), n)?,
        };
        return Ok(OperationPreviewFields {
            element_count: Some(n),
            median: Some(median),
            ..OperationPreviewFields::default()
        });
    }
    let layout = partial_axis_layout(plan, axes)?;
    match backing {
        LogicalF32Backing::TempSpill(temp) => {
            let f64_cells = gather_partial_f64(n, shape, &layout, |li| {
                read_f64_at_spill_f32(temp.path(), li)
            })?;
            let medians: Vec<f64> = f64_cells
                .iter()
                .map(|c| median_f64(&mut c.clone()))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(OperationPreviewFields {
                element_count: Some(n),
                reduced_shape: Some(layout.out_shape),
                reduced_median: Some(medians),
                ..OperationPreviewFields::default()
            })
        }
        LogicalF32Backing::InMemory(v) => {
            let mut cells = gather_partial_in_memory(v, shape, &layout)?;
            let medians: Vec<f64> = cells
                .iter_mut()
                .map(|c| median_f32(c))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(OperationPreviewFields {
                element_count: Some(n),
                reduced_shape: Some(layout.out_shape),
                reduced_median: Some(medians),
                ..OperationPreviewFields::default()
            })
        }
    }
}

fn run_tier_c_median_f64(
    backing: &LogicalF64Backing,
    plan: &ReadPlan,
    axes: &[String],
    n: usize,
    shape: &[u64],
) -> Result<OperationPreviewFields, TetError> {
    if axes.is_empty() {
        let median = match backing {
            LogicalF64Backing::InMemory(v) => median_f64(&mut v.clone())?,
            LogicalF64Backing::TempSpill(temp) => median_f64_spill_file(temp.path(), n)?,
        };
        return Ok(OperationPreviewFields {
            element_count: Some(n),
            median: Some(median),
            ..OperationPreviewFields::default()
        });
    }
    let layout = partial_axis_layout(plan, axes)?;
    let mut cells = match backing {
        LogicalF64Backing::InMemory(v) => gather_partial_in_memory(v, shape, &layout)?,
        LogicalF64Backing::TempSpill(temp) => gather_partial_f64(n, shape, &layout, |li| {
            read_f64_at_spill_f64(temp.path(), li)
        })?,
    };
    let medians: Vec<f64> = cells
        .iter_mut()
        .map(|c| median_f64(c))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(OperationPreviewFields {
        element_count: Some(n),
        reduced_shape: Some(layout.out_shape),
        reduced_median: Some(medians),
        ..OperationPreviewFields::default()
    })
}

pub(crate) fn run_tier_c_operation(
    materialized: &MaterializedLogical,
    plan: &ReadPlan,
    op: &Operation,
) -> Result<OperationPreviewFields, TetError> {
    if matches!(
        materialized,
        MaterializedLogical::I32 { .. }
            | MaterializedLogical::I64 { .. }
            | MaterializedLogical::U8 { .. }
            | MaterializedLogical::U16 { .. }
            | MaterializedLogical::I16 { .. }
    ) {
        let vec = materialized_logical_as_f64(materialized)?;
        let synthetic = MaterializedLogical::F64 {
            backing: LogicalF64Backing::InMemory(vec),
            total_bytes_read_from_disk: 0,
            strategy: MemoryStrategy::InMemoryMaterialize,
        };
        return run_tier_c_operation(&synthetic, plan, op);
    }

    let n = plan.logical_element_count;
    let shape = &plan.logical_selection_shape;
    let axes = op.axes();

    match (materialized, op) {
        (MaterializedLogical::F32 { backing, .. }, Operation::Median { .. }) => {
            run_tier_c_median_f32(backing, plan, axes, n, shape)
        }
        (MaterializedLogical::F64 { backing, .. }, Operation::Median { .. }) => {
            run_tier_c_median_f64(backing, plan, axes, n, shape)
        }
        (MaterializedLogical::F32 { backing, .. }, Operation::Quantile { q, .. }) => {
            run_quantile_f32(backing, plan, axes, n, shape, *q)
        }
        (MaterializedLogical::F64 { backing, .. }, Operation::Quantile { q, .. }) => {
            run_quantile_f64(backing, plan, axes, n, shape, *q)
        }
        (MaterializedLogical::F32 { backing, .. }, Operation::Histogram { bins, min, max, .. }) => {
            run_histogram_f32(
                backing,
                &HistogramTierCParams {
                    plan,
                    axes,
                    n,
                    shape,
                    bins: *bins,
                    range_min: *min,
                    range_max: *max,
                },
            )
        }
        (MaterializedLogical::F64 { backing, .. }, Operation::Histogram { bins, min, max, .. }) => {
            run_histogram_f64(
                backing,
                &HistogramTierCParams {
                    plan,
                    axes,
                    n,
                    shape,
                    bins: *bins,
                    range_min: *min,
                    range_max: *max,
                },
            )
        }
        (_, Operation::Covariance { .. } | Operation::Correlation { .. }) => {
            let obs_axis = observation_axis_index(axes)?;
            let values = materialized_logical_as_f64(materialized)?;
            let correlation = matches!(op, Operation::Correlation { .. });
            run_covariance_correlation(&values, shape, obs_axis, correlation)
        }
        _ => Err(TetError::Validation(format!(
            "unsupported materialize-required operation: {op:?}"
        ))),
    }
}

fn run_quantile_f64(
    backing: &LogicalF64Backing,
    plan: &ReadPlan,
    axes: &[String],
    n: usize,
    shape: &[u64],
    q: f64,
) -> Result<OperationPreviewFields, TetError> {
    if axes.is_empty() {
        let mut cell = match backing {
            LogicalF64Backing::InMemory(v) => v.clone(),
            LogicalF64Backing::TempSpill(temp) => (0..n)
                .map(|li| read_f64_at_spill_f64(temp.path(), li))
                .collect::<Result<Vec<_>, _>>()?,
        };
        let val = quantile_f64(&mut cell, q)?;
        return Ok(OperationPreviewFields {
            element_count: Some(n),
            quantile: Some(val),
            ..OperationPreviewFields::default()
        });
    }
    let layout = partial_axis_layout(plan, axes)?;
    let mut cells = match backing {
        LogicalF64Backing::InMemory(v) => gather_partial_in_memory(v, shape, &layout)?,
        LogicalF64Backing::TempSpill(temp) => gather_partial_f64(n, shape, &layout, |li| {
            read_f64_at_spill_f64(temp.path(), li)
        })?,
    };
    let qs: Vec<f64> = cells
        .iter_mut()
        .map(|c| quantile_f64(c, q))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(OperationPreviewFields {
        element_count: Some(n),
        reduced_shape: Some(layout.out_shape),
        reduced_quantile: Some(qs),
        ..OperationPreviewFields::default()
    })
}

fn run_quantile_f32(
    backing: &LogicalF32Backing,
    plan: &ReadPlan,
    axes: &[String],
    n: usize,
    shape: &[u64],
    q: f64,
) -> Result<OperationPreviewFields, TetError> {
    if axes.is_empty() {
        let mut cell: Vec<f64> = match backing {
            LogicalF32Backing::InMemory(v) => v.iter().map(|&x| f64::from(x)).collect(),
            LogicalF32Backing::TempSpill(temp) => (0..n)
                .map(|li| read_f64_at_spill_f32(temp.path(), li))
                .collect::<Result<Vec<_>, _>>()?,
        };
        let val = quantile_f64(&mut cell, q)?;
        return Ok(OperationPreviewFields {
            element_count: Some(n),
            quantile: Some(val),
            ..OperationPreviewFields::default()
        });
    }
    let layout = partial_axis_layout(plan, axes)?;
    let mut cells = match backing {
        LogicalF32Backing::InMemory(v) => gather_partial_in_memory_f64_from_f32(v, shape, &layout)?,
        LogicalF32Backing::TempSpill(temp) => gather_partial_f64(n, shape, &layout, |li| {
            read_f64_at_spill_f32(temp.path(), li)
        })?,
    };
    let qs: Vec<f64> = cells
        .iter_mut()
        .map(|c| quantile_f64(c, q))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(OperationPreviewFields {
        element_count: Some(n),
        reduced_shape: Some(layout.out_shape),
        reduced_quantile: Some(qs),
        ..OperationPreviewFields::default()
    })
}

fn run_histogram_f64(
    backing: &LogicalF64Backing,
    p: &HistogramTierCParams<'_>,
) -> Result<OperationPreviewFields, TetError> {
    let HistogramTierCParams {
        plan,
        axes,
        n,
        shape,
        bins,
        range_min,
        range_max,
    } = *p;
    if axes.is_empty() {
        let values = match backing {
            LogicalF64Backing::InMemory(v) => v.clone(),
            LogicalF64Backing::TempSpill(temp) => (0..n)
                .map(|li| read_f64_at_spill_f64(temp.path(), li))
                .collect::<Result<Vec<_>, _>>()?,
        };
        let (counts, edges) = histogram_f64(&values, bins, range_min, range_max)?;
        return Ok(OperationPreviewFields {
            element_count: Some(n),
            histogram_counts: Some(counts),
            histogram_edges: Some(edges),
            ..OperationPreviewFields::default()
        });
    }
    let layout = partial_axis_layout(plan, axes)?;
    let cells = match backing {
        LogicalF64Backing::InMemory(v) => gather_partial_in_memory(v, shape, &layout)?,
        LogicalF64Backing::TempSpill(temp) => gather_partial_f64(n, shape, &layout, |li| {
            read_f64_at_spill_f64(temp.path(), li)
        })?,
    };
    let mut flat = Vec::with_capacity(cells.len() * bins as usize);
    for cell in &cells {
        let (counts, _) = histogram_f64(cell, bins, range_min, range_max)?;
        flat.extend(counts);
    }
    Ok(OperationPreviewFields {
        element_count: Some(n),
        reduced_shape: Some(layout.out_shape),
        reduced_histogram_counts: Some(flat),
        ..OperationPreviewFields::default()
    })
}

fn run_histogram_f32(
    backing: &LogicalF32Backing,
    p: &HistogramTierCParams<'_>,
) -> Result<OperationPreviewFields, TetError> {
    let HistogramTierCParams {
        plan,
        axes,
        n,
        shape,
        bins,
        range_min,
        range_max,
    } = *p;
    if axes.is_empty() {
        let values: Vec<f64> = match backing {
            LogicalF32Backing::InMemory(v) => v.iter().map(|&x| f64::from(x)).collect(),
            LogicalF32Backing::TempSpill(temp) => (0..n)
                .map(|li| read_f64_at_spill_f32(temp.path(), li))
                .collect::<Result<Vec<_>, _>>()?,
        };
        let (counts, edges) = histogram_f64(&values, bins, range_min, range_max)?;
        return Ok(OperationPreviewFields {
            element_count: Some(n),
            histogram_counts: Some(counts),
            histogram_edges: Some(edges),
            ..OperationPreviewFields::default()
        });
    }
    let layout = partial_axis_layout(plan, axes)?;
    let cells = match backing {
        LogicalF32Backing::InMemory(v) => gather_partial_in_memory_f64_from_f32(v, shape, &layout)?,
        LogicalF32Backing::TempSpill(temp) => gather_partial_f64(n, shape, &layout, |li| {
            read_f64_at_spill_f32(temp.path(), li)
        })?,
    };
    let mut flat = Vec::with_capacity(cells.len() * bins as usize);
    for cell in &cells {
        let (counts, _) = histogram_f64(cell, bins, range_min, range_max)?;
        flat.extend(counts);
    }
    Ok(OperationPreviewFields {
        element_count: Some(n),
        reduced_shape: Some(layout.out_shape),
        reduced_histogram_counts: Some(flat),
        ..OperationPreviewFields::default()
    })
}

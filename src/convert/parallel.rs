//! Parallel convert helpers: per-rayon-thread source file handles.
//!
//! HDF5/NetCDF tile readers are imported only when the matching `tetration-*` feature is
//! enabled (required for docs.rs `--no-default-features` builds).

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::Arc;

use crate::catalog::StreamTileJob;

use super::ConvertError;
use super::shared::ImportPlan;
#[cfg(feature = "tetration-hdf5")]
use super::tile_io::read_hdf5_tile_le_into;
#[cfg(feature = "tetration-netcdf")]
use super::tile_io::read_netcdf_tile_le_into;

/// Default worker count for parallel chunk reads (`available_parallelism`, clamped to 1..=64).
#[must_use]
pub fn default_parallel_jobs() -> usize {
    std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::<usize>::get)
        .clamp(1, 64)
}

/// Resolve `0` (auto) to [`default_parallel_jobs`]; otherwise clamp to 1..=64.
#[must_use]
pub fn resolve_parallel_jobs(jobs: usize) -> usize {
    if jobs == 0 {
        default_parallel_jobs()
    } else {
        jobs.clamp(1, 64)
    }
}

#[cfg(feature = "tetration-hdf5")]
pub(crate) struct H5ParallelSource {
    input: PathBuf,
    plans: Arc<[ImportPlan]>,
}

#[cfg(feature = "tetration-hdf5")]
impl H5ParallelSource {
    pub(crate) fn new(input: PathBuf, plans: Vec<ImportPlan>) -> Self {
        Self {
            input,
            plans: plans.into(),
        }
    }

    pub(crate) fn fill_tile(
        &self,
        job: &StreamTileJob<'_>,
        buf: &mut [u8],
    ) -> Result<(), ConvertError> {
        thread_local! {
            static CTX: RefCell<Option<H5ThreadCtx>> = const { RefCell::new(None) };
        }

        CTX.with(|slot| {
            let mut ctx = slot.borrow_mut();
            if ctx
                .as_ref()
                .is_none_or(|c| c.input != self.input || c.plans_len != self.plans.len())
            {
                let file = hdf5_metno::File::open(&self.input)
                    .map_err(|e| ConvertError::Hdf5(e.to_string()))?;
                *ctx = Some(H5ThreadCtx {
                    input: self.input.clone(),
                    plans_len: self.plans.len(),
                    file,
                });
            }
            let ctx = ctx.as_ref().expect("h5 thread ctx");
            let plan = &self.plans[job.dataset_id];
            let ds = ctx
                .file
                .dataset(&plan.name)
                .map_err(|e| ConvertError::Hdf5(e.to_string()))?;
            let spec = plan.tile_read(job);
            read_hdf5_tile_le_into(&ds, spec, buf)
        })
    }
}

#[cfg(feature = "tetration-hdf5")]
struct H5ThreadCtx {
    input: PathBuf,
    plans_len: usize,
    file: hdf5_metno::File,
}

#[cfg(feature = "tetration-netcdf")]
pub(crate) struct NetcdfParallelSource {
    input: PathBuf,
    plans: Arc<[ImportPlan]>,
}

#[cfg(feature = "tetration-netcdf")]
impl NetcdfParallelSource {
    pub(crate) fn new(input: PathBuf, plans: Vec<ImportPlan>) -> Self {
        Self {
            input,
            plans: plans.into(),
        }
    }

    pub(crate) fn fill_tile(
        &self,
        job: &StreamTileJob<'_>,
        buf: &mut [u8],
    ) -> Result<(), ConvertError> {
        thread_local! {
            static CTX: RefCell<Option<NetcdfThreadCtx>> = const { RefCell::new(None) };
        }

        CTX.with(|slot| {
            let mut ctx = slot.borrow_mut();
            if ctx
                .as_ref()
                .is_none_or(|c| c.input != self.input || c.plans_len != self.plans.len())
            {
                let file =
                    netcdf::open(&self.input).map_err(|e| ConvertError::Netcdf(e.to_string()))?;
                *ctx = Some(NetcdfThreadCtx {
                    input: self.input.clone(),
                    plans_len: self.plans.len(),
                    file,
                });
            }
            let ctx = ctx.as_ref().expect("netcdf thread ctx");
            let plan = &self.plans[job.dataset_id];
            let var = ctx
                .file
                .variable(&plan.name)
                .ok_or_else(|| ConvertError::Netcdf(format!("variable `{}` missing", plan.name)))?;
            let spec = plan.tile_read(job);
            read_netcdf_tile_le_into(&var, spec, buf)
        })
    }
}

#[cfg(feature = "tetration-netcdf")]
struct NetcdfThreadCtx {
    input: PathBuf,
    plans_len: usize,
    file: netcdf::File,
}

pub(crate) struct ZarrParallelSource {
    store: PathBuf,
    plans: Arc<[ImportPlan]>,
}

impl ZarrParallelSource {
    pub(crate) fn new(store: PathBuf, plans: Vec<ImportPlan>) -> Self {
        Self {
            store,
            plans: plans.into(),
        }
    }

    pub(crate) fn fill_tile(
        &self,
        job: &StreamTileJob<'_>,
        buf: &mut [u8],
    ) -> Result<(), ConvertError> {
        let plan = &self.plans[job.dataset_id];
        let array_rel = plan.zarr_array_rel.as_deref().unwrap_or(plan.name.as_str());
        let spec = plan.tile_read(job);
        super::zarr::read_zarr_tile_le_into(
            &self.store,
            array_rel,
            plan.zarr_zstd,
            plan.zarr_v2_sep,
            spec,
            buf,
        )
    }
}

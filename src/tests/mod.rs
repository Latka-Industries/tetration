//! In-crate integration tests (`cargo test --lib`); uses submodule paths and `pub(crate)` helpers.

#![allow(dead_code)]

/// Parallel `cargo test` opens the same read-only HDF5/NetCDF fixtures (and freshly
/// written temp exports) from many threads. HDF5 advisory locks then fail with
/// `errno = 11` / NetCDF `-101`. Disable locking once for the test process.
#[cfg(any(feature = "tetration-hdf5", feature = "tetration-netcdf"))]
pub(crate) fn ensure_hdf5_no_file_locking() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // SAFETY: test harness only; must run before any HDF5/NetCDF open in this process.
        unsafe {
            std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE");
        }
    });
}

#[cfg(any(feature = "tetration-hdf5", feature = "tetration-netcdf"))]
#[test]
fn init_hdf5_no_file_locking_for_parallel_tests() {
    ensure_hdf5_no_file_locking();
}

mod catalog;
mod cell_filter;
mod cli_convert;
mod cli_history;
mod cli_info;
mod cli_output;
mod cli_package;
mod cli_run;
mod cli_semantics;
mod concurrent_query;
mod conformance;
mod convert;
mod convert_s3;
mod covariance;
mod device;
mod distributed;
mod embed_materialize;
mod export;
#[cfg(feature = "tetration-ffi")]
mod ffi;
mod fixture;
mod fold;
mod fs_device;
mod gc;
mod job;
mod layout_roundtrip;
mod metadata;
mod package;
mod package_verify;
mod publish;
mod query;
mod query_fixtures;
mod reduction;
mod reject_corpus;
mod repair;
mod resolve_axes;
mod resolve_selection;
mod secondary_index;
mod selective_query;
mod semantics;
mod session;
mod shard;
mod small_tet_fixtures;
mod store;
mod store_cache;
mod store_ingest;
mod store_prune;
mod store_query;
#[cfg(feature = "tetration-s3")]
mod store_s3;
mod utils;
mod variance_simd;
mod verify;
mod verify_fixtures;
mod write_invariants;
mod zone;

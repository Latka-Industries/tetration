//! In-crate integration tests (`cargo test --lib`); uses submodule paths and `pub(crate)` helpers.

#![allow(dead_code)]

mod catalog;
mod cli_convert;
mod cli_history;
mod cli_info;
mod cli_output;
mod cli_package;
mod concurrent_query;
mod convert;
mod convert_s3;
mod covariance;
mod device;
mod embed_materialize;
mod export;
#[cfg(feature = "tetration-ffi")]
mod ffi;
mod fixture;
mod fold;
mod fs_device;
mod gc;
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
mod session;
mod shard;
mod small_tet_fixtures;
mod store;
mod store_cache;
mod store_ingest;
mod store_query;
#[cfg(feature = "tetration-s3")]
mod store_s3;
mod utils;
mod variance_simd;
mod verify;
mod verify_fixtures;
mod write_invariants;

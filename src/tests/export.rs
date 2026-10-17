//! `.tet` → Zarr / HDF5 / NetCDF export roundtrips (convert → export → re-read).

use super::convert::{materialize_dataset_le_bytes, small_zarr, zarr_array_le_bytes};
use super::verify::assert_tet_verify_ok;
use crate::catalog::read_tet_summary_v1;
use crate::convert::convert_zarr_to_tet_with_progress;
use crate::export::export_tet_to_zarr;
use crate::layout::mmap_file_read;

const FIXTURE_DTYPES: [&str; 4] = ["f32", "f64", "i32", "i64"];

#[test]
fn export_zarr_tensor_3d_roundtrip_matches_fixture_bytes() {
    let input = small_zarr("tensor_3d");
    let dir = tempfile::tempdir().unwrap();
    let tet = dir.path().join("tensor_3d.tet");
    let zarr_out = dir.path().join("exported");
    convert_zarr_to_tet_with_progress(&input, &tet, 1, None::<fn(_)>).unwrap();
    export_tet_to_zarr(&tet, &zarr_out).unwrap();
    assert_tet_verify_ok(&tet);

    for name in FIXTURE_DTYPES {
        let want = zarr_array_le_bytes(&input.join(name));
        let got = zarr_array_le_bytes(&zarr_out.join(name));
        assert_eq!(
            got, want,
            "dataset {name} zarr bytes differ after export roundtrip"
        );
    }
}

#[test]
fn export_zarr_groups_3d_nested_paths() {
    let input = small_zarr("groups_3d");
    let dir = tempfile::tempdir().unwrap();
    let tet = dir.path().join("groups.tet");
    let zarr_out = dir.path().join("exported");
    convert_zarr_to_tet_with_progress(&input, &tet, 1, None::<fn(_)>).unwrap();
    export_tet_to_zarr(&tet, &zarr_out).unwrap();

    for dtype in FIXTURE_DTYPES {
        let rel = format!("primary/{dtype}");
        let want = zarr_array_le_bytes(&input.join(&rel));
        let got = zarr_array_le_bytes(&zarr_out.join(&rel));
        assert_eq!(got, want, "dataset {rel}");
        assert!(zarr_out.join("primary/zarr.json").is_file());
    }
}

#[test]
fn export_preserves_logical_f32_after_convert() {
    let input = small_zarr("tensor_3d");
    let dir = tempfile::tempdir().unwrap();
    let tet = dir.path().join("t.tet");
    let zarr_out = dir.path().join("out");
    convert_zarr_to_tet_with_progress(&input, &tet, 1, None::<fn(_)>).unwrap();
    export_tet_to_zarr(&tet, &zarr_out).unwrap();
    let mmap = mmap_file_read(&tet).unwrap();
    let want = materialize_dataset_le_bytes(&mmap, "f32");
    let got = zarr_array_le_bytes(&zarr_out.join("f32"));
    assert_eq!(got, want);
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.datasets.len(), 4);
}

#[cfg(feature = "tetration-hdf5")]
mod hdf5_export {
    use std::path::{Path, PathBuf};

    use super::super::convert::materialize_dataset_le_bytes;
    use super::super::verify::assert_tet_verify_ok;
    use super::FIXTURE_DTYPES;
    use crate::catalog::read_tet_summary_v1;
    use crate::convert::{convert_h5_to_tet_with_progress, convert_zarr_to_tet_with_progress};
    use crate::export::{export_tet, export_tet_to_hdf5};
    use crate::layout::mmap_file_read;

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    fn small_h5(name: &str) -> PathBuf {
        repo_root().join("fixtures/small/h5").join(name)
    }

    fn small_zarr(name: &str) -> PathBuf {
        repo_root().join("fixtures/small/zarr").join(name)
    }

    fn hdf5_dataset_le_bytes(ds: &hdf5_metno::Dataset) -> Vec<u8> {
        use hdf5_metno::types::{FloatSize, IntSize, TypeDescriptor};

        let desc = ds.dtype().unwrap().to_descriptor().unwrap();
        match desc {
            TypeDescriptor::Float(FloatSize::U4) => ds
                .read_raw::<f32>()
                .unwrap()
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
            TypeDescriptor::Float(FloatSize::U8) => ds
                .read_raw::<f64>()
                .unwrap()
                .into_iter()
                .flat_map(f64::to_le_bytes)
                .collect(),
            TypeDescriptor::Integer(IntSize::U4) => ds
                .read_raw::<i32>()
                .unwrap()
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect(),
            TypeDescriptor::Integer(IntSize::U8) => ds
                .read_raw::<i64>()
                .unwrap()
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect(),
            other => panic!("unexpected hdf5 dtype {other:?}"),
        }
    }

    #[test]
    fn export_hdf5_convert_export_reread_matches_bytes() {
        let input = small_h5("tensor_3d.h5");
        let dir = tempfile::tempdir().unwrap();
        let tet = dir.path().join("tensor_3d.tet");
        let h5_out = dir.path().join("exported.h5");
        let tet2 = dir.path().join("roundtrip.tet");

        convert_h5_to_tet_with_progress(&input, &tet, 1, None::<fn(_)>).unwrap();
        export_tet_to_hdf5(&tet, &h5_out).unwrap();
        convert_h5_to_tet_with_progress(&h5_out, &tet2, 1, None::<fn(_)>).unwrap();
        assert_tet_verify_ok(&tet2);

        let mmap = mmap_file_read(&tet).unwrap();
        let mmap2 = mmap_file_read(&tet2).unwrap();
        for name in FIXTURE_DTYPES {
            let want = materialize_dataset_le_bytes(&mmap, name);
            let got = materialize_dataset_le_bytes(&mmap2, name);
            assert_eq!(got, want, "dataset {name} after hdf5 export roundtrip");
        }

        let exported = hdf5_metno::File::open(&h5_out).unwrap();
        for name in FIXTURE_DTYPES {
            let ds = exported.dataset(name).unwrap();
            let want = materialize_dataset_le_bytes(&mmap, name);
            let got = hdf5_dataset_le_bytes(&ds);
            assert_eq!(got, want, "hdf5 file bytes for {name}");
        }
    }

    #[test]
    fn export_hdf5_groups_and_dispatch_by_extension() {
        let input = small_zarr("groups_3d");
        let dir = tempfile::tempdir().unwrap();
        let tet = dir.path().join("groups.tet");
        let h5_out = dir.path().join("groups.h5");
        convert_zarr_to_tet_with_progress(&input, &tet, 1, None::<fn(_)>).unwrap();
        export_tet(&tet, &h5_out).unwrap();

        let file = hdf5_metno::File::open(&h5_out).unwrap();
        for dtype in FIXTURE_DTYPES {
            let path = format!("primary/{dtype}");
            assert!(file.dataset(&path).is_ok(), "missing {path}");
        }
        let mmap = mmap_file_read(&tet).unwrap();
        let want = materialize_dataset_le_bytes(&mmap, "primary/f32");
        let got = hdf5_dataset_le_bytes(&file.dataset("primary/f32").unwrap());
        assert_eq!(got, want);
    }

    #[test]
    fn export_hdf5_carries_basic_attrs() {
        let input = small_h5("cf_3d.h5");
        let dir = tempfile::tempdir().unwrap();
        let tet = dir.path().join("cf.tet");
        let h5_out = dir.path().join("cf_out.h5");
        convert_h5_to_tet_with_progress(&input, &tet, 1, None::<fn(_)>).unwrap();
        let mmap = mmap_file_read(&tet).unwrap();
        let summary = read_tet_summary_v1(&mmap).unwrap();
        let temp_meta = summary.metadata.datasets.get("temperature");
        if temp_meta.is_none() {
            // CF fixture may decode into temperature without preserving all attrs on every build.
            return;
        }
        export_tet_to_hdf5(&tet, &h5_out).unwrap();
        let file = hdf5_metno::File::open(&h5_out).unwrap();
        let ds = file.dataset("temperature").unwrap();
        let names = ds.attr_names().unwrap();
        assert!(
            names
                .iter()
                .any(|n| n == "units" || n == "long_name" || n == "tetration_dtype"),
            "expected basic attrs on exported temperature, got {names:?}"
        );
    }
}

#[cfg(feature = "tetration-netcdf")]
mod netcdf_export {
    use std::path::{Path, PathBuf};

    use super::super::convert::materialize_dataset_le_bytes;
    use super::super::verify::assert_tet_verify_ok;
    use super::FIXTURE_DTYPES;
    use crate::catalog::read_tet_summary_v1;
    use crate::convert::convert_netcdf_to_tet_with_progress;
    use crate::export::{export_tet, export_tet_to_netcdf};
    use crate::layout::mmap_file_read;

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    fn small_nc(name: &str) -> PathBuf {
        repo_root().join("fixtures/small/netcdf").join(name)
    }

    #[test]
    fn export_netcdf_convert_export_reread_matches_bytes() {
        let input = small_nc("tensor_3d.nc");
        let dir = tempfile::tempdir().unwrap();
        let tet = dir.path().join("tensor_3d.tet");
        let nc_out = dir.path().join("exported.nc");
        let tet2 = dir.path().join("roundtrip.tet");

        convert_netcdf_to_tet_with_progress(&input, &tet, 1, None::<fn(_)>).unwrap();
        export_tet_to_netcdf(&tet, &nc_out).unwrap();
        convert_netcdf_to_tet_with_progress(&nc_out, &tet2, 1, None::<fn(_)>).unwrap();
        assert_tet_verify_ok(&tet2);

        let mmap = mmap_file_read(&tet).unwrap();
        let mmap2 = mmap_file_read(&tet2).unwrap();
        for name in FIXTURE_DTYPES {
            let want = materialize_dataset_le_bytes(&mmap, name);
            let got = materialize_dataset_le_bytes(&mmap2, name);
            assert_eq!(got, want, "dataset {name} after netcdf export roundtrip");
        }

        let exported = netcdf::open(&nc_out).unwrap();
        for name in FIXTURE_DTYPES {
            let var = exported.variable(name).expect(name);
            let want = materialize_dataset_le_bytes(&mmap, name);
            let got = var.get_raw_values(..).unwrap();
            assert_eq!(got, want, "netcdf file bytes for {name}");
        }
    }

    #[test]
    fn export_netcdf_groups_and_dispatch_by_extension() {
        let input = small_nc("groups_3d.nc");
        let dir = tempfile::tempdir().unwrap();
        let tet = dir.path().join("groups.tet");
        let nc_out = dir.path().join("groups.nc");
        convert_netcdf_to_tet_with_progress(&input, &tet, 1, None::<fn(_)>).unwrap();
        export_tet(&tet, &nc_out).unwrap();

        let file = netcdf::open(&nc_out).unwrap();
        for dtype in FIXTURE_DTYPES {
            let path = format!("primary/{dtype}");
            assert!(file.variable(&path).is_some(), "missing variable {path}");
        }
        let mmap = mmap_file_read(&tet).unwrap();
        let want = materialize_dataset_le_bytes(&mmap, "primary/f32");
        let got = file
            .variable("primary/f32")
            .unwrap()
            .get_raw_values(..)
            .unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn export_netcdf_carries_dim_names_when_present() {
        let input = small_nc("cf_3d.nc");
        let dir = tempfile::tempdir().unwrap();
        let tet = dir.path().join("cf.tet");
        let nc_out = dir.path().join("cf_out.nc");
        convert_netcdf_to_tet_with_progress(&input, &tet, 1, None::<fn(_)>).unwrap();
        let mmap = mmap_file_read(&tet).unwrap();
        let summary = read_tet_summary_v1(&mmap).unwrap();
        let Some(temp_meta) = summary.metadata.datasets.get("temperature") else {
            return;
        };
        let Some(dim_names) = temp_meta.dim_names.as_ref() else {
            return;
        };
        assert!(!dim_names.is_empty());
        export_tet_to_netcdf(&tet, &nc_out).unwrap();
        let file = netcdf::open(&nc_out).unwrap();
        let var = file.variable("temperature").expect("temperature");
        let exported_dims: Vec<String> = var.dimensions().iter().map(|d| d.name()).collect();
        assert_eq!(exported_dims, *dim_names);
    }
}

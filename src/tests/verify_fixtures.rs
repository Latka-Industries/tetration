//! Run [`verify_tet_file`] on writer-built and convert-produced `.tet` files (CI gate).

use std::path::Path;

use super::fixture::{
    le_row_major_2x3_f32_one_to_six, write_multichunk_2x3_f16_tiles,
    write_multichunk_2x3_f64_tiles, write_multichunk_2x3_i16_tiles, write_multichunk_2x3_i32_tiles,
    write_multichunk_2x3_i64_tiles, write_multichunk_2x3_tiles, write_multichunk_2x3_u8_tiles,
    write_multichunk_2x3_u16_tiles, write_multichunk_2x3_u32_tiles, write_multichunk_2x3_u64_tiles,
    write_multichunk_2x3_zstd,
};
use super::verify::assert_tet_verify_ok;
use crate::catalog::{FooterBlobV1, TetMetadataV1, write_footer_blob};
use crate::verify::verify_tet_file;

#[test]
fn verify_writer_multichunk_all_dtypes() {
    let dir = tempfile::tempdir().unwrap();
    let f32 = dir.path().join("f32.tet");
    write_multichunk_2x3_tiles(&f32, "a");
    assert_tet_verify_ok(&f32);

    let f64 = dir.path().join("f64.tet");
    write_multichunk_2x3_f64_tiles(&f64, "a");
    assert_tet_verify_ok(&f64);

    let i32 = dir.path().join("i32.tet");
    write_multichunk_2x3_i32_tiles(&i32, "a");
    assert_tet_verify_ok(&i32);

    let i64 = dir.path().join("i64.tet");
    write_multichunk_2x3_i64_tiles(&i64, "a");
    assert_tet_verify_ok(&i64);

    let u8 = dir.path().join("u8.tet");
    write_multichunk_2x3_u8_tiles(&u8, "a");
    assert_tet_verify_ok(&u8);

    let u16 = dir.path().join("u16.tet");
    write_multichunk_2x3_u16_tiles(&u16, "a");
    assert_tet_verify_ok(&u16);

    let i16 = dir.path().join("i16.tet");
    write_multichunk_2x3_i16_tiles(&i16, "a");
    assert_tet_verify_ok(&i16);

    let u32 = dir.path().join("u32.tet");
    write_multichunk_2x3_u32_tiles(&u32, "a");
    assert_tet_verify_ok(&u32);

    let u64 = dir.path().join("u64.tet");
    write_multichunk_2x3_u64_tiles(&u64, "a");
    assert_tet_verify_ok(&u64);

    let f16 = dir.path().join("f16.tet");
    write_multichunk_2x3_f16_tiles(&f16, "a");
    assert_tet_verify_ok(&f16);

    let zstd = dir.path().join("zstd.tet");
    write_multichunk_2x3_zstd(&zstd, "a", &le_row_major_2x3_f32_one_to_six());
    assert_tet_verify_ok(&zstd);
}

#[test]
fn verify_writer_footer_with_metadata_spill() {
    use std::collections::BTreeMap;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("spill.tet");
    write_multichunk_2x3_tiles(&path, "a");

    let label = "x".repeat(1024);
    let mut coords = BTreeMap::new();
    for ax in 0..8 {
        coords.insert(
            format!("axis_{ax}"),
            crate::catalog::CoordAxisV1 {
                labels: vec![label.clone(); 64],
            },
        );
    }
    let meta = TetMetadataV1 {
        file: None,
        datasets: [(
            "a".to_owned(),
            crate::catalog::DatasetMetadataV1 {
                attrs: BTreeMap::new(),
                dim_names: None,
                coords: Some(coords),
            },
        )]
        .into_iter()
        .collect(),
    };
    write_footer_blob(
        &path,
        &FooterBlobV1 {
            history: Vec::new(),
            metadata: Some(meta),
            metadata_ref: None,
        },
    )
    .unwrap();

    let report = verify_tet_file(&path).unwrap();
    assert!(report.ok, "{:?}", report.findings);
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.check == "footer_metadata" && f.ok)
    );
}

#[cfg(feature = "tetration-netcdf")]
mod convert_gate {
    use super::*;
    use crate::convert::convert_netcdf_to_tet_with_progress;

    fn repo_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    fn small_nc(stem: &str) -> std::path::PathBuf {
        crate::tests::ensure_hdf5_no_file_locking();
        repo_root()
            .join("fixtures/small/netcdf")
            .join(format!("{stem}.nc"))
    }

    #[test]
    fn verify_small_netcdf_convert_outputs() {
        for stem in ["tensor_3d", "tensor_4d", "tensor_5d", "groups_3d", "cf_3d"] {
            let input = small_nc(stem);
            assert!(input.is_file(), "missing {}", input.display());
            let dir = tempfile::tempdir().unwrap();
            let output = dir.path().join(format!("{stem}.tet"));
            convert_netcdf_to_tet_with_progress(&input, &output, 1, None::<fn(_)>).unwrap();
            assert_tet_verify_ok(&output);
        }
    }
}

#[cfg(feature = "tetration-hdf5")]
mod convert_gate_h5 {
    use super::*;
    use crate::convert::convert_h5_to_tet_with_progress;

    fn repo_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    fn small_h5(stem: &str) -> std::path::PathBuf {
        crate::tests::ensure_hdf5_no_file_locking();
        repo_root()
            .join("fixtures/small/h5")
            .join(format!("{stem}.h5"))
    }

    #[test]
    fn verify_small_h5_convert_outputs() {
        for stem in ["tensor_3d", "tensor_4d", "tensor_5d", "cf_3d"] {
            let input = small_h5(stem);
            assert!(input.is_file(), "missing {}", input.display());
            let dir = tempfile::tempdir().unwrap();
            let output = dir.path().join(format!("{stem}.tet"));
            convert_h5_to_tet_with_progress(&input, &output, 1, None::<fn(_)>).unwrap();
            assert_tet_verify_ok(&output);
        }
    }
}

#[cfg(feature = "tetration-hdf5")]
mod convert_gate_zarr {
    use super::*;
    use crate::convert::convert_zarr_to_tet_with_progress;

    fn repo_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    fn small_zarr(stem: &str) -> std::path::PathBuf {
        repo_root().join("fixtures/small/zarr").join(stem)
    }

    #[test]
    fn verify_small_zarr_convert_outputs() {
        for stem in ["tensor_3d", "groups_3d"] {
            let input = small_zarr(stem);
            assert!(input.is_dir(), "missing {}", input.display());
            let dir = tempfile::tempdir().unwrap();
            let output = dir.path().join(format!("{stem}.tet"));
            convert_zarr_to_tet_with_progress(&input, &output, 1, None::<fn(_)>).unwrap();
            assert_tet_verify_ok(&output);
        }
    }
}

//! Convert integration tests against tracked `fixtures/small/` tensors.

use std::path::{Path, PathBuf};

use crate::catalog::{DATASET_DTYPE_TAG_V1, read_tet_summary_v1};
use crate::convert::{ConvertError, ConvertOptions, convert_to_tet_with_options};
#[cfg(feature = "tetration-hdf5")]
use crate::convert::{
    convert_h5_to_tet_with_progress, convert_netcdf_to_tet_with_progress,
    convert_zarr_to_tet_with_progress,
};
use crate::layout::mmap_file_read;
use crate::query::{
    materialize_read_plan_f32_le, materialize_read_plan_f64_le, materialize_read_plan_i32_le,
    materialize_read_plan_i64_le, parse_query_json, plan_query_with_tet_mmap, validate_query,
};

use super::verify::assert_tet_verify_ok;

const FIXTURE_DTYPES: [&str; 4] = ["f32", "f64", "i32", "i64"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

#[cfg(feature = "tetration-hdf5")]
fn small_h5(name: &str) -> PathBuf {
    repo_root().join("fixtures/small/h5").join(name)
}

fn small_nc(name: &str) -> PathBuf {
    repo_root().join("fixtures/small/netcdf").join(name)
}

pub(crate) fn small_zarr(name: &str) -> PathBuf {
    repo_root().join("fixtures/small/zarr").join(name)
}

pub(crate) fn materialize_dataset_le_bytes(mmap: &[u8], dataset: &str) -> Vec<u8> {
    let doc = parse_query_json(&format!(r#"{{"dataset":"{dataset}"}}"#)).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, mmap, None).unwrap();
    let cat = plan.catalog.as_ref().unwrap();
    let dtype = cat.dtype.expect("matched dataset dtype");
    let rp = plan.read_plan.as_ref().unwrap();
    let tags = DATASET_DTYPE_TAG_V1;
    if tags.is_f32(dtype) {
        let (vals, _, _) = materialize_read_plan_f32_le(mmap, rp, None).unwrap();
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if tags.is_f64(dtype) {
        let (vals, _, _) = materialize_read_plan_f64_le(mmap, rp, None).unwrap();
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if tags.is_i32(dtype) {
        let (vals, _, _) = materialize_read_plan_i32_le(mmap, rp, None).unwrap();
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if tags.is_i64(dtype) {
        let (vals, _, _) = materialize_read_plan_i64_le(mmap, rp, None).unwrap();
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else {
        panic!("unexpected dtype {dtype}");
    }
}

#[cfg(feature = "tetration-hdf5")]
fn hdf5_dataset_le_bytes(ds: &hdf5_metno::Dataset, name: &str) -> Vec<u8> {
    hdf5_dataset_raw_le_bytes(ds).unwrap_or_else(|| panic!("unexpected dataset {name}"))
}

#[cfg(feature = "tetration-hdf5")]
fn hdf5_dataset_raw_le_bytes(ds: &hdf5_metno::Dataset) -> Option<Vec<u8>> {
    use hdf5_metno::types::{FloatSize, IntSize, TypeDescriptor};

    let td = ds.dtype().ok()?;
    let desc = td.to_descriptor().ok()?;
    Some(match desc {
        TypeDescriptor::Float(FloatSize::U4) => ds
            .read_raw::<f32>()
            .ok()?
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
        TypeDescriptor::Float(FloatSize::U8) => ds
            .read_raw::<f64>()
            .ok()?
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect(),
        TypeDescriptor::Integer(IntSize::U4) => ds
            .read_raw::<i32>()
            .ok()?
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect(),
        TypeDescriptor::Integer(IntSize::U8) => ds
            .read_raw::<i64>()
            .ok()?
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect(),
        _ => return None,
    })
}

#[cfg(feature = "tetration-hdf5")]
fn hdf5_dataset_path_le_bytes(file: &hdf5_metno::File, path: &str) -> Vec<u8> {
    let ds = file.dataset(path).unwrap();
    hdf5_dataset_raw_le_bytes(&ds).unwrap_or_else(|| panic!("unsupported dataset `{path}`"))
}

fn netcdf_dataset_path_le_bytes(file: &netcdf::File, path: &str) -> Vec<u8> {
    let var = file.variable(path).expect("source variable");
    var.get_raw_values(..).unwrap()
}

fn read_dataset_payload_le_bytes(mmap: &[u8], dataset: &str) -> Vec<u8> {
    let doc = parse_query_json(&format!(r#"{{"dataset":"{dataset}"}}"#)).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let mut out = Vec::new();
    for chunk in &rp.chunks {
        let start = usize::try_from(chunk.payload_offset).unwrap();
        let end = start + usize::try_from(chunk.raw_byte_len).unwrap();
        out.extend_from_slice(&mmap[start..end]);
    }
    out
}

fn assert_f32_payload_eq(got: &[u8], want: &[u8], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: byte length mismatch");
    for (i, (g, w)) in got.chunks_exact(4).zip(want.chunks_exact(4)).enumerate() {
        let got_v = f32::from_le_bytes(g.try_into().unwrap());
        let want_v = f32::from_le_bytes(w.try_into().unwrap());
        let equal = if got_v.is_nan() && want_v.is_nan() {
            true
        } else if got_v.is_nan() || want_v.is_nan() {
            false
        } else {
            got_v.to_bits() == want_v.to_bits()
                || (got_v - want_v).abs() <= 1.0e-5 * got_v.abs().max(want_v.abs()).max(1.0)
        };
        assert!(equal, "{label}[{i}] got={got_v} want={want_v}");
    }
}

fn cf_temperature_physical_le_bytes(stored: &[u8]) -> Vec<u8> {
    let scale = 0.01_f64;
    let offset = 273.15_f64;
    let fill = -9999.0_f32;
    stored
        .chunks_exact(4)
        .map(|chunk| {
            let v = f32::from_le_bytes(chunk.try_into().unwrap());
            let decoded = if (v - fill).abs() <= 1.0e-3 {
                f32::NAN
            } else {
                (f64::from(v) * scale + offset) as f32
            };
            decoded.to_le_bytes()
        })
        .flatten()
        .collect()
}

#[cfg(feature = "tetration-hdf5")]
fn assert_small_fixture_h5(stem: &str) {
    assert_small_fixture_h5_with_jobs(stem, 1);
}

#[cfg(feature = "tetration-hdf5")]
fn assert_small_fixture_h5_with_jobs(stem: &str, parallel_jobs: usize) {
    let input = small_h5(&format!("{stem}.h5"));
    assert!(input.is_file(), "missing fixture {}", input.display());
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join(format!("{stem}.tet"));
    convert_h5_to_tet_with_progress(&input, &output, parallel_jobs, None::<fn(_)>).unwrap();
    let mmap = mmap_file_read(&output).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.datasets.len(), 4);
    assert_eq!(summary.history.len(), 1);
    assert_eq!(summary.history[0].op, "convert");
    assert_eq!(summary.history[0].source, "h5");

    let src = hdf5_metno::File::open(&input).unwrap();
    for name in FIXTURE_DTYPES {
        let ds = src.dataset(name).unwrap();
        let want = hdf5_dataset_le_bytes(&ds, name);
        let got = materialize_dataset_le_bytes(&mmap, name);
        assert_eq!(got, want, "dataset {name} bytes differ for {stem}");
    }
    assert_tet_verify_ok(&output);
}

fn assert_small_fixture_netcdf(stem: &str) {
    assert_small_fixture_netcdf_with_jobs(stem, 1);
}

fn assert_small_fixture_netcdf_with_jobs(stem: &str, parallel_jobs: usize) {
    let input = small_nc(&format!("{stem}.nc"));
    assert!(input.is_file(), "missing fixture {}", input.display());
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join(format!("{stem}.tet"));
    convert_netcdf_to_tet_with_progress(&input, &output, parallel_jobs, None::<fn(_)>).unwrap();
    let mmap = mmap_file_read(&output).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.datasets.len(), 4);
    assert_eq!(summary.history.len(), 1);
    assert_eq!(summary.history[0].op, "convert");
    assert_eq!(summary.history[0].source, "nc");

    let src = netcdf::open(&input).unwrap();
    for name in FIXTURE_DTYPES {
        let var = src.variable(name).expect("source variable");
        let want = var.get_raw_values(..).unwrap();
        let got = materialize_dataset_le_bytes(&mmap, name);
        assert_eq!(got, want, "dataset {name} bytes differ for {stem}");
    }
    assert_tet_verify_ok(&output);
}

fn assert_small_fixture_zarr(stem: &str) {
    let input = small_zarr(stem);
    assert!(input.is_dir(), "missing fixture {}", input.display());
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join(format!("{stem}.tet"));
    convert_zarr_to_tet_with_progress(&input, &output, 1, None::<fn(_)>).unwrap();
    let mmap = mmap_file_read(&output).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.datasets.len(), 4);
    assert_eq!(summary.history.len(), 1);
    assert_eq!(summary.history[0].op, "convert");
    assert_eq!(summary.history[0].source, "zarr");

    for name in FIXTURE_DTYPES {
        let want_path = input.join(name);
        let want = zarr_array_le_bytes(&want_path);
        let got = materialize_dataset_le_bytes(&mmap, name);
        assert_eq!(got, want, "dataset {name} bytes differ for zarr {stem}");
    }
    assert_tet_verify_ok(&output);
}

pub(crate) fn zarr_array_le_bytes(array_dir: &Path) -> Vec<u8> {
    let chunk_path = array_dir.join("c/0/0/0");
    let on_disk = std::fs::read(&chunk_path).unwrap();
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(array_dir.join("zarr.json")).unwrap())
            .unwrap();
    let zstd = meta["codecs"].as_array().is_some_and(|codecs| {
        codecs.iter().any(|codec| {
            codec
                .get("name")
                .and_then(|v| v.as_str())
                .is_some_and(|name| name == "zstd")
        })
    });
    let raw = if zstd {
        zstd::decode_all(on_disk.as_slice()).unwrap()
    } else {
        on_disk
    };
    let data_type = meta["data_type"].as_str().unwrap_or("");
    match data_type {
        "float32" => raw,
        "float64" => raw,
        "int32" => raw,
        "int64" => raw,
        other => panic!("unexpected zarr dtype {other}"),
    }
}

#[cfg(feature = "tetration-hdf5")]
#[test]
fn convert_small_h5_tensor_3d_matches_source_bytes() {
    assert_small_fixture_h5("tensor_3d");
}

#[cfg(feature = "tetration-hdf5")]
#[test]
fn convert_small_h5_tensor_4d_matches_source_bytes() {
    assert_small_fixture_h5("tensor_4d");
}

#[cfg(feature = "tetration-hdf5")]
#[test]
fn convert_small_h5_tensor_5d_matches_source_bytes() {
    assert_small_fixture_h5("tensor_5d");
}

#[cfg(feature = "tetration-hdf5")]
#[test]
fn convert_small_h5_groups_3d_matches_nested_source_bytes() {
    let input = small_h5("groups_3d.h5");
    assert!(input.is_file(), "missing fixture {}", input.display());
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("groups_3d.tet");
    convert_h5_to_tet_with_progress(&input, &output, 1, None::<fn(_)>).unwrap();
    let mmap = mmap_file_read(&output).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.datasets.len(), 5);

    let src = hdf5_metno::File::open(&input).unwrap();
    for name in FIXTURE_DTYPES {
        let path = format!("primary/{name}");
        let want = hdf5_dataset_path_le_bytes(&src, &path);
        let got = materialize_dataset_le_bytes(&mmap, &path);
        assert_eq!(got, want, "dataset {path} bytes differ");
    }
    let want_scale = hdf5_dataset_path_le_bytes(&src, "aux/scale");
    let got_scale = materialize_dataset_le_bytes(&mmap, "aux/scale");
    assert_eq!(got_scale, want_scale);
    assert_tet_verify_ok(&output);
}

#[cfg(feature = "tetration-hdf5")]
#[test]
fn convert_small_h5_cf_3d_decodes_temperature() {
    let input = small_h5("cf_3d.h5");
    assert!(input.is_file(), "missing fixture {}", input.display());
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("cf_3d.tet");
    convert_h5_to_tet_with_progress(&input, &output, 1, None::<fn(_)>).unwrap();
    let mmap = mmap_file_read(&output).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert!(summary.datasets.len() >= 5);

    let src = hdf5_metno::File::open(&input).unwrap();
    for name in FIXTURE_DTYPES {
        let want = hdf5_dataset_le_bytes(&src.dataset(name).unwrap(), name);
        let got = materialize_dataset_le_bytes(&mmap, name);
        assert_eq!(got, want, "dataset {name} bytes differ");
    }
    let stored = src
        .dataset("temperature")
        .unwrap()
        .read_raw::<f32>()
        .unwrap();
    let want = cf_temperature_physical_le_bytes(
        &stored
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let got = read_dataset_payload_le_bytes(&mmap, "temperature");
    assert_f32_payload_eq(&got, &want, "temperature CF decode");

    let temp_meta = summary.metadata.datasets.get("temperature").unwrap();
    assert_eq!(
        temp_meta.attrs.get("long_name").map(String::as_str),
        Some("sea surface temperature")
    );
    assert_eq!(temp_meta.attrs.get("units").map(String::as_str), Some("K"));
    assert!(
        temp_meta.attrs.contains_key("scale_factor"),
        "expected scale_factor in imported attrs: {:?}",
        temp_meta.attrs
    );

    let time_meta = summary.metadata.datasets.get("coordinates/time").unwrap();
    assert_eq!(
        time_meta.attrs.get("units").map(String::as_str),
        Some("days since 2020-01-01")
    );

    let coords = temp_meta.coords.as_ref().expect("temperature coords");
    assert_eq!(coords.len(), 3);
    assert_eq!(coords.get("time").map(|c| c.labels.len()), Some(32));
    assert_eq!(coords["time"].labels.first().map(String::as_str), Some("0"));
    assert_eq!(coords.get("lat").map(|c| c.labels.len()), Some(32));
    assert_eq!(coords.get("lon").map(|c| c.labels.len()), Some(32));
    assert_tet_verify_ok(&output);
}

#[test]
fn detect_convert_format_from_extension() {
    use crate::convert::{
        ConvertInputFormat, Hdf5ConvertInput, NetcdfConvertInput, ZarrConvertInput,
        detect_convert_format,
    };

    for ext in Hdf5ConvertInput::EXTENSIONS {
        let path = format!("tensor.{ext}");
        assert_eq!(
            detect_convert_format(Path::new(&path)).unwrap(),
            ConvertInputFormat::H5,
            "expected HDF5 for .{ext}"
        );
    }
    assert_eq!(
        detect_convert_format(Path::new("a.HDF5")).unwrap(),
        ConvertInputFormat::H5
    );

    for ext in NetcdfConvertInput::EXTENSIONS {
        let path = format!("tensor.{ext}");
        assert_eq!(
            detect_convert_format(Path::new(&path)).unwrap(),
            ConvertInputFormat::Netcdf,
            "expected NetCDF for .{ext}"
        );
    }
    assert_eq!(
        detect_convert_format(Path::new("a.NETCDF")).unwrap(),
        ConvertInputFormat::Netcdf
    );
    assert_eq!(
        detect_convert_format(Path::new("model.nc.gz")).unwrap(),
        ConvertInputFormat::Netcdf
    );

    for ext in ZarrConvertInput::EXTENSIONS {
        let path = format!("tensor.{ext}");
        assert_eq!(
            detect_convert_format(Path::new(&path)).unwrap(),
            ConvertInputFormat::Zarr,
            "expected Zarr for .{ext}"
        );
    }

    assert_eq!(
        detect_convert_format(Path::new("data.csv")).unwrap(),
        ConvertInputFormat::Csv
    );
    assert_eq!(
        detect_convert_format(Path::new("data.TSV")).unwrap(),
        ConvertInputFormat::Csv
    );
    assert!(detect_convert_format(Path::new("data.txt")).is_err());
    assert_eq!(
        detect_convert_format(Path::new("sample.npy")).unwrap(),
        ConvertInputFormat::Npy
    );
    assert_eq!(
        detect_convert_format(Path::new("sample.NPZ")).unwrap(),
        ConvertInputFormat::Npz
    );
}

#[test]
fn detect_convert_format_sniffs_file_signature() {
    use crate::convert::{
        ConvertInputFormat, Hdf5ConvertInput, NetcdfConvertInput, detect_convert_format,
    };

    let dir = tempfile::tempdir().unwrap();
    let h5_path = dir.path().join("payload.bin");
    std::fs::write(&h5_path, Hdf5ConvertInput::MAGIC).unwrap();
    assert_eq!(
        detect_convert_format(&h5_path).unwrap(),
        ConvertInputFormat::H5
    );

    let nc_path = dir.path().join("payload.dat");
    std::fs::write(&nc_path, NetcdfConvertInput::NETCDF3_V1).unwrap();
    assert_eq!(
        detect_convert_format(&nc_path).unwrap(),
        ConvertInputFormat::Netcdf
    );
}

#[test]
fn detect_convert_format_sniffs_zarr_directory() {
    use crate::convert::{ConvertInputFormat, detect_convert_format};

    let path = small_zarr("tensor_3d");
    if path.is_dir() {
        assert_eq!(
            detect_convert_format(&path).unwrap(),
            ConvertInputFormat::Zarr
        );
    }
}

#[cfg(feature = "tetration-hdf5")]
#[test]
fn convert_small_h5_parallel_jobs_matches_source_bytes() {
    assert_small_fixture_h5_with_jobs("tensor_3d", 4);
}

#[test]
fn convert_small_netcdf_parallel_jobs_matches_source_bytes() {
    assert_small_fixture_netcdf_with_jobs("tensor_3d", 4);
}

#[test]
fn convert_small_netcdf_tensor_3d_matches_source_bytes() {
    assert_small_fixture_netcdf("tensor_3d");
}

#[test]
fn convert_small_netcdf_tensor_4d_matches_source_bytes() {
    assert_small_fixture_netcdf("tensor_4d");
}

#[test]
fn convert_small_netcdf_tensor_5d_matches_source_bytes() {
    assert_small_fixture_netcdf("tensor_5d");
}

#[test]
fn convert_small_netcdf_groups_3d_matches_nested_source_bytes() {
    let input = small_nc("groups_3d.nc");
    assert!(input.is_file(), "missing fixture {}", input.display());
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("groups_3d.tet");
    convert_netcdf_to_tet_with_progress(&input, &output, 1, None::<fn(_)>).unwrap();
    let mmap = mmap_file_read(&output).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.datasets.len(), 4);

    let src = netcdf::open(&input).unwrap();
    for name in FIXTURE_DTYPES {
        let path = format!("primary/{name}");
        let want = netcdf_dataset_path_le_bytes(&src, &path);
        let got = materialize_dataset_le_bytes(&mmap, &path);
        assert_eq!(got, want, "dataset {path} bytes differ");
    }
    assert_tet_verify_ok(&output);
}

#[test]
fn convert_small_netcdf_cf_3d_decodes_temperature() {
    let input = small_nc("cf_3d.nc");
    assert!(input.is_file(), "missing fixture {}", input.display());
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("cf_3d.tet");
    convert_netcdf_to_tet_with_progress(&input, &output, 1, None::<fn(_)>).unwrap();
    let mmap = mmap_file_read(&output).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();

    let src = netcdf::open(&input).unwrap();
    for name in FIXTURE_DTYPES {
        let want = netcdf_dataset_path_le_bytes(&src, name);
        let got = materialize_dataset_le_bytes(&mmap, name);
        assert_eq!(got, want, "dataset {name} bytes differ");
    }
    let stored = src
        .variable("temperature")
        .unwrap()
        .get_raw_values(..)
        .unwrap();
    let want = cf_temperature_physical_le_bytes(&stored);
    let got = read_dataset_payload_le_bytes(&mmap, "temperature");
    assert_f32_payload_eq(&got, &want, "temperature CF decode");

    let temp_meta = summary.metadata.datasets.get("temperature").unwrap();
    assert_eq!(
        temp_meta.attrs.get("long_name").map(String::as_str),
        Some("sea surface temperature")
    );
    let dim_names = temp_meta.dim_names.as_ref().expect("nc dim_names");
    assert_eq!(dim_names.len(), 3);
    assert_tet_verify_ok(&output);
}

#[test]
fn convert_small_zarr_tensor_3d_matches_source_bytes() {
    assert_small_fixture_zarr("tensor_3d");
}

#[test]
fn convert_small_zarr_imports_array_attrs() {
    let input = small_zarr("tensor_3d");
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("tensor_3d_attrs.tet");
    convert_zarr_to_tet_with_progress(&input, &output, 1, None::<fn(_)>).unwrap();
    let summary = read_tet_summary_v1(&mmap_file_read(&output).unwrap()).unwrap();
    let f32_meta = summary.metadata.datasets.get("f32").unwrap();
    assert_eq!(
        f32_meta.attrs.get("tetration_dtype").map(String::as_str),
        Some("f32")
    );

    let input = small_zarr("groups_3d");
    let output = dir.path().join("groups_3d_attrs.tet");
    convert_zarr_to_tet_with_progress(&input, &output, 1, None::<fn(_)>).unwrap();
    let summary = read_tet_summary_v1(&mmap_file_read(&output).unwrap()).unwrap();
    let nested = summary.metadata.datasets.get("primary/f32").unwrap();
    assert_eq!(
        nested.attrs.get("tetration_dtype").map(String::as_str),
        Some("f32")
    );
    assert_tet_verify_ok(&dir.path().join("tensor_3d_attrs.tet"));
    assert_tet_verify_ok(&output);
}

#[test]
fn convert_small_zarr_groups_3d_matches_nested_source_bytes() {
    let input = small_zarr("groups_3d");
    assert!(input.is_dir(), "missing fixture {}", input.display());
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("groups_3d.tet");
    convert_zarr_to_tet_with_progress(&input, &output, 1, None::<fn(_)>).unwrap();
    let mmap = mmap_file_read(&output).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.datasets.len(), 4);

    for name in FIXTURE_DTYPES {
        let path = format!("primary/{name}");
        let want = zarr_array_le_bytes(&input.join("primary").join(name));
        let got = materialize_dataset_le_bytes(&mmap, &path);
        assert_eq!(got, want, "dataset {path} bytes differ");
    }
    assert_tet_verify_ok(&output);
}

fn write_mixed_zarr(root: &Path) {
    let data = root.join("data");
    std::fs::create_dir_all(data.join("c")).unwrap();
    std::fs::create_dir_all(root.join("label")).unwrap();
    std::fs::write(
        root.join("zarr.json"),
        r#"{"zarr_format":3,"node_type":"group"}"#,
    )
    .unwrap();
    std::fs::write(
        data.join("zarr.json"),
        r#"{"zarr_format":3,"node_type":"array","shape":[2],"data_type":"float32","chunk_grid":{"name":"regular","configuration":{"chunk_shape":[2]}},"codecs":[{"name":"bytes","configuration":{"endian":"little"}}]}"#,
    )
    .unwrap();
    let mut bytes = Vec::new();
    bytes.extend(1.0f32.to_le_bytes());
    bytes.extend(2.0f32.to_le_bytes());
    std::fs::write(data.join("c/0"), bytes).unwrap();
    std::fs::write(
        root.join("label/zarr.json"),
        r#"{"zarr_format":3,"node_type":"array","shape":[2],"data_type":"string","chunk_grid":{"name":"regular","configuration":{"chunk_shape":[2]}},"codecs":[{"name":"bytes","configuration":{"endian":"little"}}]}"#,
    )
    .unwrap();
}

#[test]
fn convert_reports_skipped_dtype_and_strict_refuses_write() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("mixed.zarr");
    write_mixed_zarr(&store);
    let output = dir.path().join("mixed.tet");

    let report = convert_to_tet_with_options(
        &store,
        &output,
        ConvertOptions {
            parallel_jobs: 1,
            strict: false,
        },
        None::<fn(_)>,
    )
    .unwrap();
    assert_eq!(report.dataset_names, vec!["data".to_owned()]);
    assert_eq!(report.skipped.len(), 1);
    assert_eq!(report.skipped[0].path, "label");
    assert!(report.skipped[0].reason.contains("string"));
    assert!(output.is_file());

    let strict_out = dir.path().join("strict.tet");
    let err = convert_to_tet_with_options(
        &store,
        &strict_out,
        ConvertOptions {
            parallel_jobs: 1,
            strict: true,
        },
        None::<fn(_)>,
    )
    .unwrap_err();
    match err {
        ConvertError::StrictSkips { count, detail, .. } => {
            assert_eq!(count, 1);
            assert!(detail.contains("label"));
        }
        other => panic!("expected StrictSkips, got {other}"),
    }
    assert!(!strict_out.exists());
}

fn npy_bytes(descr: &str, shape: &[u64], fortran: bool, data: &[u8]) -> Vec<u8> {
    let shape_text = if shape.is_empty() {
        "()".to_owned()
    } else if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(|dim| dim.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let order = if fortran { "True" } else { "False" };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': {order}, 'shape': {shape_text}, }}");
    let mut header = dict.into_bytes();
    header.push(b'\n');
    let prefix = 10usize;
    let pad = (64 - ((prefix + header.len()) % 64)) % 64;
    if pad > 0 {
        header.pop();
        header.extend(std::iter::repeat_n(b' ', pad));
        header.push(b'\n');
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"\x93NUMPY");
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&u16::try_from(header.len()).unwrap().to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(data);
    out
}

fn write_npz(path: &Path, members: &[(&str, &[u8])], stored: bool) {
    use std::io::Write;
    use zip::write::SimpleFileOptions;
    use zip::{CompressionMethod, ZipWriter};

    let file = std::fs::File::create(path).unwrap();
    let mut zip = ZipWriter::new(file);
    let method = if stored {
        CompressionMethod::Stored
    } else {
        CompressionMethod::Deflated
    };
    let options = SimpleFileOptions::default().compression_method(method);
    for (name, bytes) in members {
        zip.start_file(*name, options).unwrap();
        zip.write_all(bytes).unwrap();
    }
    zip.finish().unwrap();
}

#[test]
fn convert_npy_roundtrips_bytes_across_tiles() {
    use crate::catalog::read_tet_summary_v1;
    use crate::convert::convert_numpy_with_chunk_bytes;
    use crate::layout::mmap_file_read;

    let dir = tempfile::tempdir().unwrap();
    let mut data = Vec::new();
    for value in 0..32u8 {
        data.extend_from_slice(&(f32::from(value) * 0.5).to_le_bytes());
    }
    let src = dir.path().join("sample.npy");
    std::fs::write(&src, npy_bytes("<f4", &[8, 4], false, &data)).unwrap();
    let output = dir.path().join("sample.tet");

    let report = convert_numpy_with_chunk_bytes(
        &src,
        &output,
        2,
        false,
        crate::convert::ConvertInputFormat::Npy,
        32,
        None::<fn(_)>,
    )
    .unwrap();
    assert_eq!(report.dataset_names, vec!["data".to_owned()]);
    assert!(report.skipped.is_empty());
    assert_eq!(report.history[0].source, "npy");

    let mmap = mmap_file_read(&output).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.datasets[0].chunk_shape, vec![2, 4]);
    assert_eq!(summary.datasets[0].shape, vec![8, 4]);
    let got = materialize_dataset_le_bytes(&mmap, "data");
    assert_eq!(got, data);
    assert_tet_verify_ok(&output);
}

#[test]
fn convert_npz_single_array_roundtrips_stored_and_deflated() {
    use crate::layout::mmap_file_read;

    let dir = tempfile::tempdir().unwrap();
    let mut data = Vec::new();
    for value in [1i32, 2, 3, 4, 5, 6] {
        data.extend_from_slice(&value.to_le_bytes());
    }
    let npy = npy_bytes("<i4", &[2, 3], false, &data);
    for (name, stored) in [("stored.npz", true), ("deflated.npz", false)] {
        let src = dir.path().join(name);
        write_npz(&src, &[("values.npy", npy.as_slice())], stored);
        let output = dir.path().join(format!("{name}.tet"));
        let report = convert_to_tet_with_options(
            &src,
            &output,
            ConvertOptions {
                parallel_jobs: 1,
                strict: false,
            },
            None::<fn(_)>,
        )
        .unwrap();
        assert_eq!(report.dataset_names, vec!["values".to_owned()]);
        assert_eq!(report.history[0].source, "npz");
        let mmap = mmap_file_read(&output).unwrap();
        assert_eq!(materialize_dataset_le_bytes(&mmap, "values"), data);
        assert_tet_verify_ok(&output);
    }
}

#[test]
fn convert_npz_multi_array_roundtrips_each_member() {
    use crate::layout::mmap_file_read;

    let dir = tempfile::tempdir().unwrap();
    let a = {
        let mut bytes = Vec::new();
        for value in [1.0f32, 2.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        npy_bytes("<f4", &[2], false, &bytes)
    };
    let b = {
        let mut bytes = Vec::new();
        for value in [3i32, 4, 5] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        npy_bytes("<i4", &[3], false, &bytes)
    };
    let src = dir.path().join("multi.npz");
    write_npz(
        &src,
        &[("b.npy", b.as_slice()), ("a.npy", a.as_slice())],
        true,
    );
    let output = dir.path().join("out.tet");
    let report = convert_to_tet_with_options(
        &src,
        &output,
        ConvertOptions {
            parallel_jobs: 2,
            strict: false,
        },
        None::<fn(_)>,
    )
    .unwrap();
    assert_eq!(report.dataset_names, vec!["a".to_owned(), "b".to_owned()]);
    let mmap = mmap_file_read(&output).unwrap();
    assert_eq!(
        materialize_dataset_le_bytes(&mmap, "a"),
        1.0f32
            .to_le_bytes()
            .into_iter()
            .chain(2.0f32.to_le_bytes())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        materialize_dataset_le_bytes(&mmap, "b"),
        [3i32, 4, 5]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_tet_verify_ok(&output);
}

#[test]
fn convert_npz_duplicate_names_and_fortran_member_write_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let data = 1.0f32.to_le_bytes().repeat(2);
    let ok = npy_bytes("<f4", &[2], false, &data);
    let fortran = npy_bytes("<f4", &[2], true, &data);

    let clash = dir.path().join("clash.npz");
    write_npz(
        &clash,
        &[("grp/a.npy", ok.as_slice()), ("grp\\a.npy", ok.as_slice())],
        true,
    );
    let err = convert_to_tet_with_options(
        &clash,
        &dir.path().join("clash.tet"),
        ConvertOptions::default(),
        None::<fn(_)>,
    )
    .unwrap_err();
    let ConvertError::Numpy(detail) = err else {
        panic!("expected Numpy error");
    };
    assert!(
        detail.contains("more than one array named `grp/a`"),
        "{detail}"
    );
    assert!(!dir.path().join("clash.tet").exists());

    let bad = dir.path().join("fortran.npz");
    write_npz(
        &bad,
        &[("ok.npy", ok.as_slice()), ("bad.npy", fortran.as_slice())],
        true,
    );
    let err = convert_to_tet_with_options(
        &bad,
        &dir.path().join("fortran.tet"),
        ConvertOptions::default(),
        None::<fn(_)>,
    )
    .unwrap_err();
    let ConvertError::Numpy(detail) = err else {
        panic!("expected Numpy error");
    };
    assert!(detail.contains("Fortran-order"), "{detail}");
    assert!(!dir.path().join("fortran.tet").exists());
}

#[test]
fn convert_npy_rejects_fortran_and_object_and_skips_scalar() {
    let dir = tempfile::tempdir().unwrap();
    let data = 1.0f32.to_le_bytes().repeat(4);

    let fortran = dir.path().join("fortran.npy");
    std::fs::write(&fortran, npy_bytes("<f4", &[4], true, &data)).unwrap();
    let err = convert_to_tet_with_options(
        &fortran,
        &dir.path().join("fortran.tet"),
        ConvertOptions::default(),
        None::<fn(_)>,
    )
    .unwrap_err();
    let ConvertError::Numpy(detail) = err else {
        panic!("expected Numpy error");
    };
    assert!(detail.contains("Fortran-order"), "{detail}");

    let object = dir.path().join("object.npy");
    std::fs::write(&object, npy_bytes("|O", &[1], false, &[0])).unwrap();
    let err = convert_to_tet_with_options(
        &object,
        &dir.path().join("object.tet"),
        ConvertOptions::default(),
        None::<fn(_)>,
    )
    .unwrap_err();
    let ConvertError::Numpy(detail) = err else {
        panic!("expected Numpy error");
    };
    assert!(detail.contains("pickled"), "{detail}");

    let scalar = dir.path().join("scalar.npy");
    std::fs::write(&scalar, npy_bytes("<f4", &[], false, &1.0f32.to_le_bytes())).unwrap();
    let err = convert_to_tet_with_options(
        &scalar,
        &dir.path().join("scalar.tet"),
        ConvertOptions {
            parallel_jobs: 1,
            strict: true,
        },
        None::<fn(_)>,
    )
    .unwrap_err();
    match err {
        ConvertError::StrictSkips { count, .. } => assert_eq!(count, 1),
        other => panic!("expected StrictSkips, got {other}"),
    }
    assert!(!dir.path().join("scalar.tet").exists());
}

#[test]
fn convert_csv_numeric_columns_roundtrip_and_skip_text() {
    use crate::catalog::read_tet_summary_v1;
    use crate::convert::convert_csv_with_chunk_bytes;
    use crate::layout::mmap_file_read;

    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("table.csv");
    std::fs::write(&src, "temp,note,n\n1.5,\"a,b\",2\n2.5,9,3\n").unwrap();
    let output = dir.path().join("table.tet");
    let report = convert_csv_with_chunk_bytes(&src, &output, 1, false, 16, None::<fn(_)>).unwrap();
    assert_eq!(
        report.dataset_names,
        vec!["temp".to_owned(), "n".to_owned()]
    );
    assert_eq!(report.skipped.len(), 1);
    assert_eq!(report.skipped[0].path, "note");
    assert!(report.skipped[0].reason.contains("non-numeric"));
    assert_eq!(report.history[0].source, "csv");

    let mmap = mmap_file_read(&output).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.datasets[0].chunk_shape, vec![2]);
    assert_eq!(
        materialize_dataset_le_bytes(&mmap, "temp"),
        [1.5f64, 2.5]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        materialize_dataset_le_bytes(&mmap, "n"),
        [2i64, 3]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_tet_verify_ok(&output);

    let strict_out = dir.path().join("strict.tet");
    let err = convert_to_tet_with_options(
        &src,
        &strict_out,
        ConvertOptions {
            parallel_jobs: 1,
            strict: true,
        },
        None::<fn(_)>,
    )
    .unwrap_err();
    assert!(matches!(err, ConvertError::StrictSkips { count: 1, .. }));
    assert!(!strict_out.exists());
}

#[test]
fn convert_csv_headerless_tsv_blank_ragged_and_duplicate() {
    let dir = tempfile::tempdir().unwrap();

    let bare = dir.path().join("bare.csv");
    std::fs::write(&bare, "1,2\n3,4\n").unwrap();
    let bare_out = dir.path().join("bare.tet");
    let report =
        convert_to_tet_with_options(&bare, &bare_out, ConvertOptions::default(), None::<fn(_)>)
            .unwrap();
    assert_eq!(
        report.dataset_names,
        vec!["col0".to_owned(), "col1".to_owned()]
    );

    let tsv = dir.path().join("table.tsv");
    std::fs::write(&tsv, "a\tb\n1\tx\n2\ty\n").unwrap();
    let tsv_out = dir.path().join("table.tet");
    let report =
        convert_to_tet_with_options(&tsv, &tsv_out, ConvertOptions::default(), None::<fn(_)>)
            .unwrap();
    assert_eq!(report.dataset_names, vec!["a".to_owned(), "b".to_owned()]);
    let mmap = crate::layout::mmap_file_read(&tsv_out).unwrap();
    assert_eq!(
        read_dataset_payload_le_bytes(&mmap, "b"),
        [0i64, 1]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>()
    );

    let blanks = dir.path().join("blanks.csv");
    std::fs::write(&blanks, "n\n1\n \n3\n").unwrap();
    let blanks_out = dir.path().join("blanks.tet");
    convert_to_tet_with_options(
        &blanks,
        &blanks_out,
        ConvertOptions::default(),
        None::<fn(_)>,
    )
    .unwrap();
    let mmap = crate::layout::mmap_file_read(&blanks_out).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.datasets[0].dtype, DATASET_DTYPE_TAG_V1.f64);
    let got = read_dataset_payload_le_bytes(&mmap, "n");
    let mut expected = Vec::new();
    expected.extend_from_slice(&1.0f64.to_le_bytes());
    expected.extend_from_slice(&f64::NAN.to_le_bytes());
    expected.extend_from_slice(&3.0f64.to_le_bytes());
    assert_eq!(got, expected);

    let ragged = dir.path().join("ragged.csv");
    std::fs::write(&ragged, "a,b\n1,2\n3\n").unwrap();
    let err = convert_to_tet_with_options(
        &ragged,
        &dir.path().join("ragged.tet"),
        ConvertOptions::default(),
        None::<fn(_)>,
    )
    .unwrap_err();
    let ConvertError::Csv(detail) = err else {
        panic!("expected Csv error");
    };
    assert!(detail.contains("expected 2"), "{detail}");

    let quoted = dir.path().join("quoted.csv");
    std::fs::write(&quoted, "temp,note\n1.25,\"line one\nline two\"\n").unwrap();
    let quoted_out = dir.path().join("quoted.tet");
    let report = convert_to_tet_with_options(
        &quoted,
        &quoted_out,
        ConvertOptions::default(),
        None::<fn(_)>,
    )
    .unwrap();
    assert_eq!(
        report.dataset_names,
        vec!["temp".to_owned(), "note".to_owned()]
    );
    let mmap = crate::layout::mmap_file_read(&quoted_out).unwrap();
    assert_eq!(
        read_dataset_payload_le_bytes(&mmap, "temp"),
        1.25f64.to_le_bytes()
    );
    assert_eq!(
        read_dataset_payload_le_bytes(&mmap, "note"),
        0i64.to_le_bytes()
    );
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let note = &summary.metadata.datasets["note"].attrs["tetration_categories"];
    let parsed: serde_json::Value = serde_json::from_str(note).unwrap();
    assert_eq!(parsed["codes"]["line one\nline two"], 0);

    let dup = dir.path().join("dup.csv");
    std::fs::write(&dup, "a,a\n1,2\n").unwrap();
    let err = convert_to_tet_with_options(
        &dup,
        &dir.path().join("dup.tet"),
        ConvertOptions::default(),
        None::<fn(_)>,
    )
    .unwrap_err();
    let ConvertError::Csv(detail) = err else {
        panic!("expected Csv error");
    };
    assert!(
        detail.contains("more than one column named `a`"),
        "{detail}"
    );
    assert!(!dir.path().join("dup.tet").exists());
}

#[test]
fn convert_csv_encodes_categories_and_skips_oversize_maps() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("groups.csv");
    std::fs::write(&src, "group,score\n ,1\ncontrol,2\ntreated,3\ncontrol,4\n").unwrap();
    let output = dir.path().join("groups.tet");
    let report =
        convert_to_tet_with_options(&src, &output, ConvertOptions::default(), None::<fn(_)>)
            .unwrap();
    assert_eq!(
        report.dataset_names,
        vec!["group".to_owned(), "score".to_owned()]
    );
    let mmap = mmap_file_read(&output).unwrap();
    assert_eq!(
        read_dataset_payload_le_bytes(&mmap, "group"),
        [-1i64, 0, 1, 0]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>()
    );
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let note = &summary.metadata.datasets["group"].attrs["tetration_categories"];
    let parsed: serde_json::Value = serde_json::from_str(note).unwrap();
    assert_eq!(parsed["missing"], -1);
    assert_eq!(parsed["codes"]["control"], 0);
    assert_eq!(parsed["codes"]["treated"], 1);
    assert_tet_verify_ok(&output);

    let wide = dir.path().join("wide.csv");
    let label = "x".repeat(1100);
    std::fs::write(&wide, format!("id,v\n{label},7\n")).unwrap();
    let wide_out = dir.path().join("wide.tet");
    let report =
        convert_to_tet_with_options(&wide, &wide_out, ConvertOptions::default(), None::<fn(_)>)
            .unwrap();
    assert_eq!(report.dataset_names, vec!["v".to_owned()]);
    assert!(
        report.skipped[0].reason.contains("1024-byte"),
        "{}",
        report.skipped[0].reason
    );
}

#[test]
fn convert_zarr_v2_roundtrips_raw_and_zstd_and_skips_blosc() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("sample.zarr");
    std::fs::create_dir_all(store.join("grp/bias")).unwrap();
    std::fs::create_dir_all(store.join("temp")).unwrap();
    std::fs::create_dir_all(store.join("packed")).unwrap();
    std::fs::create_dir_all(store.join("blosc")).unwrap();
    std::fs::write(store.join(".zgroup"), r#"{"zarr_format":2}"#).unwrap();
    std::fs::write(
        store.join("temp/.zarray"),
        r#"{"zarr_format":2,"shape":[4],"chunks":[2],"dtype":"<f4","order":"C","compressor":null,"filters":null,"fill_value":0}"#,
    )
    .unwrap();
    std::fs::write(store.join("temp/.zattrs"), r#"{"units":"K"}"#).unwrap();
    let mut temp = Vec::new();
    for value in [1.0f32, 2.0, 3.0, 4.0] {
        temp.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(store.join("temp/0"), &temp[..8]).unwrap();
    std::fs::write(store.join("temp/1"), &temp[8..]).unwrap();

    std::fs::write(store.join("grp/.zgroup"), r#"{"zarr_format":2}"#).unwrap();
    std::fs::write(
        store.join("grp/bias/.zarray"),
        r#"{"zarr_format":2,"shape":[2,2],"chunks":[2,2],"dtype":"<i4","order":"C","compressor":null,"filters":null,"fill_value":0}"#,
    )
    .unwrap();
    let bias: Vec<u8> = [5i32, 6, 7, 8]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect();
    std::fs::write(store.join("grp/bias/0.0"), bias).unwrap();

    std::fs::write(
        store.join("packed/.zarray"),
        r#"{"zarr_format":2,"shape":[2],"chunks":[2],"dtype":"<f8","order":"C","compressor":{"id":"zstd","level":1},"filters":null,"fill_value":0}"#,
    )
    .unwrap();
    let packed: Vec<u8> = [1.5f64, 2.5]
        .into_iter()
        .flat_map(f64::to_le_bytes)
        .collect();
    std::fs::write(
        store.join("packed/0"),
        zstd::encode_all(packed.as_slice(), 1).unwrap(),
    )
    .unwrap();

    std::fs::write(
        store.join("blosc/.zarray"),
        r#"{"zarr_format":2,"shape":[2],"chunks":[2],"dtype":"<f4","order":"C","compressor":{"id":"blosc"},"filters":null,"fill_value":0}"#,
    )
    .unwrap();

    let output = dir.path().join("sample.tet");
    let report =
        convert_to_tet_with_options(&store, &output, ConvertOptions::default(), None::<fn(_)>)
            .unwrap();
    assert_eq!(report.history[0].source, "zarr2");
    assert_eq!(
        report.dataset_names,
        vec![
            "grp/bias".to_owned(),
            "packed".to_owned(),
            "temp".to_owned()
        ]
    );
    assert_eq!(report.skipped.len(), 1);
    assert_eq!(report.skipped[0].path, "blosc");
    assert!(report.skipped[0].reason.contains("blosc"));

    let mmap = mmap_file_read(&output).unwrap();
    assert_eq!(read_dataset_payload_le_bytes(&mmap, "temp"), temp);
    assert_eq!(
        read_dataset_payload_le_bytes(&mmap, "grp/bias"),
        [5i32, 6, 7, 8]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(read_dataset_payload_le_bytes(&mmap, "packed"), packed);
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(
        summary.metadata.datasets["temp"]
            .attrs
            .get("units")
            .map(String::as_str),
        Some("K")
    );
    assert_tet_verify_ok(&output);
}

#[cfg(feature = "tetration-parquet")]
#[test]
fn convert_parquet_numeric_and_category_columns() {
    use std::sync::Arc;

    use arrow_array::{BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("table.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("temp", DataType::Float64, true),
        Field::new("n", DataType::Int64, false),
        Field::new("label", DataType::Utf8, true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Float64Array::from(vec![Some(1.5), None, Some(2.5)])),
            Arc::new(Int64Array::from(vec![2, 3, 4])),
            Arc::new(StringArray::from(vec![
                Some("control"),
                None,
                Some("treated"),
            ])),
            Arc::new(BooleanArray::from(vec![Some(true), Some(false), None])),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(&src).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let output = dir.path().join("table.tet");
    let report =
        convert_to_tet_with_options(&src, &output, ConvertOptions::default(), None::<fn(_)>)
            .unwrap();
    assert_eq!(report.history[0].source, "parquet");
    assert_eq!(
        report.dataset_names,
        vec!["temp".to_owned(), "n".to_owned(), "label".to_owned()]
    );
    assert_eq!(report.skipped[0].path, "flag");
    let mmap = mmap_file_read(&output).unwrap();
    let mut expected = Vec::new();
    expected.extend_from_slice(&1.5f64.to_le_bytes());
    expected.extend_from_slice(&f64::NAN.to_le_bytes());
    expected.extend_from_slice(&2.5f64.to_le_bytes());
    assert_eq!(read_dataset_payload_le_bytes(&mmap, "temp"), expected);
    assert_eq!(
        read_dataset_payload_le_bytes(&mmap, "n"),
        [2i64, 3, 4]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        read_dataset_payload_le_bytes(&mmap, "label"),
        [0i64, -1, 1]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>()
    );
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let note = &summary.metadata.datasets["label"].attrs["tetration_categories"];
    let parsed: serde_json::Value = serde_json::from_str(note).unwrap();
    assert_eq!(parsed["codes"]["control"], 0);
    assert_eq!(parsed["codes"]["treated"], 1);
    assert_tet_verify_ok(&output);
}

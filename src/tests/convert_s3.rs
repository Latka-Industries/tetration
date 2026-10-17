//! `s3://` single-file convert (THI-502).

use std::path::Path;

use crate::convert::{
    ConvertError, ConvertOptions, convert_s3_object_to_tet, convert_to_tet, parse_s3_convert_object,
};
use crate::query::{ExecuteQueryOptions, execute_query_json};

fn npy_f32_pair(a: f32, b: f32) -> Vec<u8> {
    let mut header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }".to_vec();
    let current = 10 + header.len() + 1;
    let pad = (64 - (current % 64)) % 64;
    header.extend(std::iter::repeat_n(b' ', pad));
    header.push(b'\n');
    assert_eq!((10 + header.len()) % 64, 0);
    let mut out = Vec::new();
    out.extend_from_slice(b"\x93NUMPY");
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(u16::try_from(header.len()).unwrap()).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&a.to_le_bytes());
    out.extend_from_slice(&b.to_le_bytes());
    out
}

fn dataset_sum(path: &Path) -> (String, f64) {
    let bytes = std::fs::read(path).unwrap();
    let summary = crate::catalog::read_tet_summary_v1(&bytes).unwrap();
    let name = summary.datasets[0].name.clone();
    let json = format!(r#"{{"dataset":"{name}","sum":[]}}"#);
    let resp = execute_query_json(
        &json,
        path,
        &bytes,
        ExecuteQueryOptions::execute_no_preview(),
        None,
    )
    .unwrap();
    let sum = resp
        .execution
        .as_ref()
        .and_then(|e| e.operation_sum)
        .unwrap();
    (name, sum)
}

#[test]
fn parse_s3_convert_accepts_npy_and_rejects_zarr() {
    let obj = parse_s3_convert_object("s3://bucket/dir/pair.npy").unwrap();
    assert_eq!(obj.bucket, "bucket");
    assert_eq!(obj.key, "dir/pair.npy");
    assert_eq!(obj.file_name, "pair.npy");

    let zarr = parse_s3_convert_object("s3://bucket/store.zarr").unwrap_err();
    assert!(matches!(zarr, ConvertError::S3ZarrNotSupported { .. }));
    let prefix = parse_s3_convert_object("s3://bucket/store/").unwrap_err();
    assert!(matches!(prefix, ConvertError::S3ZarrNotSupported { .. }));
}

#[cfg(not(feature = "tetration-s3"))]
#[test]
fn s3_convert_reports_missing_feature() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.tet");
    let err = convert_s3_object_to_tet(
        "s3://bucket/pair.npy",
        &out,
        ConvertOptions::default(),
        None::<fn(_)>,
    )
    .unwrap_err();
    assert!(matches!(err, ConvertError::S3FeatureDisabled));
}

#[cfg(feature = "tetration-s3")]
#[test]
fn minio_s3_npy_convert_matches_local_path() {
    use crate::catalog::{S3Config, S3Store, WritePart};

    let endpoint = std::env::var("TETRATION_S3_ENDPOINT").unwrap_or_default();
    if endpoint.is_empty() {
        if std::env::var("TETRATION_S3_REQUIRED").ok().as_deref() == Some("1") {
            panic!("TETRATION_S3_ENDPOINT is required but unset");
        }
        eprintln!("skipping minio convert test: set TETRATION_S3_ENDPOINT");
        return;
    }
    let bucket = std::env::var("TETRATION_S3_BUCKET").unwrap_or_else(|_| "tetration-test".into());
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let key = format!("convert-s3/{nanos}/pair.npy");
    let store = S3Store::connect(S3Config {
        endpoint,
        bucket: bucket.clone(),
        prefix: String::new(),
        region: std::env::var("TETRATION_S3_REGION").unwrap_or_else(|_| "us-east-1".into()),
        access_key_id: std::env::var("TETRATION_S3_ACCESS_KEY")
            .unwrap_or_else(|_| "tetration".into()),
        secret_access_key: std::env::var("TETRATION_S3_SECRET_KEY")
            .unwrap_or_else(|_| "tetration-secret".into()),
    })
    .unwrap();
    let npy = npy_f32_pair(1.5, 2.5);
    store.write_part(&key, &npy).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let local_npy = dir.path().join("pair.npy");
    let local_tet = dir.path().join("local.tet");
    let remote_tet = dir.path().join("remote.tet");
    std::fs::write(&local_npy, &npy).unwrap();
    convert_to_tet(&local_npy, &local_tet).unwrap();

    let uri = format!("s3://{bucket}/{key}");
    let report =
        convert_s3_object_to_tet(&uri, &remote_tet, ConvertOptions::default(), None::<fn(_)>)
            .unwrap();
    assert_eq!(report.input, uri);
    assert_eq!(report.dataset_names, ["data"]);

    let (local_name, local_sum) = dataset_sum(&local_tet);
    let (remote_name, remote_sum) = dataset_sum(&remote_tet);
    assert_eq!(local_name, remote_name);
    assert!((local_sum - remote_sum).abs() < 1e-6);
    assert!((remote_sum - 4.0).abs() < 1e-6);

    let _ = store.remove_object(&key);
}

#[test]
fn generated_npy_converts_locally() {
    let dir = tempfile::tempdir().unwrap();
    let npy = dir.path().join("pair.npy");
    std::fs::write(&npy, npy_f32_pair(1.5, 2.5)).unwrap();
    let out = dir.path().join("pair.tet");
    let report = convert_to_tet(&npy, &out).unwrap();
    assert_eq!(report.dataset_names, ["data"]);
    let (name, sum) = dataset_sum(&out);
    assert_eq!(name, "data");
    assert!((sum - 4.0).abs() < 1e-6);
}

#[test]
fn local_path_convert_still_rejects_unknown_extension() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("notes.txt");
    std::fs::write(&input, b"nope").unwrap();
    let err = convert_to_tet(&input, &dir.path().join("out.tet")).unwrap_err();
    assert!(matches!(
        err,
        ConvertError::UnsupportedInputExtension { .. }
    ));
}

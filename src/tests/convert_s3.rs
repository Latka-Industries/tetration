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
    assert!(matches!(zarr, ConvertError::S3(_)));
    let prefix = parse_s3_convert_object("s3://bucket/store/").unwrap_err();
    assert!(matches!(prefix, ConvertError::S3(_)));
}

#[cfg(not(feature = "tetration-s3"))]
#[test]
fn s3_zarr_convert_reports_missing_feature() {
    let dir = tempfile::tempdir().unwrap();
    let err = convert_s3_object_to_tet(
        "s3://bucket/store.zarr",
        &dir.path().join("out.tet"),
        ConvertOptions::default(),
        None::<fn(_)>,
    )
    .unwrap_err();
    assert!(matches!(err, ConvertError::S3FeatureDisabled));
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
    let store =
        S3Store::connect(S3Config::from_env(bucket.clone(), String::new()).unwrap()).unwrap();
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

#[cfg(feature = "tetration-s3")]
#[test]
fn minio_s3_zarr_convert_matches_local_directory() {
    use crate::catalog::{S3Config, S3Store, WritePart};
    use crate::convert::ConvertOptions;

    let endpoint = std::env::var("TETRATION_S3_ENDPOINT").unwrap_or_default();
    if endpoint.is_empty() {
        if std::env::var("TETRATION_S3_REQUIRED").ok().as_deref() == Some("1") {
            panic!("TETRATION_S3_ENDPOINT is required but unset");
        }
        eprintln!("skipping minio zarr convert test: set TETRATION_S3_ENDPOINT");
        return;
    }
    let bucket = std::env::var("TETRATION_S3_BUCKET").unwrap_or_else(|_| "tetration-test".into());
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let store =
        S3Store::connect(S3Config::from_env(bucket.clone(), String::new()).unwrap()).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let v3 = dir.path().join("v3.zarr");
    write_v3_pair(&v3);
    let v2 = dir.path().join("v2store");
    write_v2_pair(&v2);

    let v3_prefix = format!("zarr-s3/{nanos}/v3.zarr");
    let v2_prefix = format!("zarr-s3/{nanos}/v2store");
    upload_tree(&store, &v3_prefix, &v3);
    upload_tree(&store, &v2_prefix, &v2);
    store
        .write_part(&format!("{v3_prefix}/noise.bin"), b"not-a-chunk")
        .unwrap();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_s3_matches_local(
            &bucket,
            &v3_prefix,
            &v3,
            dir.path(),
            "zarr",
            ConvertOptions::default(),
        );
        assert_s3_matches_local(
            &bucket,
            &format!("{v3_prefix}/"),
            &v3,
            dir.path(),
            "zarr",
            ConvertOptions {
                parallel_jobs: 2,
                strict: false,
                chunk_mib: Some(1),
            },
        );
        assert_s3_matches_local(
            &bucket,
            &v2_prefix,
            &v2,
            dir.path(),
            "zarr2",
            ConvertOptions::default(),
        );
    }));
    let _ = store.remove_prefix(&format!("zarr-s3/{nanos}"));
    if let Err(err) = result {
        std::panic::resume_unwind(err);
    }
}

#[cfg(feature = "tetration-s3")]
fn assert_s3_matches_local(
    bucket: &str,
    prefix: &str,
    local: &std::path::Path,
    scratch: &std::path::Path,
    source: &str,
    options: crate::convert::ConvertOptions,
) {
    use crate::catalog::read_tet_summary_v1;
    use crate::convert::convert_to_tet_with_options;
    use crate::layout::mmap_file_read;

    use super::convert::materialize_dataset_le_bytes;

    let leaf = prefix
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or("z");
    let local_out = scratch.join(format!(
        "local-{leaf}-{}-{}.tet",
        options.parallel_jobs,
        options.chunk_mib.unwrap_or(0)
    ));
    let remote_out = scratch.join(format!(
        "remote-{leaf}-{}-{}.tet",
        options.parallel_jobs,
        options.chunk_mib.unwrap_or(0)
    ));
    convert_to_tet_with_options(local, &local_out, options, None::<fn(_)>).unwrap();
    let uri = format!("s3://{bucket}/{prefix}");
    let report = convert_s3_object_to_tet(&uri, &remote_out, options, None::<fn(_)>).unwrap();
    assert_eq!(report.input, uri);
    assert_eq!(report.history[0].source, source);
    let local_bytes = mmap_file_read(&local_out).unwrap();
    let remote_bytes = mmap_file_read(&remote_out).unwrap();
    let local_summary = read_tet_summary_v1(&local_bytes).unwrap();
    let remote_summary = read_tet_summary_v1(&remote_bytes).unwrap();
    assert_eq!(
        local_summary
            .datasets
            .iter()
            .map(|dataset| dataset.name.as_str())
            .collect::<Vec<_>>(),
        remote_summary
            .datasets
            .iter()
            .map(|dataset| dataset.name.as_str())
            .collect::<Vec<_>>()
    );
    for dataset in &local_summary.datasets {
        assert_eq!(
            materialize_dataset_le_bytes(&remote_bytes, &dataset.name),
            materialize_dataset_le_bytes(&local_bytes, &dataset.name),
            "{}",
            dataset.name
        );
    }
}

#[cfg(feature = "tetration-s3")]
fn upload_tree(store: &crate::catalog::S3Store, prefix: &str, root: &std::path::Path) {
    fn walk(store: &crate::catalog::S3Store, prefix: &str, root: &std::path::Path, rel: &str) {
        use crate::catalog::WritePart;
        let dir = if rel.is_empty() {
            root.to_path_buf()
        } else {
            root.join(rel)
        };
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().into_string().unwrap();
            let child = if rel.is_empty() {
                name
            } else {
                format!("{rel}/{name}")
            };
            if entry.path().is_dir() {
                walk(store, prefix, root, &child);
            } else {
                let bytes = std::fs::read(entry.path()).unwrap();
                store
                    .write_part(&format!("{prefix}/{child}"), &bytes)
                    .unwrap();
            }
        }
    }
    walk(store, prefix, root, "");
}

#[cfg(feature = "tetration-s3")]
fn write_v3_pair(root: &std::path::Path) {
    let data = root.join("data");
    std::fs::create_dir_all(data.join("c")).unwrap();
    std::fs::write(
        root.join("zarr.json"),
        r#"{"zarr_format":3,"node_type":"group"}"#,
    )
    .unwrap();
    std::fs::write(
        data.join("zarr.json"),
        r#"{"zarr_format":3,"node_type":"array","shape":[4],"data_type":"float32","chunk_grid":{"name":"regular","configuration":{"chunk_shape":[2]}},"codecs":[{"name":"bytes","configuration":{"endian":"little"}}]}"#,
    )
    .unwrap();
    let mut first = Vec::new();
    first.extend(1.0f32.to_le_bytes());
    first.extend(2.0f32.to_le_bytes());
    let mut second = Vec::new();
    second.extend(3.0f32.to_le_bytes());
    second.extend(4.0f32.to_le_bytes());
    std::fs::write(data.join("c/0"), first).unwrap();
    std::fs::write(data.join("c/1"), second).unwrap();
}

#[cfg(feature = "tetration-s3")]
fn write_v2_pair(root: &std::path::Path) {
    std::fs::create_dir_all(root.join("temp")).unwrap();
    std::fs::write(root.join(".zgroup"), r#"{"zarr_format":2}"#).unwrap();
    std::fs::write(
        root.join("temp/.zarray"),
        r#"{"zarr_format":2,"shape":[4],"chunks":[2],"dtype":"<f4","order":"C","compressor":null,"filters":null,"fill_value":0,"dimension_separator":"."}"#,
    )
    .unwrap();
    std::fs::write(root.join("temp/.zattrs"), r#"{"units":"K"}"#).unwrap();
    let mut payload = Vec::new();
    for value in [1.0f32, 2.0, 3.0, 4.0] {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(root.join("temp/0"), &payload[..8]).unwrap();
    std::fs::write(root.join("temp/1"), &payload[8..]).unwrap();
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

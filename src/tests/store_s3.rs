//! S3 store against local MinIO (THI-499). Skipped unless `TETRATION_S3_ENDPOINT` is set.
//! `TETRATION_S3_REQUIRED=1` turns a missing endpoint into a failure (CI).

use crate::catalog::{CatalogError, PublishGeneration, ReadAt, S3Config, S3Store, WritePart};

use super::store::assert_store_roundtrip;

fn minio_store(prefix: &str) -> Option<S3Store> {
    let endpoint = std::env::var("TETRATION_S3_ENDPOINT").unwrap_or_default();
    if endpoint.is_empty() {
        if std::env::var("TETRATION_S3_REQUIRED").ok().as_deref() == Some("1") {
            panic!("TETRATION_S3_ENDPOINT is required but unset");
        }
        eprintln!("skipping minio test: set TETRATION_S3_ENDPOINT");
        return None;
    }
    let bucket = std::env::var("TETRATION_S3_BUCKET").unwrap_or_else(|_| "tetration-test".into());
    let access_key_id =
        std::env::var("TETRATION_S3_ACCESS_KEY").unwrap_or_else(|_| "tetration".into());
    let secret_access_key =
        std::env::var("TETRATION_S3_SECRET_KEY").unwrap_or_else(|_| "tetration-secret".into());
    let region = std::env::var("TETRATION_S3_REGION").unwrap_or_else(|_| "us-east-1".into());
    Some(
        S3Store::connect(S3Config {
            endpoint,
            bucket,
            prefix: prefix.to_string(),
            region,
            access_key_id,
            secret_access_key,
        })
        .expect("connect minio"),
    )
}

fn unique_prefix(label: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("tests/{label}-{nanos}-{n}")
}

struct ClearOnDrop(S3Store);

impl Drop for ClearOnDrop {
    fn drop(&mut self) {
        let _ = self.0.clear();
    }
}

#[test]
fn minio_store_read_write_publish() {
    let Some(store) = minio_store(&unique_prefix("roundtrip")) else {
        return;
    };
    let store = ClearOnDrop(store);
    assert_store_roundtrip(&store.0);
}

#[test]
fn minio_lost_conditional_put_keeps_current() {
    let Some(store) = minio_store(&unique_prefix("cas")) else {
        return;
    };
    let store = ClearOnDrop(store);
    let store = &store.0;
    store.write_part("CURRENT", b"1\n").unwrap();
    let etag = store.e_tag("CURRENT").unwrap().expect("etag");
    store.write_part("CURRENT", b"2\n").unwrap();
    let err = store
        .put_if_match("CURRENT", b"3\n", &etag)
        .expect_err("stale etag");
    assert!(matches!(err, CatalogError::InvalidCurrent));
    assert_eq!(store.read_all("CURRENT").unwrap(), b"2\n");
    assert_eq!(store.current_generation().unwrap(), Some(2));
}

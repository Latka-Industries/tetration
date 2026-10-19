//! S3 store against local MinIO (THI-499). Skipped unless `TETRATION_S3_ENDPOINT` is set.
//! `TETRATION_S3_REQUIRED=1` turns a missing endpoint into a failure (CI).

use crate::catalog::{CatalogError, PublishGeneration, ReadAt, S3Config, S3Store, WritePart};

use super::store::assert_store_roundtrip;

pub(crate) fn minio_store(prefix: &str) -> Option<S3Store> {
    let endpoint = std::env::var("TETRATION_S3_ENDPOINT").unwrap_or_default();
    if endpoint.is_empty() {
        if std::env::var("TETRATION_S3_REQUIRED").ok().as_deref() == Some("1") {
            panic!("TETRATION_S3_ENDPOINT is required but unset");
        }
        eprintln!("skipping minio test: set TETRATION_S3_ENDPOINT");
        return None;
    }
    let bucket = std::env::var("TETRATION_S3_BUCKET").unwrap_or_else(|_| "tetration-test".into());
    Some(
        S3Store::connect(S3Config::from_env(bucket, prefix).expect("s3 env"))
            .expect("connect minio"),
    )
}

pub(crate) fn unique_prefix(label: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("tests/{label}-{nanos}-{n}")
}

pub(crate) struct ClearOnDrop(pub(crate) S3Store);

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

fn tiny_tet() -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("part.tet");
    let mut session = crate::catalog::TetWriterSession::create(&path);
    let data: Vec<u8> = [1.0_f32, 2.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    session
        .push_dataset(
            crate::catalog::TetDatasetWrite::f32_row_major("a", &[2], &[2], data).unwrap(),
        )
        .unwrap();
    session.commit().unwrap();
    std::fs::read(path).unwrap()
}

#[test]
fn minio_prune_dry_run_apply_and_pin() {
    use std::slice::from_ref;

    use crate::catalog::{
        GenerationPin, PruneOptions, PruneRefuseReason, PruneTarget, packed_tet_key,
        pin_packed_tet, prune_store, publish_staged_parts, stage_part,
    };

    let Some(store) = minio_store(&unique_prefix("prune")) else {
        return;
    };
    let store = ClearOnDrop(store);
    let store = &store.0;
    let bytes = tiny_tet();
    for generation in 1..=4 {
        let part = format!("p{generation}");
        stage_part(store, &format!("s{generation}"), &part, &bytes).unwrap();
        publish_staged_parts(store, generation, &[part]).unwrap();
    }
    let before = store.read_all("CURRENT").unwrap();
    let dry = prune_store(store, &PruneOptions::default()).unwrap();
    assert_eq!(dry.generations, vec![1, 2]);
    assert!(!dry.staging_sessions.is_empty());
    assert!(store.object_meta(&packed_tet_key(1)).unwrap().is_some());

    let pin = pin_packed_tet(store, GenerationPin::Explicit(1)).unwrap();
    let applied = prune_store(
        store,
        &PruneOptions::default()
            .apply()
            .with_store_pins(from_ref(&pin)),
    )
    .unwrap();
    assert_eq!(applied.generations, vec![2]);
    assert!(applied.refused.iter().any(|refusal| {
        refusal.target == PruneTarget::Generation(1) && refusal.reason == PruneRefuseReason::OpenPin
    }));
    assert!(store.object_meta(&packed_tet_key(1)).unwrap().is_some());
    assert!(store.object_meta(&packed_tet_key(2)).unwrap().is_none());
    assert!(store.object_meta(&packed_tet_key(3)).unwrap().is_some());
    assert!(store.object_meta(&packed_tet_key(4)).unwrap().is_some());
    assert!(store.list_prefix("staging/").unwrap().is_empty());
    assert_eq!(store.read_all("CURRENT").unwrap(), before);
    assert_eq!(store.current_generation().unwrap(), Some(4));
}

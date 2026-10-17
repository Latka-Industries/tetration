//! Parallel store ingest, one CAS publish, range query, lost publish (THI-471).
//! Cross-host publish claim / force stale takeover (THI-500).

use std::sync::Barrier;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use crate::catalog::store::read_generation_manifest;
use crate::catalog::{
    CatalogError, GenerationPin, MemoryStore, PackageStore, PublishGeneration, PublishOptions,
    TetDatasetWrite, TetWriterSession, WritePart, begin_publish_store, heartbeat_key,
    packed_tet_key, publish_lock_exists, publish_lock_is_stale, publish_staged_parts,
    publishing_lock_key, seal_staged_generation, stage_part,
};
use crate::query::{
    ExecuteQueryOptions, execute_query_on_store_pin, parse_query_json, validate_query,
};
use crate::verify::{VerifyOptions, verify_tet_bytes};

fn tiny_tet(name: &str, vals: &[f32]) -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("part.tet");
    let mut session = TetWriterSession::create(&path);
    let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    session
        .push_dataset(TetDatasetWrite::f32_row_major(name, &[2], &[2], data).unwrap())
        .unwrap();
    session.commit().unwrap();
    std::fs::read(path).unwrap()
}

fn assert_generation<S: PackageStore>(store: &S, generation: u64, part_ids: &[&str]) {
    let manifest = read_generation_manifest(store, generation)
        .unwrap()
        .expect("manifest");
    assert_eq!(manifest.generation, generation);
    assert_eq!(
        manifest.part_ids,
        part_ids
            .iter()
            .map(|s| (*s).to_string())
            .collect::<Vec<_>>()
    );
    let packed = store.read_all(&packed_tet_key(generation)).unwrap();
    assert_eq!(packed.len() as u64, manifest.packed_byte_len);
    let report = verify_tet_bytes(&packed, None, VerifyOptions::default());
    assert!(report.ok, "{:?}", report.findings);
    let listed = store.list_prefix("staging/").unwrap();
    for id in part_ids {
        let suffix = format!("/part-{id}.tet");
        assert!(
            listed.iter().any(|k| k.ends_with(&suffix)),
            "missing listed part {id}: {listed:?}"
        );
    }
}

fn query_sum<S: PackageStore>(store: &S, dataset: &str) -> f64 {
    let doc = parse_query_json(&format!(r#"{{"dataset":"{dataset}","sum":[]}}"#)).unwrap();
    validate_query(&doc).unwrap();
    let resp = execute_query_on_store_pin(
        &doc,
        store,
        GenerationPin::Current,
        ExecuteQueryOptions::execute_no_preview(),
    )
    .unwrap();
    resp.execution
        .as_ref()
        .and_then(|e| e.operation_sum)
        .expect("sum")
}

/// N writers stage parts, one publish wins, a stale CAS leaves that generation.
fn parallel_ingest_then_lost_publish<S: PackageStore + Sync>(store: &S) {
    let parts = [
        ("d0", 1.0, 2.0),
        ("d1", 3.0, 4.0),
        ("d2", 5.0, 6.0),
        ("d3", 7.0, 8.0),
    ];
    thread::scope(|scope| {
        for (i, (name, a, b)) in parts.iter().enumerate() {
            scope.spawn(move || {
                let bytes = tiny_tet(name, &[*a, *b]);
                stage_part(store, &format!("s{i}"), &format!("p{i}"), &bytes).unwrap();
            });
        }
    });

    let ids: Vec<String> = (0..parts.len()).map(|i| format!("p{i}")).collect();
    let manifest = publish_staged_parts(store, 1, &ids).unwrap();
    assert_eq!(manifest.part_ids, ids);
    assert_eq!(store.current_generation().unwrap(), Some(1));
    assert_generation(store, 1, &["p0", "p1", "p2", "p3"]);
    assert!((query_sum(store, "d0") - 3.0).abs() < 1e-6);
    assert!((query_sum(store, "d3") - 15.0).abs() < 1e-6);

    let before = store.read_all(&packed_tet_key(1)).unwrap();
    seal_staged_generation(store, 2, &ids).unwrap();
    let err = store.publish_generation(2, None).unwrap_err();
    assert!(matches!(err, CatalogError::InvalidCurrent));
    assert_eq!(store.current_generation().unwrap(), Some(1));
    assert_eq!(store.read_all(&packed_tet_key(1)).unwrap(), before);
    assert_generation(store, 1, &["p0", "p1", "p2", "p3"]);
    assert!((query_sum(store, "d1") - 7.0).abs() < 1e-6);
}

#[test]
fn memory_parallel_ingest_publish_and_lost_cas() {
    parallel_ingest_then_lost_publish(&MemoryStore::new());
}

/// Two committers race one generation; exactly one claim wins and `CURRENT` flips once.
fn two_publishers_one_wins<S: PackageStore + Sync>(store: &S) {
    let a = tiny_tet("a", &[1.0, 2.0]);
    let b = tiny_tet("b", &[3.0, 4.0]);
    stage_part(store, "sa", "pa", &a).unwrap();
    stage_part(store, "sb", "pb", &b).unwrap();
    let ids = vec!["pa".to_string(), "pb".to_string()];

    let barrier = Barrier::new(2);
    let wins = AtomicUsize::new(0);
    let losses = AtomicUsize::new(0);
    thread::scope(|scope| {
        for _ in 0..2 {
            scope.spawn(|| {
                barrier.wait();
                match publish_staged_parts(store, 1, &ids) {
                    Ok(_) => {
                        wins.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(
                        CatalogError::PublishLockHeld(1)
                        | CatalogError::PackagePathExists(_)
                        | CatalogError::Package(_),
                    ) => {
                        // Lost exclusive claim, or lost rename into `generations/<G>` after the
                        // winner already sealed.
                        losses.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(err) => panic!("unexpected publish error: {err:?}"),
                }
            });
        }
    });

    assert_eq!(
        wins.load(Ordering::SeqCst),
        1,
        "exactly one publisher must win"
    );
    assert_eq!(
        losses.load(Ordering::SeqCst),
        1,
        "loser must not flip CURRENT"
    );
    assert_eq!(store.current_generation().unwrap(), Some(1));
    assert_generation(store, 1, &["pa", "pb"]);
    assert!(!publish_lock_exists(store, 1).unwrap());
}

#[test]
fn memory_two_publishers_one_wins() {
    two_publishers_one_wins(&MemoryStore::new());
}

#[test]
fn memory_stale_force_takeover_live_force_held() {
    let store = MemoryStore::new();
    let bytes = tiny_tet("x", &[9.0, 8.0]);
    stage_part(&store, "sx", "px", &bytes).unwrap();

    begin_publish_store(&store, 7, &PublishOptions::default()).unwrap();
    // Live lock: force must not steal.
    let err = begin_publish_store(
        &store,
        7,
        &PublishOptions {
            force: true,
            stale_after: Duration::from_secs(3600),
        },
    )
    .unwrap_err();
    assert!(matches!(err, CatalogError::PublishLockHeld(7)));

    // Age the heartbeat into the past.
    store.write_part(&heartbeat_key(7), b"1\n").unwrap();
    assert!(publish_lock_is_stale(&store, 7, Duration::from_secs(60)).unwrap());

    begin_publish_store(
        &store,
        7,
        &PublishOptions {
            force: true,
            stale_after: Duration::from_secs(60),
        },
    )
    .unwrap();
    assert!(publish_lock_exists(&store, 7).unwrap());
    // Drop the test claim so a normal publish can take generation 7.
    store.remove_prefix(&publishing_lock_key(7)).unwrap();

    let ids = vec!["px".to_string()];
    let manifest = publish_staged_parts(&store, 7, &ids).unwrap();
    assert_eq!(manifest.generation, 7);
    assert_eq!(store.current_generation().unwrap(), Some(7));
}

#[cfg(feature = "tetration-s3")]
#[test]
fn minio_parallel_ingest_publish_and_lost_cas() {
    use super::store_s3::minio_store;

    let Some(store) = minio_store(&unique_prefix()) else {
        return;
    };
    let store = ClearOnDrop(store);
    parallel_ingest_then_lost_publish(&store.0);
}

#[cfg(feature = "tetration-s3")]
#[test]
fn minio_two_publishers_one_wins() {
    use super::store_s3::minio_store;

    let Some(store) = minio_store(&unique_prefix()) else {
        return;
    };
    let store = ClearOnDrop(store);
    two_publishers_one_wins(&store.0);
}

#[cfg(feature = "tetration-s3")]
fn unique_prefix() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("tests/ingest-{nanos}")
}

#[cfg(feature = "tetration-s3")]
struct ClearOnDrop(crate::catalog::S3Store);

#[cfg(feature = "tetration-s3")]
impl Drop for ClearOnDrop {
    fn drop(&mut self) {
        let _ = self.0.clear();
    }
}

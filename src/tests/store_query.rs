//! Store-backed range planner + query (THI-470).

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::catalog::store::{
    ByteRange, GenerationPin, MemoryStore, ObjectMeta, PublishGeneration, ReadAt, WritePart,
    merge_byte_ranges, packed_tet_key, parse_tet_uri, pin_packed_tet, plan_catalog_fetch,
    read_tet_summary_at_store,
};
use crate::catalog::{CatalogError, TetDatasetWrite, TetWriterSession, read_tet_summary_v1};
use crate::layout::read_superblock_v1;
use crate::query::{
    ExecuteQueryOptions, execute_query_on_store_pin, execute_query_with_store, parse_query_json,
    plan_query_fetch, plan_query_with_tet_mmap_ex, validate_query,
};

/// Counts `read_at` / forbids full-object `read_all` on `packed.tet`.
struct RecordingStore {
    inner: MemoryStore,
    read_at_calls: AtomicUsize,
    forbidden_full: Mutex<Vec<String>>,
}

impl RecordingStore {
    fn new(inner: MemoryStore) -> Self {
        Self {
            inner,
            read_at_calls: AtomicUsize::new(0),
            forbidden_full: Mutex::new(Vec::new()),
        }
    }
}

impl ReadAt for RecordingStore {
    fn read_at(&self, key: &str, offset: u64, len: u64) -> Result<Vec<u8>, CatalogError> {
        self.read_at_calls.fetch_add(1, Ordering::SeqCst);
        self.inner.read_at(key, offset, len)
    }

    fn read_all(&self, key: &str) -> Result<Vec<u8>, CatalogError> {
        if key.contains("packed.tet") {
            self.forbidden_full
                .lock()
                .unwrap()
                .push(format!("read_all {key}"));
            return Err(CatalogError::Package("test forbids read_all on packed.tet"));
        }
        self.inner.read_all(key)
    }

    fn object_meta(&self, key: &str) -> Result<Option<ObjectMeta>, CatalogError> {
        self.inner.object_meta(key)
    }

    fn list_prefix(&self, prefix: &str) -> Result<Vec<String>, CatalogError> {
        self.inner.list_prefix(prefix)
    }
}

impl WritePart for RecordingStore {
    fn create_dir_exclusive(&self, key: &str) -> Result<(), CatalogError> {
        self.inner.create_dir_exclusive(key)
    }
    fn write_part(&self, key: &str, bytes: &[u8]) -> Result<(), CatalogError> {
        self.inner.write_part(key, bytes)
    }
    fn remove_object(&self, key: &str) -> Result<(), CatalogError> {
        self.inner.remove_object(key)
    }
    fn remove_prefix(&self, prefix: &str) -> Result<(), CatalogError> {
        self.inner.remove_prefix(prefix)
    }
    fn rename_prefix(&self, from_prefix: &str, to_prefix: &str) -> Result<(), CatalogError> {
        self.inner.rename_prefix(from_prefix, to_prefix)
    }
}

impl PublishGeneration for RecordingStore {
    fn publish_generation(
        &self,
        generation: u64,
        expected_current: Option<u64>,
    ) -> Result<(), CatalogError> {
        self.inner.publish_generation(generation, expected_current)
    }
    fn current_generation(&self) -> Result<Option<u64>, CatalogError> {
        self.inner.current_generation()
    }
}

fn multi_chunk_tet() -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.tet");
    let mut session = TetWriterSession::create(&path);
    let mut data = vec![0u8; 4 * 4 * 4];
    for (i, chunk) in data.chunks_exact_mut(4).enumerate() {
        chunk.copy_from_slice(&(i as f32).to_le_bytes());
    }
    session
        .push_dataset(TetDatasetWrite::f32_row_major("a", &[4, 4], &[2, 2], data).unwrap())
        .unwrap();
    session.commit().unwrap();
    std::fs::read(&path).unwrap()
}

fn seeded_store() -> RecordingStore {
    let packed = multi_chunk_tet();
    let mem = MemoryStore::new();
    mem.write_part(&packed_tet_key(1), &packed).unwrap();
    mem.write_part(
        "generations/00000001/MANIFEST.json",
        format!(
            r#"{{"generation":1,"protocol_version":2,"part_ids":["a"],"packed_byte_len":{}}}"#,
            packed.len()
        )
        .as_bytes(),
    )
    .unwrap();
    mem.publish_generation(1, None).unwrap();
    RecordingStore::new(mem)
}

#[test]
fn parse_uri_shapes() {
    let p = parse_tet_uri("/tmp/x.tet").unwrap();
    assert!(matches!(
        p.target,
        crate::catalog::TetOpenTarget::LocalTet(_)
    ));
    let p = parse_tet_uri("/tmp/pkg.tet.d").unwrap();
    assert!(matches!(
        p.target,
        crate::catalog::TetOpenTarget::LocalPackage { .. }
    ));
    let p = parse_tet_uri("ds@./Foo.TET").unwrap();
    assert_eq!(p.label.as_deref(), Some("ds"));
    assert!(matches!(
        p.target,
        crate::catalog::TetOpenTarget::LocalTet(_)
    ));
    let p = parse_tet_uri("/tmp/Pkg.TET.D").unwrap();
    assert!(matches!(
        p.target,
        crate::catalog::TetOpenTarget::LocalPackage { .. }
    ));
    let p = parse_tet_uri("memory://current").unwrap();
    assert!(matches!(
        p.target,
        crate::catalog::TetOpenTarget::StorePackage {
            generation: GenerationPin::Current
        }
    ));
    let p = parse_tet_uri("data@s3://bucket/pref/packed.tet").unwrap();
    assert_eq!(p.label.as_deref(), Some("data"));
    assert!(matches!(p.target, crate::catalog::TetOpenTarget::S3 { .. }));
}

#[test]
fn parse_s3_package_uri_prefix_and_generation() {
    use crate::catalog::parse_s3_package_uri;

    let loc = parse_s3_package_uri("ds@s3://bucket/a/pkg/?generation=4").unwrap();
    assert_eq!(loc.label.as_deref(), Some("ds"));
    assert_eq!(loc.bucket, "bucket");
    assert_eq!(loc.prefix, "a/pkg");
    assert_eq!(loc.generation, GenerationPin::Explicit(4));

    let root = parse_s3_package_uri("s3://bucket").unwrap();
    assert_eq!(root.prefix, "");
    assert_eq!(root.generation, GenerationPin::Current);

    assert!(parse_s3_package_uri("s3://bucket/a/../b").is_err());
    assert!(parse_s3_package_uri("/tmp/x.tet").is_err());
}

#[test]
fn merge_ranges_coalesces() {
    let m = merge_byte_ranges([
        ByteRange::new(0, 10),
        ByteRange::new(10, 5),
        ByteRange::new(20, 1),
    ]);
    assert_eq!(m, vec![ByteRange::new(0, 15), ByteRange::new(20, 1)]);
}

#[test]
fn summary_via_store_matches_mmap() {
    let store = seeded_store();
    let pin = pin_packed_tet(&store, GenerationPin::Current).unwrap();
    let via = read_tet_summary_at_store(&store, &pin.object_key, pin.object_len).unwrap();
    let full = store.inner.read_all(&pin.object_key).unwrap();
    let mmap = read_tet_summary_v1(&full).unwrap();
    assert_eq!(via.datasets, mmap.datasets);
    assert_eq!(via.chunks.len(), mmap.chunks.len());
    assert!(
        store.read_at_calls.load(Ordering::SeqCst) > 0,
        "expected range reads"
    );
    assert!(store.forbidden_full.lock().unwrap().is_empty());
}

#[test]
fn query_sum_via_store_ranges_matches_mmap() {
    let store = seeded_store();
    let pin = pin_packed_tet(&store, GenerationPin::Current).unwrap();
    let full = store.inner.read_all(&pin.object_key).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();

    let mmap_resp = plan_query_with_tet_mmap_ex(&doc, None, &full, Some(0), None).unwrap();
    let store_resp = execute_query_with_store(
        &doc,
        &store,
        &pin,
        ExecuteQueryOptions::execute_no_preview(),
        None,
    )
    .unwrap();

    assert_eq!(
        store_resp.read_plan.as_ref().map(|p| p.chunks.len()),
        mmap_resp.read_plan.as_ref().map(|p| p.chunks.len())
    );
    let mmap_sum = mmap_resp.execution.as_ref().and_then(|e| e.operation_sum);
    let store_sum = store_resp.execution.as_ref().and_then(|e| e.operation_sum);
    assert!(
        mmap_sum.is_some(),
        "mmap baseline missing sum: {mmap_resp:?}"
    );
    assert_eq!(store_sum, mmap_sum);
    assert!(store.forbidden_full.lock().unwrap().is_empty());
    assert!(store.read_at_calls.load(Ordering::SeqCst) >= 2);
}

#[test]
fn pin_survives_current_advance() {
    let store = seeded_store();
    let pin = pin_packed_tet(&store, GenerationPin::Current).unwrap();
    assert_eq!(pin.generation, 1);

    // Publish a second generation pointing CURRENT at 2; pin still reads G=1.
    let packed = multi_chunk_tet();
    store.write_part(&packed_tet_key(2), &packed).unwrap();
    store.publish_generation(2, Some(1)).unwrap();
    assert_eq!(store.current_generation().unwrap(), Some(2));

    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let resp = execute_query_with_store(
        &doc,
        &store,
        &pin,
        ExecuteQueryOptions::execute_no_preview(),
        None,
    )
    .unwrap();
    assert!(
        resp.execution
            .as_ref()
            .and_then(|e| e.operation_sum)
            .is_some()
    );
    assert_eq!(pin.object_key, packed_tet_key(1));
}

#[test]
fn catalog_plan_includes_header() {
    let packed = multi_chunk_tet();
    let sb = read_superblock_v1(&packed).unwrap();
    let plan = plan_catalog_fetch(packed.len() as u64, &sb).unwrap();
    assert!(!plan.ranges.is_empty());
    assert_eq!(plan.ranges[0].offset, 0);
    assert!(plan.ranges[0].len >= 32);
}

#[test]
fn execute_on_store_pin_current() {
    let store = seeded_store();
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let resp = execute_query_on_store_pin(
        &doc,
        &store,
        GenerationPin::Current,
        ExecuteQueryOptions::execute_no_preview(),
    )
    .unwrap();
    assert!(
        resp.execution
            .as_ref()
            .and_then(|e| e.operation_sum)
            .is_some()
    );
}

#[test]
fn query_fetch_plan_covers_touched_chunks() {
    let store = seeded_store();
    let pin = pin_packed_tet(&store, GenerationPin::Current).unwrap();
    let full = store.inner.read_all(&pin.object_key).unwrap();
    let sb = read_superblock_v1(&full).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let planned = crate::query::plan_read_for_document(&doc, &full).unwrap();
    let fetch = plan_query_fetch(&pin.object_key, pin.object_len, &sb, &planned.read_plan).unwrap();
    assert!(fetch.ranges.len() >= 1);
    let payload_bytes: u64 = planned
        .read_plan
        .chunks
        .iter()
        .map(|c| c.stored_byte_len)
        .sum();
    let fetched_payload: u64 = fetch
        .ranges
        .iter()
        .skip(1) // first range is catalog header
        .map(|r| r.len)
        .sum();
    assert!(fetched_payload >= payload_bytes || fetch.ranges.len() == 1);
}

#[cfg(feature = "tetration-s3")]
#[test]
fn query_s3_package_uri_sum_matches_mmap() {
    use super::store_s3::{ClearOnDrop, minio_store, unique_prefix};
    use crate::catalog::{PublishGeneration, WritePart, open_s3_package, packed_tet_key};
    use crate::query::query_opened_s3_package;

    let prefix = unique_prefix("query");
    let Some(store) = minio_store(&prefix) else {
        return;
    };
    let store = ClearOnDrop(store);
    let packed = multi_chunk_tet();
    store.0.write_part(&packed_tet_key(1), &packed).unwrap();
    store
        .0
        .write_part(
            "generations/00000001/MANIFEST.json",
            format!(
                r#"{{"generation":1,"protocol_version":2,"part_ids":["a"],"packed_byte_len":{}}}"#,
                packed.len()
            )
            .as_bytes(),
        )
        .unwrap();
    store.0.publish_generation(1, None).unwrap();

    let bucket = std::env::var("TETRATION_S3_BUCKET").unwrap_or_else(|_| "tetration-test".into());
    let uri = format!("s3://{bucket}/{prefix}?generation=1");
    let opened = open_s3_package(&uri).unwrap();
    assert_eq!(opened.pin().generation, 1);

    let current = open_s3_package(&format!("s3://{bucket}/{prefix}")).unwrap();
    assert_eq!(current.pin().generation, 1);
    assert_eq!(current.pin().object_key, opened.pin().object_key);

    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let via = query_opened_s3_package(
        &doc,
        &opened,
        true,
        ExecuteQueryOptions::execute_no_preview(),
        None,
        Some(&uri),
    )
    .unwrap();
    let mmap = plan_query_with_tet_mmap_ex(&doc, None, &packed, Some(0), None).unwrap();
    assert_eq!(via.tet_file.as_deref(), Some(uri.as_str()));
    assert_eq!(
        via.execution.as_ref().and_then(|e| e.operation_sum),
        mmap.execution.as_ref().and_then(|e| e.operation_sum)
    );

    let plan_doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&plan_doc).unwrap();
    let planned = query_opened_s3_package(
        &plan_doc,
        &opened,
        false,
        ExecuteQueryOptions::plan_only(),
        None,
        Some(&uri),
    )
    .unwrap();
    assert!(planned.execution.is_none());
    assert!(planned.read_plan.is_some());
    assert_eq!(planned.tet_file.as_deref(), Some(uri.as_str()));
}

//! Optional packed.tet block cache (THI-501).

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::catalog::store::{
    BlockCache, CachingStore, GenerationPin, MemoryStore, ObjectMeta, PublishGeneration, ReadAt,
    WritePart, packed_tet_key, pin_packed_tet, publish_staged_parts, stage_part,
};
use crate::catalog::{CatalogError, TetDatasetWrite, TetWriterSession};

/// Counts `read_at` calls on the inner store.
struct CountingStore {
    inner: MemoryStore,
    read_at_calls: AtomicUsize,
}

impl CountingStore {
    fn new(inner: MemoryStore) -> Self {
        Self {
            inner,
            read_at_calls: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.read_at_calls.load(Ordering::SeqCst)
    }
}

impl ReadAt for CountingStore {
    fn read_at(&self, key: &str, offset: u64, len: u64) -> Result<Vec<u8>, CatalogError> {
        self.read_at_calls.fetch_add(1, Ordering::SeqCst);
        self.inner.read_at(key, offset, len)
    }

    fn object_meta(&self, key: &str) -> Result<Option<ObjectMeta>, CatalogError> {
        self.inner.object_meta(key)
    }

    fn list_prefix(&self, prefix: &str) -> Result<Vec<String>, CatalogError> {
        self.inner.list_prefix(prefix)
    }
}

impl WritePart for CountingStore {
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

impl PublishGeneration for CountingStore {
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

fn publish_one(
    store: &impl crate::catalog::PackageStore,
    generation: u64,
    name: &str,
    vals: &[f32],
) {
    let bytes = tiny_tet(name, vals);
    let part = format!("p{generation}");
    stage_part(store, &format!("s{generation}"), &part, &bytes).unwrap();
    publish_staged_parts(store, generation, &[part]).unwrap();
}

#[test]
fn repeated_range_served_from_cache() {
    let mem = MemoryStore::new();
    publish_one(&mem, 1, "a", &[1.0, 2.0]);
    let pin = pin_packed_tet(&mem, GenerationPin::Current).unwrap();
    let counting = CountingStore::new(mem);
    let cached = CachingStore::new(counting, pin.clone());

    let first = cached.read_at(&pin.object_key, 0, 32).unwrap();
    assert_eq!(first.len(), 32);
    assert_eq!(cached.inner().calls(), 1);

    let second = cached.read_at(&pin.object_key, 0, 32).unwrap();
    assert_eq!(second, first);
    assert_eq!(
        cached.inner().calls(),
        1,
        "second identical range must not refetch"
    );

    let prefix = cached.read_at(&pin.object_key, 0, 16).unwrap();
    assert_eq!(prefix, first[..16]);
    assert_eq!(
        cached.inner().calls(),
        1,
        "shorter prefix of a cached range is a hit"
    );
}

#[test]
fn without_cache_always_hits_inner() {
    let mem = MemoryStore::new();
    publish_one(&mem, 1, "a", &[1.0, 2.0]);
    let pin = pin_packed_tet(&mem, GenerationPin::Current).unwrap();
    let counting = CountingStore::new(mem);
    let passthrough = CachingStore::without_cache(counting, pin.clone());

    let _ = passthrough.read_at(&pin.object_key, 0, 16).unwrap();
    let _ = passthrough.read_at(&pin.object_key, 0, 16).unwrap();
    assert_eq!(passthrough.inner().calls(), 2);
}

#[test]
fn block_cache_keys_by_generation() {
    let mut cache = BlockCache::new();
    cache.insert(1, 0, vec![1, 2, 3, 4]);
    cache.insert(2, 0, vec![9, 8, 7, 6]);
    assert_eq!(cache.get(1, 0, 4).unwrap(), [1, 2, 3, 4]);
    assert_eq!(cache.get(2, 0, 4).unwrap(), [9, 8, 7, 6]);
    assert!(cache.get(3, 0, 1).is_none());
}

#[test]
fn new_pin_does_not_reuse_other_generation_blocks() {
    let mem = MemoryStore::new();
    publish_one(&mem, 1, "g1", &[1.0, 2.0]);
    let packed_g1 = mem.read_all(&packed_tet_key(1)).unwrap();
    publish_one(&mem, 2, "g2", &[9.0, 8.0]);
    let packed_g2 = mem.read_all(&packed_tet_key(2)).unwrap();
    let offset = packed_g1
        .iter()
        .zip(packed_g2.iter())
        .position(|(a, b)| a != b)
        .expect("generations must differ somewhere") as u64;
    let len = 8u64;
    assert!(offset + len <= packed_g1.len() as u64);
    assert!(offset + len <= packed_g2.len() as u64);
    assert_eq!(mem.current_generation().unwrap(), Some(2));

    let shared = Arc::new(Mutex::new(BlockCache::new()));
    let pin1 = pin_packed_tet(&mem, GenerationPin::Explicit(1)).unwrap();
    let pin2 = pin_packed_tet(&mem, GenerationPin::Current).unwrap();
    assert_eq!(pin2.generation, 2);

    let wrap1 = CachingStore::with_cache(mem, pin1.clone(), Arc::clone(&shared));
    let from_g1 = wrap1.read_at(&pin1.object_key, offset, len).unwrap();
    let start = offset as usize;
    let end = start + len as usize;
    assert_eq!(from_g1, packed_g1[start..end]);
    assert!(shared.lock().unwrap().contains(1, offset, len));
    assert!(!shared.lock().unwrap().contains(2, offset, len));

    // CURRENT already advanced to 2; older pin still returns G1 from cache.
    let again_g1 = wrap1.read_at(&pin1.object_key, offset, len).unwrap();
    assert_eq!(again_g1, from_g1);

    let mem = wrap1.into_inner();
    let wrap2 = CachingStore::with_cache(mem, pin2.clone(), Arc::clone(&shared));
    let from_g2 = wrap2.read_at(&pin2.object_key, offset, len).unwrap();
    assert_eq!(from_g2, packed_g2[start..end]);
    assert_ne!(from_g2, from_g1);
    assert!(shared.lock().unwrap().contains(2, offset, len));

    // Re-wrap the older pin against the shared map: still G1 bytes, not G2.
    let mem = wrap2.into_inner();
    let key_g1 = pin1.object_key.clone();
    let still_g1 = CachingStore::with_cache(mem, pin1, shared)
        .read_at(&key_g1, offset, len)
        .unwrap();
    assert_eq!(still_g1, from_g1);
}

#[test]
fn non_packed_keys_bypass_cache() {
    let mem = MemoryStore::new();
    publish_one(&mem, 1, "a", &[1.0, 2.0]);
    let pin = pin_packed_tet(&mem, GenerationPin::Current).unwrap();
    let counting = CountingStore::new(mem);
    let cached = CachingStore::new(counting, pin);

    let _ = cached.read_at("CURRENT", 0, 1).unwrap();
    let _ = cached.read_at("CURRENT", 0, 1).unwrap();
    assert_eq!(cached.inner().calls(), 2);
    assert!(cached.cache().unwrap().lock().unwrap().is_empty());
}

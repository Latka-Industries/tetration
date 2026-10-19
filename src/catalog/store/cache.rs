//! Optional in-memory block cache for pinned `packed.tet` range reads (THI-501).
//!
//! Keys are `(generation_id, offset)`. A pin for generation `G` never serves bytes
//! cached under another generation. Cache is opt-in: use the inner store alone, or
//! wrap with [`CachingStore`] / [`CachingStore::without_cache`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::prune::{self, PruneOptions, PruneReport};
use super::{
    CatalogError, ObjectMeta, PackageStore, PinnedPackedTet, PublishGeneration, ReadAt, WritePart,
};

/// In-memory map of downloaded packed ranges, keyed by generation id + byte offset.
#[derive(Debug, Default)]
pub struct BlockCache {
    /// `(generation, offset) → bytes fetched starting at that offset`.
    entries: HashMap<(u64, u64), Vec<u8>>,
}

impl BlockCache {
    /// Empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of stored ranges.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when no ranges are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop every entry.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Look up a range for `generation` at `offset`.
    ///
    /// Hit when a stored blob starts at `offset` and is at least `len` bytes long;
    /// returns a prefix of that length. Shorter stored blobs are treated as a miss.
    #[must_use]
    pub fn get(&self, generation: u64, offset: u64, len: u64) -> Option<Vec<u8>> {
        let want = usize::try_from(len).ok()?;
        let bytes = self.entries.get(&(generation, offset))?;
        if bytes.len() < want {
            return None;
        }
        Some(bytes[..want].to_vec())
    }

    /// Store (or replace) the blob for `(generation, offset)`.
    ///
    /// Replaces an existing entry only when the new blob is at least as long, so a
    /// longer refetch can widen a prior short cache line.
    pub fn insert(&mut self, generation: u64, offset: u64, bytes: Vec<u8>) {
        let key = (generation, offset);
        match self.entries.get(&key) {
            Some(existing) if existing.len() >= bytes.len() => {}
            _ => {
                self.entries.insert(key, bytes);
            }
        }
    }

    /// Whether `(generation, offset)` has a blob of at least `len` bytes.
    #[must_use]
    pub fn contains(&self, generation: u64, offset: u64, len: u64) -> bool {
        self.get(generation, offset, len).is_some()
    }
}

/// [`ReadAt`] / [`PackageStore`] decorator with an optional [`BlockCache`].
///
/// Only ranges against [`PinnedPackedTet::object_key`] are cached, and only under
/// [`PinnedPackedTet::generation`]. Other keys always hit the inner store.
#[derive(Debug)]
pub struct CachingStore<S> {
    inner: S,
    pin: PinnedPackedTet,
    cache: Option<Arc<Mutex<BlockCache>>>,
}

impl<S> CachingStore<S> {
    /// Wrap `inner` with a fresh shared cache for `pin`.
    #[must_use]
    pub fn new(inner: S, pin: PinnedPackedTet) -> Self {
        Self {
            inner,
            pin,
            cache: Some(Arc::new(Mutex::new(BlockCache::new()))),
        }
    }

    /// Wrap `inner` with an existing shared cache (e.g. reuse across queries on the same pin).
    #[must_use]
    pub fn with_cache(inner: S, pin: PinnedPackedTet, cache: Arc<Mutex<BlockCache>>) -> Self {
        Self {
            inner,
            pin,
            cache: Some(cache),
        }
    }

    /// Wrap without caching — every `read_at` goes to `inner` (optional-off path).
    #[must_use]
    pub fn without_cache(inner: S, pin: PinnedPackedTet) -> Self {
        Self {
            inner,
            pin,
            cache: None,
        }
    }

    /// Borrow the inner store.
    #[must_use]
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Consume the wrapper and return the inner store.
    #[must_use]
    pub fn into_inner(self) -> S {
        self.inner
    }

    /// Generation this wrapper pins for cache lookups.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.pin.generation
    }

    /// Shared cache handle when caching is enabled.
    #[must_use]
    pub fn cache(&self) -> Option<&Arc<Mutex<BlockCache>>> {
        self.cache.as_ref()
    }

    /// Prune the inner store, always protecting this wrapper's pinned generation.
    ///
    /// Dry-run unless [`PruneOptions::apply`] is set. Does not write `CURRENT`.
    /// Other open pins still have to be passed in `opts`.
    ///
    /// # Errors
    ///
    /// Same as [`prune::prune_store`].
    pub fn prune(&self, opts: &PruneOptions) -> Result<PruneReport, CatalogError>
    where
        S: PackageStore,
    {
        let mut opts = opts.clone();
        opts.pinned_generations.push(self.pin.generation);
        prune::prune_store(&self.inner, &opts)
    }

    fn is_pinned_packed_key(&self, key: &str) -> bool {
        key == self.pin.object_key
    }
}

impl<S: ReadAt> ReadAt for CachingStore<S> {
    fn read_at(&self, key: &str, offset: u64, len: u64) -> Result<Vec<u8>, CatalogError> {
        if len == 0 {
            return Ok(Vec::new());
        }
        if self.is_pinned_packed_key(key)
            && let Some(cache) = &self.cache
        {
            {
                let guard = cache
                    .lock()
                    .map_err(|_| CatalogError::Package("block cache lock poisoned"))?;
                if let Some(hit) = guard.get(self.pin.generation, offset, len) {
                    return Ok(hit);
                }
            }
            let bytes = self.inner.read_at(key, offset, len)?;
            let mut guard = cache
                .lock()
                .map_err(|_| CatalogError::Package("block cache lock poisoned"))?;
            guard.insert(self.pin.generation, offset, bytes.clone());
            return Ok(bytes);
        }
        self.inner.read_at(key, offset, len)
    }

    fn object_meta(&self, key: &str) -> Result<Option<ObjectMeta>, CatalogError> {
        self.inner.object_meta(key)
    }

    fn list_prefix(&self, prefix: &str) -> Result<Vec<String>, CatalogError> {
        self.inner.list_prefix(prefix)
    }
}

impl<S: WritePart> WritePart for CachingStore<S> {
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

impl<S: PublishGeneration> PublishGeneration for CachingStore<S> {
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

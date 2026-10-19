//! Store prune: staging leftovers + old generations (THI-557).

use std::slice::from_ref;

use crate::catalog::{
    CachingStore, CatalogError, GenerationPin, LocalStore, MemoryStore, PackageStore, PruneOptions,
    PruneRefuseReason, PruneReport, PruneTarget, PublishGeneration, PublishOptions, ReadAt,
    TetDatasetWrite, TetPackage, TetPackagePin, TetShardWriter, WritePart, begin_publish_store,
    heartbeat_key, packed_tet_key, pin_packed_tet, prune_store, publish_lock_exists,
    publish_lock_is_stale, publish_parts, publish_staged_parts, seal_staged_generation, stage_part,
};

fn tiny_tet() -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("part.tet");
    let mut session = crate::catalog::TetWriterSession::create(&path);
    let data: Vec<u8> = [1.0_f32, 2.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    session
        .push_dataset(TetDatasetWrite::f32_row_major("a", &[2], &[2], data).unwrap())
        .unwrap();
    session.commit().unwrap();
    std::fs::read(path).unwrap()
}

fn publish_gens(store: &impl PackageStore, n: u64) {
    let bytes = tiny_tet();
    for generation in 1..=n {
        let part = format!("p{generation}");
        stage_part(store, &format!("s{generation}"), &part, &bytes).unwrap();
        publish_staged_parts(store, generation, &[part]).unwrap();
    }
}

fn packed_exists(store: &impl PackageStore, generation: u64) -> bool {
    store
        .object_meta(&packed_tet_key(generation))
        .unwrap()
        .is_some()
}

fn assert_current_unchanged(store: &impl PackageStore, before: &[u8]) {
    assert_eq!(store.read_all("CURRENT").unwrap(), before);
}

fn assert_gen1_candidate(report: &PruneReport) {
    assert_eq!(report.generations, vec![1]);
    assert!(!report.staging_sessions.is_empty());
    assert!(!report.keys.is_empty());
    assert!(
        report
            .keys
            .iter()
            .all(|key| key.starts_with("staging/") || key.starts_with("generations/00000001/"))
    );
    assert!(!report.keys.iter().any(|key| {
        key == "CURRENT"
            || key.starts_with("generations/00000002/")
            || key.starts_with("generations/00000003/")
    }));
}

#[test]
fn memory_dry_run_keeps_staging_and_old_gens() {
    let store = MemoryStore::new();
    publish_gens(&store, 3);
    let before = store.read_all("CURRENT").unwrap();
    let report = prune_store(&store, &PruneOptions::default()).unwrap();
    assert_gen1_candidate(&report);
    assert!(report.refused.is_empty());
    assert!(packed_exists(&store, 1));
    assert!(!store.list_prefix("staging/").unwrap().is_empty());
    assert_current_unchanged(&store, &before);
    assert_eq!(store.current_generation().unwrap(), Some(3));
}

#[test]
fn memory_apply_drops_staging_and_gen1_keeps_current_and_prior() {
    let store = MemoryStore::new();
    publish_gens(&store, 3);
    let before = store.read_all("CURRENT").unwrap();
    let report = prune_store(
        &store,
        &PruneOptions {
            apply: true,
            ..PruneOptions::default()
        },
    )
    .unwrap();
    assert_gen1_candidate(&report);
    assert!(!packed_exists(&store, 1));
    assert!(packed_exists(&store, 2));
    assert!(packed_exists(&store, 3));
    assert!(store.list_prefix("staging/").unwrap().is_empty());
    assert_current_unchanged(&store, &before);
    assert_eq!(store.current_generation().unwrap(), Some(3));
}

#[test]
fn memory_pin_refuses_old_generation_and_still_drops_unpinned() {
    let store = MemoryStore::new();
    publish_gens(&store, 4);
    let before = store.read_all("CURRENT").unwrap();
    let pin = pin_packed_tet(&store, GenerationPin::Explicit(1)).unwrap();
    let report = prune_store(
        &store,
        &PruneOptions::default()
            .apply()
            .with_store_pins(from_ref(&pin)),
    )
    .unwrap();
    assert_eq!(report.generations, vec![2]);
    assert!(report.refused.iter().any(|refusal| {
        refusal.target == PruneTarget::Generation(1) && refusal.reason == PruneRefuseReason::OpenPin
    }));
    assert!(
        !report
            .keys
            .iter()
            .any(|key| key.starts_with("generations/00000001/"))
    );
    assert!(packed_exists(&store, 1));
    assert!(!packed_exists(&store, 2));
    assert!(packed_exists(&store, 3));
    assert!(packed_exists(&store, 4));
    assert_current_unchanged(&store, &before);
    assert_eq!(store.current_generation().unwrap(), Some(4));
}

#[test]
fn memory_caching_store_pin_is_protected() {
    let store = MemoryStore::new();
    publish_gens(&store, 4);
    let pin = pin_packed_tet(&store, GenerationPin::Explicit(1)).unwrap();
    let cached = CachingStore::new(store, pin);
    let before = cached.inner().read_all("CURRENT").unwrap();
    let report = cached
        .prune(&PruneOptions {
            apply: true,
            ..PruneOptions::default()
        })
        .unwrap();
    assert_eq!(report.generations, vec![2]);
    assert!(packed_exists(cached.inner(), 1));
    assert!(!packed_exists(cached.inner(), 2));
    assert!(packed_exists(cached.inner(), 3));
    assert!(packed_exists(cached.inner(), 4));
    assert_eq!(cached.inner().read_all("CURRENT").unwrap(), before);
}

#[test]
fn memory_active_claim_refuses_staging_and_does_not_flip_current() {
    let store = MemoryStore::new();
    publish_gens(&store, 3);
    let bytes = tiny_tet();
    stage_part(&store, "live", "plive", &bytes).unwrap();
    begin_publish_store(&store, 4, &PublishOptions::default()).unwrap();
    let before = store.read_all("CURRENT").unwrap();
    let report = prune_store(
        &store,
        &PruneOptions {
            apply: true,
            ..PruneOptions::default()
        },
    )
    .unwrap();
    assert_eq!(report.generations, vec![1]);
    assert!(report.staging_sessions.is_empty());
    assert!(report.refused.iter().any(|refusal| {
        matches!(refusal.target, PruneTarget::StagingSession(_))
            && refusal.reason == PruneRefuseReason::ActivePublishClaim
    }));
    assert!(!store.list_prefix("staging/").unwrap().is_empty());
    assert!(publish_lock_exists(&store, 4).unwrap());
    assert!(!packed_exists(&store, 1));
    assert!(packed_exists(&store, 2));
    assert!(packed_exists(&store, 3));
    assert_current_unchanged(&store, &before);
}

#[test]
fn memory_stale_claim_does_not_keep_staging() {
    let store = MemoryStore::new();
    publish_gens(&store, 3);
    begin_publish_store(&store, 4, &PublishOptions::default()).unwrap();
    store.write_part(&heartbeat_key(4), b"0\n").unwrap();
    assert!(publish_lock_is_stale(&store, 4, crate::catalog::DEFAULT_PUBLISH_LOCK_STALE).unwrap());
    let report = prune_store(
        &store,
        &PruneOptions {
            apply: true,
            ..PruneOptions::default()
        },
    )
    .unwrap();
    assert!(!report.staging_sessions.is_empty());
    assert!(store.list_prefix("staging/").unwrap().is_empty());
    assert!(publish_lock_exists(&store, 4).unwrap());
    assert_eq!(store.current_generation().unwrap(), Some(3));
}

#[test]
fn memory_keeps_current_previous_and_newer_sealed() {
    let store = MemoryStore::new();
    publish_gens(&store, 1);
    seal_staged_generation(&store, 2, &[String::from("p1")]).unwrap();
    assert_eq!(store.current_generation().unwrap(), Some(1));
    let report = prune_store(
        &store,
        &PruneOptions {
            apply: true,
            ..PruneOptions::default()
        },
    )
    .unwrap();
    assert!(report.generations.is_empty());
    assert!(packed_exists(&store, 1));
    assert!(packed_exists(&store, 2));
    assert!(store.list_prefix("staging/").unwrap().is_empty());
    assert_eq!(store.current_generation().unwrap(), Some(1));

    let retain = MemoryStore::new();
    publish_gens(&retain, 2);
    let report = prune_store(
        &retain,
        &PruneOptions {
            apply: true,
            ..PruneOptions::default()
        },
    )
    .unwrap();
    assert!(report.generations.is_empty());
    assert!(packed_exists(&retain, 1));
    assert!(packed_exists(&retain, 2));
    assert_eq!(retain.current_generation().unwrap(), Some(2));
}

#[test]
fn memory_without_current_deletes_nothing() {
    let store = MemoryStore::new();
    let bytes = tiny_tet();
    stage_part(&store, "s", "p", &bytes).unwrap();
    let err = prune_store(&store, &PruneOptions::default()).unwrap_err();
    assert!(matches!(err, CatalogError::NoCurrentGeneration));
    assert!(!store.list_prefix("staging/").unwrap().is_empty());
}

#[test]
fn local_store_apply_matches_memory() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("pkg.tet.d");
    std::fs::create_dir_all(&root).unwrap();
    let store = LocalStore::new(root);
    publish_gens(&store, 3);
    let before = store.read_all("CURRENT").unwrap();
    let dry = prune_store(&store, &PruneOptions::default()).unwrap();
    assert_gen1_candidate(&dry);
    assert!(packed_exists(&store, 1));
    let report = prune_store(
        &store,
        &PruneOptions {
            apply: true,
            ..PruneOptions::default()
        },
    )
    .unwrap();
    assert_eq!(report.generations, vec![1]);
    assert!(!packed_exists(&store, 1));
    assert!(packed_exists(&store, 2));
    assert!(packed_exists(&store, 3));
    assert!(store.list_prefix("staging/").unwrap().is_empty());
    assert_current_unchanged(&store, &before);
}

fn publish_named(pkg: &TetPackage, generation: u64, session: &str, part: &str, name: &str) {
    let mut writer = TetShardWriter::create(pkg, session, part).unwrap();
    let data: Vec<u8> = [1.0_f32, 2.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    writer
        .push_dataset(TetDatasetWrite::f32_row_major(name, &[2], &[2], data).unwrap())
        .unwrap();
    writer.seal().unwrap();
    publish_parts(pkg, generation, &[part.into()]).unwrap();
}

#[test]
fn local_package_dry_run_then_apply() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("apply.tet.d")).unwrap();
    publish_named(&pkg, 1, "s1", "p1", "a");
    publish_named(&pkg, 2, "s2", "p2", "b");
    publish_named(&pkg, 3, "s3", "p3", "c");
    let before = pkg.store().read_all("CURRENT").unwrap();
    let dry = prune_store(pkg.store(), &PruneOptions::default()).unwrap();
    assert_eq!(dry.generations, vec![1]);
    assert!(!dry.staging_sessions.is_empty());
    assert!(pkg.generation_dir(1).is_dir());

    let applied = prune_store(
        pkg.store(),
        &PruneOptions {
            apply: true,
            ..PruneOptions::default()
        },
    )
    .unwrap();
    assert_eq!(applied.generations, vec![1]);
    assert!(!pkg.generation_dir(1).exists());
    assert!(pkg.generation_dir(2).is_dir());
    assert!(pkg.generation_dir(3).is_dir());
    assert_eq!(pkg.current().unwrap(), Some(3));
    assert_eq!(pkg.store().read_all("CURRENT").unwrap(), before);
    assert!(pkg.store().object_meta("PROTOCOL").unwrap().is_some());
}

#[test]
fn local_package_pin_blocks_that_generation() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("pin.tet.d")).unwrap();
    publish_named(&pkg, 1, "s1", "p1", "a");
    publish_named(&pkg, 2, "s2", "p2", "b");
    publish_named(&pkg, 3, "s3", "p3", "c");
    publish_named(&pkg, 4, "s4", "p4", "d");
    let pin: TetPackagePin = pkg.pin_generation(1).unwrap();
    let before = pkg.store().read_all("CURRENT").unwrap();
    let report = prune_store(
        pkg.store(),
        &PruneOptions::default()
            .apply()
            .with_package_pins(from_ref(&pin)),
    )
    .unwrap();
    assert_eq!(report.generations, vec![2]);
    assert!(report.refused.iter().any(|refusal| {
        refusal.target == PruneTarget::Generation(1) && refusal.reason == PruneRefuseReason::OpenPin
    }));
    assert!(pkg.generation_dir(1).is_dir());
    assert!(!pkg.generation_dir(2).exists());
    assert!(pkg.generation_dir(3).is_dir());
    assert!(pkg.generation_dir(4).is_dir());
    assert_eq!(pkg.current().unwrap(), Some(4));
    assert_eq!(pkg.store().read_all("CURRENT").unwrap(), before);
    assert_eq!(pin.generation(), 1);
}

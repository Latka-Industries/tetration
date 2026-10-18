//! Package GC: staging leftovers + old generations (THI-521).

use crate::catalog::{GcOptions, TetDatasetWrite, TetPackage, TetShardWriter, publish_parts};
use crate::verify::{VerifySeverity, verify_tet_package};

fn f32_le(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn seal_named(pkg: &TetPackage, session: &str, part: &str, name: &str, vals: &[f32]) {
    let mut w = TetShardWriter::create(pkg, session, part).unwrap();
    w.push_dataset(TetDatasetWrite::f32_row_major(name, &[2], &[2], f32_le(vals)).unwrap())
        .unwrap();
    w.seal().unwrap();
}

fn publish_named(pkg: &TetPackage, generation: u64, session: &str, part: &str, name: &str) {
    seal_named(pkg, session, part, name, &[1.0, 2.0]);
    publish_parts(pkg, generation, &[part.into()]).unwrap();
}

#[test]
fn gc_dry_run_keeps_staging_and_old_gens() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("dry.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    publish_named(&pkg, 1, "s1", "p1", "a");
    publish_named(&pkg, 2, "s2", "p2", "b");
    publish_named(&pkg, 3, "s3", "p3", "c");
    // publish leaves staging sessions; ensure at least one remains.
    assert!(
        pkg.staging_dir()
            .read_dir()
            .unwrap()
            .filter_map(Result::ok)
            .any(|e| e.path().is_dir())
    );

    let plan = pkg.gc(&GcOptions { apply: false }).unwrap();
    assert!(!plan.staging_sessions.is_empty());
    assert_eq!(plan.generations, vec![1]);
    assert!(pkg.generation_dir(1).is_dir());
    assert!(pkg.staging_dir().read_dir().unwrap().next().is_some());
}

#[test]
fn gc_apply_drops_staging_and_gen1_keeps_current_and_prior() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("apply.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    publish_named(&pkg, 1, "s1", "p1", "a");
    publish_named(&pkg, 2, "s2", "p2", "b");
    publish_named(&pkg, 3, "s3", "p3", "c");
    assert_eq!(pkg.current().unwrap(), Some(3));

    let plan = pkg.gc(&GcOptions { apply: true }).unwrap();
    assert_eq!(plan.generations, vec![1]);
    assert!(!plan.staging_sessions.is_empty());
    assert!(!pkg.generation_dir(1).exists());
    assert!(pkg.generation_dir(2).is_dir());
    assert!(pkg.generation_dir(3).is_dir());
    assert_eq!(pkg.current().unwrap(), Some(3));
    assert!(
        pkg.staging_dir()
            .read_dir()
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().is_dir())
            .count()
            == 0
    );

    let report = verify_tet_package(&root).unwrap();
    assert!(report.ok, "{:?}", report.findings);
    assert!(
        !report
            .findings
            .iter()
            .any(|f| { f.check == "staging" && f.severity == VerifySeverity::Warning && !f.ok }),
        "staging warn should be gone: {:?}",
        report.findings
    );
}

#[test]
fn gc_never_plans_current_or_current_minus_one() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("retain.tet.d")).unwrap();
    publish_named(&pkg, 1, "s1", "p1", "a");
    publish_named(&pkg, 2, "s2", "p2", "b");
    let plan = pkg.plan_gc().unwrap();
    assert!(plan.generations.is_empty());
    pkg.gc(&GcOptions { apply: true }).unwrap();
    assert!(pkg.generation_dir(1).is_dir());
    assert!(pkg.generation_dir(2).is_dir());
}

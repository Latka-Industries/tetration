//! Localhost N-process distributed fold + dense materialize (THI-618 / THI-617).

use std::path::PathBuf;

use tetration::catalog::{TetDatasetWrite, TetWriterSession};
use tetration::layout::mmap_file_read;
use tetration::query::distributed::{
    MultiprocessDenseOpts, MultiprocessFoldOpts, ShardPartition,
    execute_sharded_scalar_fold_multiprocess, execute_sharded_tier_c_multiprocess,
};
use tetration::query::{
    SpillPathAllowlist, parse_query_json, plan_query_with_tet_mmap_ex, validate_query,
};

fn tet_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tet"))
}

fn write_tiny_f32(path: &std::path::Path) {
    let shape = [2u64, 3];
    let chunk = [2u64, 2];
    let mut data = vec![0u8; 24];
    for (slot, n) in data.as_chunks_mut::<4>().0.iter_mut().zip(1_u8..=6) {
        slot.copy_from_slice(&f32::from(n).to_le_bytes());
    }
    let mut session = TetWriterSession::create(path);
    session
        .push_dataset(TetDatasetWrite::f32_row_major("a", &shape, &chunk, data).expect("dataset"))
        .expect("push");
    session.commit().expect("commit");
}

#[test]
fn multiprocess_sum_matches_monolithic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mp_sum.tet");
    write_tiny_f32(&path);

    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let mono = plan_query_with_tet_mmap_ex(&doc, None, &mmap, Some(0), None).unwrap();
    let mono_sum = mono.execution.as_ref().unwrap().operation_sum.unwrap();
    let plan = mono.read_plan.as_ref().unwrap();
    let op = doc.operation.as_ref().unwrap();
    let worker_bin = tet_bin();

    let mp = execute_sharded_scalar_fold_multiprocess(MultiprocessFoldOpts {
        tet_path: &path,
        plan,
        doc: &doc,
        op,
        n_shards: 3,
        partition: ShardPartition::ContiguousRanges,
        write: None,
        worker_bin: &worker_bin,
    })
    .unwrap();
    assert!(
        (mp.operation_sum.unwrap() - mono_sum).abs() < 1e-9,
        "multiprocess sum {} != mono {}",
        mp.operation_sum.unwrap(),
        mono_sum
    );
}

#[test]
fn multiprocess_median_matches_monolithic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mp_median.tet");
    write_tiny_f32(&path);

    let doc = parse_query_json(r#"{"dataset":"a","median":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let policy = SpillPathAllowlist::default_for_tet(&path, std::iter::empty::<PathBuf>()).unwrap();
    let tet = path.to_str().unwrap();
    let mono = plan_query_with_tet_mmap_ex(&doc, Some(tet), &mmap, Some(0), Some(&policy)).unwrap();
    let want = mono.execution.as_ref().unwrap().operation_median.unwrap();
    let plan = mono.read_plan.as_ref().unwrap();
    let op = doc.operation.as_ref().unwrap();
    let worker_bin = tet_bin();

    let mp = execute_sharded_tier_c_multiprocess(
        MultiprocessDenseOpts {
            tet_path: &path,
            plan,
            doc: &doc,
            n_shards: 3,
            partition: ShardPartition::ContiguousRanges,
            write: None,
            worker_bin: &worker_bin,
        },
        op,
    )
    .unwrap();
    assert!(
        (mp.operation_median.unwrap() - want).abs() < 1e-9,
        "multiprocess median {} != mono {want}",
        mp.operation_median.unwrap()
    );
}

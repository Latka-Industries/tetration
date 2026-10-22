//! Distributed fold + dense materialize / transform harness (THI-576 / THI-617).

use std::path::PathBuf;

use super::fixture::write_multichunk_2x3_tiles;
use crate::layout::mmap_file_read;
use crate::query::distributed::{
    FoldPartialState, ShardPartition, TransformRamLocalOpts, WORKER_PROTOCOL_V1, WorkerJobV1,
    WorkerPartialV1, WorkerPinV1, execute_sharded_materialize_local,
    execute_sharded_scalar_fold_local, execute_sharded_tier_c_local,
    execute_sharded_transform_ram_local, execute_worker_job_v1, operation_is_dense_distributable,
    operation_is_shard_mergeable, plan_chunk_shards, refuse_distributed_spill, split_read_plan,
};
use crate::query::embed_materialize::{
    DenseBuffer, materialize_query_selection, materialize_query_transform_ram,
};
use crate::query::engine::plan_read_for_document;
use crate::query::types::{Operation, WriteHints, WriteTarget};
use crate::query::{
    SpillPathAllowlist, parse_query_json, plan_query_with_tet_mmap_ex, validate_query,
};
use crate::utils::dtype::ElementDtype;

fn spill_policy(path: &std::path::Path) -> SpillPathAllowlist {
    SpillPathAllowlist::default_for_tet(path, std::iter::empty::<PathBuf>()).unwrap()
}

#[test]
fn sharded_sum_matches_monolithic_contiguous_and_interleaved() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dist_sum.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let mono = plan_query_with_tet_mmap_ex(&doc, None, &mmap, Some(0), None).unwrap();
    let mono_sum = mono.execution.as_ref().unwrap().operation_sum.unwrap();
    let plan = mono.read_plan.as_ref().unwrap();
    let op = doc.operation.as_ref().unwrap();

    for part in [
        ShardPartition::ContiguousRanges,
        ShardPartition::Interleaved,
    ] {
        for n_shards in [1usize, 2, 3, 5] {
            let sharded = execute_sharded_scalar_fold_local(
                &mmap,
                plan,
                op,
                ElementDtype::F32,
                n_shards,
                part,
                None,
            )
            .unwrap();
            let got = sharded.operation_sum.unwrap();
            assert!(
                (got - mono_sum).abs() < 1e-9,
                "part={part:?} n={n_shards} got={got} want={mono_sum}"
            );
        }
    }
}

#[test]
fn sharded_mean_var_min_argmin_match_monolithic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dist_ops.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();

    let cases = [
        r#"{"dataset":"a","mean":[]}"#,
        r#"{"dataset":"a","var":[]}"#,
        r#"{"dataset":"a","min":[]}"#,
        r#"{"dataset":"a","arg_min":[]}"#,
    ];
    for json in cases {
        let doc = parse_query_json(json).unwrap();
        validate_query(&doc).unwrap();
        let mono = plan_query_with_tet_mmap_ex(&doc, None, &mmap, Some(0), None).unwrap();
        let ex = mono.execution.as_ref().unwrap();
        let plan = mono.read_plan.as_ref().unwrap();
        let op = doc.operation.as_ref().unwrap();
        let sharded = execute_sharded_scalar_fold_local(
            &mmap,
            plan,
            op,
            ElementDtype::F32,
            3,
            ShardPartition::ContiguousRanges,
            None,
        )
        .unwrap();
        match op {
            Operation::Mean { .. } => {
                assert!(
                    (ex.operation_mean.unwrap() - sharded.operation_mean.unwrap()).abs() < 1e-9
                );
            }
            Operation::Var { .. } => {
                assert!((ex.operation_var.unwrap() - sharded.operation_var.unwrap()).abs() < 1e-9);
            }
            Operation::Min { .. } => {
                assert!((ex.operation_min.unwrap() - sharded.operation_min.unwrap()).abs() < 1e-9);
            }
            Operation::ArgMin { .. } => {
                assert_eq!(ex.operation_argmin_index, sharded.operation_argmin_index);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn split_read_plan_preserves_geometry_and_subsets_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("split.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let planned = plan_query_with_tet_mmap_ex(&doc, None, &mmap, Some(0), None).unwrap();
    let plan = planned.read_plan.as_ref().unwrap();
    assert!(plan.chunk_count >= 2);
    let specs = plan_chunk_shards(plan.chunks.len(), 2, ShardPartition::ContiguousRanges);
    let left = split_read_plan(plan, &specs[0]);
    assert_eq!(left.logical_element_count, plan.logical_element_count);
    assert_eq!(left.selection_box_start, plan.selection_box_start);
    assert_eq!(left.chunk_count, specs[0].chunk_indices.len());
    assert_eq!(left.chunks.len(), specs[0].chunk_indices.len());
}

#[test]
fn tier_c_not_shard_mergeable_but_dense_distributable() {
    let median = Operation::Median { axes: vec![] };
    assert!(!operation_is_shard_mergeable(&median));
    assert!(operation_is_dense_distributable(&median));
    assert!(operation_is_shard_mergeable(&Operation::Sum {
        axes: vec![]
    }));
    assert!(!operation_is_dense_distributable(&Operation::Sum {
        axes: vec![]
    }));
}

#[test]
fn refuse_spill_write_targets() {
    refuse_distributed_spill(None).unwrap();
    refuse_distributed_spill(Some(&WriteHints {
        target: WriteTarget::Ram,
        path: None,
        timestamp: None,
    }))
    .unwrap();
    let err = refuse_distributed_spill(Some(&WriteHints {
        target: WriteTarget::Spill,
        path: None,
        timestamp: None,
    }))
    .unwrap_err();
    assert!(err.to_string().contains("spill"));
}

#[test]
fn worker_job_partial_json_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("job.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let planned = plan_query_with_tet_mmap_ex(&doc, None, &mmap, Some(0), None).unwrap();
    let plan = planned.read_plan.as_ref().unwrap().clone();
    let job = WorkerJobV1 {
        protocol: WORKER_PROTOCOL_V1,
        job_id: "j1".into(),
        shard_id: 0,
        n_shards: 1,
        pin: WorkerPinV1 {
            generation: 1,
            object_key: "generations/00000001/packed.tet".into(),
            object_len: u64::try_from(mmap.len()).unwrap(),
            package_uri: path.display().to_string(),
        },
        document: doc,
        chunk_indices: (0..plan.chunks.len()).collect(),
        plan,
        preview_max: 0,
    };
    let json = serde_json::to_string(&job).unwrap();
    let back: WorkerJobV1 = serde_json::from_str(&json).unwrap();
    assert_eq!(back.protocol, WORKER_PROTOCOL_V1);
    assert_eq!(back.shard_id, 0);
    assert_eq!(back.pin.generation, 1);

    let partial = WorkerPartialV1 {
        protocol: WORKER_PROTOCOL_V1,
        job_id: "j1".into(),
        shard_id: 0,
        pin_generation: 1,
        partial: FoldPartialState::empty(),
        dense: None,
    };
    let pjson = serde_json::to_string(&partial).unwrap();
    let pback: WorkerPartialV1 = serde_json::from_str(&pjson).unwrap();
    assert_eq!(pback.partial.bytes_read(), 0);
}

#[test]
fn execute_worker_job_v1_sum_partial() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("worker_job.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let planned = plan_query_with_tet_mmap_ex(&doc, None, &mmap, Some(0), None).unwrap();
    let plan = planned.read_plan.as_ref().unwrap().clone();
    let job = WorkerJobV1 {
        protocol: WORKER_PROTOCOL_V1,
        job_id: "j1".into(),
        shard_id: 0,
        n_shards: 1,
        pin: WorkerPinV1 {
            generation: 0,
            object_key: String::new(),
            object_len: u64::try_from(mmap.len()).unwrap(),
            package_uri: path.display().to_string(),
        },
        document: doc,
        chunk_indices: (0..plan.chunks.len()).collect(),
        plan,
        preview_max: 0,
    };
    let partial = execute_worker_job_v1(&job).unwrap();
    assert_eq!(partial.shard_id, 0);
    assert!(partial.partial.bytes_read() > 0);
    assert!(partial.dense.is_none());
}

#[test]
fn sharded_materialize_matches_monolithic_selection() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dist_mat.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let mono = materialize_query_selection(&doc, &mmap).unwrap();
    let DenseBuffer::F32(want) = mono.buffer else {
        panic!("expected f32");
    };
    let planned = plan_read_for_document(&doc, &mmap).unwrap();
    let plan = &planned.read_plan;

    for n_shards in [1usize, 2, 3, 5] {
        let (got, _) = execute_sharded_materialize_local(
            &mmap,
            plan,
            ElementDtype::F32,
            n_shards,
            ShardPartition::ContiguousRanges,
            None,
        )
        .unwrap();
        assert_eq!(got, want, "n_shards={n_shards}");
    }
}

#[test]
fn sharded_median_matches_monolithic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dist_median.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","median":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let policy = spill_policy(&path);
    let tet = path.to_str().unwrap();
    let mono = plan_query_with_tet_mmap_ex(&doc, Some(tet), &mmap, Some(0), Some(&policy)).unwrap();
    let want = mono.execution.as_ref().unwrap().operation_median.unwrap();
    let plan = mono.read_plan.as_ref().unwrap();
    let op = doc.operation.as_ref().unwrap();

    let sharded = execute_sharded_tier_c_local(
        &mmap,
        plan,
        op,
        ElementDtype::F32,
        3,
        ShardPartition::Interleaved,
        None,
    )
    .unwrap();
    assert!(
        (sharded.operation_median.unwrap() - want).abs() < 1e-9,
        "got {:?} want {want}",
        sharded.operation_median
    );
}

#[test]
fn sharded_transform_zscore_ram_matches_monolithic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dist_zscore.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","transform":{"method":"zscore"},"write":"ram"}"#)
        .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let mono = materialize_query_transform_ram(&doc, &mmap, &path, None).unwrap();
    let DenseBuffer::F32(want) = mono.buffer else {
        panic!("expected f32");
    };
    let planned = plan_read_for_document(&doc, &mmap).unwrap();
    let plan = &planned.read_plan;
    let op = doc.operation.as_ref().unwrap();
    let write = doc.write.as_ref();

    let (got, _) = execute_sharded_transform_ram_local(TransformRamLocalOpts {
        mmap: &mmap,
        plan,
        op,
        dtype: ElementDtype::F32,
        n_shards: 3,
        partition: ShardPartition::ContiguousRanges,
        write,
        tet_path: Some(&path),
    })
    .unwrap();
    assert_eq!(got.len(), want.len());
    for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
        assert!((a - b).abs() < 1e-5, "index {i}: sharded {a} != mono {b}");
    }
}

#[test]
fn execute_worker_job_v1_dense_median_shard() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("worker_dense.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","median":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let planned = plan_read_for_document(&doc, &mmap).unwrap();
    let plan = planned.read_plan.clone();
    let job = WorkerJobV1 {
        protocol: WORKER_PROTOCOL_V1,
        job_id: "j-dense".into(),
        shard_id: 0,
        n_shards: 1,
        pin: WorkerPinV1 {
            generation: 0,
            object_key: String::new(),
            object_len: u64::try_from(mmap.len()).unwrap(),
            package_uri: path.display().to_string(),
        },
        document: doc,
        chunk_indices: (0..plan.chunks.len()).collect(),
        plan,
        preview_max: 0,
    };
    let partial = execute_worker_job_v1(&job).unwrap();
    let dense = partial.dense.as_ref().expect("dense payload");
    assert!(!dense.indices.is_empty());
    assert_eq!(dense.indices.len(), dense.values_f32.len());
}

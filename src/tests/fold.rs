//! Fold I/O policy tests.

use crate::query::engine::budget::ExecutionBudget;
use crate::query::fold::fold_policy::{
    FoldIoPolicy, IoRegime, PARALLEL_MIN_LOGICAL_BYTES, chunk_indices_for_fold, resolve_io_regime,
};
use crate::query::fold::linear_scan;
use crate::query::types::{
    CellPredicate, ChunkTouchPolicy, ExecutionHints, LabelEqPred, PlannedChunkIo, ReadPlan,
    ValueCompareOp, ValueFilter, ValuePredicate,
};
use crate::utils::dtype::ElementDtype;

fn sample_plan(chunks: Vec<PlannedChunkIo>, shape: u64) -> ReadPlan {
    ReadPlan {
        chunk_touch_policy: ChunkTouchPolicy::DenseHalfOpenUnitStep,
        chunk_count: chunks.len(),
        zone_filter_skipped_chunks: 0,
        index_filter_skipped_chunks: 0,
        total_stored_bytes: 0,
        chunks,
        dataset_shape: vec![shape],
        chunk_shape: vec![shape],
        selection_box_start: vec![0],
        selection_box_stop_exclusive: vec![shape],
        selection_step: vec![1],
        logical_selection_shape: vec![shape],
        logical_element_count: shape as usize,
        cell_filter: None,
        label_eq: None,
        label_group_remap: None,
        dataset_fill: None,
    }
}

#[test]
fn full_selection_detected() {
    let plan = sample_plan(vec![], 1024);
    assert!(FoldIoPolicy::is_dense_unit_step_full_selection(&plan));
}

#[test]
fn partial_selection_not_full_dense() {
    let mut plan = sample_plan(vec![], 1024);
    plan.selection_box_stop_exclusive = vec![512];
    assert!(!FoldIoPolicy::is_dense_unit_step_full_selection(&plan));
}

#[test]
fn out_of_core_when_logical_exceeds_headroom() {
    let budget = ExecutionBudget {
        memory_budget_bytes: 1,
        host_available_ram_bytes: Some(10 * 1024 * 1024 * 1024),
        memory_budget_percent_bps: 2500,
    };
    let logical = 20 * 1024 * 1024 * 1024;
    assert_eq!(resolve_io_regime(&budget, logical), IoRegime::OutOfCore);
}

#[test]
fn in_core_when_logical_fits_headroom() {
    let budget = ExecutionBudget {
        memory_budget_bytes: 1,
        host_available_ram_bytes: Some(10 * 1024 * 1024 * 1024),
        memory_budget_percent_bps: 2500,
    };
    let logical = 6 * 1024 * 1024 * 1024;
    assert_eq!(resolve_io_regime(&budget, logical), IoRegime::InCore);
}

#[test]
fn chunk_indices_sorted_by_payload_offset() {
    let chunks = vec![
        PlannedChunkIo {
            chunk_index: vec![2],
            payload_offset: 200,
            stored_byte_len: 10,
            raw_byte_len: 10,
            codec: 0,
            zone: None,
        },
        PlannedChunkIo {
            chunk_index: vec![0],
            payload_offset: 0,
            stored_byte_len: 10,
            raw_byte_len: 10,
            codec: 0,
            zone: None,
        },
        PlannedChunkIo {
            chunk_index: vec![1],
            payload_offset: 100,
            stored_byte_len: 10,
            raw_byte_len: 10,
            codec: 0,
            zone: None,
        },
    ];
    let plan = sample_plan(chunks, 3);
    let idx = chunk_indices_for_fold(&plan, true);
    assert_eq!(idx, vec![1, 2, 0]);
}

fn two_chunks() -> Vec<PlannedChunkIo> {
    vec![
        PlannedChunkIo {
            chunk_index: vec![0],
            payload_offset: 0,
            stored_byte_len: 64,
            raw_byte_len: 64,
            codec: 0,
            zone: None,
        },
        PlannedChunkIo {
            chunk_index: vec![1],
            payload_offset: 64,
            stored_byte_len: 64,
            raw_byte_len: 64,
            codec: 0,
            zone: None,
        },
    ]
}

/// f32 element count that meets [`PARALLEL_MIN_LOGICAL_BYTES`].
fn parallel_min_f32_elems() -> u64 {
    PARALLEL_MIN_LOGICAL_BYTES.div_ceil(4)
}

#[test]
fn policy_parallel_default_all_regimes() {
    let budget = ExecutionBudget {
        memory_budget_bytes: 1,
        host_available_ram_bytes: Some(10 * 1024 * 1024 * 1024),
        memory_budget_percent_bps: 2500,
    };
    let plan = sample_plan(two_chunks(), parallel_min_f32_elems());
    let policy =
        FoldIoPolicy::resolve(&plan, &budget, None, ElementDtype::F32, true).expect("resolve");
    assert_eq!(policy.io_regime, IoRegime::InCore);
    assert!(policy.parallel);
    assert!(!policy.sequential_io);
    assert_eq!(policy.fold_workers, None);

    // Out-of-core linear_scan needs payload bytes == logical bytes; keep a tiny
    // contiguous plan for that check (independent of the parallel size floor).
    let tight_budget = ExecutionBudget {
        memory_budget_bytes: 1,
        host_available_ram_bytes: Some(100),
        memory_budget_percent_bps: 2500,
    };
    let tiny = sample_plan(two_chunks(), 32); // 128 B f32, matches two×64 payloads
    let policy = FoldIoPolicy::resolve(&tiny, &tight_budget, None, ElementDtype::F32, true)
        .expect("resolve");
    assert_eq!(policy.io_regime, IoRegime::OutOfCore);
    assert!(policy.linear_scan);
    assert!(!policy.parallel);
    assert!(!policy.sequential_io);
    assert_eq!(policy.fold_workers, None);
}

#[test]
fn policy_out_of_core_skips_linear_scan_when_label_eq() {
    let tight_budget = ExecutionBudget {
        memory_budget_bytes: 1,
        host_available_ram_bytes: Some(100),
        memory_budget_percent_bps: 2500,
    };
    let mut plan = sample_plan(two_chunks(), 32);
    plan.label_eq = Some(LabelEqPred {
        axis: 0,
        matching: vec![true, false],
    });
    let policy = FoldIoPolicy::resolve(&plan, &tight_budget, None, ElementDtype::F32, true)
        .expect("resolve");
    assert_eq!(policy.io_regime, IoRegime::OutOfCore);
    assert!(
        !policy.linear_scan,
        "label_eq must not use byte-stream fold (would ignore position filter)"
    );
}

#[test]
fn policy_out_of_core_skips_linear_scan_when_cell_filter() {
    let tight_budget = ExecutionBudget {
        memory_budget_bytes: 1,
        host_available_ram_bytes: Some(100),
        memory_budget_percent_bps: 2500,
    };
    let mut plan = sample_plan(two_chunks(), 32);
    plan.cell_filter = Some(CellPredicate::from_filter(
        &ValueFilter {
            predicates: vec![ValuePredicate::Compare {
                op: ValueCompareOp::Gt,
                value: 0.0,
            }],
        },
        None,
    ));
    let policy = FoldIoPolicy::resolve(&plan, &tight_budget, None, ElementDtype::F32, true)
        .expect("resolve");
    assert!(!policy.linear_scan);
}

#[test]
fn policy_out_of_core_skips_linear_scan_when_disallowed() {
    // Window folds pass allows_linear_scan=false so OOC keeps parallel chunks
    // instead of single-threaded sequential scatter (no window linear path).
    let tight_budget = ExecutionBudget {
        memory_budget_bytes: 1,
        host_available_ram_bytes: Some(100),
        memory_budget_percent_bps: 2500,
    };
    let elems = parallel_min_f32_elems();
    let raw_per = (elems * 4) / 2;
    let chunks = vec![
        PlannedChunkIo {
            chunk_index: vec![0],
            payload_offset: 0,
            stored_byte_len: raw_per,
            raw_byte_len: raw_per,
            codec: 0,
            zone: None,
        },
        PlannedChunkIo {
            chunk_index: vec![1],
            payload_offset: raw_per,
            stored_byte_len: raw_per,
            raw_byte_len: raw_per,
            codec: 0,
            zone: None,
        },
    ];
    let plan = sample_plan(chunks, elems);
    let with_linear = FoldIoPolicy::resolve(&plan, &tight_budget, None, ElementDtype::F32, true)
        .expect("resolve");
    assert!(with_linear.linear_scan);
    assert!(!with_linear.parallel);

    let windowish = FoldIoPolicy::resolve(&plan, &tight_budget, None, ElementDtype::F32, false)
        .expect("resolve");
    assert_eq!(windowish.io_regime, IoRegime::OutOfCore);
    assert!(
        !windowish.linear_scan,
        "ops without a linear_scan path must not select it"
    );
    assert!(
        windowish.parallel,
        "OOC window-sized selection should keep Rayon chunk fold"
    );
}

#[test]
fn policy_tiny_multi_chunk_stays_sequential_by_default() {
    let budget = ExecutionBudget {
        memory_budget_bytes: 1,
        host_available_ram_bytes: Some(10 * 1024 * 1024 * 1024),
        memory_budget_percent_bps: 2500,
    };
    // 16 MiB f32 selection — below PARALLEL_MIN_LOGICAL_BYTES.
    let plan = sample_plan(two_chunks(), (16 * 1024 * 1024) / 4);
    let policy =
        FoldIoPolicy::resolve(&plan, &budget, None, ElementDtype::F32, true).expect("resolve");
    assert!(!policy.parallel);
    assert!(!policy.linear_scan);

    let hints = ExecutionHints {
        memory_budget_bytes: None,
        memory_budget_percent_bps: None,
        fold_parallel: Some(true),
        device: None,
    };
    let forced = FoldIoPolicy::resolve(&plan, &budget, Some(&hints), ElementDtype::F32, true)
        .expect("resolve");
    assert!(forced.parallel);
}

#[test]
fn detect_contiguous_raw_span_requires_adjacent_payloads() {
    let chunks = vec![
        PlannedChunkIo {
            chunk_index: vec![0],
            payload_offset: 0,
            stored_byte_len: 8,
            raw_byte_len: 8,
            codec: 0,
            zone: None,
        },
        PlannedChunkIo {
            chunk_index: vec![1],
            payload_offset: 16,
            stored_byte_len: 8,
            raw_byte_len: 8,
            codec: 0,
            zone: None,
        },
    ];
    let plan = sample_plan(chunks, 4);
    assert!(linear_scan::detect_contiguous_raw_span(&plan, 4).is_none());
}

#[test]
fn detect_contiguous_raw_span_ok_for_sequential_raw() {
    let chunks = vec![
        PlannedChunkIo {
            chunk_index: vec![0],
            payload_offset: 100,
            stored_byte_len: 8,
            raw_byte_len: 8,
            codec: 0,
            zone: None,
        },
        PlannedChunkIo {
            chunk_index: vec![1],
            payload_offset: 108,
            stored_byte_len: 8,
            raw_byte_len: 8,
            codec: 0,
            zone: None,
        },
    ];
    let plan = sample_plan(chunks, 4);
    let span = linear_scan::detect_contiguous_raw_span(&plan, 4).expect("span");
    assert_eq!(span.start, 100);
    assert_eq!(span.len, 16);
}

#[test]
fn policy_in_core_keeps_parallel_not_linear_scan() {
    let budget = ExecutionBudget {
        memory_budget_bytes: 1,
        host_available_ram_bytes: Some(10 * 1024 * 1024 * 1024),
        memory_budget_percent_bps: 2500,
    };
    let plan = sample_plan(two_chunks(), parallel_min_f32_elems());
    let policy =
        FoldIoPolicy::resolve(&plan, &budget, None, ElementDtype::F32, true).expect("resolve");
    assert!(!policy.linear_scan);
    assert!(policy.parallel);
}

#[test]
fn policy_fold_parallel_false_enables_sequential_io() {
    let budget = ExecutionBudget {
        memory_budget_bytes: 1,
        host_available_ram_bytes: Some(10 * 1024 * 1024 * 1024),
        memory_budget_percent_bps: 2500,
    };
    let plan = sample_plan(two_chunks(), parallel_min_f32_elems());
    let hints = ExecutionHints {
        memory_budget_bytes: None,
        memory_budget_percent_bps: None,
        fold_parallel: Some(false),
        device: None,
    };
    let policy = FoldIoPolicy::resolve(&plan, &budget, Some(&hints), ElementDtype::F32, true)
        .expect("resolve");
    assert!(!policy.parallel);
    assert!(policy.sequential_io);
    assert!(!policy.linear_scan);
    assert_eq!(policy.fold_workers, None);
}

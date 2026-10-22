//! `f16` and `f64` logical materialize, spill, and scalar fold.
//!
//! Both dtypes share one macro. The `f32` into-buffer path stays in `f32.rs`.

use std::path::Path;

use crate::query::{
    fold,
    types::{ReadPlan, TetError},
};

use super::shared::{
    check_materialized_nan_slice, materialize_into_vec_dispatch, materialize_read_plan_pod_le_core,
    scatter_fill_chunks, spill_byte_len_from_elem_count, spill_read_plan_pod_le_impl,
};

/// Per-dtype materialize / spill / fold / preview for `f16` and `f64`.
macro_rules! define_float_materialize {
    (
        ty $elem:ty;
        type_label $type_label:literal;
        bytes_from $bytes_from:path;
        scatter $scatter:path;
        visit $visit:path;
        parallel_mat $parallel_mat:path;
        parallel_fold $parallel_fold:path;
        backing $backing:ident;
        preview_variant $preview_variant:ident;
        core_fn $core_fn:ident;
        check_fn $check_fn:ident;
        scatter_fn $scatter_fn:ident;
        read_fn $read_fn:ident;
        spill_fn $spill_fn:ident;
        fold_fn $fold_fn:ident;
        into_vec_fn $into_vec_fn:ident;
        preview_mat_fn $preview_mat_fn:ident;
        preview_spill_fn $preview_spill_fn:ident;
        promote |$v:ident| $promote:expr;
    ) => {
        fn $check_fn(out: &[$elem]) -> Result<(), TetError> {
            check_materialized_nan_slice(out, <$elem>::is_nan)
        }

        pub(crate) fn $core_fn(
            mmap: &[u8],
            plan: &ReadPlan,
            max_elements: Option<usize>,
            scatter_fill: fn(&[u8], &ReadPlan, &mut [$elem]) -> Result<u64, TetError>,
        ) -> Result<(Vec<$elem>, bool, u64), TetError> {
            materialize_read_plan_pod_le_core(
                mmap,
                plan,
                max_elements,
                scatter_fill,
                <$elem>::NAN,
                $check_fn,
            )
        }

        /// Decode planned raw `$type_label` chunk payloads (little-endian) into **logical row-major**
        /// order for the strided selection encoded on [`ReadPlan`].
        ///
        /// `max_elements`: `None` decodes every value in the logical tensor. `Some(0)` returns an empty
        /// vector and reads nothing from disk. `Some(n)` for `n > 0` returns the first `n` values in
        /// logical row-major order and sets `truncated` when the logical tensor is longer.
        ///
        /// # Errors
        ///
        /// Returns [`TetError::Validation`] when chunk payloads disagree with tile geometry, the
        /// strided selection is not fully covered by planned chunks, mmap bounds fail, or zstd decode
        /// fails.
        pub fn $read_fn(
            mmap: &[u8],
            plan: &ReadPlan,
            max_elements: Option<usize>,
        ) -> Result<(Vec<$elem>, bool, u64), TetError> {
            $core_fn(mmap, plan, max_elements, $scatter_fn)
        }

        pub(crate) fn $scatter_fn(
            mmap: &[u8],
            plan: &ReadPlan,
            out: &mut [$elem],
        ) -> Result<u64, TetError> {
            scatter_fill_chunks(mmap, plan, out, $scatter)
        }

        /// Spill the full logical selection as row-major `$type_label` LE to `path` using a file-backed
        /// mmap (disk-resident; does not allocate a dense `Vec` in RAM).
        ///
        /// # Errors
        ///
        /// Same validation failures as [`$read_fn`], plus I/O or mmap errors on `path`.
        pub fn $spill_fn(mmap: &[u8], plan: &ReadPlan, path: &Path) -> Result<u64, TetError> {
            let byte_len = spill_byte_len_from_elem_count(plan.logical_element_count, $bytes_from)?;
            spill_read_plan_pod_le_impl(mmap, plan, path, byte_len, $scatter_fn, Some($check_fn))
        }

        pub(crate) fn $fold_fn(
            mmap: &[u8],
            plan: &ReadPlan,
            max_preview: usize,
            kind: fold::reduction::ReductionKind,
            policy: &crate::query::fold::fold_policy::FoldIoPolicy,
        ) -> Result<fold::FoldPlanOutcome, TetError> {
            if crate::query::fold::parallel::use_parallel_fold(plan, policy) {
                return $parallel_fold(mmap, plan, max_preview, kind, policy.fold_workers);
            }
            let n = plan.logical_element_count;
            let preview_cap = max_preview.min(n);
            let mut preview = vec![<$elem>::NAN; preview_cap];
            let mut total_bytes_read_from_disk: u64 = 0;
            let chunk_order =
                crate::query::fold::fold_policy::chunk_indices_for_fold(plan, policy.sequential_io);

            let operation = match kind {
                fold::reduction::ReductionKind::ArgMin | fold::reduction::ReductionKind::ArgMax => {
                    let mut acc = fold::reduction::ArgIndexAccum::default();
                    for i in chunk_order {
                        let c = &plan.chunks[i];
                        let chunk_bytes = $visit(mmap, plan, c, |li, v| {
                            let $v = v;
                            if crate::query::types::fold_keeps_element(plan, li, $promote) {
                                acc.push_f64(li as u64, $promote, kind);
                            }
                            if li < preview_cap {
                                preview[li] = v;
                            }
                            Ok(())
                        })?;
                        total_bytes_read_from_disk = total_bytes_read_from_disk
                            .checked_add(chunk_bytes)
                            .ok_or_else(|| {
                                TetError::Validation("total bytes read overflow".into())
                            })?;
                    }
                    if acc.is_empty() && plan.cell_filter.is_none() {
                        return Err(TetError::Validation(
                            "operation requires at least one decoded value from the read plan"
                                .into(),
                        ));
                    }
                    if preview_cap > 0 && preview.iter().any(|v| v.is_nan()) {
                        return Err(TetError::Validation(
                            "materialized selection has unset preview elements".into(),
                        ));
                    }
                    acc.finish_scalar(kind, n).into()
                }
                _ => {
                    let mut acc = fold::reduction::ValueAccum::default();
                    for i in chunk_order {
                        let c = &plan.chunks[i];
                        let chunk_bytes = $visit(mmap, plan, c, |li, v| {
                            let $v = v;
                            if crate::query::types::fold_keeps_element(plan, li, $promote) {
                                acc.push_f64($promote);
                            }
                            if li < preview_cap {
                                preview[li] = v;
                            }
                            Ok(())
                        })?;
                        total_bytes_read_from_disk = total_bytes_read_from_disk
                            .checked_add(chunk_bytes)
                            .ok_or_else(|| {
                                TetError::Validation("total bytes read overflow".into())
                            })?;
                    }
                    if acc.is_empty() && plan.cell_filter.is_none() {
                        return Err(TetError::Validation(
                            "operation requires at least one decoded value from the read plan"
                                .into(),
                        ));
                    }
                    if preview_cap > 0 && preview.iter().any(|v| v.is_nan()) {
                        return Err(TetError::Validation(
                            "materialized selection has unset preview elements".into(),
                        ));
                    }
                    acc.finish_scalar(kind).into()
                }
            };

            Ok(fold::shared::build_fold_plan_outcome_typed(
                fold::shared::FoldPreviewBuffer::$preview_variant(preview),
                max_preview,
                n,
                total_bytes_read_from_disk,
                operation,
            ))
        }

        pub(crate) fn $into_vec_fn(
            mmap: &[u8],
            plan: &ReadPlan,
        ) -> Result<(Vec<$elem>, u64), TetError> {
            materialize_into_vec_dispatch(mmap, plan, $read_fn, $parallel_mat)
        }

        pub(crate) fn $preview_mat_fn(
            backing: &super::types::$backing,
            logical_len: usize,
            max: usize,
        ) -> Result<(Vec<$elem>, bool), TetError> {
            use super::shared::preview_from_backing_in_memory;

            match backing {
                super::types::$backing::InMemory(v) => {
                    Ok(preview_from_backing_in_memory(v, logical_len, max))
                }
                super::types::$backing::TempSpill(temp) => {
                    super::shared::preview_from_spill_file_pod(
                        temp.path(),
                        max.min(logical_len),
                        logical_len,
                    )
                }
            }
        }

        pub(crate) fn $preview_spill_fn(
            path: &Path,
            cap: usize,
            logical_len: usize,
        ) -> Result<(Vec<$elem>, bool), TetError> {
            super::shared::preview_from_spill_file_pod(path, cap, logical_len)
        }
    };
}

define_float_materialize! {
    ty half::f16;
    type_label "f16";
    bytes_from crate::utils::f16_le::bytes_from_elem_count;
    scatter crate::query::decode::chunk_decode::scatter_chunk_into_plan_f16;
    visit crate::query::decode::chunk_decode::visit_planned_chunk_f16;
    parallel_mat super::parallel::materialize_read_plan_f16_le_parallel;
    parallel_fold crate::query::fold::parallel::fold_read_plan_scalar_operation_f16_parallel;
    backing LogicalF16Backing;
    preview_variant F16;
    core_fn materialize_read_plan_f16_le_core;
    check_fn check_materialized_complete_f16;
    scatter_fn materialize_scatter_fill_f16;
    read_fn materialize_read_plan_f16_le;
    spill_fn spill_read_plan_f16_le;
    fold_fn fold_read_plan_scalar_operation_f16;
    into_vec_fn materialize_into_vec_f16;
    preview_mat_fn preview_from_materialized_f16;
    preview_spill_fn preview_from_spill_file_f16;
    promote |v| f64::from(v);
}

define_float_materialize! {
    ty f64;
    type_label "f64";
    bytes_from crate::utils::f64_le::bytes_from_elem_count;
    scatter crate::query::decode::chunk_decode::scatter_chunk_into_plan_f64;
    visit crate::query::decode::chunk_decode::visit_planned_chunk_f64;
    parallel_mat super::parallel::materialize_read_plan_f64_le_parallel;
    parallel_fold crate::query::fold::parallel::fold_read_plan_scalar_operation_f64_parallel;
    backing LogicalF64Backing;
    preview_variant F64;
    core_fn materialize_read_plan_f64_le_core;
    check_fn check_materialized_complete_f64;
    scatter_fn materialize_scatter_fill_f64;
    read_fn materialize_read_plan_f64_le;
    spill_fn spill_read_plan_f64_le;
    fold_fn fold_read_plan_scalar_operation_f64;
    into_vec_fn materialize_into_vec_f64;
    preview_mat_fn preview_from_materialized_f64;
    preview_spill_fn preview_from_spill_file_f64;
    promote |v| v;
}

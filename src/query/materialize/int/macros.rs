//! `define_int_materialize!` — per-integer-dtype materialize/spill surface.

/// Per-dtype materialize/spill/preview surface (`i32` / `i64` / `u8` / `u16` / `i16`).
macro_rules! define_int_materialize {
    (
        $elem:ty;
        backing $backing:ident;
        le_mod $le_mod:ident;
        scatter $scatter:path;
        scatter_seq $scatter_seq:ident;
        scatter_ty $scatter_ty:ident;
        scatter_par $scatter_par:path;
        core_fn $core_fn:ident;
        read_fn $read_fn:ident;
        spill_fn $spill_fn:ident;
        type_label $type_label:literal;
        into_vec_fn $into_vec_fn:ident;
        spill_file_fn $spill_file_fn:ident;
        preview_mat_fn $preview_mat_fn:ident;
        as_f64_fn $as_f64_fn:ident;
        promote_inmem |$v:ident| $promote_inmem:expr;
        promote_spill |$s:ident| $promote_spill:expr;
    ) => {
        pub(crate) enum $backing {
            InMemory(Vec<$elem>),
            TempSpill(TempSpillFile),
        }

        fn $scatter_seq(
            mmap: &[u8],
            plan: &ReadPlan,
            out: &mut [Option<$elem>],
        ) -> Result<u64, TetError> {
            validate_read_plan_geometry(plan, out.len())?;
            let mut total_bytes_read_from_disk: u64 = 0;
            for c in &plan.chunks {
                let n = $scatter(mmap, plan, c, out)?;
                accumulate_chunk_read_bytes(&mut total_bytes_read_from_disk, n)?;
            }
            Ok(total_bytes_read_from_disk)
        }

        type $scatter_ty = fn(&[u8], &ReadPlan, &mut [Option<$elem>]) -> Result<u64, TetError>;

        pub(crate) fn $core_fn(
            mmap: &[u8],
            plan: &ReadPlan,
            max_elements: Option<usize>,
            scatter_fill: $scatter_ty,
        ) -> Result<(Vec<$elem>, bool, u64), TetError> {
            materialize_read_plan_int_le_core(mmap, plan, max_elements, scatter_fill)
        }

        /// Decode planned raw [`$type_label`] chunk payloads (little-endian) into **logical row-major**
        /// order for the strided selection on [`ReadPlan`].
        ///
        /// `max_elements`: `None` decodes the full logical tensor. `Some(0)` returns an empty vector
        /// and reads nothing. `Some(n)` for `n > 0` returns the first `n` logical values and sets
        /// `truncated` when the logical tensor is longer.
        ///
        /// # Errors
        ///
        /// Returns [`TetError::Validation`] when chunk payloads disagree with tile geometry, the
        /// strided selection is not fully covered by planned chunks, or mmap bounds fail.
        pub fn $read_fn(
            mmap: &[u8],
            plan: &ReadPlan,
            max_elements: Option<usize>,
        ) -> Result<(Vec<$elem>, bool, u64), TetError> {
            $core_fn(mmap, plan, max_elements, $scatter_seq)
        }

        /// Spill the full logical selection as row-major [`$type_label`] LE to `path` via file-backed mmap.
        ///
        /// # Errors
        ///
        /// Same validation failures as [`$read_fn`], plus logical element count or spill byte length
        /// overflow, or I/O or mmap errors on `path`.
        pub fn $spill_fn(mmap: &[u8], plan: &ReadPlan, path: &Path) -> Result<u64, TetError> {
            let byte_len = $crate::query::materialize::shared::spill_byte_len_from_elem_count(
                plan.logical_element_count,
                $le_mod::bytes_from_elem_count,
            )?;
            spill_read_plan_int_le_impl(mmap, plan, path, byte_len, $scatter_seq)
        }

        pub(crate) fn $into_vec_fn(
            mmap: &[u8],
            plan: &ReadPlan,
        ) -> Result<(Vec<$elem>, u64), TetError> {
            let scatter = if plan.chunks.len() <= 1 {
                $scatter_seq
            } else {
                $scatter_par
            };
            $core_fn(mmap, plan, None, scatter).map(|(v, truncated, bytes)| {
                debug_assert!(!truncated);
                (v, bytes)
            })
        }

        pub(crate) fn $spill_file_fn(
            path: &Path,
            cap: usize,
            logical_len: usize,
        ) -> Result<(Vec<$elem>, bool), TetError> {
            $crate::query::materialize::shared::preview_from_spill_file_pod(path, cap, logical_len)
        }

        pub(crate) fn $preview_mat_fn(
            backing: &$backing,
            logical_len: usize,
            max: usize,
        ) -> Result<(Vec<$elem>, bool), TetError> {
            match backing {
                $backing::InMemory(v) => Ok(
                    $crate::query::materialize::shared::preview_from_backing_in_memory(
                        v,
                        logical_len,
                        max,
                    ),
                ),
                $backing::TempSpill(temp) => {
                    $spill_file_fn(temp.path(), max.min(logical_len), logical_len)
                }
            }
        }

        pub(crate) fn $as_f64_fn(backing: &$backing) -> Result<Vec<f64>, TetError> {
            match backing {
                $backing::InMemory(v) => Ok(v.iter().map(|&$v| $promote_inmem).collect()),
                $backing::TempSpill(temp) => {
                    let mmap = $crate::query::materialize::shared::mmap_spill(temp.path())?;
                    Ok(bytemuck::cast_slice::<u8, $elem>(&mmap)
                        .iter()
                        .map(|&$s| $promote_spill)
                        .collect())
                }
            }
        }
    };
}

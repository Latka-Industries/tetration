//! In-memory column buffers tiled into `.tet` (CSV and Parquet).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use crate::catalog::HistoryEvent;
use crate::utils::dtype::ElementDtype;

use super::import_metadata::finish_convert_footer;
use super::numpy::numpy_chunk_shape;
use super::shared::{ImportPlan, record_skip, write_plans_streaming};
use super::{ConvertError, ConvertProgress, ConvertSkip};

pub(crate) struct BufferedColumn {
    name: String,
    elem_size: usize,
    chunk_elems: u64,
    bytes: Vec<u8>,
}

pub(crate) struct BufferedColumnSpec {
    pub name: String,
    pub dtype: ElementDtype,
    pub bytes: Vec<u8>,
    pub attrs: BTreeMap<String, String>,
}

pub(crate) fn accept_buffered_column(
    plans: &mut Vec<ImportPlan>,
    payloads: &mut Vec<BufferedColumn>,
    skipped: &mut Vec<ConvertSkip>,
    spec: BufferedColumnSpec,
    chunk_bytes: u64,
) {
    let elem_size = spec.dtype.elem_size();
    let nrows = u64::try_from(spec.bytes.len() / elem_size).unwrap_or(0);
    if nrows == 0 {
        record_skip(skipped, spec.name, "empty");
        return;
    }
    let chunk_shape = numpy_chunk_shape(&[nrows], elem_size, chunk_bytes);
    payloads.push(BufferedColumn {
        name: spec.name.clone(),
        elem_size,
        chunk_elems: chunk_shape.first().copied().unwrap_or(1),
        bytes: spec.bytes,
    });
    let mut plan = ImportPlan::new(spec.name, spec.dtype, vec![nrows], chunk_shape, None);
    if !spec.attrs.is_empty() {
        plan = plan.with_import(spec.attrs, None, None);
    }
    plans.push(plan);
}

pub(crate) fn stream_buffered_columns(
    output: &Path,
    plans: &[ImportPlan],
    payloads: Vec<BufferedColumn>,
    parallel_jobs: usize,
    history_source: &str,
    mut progress: Option<impl FnMut(ConvertProgress)>,
    map_err: fn(String) -> ConvertError,
) -> Result<Vec<HistoryEvent>, ConvertError> {
    let mut progress_bridge = |done: u64, total: u64, dataset: &str| {
        if let Some(ref mut cb) = progress {
            cb(ConvertProgress {
                chunks_done: done,
                chunks_total: total,
                dataset: dataset.to_owned(),
            });
        }
    };
    let parallel_jobs = super::parallel::resolve_parallel_jobs(parallel_jobs);
    let payloads: Arc<[BufferedColumn]> = payloads.into();
    write_plans_streaming(
        output,
        plans,
        parallel_jobs,
        |job, buf| {
            let column = &payloads[job.dataset_id];
            let start = usize::try_from(job.chunk_coord[0] * column.chunk_elems)
                .map_err(|_| map_err("chunk offset overflow".to_owned()))?
                .saturating_mul(column.elem_size);
            let end = start + buf.len();
            if end > column.bytes.len() {
                return Err(map_err(format!(
                    "column `{}` tile exceeds parsed rows",
                    column.name
                )));
            }
            buf.copy_from_slice(&column.bytes[start..end]);
            Ok(())
        },
        Some(&mut progress_bridge as &mut dyn FnMut(u64, u64, &str)),
    )?;
    finish_convert_footer(output, history_source, plans).map_err(ConvertError::from)
}

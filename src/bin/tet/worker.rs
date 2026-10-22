//! `tet worker` — execute one `WorkerJobV1`, emit `WorkerPartialV1` JSON.

use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;

use tetration::query::distributed::{
    WorkerJobV1, execute_worker_job_v1, write_worker_partial_json,
};

/// Run `tet worker`: read job JSON, fold shard, write partial JSON to stdout.
pub fn run_worker(job: Option<PathBuf>) -> Result<(), String> {
    let bytes = match job {
        Some(path) if path.as_os_str() != "-" => {
            fs::read(&path).map_err(|e| format!("read worker job {}: {e}", path.display()))?
        }
        _ => {
            let mut buf = Vec::new();
            io::stdin()
                .read_to_end(&mut buf)
                .map_err(|e| format!("read worker job from stdin: {e}"))?;
            buf
        }
    };
    let parsed: WorkerJobV1 =
        serde_json::from_slice(&bytes).map_err(|e| format!("parse WorkerJobV1 JSON: {e}"))?;
    let partial = execute_worker_job_v1(&parsed).map_err(|e| e.to_string())?;
    let mut out = io::stdout().lock();
    write_worker_partial_json(&partial, &mut out).map_err(|e| e.to_string())?;
    out.write_all(b"\n").map_err(|e| format!("stdout: {e}"))?;
    Ok(())
}

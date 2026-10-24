//! Create a small `.tet` file and run an in-process query (no `tet` subprocess).
//!
//! Uses [`TetWriterSession`] + [`execute_query_document`]. See also [`session_write`].
//!
//! ```bash
//! cargo run --example create_and_query
//! ```

use tetration::catalog::{TetDatasetWrite, TetFile, TetWriterSession};
use tetration::prelude::*;
use tetration::query::cli::{QueryOutputFormat, format_query_response};

const SHAPE: [u64; 2] = [2, 3];
const CHUNK_SHAPE: [u64; 2] = [2, 2];
const DATASET: &str = "temperature";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join("tetration_create_and_query_example.tet");

    let mut session = TetWriterSession::create(&path);
    session.push_history_event("convert", "example");
    session.push_dataset(TetDatasetWrite::f32_row_major(
        DATASET,
        &SHAPE,
        &CHUNK_SHAPE,
        f32_tensor_one_to_six(),
    )?)?;
    let path = session.commit()?;

    let file = TetFile::open(&path)?;
    let summary = read_tet_summary_v1(file.mmap())?;
    println!(
        "wrote {} ({} dataset(s), {} chunk row(s), {} history row(s))",
        path.display(),
        summary.datasets.len(),
        summary.chunks.len(),
        summary.history.len()
    );

    let doc = parse_query_json(&format!(r#"{{"dataset":"{DATASET}","mean":[]}}"#))?;
    validate_query(&doc)?;
    let response = execute_query_document(
        &doc,
        file.path(),
        file.mmap(),
        ExecuteQueryOptions::execute_no_preview(),
        None,
    )?;
    println!(
        "{}",
        format_query_response(&response, QueryOutputFormat::Quiet)?
    );

    Ok(())
}

fn f32_tensor_one_to_six() -> Vec<u8> {
    let mut data = vec![0u8; 24];
    for (slot, n) in data.as_chunks_mut::<4>().0.iter_mut().zip(1_u8..=6) {
        slot.copy_from_slice(&f32::from(n).to_le_bytes());
    }
    data
}

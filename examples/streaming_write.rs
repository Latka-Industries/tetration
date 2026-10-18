//! Stream a large grid without holding the full tensor in RAM.
//!
//! ```bash
//! cargo run --example streaming_write
//! ```

use tetration::catalog::{TetDatasetStreamSpec, TetFile, TetWriterSession};
use tetration::prelude::*;
use tetration::query::cli::{QueryOutputFormat, format_query_response};

const SHAPE: [u64; 2] = [128, 128];
const CHUNK_SHAPE: [u64; 2] = [32, 32];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join("tetration_streaming_write_example.tet");

    let mut session = TetWriterSession::create(&path);
    session.metadata.tool = Some("tetration-example".to_owned());
    session.push_history_event("write", "streaming_write");
    let mut spec = TetDatasetStreamSpec::f32_row_major("field", &SHAPE, &CHUNK_SHAPE)?;
    spec.attrs.insert("units".to_owned(), "1".to_owned());
    session.push_dataset_streaming(spec)?;

    let path = session.commit_with_fill(1, |job, buf| {
        // Synthetic tile: value = row index of the chunk's first row (cheap stand-in for I/O).
        let row0 = job.chunk_coord[0] * CHUNK_SHAPE[0];
        for slot in buf.chunks_exact_mut(4) {
            slot.copy_from_slice(&(row0 as f32).to_le_bytes());
        }
        Ok(())
    })?;
    println!("committed {}", path.display());

    let file = TetFile::open(&path)?;
    let line = format_query_response(
        &execute_query_json(
            r#"{"dataset":"field","mean":[]}"#,
            file.path(),
            file.mmap(),
            ExecuteQueryOptions::execute_no_preview(),
            None,
        )?,
        QueryOutputFormat::Quiet,
    )?;
    println!("{line}");

    Ok(())
}

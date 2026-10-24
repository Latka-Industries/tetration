//! Create a `.tet` with [`TetWriterSession`] + [`TetDatasetWrite::f32_row_major`]
//! (also `f64_row_major` / `row_major` for other wire dtypes), then query via
//! [`TetFile`] + [`execute_query_json`].
//!
//! ```bash
//! cargo run --example session_write
//! ```

use std::collections::BTreeMap;

use tetration::catalog::{CoordAxisV1, TetDatasetWrite, TetFile, TetWriterSession};
use tetration::prelude::*;
use tetration::query::cli::{QueryOutputFormat, format_query_response};

const SHAPE: [u64; 2] = [2, 3];
const CHUNK_SHAPE: [u64; 2] = [2, 2];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join("tetration_session_write_example.tet");

    let mut session = TetWriterSession::create(&path);
    session.metadata.tool = Some("tetration-example".to_owned());
    session.push_history_event("write", "session_write");
    let mut ds = TetDatasetWrite::f32_row_major(
        "temperature",
        &SHAPE,
        &CHUNK_SHAPE,
        f32_tensor_one_to_six(),
    )?;
    ds.attrs.insert("units".to_owned(), "K".to_owned());
    ds.dim_names = Some(vec!["row".to_owned(), "col".to_owned()]);
    let mut coords = BTreeMap::new();
    coords.insert(
        "row".to_owned(),
        CoordAxisV1 {
            labels: vec!["r0".to_owned(), "r1".to_owned()],
        },
    );
    ds.coords = Some(coords);
    session.push_dataset(ds)?;

    let path = session.commit()?;
    println!("committed {}", path.display());

    let file = TetFile::open(&path)?;
    for ds in file.datasets()? {
        println!("  dataset {} dtype={}", ds.name, ds.dtype);
    }

    let line = format_query_response(
        &execute_query_json(
            r#"{"dataset":"temperature","mean":[]}"#,
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

fn f32_tensor_one_to_six() -> Vec<u8> {
    let mut data = vec![0u8; 24];
    for (slot, n) in data.as_chunks_mut::<4>().0.iter_mut().zip(1_u8..=6) {
        slot.copy_from_slice(&f32::from(n).to_le_bytes());
    }
    data
}

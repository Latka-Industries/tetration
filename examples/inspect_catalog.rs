//! Mmap a `.tet` file and print catalog summary (parity with `tet info` / `--json` data).
//!
//! With no args, builds the same fixture as [`create_and_query`] in a temp file.
//!
//! ```bash
//! cargo run --example inspect_catalog
//! cargo run --example inspect_catalog -- /path/to/file.tet
//! ```

use std::path::{Path, PathBuf};

use tetration::catalog::{
    CHUNK_PAYLOAD_CODEC_V1, DATASET_DTYPE_TAG_V1, RawArrayWrite, read_tet_summary_v1,
    write_raw_array_file,
};
use tetration::layout::mmap_file_read;

const SHAPE: [u64; 2] = [2, 3];
const CHUNK_SHAPE: [u64; 2] = [2, 2];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = resolve_tet_path()?;
    let mmap = mmap_file_read(&path)?;
    let summary = read_tet_summary_v1(&mmap)?;

    println!("file: {}", path.display());
    println!(
        "layout_version={} datasets={} chunks={} history_events={}",
        summary.superblock.layout_version,
        summary.datasets.len(),
        summary.chunks.len(),
        summary.history.len()
    );

    if let Some(file) = &summary.metadata.file {
        println!("file metadata:");
        if let Some(tool) = &file.tool {
            println!("  tool: {tool}");
        }
        if let Some(ver) = &file.library_version {
            println!("  library_version: {ver}");
        }
    }

    for ds in &summary.datasets {
        let shape = ds
            .shape
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join("×");
        println!("  {}  dtype={}  shape={shape}", ds.name, ds.dtype);
        if let Some(meta) = summary.metadata.datasets.get(&ds.name) {
            if let Some(dim_names) = &meta.dim_names {
                println!("       dim_names: {}", dim_names.join(", "));
            }
            for (k, v) in &meta.attrs {
                println!("       {k}: {v}");
            }
        }
    }

    if !summary.history.is_empty() {
        println!("history:");
        for ev in &summary.history {
            println!("  {}  {}  {}", ev.op, ev.source, ev.at);
        }
    }

    Ok(())
}

fn resolve_tet_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    if let Some(path) = args.next() {
        return Ok(PathBuf::from(path));
    }
    let path = std::env::temp_dir().join("tetration_inspect_catalog_example.tet");
    write_demo_file(&path)?;
    Ok(path)
}

fn write_demo_file(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut data = vec![0u8; 24];
    for (slot, n) in data.as_chunks_mut::<4>().0.iter_mut().zip(1_u8..=6) {
        slot.copy_from_slice(&f32::from(n).to_le_bytes());
    }
    write_raw_array_file(
        path,
        &RawArrayWrite {
            name: "demo",
            dtype: DATASET_DTYPE_TAG_V1.f32,
            shape: &SHAPE,
            chunk_shape: &CHUNK_SHAPE,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data: &data,
            file_execution: None,
        },
    )?;
    Ok(())
}

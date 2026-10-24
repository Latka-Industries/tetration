//! `tet info` formatting and filters.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

use crate::catalog::{
    DatasetMetadataV1, FooterBlobV1, TetMetadataV1, read_tet_summary_v1, write_footer_blob,
};
use crate::layout::mmap_file_read;
use crate::query::cli::{
    InfoListFilter, InfoMetadataDisplay, InfoViewSections, format_info_json, format_info_quiet,
    format_info_text,
};

use super::fixture::write_multichunk_2x3_tiles;

/// Resolve the `tet` binary for spawn smoke tests (lib tests do not set `CARGO_BIN_EXE_tet`).
fn tet_info_command(tet_path: &Path) -> Command {
    let path_arg = tet_path.to_str().expect("fixture path must be UTF-8");
    if let Ok(exe) = std::env::var("CARGO_BIN_EXE_tet") {
        let mut cmd = Command::new(exe);
        cmd.args(["info", path_arg]);
        return cmd;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for rel in ["target/debug/tet", "target/release/tet"] {
        let bin = root.join(rel);
        if bin.is_file() {
            let mut cmd = Command::new(bin);
            cmd.args(["info", path_arg]);
            return cmd;
        }
    }
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "--quiet", "--bin", "tet", "--", "info", path_arg]);
    cmd
}

#[test]
fn info_default_table_lists_datasets() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let text = format_info_text(
        Some(&path),
        mmap.len() as u64,
        &summary,
        None,
        InfoViewSections::default_table(),
        32,
        InfoMetadataDisplay::WhenPresent,
    );
    assert!(text.contains("datasets:"));
    assert!(text.contains("a"));
    assert!(text.contains("f32"));
    assert!(!text.contains("\"superblock\""));
}

#[test]
fn info_default_shows_footer_metadata_under_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("meta.tet");
    write_multichunk_2x3_tiles(&path, "temperature");
    write_footer_blob(
        &path,
        &FooterBlobV1 {
            history: Vec::new(),
            metadata_ref: None,
            metadata: Some(TetMetadataV1 {
                file: None,
                datasets: [(
                    "temperature".to_owned(),
                    DatasetMetadataV1 {
                        attrs: [("units".to_owned(), "K".to_owned())].into_iter().collect(),
                        dim_names: Some(vec!["y".to_owned(), "x".to_owned()]),
                        coords: None,
                    },
                )]
                .into_iter()
                .collect(),
            }),
        },
    )
    .unwrap();

    let mmap = mmap_file_read(&path).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let text = format_info_text(
        Some(&path),
        mmap.len() as u64,
        &summary,
        None,
        InfoViewSections::default_table(),
        32,
        InfoMetadataDisplay::WhenPresent,
    );
    assert!(text.contains("dim_names"));
    assert!(text.contains("y, x"));
    assert!(text.contains("units"));
    assert!(text.contains('K'));
}

#[test]
fn info_grep_matches_footer_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("grep_meta.tet");
    write_multichunk_2x3_tiles(&path, "temperature");
    write_footer_blob(
        &path,
        &FooterBlobV1 {
            history: Vec::new(),
            metadata_ref: None,
            metadata: Some(TetMetadataV1 {
                file: None,
                datasets: [(
                    "temperature".to_owned(),
                    DatasetMetadataV1 {
                        attrs: [("units".to_owned(), "Kelvin".to_owned())]
                            .into_iter()
                            .collect(),
                        dim_names: None,
                        coords: None,
                    },
                )]
                .into_iter()
                .collect(),
            }),
        },
    )
    .unwrap();

    let mmap = mmap_file_read(&path).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let filter = InfoListFilter {
        dataset: None,
        grep: Some("kelvin".to_owned()),
    };
    let text = format_info_text(
        Some(&path),
        mmap.len() as u64,
        &summary,
        Some(&filter),
        InfoViewSections::default_table(),
        32,
        InfoMetadataDisplay::WhenPresent,
    );
    assert!(text.contains("temperature"));
    assert!(text.contains("units"));
    assert!(text.contains("Kelvin"));
}

#[test]
fn info_json_includes_full_summary() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("one.tet");
    write_multichunk_2x3_tiles(&path, "temp");
    let mmap = mmap_file_read(&path).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let json = format_info_json(Some(&path), mmap.len() as u64, &summary, None).unwrap();
    assert!(json.contains("\"superblock\""));
    assert!(json.contains("\"datasets\""));
    assert!(json.contains("temp"));
}

#[test]
fn info_grep_filters_datasets() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "temperature");
    let mmap = mmap_file_read(&path).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let filter = InfoListFilter {
        dataset: None,
        grep: Some("temp".to_owned()),
    };
    let text = format_info_text(
        Some(&path),
        mmap.len() as u64,
        &summary,
        Some(&filter),
        InfoViewSections::default_table(),
        32,
        InfoMetadataDisplay::WhenPresent,
    );
    assert!(text.contains("temperature"));
    assert!(text.contains("filter"));
    assert!(text.contains("grep~temp"));
}

#[test]
fn tet_info_binary_runs_on_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let out = tet_info_command(&path).output().unwrap();
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("datasets:"));
    assert!(stdout.contains("a"));
}

#[test]
fn info_quiet_one_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let line = format_info_quiet(Some(&path), mmap.len() as u64, &summary, None);
    assert!(line.contains("path="));
    assert!(line.contains("datasets=1"));
}

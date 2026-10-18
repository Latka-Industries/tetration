//! Sealed `.tet` files: many concurrent readers (library + optional CLI smoke).

use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::mpsc,
    thread,
};

use crate::catalog::TetFile;
use crate::query::cli::{QueryOutputFormat, format_query_response};
use crate::query::{ExecuteQueryOptions, execute_query_document, parse_query_json, validate_query};

use super::fixture::write_multichunk_2x3_tiles;

const WORKERS: usize = 8;
const EXPECTED_SUM: f64 = 21.0; // values 1..=6 on [2, 3] grid

fn sum_query_doc(dataset: &str) -> crate::query::QueryDocument {
    let doc = parse_query_json(&format!(r#"{{"dataset":"{dataset}","sum":[]}}"#)).unwrap();
    validate_query(&doc).unwrap();
    doc
}

fn assert_sum_response(response: &crate::query::QueryResponse) {
    let line = format_query_response(response, QueryOutputFormat::Quiet).unwrap();
    assert!(
        line.contains("sum=21") || line.contains("sum=21.0"),
        "expected sum=21 in quiet line, got: {line}"
    );
    let exec = response.execution.as_ref().expect("execution block");
    let sum = exec.operation_sum.expect("operation_sum");
    assert!((sum - EXPECTED_SUM).abs() < 1e-9, "sum={sum}");
}

#[test]
fn concurrent_library_queries_on_sealed_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sealed.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let path = path.to_path_buf();

    let (tx, rx) = mpsc::channel();
    let handles: Vec<_> = (0..WORKERS)
        .map(|_| {
            let path = path.clone();
            let tx = tx.clone();
            thread::spawn(move || {
                let file = TetFile::open(&path).unwrap();
                let doc = sum_query_doc("a");
                let response = execute_query_document(
                    &doc,
                    file.path(),
                    file.mmap(),
                    ExecuteQueryOptions::execute_no_preview(),
                    None,
                )
                .unwrap();
                tx.send(response).unwrap();
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    drop(tx);

    for response in rx.iter().take(WORKERS) {
        assert_sum_response(&response);
    }
}

fn tet_query_command(tet_path: &Path, query_json: &str) -> Command {
    let path_arg = tet_path.to_str().expect("fixture path must be UTF-8");
    let base_args = [
        "query",
        query_json,
        "-t",
        path_arg,
        "-x",
        "-q",
        "--preview",
        "0",
    ];
    if let Ok(exe) = std::env::var("CARGO_BIN_EXE_tet") {
        let mut cmd = Command::new(exe);
        cmd.args(base_args);
        return cmd;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for rel in ["target/debug/tet", "target/release/tet"] {
        let bin = root.join(rel);
        if bin.is_file() {
            let mut cmd = Command::new(bin);
            cmd.args(base_args);
            return cmd;
        }
    }
    let mut cmd = Command::new(env!("CARGO"));
    let mut args = vec![
        "run".to_owned(),
        "--quiet".to_owned(),
        "--bin".to_owned(),
        "tet".to_owned(),
        "--".to_owned(),
    ];
    args.extend(base_args.iter().map(|s| (*s).to_owned()));
    cmd.args(args);
    cmd
}

#[test]
fn concurrent_process_tet_query_smoke() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sealed.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let query = r#"{"dataset":"a","sum":[]}"#;
    let path = path.to_path_buf();

    let (tx, rx) = mpsc::channel();
    let workers = WORKERS.min(4);
    let handles: Vec<_> = (0..workers)
        .map(|_| {
            let path = path.clone();
            let query = query.to_owned();
            let tx = tx.clone();
            thread::spawn(move || {
                let output = tet_query_command(&path, &query)
                    .output()
                    .expect("spawn tet query");
                tx.send(output).unwrap();
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    drop(tx);

    for output in rx.iter().take(workers) {
        assert!(
            output.status.success(),
            "tet query failed: status={:?} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("sum=21"),
            "expected sum=21 on stdout, got: {stdout}"
        );
    }
}

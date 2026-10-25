//! `tet run` job files: parse / validate, dry run, fail-fast, and an end-to-end pipeline (THI-646).

use std::path::{Path, PathBuf};

use crate::job::{
    Job, JobError, JobFormat, OPS, RunJobOptions, Step, StepStatus, detect_job_format, parse_job,
    run_job, run_job_path,
};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/jobs")
}

/// Temp dir holding `raw/table.csv` (the fixture input) so jobs can write `work/` safely.
fn scratch_with_csv() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("raw")).unwrap();
    std::fs::copy(
        fixture_dir().join("raw/table.csv"),
        dir.path().join("raw/table.csv"),
    )
    .unwrap();
    dir
}

fn opts(root: &Path) -> RunJobOptions {
    RunJobOptions {
        root: Some(root.to_path_buf()),
        ..RunJobOptions::default()
    }
}

fn toml_job(text: &str) -> Result<Job, JobError> {
    parse_job(text, JobFormat::Toml)
}

fn parse_err(text: &str) -> String {
    toml_job(text).unwrap_err().to_string()
}

#[test]
fn parse_toml_keeps_step_order() {
    let job = toml_job(
        r#"
schema_version = 1
[[step]]
op = "convert"
from = "raw/x.csv"
to = "work/x.tet"
[[step]]
op = "verify"
path = "work/x.tet"
[[step]]
op = "index"
path = "work/x.tet"
"#,
    )
    .unwrap();
    let ops: Vec<_> = job.steps.iter().map(Step::op).collect();
    assert_eq!(ops, ["convert", "verify", "index"]);
    let Step::Convert(c) = &job.steps[0] else {
        panic!("expected convert");
    };
    assert_eq!(c.from, "raw/x.csv");
    assert_eq!(c.to, Path::new("work/x.tet"));
    assert_eq!(c.jobs, 0);
}

#[test]
fn parse_json_steps_and_step_alias() {
    let steps = r#"{"schema_version":1,"steps":[{"op":"verify","path":"a.tet"}]}"#;
    assert_eq!(parse_job(steps, JobFormat::Auto).unwrap().steps.len(), 1);
    let alias = r#"{"schema_version":1,"step":[{"op":"verify","path":"a.tet"}]}"#;
    assert_eq!(parse_job(alias, JobFormat::Json).unwrap().steps.len(), 1);
    let both = r#"{"schema_version":1,"step":[],"steps":[]}"#;
    assert!(parse_job(both, JobFormat::Json).is_err());
    // TOML accepts the `steps` spelling too.
    let toml_steps =
        toml_job("schema_version = 1\nsteps = [{ op = \"verify\", path = \"a.tet\" }]");
    assert_eq!(toml_steps.unwrap().steps.len(), 1);
}

#[test]
fn detect_format_from_extension_and_content() {
    assert_eq!(
        detect_job_format(Some(Path::new("j.json")), "x"),
        JobFormat::Json
    );
    assert_eq!(
        detect_job_format(Some(Path::new("j.toml")), "{"),
        JobFormat::Toml
    );
    assert_eq!(detect_job_format(None, "  {\"a\":1}"), JobFormat::Json);
    assert_eq!(
        detect_job_format(None, "schema_version = 1"),
        JobFormat::Toml
    );
}

#[test]
fn parse_every_op_in_the_closed_set() {
    let minimal = |op: &str| match op {
        "info"
        | "verify"
        | "repair"
        | "index"
        | "semantics.scaffold"
        | "semantics.wizard"
        | "semantics.lint"
        | "semantics.package-write" => {
            format!("op = \"{op}\"\npath = \"a.tet\"")
        }
        "query" => "op = \"query\"\nquery = { dataset = \"a\", mean = [] }".to_owned(),
        "convert" => "op = \"convert\"\nfrom = \"a.csv\"\nto = \"a.tet\"".to_owned(),
        "export" => "op = \"export\"\nfrom = \"a.tet\"\nto = \"a.zarr\"".to_owned(),
        "semantics.apply" => {
            "op = \"semantics.apply\"\npath = \"a.tet\"\nfrom = \"s.toml\"".to_owned()
        }
        "semantics.set" => {
            "op = \"semantics.set\"\npath = \"a.tet\"\ndataset = \"a\"\nset = { k = \"v\" }"
                .to_owned()
        }
        "ingest" => "op = \"ingest\"\nfrom = [\"a.tet\", \"b.tet\"]\nto = \"p.tet.d\"".to_owned(),
        "publish" | "compact" => format!("op = \"{op}\"\npackage = \"p.tet.d\""),
        "pack" => "op = \"pack\"\nfrom = \"p.tet.d\"\nto = \"a.tet\"".to_owned(),
        "gc" => "op = \"gc\"\ntarget = \"p.tet.d\"".to_owned(),
        other => panic!("test table missing {other}"),
    };
    for op in OPS {
        let text = format!("schema_version = 1\n[[step]]\n{}\n", minimal(op));
        let job = toml_job(&text).unwrap_or_else(|e| panic!("{op}: {e}"));
        assert_eq!(job.steps[0].op(), *op);
    }
    assert_eq!(OPS.len(), 18);
}

#[test]
fn parse_rejects_worker_qhist_and_unknown_ops() {
    for op in ["worker", "qhist", "semantics.package-show", "nope"] {
        let msg = parse_err(&format!("schema_version = 1\n[[step]]\nop = \"{op}\"\n"));
        assert!(msg.contains("unknown op"), "{op}: {msg}");
        assert!(msg.contains("step 1"), "{msg}");
    }
}

#[test]
fn parse_rejects_schema_problems() {
    assert!(parse_err("[[step]]\nop = \"verify\"\npath = \"a\"").contains("schema_version"));
    let v2 = parse_err("schema_version = 2\n[[step]]\nop = \"verify\"\npath = \"a\"");
    assert!(v2.contains("unsupported schema_version 2"), "{v2}");
    assert!(parse_err("schema_version = 1\n").contains("missing"));
    assert!(parse_err("schema_version = 1\nsteps = []\n").contains("no steps"));
    let extra =
        parse_err("schema_version = 1\nretries = 3\n[[step]]\nop = \"verify\"\npath = \"a\"");
    assert!(extra.contains("unknown top-level key `retries`"), "{extra}");
    assert!(parse_err("schema_version = 1\n[[step]]\npath = \"a\"").contains("missing `op`"));
}

#[test]
fn parse_rejects_unknown_step_fields() {
    // No control flow: `on_error`, `retries`, `when` are not part of the schema.
    for field in ["on_error = \"continue\"", "retries = 2", "when = \"x\""] {
        let msg = parse_err(&format!(
            "schema_version = 1\n[[step]]\nop = \"verify\"\npath = \"a.tet\"\n{field}\n"
        ));
        assert!(msg.contains("unknown field"), "{field}: {msg}");
        assert!(msg.contains("step 1 (verify)"), "{msg}");
    }
}

#[test]
fn query_step_doc_and_inline_are_mutually_exclusive() {
    let both = parse_err(
        "schema_version = 1\n[[step]]\nop = \"query\"\ndoc = \"q.json\"\nquery = { dataset = \"a\", mean = [] }\n",
    );
    assert!(both.contains("mutually exclusive"), "{both}");
    let neither = parse_err("schema_version = 1\n[[step]]\nop = \"query\"\n");
    assert!(neither.contains("`doc"), "{neither}");
    assert!(toml_job("schema_version = 1\n[[step]]\nop = \"query\"\ndoc = \"q.json\"\n").is_ok());
}

#[test]
fn step_flag_conflicts_are_rejected_at_parse_time() {
    let cases = [
        (
            "verify",
            "path = \"a\"\njson = true\nquiet = true",
            "conflict",
        ),
        (
            "query",
            "doc = \"q.json\"\npreview = 4",
            "`preview` requires `execute",
        ),
        (
            "query",
            "doc = \"q.json\"\nexecute = true",
            "requires `tet`",
        ),
        (
            "publish",
            "package = \"p\"\nabort = true\ndry_run = true",
            "cannot be combined",
        ),
        (
            "publish",
            "package = \"p\"\nabort = true",
            "requires `generation`",
        ),
        (
            "semantics.apply",
            "path = \"a\"\nfrom = \"s.toml\"\noutput = \"o.toml\"",
            "requires `annotate",
        ),
        (
            "semantics.set",
            "path = \"a\"\ndataset = \"d\"",
            "at least one",
        ),
        ("ingest", "from = []\nto = \"p\"", "at least one"),
        (
            "query",
            "doc = \"q\"\nformat = \"nope\"",
            "unknown output format",
        ),
    ];
    for (op, body, want) in cases {
        let msg = parse_err(&format!(
            "schema_version = 1\n[[step]]\nop = \"{op}\"\n{body}\n"
        ));
        assert!(msg.contains(want), "{op}: wanted {want:?} in {msg}");
    }
}

#[test]
fn dry_run_plans_every_step_and_writes_nothing() {
    let dir = scratch_with_csv();
    let report = run_job_path(
        &fixture_dir().join("csv_verify_query.toml"),
        &RunJobOptions {
            root: Some(dir.path().to_path_buf()),
            dry_run: true,
            ..RunJobOptions::default()
        },
    )
    .unwrap();
    assert!(report.dry_run);
    assert_eq!(report.steps.len(), 3);
    assert!(report.steps.iter().all(|s| s.status == StepStatus::Planned));
    assert!(report.steps[0].description.starts_with("convert "));
    assert!(report.steps[0].description.contains("work/table.tet"));
    assert!(
        report.steps[2]
            .description
            .starts_with("query inline query")
    );
    assert!(
        !dir.path().join("work").exists(),
        "dry run must not execute"
    );
}

#[test]
fn dry_run_validates_query_docs_before_anything_runs() {
    let dir = tempfile::tempdir().unwrap();
    let missing =
        toml_job("schema_version = 1\n[[step]]\nop = \"query\"\ndoc = \"nope.json\"\n").unwrap();
    let err = run_job(
        &missing,
        &RunJobOptions {
            dry_run: true,
            ..opts(dir.path())
        },
    )
    .unwrap_err();
    assert!(matches!(err, JobError::Invalid(_)), "{err}");
    assert!(err.to_string().contains("step 1 (query)"), "{err}");

    std::fs::write(dir.path().join("bad.json"), r#"{"dataset":"a","bogus":1}"#).unwrap();
    let bad =
        toml_job("schema_version = 1\n[[step]]\nop = \"query\"\ndoc = \"bad.json\"\n").unwrap();
    let err = run_job(&bad, &opts(dir.path())).unwrap_err();
    assert!(matches!(err, JobError::Invalid(_)), "{err}");

    let inline = toml_job(
        "schema_version = 1\n[[step]]\nop = \"query\"\nquery = { dataset = \"a\", bogus = 1 }\n",
    )
    .unwrap();
    assert!(matches!(
        run_job(&inline, &opts(dir.path())).unwrap_err(),
        JobError::Invalid(_)
    ));
}

#[test]
fn preflight_failure_runs_no_steps() {
    let dir = scratch_with_csv();
    let job = toml_job(
        r#"
schema_version = 1
[[step]]
op = "convert"
from = "raw/table.csv"
to = "work/table.tet"
[[step]]
op = "query"
doc = "missing.json"
"#,
    )
    .unwrap();
    let err = run_job(&job, &opts(dir.path())).unwrap_err();
    assert!(matches!(err, JobError::Invalid(_)), "{err}");
    assert!(!dir.path().join("work").exists());
}

#[test]
fn wizard_is_refused_unless_interactive_is_allowed() {
    let job =
        toml_job("schema_version = 1\n[[step]]\nop = \"semantics.wizard\"\npath = \"a.tet\"\n")
            .unwrap();
    for dry_run in [true, false] {
        let err = run_job(
            &job,
            &RunJobOptions {
                dry_run,
                ..RunJobOptions::default()
            },
        )
        .unwrap_err();
        let JobError::Invalid(msg) = &err else {
            panic!("expected Invalid, got {err}");
        };
        assert!(msg.contains("interactive"), "{msg}");
    }
    // With the flag the dry run plans it (never prompts).
    let report = run_job(
        &job,
        &RunJobOptions {
            dry_run: true,
            allow_interactive: true,
            ..RunJobOptions::default()
        },
    )
    .unwrap();
    assert_eq!(report.steps[0].op, "semantics.wizard");
}

#[test]
fn pipeline_csv_convert_verify_query_toml_and_json() {
    for name in ["csv_verify_query.toml", "csv_verify_query.json"] {
        let dir = scratch_with_csv();
        let report = run_job_path(&fixture_dir().join(name), &opts(dir.path())).unwrap();
        assert!(!report.dry_run);
        let ops: Vec<_> = report.steps.iter().map(|s| s.op).collect();
        assert_eq!(ops, ["convert", "verify", "query"], "{name}");
        assert!(report.steps.iter().all(|s| s.status == StepStatus::Ok));
        assert!(dir.path().join("work/table.tet").is_file());

        let convert = &report.steps[0];
        assert!(
            convert
                .note
                .as_deref()
                .unwrap()
                .contains("converted 2 dataset")
        );
        assert_eq!(convert.data.as_ref().unwrap()["dataset_count"], 2);

        let verify = &report.steps[1];
        assert!(
            verify.stdout.as_deref().unwrap().contains("ok"),
            "{verify:?}"
        );

        let query = &report.steps[2];
        let line = query.stdout.as_deref().unwrap();
        assert!(line.contains("dataset=temp"), "{line}");
        // mean(1.5, 2.5, 4.0) = 2.666…
        assert!(line.contains("2.66"), "{line}");
        assert!(query.data.is_some());
    }
}

#[test]
fn run_job_path_defaults_root_to_job_dir_and_root_overrides_it() {
    let dir = scratch_with_csv();
    let job_path = dir.path().join("job.toml");
    std::fs::copy(fixture_dir().join("csv_verify_query.toml"), &job_path).unwrap();

    let report = run_job_path(&job_path, &RunJobOptions::default()).unwrap();
    assert_eq!(report.root, dir.path());
    assert!(dir.path().join("work/table.tet").is_file());

    // `root` points elsewhere: raw/table.csv is not there, so convert fails.
    let other = tempfile::tempdir().unwrap();
    let err = run_job_path(&job_path, &opts(other.path())).unwrap_err();
    assert!(matches!(err, JobError::Step { index: 0, .. }), "{err}");
}

#[test]
fn run_job_path_reports_io_errors() {
    let err = run_job_path(
        Path::new("definitely/missing/job.toml"),
        &RunJobOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(err, JobError::Io { .. }), "{err}");
}

#[test]
fn fail_fast_stops_at_first_failing_step() {
    let dir = scratch_with_csv();
    let job = toml_job(
        r#"
schema_version = 1
[[step]]
op = "convert"
from = "raw/table.csv"
to = "work/a.tet"
[[step]]
op = "verify"
path = "work/does-not-exist.tet"
[[step]]
op = "convert"
from = "raw/table.csv"
to = "work/never.tet"
"#,
    )
    .unwrap();
    let mut seen = Vec::new();
    let err = crate::job::run_job_observed(&job, &opts(dir.path()), |s, _| seen.push(s.index))
        .unwrap_err();
    let JobError::Step {
        index,
        op,
        completed,
        ..
    } = &err
    else {
        panic!("expected Step error, got {err}");
    };
    assert_eq!((*index, *op), (1, "verify"));
    assert_eq!(completed.len(), 1);
    assert_eq!(seen, [0]);
    assert!(
        err.to_string().starts_with("step 2 (verify) failed"),
        "{err}"
    );
    // The failing step names the resolved (absolute) path once, at the end.
    let missing = dir.path().join("work/does-not-exist.tet");
    assert!(
        err.to_string()
            .ends_with(&format!(": {}", missing.display())),
        "{err}"
    );
    assert_eq!(
        err.to_string().matches("does-not-exist.tet").count(),
        1,
        "{err}"
    );
    assert!(dir.path().join("work/a.tet").is_file());
    assert!(!dir.path().join("work/never.tet").exists());
}

#[test]
fn strict_convert_failure_is_a_step_error() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("t.csv"), "temp,note\n1.5,\"a,b\"\n2.5,9\n").unwrap();
    let job = toml_job(
        "schema_version = 1\n[[step]]\nop = \"convert\"\nfrom = \"t.csv\"\nto = \"t.tet\"\nstrict = true\n",
    )
    .unwrap();
    let err = run_job(&job, &opts(dir.path())).unwrap_err();
    assert!(matches!(err, JobError::Step { index: 0, .. }), "{err}");
}

#[test]
fn semantics_set_lint_and_info_steps() {
    let dir = scratch_with_csv();
    let job = toml_job(
        r#"
schema_version = 1
[[step]]
op = "convert"
from = "raw/table.csv"
to = "work/table.tet"
[[step]]
op = "semantics.set"
path = "work/table.tet"
dataset = "temp"
set = { frame_rate = 30, media_kind = "video" }
[[step]]
op = "semantics.lint"
path = "work/table.tet"
[[step]]
op = "semantics.scaffold"
path = "work/table.tet"
output = "out/policy.toml"
[[step]]
op = "info"
path = "work/table.tet"
json = true
dataset = "temp"
"#,
    )
    .unwrap();
    let report = run_job(&job, &opts(dir.path())).unwrap();
    assert_eq!(report.steps.len(), 5);
    assert!(
        report.steps[1]
            .note
            .as_deref()
            .unwrap()
            .contains("updated dataset")
    );
    assert!(report.steps[2].stdout.is_some());
    let policy = std::fs::read_to_string(dir.path().join("out/policy.toml")).unwrap();
    assert!(policy.contains("frame_rate"), "{policy}");
    let info = report.steps[4].stdout.as_deref().unwrap();
    assert!(info.contains("\"frame_rate\""), "{info}");
    assert!(info.contains("\"30\""), "{info}");
}

#[test]
fn package_steps_ingest_publish_verify_pack_gc() {
    let dir = scratch_with_csv();
    let job = toml_job(
        r#"
schema_version = 1
[[step]]
op = "convert"
from = "raw/table.csv"
to = "work/a.tet"
[[step]]
op = "ingest"
from = "work/a.tet"
to = "work/pkg.tet.d"
jobs = 1
[[step]]
op = "publish"
package = "work/pkg.tet.d"
dry_run = true
[[step]]
op = "publish"
package = "work/pkg.tet.d"
[[step]]
op = "verify"
path = "work/pkg.tet.d"
quiet = true
[[step]]
op = "query"
tet = "work/pkg.tet.d"
doc = "q.json"
execute = true
quiet = true
[[step]]
op = "pack"
from = "work/pkg.tet.d"
to = "out/packed.tet"
[[step]]
op = "verify"
path = "out/packed.tet"
quiet = true
[[step]]
op = "gc"
target = "work/pkg.tet.d"
"#,
    )
    .unwrap();
    std::fs::write(dir.path().join("q.json"), r#"{"dataset":"n","sum":[]}"#).unwrap();
    let report = run_job(&job, &opts(dir.path())).unwrap();
    assert_eq!(report.steps.len(), 9);
    assert!(
        report.steps[2]
            .note
            .as_deref()
            .unwrap()
            .contains("dry-run ok")
    );
    assert!(
        report.steps[3]
            .note
            .as_deref()
            .unwrap()
            .contains("published generation=1")
    );
    assert!(
        report.steps[5]
            .stdout
            .as_deref()
            .unwrap()
            .contains("dataset=n")
    );
    assert!(dir.path().join("out/packed.tet").is_file());
    assert!(
        report.steps[8]
            .note
            .as_deref()
            .unwrap()
            .contains("gc dry-run")
    );
}

#[test]
fn quiet_package_steps_emit_no_note() {
    let dir = scratch_with_csv();
    let job = toml_job(
        r#"
schema_version = 1
[[step]]
op = "convert"
from = "raw/table.csv"
to = "a.tet"
[[step]]
op = "index"
path = "a.tet"
quiet = true
"#,
    )
    .unwrap();
    let report = run_job(&job, &opts(dir.path())).unwrap();
    assert!(report.steps[1].note.is_none());
}

fn verify_step_stdout(extra: &str) -> String {
    let dir = scratch_with_csv();
    let job = toml_job(&format!(
        "schema_version = 1\n[[step]]\nop = \"convert\"\nfrom = \"raw/table.csv\"\nto = \"a.tet\"\n\
         [[step]]\nop = \"verify\"\npath = \"a.tet\"\n{extra}\n"
    ))
    .unwrap();
    let report = run_job(&job, &opts(dir.path())).unwrap();
    report.steps[1].stdout.clone().unwrap()
}

#[test]
fn verify_omitted_quiet_defaults_to_one_line() {
    let out = verify_step_stdout("");
    assert_eq!(out.trim_end().lines().count(), 1, "{out}");
    assert!(out.contains("status=ok"), "{out}");
}

#[test]
fn verify_explicit_quiet_false_restores_full_text() {
    let full = verify_step_stdout("quiet = false");
    assert!(full.trim_end().lines().count() > 1, "{full}");
    // `quiet = true` is the same one-liner as the default.
    assert_eq!(
        verify_step_stdout("quiet = true")
            .trim_end()
            .lines()
            .count(),
        1
    );
    // `json` wins over the quiet default.
    let json = verify_step_stdout("json = true");
    assert!(json.trim_start().starts_with('{'), "{json}");
}

#[test]
fn query_omitted_quiet_is_quiet_and_format_wins() {
    let run = |extra: &str| {
        let dir = scratch_with_csv();
        let job = toml_job(&format!(
            "schema_version = 1\n[[step]]\nop = \"convert\"\nfrom = \"raw/table.csv\"\nto = \"a.tet\"\n\
             [[step]]\nop = \"query\"\ntet = \"a.tet\"\nexecute = true\n{extra}\n\
             query = {{ dataset = \"temp\", mean = [] }}\n"
        ))
        .unwrap();
        run_job(&job, &opts(dir.path())).unwrap().steps[1]
            .stdout
            .clone()
            .unwrap()
    };
    let default = run("");
    assert_eq!(default.trim_end().lines().count(), 1, "{default}");
    assert_eq!(default, run("quiet = true"));
    assert_ne!(default, run("quiet = false"));
    // Explicit format is allowed without `quiet` and is not overridden by the default.
    let json = run("format = \"json\"");
    assert!(json.trim_start().starts_with('{'), "{json}");
}

#[test]
fn quiet_true_still_conflicts_with_json_or_format_but_omitted_does_not() {
    for (op, body) in [
        ("info", "path = \"a\"\njson = true\nquiet = true"),
        ("query", "doc = \"q.json\"\nformat = \"json\"\nquiet = true"),
    ] {
        let msg = parse_err(&format!(
            "schema_version = 1\n[[step]]\nop = \"{op}\"\n{body}\n"
        ));
        assert!(msg.contains("conflict"), "{op}: {msg}");
    }
    for (op, body) in [
        ("info", "path = \"a\"\njson = true"),
        ("info", "path = \"a\"\njson = true\nquiet = false"),
        ("query", "doc = \"q.json\"\nformat = \"json\""),
        (
            "query",
            "doc = \"q.json\"\nformat = \"json\"\nquiet = false",
        ),
    ] {
        toml_job(&format!(
            "schema_version = 1\n[[step]]\nop = \"{op}\"\n{body}\n"
        ))
        .unwrap_or_else(|e| panic!("{op} {body}: {e}"));
    }
}

#[test]
fn scaffold_format_is_parsed_once_at_deserialize_time() {
    let job = toml_job(
        "schema_version = 1\n[[step]]\nop = \"semantics.scaffold\"\npath = \"a.tet\"\nformat = \"JSON\"\n",
    )
    .unwrap();
    let Step::SemanticsScaffold(s) = &job.steps[0] else {
        panic!("expected scaffold");
    };
    assert_eq!(s.format, Some(crate::catalog::ScaffoldFormat::Json));
    let msg = parse_err(
        "schema_version = 1\n[[step]]\nop = \"semantics.scaffold\"\npath = \"a.tet\"\nformat = \"yaml\"\n",
    );
    assert!(msg.contains("step 1 (semantics.scaffold)"), "{msg}");
    assert!(msg.contains("toml or json"), "{msg}");
}

#[test]
fn step_errors_carry_the_resolved_path() {
    let dir = tempfile::tempdir().unwrap();
    let job = toml_job(
        "schema_version = 1\n[[step]]\nop = \"ingest\"\nfrom = \"gone.tet\"\nto = \"p.tet.d\"\n",
    )
    .unwrap();
    let err = run_job(&job, &opts(dir.path())).unwrap_err();
    let JobError::Step { path, message, .. } = &err else {
        panic!("expected Step, got {err}");
    };
    assert_eq!(path.as_deref(), Some(dir.path().join("gone.tet").as_path()));
    assert_eq!(message, "input not found");
    assert_eq!(
        err.to_string(),
        format!(
            "step 1 (ingest) failed: input not found: {}",
            dir.path().join("gone.tet").display()
        )
    );
}

#[test]
fn ingest_creates_a_missing_package_regardless_of_create() {
    for create in ["", "create = false\n", "create = true\n"] {
        let dir = scratch_with_csv();
        let job = toml_job(&format!(
            "schema_version = 1\n[[step]]\nop = \"convert\"\nfrom = \"raw/table.csv\"\nto = \"a.tet\"\n\
             [[step]]\nop = \"ingest\"\nfrom = \"a.tet\"\nto = \"deep/p.tet.d\"\n{create}"
        ))
        .unwrap();
        run_job(&job, &opts(dir.path())).unwrap();
        assert!(dir.path().join("deep/p.tet.d/PROTOCOL").is_file());
    }
}

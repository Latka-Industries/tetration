# Job files (`tet run`)

A job file is an ordered list of stock `tet` steps in TOML or JSON. `tet run job.toml` runs them top to bottom and stops at the first failure.

**This is not Airflow.** No DAG, conditionals, loops, retries, schedules, or `on_error = continue`. It is also **not query-pipeline ETL**: chained transforms inside one query belong in the query `pipeline` wire ([`query_engine.md`](query_engine.md)). A job only sequences whole CLI verbs (convert, verify, query, publish, …).

```bash
tet run job.toml                    # run; relative paths resolve against the job file's directory
tet run job.toml --dry-run          # validate + print the plan; execute nothing
tet run job.toml --root /data/run1  # resolve relative paths against DIR instead
```

| Flag        | Effect                                                                       |
| ----------- | ---------------------------------------------------------------------------- |
| `--dry-run` | Parse, validate (including query documents), print each step; run nothing    |
| `--root DIR`| Base directory for relative paths (default: the job file's parent directory) |

`s3://…` values (convert `from`, query `tet`, gc `target`) are passed through untouched.

## Schema (`schema_version = 1`)

```toml
schema_version = 1

[[step]]
op = "convert"
from = "raw/table.csv"
to = "work/table.tet"

[[step]]
op = "verify"
path = "work/table.tet"
quiet = true

[[step]]
op = "query"
tet = "work/table.tet"
execute = true
quiet = true
query = { dataset = "temp", mean = [] }
```

JSON uses `"steps": [ { "op": "…", … } ]`. Either spelling (`step` / `steps`) is accepted in either format. Unknown top-level keys, unknown `op` values, and unknown step fields are errors. A job needs at least one step.

Example files: [`fixtures/jobs/`](../fixtures/jobs/).

## Ops (closed set)

Field names mirror the CLI flags (`--dry-run` → `dry_run`). Booleans default to `false`.

| `op`                      | Fields (besides `op`)                                                                                       | CLI                    |
| ------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------- |
| `info`                    | `path`, `json`, `quiet`, `all`, `layout`, `execution`, `datasets`, `chunks`, `history`, `limit`, `dataset`, `grep`, `metadata` | `tet info`             |
| `verify`                  | `path`, `json`, `quiet`, `repair`, `deep`. A failed verify fails the step.                                  | `tet verify`           |
| `repair`                  | `path`, `json`, `dry_run`, `apply = ["CODE", …]` (plan only when `apply` is empty)                          | `tet repair`           |
| `query`                   | `doc` **or** `query = { … }` (exclusive); `tet`, `execute`, `format`, `quiet`, `preview`, `spill_allow`, `device` | `tet query`            |
| `convert`                 | `from`, `to`, `jobs`, `strict`, `chunk_mib`                                                                 | `tet convert`          |
| `export`                  | `from`, `to`                                                                                                | `tet export`           |
| `index`                   | `path`, `clear`, `quiet`                                                                                    | `tet index`            |
| `semantics.apply`         | `path`, `from` (the semantics TOML), `annotate`, `output`                                                   | `tet semantics apply`  |
| `semantics.scaffold`      | `path`, `output`, `format` (`toml` / `json`)                                                                | `tet semantics scaffold` |
| `semantics.set`           | `path`, `dataset`, `set = { key = value, … }`, `clear = ["key", …]`                                         | `tet semantics set`    |
| `semantics.wizard`        | `path`, `dataset`, `output`, `no_apply` — interactive, see below                                            | `tet semantics wizard` |
| `semantics.lint`          | `path`, `generation`                                                                                        | `tet semantics lint`   |
| `semantics.package-write` | `path`, `generation`, `no_manifest`                                                                         | `tet semantics package-write` |
| `ingest`                  | `from` (path or list), `to` (`.tet.d`), `jobs`, `create`, `quiet`                                           | `tet ingest`           |
| `publish`                 | `package`, `generation`, `parts`, `abort`, `dry_run`, `replace`, `quiet`                                    | `tet publish`          |
| `compact`                 | `package`, `generation`, `parts`, `replace`, `quiet`                                                        | `tet compact`          |
| `pack`                    | `from` (`.tet.d`), `to` (`.tet`), `generation`, `remove_package`, `quiet`                                   | `tet pack`             |
| `gc`                      | `target`, `apply`, `protect_generation = [G, …]`, `quiet`                                                   | `tet gc`               |

Not in v1: `worker`, `qhist`, `semantics package-show`, Python steps.

Notes:

- **Query steps** take exactly one of `doc = "path.json|toml"` or an inline `query` table with the same schema as a query file. `execute = true` needs `tet`; `preview`, `spill_allow`, and `device` need `execute = true`. Query history is **off** in jobs.
- **`semantics.set`** values may be strings, numbers, or booleans (`set = { frame_rate = 30 }` ≡ `--set frame_rate=30`).
- **Output directories** for `convert`, `export`, `pack`, `ingest`, and file outputs are created as needed.
- **`semantics.wizard`** prompts on stdin. It is refused (before any step runs, including with `--dry-run`) unless stdin is a terminal. Library callers opt in with `allow_interactive`.

## Execution model

1. **Pre-flight** (also under `--dry-run`): schema and flag rules, then every query document is loaded and validated and the wizard rule is applied. A failure here means **no step ran**.
2. **Run** steps in order. The first error stops the job; later steps do not run. Exit code is non-zero and stderr names the step (`error: step 2 (verify) failed: …`).
3. Each step prints what the matching command would: query / info / verify output goes to **stdout**; progress and summary lines go to **stderr** (indented under `✓ step N`).

Steps call the same library functions as the CLI (no subprocess), so a step behaves like the equivalent `tet` command.

## Library API

`tetration::job` is the same engine, quiet and structured (no `indicatif`, no printing):

```rust
use tetration::job::{RunJobOptions, parse_job, JobFormat, run_job, run_job_path};

let report = run_job_path(
    std::path::Path::new("job.toml"),
    &RunJobOptions { dry_run: false, ..Default::default() },
)?;
for step in &report.steps {
    println!("{} {} ({:.2}s)", step.op, step.description, step.elapsed_secs);
    // step.stdout / step.note / step.data (convert / export / verify reports, query response)
}
```

| Item | Role |
| ---- | ---- |
| `parse_job(text, JobFormat)` | Parse + statically validate TOML / JSON → `Job` |
| `run_job(&Job, &RunJobOptions)` | Pre-flight, then run (or plan with `dry_run`) → `JobReport` |
| `run_job_path(path, &RunJobOptions)` | Read file, default `root` to its parent, run |
| `run_job_observed` / `run_job_path_observed` | Same, with a callback after each step |
| `JobError::Step { index, op, message, completed }` | Fail-fast error; `completed` holds the steps that finished |

`RunJobOptions`: `root`, `dry_run`, `allow_interactive` (default `false`), `record_query_history` (default `false`).

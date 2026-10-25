# Job files (`tet run`)

A job file is an ordered list of stock `tet` steps in TOML or JSON. `tet run job.toml` runs them top to bottom and stops at the first failure.

**This is not Airflow.** No DAG, conditionals, loops, retries, schedules, or `on_error = continue`. It is also **not query-pipeline ETL**: chained transforms inside one query belong in the query `pipeline` wire ([`query_engine.md`](query_engine.md)). A job only sequences whole CLI verbs (convert, verify, query, publish, …).

```bash
tet run job.toml                    # run; relative paths resolve against the job file's directory
tet run job.toml --dry-run          # validate + print the plan; execute nothing
tet run job.toml --root /data/run1  # resolve relative paths against DIR instead
tet run job.toml --json > report.json  # full JobReport as JSON on stdout
```

| Flag        | Effect                                                                       |
| ----------- | ---------------------------------------------------------------------------- |
| `--dry-run` | Parse, validate (including query documents), print each step; run nothing    |
| `--root DIR`| Base directory for relative paths (default: the job file's parent directory) |
| `--json`    | Print the full `JobReport` as pretty JSON on **stdout**; progress stays on stderr (see [Machine-readable output](#machine-readable-output)) |

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

[[step]]
op = "query"
tet = "work/table.tet"
execute = true
query = { dataset = "temp", mean = [] }
```

JSON uses `"steps": [ { "op": "…", … } ]`. Either spelling (`step` / `steps`) is accepted in either format. Unknown top-level keys, unknown `op` values, and unknown step fields are errors. A job needs at least one step.

Example files: [`fixtures/jobs/`](../fixtures/jobs/).

## Ops (closed set)

Field names mirror the CLI flags (`--dry-run` → `dry_run`). Booleans default to `false`, except `quiet` on `info` / `verify` / `query` (see below).

**`quiet` defaults are job-specific.** For **`info`**, **`verify`**, and **`query`** only, an omitted `quiet` means **quiet** (one-line summary) so job logs stay short. `quiet = false` restores the full text the CLI prints. `json = true` (info / verify) or an explicit `format = "…"` (query) always wins over the quiet default, and `quiet = true` together with `json` / `format` is still a conflict. Every other op keeps `quiet = false` as the default (ingest, publish, compact, index, pack, gc: notes are printed unless you set `quiet = true`).

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
| `ingest`                  | `from` (path or list), `to` (`.tet.d`), `jobs`, `create` (no-op), `quiet`                                         | `tet ingest`           |
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
2. **Run** steps in order. The first error stops the job; later steps do not run. Exit code is non-zero and stderr names the step and, when the failing operation touched a file, its resolved absolute path: `error: step 2 (verify) failed: No such file or directory (os error 2): /abs/path/work/x.tet`.
3. Progress on **stderr**: `[N/TOTAL] <description>  ok  0.12s` (`plan` under `--dry-run`). Quiet one-line results (`info` / `verify` / `query` defaults) nest under the step as a light vertical kv table (`UTF8_NO_BORDERS`). Multi-line dumps (`quiet = false`, JSON, tables) still go to **stdout**. Colour only when stderr is a terminal.

Steps call the same library functions as the CLI (no subprocess), so a step behaves like the equivalent `tet` command.

## Machine-readable output

`tet run job.toml --json` prints the whole `JobReport` (`dry_run`, `root`, `steps`, `elapsed_secs`) as pretty JSON on **stdout** after the job finishes. Nothing else goes to stdout: the `[N/TOTAL]` progress lines, kv tables, and notes still print on **stderr** so a human watching sees progress, and multi-line step dumps (`quiet = false`, tables) move to stderr too (they are also in each step's `stdout` field). Combine with `--dry-run` for a planned report (`status = "planned"`, `data = null`).

```bash
tet run fixtures/jobs/csv_verify_query.toml --json 2>/dev/null | jq '.steps[].data | keys'
```

Each entry in `steps` is a `StepReport`: `index`, `op`, `status`, `description`, `elapsed_secs`, `stdout` (text the CLI prints), `note` (text for stderr), and **`data`**.

**`step.data` is the programmatic grab bag.** Every executed step fills it with the structured result (serde of the library report type where one exists), so tooling never has to parse `stdout` / `note`. The Python bindings (tet-py) wrap the same `JobReport`.

| `op` | `data` |
| ---- | ------ |
| `info` | The `tet info --json` envelope (`path`, `file_len`, `filter`, `summary`), regardless of `json` / `quiet` |
| `verify` | `TetVerifyReport` |
| `repair` | `RepairPlan` (no `apply`) or `TetRepairReport` |
| `query` | `QueryResponse` |
| `convert` / `export` | `ConvertReport` / `ExportReport` |
| `index` | `{ ok, path, cleared, packed_sha256, zone_maps_sha256, index_present }` (`cleared = true` carries only `ok` / `path` / `cleared`) |
| `semantics.apply` | `SemanticsApplyReport` (`stamped`, `profiles_left`, `units_left`) |
| `semantics.scaffold` | `{ ok, path, format, output }` (`output` is `null` when the text went to `stdout`) |
| `semantics.set` | `{ ok, path, dataset, set, clear }` |
| `semantics.wizard` | `WizardResult` (`dataset`, `toml_path`, `apply`) |
| `semantics.lint` | `SemanticsLintReport` (`findings`) |
| `semantics.package-write` | `{ ok, path, generation, sha256, semantics }` (`semantics` is the written `SEMANTICS.json` document) |
| `ingest` | `{ package, jobs, parts: [{ source, part }] }` (`part` is `PartMeta`) |
| `publish` / `compact` | `{ mode: "published", package, manifest }` (`GenerationManifest`), `{ mode: "dry_run", package, validation }` (`PublishValidation`), or `{ mode: "aborted", package, generation, current_before }` |
| `pack` | `{ ok, package, output, generation, remove_package }` |
| `gc` | `{ target, apply, plan }` (local, `GcPlan`) or `{ target, apply, report }` (`s3://`, `PruneReport`) |

`data` is set even when `quiet = true` suppresses the matching `note`.

**Failures.** The exit code is still 1 and stderr still says `error: step N (op) failed: …`. If at least one step ran, stdout additionally holds the steps that completed as a `JobReport` plus an `error` object (`index`, `op`, `message`, `path`), so partial results are not lost. Parse, pre-flight, and I/O errors happen before step 1 and print nothing on stdout. Always check the exit code (or the `error` key) before treating stdout as a success report.

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
    // step.stdout / step.note / step.data (structured result for every executed step)
}
```

| Item | Role |
| ---- | ---- |
| `parse_job(text, JobFormat)` | Parse + statically validate TOML / JSON → `Job` |
| `run_job(&Job, &RunJobOptions)` | Pre-flight, then run (or plan with `dry_run`) → `JobReport` |
| `run_job_path(path, &RunJobOptions)` | Read file, default `root` to its parent, run |
| `run_job_observed` / `run_job_path_observed` | Same, with a callback `(&StepReport, total_steps)` after each step |
| `JobError::Step { index, op, message, path, completed }` | Fail-fast error; `path` is the resolved path the failing operation touched (shown after the message), `completed` holds the steps that finished |

Library callers get the same quiet defaults (`StepReport::stdout` is the one-line form unless `quiet = false`).

`RunJobOptions`: `root`, `dry_run`, `allow_interactive` (default `false`), `record_query_history` (default `false`).

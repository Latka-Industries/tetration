# `query` — JSON/TOML query engine

Control plane for analytics over mmap'd `.tet` files: parse documents, build [`ReadPlan`](types/README.md), decode intersecting chunks, run reductions/transforms, return [`QueryResponse`](types/README.md).

Full wire spec: [`docs/query_engine.md`](../../docs/query_engine.md).

## End-to-end flow

```text
QueryDocument (JSON/TOML)
    → validate (document.rs, document_wire.rs, document_toml.rs)
    → resolve dataset + axes (resolve_selection.rs, resolve_axes.rs)
    → ReadPlan (plan/)
    → execute (engine/) or plan-only
         → decode chunks (decode/)
         → materialize logical buffer OR spill OR streaming fold (materialize/, fold/)
         → optional GPU (gpu/, device.rs)
         → optional transform two-pass (transform/) — ram, spill, or sidecar `.tet` publish
    → QueryResponse + CLI format (cli/)
         OR embedder dense export (embed_materialize.rs) — full buffers without preview cap
```

## Public API (library)

| Entry                                                        | Role                                               |
| ------------------------------------------------------------ | -------------------------------------------------- |
| `parse_query_json` / `parse_query_toml` / `parse_query_text` | Parse + limits                                     |
| `validate_query`                                             | Schema / policy checks                             |
| `plan_query_with_tet_mmap_ex`                                | Attach catalog + `ReadPlan`                        |
| `plan_read_for_document`                                     | Resolve `ReadPlan` only (no execution)             |
| `execute_query_document` / `execute_query_json`              | Plan + run operation                               |
| `materialize_query_selection`                                | Full dense decode (selection-only; budget preflight) |
| `estimate_query_selection_bytes`                             | Logical dense size without decode                    |
| `materialize_query_transform_ram`                            | Full dense transform (`write: ram`, `f32`/`f64`)   |
| `format_query_response`                                      | CLI-oriented stdout (also used by embedders)       |

Re-exported from [`prelude`](../lib.rs).

## Top-level files (not directories)

| File                   | Role                                                           |
| ---------------------- | -------------------------------------------------------------- |
| `document.rs`          | JSON parse, `QueryLimits`, `detect_query_input_format`         |
| `document_toml.rs`     | TOML parse (same `QueryDocument`)                              |
| `document_wire.rs`     | Serde wire shapes for flat query keys (`mean`, `transform`, …) |
| `execute.rs`           | `ExecuteQueryOptions`, thin wrapper over engine                |
| `embed_materialize.rs` | Dense tensor export for embedders (no preview cap)             |
| `dispatch.rs`          | Dtype-specific dispatch: materialize vs fold vs spill          |
| `device.rs`            | `execution.device` routing (`cpu`, `cuda`, `metal`, `auto`, …) |
| `resolve_selection.rs` | Index/coord-label selection → logical ranges                   |
| `resolve_axes.rs`      | Named axis slices                                              |

## Submodules

| Directory                               | Role                                                 |
| --------------------------------------- | ---------------------------------------------------- |
| [`types/`](types/README.md)             | `QueryDocument`, `ReadPlan`, `QueryResponse`, errors |
| [`plan/`](plan/README.md)               | Selection geometry → chunk list                      |
| [`engine/`](engine/README.md)           | Orchestration, budget, spill policy, `run`           |
| [`decode/`](decode/README.md)           | Mmap slice + per-chunk decode                        |
| [`materialize/`](materialize/README.md) | Dense logical tensors, spill export                  |
| [`fold/`](fold/README.md)               | Scalar/partial reductions, parallel merge            |
| [`transform/`](transform/README.md)     | Two-pass zscore, minmax, softmax, …                  |
| [`gpu/`](gpu/README.md)                 | Optional CUDA / ROCm / Metal reductions              |
| [`cli/`](cli/README.md)                 | `tet query` / `tet info` / `qhist` formatting        |

## CLI mapping

- `tet query -t file.tet` → plan only
- `tet query -t file.tet -x` → `execute_query_json`
- `--format quiet|stats|table|plan` → `cli/output/`

## Security note

Query JSON/TOML is untrusted input: size caps, spill path allowlists, and validation live in `document` + `engine::spill_policy`. See docs for embedder guidance.

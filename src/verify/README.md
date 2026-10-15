# `verify` — read-only `.tet` health checks

Runs **checks** beyond catalog parsing: layout consistency, chunk payload bounds, optional full decode, footer JSON validity. Emits [`TetVerifyReport`](report.rs) with findings and repair suggestions.

Distinct from [`catalog`](../catalog/README.md): parsing may succeed while verify fails (e.g. truncated payload).

## Public API

- `verify_tet_file` / `verify_tet_file_with_options`
- `verify_tet_bytes` — in-memory buffer + optional path for messages
- `VerifyOptions` — `--deep` decode all chunks vs quick sample (`DEEP_DECODE_MAX_CHUNKS`)
- Formatters: `format_verify_text`, `format_verify_json`, `format_verify_quiet`

## Submodules

| File           | Role                                              |
| -------------- | ------------------------------------------------- |
| `run.rs`       | Orchestrate all check phases                      |
| `datasets.rs`  | Dataset record vs chunk grid / byte lengths       |
| `chunks.rs`    | Index entries, payload spans, codec decode sample |
| `footer.rs`    | `THST` magic, history JSON, metadata              |
| `recommend.rs` | Map findings → `VerifyRecommendation` codes       |
| `report.rs`    | `TetVerifyReport`, `VerifyFinding`, severity      |
| `options.rs`   | Deep vs quick, limits                             |
| `format.rs`    | CLI output                                        |

## Repair integration

Read-only by default. [`repair`](../repair/README.md) applies fixes for codes like `footer_invalid`; verify enriches recommendations with `tet repair …` command hints.

## CLI

`tet verify` → exit **1** on failure (CI-friendly). `tet verify --repair` runs safe repairs after verify.

## Fuzzing

LibFuzzer harness over `verify_tet_bytes` (untrusted bytes must not panic):

```bash
mise fuzz
# or: cargo +nightly fuzz run verify_tet_bytes
```

See [`fuzz/README.md`](../../fuzz/README.md) (THI-376).

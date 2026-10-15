# `verify` — read-only `.tet` health checks

Runs **checks** beyond catalog parsing: layout consistency, chunk payload bounds, optional full decode, footer JSON validity. Emits [`TetVerifyReport`](report.rs) with findings and repair suggestions.

Distinct from [`catalog`](../catalog/README.md): parsing may succeed while verify fails (e.g. truncated payload).

## Public API

- `verify_tet_file` / `verify_tet_file_with_options`
- `verify_tet_bytes` — in-memory buffer + optional path for messages
- `verify_tet_package` / `verify_tet_package_with_options` / `looks_like_tet_package` — `.tet.d` (THI-466)
- `VerifyOptions` — `--deep` decode all chunks vs quick sample (`DEEP_DECODE_MAX_CHUNKS`)
- Formatters: `format_verify_text`, `format_verify_json`, `format_verify_quiet`

## Submodules

| File           | Role                                              |
| -------------- | ------------------------------------------------- |
| `run.rs`       | Orchestrate all check phases                      |
| `package.rs`   | `.tet.d` PROTOCOL / CURRENT / MANIFEST / packed   |
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

See [`fuzz/README.md`](../../fuzz/README.md) (THI-376). Optional manual CI smoke (60s):
[`.github/workflows/fuzz-smoke.yml`](../../.github/workflows/fuzz-smoke.yml) (THI-379).

## Reject corpus (THI-378)

Checked-in malformed `.tet` files under [`fixtures/small/tet/reject/`](../../fixtures/small/tet/reject/) pin TIDX / payload failure classes. CI asserts each fails `verify_tet_bytes` with a stable finding `check` (`src/tests/reject_corpus.rs`). `mise fuzz-reseed` copies them into the fuzz corpora.

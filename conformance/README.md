# Tetration conformance pack (0.5.2)

**Status:** normative pack format for THI-592.  
**SPEC:** [`SPEC/README.md`](../SPEC/README.md) · [`SPEC/COMPATIBILITY.md`](../SPEC/COMPATIBILITY.md).

This directory is the **portable** description of “compatible”: accept fixtures with expected query answers, reject fixtures that must fail verify or query parse. The reference Rust reader gates CI by running [`src/tests/conformance.rs`](../src/tests/conformance.rs) against [`manifest.v1.json`](manifest.v1.json).

External implementers **SHOULD** consume the same manifest (or an equivalent generated from it). Matching the reference answers on `accept` and failing on `reject_*` is how you claim compatibility for the covered surface.

## Layout

```text
conformance/
  README.md              # this file
  manifest.v1.json       # machine-readable case index (paths relative to repo root)
  reject/queries/        # query documents that MUST fail parse/validate
```

Binary `.tet` accept/reject files live under [`fixtures/small/tet/`](../fixtures/small/tet/) (already tracked). Query profiles under [`fixtures/queries/`](../fixtures/queries/). The manifest points at those paths so the pack does not duplicate payloads.

## Manifest schema (`pack_version: 1`)

Top-level object:

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `pack_version` | `1` | This schema |
| `spec_milestone` | string | e.g. `"0.5.2"` |
| `accept` | array | Cases that must succeed |
| `reject_verify` | array | `.tet` files that must fail `verify` |
| `reject_query` | array | Query docs that must fail parse/validate |

### `accept[]`

| Field | Required | Meaning |
| ----- | -------- | ------- |
| `id` | yes | Stable case id |
| `tet` | yes | Path to `.tet` (repo-root relative) |
| `verify` | no | `"ok"` → verify must succeed (default when omitted and no `query`) |
| `query` | no | Path to query JSON (flat wire) |
| `expect` | if `query` | Map of dotted result paths → number/bool/string |

Supported `expect` keys today (reference runner):

- `execution.operation_mean`
- `execution.operation_sum`
- `execution.operation_var`
- `execution.operation_element_count`

Float compares use absolute tolerance `1e-5` unless `expect_tol` is set on the case.

### `reject_verify[]`

| Field | Required | Meaning |
| ----- | -------- | ------- |
| `id` | yes | Stable case id |
| `tet` | yes | Path to malformed `.tet` |
| `expect_check` | yes | Verify finding `check` name that must fail (`ok == false`) |

Finding names match the reference `tet verify` / `verify_tet_bytes` report (see reject corpus).

### `reject_query[]`

| Field | Required | Meaning |
| ----- | -------- | ------- |
| `id` | yes | Stable case id |
| `query` | yes | Path to query JSON that must be rejected |
| `expect_error_substr` | yes | Case-insensitive substring of the validation error |

These encode the closed-key / refuse-unknown policy in [`SPEC/COMPATIBILITY.md`](../SPEC/COMPATIBILITY.md).

## Running (reference)

```bash
# Full lib suite includes conformance
cargo test --lib conformance

# Or mise (if configured)
mise run check
```

CI runs `cargo test` on non-markdown PRs; the conformance module is part of that gate.

## Second-language smoke (optional)

Python stdlib reader that opens a fixture and reads catalog + one chunk: [`smoke/`](smoke/README.md) (THI-589). Not part of pack_version answer equality; it proves the binary layout section of the SPEC is implementable without Rust.

## Extending the pack

1. Add or reuse a fixture under `fixtures/`.
2. Append a case to `manifest.v1.json`.
3. For new reject query shapes, drop JSON under `conformance/reject/queries/`.
4. Run `cargo test --lib conformance` and update this README if new `expect` keys are needed.

Do **not** put Linear / crate-internal “out of scope” framing in public site docs; keep that in Linear and contributor notes.

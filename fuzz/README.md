# Fuzzing Tetration

In-memory layout checks via [`verify_tet_bytes`](../src/verify/run.rs) (no mmap).
Tracks [THI-376](https://linear.app/thicclatka/issue/THI-376) / parent [THI-375](https://linear.app/thicclatka/issue/THI-375).

Package (`.tet.d`) checks via [`verify_tet_package`](../src/verify/package.rs) —
[THI-481](https://linear.app/thicclatka/issue/THI-481).

Structured **valid-write** offset/round-trip invariants (CI `proptest`, not libFuzzer) live in
[`src/tests/write_invariants.rs`](../src/tests/write_invariants.rs) — [THI-377](https://linear.app/thicclatka/issue/THI-377).

Conformance **reject** fixtures (TIDX / payload faults) live under
[`fixtures/small/tet/reject/`](../fixtures/small/tet/reject/) and are tested in
[`src/tests/reject_corpus.rs`](../src/tests/reject_corpus.rs) — [THI-378](https://linear.app/thicclatka/issue/THI-378).
`mise fuzz-reseed` copies them into the file corpora and packs selected ones into package seeds.

## Setup

```bash
cargo install cargo-fuzz
rustup component add rust-src --toolchain nightly
```

`cargo-fuzz` needs **nightly** (AddressSanitizer uses `-Z` flags). Plain
`cargo fuzz …` on stable will fail with `the option Z is only accepted on the
nightly compiler`.

## Targets

| Target | Options | Notes |
| ------ | ------- | ----- |
| `verify_tet_bytes` | quick scan (`deep_decode: false`) | Hot loop — prefer this for file layout |
| `verify_tet_bytes_deep` | `deep_decode: true` | Slower; decode walk |
| `verify_tet_package` | package tree from length-prefixed blob | PROTOCOL / CURRENT / MANIFEST / packed.tet |
| `parse_query_json` | UTF-8 query document text (capped) | Closed wire parse; no panic (THI-604; window pad/step/align/`min_count` seeds THI-607; label-range / `label_filter` seeds THI-613/614). **Post-0.6 shapes covered** via conformance accept/reject + named stubs (window, `group_by` labels, align, filter) — see fuzz-reseed notes. |
| `parse_job` | UTF-8 job TOML/JSON text (capped) | `parse_job` parse + static validate only; no executor (THI-651). Seeds: `fixtures/jobs/*` + reject stubs (unknown op, bad `schema_version`, huge nesting). |
| `secondary_index_json` | `INDEX.json` bytes (capped) | Deserialize + encode / kind filters; unknown `kind` → `Unknown` (THI-613) |
| `secondary_index_seal` | tiny packed `.tet` from mutated tiles | Write + `prepare_secondary_index` / embed; value-postings cardinality early-abort |

### Package input shape (`verify_tet_package`)

Not tar — four little-endian `u32` length-prefixed fields so mutations hit each
artifact independently:

```text
[u32 LE][PROTOCOL bytes]
[u32 LE][CURRENT bytes]      # empty length → omit CURRENT
[u32 LE][MANIFEST.json]
[u32 LE][packed.tet bytes]   # clamped to 256 KiB in the harness
```

The harness materializes `fuzz.tet.d` under a tempfile (`generations/00000001/`,
empty `staging/` + `publishing/`) and calls `verify_tet_package`. Seeds are
`*.pkgfuzz` under `corpus/verify_tet_package/` (built by `mise fuzz-reseed` via
[`scripts/encode_pkgfuzz.py`](scripts/encode_pkgfuzz.py)).

## Run

```bash
mise fuzz              # preferred (quick file target)
mise fuzz-package      # package target (THI-481)
mise fuzz-query        # query wire (THI-604)
mise fuzz-job          # job file parse/validate (THI-651)
mise fuzz-index        # INDEX.json / sorted_axis_range (THI-613)
mise fuzz-index-seal   # seal-time value-postings from tiny packed.tet
# or:
cargo +nightly fuzz run verify_tet_bytes
cargo +nightly fuzz run verify_tet_bytes_deep
cargo +nightly fuzz run verify_tet_package
cargo +nightly fuzz run parse_query_json
cargo +nightly fuzz run parse_job
cargo +nightly fuzz run secondary_index_json
cargo +nightly fuzz run secondary_index_seal
```

Seeds are under `corpus/verify_tet_bytes/`, `corpus/verify_tet_bytes_deep/`
(`*.tet` from `fixtures/small/tet/` plus tiny reject stubs),
`corpus/verify_tet_package/` (`*.pkgfuzz`), `corpus/parse_query_json/` (`*.json`),
`corpus/parse_job/` (`*.json` / `*.toml` from `fixtures/jobs/` plus reject stubs),
`corpus/secondary_index_json/` (`INDEX.json` stubs), and
`corpus/secondary_index_seal/` (`*.bin` tile headers + payloads). LibFuzzer mutates more
inputs beside them; those hash-named files and `artifacts/` are gitignored.

```bash
mise fuzz-clean    # drop mutated corpus + crash artifacts; keep seed extensions
mise fuzz-reseed   # refresh seeds via fuzz/scripts/fuzz-reseed.sh
```

## Build-only smoke

```bash
mise fuzz-build
# or:
cargo +nightly fuzz build verify_tet_bytes
cargo +nightly fuzz build verify_tet_package
```

## CI / local smoke (THI-379 / THI-481 / THI-651)

Manual GitHub Actions job [`.github/workflows/fuzz-smoke.yml`](../.github/workflows/fuzz-smoke.yml)
(`workflow_dispatch` only — no nightly schedule):

- Ubuntu + nightly toolchain
- `verify_tet_bytes`, `verify_tet_package`, `parse_query_json`, `parse_job`, and index targets each for 60s
- Not part of the PR Test Suite / MSRV matrix

```bash
# local equivalents (prefer these day-to-day)
mise fuzz-smoke
mise fuzz-package-smoke
mise fuzz-query-smoke
mise fuzz-job-smoke
mise fuzz-index-seal-smoke
# or:
cargo +nightly fuzz run verify_tet_bytes -- -max_total_time=60 -timeout=10
cargo +nightly fuzz run verify_tet_package -- -max_total_time=60 -timeout=10
cargo +nightly fuzz run parse_job -- -max_total_time=60 -timeout=10
cargo +nightly fuzz run secondary_index_seal -- -max_total_time=60 -timeout=10
```

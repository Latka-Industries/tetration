# Fuzzing Tetration

In-memory layout checks via [`verify_tet_bytes`](../src/verify/run.rs) (no mmap).
Tracks [THI-376](https://linear.app/thicclatka/issue/THI-376) / parent [THI-375](https://linear.app/thicclatka/issue/THI-375).

Structured **valid-write** offset/round-trip invariants (CI `proptest`, not libFuzzer) live in
[`src/tests/write_invariants.rs`](../src/tests/write_invariants.rs) — [THI-377](https://linear.app/thicclatka/issue/THI-377).

Conformance **reject** fixtures (TIDX / payload faults) live under
[`fixtures/small/tet/reject/`](../fixtures/small/tet/reject/) and are tested in
[`src/tests/reject_corpus.rs`](../src/tests/reject_corpus.rs) — [THI-378](https://linear.app/thicclatka/issue/THI-378).
`mise fuzz-reseed` copies them into both corpora below.

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
| `verify_tet_bytes` | quick scan (`deep_decode: false`) | Hot loop — prefer this |
| `verify_tet_bytes_deep` | `deep_decode: true` | Slower; decode walk |

## Run

```bash
mise fuzz              # preferred (quick target)
# or:
cargo +nightly fuzz run verify_tet_bytes
cargo +nightly fuzz run verify_tet_bytes_deep
```

Seeds are under `corpus/verify_tet_bytes/` and `corpus/verify_tet_bytes_deep/`
(`*.tet` from `fixtures/small/tet/` plus tiny reject stubs). LibFuzzer mutates
more inputs beside them; those hash-named files and `artifacts/` are gitignored.

```bash
mise fuzz-clean    # drop mutated corpus + crash artifacts; keep *.tet
mise fuzz-reseed   # refresh *.tet seeds from fixtures/small/tet/
```

## Build-only smoke

```bash
mise fuzz-build
# or:
cargo +nightly fuzz build verify_tet_bytes
```

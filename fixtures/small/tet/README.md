# Small `.tet` fixtures (verify / repair / query)

Tracked binaries for **manual** CLI checks and `src/tests/small_tet_fixtures.rs`.

| File                 | Role                                                                     |
| -------------------- | ------------------------------------------------------------------------ |
| `sample.tet`         | Multichunk `f32` `[2,3]` / tiles `[2,2]` (values 1..6)                   |
| `embedded_zone.tet`  | 1-D `f32` `[4]` / chunks `[2]` with embedded ZONE/INDEX (`history_version=2`, 0.7.0) |
| `large.tet`          | `f32` `[34,64]` / `[4,4]` → **144** chunks (quick verify samples 128)    |
| `plan.tet`           | Same as `sample` + **invalid** history footer (`tet repair` plan target) |
| `multichunk_u8.tet`  | Multichunk `u8`, values 1..6                                             |
| `multichunk_u32.tet` | Multichunk `u32`, values 1..6                                            |
| `multichunk_f16.tet` | Multichunk `f16`, values 1..6                                            |
| `recording.tet`      | Media worked example: `frames` `[6,2,2]` + `frame_rate` / `time_axis` (THI-583) |
| `group_by_labels.tet` | Multichunk `f32` `[2,3]` + repeated `col` labels (`east`,`east`,`west`) for label `group_by` (THI-596) |
| `align_pair.tet`     | Two datasets `a`/`b` for multi-dataset `align` checks (THI-594) |
| `reject/overflow_tile_extent.tet` | Malformed chunk coords (libFuzzer); `tet verify` must fail without panic |
| `reject/oom_dataset_count.tet` | Inflated superblock `dataset_count` (libFuzzer); must reject without OOM |
| `reject/zero_chunk_shape.tet` | Zero `chunk_shape` entry (libFuzzer); must fail without `div_ceil` panic |
| `reject/payload_offset_past_eof.tet` | TIDX `payload_offset` past EOF (THI-378) |
| `reject/duplicate_payload_offset.tet` | Two index rows share `payload_offset` |
| `reject/raw_stored_len_mismatch.tet` | Raw codec with `raw_byte_len != stored_byte_len` |
| `reject/chunk_coord_out_of_grid.tet` | Chunk grid coord outside `ceil(shape/chunk)` |
| `reject/truncated_payload.tet` | File truncated mid-payload |
| `reject/truncated_tidx.tet` | File truncated mid-TIDX |
| `reject/zstd_garbage_payload.tet` | Zstd frame bytes corrupted |
| `reject/inflated_stored_past_eof.tet` | `stored_byte_len` span past EOF |

Reject corpus tests: `src/tests/reject_corpus.rs`. Regenerate THI-378 cases:

```bash
UPDATE_REJECT_CORPUS=1 cargo test --lib regenerate_reject_corpus -- --ignored --nocapture
mise fuzz-reseed
```

**Dataset names:** `sample.tet` → **`temperature`**; `multichunk_*.tet` → **`a`**.

Example query profiles (JSON + TOML): [`fixtures/queries/`](../../queries/README.md). C ABI smoke: [`examples/ffi_query.c`](../../../examples/ffi_query.c) or `./.github/scripts/build-ffi-example.sh`.

## Manual (from repo root)

```bash
cargo build --release
BIN=./target/release/tet
SAMPLE=fixtures/small/tet/sample.tet
U8=fixtures/small/tet/multichunk_u8.tet

$BIN info $SAMPLE
$BIN verify $SAMPLE -q
$BIN verify fixtures/small/tet/large.tet          # ok; warns about skipped decode past chunk 128
$BIN verify --deep fixtures/small/tet/large.tet -q
$BIN repair fixtures/small/tet/plan.tet            # plan (footer_invalid)
$BIN repair fixtures/small/tet/plan.tet --apply footer_invalid

Q=fixtures/queries

# sample.tet — dataset "temperature" (mean of 1..6 = 3.5)
$BIN query $Q/mean_temperature.toml -t $SAMPLE -x -q

# slice value grid (2×3 tensor, values 1..6)
$BIN query $Q/slice_full_temperature.toml -t $SAMPLE -x --preview 6 --format table

# 2×2 sub-slice (values 1, 2, 4, 5)
$BIN query $Q/slice_2x2_temperature.toml -t $SAMPLE -x --preview 4 --format table

# multichunk_u8.tet — dataset "a" (sum of 1..6 = 21)
$BIN query $Q/sum_a.toml -t $U8 -x -q
$BIN query $Q/var_a.json -t $U8 -x -q
$BIN query '{"dataset":"a","sum":[]}' -t fixtures/small/tet/multichunk_u32.tet -x -q
$BIN query '{"dataset":"a","var":[]}' -t fixtures/small/tet/multichunk_f16.tet -x -q
```

Expected aggregates on 1..6: **sum=21**, **var≈2.916667** (population, `ddof=0`).

## Regenerate

```bash
UPDATE_SMALL_TET=1 cargo test --lib regenerate_tracked_small_tet_fixtures -- --ignored --nocapture
```

Then commit `fixtures/small/tet/*.tet`.

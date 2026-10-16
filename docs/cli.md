# `tet` CLI reference

Full flag lists: **`tet -h`** and **`tet <command> -h`** (always match the installed binary).

| Command                                            | Alias  | Role                                                        |
| -------------------------------------------------- | ------ | ----------------------------------------------------------- |
| [`tet info`](#tet-info) `<path.tet\|.tet.d>`       | —      | Summarize a file or package CURRENT (default: dataset table) |
| [`tet verify`](#tet-verify) `<path.tet\|.tet.d>`   | —      | Layout / package health check (exit 1 on failure); `--json` / `-q` |
| [`tet repair`](#tet-repair) `<path.tet\|.tet.d>`   | —      | Plan or apply safe in-place fixes (footer, stale publish)   |
| [`tet query`](#tet-query) `[QUERY]`                | `q`    | Validate JSON/TOML; optional catalog + execute against `-t` |
| [`tet qhist`](#tet-qhist) `[list\|run]`            | `hist` | Recent queries (platform cache; **not** the `.tet` footer)  |
| [`tet convert`](#tet-convert) `<in> <out.tet>`     | —      | HDF5 / NetCDF / Zarr v3 / NumPy / CSV → `.tet`              |
| [`tet export`](#tet-export) `<in.tet> <out.zarr/>` | —      | `.tet` → Zarr v3 directory store                            |
| [`tet ingest`](#tet-ingest--publish--compact--pack) | —    | Stage `.tet` inputs into a `.tet.d` package                 |
| [`tet publish`](#tet-ingest--publish--compact--pack) / [`compact`](#tet-ingest--publish--compact--pack) | — | Seal staging → generation / flip `CURRENT` |
| [`tet pack`](#tet-ingest--publish--compact--pack)  | —      | Copy sealed `packed.tet` → standalone `.tet`                |

## `tet info`

| Flag                                                                 | Effect                                                            |
| -------------------------------------------------------------------- | ----------------------------------------------------------------- |
| _(default)_                                                          | Dataset catalog table                                             |
| `--json`                                                             | Full pretty JSON (superblock, catalog, chunks, history)           |
| `-q`, `--quiet`                                                      | One-line summary                                                  |
| `--all`                                                              | All text sections                                                 |
| `--layout` / `--execution` / `--datasets` / `--chunks` / `--history` | One section each (`--history` = convert footer; not `qhist`)      |
| `--metadata`                                                         | Footer `dim_names` / `coords` previews under dataset rows         |
| `-n`, `--limit N`                                                    | Max chunk rows with `--chunks` or `--all` (default 32; `0` = all) |
| `--dataset`, `--grep`                                                | Case-insensitive filters on dataset name (and dtype for `--grep`) |

## `tet verify`

Accepts a standalone `.tet` **or** a `.tet.d` package directory (`PROTOCOL` present).

| Flag        | Effect                                                                               |
| ----------- | ------------------------------------------------------------------------------------ |
| _(default)_ | Human-readable check list + summary (decodes up to **128** chunks on large files)    |
| `--deep`    | Decode **every** chunk payload (not just the quick sample; package: each `packed.tet`) |
| `--repair`  | After verify, apply safe in-place repairs for repairable findings (see `tet repair`) |
| `--json`    | Pretty JSON [`TetVerifyReport`](../src/verify/report.rs)                             |
| `-q`        | One line (`status=ok` / `failed`)                                                    |

**Standalone `.tet`:** superblock, catalog, chunk index, payloads, duplicate chunk coords, footer.

**Package `.tet.d`:** `PROTOCOL` / `CURRENT` → sealed generation, `MANIFEST` length + SHA-256, layout-v1 on `packed.tet`, warn on incomplete/stale `publishing/` and orphan `staging/`. See [writer protocol v2](writer_protocol_v2.md#verify--repair-expectations).

#### Package finding `check` names

| `check` | Severity | Typical cause |
| ------- | -------- | ------------- |
| `protocol` | Error | Missing / unreadable `PROTOCOL`, or `protocol_version ≠ 2` |
| `current` | Error | Unparseable `CURRENT`, or `CURRENT` absent while `generations/` has sealed dirs |
| `current_packed` / `generation_packed` | Error | `generations/<G>/packed.tet` missing |
| `current_manifest` / `generation_manifest` | Error | `MANIFEST.json` missing / unreadable |
| `manifest_generation` | Error | MANIFEST `generation` ≠ directory id |
| `manifest_protocol` | Error | MANIFEST `protocol_version ≠ 2` |
| `manifest_byte_len` | Error | `packed.tet` length ≠ MANIFEST `packed_byte_len` |
| `manifest_sha256` | Error / Warn | SHA mismatch (error) or SHA absent / legacy (warn on CURRENT) |
| `packed_layout` | Error | Sealed `packed.tet` failed layout-v1 verify |
| `chunk_coords` | Error | Duplicate `(dataset_id, chunk_index)` inside `packed.tet` |
| `publishing_incomplete` | Warn | Live `publishing/<G>/` lock (CURRENT unchanged) |
| `publishing_stale` | Warn | Stale publish lock (repair: `quarantine_publish`) |
| `publishing` | Warn / Ok | Quarantine leftovers or clean (no active locks) |
| `staging` | Warn / Ok | Orphan staging sessions (optional GC) |
| `generations` | Warn | Unexpected entry under `generations/` |

Reject / mid-publish coverage: `src/tests/package_verify.rs`, `src/tests/publish.rs` ([THI-480](https://linear.app/thicclatka/issue/THI-480)).

Exit code **1** when verification fails (CI-friendly). Manual smoke fixtures: [`fixtures/small/tet/README.md`](../fixtures/small/tet/README.md).

## `tet repair`

| Flag           | Effect                                                                   |
| -------------- | ------------------------------------------------------------------------ |
| _(default)_    | Plan from verify recommendations (no writes)                             |
| `--apply CODE` | Apply fix (repeatable): `footer_invalid`, `quarantine_publish`           |
| `--dry-run`    | With `--apply`, show changes without writing                             |
| `--json`       | Pretty JSON plan or repair report                                        |

## `tet query`

`QUERY`: path to `.json` / `.toml`, inline JSON/TOML, `-` for stdin, or omit to read stdin. Leading `{` → JSON; `.toml` extension → TOML.

| Flag                | Effect                                                                                                                                                           |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-t`, `--tet PATH`  | Attach catalog / read plan (required for `-x`); `.tet` or `.tet.d` (uses `CURRENT`)                                                                              |
| `-x`, `--execute`   | Decode tiles, run `operation`, attach `execution`                                                                                                                |
| `--format`          | `full` (default), `json`, `stats`, `plan`, `quiet`, `table`                                                                                                      |
| `-q`, `--quiet`     | Shorthand for `--format quiet` (one-line stdout)                                                                                                                 |
| `--preview N`       | Cap preview sample values when executing (all dtypes; `--preview-f32` alias; default **64** for `full`/`json`, **0** for `stats`/`plan`/`quiet`/`table`)         |
| `--device DEVICE`   | Tier-A/B device routing (`cpu`, `auto`, `metal`, `cuda`, `cuda:N`, `rocm`, `rocm:N`, `cuda:multi`, `rocm:multi`); overrides query `execution.device`; needs `-x` |
| `--spill-allow DIR` | Extra spill roots (repeatable; needs `-x` and `-t`)                                                                                                              |

**Transform `write` targets** (with `-x` and `-t`): `ram` / `switch` / `spill` / **`sidecar`**. Sidecar publishes a one-chunk `.tet` beside the source (default filename `{stem}.{method}.{timestamp}.tet`; set `"timestamp": false` for a stable name). Published path appears in `execution.spill_f32_path`. Example:

```bash
tet query '{"dataset":"temperature","transform":{"method":"zscore"},"write":{"target":"sidecar","timestamp":false}}' -t data.tet -x -q
```

See [query engine — transform sidecar](query_engine.md#embedder-dense-export) and [JSON examples](query_engine.md#json-examples).

## `tet qhist`

Stored under the platform cache (`query_history.jsonl`), not in the `.tet` file. Env: `TET_NO_QUERY_HISTORY`, `TET_QUERY_HISTORY_FILE`, `TET_QUERY_HISTORY_MAX`. Details: [query engine — end-to-end flow](query_engine.md#end-to-end-flow) (`tet qhist`); roadmap row under [operations](query_engine.md#operations-roadmap-planned).

| Subcommand / flag                                                | Effect                                                                                                              |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `list` _(default)_                                               | Compact table of recent queries                                                                                     |
| `run N`                                                          | Re-run saved row (`1` = newest in filtered view); honors today's `--format` / `-q`; `-t` / `-x` / `--plan` override |
| `--clear`                                                        | Remove the history file                                                                                             |
| `list --all`, `--dataset`, `--tet`, `--mode`, `--grep`, `--json` | Filters / full JSON export on `list`                                                                                |

## `tet convert`

| Input   | Sniff / extensions                                        |
| ------- | --------------------------------------------------------- |
| HDF5    | `.h5`, `.hdf5`, `.hdf`, `.he2`, `.he5`, or file signature |
| NetCDF  | `.nc`, `.netcdf`, `.nc4`, `.nc3`, `.cdf`, or signature    |
| Zarr v3 | Directory with root `zarr.json`                           |
| NumPy   | `.npy`, `.npz` (one dataset per member), or `\x93NUMPY` magic |
| CSV     | `.csv` (comma) or `.tsv` (tab). One 1-D dataset per numeric column. Text labels become `i64` codes plus a `tetration_categories` footer attr when the map fits in 1024 bytes |

| Flag       | Effect                                                                         |
| ---------- | ------------------------------------------------------------------------------ |
| `--jobs N` | Parallel chunk read workers (`0` = host `available_parallelism`, capped at 64) |
| `--strict` | Fail (no output file) when any source array is skipped |

## `tet export`

| Flag / arg | Effect                                                                                 |
| ---------- | -------------------------------------------------------------------------------------- |
| `<in.tet>` | Source file (mmap read + catalog summary)                                              |
| `<out>`    | Zarr v3 **directory**; must be missing or **empty** (creates `zarr.json` + chunk tree) |
| _(stderr)_ | Progress line: dataset count, chunks written, elapsed seconds                          |

Preserves per-dataset **raw** or **zstd** chunk bytes; slash-separated dataset names become nested groups (`primary/f32`). Library: [`export_tet_to_zarr`](https://docs.rs/tetration/latest/tetration/export/fn.export_tet_to_zarr.html).

Import/export matrix, roadblocks, and future formats: [`conversions.md`](conversions.md).

## `tet ingest` / `publish` / `compact` / `pack`

Local concurrent package workflow ([writer protocol v2](writer_protocol_v2.md)):

```bash
tet ingest a.tet b.tet --out climate.tet.d --jobs 2
tet publish climate.tet.d
tet query '{"dataset":"temp","mean":[]}' -t climate.tet.d -x -q
tet pack climate.tet.d -o climate.tet
```

### `tet ingest`

| Flag / arg | Effect |
| ---------- | ------ |
| `<INPUT.tet>…` | One or more sealed `.tet` files (each → one staging part) |
| `--out PKG.tet.d` | Package directory (created if missing) |
| `--jobs N` | Parallel ingest workers (`0` = host parallelism, capped at 64) |
| `--create` | Explicit create (default already creates when path is absent) |
| `-q` | Quiet |

### `tet publish` / `tet compact`

Freeze sealed staging parts → merge into `packed.tet` → flip `CURRENT`. **`tet compact` is an alias**: compact/merge happens during publish (not a separate visibility step).

| Flag | Effect |
| ---- | ------ |
| `--generation G` | Generation id (default: `CURRENT+1`, or `1`) |
| `--parts ID,…` | Explicit part-id list (default: all sealed staging parts) |
| `-q` | Quiet |

### `tet pack`

| Flag / arg | Effect |
| ---------- | ------ |
| `PKG.tet.d` | Package root |
| `-o OUT.tet` | Destination standalone `.tet` |
| `--generation G` | Generation to copy (default: `CURRENT`) |

More examples: [`fixtures/queries/`](../fixtures/queries/), [`query_engine.md`](query_engine.md).

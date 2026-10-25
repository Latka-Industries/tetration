# `tet` CLI reference

Full flag lists: **`tet -h`** and **`tet <command> -h`** (always match the installed binary).

| Command                                            | Alias  | Role                                                        |
| -------------------------------------------------- | ------ | ----------------------------------------------------------- |
| [`tet info`](#tet-info) `<path.tet\|.tet.d>`       | —      | Summarize a file or package CURRENT (default: dataset table) |
| [`tet verify`](#tet-verify) `<path.tet\|.tet.d>`   | —      | Layout / package health check (exit 1 on failure); `--json` / `-q` |
| [`tet repair`](#tet-repair) `<path.tet\|.tet.d>`   | —      | Plan or apply safe in-place fixes (footer, stale publish)   |
| [`tet query`](#tet-query) `[QUERY]`                | `q`    | Validate JSON/TOML; optional catalog + execute against `-t` |
| [`tet qhist`](#tet-qhist) `[list\|run]`            | `hist` | Recent queries (platform cache; **not** the `.tet` footer)  |
| [`tet semantics`](#tet-semantics) `apply\|scaffold\|set`  | —      | Stamp dims/attrs from TOML, or scaffold caller policy         |
| [`tet run`](#tet-run) `<job.toml\|json>`          | —      | Run an ordered list of `tet` steps from a job file (fail-fast; [`job.md`](job.md)) |
| [`tet worker`](#tet-worker) `[--job PATH]`         | —      | Distributed harness: `WorkerJobV1` JSON → `WorkerPartialV1` JSON |
| [`tet convert`](#tet-convert) `<in> <out.tet>`     | —      | HDF5 / NetCDF / Zarr v3 / NumPy / CSV → `.tet` (auto-embeds ZONE/INDEX) |
| [`tet index`](#tet-index) `<path.tet>`             | —      | (Re)embed or `--clear` ZONE/INDEX footer spills (0.7.0)     |
| [`tet export`](#tet-export) `<in.tet> <out>`       | —      | `.tet` → Zarr v3 / HDF5 / NetCDF (from output path)         |
| [`tet ingest`](#tet-ingest--publish--compact--pack) | —    | Stage `.tet` inputs into a `.tet.d` package                 |
| [`tet publish`](#tet-ingest--publish--compact--pack) / [`compact`](#tet-ingest--publish--compact--pack) | — | Seal staging → generation / flip `CURRENT` |
| [`tet pack`](#tet-ingest--publish--compact--pack)  | —      | Copy sealed `packed.tet` → standalone `.tet` (`--remove-package` optional) |
| [`tet gc`](#tet-gc) `<path.tet.d\|s3://>`       | —      | Drop leftover staging + old generations (dry-run; `--apply` to delete) |

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
| `-t`, `--tet PATH`  | Attach catalog / read plan (required for `-x`); `.tet`, `.tet.d`, or `s3://bucket/prefix` (`?generation=N`). Package root and `packed.tet` both load generation `ZONE.json` / `INDEX.json` for pushdown; standalone `.tet` has no sidecars (through 0.6.x). A package pins the generation it opened and does not follow a newer publish ([pin](query_engine.md#pin--consistency)). `s3://` needs `--features tetration-s3` and `TETRATION_S3_ENDPOINT` |
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

## `tet semantics`

Declared science/media facts in the `.tet` footer, plus caller-policy stubs. Spec: [semantics.md](semantics.md); profile contracts: [semantics_profiles.md](semantics_profiles.md). Not a plugin host; attrs do not change query answers by themselves.

| Subcommand | Effect |
| ---------- | ------ |
| `apply <path.tet> --from FILE` | Stamp `dim_names` and attrs from one-file `tet-semantics.toml`. Prints what was stamped vs left in the TOML (profiles and units are not written). Does not rewrite `--from`. Refuses a sealed `.tet.d`. `--annotate` prints a combined TOML with live values and `# key =` comments for missing recommended attrs; `-o FILE` writes that copy and must not be `--from`. |
| `scaffold <path.tet>` | Emit combined `tet-semantics.toml` (default) or `--format json`: declare sections plus policy. Missing recommended keys are TOML comments. With none → catalog lines + doc URLs only (no fake config). `-o FILE` writes instead of stdout. |
| `set <path.tet> --dataset NAME --set KEY=VAL` | Declare attrs (repeatable `--set` / `--clear KEY`). Rewrites footer metadata; appends a `semantics` history row. |
| `package-write <path.tet.d>` | Write declare-only `generations/<G>/SEMANTICS.json` from packed footer attrs and `dim_names` (default `CURRENT`; `--generation N`; `--no-manifest` skips MANIFEST stamp). Does not rewrite `packed.tet`. |
| `package-show <path.tet.d>` | Print `SEMANTICS.json` for a sealed generation (default `CURRENT`). |
| `lint <path.tet\|.tet.d>` | Advisory profile-contract lint (missing recommended attrs). Does not refuse open; exit 0 always. |

Worked recording fixture: `fixtures/small/tet/recording.tet` (`frame_rate`, `time_axis`, provenance).

## `tet worker`

Localhost distributed harness (THI-618). Reads one `WorkerJobV1` JSON (`--job PATH`, or stdin), opens the pin path, folds the shard plan, writes `WorkerPartialV1` JSON to stdout. Not a cluster scheduler — see [`distributed_boundary.md`](distributed_boundary.md) and [`query::distributed`](../src/query/distributed/README.md).

| Flag | Effect |
| ---- | ------ |
| `--job PATH` | Job JSON file (`-` or omit = stdin) |

Smoke: `mise distributed:smoke` (in-process + N-process integration test).

## `tet convert`

| Input   | Sniff / extensions                                        |
| ------- | --------------------------------------------------------- |
| HDF5    | `.h5`, `.hdf5`, `.hdf`, `.he2`, `.he5`, or file signature |
| NetCDF  | `.nc`, `.netcdf`, `.nc4`, `.nc3`, `.cdf`, or signature    |
| Zarr    | v3 directory with root `zarr.json`, or v2 `.zgroup` / `.zarray` |
| NumPy   | `.npy`, `.npz` (one dataset per member), or `\x93NUMPY` magic |
| CSV     | `.csv` (comma) or `.tsv` (tab). One 1-D dataset per numeric column. Text labels become `i64` codes plus a `tetration_categories` footer attr when the map fits in 1024 bytes |
| Parquet | `.parquet`, `.pq`, or `PAR1`. Same column mapping as CSV. Requires `--features tetration-parquet` |
| `s3://` | A single file (extensions above) downloads the whole object, then the same importer. A Zarr v2 or v3 prefix (`store`, `store.zarr`, or a trailing slash) fetches each chunk as the tile is written; the `.tet` matches a local convert. Requires `--features tetration-s3` and `TETRATION_S3_ENDPOINT` (`TETRATION_S3_ACCESS_KEY`, `TETRATION_S3_SECRET_KEY`, `TETRATION_S3_REGION`) |

| Flag            | Effect                                                                                          |
| --------------- | ----------------------------------------------------------------------------------------------- |
| `--jobs N`      | Parallel chunk read workers (`0` = host `available_parallelism`, capped at 64)                  |
| `--strict`      | Fail (no output file) when any source array is skipped                                          |
| `--chunk-mib N` | Retile every dataset to about N MiB along the leading axis. Omit to keep source chunks (or 64 MiB for NumPy, CSV, and Parquet) |

Convert **auto-embeds** zone maps + optional secondary indexes into the output `.tet` (`history_version = 2`). See [`embedded_indexes.md`](embedded_indexes.md).

## `tet index`

(Re)build or drop embedded ZONE/INDEX spills on a sealed standalone `.tet`. Convert and writer commit already embed; use this for older bare files or a no-pushdown baseline.

| Flag / arg | Effect |
| ---------- | ------ |
| `<path.tet>` | Target sealed file |
| `--clear` | Drop embeds (`history_version = 1`; keep history/metadata) |
| `-q`, `--quiet` | Suppress the one-line summary |

## `tet export`

| Flag / arg | Effect                                                                                 |
| ---------- | -------------------------------------------------------------------------------------- |
| `<in.tet>` | Source file (mmap read + catalog summary)                                              |
| `<out>`    | Format from path: Zarr **directory** (missing/empty), `.h5`/`.hdf5`, or `.nc`/`.netcdf` |
| _(stderr)_ | Progress line: dataset count, chunks written, elapsed seconds                          |

| Output | Behavior |
| ------ | -------- |
| Zarr directory / `.zarr` | Copies per-dataset **raw** or **zstd** chunk bytes; slash names → nested groups |
| `.h5` / `.hdf5` / … | Dense numeric HDF5 (`tetration-hdf5`); path must not exist |
| `.nc` / `.netcdf` / … | Dense numeric NetCDF-4 (`tetration-netcdf`); carries `dim_names` when present |

Slash-separated dataset names become nested groups (`primary/f32`). Library: [`export_tet`](https://docs.rs/tetration/latest/tetration/export/fn.export_tet.html) (or format-specific `export_tet_to_*`).

Import/export matrix, roadblocks, and future formats: [`conversions.md`](conversions.md).

## `tet ingest` / `publish` / `compact` / `pack`

Local concurrent package workflow ([writer protocol v2](writer_protocol_v2.md)):

```bash
tet ingest a.tet b.tet --out climate.tet.d --jobs 2
tet publish climate.tet.d
tet query '{"dataset":"temp","mean":[]}' -t climate.tet.d -x -q
tet query '{"dataset":"temp","mean":[]}' -t 's3://bucket/prefix?generation=3' -x -q
tet pack climate.tet.d -o climate.tet
tet pack climate.tet.d -o climate.tet --remove-package
```

### `tet ingest`

Copies each input `.tet` into staging as a **full-grid** part (whole datasets). Use this to **add** sealed arrays to a package — not to tile one array across workers. For disjoint chunk-range tiling of a single dataset, use the library [`TetShardWriter`](https://docs.rs/tetration/latest/tetration/catalog/struct.TetShardWriter.html) with [`DatasetClaim::ranges`](https://docs.rs/tetration/latest/tetration/catalog/struct.DatasetClaim.html#method.ranges) (writer protocol decision E).

| Flag / arg | Effect |
| ---------- | ------ |
| `<INPUT.tet>…` | One or more sealed `.tet` files (each → one staging part) |
| `--out PKG.tet.d` | Package directory (created if missing) |
| `--jobs N` | Parallel ingest workers (`0` = host parallelism, capped at 64) |
| `--create` | No-op (kept for compatibility): a missing package is always created; an existing non-package path always fails |
| `-q` | Quiet |

### `tet publish` / `tet compact`

Freeze sealed staging parts → merge into `packed.tet` → write per-chunk zone maps (`ZONE.json`) and optional `INDEX.json` (`value_postings`, `sorted_axis_range`, `coord_label_postings` when each kind qualifies) → flip `CURRENT`. **`tet compact` is an alias**: compact/merge happens during publish (not a separate visibility step). Staging parts are not updated with those sidecars. A missing index does not change query answers. Local packages use the same exclusive `publishing/<G>/` + heartbeat rule as [writer protocol v2](writer_protocol_v2.md). Object-store / multi-host publish is the library `PackageStore` path (`publish_staged_parts` / `PublishOptions`); the CLI still targets a local `.tet.d`.

| Flag | Effect |
| ---- | ------ |
| `--generation G` | Generation id (default: `CURRENT+1`, or `1`; required with `--abort`) |
| `--parts ID,…` | Explicit part-id list (default: all sealed staging parts) |
| `--dry-run` | Claim / schema checks only; do not seal or flip `CURRENT` |
| `--replace DATASET` | Allow dtype/shape change for that name vs sealed `CURRENT` (repeatable; never mutates sealed `G`) |
| `--abort` | Drop a stuck `publishing/<G>/` lock (`--generation` required; refuses `CURRENT`) |
| `-q` | Quiet |

```bash
tet publish climate.tet.d --dry-run
tet publish climate.tet.d --replace temp
tet publish climate.tet.d --abort --generation 2
```

### `tet pack`

| Flag / arg | Effect |
| ---------- | ------ |
| `PKG.tet.d` | Package root |
| `-o OUT.tet` | Destination standalone `.tet` |
| `--generation G` | Generation to copy (default: `CURRENT`) |
| `--remove-package` | After a verified copy (size + SHA-256 + layout verify), delete the `.tet.d` tree. Default keeps the package. Close any readers pinned on that package first. |

## `tet gc`

Drop leftover `staging/` sessions and sealed generations older than `CURRENT` / `CURRENT-1` (writer protocol decisions D + J). **Dry-run by default.** Does not change `CURRENT`.

```bash
tet gc climate.tet.d
tet gc climate.tet.d --apply
tet gc s3://bucket/prefix
tet gc s3://bucket/prefix --apply --protect-generation 1
```

| Flag / arg | Effect |
| ---------- | ------ |
| `PKG.tet.d` | Local package root (`TetPackage::gc`) |
| `s3://bucket/prefix` | Store prune (`prune_store` on `S3Store`). Feature `tetration-s3`. Credentials are `TETRATION_S3_*` (`ENDPOINT` required), not the URI. Optional `?generation=G` protects that generation. |
| `--apply` | Actually delete (default: print the plan only) |
| `--protect-generation G` | Repeatable. `s3://` only. Keep generation `G` even when it is older than `CURRENT-1` (an open pin this process cannot see). |
| `-q` | Quiet |

Never deletes the generation named by `CURRENT` or `CURRENT-1`. A live `publishing/<G>` claim (heartbeat not stale) keeps staging and that generation.

**Pin safety (`s3://`):** `tet gc` cannot see `TetPackagePin` / `PinnedPackedTet` handles held by other processes. Those `packed.tet` objects are deleted if they are older than `CURRENT-1` unless you pass `--protect-generation` or `?generation=G`. `CachingStore::prune` protects its own pin; the CLI does not open a pin for you. Local `.tet.d` gc is unchanged and does not take `--protect-generation`.

## `tet run`

Run a TOML or JSON **job file**: an ordered list of stock `tet` steps, fail-fast. Not Airflow (no DAG, conditionals, loops, or retries) and not query-pipeline ETL. Schema, op list, and library API: [`job.md`](job.md).

```bash
tet run job.toml
tet run job.toml --dry-run
tet run job.toml --root /data/run1
```

| Flag / arg  | Effect |
| ----------- | ------ |
| `JOB`       | `.toml` or `.json` (`schema_version = 1`, `[[step]]` / `"steps"`) |
| `--dry-run` | Validate (including query documents) and print the plan; execute nothing |
| `--root DIR` | Resolve relative paths against `DIR` (default: the job file's directory) |

Ops: `info`, `verify`, `repair`, `query`, `convert`, `export`, `index`, `semantics.apply` / `scaffold` / `set` / `wizard` / `lint` / `package-write`, `ingest`, `publish`, `compact`, `pack`, `gc`. Query history is off. `semantics.wizard` needs a TTY on stdin.

More examples: [`fixtures/queries/`](../fixtures/queries/), [`query_engine.md`](query_engine.md).

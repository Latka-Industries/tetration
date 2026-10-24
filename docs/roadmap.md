# Roadmap and prospective phases

**Status:** informal / prospective — not a commitment or release schedule.

Tetration v1 already covers efficient **partial I/O** over a local mmap-friendly `.tet` file: chunk index, JSON/TOML query control plane, streaming reductions, transforms, optional GPU (Phase 10), and a C ABI (Phase 11). The next steps are less about obvious missing flags and more about **what kind of system Tetration should become**.

This document captures directions that would matter to teams behind formats and engines such as Apache Arrow, Parquet, HDF5, and large-scale storage at cloud vendors — adapted to Tetration’s current design ([`layout_v1.md`](layout_v1.md), [`query_engine.md`](query_engine.md)).

## Shipped phase index (reference)

| Phase | Theme | Where documented |
| ----- | ----- | ---------------- |
| 5 | Foreign format → `.tet` (`tet convert`) | [`src/convert/`](../src/convert/mod.rs) |
| 7 | Session writer, footer metadata / history | [`layout_v1.md`](layout_v1.md), [`session.rs`](../src/catalog/session.rs) |
| 9 | Named axes, coord label slices, QC counts, covariance | [`query_engine.md`](query_engine.md) |
| 10 | Optional GPU device routing (experimental) | [`query_engine.md` — Phase 10](query_engine.md#phase-10--optional-gpu-experimental) |
| 11 | C ABI (`tetration-ffi`) | [`ffi.md`](ffi.md) |

User-facing copy (no issue tracker links): **[docs.tetration.sh/roadmap](https://docs.tetration.sh/roadmap)**.

Near-term work also tracked in [`query_engine.md` — operations roadmap](query_engine.md#operations-roadmap-planned), [hardening](query_engine.md#hardening-roadmap), [`conversions.md`](conversions.md), and [**tet-py**](https://github.com/Latka-Industries/tet-py).

---

## Five strategic directions

### 1. Cloud-native operation (Phase 12)

**Shipped in 0.3.0:** [`PackageStore`](../src/catalog/store/mod.rs) (`LocalStore` / `MemoryStore`), URI parse (`file://`, paths, `memory://`, `s3://` shape), range-read planner (catalog then chunk payloads), and [`TetPackagePin`](../src/catalog/package.rs) so a handle keeps one generation when `CURRENT` advances. See [`query_engine.md` — store-backed reads](query_engine.md#store-backed-reads-030--phase-12-foundation).

**Shipped in 0.3.1:** optional `S3Store` (feature `tetration-s3`, MinIO in tests) and store-level `stage_part` / `publish_staged_parts` (one CAS publish; a lost precondition keeps the previous generation).

**Shipped in 0.3.2:** cross-host exclusive claim on `publishing/<G>` (heartbeat + stale window; takeover only with explicit `force`). A second host that loses the claim does not write `CURRENT`.

**Reader rule (normative):** a query keeps the generation it opened. A publish that moves `CURRENT` is invisible until the next open. Replacing the pinned `packed.tet` is not a new generation — see [Pin / consistency](query_engine.md#pin--consistency) and [`layout_v1.md` — package readers](layout_v1.md#package-readers).

**Shipped in 0.3.3:** optional in-memory [`BlockCache`](../src/catalog/store/cache.rs) / [`CachingStore`](../src/catalog/store/cache.rs) — packed ranges keyed by `(generation, offset)`; cache off unless wrapped. `tet convert s3://bucket/file.ext` downloads that single object, then the existing importer (`tetration-s3`).

**Shipped in 0.3.4:** `tet query -t s3://bucket/prefix` and `open_s3_package` pin that package (`CURRENT`, or `?generation=N`) and read through the existing range planner. Credentials stay in `TETRATION_S3_*`.

**Shipped in 0.3.5:** `tet convert` of a Zarr v2 or v3 prefix on `s3://` fetches each chunk as the tile is written. The `.tet` matches a local-directory convert. A single-file `s3://bucket/file.ext` still downloads the whole object.

**Shipped in 0.3.6:** local `.tet.d` package cleanup the writer protocol already allowed: `tet pack --remove-package`, `tet gc` (staging + old generations; dry-run default), and `tet publish --abort` (THI-520). S3 prune is its own cut after 0.4.0 (THI-535), not part of zone maps.

**Shipped in 0.3.7:** publish conflict UX — clearer overlap rejects, `tet publish --dry-run`, identical-byte overlap OK, opt-in `--replace` for G+1 schema change, and disjoint shard claims (THI-524). Default remains reject-on-differing-overlap.

**Shipped in 0.3.8:** library hygiene only. `f16` and `f64` materialize share one macro. Matchable package failures are `CatalogError` variants (display text unchanged). `tetration::query` CLI formatters were deprecated here (THI-536) and removed in 0.4.0 (THI-541).

**Shipped in 0.4.0:** per-chunk min/max/null zone maps at compact/publish (`generations/<G>/ZONE.json`; staging parts unchanged; verify is required for new seals and advisory for older ones) (THI-473). `ReadPlan.logical_f32_element_count` is `logical_element_count` (THI-540). The deprecated `tetration::query` CLI re-exports are gone; `tet` uses `query::cli` (THI-541). **tet-py 0.4.0** on PyPI pins this crate.

**Shipped in 0.4.1:** value-filter wire + conservative zone-map chunk skip (`filter` on the query document; `ReadPlan.zone_filter_skipped_chunks`) (THI-543).

**Shipped in 0.4.2:** `group_by` over dimension names, with the reduction streamed per group when the op is tier-A/B, plus a closed `pipeline` step list (THI-551).

**Shipped in 0.4.3:** store prune (`prune_store`, `tet gc s3://bucket/prefix`): leftover staging and sealed generations older than `CURRENT` / `CURRENT-1`. Dry-run by default. Open pins and a live publish claim are refused. Does not write `CURRENT` (THI-535).

**Shipped in 0.4.4:** element-wise `filter`. Chunk skip from 0.4.1 (`zone_filter_skipped_chunks`) is unchanged. Folds ignore non-matching cells inside kept chunks. Materialize and transform keep the selection shape: NaN for floats, dataset fill for integers (refused when no fill is set). PyPI tet-py stays **0.4.3** until that package is cut.

Example query:

```text
tet query '{"dataset":"temp","mean":[]}' -t 's3://bucket/path/climate?generation=3' -x
```

**Why the rest is hard:** object stores are not POSIX. Latency and partial reads differ from mmap. Conditional publish of `CURRENT` is the store CAS. The pin rule above does not change with the transport.

**Relation to v1:** chunk index + fixed payload offsets are unchanged. A pinned `s3://` package is read with the same range planner. A Zarr prefix is a foreign store, converted into a `.tet`, not opened as a package.

---

### 2. Smarter indexing (prospective Phase 13)

**Today:** organization is **geometric** — logical coordinates → chunk coordinates via the chunk index. Footer metadata supports `dim_names`, coordinate **labels**, and attrs. Value predicates also have optional sidecars beside `packed.tet`: zone maps, and value postings when a dataset qualifies.

**Shipped in 0.4.0:** compact/publish writes per-chunk min/max/null zone maps to `generations/<G>/ZONE.json` and copies them onto `ReadPlan` chunks. Staging writers do not update those stats.

**Shipped in 0.4.1:** query `filter` predicates skip chunks that cannot match those zone stats (conservative; missing zones → no skip). `zone_filter_skipped_chunks` is that chunk-skip stat.

**Shipped in 0.4.4:** the same predicate filters cells inside a kept chunk. A fold under `filter` ignores non-matches. Materialize and transform keep the selection shape and write NaN (floats) or the dataset fill (integers; refused when no fill is set).

**Shipped in 0.4.5:** compact/publish may also write `generations/<G>/INDEX.json`, a value-posting list for exact `filter.eq` when a dataset has at most 64 distinct non-null values. Zone maps still run first. Posting drops are `index_filter_skipped_chunks`. A missing, unknown, or unverifiable index is ignored and the planner uses zone maps (or a full scan). Old readers ignore the new manifest keys. Sorted-axis range indexes and coordinate-label postings are later kinds.

**Shipped in 0.5.0:** distributed fold library path — see [Phase 14](#3-distributed-execution-phase-14).

**Shipped in 0.5.1:** declared science/media attrs + `tet semantics` CLI / helpers — see [Phase 16](#5-scientific-semantics-phase-16).

**Shipped in 0.5.2:** normative SPEC + compatibility policy + conformance pack (+ optional Python smoke) — see [Phase 18](#prospective-phase-summary).

**Shipped in 0.5.3:** larger conformance accept / reject surface (`spec_milestone` **0.5.3**) — same layout + query wire; no ABI break.

**Shipped in 0.5.4:** crate hygiene (clippy allows, typed convert/export I/O, optional `tetration-cli`) + `parse_query_json` fuzz — no layout / query ABI / pack change. PyPI **tet-py 0.5.4** pins this crate.

**Shipped in 0.6.0:** closed `window` query wire (stats along a named axis) + conformance pack milestone **0.6.0** (THI-595).

**Shipped in 0.6.1:** `window.mode` → `bins` / `rolling` (reject-closed on `tumbling` / `sliding`); multi-dataset **`align`** checks; conformance pack milestone **0.6.1** (THI-594). Layout / package / C ABI unchanged. PyPI **tet-py 0.6.1** pins this crate.

**Shipped in 0.6.2:** `group_by` labels form (`{ "by": "labels", "axes": [...] }`) collapses repeated footer coordinate labels; array form unchanged; conformance pack milestone **0.6.2** (THI-596). Layout / package / C ABI unchanged. PyPI **tet-py 0.6.2** pins this crate.

**Shipped in 0.6.3:** `window` `pad` / `step` / `align` / `min_count` + multi-dtype windowed folds; conformance pack milestone **0.6.3** (THI-606). Layout / package / C ABI unchanged. PyPI **tet-py 0.6.3** pins this crate.

**Shipped in 0.6.4:** hygiene after the 0.6.x query wave — DRY window/fold paths, pedantic + window-edge fuzz, broader `mise bench:*` coverage (windows / transforms / scalar), and ephemeral `write: switch` spill cleanup (THI-607 / THI-608). Layout / package / C ABI / pack milestone unchanged (**0.6.3**). **tet-py 0.6.4** pins this crate.

**Shipped in 0.6.5:** optional `sorted_axis_range` in `INDEX.json` + planner pushdown (THI-613); fold sum/mean dense-slab speedups; `bench:index` small/large vs HDF5+NumPy; `secondary_index_json` fuzz. Layout / package / C ABI / pack milestone unchanged (**0.6.3**). PyPI **tet-py 0.6.5** pins this crate.

**Shipped in 0.6.6:** optional `coord_label_postings` + closed `label_filter: { axis, eq }` (pandas-style footer label equality; AND with cell `filter`; THI-614). Package-root `.tet.d` queries load generation `ZONE`/`INDEX` sidecars; indexed query no longer SHA-256s the whole `packed.tet`; OOC folds refuse `linear_scan` when cell/`label_eq` filters are set. Layout / package / C ABI / pack milestone unchanged (**0.6.3**). PyPI **tet-py 0.6.6** pins this crate.

**Shipped in 0.7.0:** embed zone maps + secondary indexes **inside** the sealed `.tet` (footer spills + `history_version = 2`; `layout_version` stays **1**) so a standalone file is complete for pushdown — no publish-to-`.tet.d` required ([`embedded_indexes.md`](embedded_indexes.md); THI-627). Convert / writer commit auto-embed (write-time zones; standalone `packed_sha256` placeholder); package seal still hashes for MANIFEST/sidecars; `tet index` / `--clear`; value-postings streams to the cardinality cap; `secondary_index_seal` fuzz. Query prefers embed then sidecar. Conformance pack milestone **0.7.0**. `.tet.d` stays the multi-writer path only. Parent ladder: [THI-626](https://linear.app/thicclatka/issue/THI-626) / [THI-609](https://linear.app/thicclatka/issue/THI-609).

**Why it matters:** at petabyte scale, **I/O avoidance** often beats raw decode speed. Zone maps answer “can this chunk’s range contain the predicate?” They keep every chunk when ranges overlap. Postings answer “which chunks contain this exact value?” and skip the rest. Sorted-axis segments answer “which chunks cover this numeric coord span?” Coord-label postings answer “which chunks contain this category name?” Through 0.6.x indexes stay optional sidecars beside `packed.tet`; **0.7.0** moves them into the image.

**Relation to v1:** filter-by-value chunk skip is in **0.4.1**. Group-by over dimension names is in **0.4.2**. Cell filter on kept chunks ships in **0.4.4**. Through 0.6.x the posting list does not change `packed.tet`; **0.7.0** appends footer spills (`history_version = 2`) without bumping `layout_version`.

---

### 3. Distributed execution (Phase 14)

**Shipped in 0.5.0 (library):** [`query::distributed`](../src/query/distributed/mod.rs) — split a [`ReadPlan`](../src/query/types/plan.rs) into chunk shards (contiguous or interleaved), JSON worker job / partial shapes (`WorkerJobV1`), merge associative tier-A/B [`FoldPartialState`](../src/query/distributed/merge.rs) (same laws as parallel/GPU host merge), and an in-process multi-shard harness. Workers must share one **pinned** generation (never re-resolve `CURRENT`). Spill/sidecar/switch write targets are **refused** on the distributed path (RAM partials only; THI-577).

**0.7.1 boundary (locked):** [`distributed_boundary.md`](distributed_boundary.md) — stay format + thin protocol (not a worker-daemon product). Spill remains refuse-only (THI-615 / THI-616).

**Localhost N-process harness (THI-618):** `tet worker` runs one `WorkerJobV1` → `WorkerPartialV1`; `execute_sharded_scalar_fold_multiprocess` spawns N workers over one pinned `.tet` and merges partials. Still no network RPC / K8s fleet.

**Shipped in 0.7.2 (THI-617):** distributed materialize / transform / tier-C on the thin protocol (`DenseShardPartial`, `write: ram` only; multiprocess materialize + median). Spill/sidecar/switch stay refuse-only. **Shipped in 0.7.3 (THI-612):** semantics profiles (four PRs + module split). **Shipped in 0.8.0 (THI-632):** one-file TOML apply (dims + attrs), combined scaffold/annotate, package `SEMANTICS.json` `dim_names`. **Shipped in 0.8.1 (THI-638 / #189):** metadata wizard; chunk-aligned `group_by` fast path. **Later (THI-630):** network RPC. Still not shipped: cluster scheduler / worker-local spill / multiprocess transform stats broadcast.

**Hygiene (before protocol):** THI-597 / THI-598 landed on `main` (squash [#112](https://github.com/Latka-Industries/tetration/pull/112)); THI-599 optional backlog.

**Trade-off:** stay a **format + embeddable library** with a thin protocol — not Spark/Flink.

**Relation to v1:** streaming fold + chunk-local partials are the merge semantics; multi-GPU sharding (Phase 10) is the local prototype of the same pattern.

---

### 4. Richer query language (prospective Phase 15)

**Today:** flat JSON/TOML documents — one dataset, selection, optional `operation`, `transform`, `write` ([`query_engine.md`](query_engine.md)). SQL and arbitrary joins are explicit **non-goals** for v1 ([README](../README.md)).

**Direction:** richer **declarative** operations while keeping chunk-based execution — e.g. group-by over named dimensions, windowed stats, multi-dataset alignment checks, composable pipelines — without becoming a general SQL engine:

```sql
-- illustrative target, not current syntax
SELECT region, AVG(temp)
FROM climate
GROUP BY region
```

**Design tension:** scientists want expressive analytics; the format wants a small, auditable control plane. Any extension should declare **tiers** (scalar fold vs materialize-required) and stay JSON/TOML-serializable for agents and FFI.

**Relation to v1:** [`join` vs append](query_engine.md) is already documented. Group-by over dimension names shipped in 0.4.2. **Windowed stats** along a named axis shipped in **0.6.0** (`window` wire); edge controls + multi-dtype in **0.6.3**. Multi-dataset alignment shipped in **0.6.1**. Label-value group-by shipped in **0.6.2**. Filter-by-value pushdown shipped in 0.4.1.

---

### 5. Scientific semantics (Phase 16)

**Shipped in 0.5.1 (declared + policy split):** optional **declared** science/media facts in footer `metadata.datasets[<name>].attrs` and append-only history, plus caller **policy** outside the file (`tet-semantics.toml` / `.json`). Helpers and CLI: `tet semantics set` / `scaffold`. Spec: [`semantics.md`](semantics.md). Worked recording fixture: `fixtures/small/tet/recording.tet`. Climate appears only as attrs / a doc profile id — **not** a climate runtime in the crate. Declared attrs do **not** change fold/query answers by themselves.

**Shipped in 0.7.3 (THI-612):** profile registry contracts ([`semantics_profiles.md`](semantics_profiles.md)); optional declare-only package `SEMANTICS.json` (THI-620); policy linter + reference snippets; richer PROV-shaped history helpers; `catalog::semantics` module split. Still fold-inert.

**Shipped in 0.8.0 (THI-632):** one-file `tet-semantics.toml` + `tet semantics apply` (stamp `dim_names` + attrs; policy stays in TOML); combined scaffold / `apply --annotate`; `ProfileContract` palette; package `SEMANTICS.json` includes `dim_names`. Still fold-inert.

**Shipped in 0.8.1 (THI-638):** interactive `tet semantics wizard` (dialoguer + comfy-table; empty keeps last selection). Docs-site pins (THI-637) remain a follow-up.

**Still prospective:** full CF / PROV compliance suites, UCUM conversion graphs inside `packed.tet`, plugin / dynamic code load, required in-tree policy interpreter that mutates folds.

**Direction:** richer meaning alongside bytes — convertible units on read, geospatial CRS bindings, richer uncertainty, domain vocabularies — without becoming a plugin host.

| Capability | Example |
| ---------- | ------- |
| Physical units | `K`, `uV` as declared string tokens; display convert in caller policy |
| Coordinate reference systems | `crs` attr (e.g. `EPSG:4326`); geo crops stay caller-side |
| Provenance | history rows (`op`, `source`, `at`, `params`; PROV-inspired keys under `prov_log.v1`) |
| Uncertainty | `uncertainty_kind` + `uncertainty_sibling` flat attrs |
| Domain vocabularies | `semantics_profile` doc ids — registry in [`semantics_profiles.md`](semantics_profiles.md) |

**Why it may beat another 10% decode speed:** many scientific workflows fail on **metadata and lineage**, not raw throughput. NetCDF and CF conventions are partial precedents; Tetration embeds a small declared subset with spec’d invariants.

**Relation to v1:** optional history footer ([`layout_v1.md`](layout_v1.md)) holds provenance; Phase 7 metadata holds declared attrs.

---

## What impresses elite systems architects

Not always “more features.” Three cross-cutting qualities:

### Transaction safety and mutability

v1 is **write-once, read-many** on a single `.tet` — no locking, no live append ([`layout_v1.md` — Concurrency](layout_v1.md#concurrency-informative)). Concurrent ingest is **shards → publish → sealed generation** (not multi-writer on one live TIDX): [`writer_protocol_v2.md`](writer_protocol_v2.md) (**shipped** in 0.2 / Phase **17**).

### Formal specification

Many formats fail because the spec lives only in code. Independent readers (Julia, Go, browser WASM) need:

- complete binary layout for every record type
- compatibility and deprecation rules
- stated invariants (index bounds, dtype tags, footer mutual exclusion)
- conformance tests shared across implementations

**0.5.2 draft:** canonical contract index at [`SPEC/README.md`](../SPEC/README.md) (layout v1, writer v2, query wire, C ABI), incorporating [`layout_v1.md`](layout_v1.md) / [`writer_protocol_v2.md`](writer_protocol_v2.md) / [`query_engine.md`](query_engine.md) / [`ffi.md`](ffi.md) by reference. Compatibility flags and the CI conformance pack follow in the same cut.

HDF5 and Parquet history suggests **spec quality** often outlasts any one implementation.

### Long-term compatibility

The hardest problem: **read a file written today in 2046.** That implies disciplined schema evolution, reserved fields, capability flags, and a policy for breaking vs non-breaking wire changes — including query JSON and ABI versioning ([`ffi.md`](ffi.md)).

---

## From access to meaning

A common progression in large data systems:

```text
Data → Storage → Retrieval → Structure → Meaning
```

Tetration v1 sits strongly in **storage + retrieval** (partial chunk access, aggregates, transforms). The longer arc is **structure and meaning**: how an array was produced, which transforms apply, confidence intervals, causal or pipeline dependencies, and relationships among datasets — without giving up mmap-friendly partial reads.

A plausible north star: **scientific knowledge infrastructure** — not only a faster array file, but a durable place to attach semantics and lineage that current stacks handle ad hoc in sidecar JSON and notebooks.

---

## Prospective phase summary

| Phase | Working title | Depends on | Risk / note |
| ----- | ------------- | ---------- | ----------- |
| **12** | Object-store / cloud-native reads | Stable layout v1 | High engineering; cache + consistency semantics |
| **13** | Predicate indexes / chunk skipping | Metadata + query planner | Index maintenance on write; verify story |
| **14** | Distributed query execution | Tier-A/B merge semantics | Product boundary vs “just a format” |
| **15** | Group-by, pipelines, controlled joins | Axes + metadata | Avoid full SQL complexity |
| **16** | Units, CRS, provenance, uncertainty | Footer history + attrs | Standards alignment (CF, PROV, etc.) |
| **17** | Shards, generations, atomic publish | Writer protocol v2 | **Shipped** in 0.2 — [`writer_protocol_v2.md`](writer_protocol_v2.md); WORM per sealed `G` |
| **18** | Normative spec + conformance suite | Layout + query wire frozen | **SPEC** + [`COMPATIBILITY.md`](../SPEC/COMPATIBILITY.md) + [`conformance/`](../conformance/README.md) (THI-586/587/592) — ships as **0.5.2** |

Phases are **not ordered commitments** — e.g. formal spec (18) might parallelize early work on cloud reads (12).

---

## How to use this document

- **Contributors:** pick a direction, open a design issue with constraints from [`layout_v1.md`](layout_v1.md) and non-goals in the README.
- **Embedders:** treat everything above as **exploratory**; ship against v1 layout and documented query wire only.
- **Discussion:** [GitHub issue #16](https://github.com/Latka-Industries/tetration/issues/16) — long-term architecture and prospective phases.

When a direction ships, move detail into the relevant doc (`layout_v1.md`, `query_engine.md`, `ffi.md`) and mark the phase row **shipped** here.

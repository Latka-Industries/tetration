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

Near-term work already tracked elsewhere: Python bindings (separate repo), [`query_engine.md` — operations roadmap](query_engine.md#operations-roadmap-planned), [`query_engine.md` — hardening roadmap](query_engine.md#hardening-roadmap), [`conversions.md`](conversions.md) (import/export formats and roadblocks).

---

## Five strategic directions

### 1. Cloud-native operation (prospective Phase 12)

**Today:** queries assume a **local** `.tet` path; the engine mmap’s a sealed byte image on a normal filesystem.

**Direction:** treat object stores (S3, GCS, Azure Blob) as first-class without downloading whole files — e.g. a URI-shaped target and range reads aligned to chunk boundaries:

```text
tet query climate.tet@s3://bucket/path/climate.tet …
```

**Why it is hard:** object stores are not POSIX filesystems. Latency, partial reads, caching, and consistency differ from mmap. A serious design needs a **read planner** that minimizes round trips (index + footer first, then payload ranges), optional local block cache, and clear semantics when the object is replaced mid-read.

**Relation to v1:** chunk index + fixed payload offsets are a good foundation; the gap is transport and cache policy, not on-disk layout.

---

### 2. Smarter indexing (prospective Phase 13)

**Today:** organization is **geometric** — logical coordinates → chunk coordinates via the chunk index. Footer metadata supports `dim_names`, coordinate **labels**, and attrs, but there is no secondary index for value predicates.

**Direction:** optional metadata indexes so predicates such as `temperature > 100`, `year = 2025`, or `region = Arizona` can skip most chunks — analogous to database indexes, min/max zone maps, or Parquet statistics.

**Why it matters:** at petabyte scale, **I/O avoidance** often beats raw decode speed. Indexes must be optional, verifiable, and evolvable without breaking v1 readers.

**Relation to v1:** [`filter-by-value` and group-by remain deferred](query_engine.md#intentional-gaps-v1); this direction generalizes “slice by label” into predicate pushdown and chunk skipping.

---

### 3. Distributed execution (prospective Phase 14)

**Today:** one machine, many cores (Rayon), optional GPUs; scale-out is **N independent queries × N processes**, not one query sharded across nodes ([README — Concurrency](../README.md#concurrency-and-scale)).

**Direction:** a coordinator splits a single query’s read plan across workers (chunk shards), merges associative partials (tier-A/B folds already compose), and handles spill/export policy cluster-wide.

**Trade-off:** at that point Tetration competes with distributed analytics engines, not only file formats. The open question is whether to stay a **format + embeddable library** with a thin cluster protocol, or grow a full execution service.

**Relation to v1:** streaming fold + chunk-local partials are the right merge semantics; multi-GPU sharding (Phase 10) is a local prototype of the same pattern.

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

**Relation to v1:** [`join` vs append](query_engine.md) is already documented; group-by and filter-by-value belong here, not in ad hoc op sprawl.

---

### 5. Scientific semantics (prospective Phase 16)

**Today:** users think in **chunk coordinates** and **axis indices**; footer metadata carries labels and attrs but not full domain semantics.

**Direction:** first-class **meaning** alongside bytes:

| Capability | Example |
| ---------- | ------- |
| Physical units | `celsius`, `pascal`, convertible on read |
| Coordinate reference systems | geospatial bindings on lat/lon axes |
| Provenance | pipeline steps, parent datasets, tool versions |
| Uncertainty | per-element or per-chunk error bars |
| Domain vocabularies | cell type, patient id, instrument channel |

**Why it may beat another 10% decode speed:** many scientific workflows fail on **metadata and lineage**, not raw throughput. NetCDF and CF conventions are partial precedents; Tetration could embed a subset in footer history + dataset attrs with spec’d invariants.

**Relation to v1:** optional history footer ([`layout_v1.md`](layout_v1.md)) is a seed for provenance; Phase 7 metadata is a seed for named science dimensions.

---

## What impresses elite systems architects

Not always “more features.” Three cross-cutting qualities:

### Transaction safety and mutability

v1 is **write-once, read-many** — no locking, no live append ([`layout_v1.md` — Concurrency](layout_v1.md#concurrency-informative)). A mature storage story might add crash-safe incremental writes, snapshots, version history, and atomic dataset replacement — closer to modern storage engines than a static export format.

### Formal specification

Many formats fail because the spec lives only in code. Independent readers (Julia, Go, browser WASM) need:

- complete binary layout for every record type
- compatibility and deprecation rules
- stated invariants (index bounds, dtype tags, footer mutual exclusion)
- conformance tests shared across implementations

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
| **17** | Snapshots, atomic update, recovery | Writer protocol spec | Conflicts with pure WORM simplicity |
| **18** | Normative spec + conformance suite | Layout + query wire frozen | Enables third-party implementations |

Phases are **not ordered commitments** — e.g. formal spec (18) might parallelize early work on cloud reads (12).

---

## How to use this document

- **Contributors:** pick a direction, open a design issue with constraints from [`layout_v1.md`](layout_v1.md) and non-goals in the README.
- **Embedders:** treat everything above as **exploratory**; ship against v1 layout and documented query wire only.
- **Discussion:** [GitHub issue #16](https://github.com/Latka-Industries/tetration/issues/16) — long-term architecture and prospective phases.

When a direction ships, move detail into the relevant doc (`layout_v1.md`, `query_engine.md`, `ffi.md`) and mark the phase row **shipped** here.

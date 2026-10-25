# Distributed execution — product boundary (0.7.1 → 0.7.2)

**Status:** 0.7.1 boundary locked (THI-615 / THI-616 / THI-618). **0.7.2 shipped** — materialize / transform / tier-C open (THI-617 / [#169](https://github.com/Latka-Industries/tetration/pull/169)) per the RFC below. Semantics profiles shipped in **0.7.3** (THI-612); one-file TOML apply in **0.8.0** (THI-632); metadata wizard in **0.8.1** (THI-638); `tet run` job sequencer in **0.8.2** (THI-645); trust-boundary / `parse_job` fuzz in **0.8.3** (THI-650).  
**Module:** [`query::distributed`](../src/query/distributed/README.md)

## Product shape

Tetration stays a **format + embeddable library** with a **thin protocol**:

- One query over **one pinned sealed generation** (never re-resolve `CURRENT` mid-query).
- Coordinator shards a [`ReadPlan`](query_engine.md); workers process chunk subsets; coordinator merges.
- Wire: JSON [`WorkerJobV1`](../src/query/distributed/job.rs) / [`WorkerPartialV1`](../src/query/distributed/job.rs) / [`WorkerPinV1`](../src/query/distributed/job.rs) (protocol `1`).
- Fold path: associative tier-A/B [`FoldPartialState`](../src/query/distributed/merge.rs) (same laws as parallel / multi-GPU host merge).
- Dense path (0.7.2): sparse [`DenseShardPartial`](../src/query/distributed/dense.rs) contributions assembled by logical index.

This is **not** a cluster scheduler, Spark/Flink replacement, or a shipped worker-daemon product. Hosts embed the library (and may optionally spawn processes with the protocol).

## In / out for 0.7.1–0.7.2

| Cut | In scope | Out |
| --- | -------- | --- |
| **0.7.1** | Boundary RFC; spill/sidecar/switch **refused** on the distributed path (RAM partials only); clearer refuse errors | Worker-local spill files; shared DFS; advancing `CURRENT` mid-query |
| **0.7.1+ harness** | Localhost **N-process** harness (`tet worker` + coordinator spawn) proving the JSON protocol across OS processes | Network RPC service; Docker/K8s worker fleet; tet-py cluster manager |
| **0.7.2** | Distributed materialize / transform / tier-C on the thin protocol (THI-617) — **shipped** with `write: ram` only | General cluster orchestration; network RPC (later THI-630); distributed spill; multiprocess transform stats broadcast |

## Spill / export policy (THI-616, carried into 0.7.2)

`DistributedSpillPolicy::RamOnlyRefuseSpill` remains the only policy:

- `write: ram` or omit write → OK.
- `write: spill|sidecar|switch` on a distributed job → [`refuse_distributed_spill`](../src/query/distributed/spill.rs) validation error.

**0.7.2 does not open worker-local spill.** Dense outcomes stay in coordinator RAM after assemble. Spill/sidecar/switch remain single-process.

## Pin rule

Workers open the same `(generation, object_key, object_len, package_uri)` from `WorkerPinV1`. For the localhost harness, `package_uri` is a filesystem path (or `file://` URI) to a sealed `.tet`.

---

## 0.7.2 RFC — distributed materialize / transform (THI-617)

### Goals

Prove **parity with single-process** for a supported subset over a pinned `.tet`, using the same localhost N-shard / N-process pattern as scalar folds. Default intent: **open** the path (not permanent refuse).

### 1. What runs on workers vs coordinator

| Query class | Workers | Coordinator |
| --- | --- | --- |
| **Tier-A/B scalar fold** (unchanged) | Fold chunk shard → `FoldPartialState` | `merge_fold_partials` → `finish_fold_partial` |
| **Dense selection / tier-C** (`median`, `quantile`, `histogram`, covariance/correlation; selection materialize) | Decode shard chunks → sparse [`DenseShardPartial`](../src/query/distributed/dense.rs) (logical index + value) | Assemble full logical buffer by index; run tier-C (or return dense) **locally** |
| **Transform + `write: ram`** | **Pass-1** (in-process 0.7.2): coordinator collects global stats on the full plan (same mmap). **Pass-2**: workers apply transform on shard chunks → `DenseShardPartial` | Assemble dense RAM buffer; preview/stats from pass-1 |

**Why tier-C finishes on the coordinator:** median / quantile / histogram are not associative over disjoint chunk shards. Shipping wrong shard-local medians would silently disagree with single-process. Correct merge = materialize then finish once.

**Why transform is two-phase:** pass-1 stats (mean/std/min/…) are **global** over the selection. Shard-local zscore would be wrong. Pass-2 is embarrassingly parallel per element once stats are known.

**Multiprocess note (0.7.2):** workers already handle dense materialize jobs (no pass-1 stats on the wire). Transform multiprocess pass-2 (broadcast stats on `WorkerJobV1`) is a follow-on on the same shapes; in-process harness covers transform parity now.

### 2. How results combine (not `FoldPartialState`)

Dense shards are **sparse contributions**:

```text
DenseShardPartial { dtype, indices[], values_f32[], bytes_read }
```

- Each `(index, value)` is a logical row-major slot the worker wrote (including filter-mask NaNs).
- Coordinator allocates `logical_element_count` slots, scatters each shard’s pairs, rejects overlapping writes, then checks completeness (unless cell/label filters allow intentional holes).
- Protocol stays **`1`**: `WorkerPartialV1` keeps `partial` for folds; optional `dense` for materialize/transform shards. Exactly one payload is meaningful per response.

No protocol version bump required for this additive field.

### 3. Write targets

Keep **refuse** for `spill|sidecar|switch` on every distributed path. Supported write surface: omit write, or `write: ram`. Transform distributed requires `write: ram` (same as embed `materialize_query_transform_ram`).

### 4. Parity

Fixtures/tests assert N-shard in-process (and multiprocess where wired) answers match monolithic `plan_query_with_tet_mmap_ex` / `materialize_query_*` for:

- f32 dense selection assemble
- tier-C scalar `median` (global axes)
- transform `zscore` + `write: ram` (in-process)

### Still refused / deferred

- Distributed spill / sidecar / switch
- Network RPC (THI-630)
- Semantics profiles (0.7.3 / THI-612)
- Transform pass-1 sharded fold merge (optional optimization; correctness uses full-plan pass-1 today)
- Multiprocess transform stats broadcast (same wire, later harness)
- Non-f32 dense wire (f64/int follow the same sparse shape when needed)

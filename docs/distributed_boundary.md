# Distributed execution — product boundary (0.7.1)

**Status:** locked for the 0.7.1–0.7.3 wave (THI-615 / THI-616).  
**Module:** [`query::distributed`](../src/query/distributed/README.md)

## Product shape

Tetration stays a **format + embeddable library** with a **thin protocol**:

- One query over **one pinned sealed generation** (never re-resolve `CURRENT` mid-query).
- Coordinator shards a [`ReadPlan`](query_engine.md); workers fold chunk subsets; coordinator merges associative tier-A/B [`FoldPartialState`](../src/query/distributed/merge.rs).
- Wire: JSON [`WorkerJobV1`](../src/query/distributed/job.rs) / [`WorkerPartialV1`](../src/query/distributed/job.rs) / [`WorkerPinV1`](../src/query/distributed/job.rs) (protocol `1`).
- Same merge laws as parallel / multi-GPU host merge.

This is **not** a cluster scheduler, Spark/Flink replacement, or a shipped worker-daemon product. Hosts embed the library (and may optionally spawn processes with the protocol).

## In / out for 0.7.1–0.7.3

| Cut | In scope | Out |
| --- | -------- | --- |
| **0.7.1** | Boundary RFC (this doc); spill/sidecar/switch **refused** on the distributed path (RAM partials only); clearer refuse errors | Worker-local spill files; shared DFS; advancing `CURRENT` mid-query |
| **0.7.1+ harness** | Localhost **N-process** harness (`tet worker` + coordinator spawn) proving the JSON protocol across OS processes | Network RPC service; Docker/K8s worker fleet; tet-py cluster manager |
| **0.7.2** | Distributed materialize / transform **only if** a later RFC opens it | Default: keep refuse |
| **0.7.3** | Optional minimal transport polish on the same job/partial shapes | General cluster orchestration |

## Spill / export policy (THI-616)

`DistributedSpillPolicy::RamOnlyRefuseSpill` remains the only policy:

- `write: ram` or omit write → OK.
- `write: spill|sidecar|switch` on a distributed job → [`refuse_distributed_spill`](../src/query/distributed/spill.rs) validation error.

Scalar distributed folds do not materialize. Spill/sidecar/switch stay single-process query features until a future cut explicitly opens worker-local spill.

## Pin rule

Workers open the same `(generation, object_key, object_len, package_uri)` from `WorkerPinV1`. For the localhost harness, `package_uri` is a filesystem path (or `file://` URI) to a sealed `.tet`.

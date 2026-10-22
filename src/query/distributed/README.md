# `query::distributed` — shard a fold / dense op across workers

Thin protocol for **one query** over **one pinned sealed generation**, split by chunk
shards. Not a cluster scheduler.

**Product boundary:** [`docs/distributed_boundary.md`](../../../docs/distributed_boundary.md)
(0.7.1 THI-615 / THI-616 / THI-618; **0.7.2** materialize/transform THI-617).

## What ships

| Piece | Role |
| --- | --- |
| `plan_chunk_shards` / `split_read_plan` | Contiguous or interleaved partition of `ReadPlan.chunks` |
| `WorkerJobV1` / `WorkerPinV1` / `WorkerPartialV1` | JSON job + pin + partial (protocol `1`) |
| `FoldPartialState` + `merge_fold_partials` / `finish_fold_partial` | Associative tier-A/B merge |
| `DenseShardPartial` + `assemble_dense_f32` | Sparse dense contributions (0.7.2) |
| `execute_sharded_scalar_fold_local` | In-process N-shard scalar fold |
| `execute_sharded_materialize_local` / `execute_sharded_tier_c_local` | In-process dense materialize / tier-C |
| `execute_sharded_transform_ram_local` | In-process transform `write: ram` (pass-1 local, pass-2 sharded) |
| `execute_worker_job_v1` | Run one job against a local pin (library + `tet worker`) |
| `execute_sharded_scalar_fold_multiprocess` | Spawn N `tet worker` processes; merge fold partials |
| `execute_sharded_materialize_multiprocess` / `execute_sharded_tier_c_multiprocess` | Spawn N workers; assemble dense / finish tier-C |
| `refuse_distributed_spill` | Spill/sidecar/switch write targets → clear error |

## Pin rule

Workers open the **same** `(generation, object_key, object_len)` from [`WorkerPinV1`].
Do not re-read package `CURRENT` mid-query.

## Local demo (in-process dense / tier-C)

```rust
use tetration::query::distributed::{
    ShardPartition, execute_sharded_tier_c_local, operation_is_dense_distributable,
};
use tetration::query::{parse_query_json, plan_query_with_tet_mmap_ex, validate_query};
use tetration::layout::mmap_file_read;
use tetration::utils::dtype::ElementDtype;

let doc = parse_query_json(r#"{"dataset":"a","median":[]}"#)?;
validate_query(&doc)?;
let mmap = mmap_file_read(path)?;
let planned = plan_query_with_tet_mmap_ex(&doc, None, &mmap, Some(0), None)?;
let plan = planned.read_plan.as_ref().unwrap();
let op = doc.operation.as_ref().unwrap();
assert!(operation_is_dense_distributable(op));
let preview = execute_sharded_tier_c_local(
    &mmap,
    plan,
    op,
    ElementDtype::F32,
    4,
    ShardPartition::ContiguousRanges,
    None,
)?;
```

Tests in `src/tests/distributed.rs` assert N-shard results match monolithic execute for
sum / mean / var / min / arg_min, dense selection, median, and transform zscore (`write: ram`).
Integration test `tests/distributed_multiprocess.rs` spawns real `tet worker` processes
(`mise distributed:smoke`).

## CLI

```bash
# Job JSON on stdin or --job PATH; WorkerPartialV1 JSON on stdout
tet worker --job shard0.json
tet worker < shard0.json
```

## Spill policy (locked 0.7.1–0.7.2)

`write: ram` (or omit) only. `refuse_distributed_spill` rejects `spill` / `sidecar` /
`switch`. Dense outcomes assemble in coordinator RAM. See the [boundary RFC](../../../docs/distributed_boundary.md).

## Non-goals (through 0.7.3 unless reopened)

- Network RPC / worker daemon product
- Worker-local spill files or a shared distributed filesystem
- Advancing `CURRENT` while a query is open
- Docker/Kubernetes worker fleets
- Multiprocess transform stats broadcast (in-process transform is open; wire follow-on)

# `query::distributed` — shard a fold across workers (0.5.0)

Thin protocol for **one query** over **one pinned sealed generation**, split by chunk
shards. Not a cluster scheduler.

## What ships

| Piece | Role |
| --- | --- |
| `plan_chunk_shards` / `split_read_plan` | Contiguous or interleaved partition of `ReadPlan.chunks` |
| `WorkerJobV1` / `WorkerPinV1` / `WorkerPartialV1` | JSON job + pin + partial (protocol `1`) |
| `FoldPartialState` + `merge_fold_partials` / `finish_fold_partial` | Associative tier-A/B merge |
| `execute_sharded_scalar_fold_local` | In-process N-shard harness (no RPC) |
| `refuse_distributed_spill` | Spill/sidecar/switch write targets → clear error |

## Pin rule

Workers open the **same** `(generation, object_key, object_len)` from [`WorkerPinV1`].
Do not re-read package `CURRENT` mid-query.

## Local demo (harness)

```rust
use tetration::query::distributed::{
    ShardPartition, execute_sharded_scalar_fold_local, operation_is_shard_mergeable,
};
use tetration::query::{parse_query_json, plan_query_with_tet_mmap_ex, validate_query};
use tetration::layout::mmap_file_read;
use tetration::utils::dtype::ElementDtype;

let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#)?;
validate_query(&doc)?;
let mmap = mmap_file_read(path)?;
let planned = plan_query_with_tet_mmap_ex(&doc, None, &mmap, Some(0), None)?;
let plan = planned.read_plan.as_ref().unwrap();
let op = doc.operation.as_ref().unwrap();
assert!(operation_is_shard_mergeable(op));
let preview = execute_sharded_scalar_fold_local(
    &mmap,
    plan,
    op,
    ElementDtype::F32,
    4, // n_shards
    ShardPartition::ContiguousRanges,
    None,
)?;
```

Tests in `src/tests/distributed.rs` assert N-shard results match monolithic execute for
sum / mean / var / min / arg_min.

## Non-goals (0.5.0)

- RPC / networking (JSON shapes are the protocol skeleton)
- Distributed materialize, transform, median/quantile/histogram
- Worker-local spill files or a shared distributed filesystem
- Advancing `CURRENT` while a query is open

## Spill policy

`write: ram` (or omit) only. `refuse_distributed_spill` rejects `spill` / `sidecar` /
`switch`. Keep partials in RAM; finish scalars at the coordinator.

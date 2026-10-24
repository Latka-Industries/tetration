# `query/types` — query wire types

Serde models for the **flat** query document and execution response. No I/O here — parsing lives in `document*.rs`, planning in `plan/`, execution in `engine/`.

## Files

| File                  | Main types                                                                |
| --------------------- | ------------------------------------------------------------------------- |
| `document.rs`         | `QueryDocument`, `Operation`, `AxisSlice`, `AxisToken`, spill vs `write` helpers |
| `simple_op.rs`        | `SimpleOp` — axes-only reduction kinds (`sum`…`nan_std`)                  |
| `plan.rs`             | `ReadPlan`, `PlannedChunkIo`, `ChunkTouchPolicy` (enum + wire tokens)     |
| `response.rs`         | `QueryResponse`, `QueryExecutionPreview`, `DatasetResolution`             |
| `error.rs`            | `TetError`                                                                |
| `transform_method.rs` | `TransformMethod` enum (`zscore`, `minmax`, `softmax`, …)                 |

## `QueryDocument` shape

Flat keys at the top level (no nested `"operation"` objects):

```json
{ "dataset": "temperature", "mean": [], "selection": { … } }
```

Operations are mutually exclusive per document (one reduction or transform family).

## `ReadPlan`

Output of planning: logical selection shape, dtype, ordered chunk I/O descriptors (mmap offset, stored length, codec, in-chunk slice). Consumed by `decode` and `materialize`.

## `QueryResponse`

Combines validation result, optional `read_plan`, catalog snapshot, and `execution` block (stats, preview samples, spill paths, warnings).

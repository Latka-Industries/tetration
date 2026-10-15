# `utils` — shared low-level helpers

Internal utilities used across `layout`, `catalog`, `query`, and `convert`. Only **`dtype`**, **`fs_device`**, and **`host_memory`** are public.

## Modules

| Module           | Visibility    | Role                                                                             |
| ---------------- | ------------- | -------------------------------------------------------------------------------- |
| `dtype.rs`       | **pub**       | `ElementDtype` — wire tag ↔ Rust type, elem sizes, tensor byte math              |
| `wire.rs`        | crate-private | Re-exports [`argus`](https://docs.rs/argus-chunk) align8 / span checks; LE I/O via `LeReader`/`LeWriter` at call sites |
| `le_pod.rs`      | mixed         | `f32_le`, `f64_le`, … — decode POD slices from chunk bytes                       |
| `fs_device.rs`   | **pub**       | Volume identity + `publish_file` (rename vs cross-volume copy for spill/sidecar) |
| `host_memory.rs` | **pub**       | Host RAM capacity probes for materialize budget                                  |

## `ElementDtype`

Central mapping for catalog wire tags 1–10 (`F32`, `F64`, `I32`, …). Query `dispatch.rs` branches on this enum for per-type materialize/fold paths.

## Why separate from `catalog`?

Keeps wire-format primitives free of catalog error types and allows `query` + `convert` to share dtype logic without circular dependencies.

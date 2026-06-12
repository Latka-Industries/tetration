# C ABI (Phase 11)

Stable **`extern "C"`** entry points for non-Rust runtimes. The Rust crate (`tetration`) remains the full API; the shared library exposes a **narrow** subset aligned with CLI parity where it matters.

## Python

Official Python bindings are **not** built from this repository. They live in [**tet-py**](https://github.com/Latka-Industries/tet-py): PyPI package **`tet-py`**, **`import tet`**, wheels built with PyO3 / maturin and a pinned **`tetration`** release on [crates.io](https://crates.io/crates/tetration) (e.g. **0.1.9** for tet-py **0.1.1**). Do not `pip install tetration` — that PyPI name is unrelated math code.

**Shipped in tet-py:** `tet.open`, JSON/TOML query + reductions, NumPy read (ram / spill / sidecar), `TetWriter` / `write_dataset` (f32/f64), `preview=N` on reductions. **Planned there:** optional convert extras (`h5py`, `netCDF4`, `zarr`), `read_numpy` preflight, integer write dtypes, zero-copy mmap reads.

For hosts that are not Python: shell **`tet`**, embed **Rust**, use this C ABI, or implement readers from [`layout_v1.md`](layout_v1.md).

## Status

**ABI v1** behind Cargo feature **`tetration-ffi`**:

| Symbol                               | Role                                               |
| ------------------------------------ | -------------------------------------------------- |
| `tet_abi_version`                    | Must match `#define TET_ABI_VERSION` in the header |
| `tet_open` / `tet_close`             | Read-only `.tet` handle                            |
| `tet_last_error` / `tet_clear_error` | Thread-local UTF-8 error text                      |
| `tet_summary_json`                   | Catalog summary JSON                               |
| `tet_query_json`                     | Query document JSON → `QueryResponse` JSON         |
| `tet_verify_json`                    | Quick verify report JSON (path only)               |
| `tet_string_free`                    | Free buffers from `*_json`                         |

Header: [`include/tetration.h`](../include/tetration.h). Example: [`examples/ffi_query.c`](../examples/ffi_query.c).

## Build and test

```bash
# Rust FFI unit tests (use default features so the lib test crate compiles)
cargo test --lib --features tetration-ffi ffi

# Lean shared library (no HDF5 / NetCDF inside libtetration)
cargo build --release --no-default-features --features tetration-ffi
```

Lean builds only compile Zarr-side convert code; `src/convert/mod.rs` allows expected `dead_code` / `unused_imports` there when both import features are off. Default-feature builds are unchanged.

Artifacts:

| Platform | Library                             |
| -------- | ----------------------------------- |
| Linux    | `target/release/libtetration.so`    |
| macOS    | `target/release/libtetration.dylib` |
| Windows  | `target/release/tetration.dll`      |

Header / symbol sync (CI and local):

```bash
./.github/scripts/check-ffi-header.sh
```

### C example

```bash
cargo build --release --no-default-features --features tetration-ffi

cc -std=c11 -Wall -Wextra -I include examples/ffi_query.c \
  -L target/release -ltetration -o target/release/ffi_query

# macOS
DYLD_LIBRARY_PATH=target/release target/release/ffi_query fixtures/small/tet/sample.tet

# Linux
LD_LIBRARY_PATH=target/release target/release/ffi_query fixtures/small/tet/sample.tet
```

Or run the helper (build + smoke on `sample.tet`):

```bash
./.github/scripts/build-ffi-example.sh
```

### GitHub Releases

On tag push (`v*`), CI attaches per-platform archives to the draft release:

- `tetration-ffi-<tag>-linux-x86_64.tar.gz`
- `tetration-ffi-<tag>-macos-aarch64.tar.gz`
- `tetration-ffi-<tag>-windows-x86_64.tar.gz`

Each contains `include/tetration.h`, `lib/` (`.so` / `.dylib` / `.dll`), and `README.txt`. Built with `--no-default-features --features tetration-ffi`.

## Linking notes

- Link against **`libtetration`** and ship the shared library next to your binary, or set `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` during development.
- Compile with **`-I include`** so `#include "tetration.h"` resolves.
- Call **`tet_abi_version()`** at startup and compare to **`TET_ABI_VERSION`** from the header you compiled against.
- Every `*_json` return value must be released with **`tet_string_free`** (including after errors that still allocated — none today, but check null first).
- **`tet_last_error`** is valid until the next FFI call on the same thread; copy the string if you need it later.

## Design principles (v1)

1. **JSON for documents** — query in and structured result out match `tet query -x` / `execute_query_json`, so bindings do not marshal Rust structs by hand.
2. **Opaque handles** — `TetHandle` owns mmap + path; no Rust types across the FFI edge.
3. **Lean library** — build with **`default-features = false`** when you only need open/query/verify.
4. **`TET_ABI_VERSION`** — bump on breaking C symbol or calling-convention changes (independent of crate semver patch).

**Out of v1:** convert/import, writer session, GPU device selection, query history.

## Stability

- **0.x crate:** C ABI may change between minors until `1.0`; watch release notes and `TET_ABI_VERSION`.
- **Panics:** FFI entry points use `catch_unwind`; release `cdylib` builds use `panic = "abort"` in `[profile.release]`.

## See also

- Embedder Rust API: [`README.md` — Library use](../README.md#library-use)
- Query JSON wire: [`query_engine.md`](query_engine.md)
- Layout for standalone readers: [`layout_v1.md`](layout_v1.md)

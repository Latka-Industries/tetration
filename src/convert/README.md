# `convert` — foreign formats → `.tet`

Importers for **HDF5**, **NetCDF**, and **Zarr v3** directory stores. Used by `tet convert` and `convert_to_tet` in the library.

**Docs:** [`docs/conversions.md`](../../docs/conversions.md) — supported formats, roadblocks, potential importers.

## Features

| Cargo feature                | Enables                             |
| ---------------------------- | ----------------------------------- |
| `tetration-hdf5` (default)   | `.h5` / HDF5 signature              |
| `tetration-netcdf` (default) | `.nc` / NetCDF signature            |
| _(always)_                   | Zarr v3 (`zarr.json` at store root) |

Build with `--no-default-features` for Zarr-only + query (no system HDF5/NetCDF libs).

## Public API

- `convert_to_tet` / `convert_to_tet_with_progress` — auto-detect format, stream chunks
- `detect_convert_format` — extension + magic sniff
- `default_parallel_jobs` / `resolve_parallel_jobs` — `--jobs` semantics
- Per-format: `convert_h5_to_tet_*`, `convert_netcdf_to_tet_*`, `convert_zarr_to_tet_*`

Returns `ConvertReport` (dataset names, dims, history row, elapsed).

## Submodules

| File                         | Role                                                       |
| ---------------------------- | ---------------------------------------------------------- |
| `sniff.rs`                   | Extension peel (`.gz`, etc.) + file signature              |
| `shared.rs`                  | `ImportPlan` — shape, chunk_shape, dtype, name per dataset |
| `tile_io.rs`                 | Read one logical tile from source (hyperslab)              |
| `parallel.rs`                | Rayon workers with per-thread file handles                 |
| `import_metadata.rs`         | Map source attrs → footer `metadata`                       |
| `cf.rs`                      | CF conventions: `scale_factor`, `add_offset`, `_FillValue` |
| `hdf5.rs` / `hdf5_shared.rs` | HDF5 traversal and dtype mapping                           |
| `netcdf.rs`                  | NetCDF variables and dims                                  |
| `zarr.rs`                    | Zarr v3 metadata + chunk keys                              |

## Pipeline

```text
sniff → list datasets → ImportPlan per array → parallel tile read → catalog::stream_write → THST history
```

Output files are sealed single-writer artifacts; readers mmap after convert completes.

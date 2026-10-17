# `convert` — foreign formats → `.tet`

Importers for **HDF5**, **NetCDF**, **Zarr** v2/v3, **NumPy** `.npy`/`.npz`, **CSV**/TSV, and **Parquet** (`tetration-parquet`). Used by `tet convert` and `convert_to_tet` in the library.

**Docs:** [`docs/conversions.md`](../../docs/conversions.md) — supported formats, roadblocks, potential importers. Export (the inverse): [`export/`](../export/README.md).

## Features

| Cargo feature                | Enables                                      |
| ---------------------------- | -------------------------------------------- |
| `tetration-hdf5` (default)   | `.h5` / HDF5 signature                       |
| `tetration-netcdf` (default) | `.nc` / NetCDF signature                     |
| `tetration-parquet`          | `.parquet` / `.pq` (off by default)          |
| _(always)_                   | Zarr v2/v3, NumPy, CSV / TSV                 |

Build with `--no-default-features` for Zarr/NumPy/CSV + query (no system HDF5/NetCDF libs).

## Public API

- `convert_to_tet` / `convert_to_tet_with_progress` / `convert_to_tet_with_options` — auto-detect format, stream chunks (`ConvertOptions.strict` / `chunk_mib`)
- `detect_convert_format` — extension + magic sniff
- `default_parallel_jobs` / `resolve_parallel_jobs` — `--jobs` semantics
- Per-format: `convert_h5_to_tet_*`, `convert_netcdf_to_tet_*`, `convert_zarr_to_tet_*`, `convert_npy_to_tet_*`, `convert_npz_to_tet_*`, `convert_csv_to_tet_*`, `convert_parquet_to_tet_*`

Returns `ConvertReport` (dataset names, dims, `skipped` paths, history row, elapsed).

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
| `zarr.rs`                    | Zarr v2/v3 metadata + chunk keys                           |
| `numpy.rs`                   | `.npy` / `.npz`                                     |
| `csv.rs` / `category.rs`     | CSV/TSV columns + category codes                           |
| `parquet.rs`                 | Parquet columns (feature-gated)                            |

## Pipeline

```text
sniff → list datasets → ImportPlan per array → parallel tile read → catalog::stream_write → THST history
```

Output files are sealed single-writer artifacts; readers mmap after convert completes.

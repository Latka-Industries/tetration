# `export` — `.tet` → Zarr v3 / HDF5 / NetCDF

Round-trip interchange from an existing `.tet`:

| Target | API | Feature | Notes |
| ------ | --- | ------- | ----- |
| **Zarr v3** directory | `export_tet_to_zarr` | always | Copies per-chunk **raw or zstd** bytes (no recompression) |
| **HDF5** file | `export_tet_to_hdf5` | `tetration-hdf5` | Dense numeric write; slash names → groups |
| **NetCDF-4** file | `export_tet_to_netcdf` | `tetration-netcdf` | Dense numeric write; `dim_names` → dimensions |

Dispatcher: `export_tet` / `export_tet_with_progress` pick the writer from the output path (directory / `.zarr` → Zarr; `.h5`/`.hdf5` → HDF5; `.nc`/`.netcdf` → NetCDF).

## Public API

- `export_tet` / `export_tet_with_progress`
- `export_tet_to_zarr` / `export_tet_to_hdf5` / `export_tet_to_netcdf` (+ `_with_progress`)
- `ExportOutputFormat`, `ExportReport`, `ExportProgress`, `ExportError`

## Implementation

- `common.rs` — densify (decode + scatter tiles), shared report/error types, basic attr filter
- `zarr.rs` — chunk-preserving Zarr v3 directory store
- `hdf5.rs` / `netcdf.rs` — feature-gated dense writers (attrs: `units` / `long_name` / …; typed CF packing attrs skipped)

Used by `tet export <in.tet> <out>`. Zarr output must be missing or empty; HDF5/NetCDF paths must not already exist.

**Docs:** [`docs/conversions.md`](../../docs/conversions.md) — export limits, round-trip notes.

## Related

- Import path: [`convert/`](../convert/README.md)
- On-disk `.tet` layout: [`catalog`](../catalog/README.md)

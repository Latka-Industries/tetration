# Format conversions — status, roadblocks, and prospects

**Status:** reference for `tet convert` / `tet export` and library importers. Describes **what works today**, **known gaps**, and **potential future** interchange paths.

Implementation: [`src/convert/`](../src/convert/), [`src/export/`](../src/export/). CLI: [`cli.md`](cli.md). On-disk target: [`layout_v1.md`](layout_v1.md).

---

## At a glance

| Direction | Formats | CLI / API |
| --------- | ------- | --------- |
| **Import → `.tet`** | HDF5, NetCDF, Zarr v3 | `tet convert <in> <out.tet>` · `convert_to_tet` |
| **Export `.tet` →** | Zarr v3 directory only | `tet export <in.tet> <out.zarr/>` · `export_tet_to_zarr` |
| **Not implemented** | HDF5 / NetCDF / Parquet / Arrow / `.npy` export; Zarr v2 import; cloud URIs as convert input | — |

Tetration’s interchange model is **dense numeric N-D arrays** in a **single sealed `.tet` file**. Converters map foreign arrays into that model; they do not preserve every source construct (VLEN strings, compound types, ragged tables, SQL schemas).

---

## Convert pipeline (all importers)

```text
sniff format → walk groups/variables/arrays → ImportPlan per dataset
  → parallel tile read (hyperslab) → optional CF decode → stream_write chunks
  → THST footer (history + metadata)
```

| Stage | Behavior |
| ----- | -------- |
| **Sniff** | Extension (case-insensitive), optional compression suffix peel (`.gz`, `.bz2`, … — **name only**, file is not decompressed), then magic bytes or Zarr v3 `zarr.json` |
| **Plan** | One catalog dataset per supported numeric array; nested groups → slash names (`group/var`) |
| **Chunk grid** | Reuse source chunking when rank matches; else **one tile = full array** |
| **CF decode** | `scale_factor`, `add_offset`, `_FillValue` applied at tile read (HDF5 / NetCDF); stored values become decoded floats + NaN for fill |
| **Write** | Streaming multi-dataset `.tet`; payloads **raw** or **zstd** (v1 catalog codecs) |
| **Metadata** | Selected attrs, `dim_names`, coordinate labels → footer `metadata` (64 KiB inline limit; larger metadata spills — see layout doc) |

Readers mmap the sealed output; convert is **single-writer**, not read-while-write.

---

## Supported today

### Import: HDF5 (feature `tetration-hdf5`, default)

| Topic | Detail |
| ----- | ------ |
| **Sniff** | `.h5`, `.hdf5`, `.hdf`, `.he2`, `.he5`, or `\x89HDF\r\n\x1a\n` signature |
| **Walk** | Recursive groups; unsupported datasets **skipped silently** |
| **Dtypes** | `f32`, `f64`, `u8` (incl. boolean), `i16`, `u16`, `i32`, `u32`, `i64`, `u64` |
| **Skipped** | Scalars, 0-sized arrays, `f16`, strings, compounds, enums, opaque, VLEN, most exotic HDF5 types |
| **CF** | Decode at import when attrs present; **requires `f32`/`f64` storage** for packed variables |
| **Chunking** | HDF5 chunk shape when no CF packing; CF-packed vars → full-array chunk grid |
| **Metadata** | Preferred attrs (`units`, `long_name`, `standard_name`, …); CF coordinate enrichment for 1-D numeric coord datasets |
| **Parallelism** | `--jobs N` (0 = host default, cap 64); per-worker HDF5 file handle |

### Import: NetCDF (feature `tetration-netcdf`, default)

| Topic | Detail |
| ----- | ------ |
| **Sniff** | `.nc`, `.netcdf`, `.nc4`, `.nc3`, `.cdf`, … or NetCDF-3 `CDF\x01` / `CDF\x02` magic |
| **NetCDF-4** | Files stored in HDF5 — use a **NetCDF extension** so the NetCDF importer runs (signature-only sniff may pick HDF5) |
| **Walk** | Groups + root-level variables; 0-D variables skipped |
| **Dtypes** | `f32`, `f64`, `u8`/`i8`, `u16`, `i16`, `i32`, `i64` — **not** `u32`/`u64`/`f16` |
| **CF** | Same decode rules as HDF5 |
| **Chunking** | NetCDF chunking when present; CF-packed → full array |
| **Metadata** | Variable attrs, dimension names, self-coordinate labels when discoverable |

### Import: Zarr v3 (always available)

| Topic | Detail |
| ----- | ------ |
| **Sniff** | Directory with root `zarr.json`, `zarr_format: 3` |
| **Walk** | `group` / `array` nodes; unsupported arrays skipped |
| **Dtypes** | Wire-aligned set: `float16`–`float64`, `uint8`–`uint64`, `int16`–`int64`, `bool`/`int8` → `u8` |
| **Chunk grid** | `chunk_grid.name == "regular"` only |
| **Codecs** | **`bytes`** (little-endian) or **`bytes` + `zstd`** on chunk files — other codec chains fail at read |
| **CF** | Not applied (values stored as in Zarr) |
| **Metadata** | Array `attributes` copied to footer attrs |

### Export: `.tet` → Zarr v3

| Topic | Detail |
| ----- | ------ |
| **Output** | New or **empty** directory; creates nested groups for slash-separated names |
| **Payloads** | Copies stored chunk bytes (**raw** or **zstd**); no recompression |
| **Dtypes** | All v1 wire tags (`f32` … `u64`, `f16`) |
| **Metadata** | Footer dataset attrs merged into Zarr `attributes` + `tetration_dtype` stamp |
| **Limit** | Per-dataset chunks must be **all raw or all zstd** — mixed codecs error |

### Build / deploy variants

| Build | Convert capability |
| ----- | ------------------ |
| **Default** (`tetration-hdf5` + `tetration-netcdf`) | HDF5, NetCDF, Zarr |
| **`--no-default-features`** | Zarr import + query on existing `.tet` (no system HDF5/NetCDF libs) |
| **`tetration-ffi` lean lib** | Same as no-default-features inside shared library |

System packages for default builds: see [README — cargo install](../README.md).

---

## Known roadblocks (today)

These are intentional limits, silent skips, or sharp edges — not necessarily bugs.

### Silent dataset drops

Importers **skip** unsupported variables/arrays (`UnsupportedDtype`) and continue. Convert **fails only when zero** supported datasets remain (`NoDatasets`). A file that “mostly converted” may be missing string metadata datasets, compound tables, or auxiliary types with no error per skipped name.

**Mitigation today:** compare `tet info` dataset list to source; check `ConvertReport.dataset_names` in library use.

**Future:** verbose skip log, `--strict` mode, or convert report JSON listing skipped paths.

### Type system mismatch

| Source concept | Tetration v1 |
| -------------- | ------------ |
| HDF5 compound / enum / VLEN string | Not imported |
| NetCDF char / string variables | Not imported |
| Categorical / labeled integers | Stored as integers; labels only if copied as attrs |
| Booleans | `u8` (0/1) |
| Complex numbers | Not supported |
| Ragged / jagged arrays | Not supported |

### CF conventions (partial)

| Supported at import | Not supported |
| ------------------- | ------------- |
| `scale_factor`, `add_offset`, `_FillValue` → decode + NaN | `valid_min` / `valid_max`, `missing_value` as non-fill sentinels (unless mapped to fill) |
| Decode for **`f32`/`f64` storage** | CF packing stored as integers (`i16`, …) — **rejected** |
| Attr copy (`units`, `long_name`, …) | Full CF compliance checker; CRS, cell methods, bounds |

### Compression and transport

| Issue | Detail |
| ----- | ------ |
| **`.nc.gz` extension peel** | Sniff sees `.nc`, but file body is still gzip — open/decode **not** implemented |
| **Zarr codecs** | Only raw `bytes` or `bytes`+`zstd`; Blosc, Gzip, Sharding, `transpose` codecs → read failure |
| **HDF5 filters** | Read via HDF5 library (filter plugins must be available); exotic filters may fail at source |
| **Cloud objects** | Convert expects a **local path**; no `s3://` input ([roadmap — Phase 12](roadmap.md#1-cloud-native-operation-prospective-phase-12)) |

### Chunking and geometry

| Issue | Detail |
| ----- | ------ |
| **Rechunk on import** | Source chunk shape reused when valid; mismatch or missing chunking → **one chunk = full array** (large RAM pressure during convert of huge arrays) |
| **CF-packed arrays** | Chunk shape forced to full array (decode requires full tile context) |
| **Non-regular Zarr grid** | Only `chunk_grid: regular` |
| **Rotate / permute layout** | Row-major `.tet` only; no automatic layout transform |

### Metadata and round-trip

| Issue | Detail |
| ----- | ------ |
| **Footer size cap** | Large metadata may spill; importers trim/prefer attrs — not every source attr preserved |
| **Zarr-only export** | No `tet export` to HDF5/NetCDF; round-trip is **`.tet` ↔ Zarr v3** for interchange |
| **History** | Convert writes one `convert` history row; full provenance chain not reconstructed from source |
| **Dim/coord fidelity** | NetCDF dim names and 1-D coords imported when found; arbitrary HDF5 soft-link graphs not preserved |

### Operational

| Issue | Detail |
| ----- | ------ |
| **Single output file** | Entire archive → one `.tet`; no sharded `.tet` family on convert |
| **Memory** | Parallel workers × chunk buffers; very large tiles can stress RAM |
| **Windows** | HDF5/NetCDF dev libs harder to install; CI hints in [`.github/scripts/`](../.github/scripts/) |

---

## Error reference

| Error | Typical cause |
| ----- | ------------- |
| `UnsupportedInputExtension` | Unknown extension and unrecognized signature |
| `ConvertFeatureDisabled` | HDF5/NetCDF path with `--no-default-features` build |
| `NoDatasets` | Every array filtered out (types, empty file) |
| `UnsupportedDtype { name, detail }` | Per-dataset (often swallowed during walk) |
| `Hdf5` / `Netcdf` / `Zarr` | Library I/O, corrupt file, codec mismatch |
| `OutputNotEmpty` (export) | Target Zarr directory not empty |

---

## Potential future conversions

Prioritized by fit with Tetration’s **dense chunked array** model. Effort is qualitative, not a schedule.

### Tier A — natural extensions (same shape: dense N-D arrays)

| Format | Value | Main roadblocks |
| ------ | ----- | ---------------- |
| **NumPy `.npy` / `.npz`** | Ubiquitous in Python science stack | `.npz` multi-array naming; default chunk policy; Fortran-order arrays; pickling in `.npz` (reject) |
| **Zarr v2** | Large installed base | Different metadata (`/.zgroup`, `.zarray`); codec matrix; v2→v3 mapping |
| **HDF5 export** | Round-trip with HDF5 ecosystems | Writer API, filter/chunk choices, group attrs, compound types not in v1 |
| **NetCDF export** | CF workflows, tools expecting `.nc` | Shared dimensions, coordinate variables, CF attrs, unlimited dims |
| **`f16` HDF5** | Match wire tag 9 | Half-float HDF5 type support in reader |

### Tier B — columnar / table formats (mapping required)

| Format | Value | Main roadblocks |
| ------ | ----- | ---------------- |
| **Apache Parquet** | Analytics, cloud warehouses | Columnar vs N-D tensor; nested/repeated fields; row groups ≠ chunk grid; statistics as index seed ([roadmap — Phase 13](roadmap.md#2-smarter-indexing-prospective-phase-13)) |
| **Apache Arrow / Feather / IPC** | Zero-copy interchange | Record batches vs single ndarray; schema with nulls; chunked arrays |
| **CSV / TSV** | Human interchange | Not chunked; infer types; size limits; poor fit for `.tet` strengths |

**Design question:** import as **one dataset per column** (wide table) vs **materialized tensor** (requires reshape rules).

### Tier C — domain geospatial / signal formats

| Format | Value | Main roadblocks |
| ------ | ----- | ---------------- |
| **GeoTIFF / COG** | GIS rasters | 2D/3D bias; tile compression (JPEG, LZW); CRS/geotransform semantics ([roadmap — Phase 16](roadmap.md#5-scientific-semantics-prospective-phase-16)); irregular grids |
| **GRIB / GRIB2** | Weather/climate | Complex templates; spectral vs grid; eccodes dependency |
| **FITS** | Astronomy | HDU model; BSCALE/BZERO overlap with CF; WCS |
| **NIfTI** | Medical imaging | 3D/4D; affine headers; gzip |

These often need **scientific semantics** (units, CRS, bounds) before raw convert is useful — see [`roadmap.md`](roadmap.md).

### Tier D — poor fit or out of scope

| Format | Why deferred |
| ------ | ------------ |
| **SQLite / DuckDB / Postgres** | Relational; joins and SQL are [non-goals](../README.md) for v1 query |
| **JSON / BSON documents** | Schema-less; no regular chunk grid |
| **Protobuf / Avro** | Row/event models |
| **Video / audio codecs** | Not dense strided arrays |

### Cross-cutting features (any new importer)

| Feature | Why it matters |
| ------- | -------------- |
| **Convert report JSON** (`skipped`, `warnings`, dtype map) | Operability when datasets are silently dropped |
| **`--strict`** | Fail on first unsupported dataset |
| **Rechunk policy flag** | Target chunk bytes (e.g. 4–64 MiB) on import |
| **Cloud read source** | Convert from object store without full download |
| **Streaming write from URL** | Pairs with cloud-native Phase 12 |
| **Python bindings repo** | [`tet-py`](https://github.com/Latka-Industries/tet-py) — `tet.convert` planned ([#10](https://github.com/Latka-Industries/tet-py/issues/10)); `h5py` / `xarray` / `zarr` via optional extras — see [`ffi.md`](ffi.md) |

Fixtures policy: [`fixtures/README.md`](../fixtures/README.md) — new formats get golden files **when** convert support lands.

---

## Round-trip fidelity checklist

Use when validating **source → `.tet` → Zarr → `.tet`** or comparing to source:

1. **Dataset names** — same count and paths (slash groups)?
2. **Shape / dtype** — wire tag matches expected promotion (e.g. bool → `u8`)?
3. **Values** — spot-check reductions (`mean`, `nan_count`) on full array; CF-decoded NetCDF vs raw Zarr path differs by design
4. **Chunk count** — may differ if source chunking was missing (full-array tiles in `.tet`)
5. **Metadata** — `tet info --metadata` vs source attrs; spill if large
6. **Codecs** — export preserves raw/zstd; re-import Zarr only supports same codec subset

---

## Related docs

| Doc | Contents |
| --- | -------- |
| [`cli.md`](cli.md) | `tet convert` / `tet export` flags |
| [`layout_v1.md`](layout_v1.md) | Wire dtypes, footer metadata, codecs |
| [`roadmap.md`](roadmap.md) | Cloud I/O, indexing, semantics (affects convert priorities) |
| [`src/convert/README.md`](../src/convert/README.md) | Module map for contributors |
| [`src/export/README.md`](../src/export/README.md) | Zarr export internals |

**Discussion:** extend this doc via PR; track new format requests in [issue #17](https://github.com/Latka-Industries/tetration/issues/17). Python-side convert orchestration: [tet-py#10](https://github.com/Latka-Industries/tet-py/issues/10).

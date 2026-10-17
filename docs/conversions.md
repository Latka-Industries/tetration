# Format conversions — status, roadblocks, and prospects

**Status:** reference for `tet convert` / `tet export` and library importers. Describes **what works today**, **known gaps**, and **potential future** interchange paths.

Implementation: [`src/convert/`](../src/convert/), [`src/export/`](../src/export/). CLI: [`cli.md`](cli.md). On-disk target: [`layout_v1.md`](layout_v1.md).

---

## At a glance

| Direction | Formats | CLI / API |
| --------- | ------- | --------- |
| **Import → `.tet`** | HDF5, NetCDF, Zarr v2 / v3, NumPy `.npy` / `.npz`, CSV / TSV, Parquet (`tetration-parquet`) | `tet convert <in> <out.tet>` · `convert_to_tet` |
| **Export `.tet` →** | Zarr v3 directory; HDF5 / NetCDF (`tetration-hdf5` / `tetration-netcdf`) | `tet export <in.tet> <out>` · `export_tet` |
| **Not implemented** | Parquet / Arrow / `.npy` export; Zarr-on-S3 chunked reads | — |

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
| **Chunk grid** | Reuse source chunking when rank matches; else **one tile = full array**. NumPy, CSV, and Parquet have no source chunks, so they use a **64 MiB** leading-axis tile. `--chunk-mib N` retile every dataset to about N MiB and replaces a source grid |
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
| **Walk** | Recursive groups; unsupported datasets are **skipped** and listed on `ConvertReport.skipped` (CLI stderr). `--strict` fails instead of writing |
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
| **Walk** | `group` / `array` nodes; unsupported arrays skipped and listed (`--strict` fails the convert) |
| **Dtypes** | Wire-aligned set: `float16`–`float64`, `uint8`–`uint64`, `int16`–`int64`, `bool`/`int8` → `u8` |
| **Chunk grid** | `chunk_grid.name == "regular"` only |
| **Codecs** | **`bytes`** (little-endian) or **`bytes` + `zstd`** on chunk files — other codec chains fail at read |
| **CF** | Not applied (values stored as in Zarr) |
| **Metadata** | Array `attributes` copied to footer attrs |

### Import: Zarr v2 (always available)

| Topic | Detail |
| ----- | ------ |
| **Sniff** | Directory with root `.zgroup` or `.zarray`, or a `.zarr` path. v3 `zarr.json` still wins when both could apply |
| **Walk** | `.zgroup` children; a root `.zarray` with no group name becomes dataset `data` |
| **Dtypes** | Little-endian `<f2`, `<f4`, `<f8`, `<i2`, `<i4`, `<i8`, `<u2`, `<u4`, `<u8`, `\|u1`. `\|i1` / `bool` → `u8`. Big-endian is a skip |
| **Order** | C-order only. Fortran-order is a skip |
| **Codecs** | No compressor, or `{"id":"zstd"}`. Blosc, gzip, and filters are skips. Chunk keys use `.` or `/` |
| **History** | Source tag `zarr2` |
| **Metadata** | `.zattrs` copied to footer attrs |
| **CF** | Not applied |

### Import: NumPy `.npy` / `.npz` (always available)

| Topic | Detail |
| ----- | ------ |
| **Sniff** | `.npy`, `.npz`, or `\x93NUMPY` magic. A bare zip signature is not treated as `.npz` |
| **`.npy`** | One C-order numeric array → dataset `data`. Payload is read from the header data offset in tiles (the array is not loaded whole) |
| **`.npz`** | One dataset per `.npy` member (`weights.npy` → `weights`, `group/bias.npy` → `group/bias`). Duplicate catalog names are an error. Stored members are sliced in the zip; compressed members are inflated one at a time to a temp file, then tiled |
| **Rejected** | Fortran-order, object / pickled dtypes (`\|O`). These fail the convert; they are not skips |
| **Skipped** | Scalars, 0-sized arrays, big-endian, bool, complex, strings, structured dtypes. `--strict` still refuses to write |
| **Dtypes** | Little-endian `<f2` (`f16`), `<f4`, `<f8`, `<i2`, `<i4`, `<i8`, `\|u1`, `<u2`, `<u4`, `<u8` |
| **Chunking** | No source grid. Default tiles are about **64 MiB**, splitting the leading axis so each tile is one contiguous read. Smaller arrays stay one tile |
| **CF** | Not applied |

### Import: CSV / TSV (always available)

| Topic | Detail |
| ----- | ------ |
| **Sniff** | `.csv` or `.tsv`. Delimiter is comma or tab from the extension. No content sniff |
| **Header** | The first row is a header when any cell is not a number. Otherwise columns are `col0`, `col1`, … and that row is data. Cells are trimmed. A line with no fields is ignored |
| **Mapping** | One 1-D dataset per numeric column, in file order. Not one 2-D table (columns may differ in dtype) |
| **Dtypes** | A column of plain integers is `i64`. A decimal, exponent, `nan`, or `inf` makes the column `f64`. A blank cell is `NaN` and promotes an integer column to `f64` |
| **Categories** | A column of text labels becomes an `i64` dataset. Codes start at 0 in first-seen order. A blank is `-1`. Footer attr `tetration_categories` is `{"codes":{…},"missing":-1}`. The JSON must fit in one 1024-byte attribute |
| **Skipped** | Mixed number/text columns, category maps over that 1024-byte limit, and columns that are empty or only blank. Listed on `ConvertReport.skipped`. `--strict` still refuses to write |
| **Rejected** | Ragged rows, and two columns with the same name. No `.tet` is written |
| **Chunking** | ~64 MiB tiles along the column, same default as NumPy. CSV is parsed into column buffers first (rows are not hyperslabs). `--chunk-mib N` overrides the tile size |
| **CF** | Not applied |

### Import: Parquet (feature `tetration-parquet`)

| Topic | Detail |
| ----- | ------ |
| **Sniff** | `.parquet`, `.pq`, or `PAR1` magic |
| **Mapping** | One 1-D dataset per primitive numeric column, same family as CSV. The file is read into column buffers, then tiled at ~64 MiB |
| **Dtypes** | `int16`/`int32`/`int64`, `uint8`–`uint64`, `float32`, `float64`. A null in an integer column promotes that column to `f64` with `NaN`. A null in a float column is `NaN` |
| **Categories** | UTF-8 columns use the CSV category rule: `i64` codes, blank/null is `-1`, footer attr `tetration_categories` when the JSON fits in 1024 bytes |
| **Skipped** | Bool, timestamps, decimals, nested lists and structs, and other Arrow types. `--strict` still refuses to write |
| **Build** | Off by default. `cargo build --features tetration-parquet` |

### Export: `.tet` → Zarr v3

| Topic | Detail |
| ----- | ------ |
| **Output** | New or **empty** directory; creates nested groups for slash-separated names |
| **Payloads** | Copies stored chunk bytes (**raw** or **zstd**); no recompression |
| **Dtypes** | All v1 wire tags (`f32` … `u64`, `f16`) |
| **Metadata** | Footer dataset attrs merged into Zarr `attributes` + `tetration_dtype` stamp |
| **Limit** | Per-dataset chunks must be **all raw or all zstd** — mixed codecs error |

### Export: `.tet` → HDF5 (feature `tetration-hdf5`, default)

| Topic | Detail |
| ----- | ------ |
| **Output** | New `.h5` / `.hdf5` / … file (must not exist); slash names → nested groups |
| **Payloads** | Densifies decoded tiles into one contiguous array per dataset (chunk grid preserved when writing) |
| **Dtypes** | `f32`/`f64`/`i16`/`u16`/`i32`/`u32`/`i64`/`u64`/`u8` — **not** `f16` |
| **Metadata** | Basic string attrs (`units`, `long_name`, `standard_name`, …); `dim_names` as a comma-joined attr when present; typed CF packing attrs (`scale_factor`, `_FillValue`, …) are **not** rewritten |

### Export: `.tet` → NetCDF-4 (feature `tetration-netcdf`, default)

| Topic | Detail |
| ----- | ------ |
| **Output** | New `.nc` / `.netcdf` / … file (must not exist); slash names → nested groups |
| **Payloads** | Same densify path as HDF5 export |
| **Dtypes** | `f32`/`f64`/`u8`/`i16`/`u16`/`i32`/`i64` — **not** `f16` / `u32` / `u64` |
| **Dimensions** | Footer `dim_names` when present and length-matched; otherwise synthetic `{leaf}_d{i}` per dataset. Shared dims reuse matching name+length in a group |
| **Metadata** | Same basic string attrs as HDF5 export; not full CF rewrite |

CLI / library pick the writer from the output path via [`export_tet`](../src/export/mod.rs).

### Build / deploy variants

| Build | Convert / export capability |
| ----- | ------------------ |
| **Default** (`tetration-hdf5` + `tetration-netcdf`) | HDF5 + NetCDF import **and** export; Zarr v2/v3, NumPy, CSV import; Zarr export |
| **`--features tetration-parquet`** | Adds Parquet import (Arrow). Not in the default build |
| **`--no-default-features`** | Zarr v2/v3, NumPy, and CSV import; Zarr-only export; query on existing `.tet` (no system HDF5/NetCDF libs) |
| **`tetration-ffi` lean lib** | Same as no-default-features inside shared library |

System packages for default builds: see [README — cargo install](../README.md).

---

## Known roadblocks (today)

These are intentional limits, silent skips, or sharp edges — not necessarily bugs.

### Silent dataset drops

Importers **skip** unsupported variables/arrays (`UnsupportedDtype`) and continue. Convert **fails only when zero** supported datasets remain (`NoDatasets`), unless **`--strict`** is set.

Each skip is a `{path, reason}` on [`ConvertReport.skipped`](../src/convert/mod.rs) and in the CLI JSON report. The CLI also prints `skipped N dataset(s):` on stderr. `--strict` returns `StrictSkips` **before writing** the `.tet`.

Scalars / 0-d NetCDF variables are recorded the same way.

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
| **Cloud objects** | Single-file `s3://bucket/object.ext` downloads the whole object to a temp file, then the existing importer (`tetration-s3`, `TETRATION_S3_ENDPOINT`). Zarr-on-S3 chunked reads are later |

### Chunking and geometry

| Issue | Detail |
| ----- | ------ |
| **Rechunk on import** | Default reuses a valid source chunk shape. Missing chunking is still one full-array tile, except NumPy / CSV / Parquet (64 MiB). `--chunk-mib N` forces a leading-axis tile of about N MiB, including over a source grid (Zarr chunk files are assembled into the new tiles) |
| **CF-packed arrays** | Without `--chunk-mib`, chunk shape stays the full array. The flag retile those arrays too; CF scale and fill are applied per element |
| **Non-regular Zarr grid** | Only `chunk_grid: regular` |
| **Rotate / permute layout** | Row-major `.tet` only; no automatic layout transform |

### Metadata and round-trip

| Issue | Detail |
| ----- | ------ |
| **Footer size cap** | Large metadata may spill; importers trim/prefer attrs — not every source attr preserved |
| **Export subset** | HDF5/NetCDF export densifies (not chunk-byte identity); typed CF packing attrs are not rewritten; `f16` (and NetCDF `u32`/`u64`) rejected |
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
| `OutputNotEmpty` / `OutputExists` (export) | Target Zarr directory not empty, or HDF5/NetCDF path already exists |
| `ExportFeatureDisabled` | HDF5/NetCDF export with `--no-default-features` build |

---

## Potential future conversions

Prioritized by fit with Tetration’s **dense chunked array** model. Effort is qualitative, not a schedule.

### Tier A — natural extensions (same shape: dense N-D arrays)

| Format | Value | Main roadblocks |
| ------ | ----- | ---------------- |
| **Zarr v2** | Shipped: C-order numeric arrays, raw or zstd chunks | Blosc, gzip, and other filters stay skips |
| **HDF5 export** | Shipped (`tetration-hdf5`): dense numeric + basic attrs | Compounds, `f16`, typed CF packing attrs, exotic filters |
| **NetCDF export** | Shipped (`tetration-netcdf`): dense numeric + `dim_names` | Full CF/CRS, unlimited dims, `u32`/`u64`/`f16` |
| **`f16` HDF5** | Match wire tag 9 | Half-float HDF5 type support in reader |

### Tier B — columnar / table formats (mapping required)

| Format | Value | Main roadblocks |
| ------ | ----- | ---------------- |
| **Apache Parquet** | Shipped behind `tetration-parquet`: one 1-D dataset per numeric column, category codes for UTF-8 | Nested lists, structs, and Parquet export are not implemented |
| **Apache Arrow / Feather / IPC** | Zero-copy interchange | Record batches vs single ndarray; schema with nulls; chunked arrays |
| **CSV / TSV** | Shipped: one 1-D dataset per numeric column, plus integer codes for category columns | Delimiter is the extension only (comma or tab). No custom delimiter flag |

**Design question:** import as **one dataset per column** (wide table) vs **materialized tensor** (requires reshape rules). CSV uses one 1-D dataset per numeric column.

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
| **Convert report JSON** (`skipped` shipped; extra warnings / dtype map still open) | Operability when datasets are left out |
| **`--strict`** | Shipped: fail before write when any dataset was skipped |
| **Rechunk policy flag** | Shipped: `--chunk-mib N` / `ConvertOptions::chunk_mib` |
| **Cloud read source** | Convert from object store without full download |
| **Streaming write from URL** | Pairs with cloud-native Phase 12 |
| **Python bindings repo** | [`tet-py`](https://github.com/Latka-Industries/tet-py) — `tet.convert` planned ([#10](https://github.com/Latka-Industries/tet-py/issues/10)); `h5py` / `xarray` / `zarr` via optional extras — see [`ffi.md`](ffi.md) |

Fixtures policy: [`fixtures/README.md`](../fixtures/README.md) — new formats get golden files **when** convert support lands.

---

## Round-trip fidelity checklist

Use when validating **source → `.tet` → Zarr/HDF5/NetCDF → `.tet`** or comparing to source:

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
| [`src/export/README.md`](../src/export/README.md) | Zarr / HDF5 / NetCDF export internals |

**Discussion:** extend this doc via PR; track new format requests in [issue #17](https://github.com/Latka-Industries/tetration/issues/17). Python-side convert orchestration: [tet-py#10](https://github.com/Latka-Industries/tet-py/issues/10).

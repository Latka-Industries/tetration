# `export` — `.tet` → Zarr v3

Round-trip interchange: writes a **Zarr v3 directory store** from an existing `.tet`, preserving per-chunk **raw or zstd** bytes (no recompression).

## Public API

- `export_tet_to_zarr` / `export_tet_to_zarr_with_progress`
- `ExportReport` — datasets exported, chunk count, elapsed
- `ExportError`

## Implementation

All logic is in `zarr.rs`:

- Mmap-read source `.tet` via catalog summary
- Create `zarr.json` + nested groups for slash-separated dataset names
- Copy stored chunk payloads and metadata compatible with Zarr v3 layout

Used by `tet export <in.tet> <out.zarr/>`. Output directory must be missing or empty.

**Docs:** [`docs/conversions.md`](../../docs/conversions.md) — export limits, round-trip notes, future formats.

## Related

- Import path: [`convert/zarr.rs`](../convert/README.md)
- On-disk `.tet` layout: [`catalog`](../catalog/README.md)

"""Paths and sizing constants for fixture generation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

FIXTURES_ROOT = Path(__file__).resolve().parent.parent

SMALL_H5 = FIXTURES_ROOT / "small" / "h5"
SMALL_NC = FIXTURES_ROOT / "small" / "netcdf"
SMALL_ZARR = FIXTURES_ROOT / "small" / "zarr"
LARGE_H5 = FIXTURES_ROOT / "large" / "h5"
LARGE_NC = FIXTURES_ROOT / "large" / "netcdf"
LARGE_ZARR = FIXTURES_ROOT / "large" / "zarr"
EXTRA_LARGE_H5 = FIXTURES_ROOT / "extra_large" / "h5"
EXTRA_LARGE_NC = FIXTURES_ROOT / "extra_large" / "netcdf"
EXTRA_LARGE_ZARR = FIXTURES_ROOT / "extra_large" / "zarr"
# Bench-only small stress tier (not fixtures/small smoke shapes).
BENCH_SMALL_H5 = FIXTURES_ROOT / "bench_small" / "h5"
BENCH_SMALL_NC = FIXTURES_ROOT / "bench_small" / "netcdf"
BENCH_SMALL_ZARR = FIXTURES_ROOT / "bench_small" / "zarr"

SMALL_SHAPES: dict[int, tuple[int, ...]] = {
    3: (32, 32, 32),
    4: (16, 16, 16, 16),
    5: (8, 8, 8, 8, 8),
}

NUMERIC_DTYPES: tuple[str, ...] = ("f32", "f64", "i32", "i64")

DtypeName = Literal["f32", "f64", "i32", "i64"]

# Large stress suite: ~20 GiB logical f32 **total** split across HDF5, NetCDF, and Zarr.
LARGE_SUITE_LOGICAL_BYTES = 20 * 1024**3
LARGE_FORMATS = 3
LARGE_PER_FORMAT_BYTES = LARGE_SUITE_LOGICAL_BYTES // LARGE_FORMATS
FLOAT32_BYTES = 4
LARGE_PER_FORMAT_ELEMS = LARGE_PER_FORMAT_BYTES // FLOAT32_BYTES

# Single-file stress tensors (one format, full 20 GiB logical f32).
EXTRA_LARGE_LOGICAL_BYTES = 20 * 1024**3
EXTRA_LARGE_ELEM_COUNT = EXTRA_LARGE_LOGICAL_BYTES // FLOAT32_BYTES

# Bench-large "small" tier: multi-chunk, above WINDOW_BENCH_SIZE, fast to generate.
# 512 MiB f32 → 8 × 64 MiB slabs; distinct from tracked fixtures/small/.
BENCH_SMALL_LOGICAL_BYTES = 512 * 1024**2
BENCH_SMALL_ELEM_COUNT = BENCH_SMALL_LOGICAL_BYTES // FLOAT32_BYTES

# ~64 MiB of f32 per write slab (keeps peak RAM low while building multi-GiB files).
CHUNK_ELEMS = 16 * 1024 * 1024

SEED_SMALL = 0
SEED_LARGE = 1
SEED_BENCH_SMALL = 21

GenerateTarget = Literal[
    "small",
    "large",
    "large-h5",
    "large-netcdf",
    "large-zarr",
    "all",
    "extra-large-h5",
    "extra-large-netcdf",
    "extra-large-zarr",
    "bench-small",
    "bench-small-h5",
    "bench-small-netcdf",
    "bench-small-zarr",
]

ExtraLargeTarget = Literal["extra-large-h5", "extra-large-netcdf", "extra-large-zarr"]
LargeSingleTarget = Literal["large-h5", "large-netcdf", "large-zarr"]
BenchSmallTarget = Literal["bench-small-h5", "bench-small-netcdf", "bench-small-zarr"]

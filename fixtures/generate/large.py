"""Gitignored large / extra_large fixture tensors."""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import netCDF4 as nc
import numpy as np
import zarr

from generate.constants import (
    BENCH_SMALL_H5,
    BENCH_SMALL_LOGICAL_BYTES,
    BENCH_SMALL_NC,
    BENCH_SMALL_ZARR,
    CHUNK_ELEMS,
    EXTRA_LARGE_H5,
    EXTRA_LARGE_LOGICAL_BYTES,
    EXTRA_LARGE_NC,
    EXTRA_LARGE_ZARR,
    FIXTURES_ROOT,
    FLOAT32_BYTES,
    LARGE_H5,
    LARGE_NC,
    LARGE_PER_FORMAT_BYTES,
    LARGE_SUITE_LOGICAL_BYTES,
    LARGE_ZARR,
    SEED_BENCH_SMALL,
    SEED_LARGE,
)
from generate.util import ZARR_RAW_ARRAY_KWARGS, iter_slabs, progress, slab_count


def write_h5_f32_slabs(
    path: Path,
    *,
    logical_bytes: int,
    fixture: str,
    suite_total_bytes: int | None,
    quiet: bool,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    total = logical_bytes // FLOAT32_BYTES
    shape = (total,)
    n_slabs = slab_count(total)
    gib = logical_bytes / (1024**3)

    if not quiet:
        print(f"HDF5: {path} ({gib:.2f} GiB logical f32, {n_slabs} slabs)")

    with h5py.File(path, "w") as f:
        f.attrs["tetration_fixture"] = fixture
        f.attrs["tetration_logical_bytes"] = int(logical_bytes)
        if suite_total_bytes is not None:
            f.attrs["tetration_suite_total_bytes"] = int(suite_total_bytes)
        dset = f.create_dataset(
            "data",
            shape=shape,
            dtype=np.float32,
            chunks=(CHUNK_ELEMS,),
        )
        for start, end in progress(
            iter_slabs(total),
            total=n_slabs,
            desc=path.name,
            quiet=quiet,
            unit="slab",
        ):
            dset[start:end] = rng.random(end - start, dtype=np.float32)

    if not quiet:
        print(f"  finished {path}")


def write_nc_f32_slabs(
    path: Path,
    *,
    logical_bytes: int,
    fixture: str,
    suite_total_bytes: int | None,
    quiet: bool,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    total = logical_bytes // FLOAT32_BYTES
    n_slabs = slab_count(total)
    gib = logical_bytes / (1024**3)

    if not quiet:
        print(f"NetCDF: {path} ({gib:.2f} GiB logical f32, {n_slabs} slabs)")

    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.setncattr("tetration_fixture", fixture)
        ds.setncattr("tetration_logical_bytes", int(logical_bytes))
        if suite_total_bytes is not None:
            ds.setncattr("tetration_suite_total_bytes", int(suite_total_bytes))
        ds.createDimension("i", total)
        var = ds.createVariable("data", "f4", ("i",), chunksizes=(CHUNK_ELEMS,))
        for start, end in progress(
            iter_slabs(total),
            total=n_slabs,
            desc=path.name,
            quiet=quiet,
            unit="slab",
        ):
            var[start:end] = rng.random(end - start, dtype=np.float32)

    if not quiet:
        print(f"  finished {path}")


def write_zarr_f32_slabs(
    path: Path,
    *,
    logical_bytes: int,
    fixture: str,
    suite_total_bytes: int | None,
    quiet: bool,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)
    rng = np.random.default_rng(seed)
    total = logical_bytes // FLOAT32_BYTES
    shape = (total,)
    n_slabs = slab_count(total)
    gib = logical_bytes / (1024**3)

    if not quiet:
        print(f"Zarr: {path} ({gib:.2f} GiB logical f32, {n_slabs} slabs)")

    root = zarr.open_group(str(path), mode="w")
    attrs: dict[str, object] = {
        "tetration_fixture": fixture,
        "tetration_logical_bytes": int(logical_bytes),
    }
    if suite_total_bytes is not None:
        attrs["tetration_suite_total_bytes"] = int(suite_total_bytes)
    root.attrs.update(attrs)
    arr = root.create_array(
        "data",
        shape=shape,
        chunks=(CHUNK_ELEMS,),
        dtype="f4",
        **ZARR_RAW_ARRAY_KWARGS,
    )
    for start, end in progress(
        iter_slabs(total),
        total=n_slabs,
        desc=path.name,
        quiet=quiet,
        unit="slab",
    ):
        arr[start:end] = rng.random(end - start, dtype=np.float32)

    if not quiet:
        print(f"  finished {path}")


def _write_h5_large(path: Path, *, quiet: bool) -> None:
    write_h5_f32_slabs(
        path,
        logical_bytes=LARGE_PER_FORMAT_BYTES,
        fixture="large_suite_h5",
        suite_total_bytes=LARGE_SUITE_LOGICAL_BYTES,
        quiet=quiet,
        seed=SEED_LARGE,
    )


def _write_nc_large(path: Path, *, quiet: bool) -> None:
    write_nc_f32_slabs(
        path,
        logical_bytes=LARGE_PER_FORMAT_BYTES,
        fixture="large_suite_nc",
        suite_total_bytes=LARGE_SUITE_LOGICAL_BYTES,
        quiet=quiet,
        seed=SEED_LARGE + 1,
    )


def _write_zarr_large(path: Path, *, quiet: bool) -> None:
    write_zarr_f32_slabs(
        path,
        logical_bytes=LARGE_PER_FORMAT_BYTES,
        fixture="large_suite_zarr",
        suite_total_bytes=LARGE_SUITE_LOGICAL_BYTES,
        quiet=quiet,
        seed=SEED_LARGE + 2,
    )


def generate_extra_large_h5(*, quiet: bool) -> None:
    if not quiet:
        gib = EXTRA_LARGE_LOGICAL_BYTES / (1024**3)
        print(f"Generating extra-large HDF5 under {EXTRA_LARGE_H5} ({gib:.0f} GiB)")
    write_h5_f32_slabs(
        EXTRA_LARGE_H5 / "tensor_20gb.h5",
        logical_bytes=EXTRA_LARGE_LOGICAL_BYTES,
        fixture="extra_large_h5",
        suite_total_bytes=None,
        quiet=quiet,
        seed=SEED_LARGE + 10,
    )
    if not quiet:
        print(f"Done: {EXTRA_LARGE_H5}")


def generate_extra_large_netcdf(*, quiet: bool) -> None:
    if not quiet:
        gib = EXTRA_LARGE_LOGICAL_BYTES / (1024**3)
        print(f"Generating extra-large NetCDF under {EXTRA_LARGE_NC} ({gib:.0f} GiB)")
    write_nc_f32_slabs(
        EXTRA_LARGE_NC / "tensor_20gb.nc",
        logical_bytes=EXTRA_LARGE_LOGICAL_BYTES,
        fixture="extra_large_nc",
        suite_total_bytes=None,
        quiet=quiet,
        seed=SEED_LARGE + 11,
    )
    if not quiet:
        print(f"Done: {EXTRA_LARGE_NC}")


def generate_extra_large_zarr(*, quiet: bool) -> None:
    if not quiet:
        gib = EXTRA_LARGE_LOGICAL_BYTES / (1024**3)
        print(f"Generating extra-large Zarr under {EXTRA_LARGE_ZARR} ({gib:.0f} GiB)")
    write_zarr_f32_slabs(
        EXTRA_LARGE_ZARR / "tensor_20gb",
        logical_bytes=EXTRA_LARGE_LOGICAL_BYTES,
        fixture="extra_large_zarr",
        suite_total_bytes=None,
        quiet=quiet,
        seed=SEED_LARGE + 12,
    )
    if not quiet:
        print(f"Done: {EXTRA_LARGE_ZARR}")


def generate_large(*, quiet: bool) -> None:
    suite_gib = LARGE_SUITE_LOGICAL_BYTES / (1024**3)
    per_gib = LARGE_PER_FORMAT_BYTES / (1024**3)
    if not quiet:
        print(
            f"Generating large suite under {FIXTURES_ROOT / 'large'} "
            f"({suite_gib:.0f} GiB total ≈ {per_gib:.2f} GiB per format)"
        )
    _write_h5_large(LARGE_H5 / "tensor_large.h5", quiet=quiet)
    _write_nc_large(LARGE_NC / "tensor_large.nc", quiet=quiet)
    _write_zarr_large(LARGE_ZARR / "tensor_large", quiet=quiet)
    if not quiet:
        print(f"Done: {FIXTURES_ROOT / 'large'}")


def generate_large_h5(*, quiet: bool) -> None:
    if not quiet:
        gib = LARGE_PER_FORMAT_BYTES / (1024**3)
        print(f"Generating large HDF5 under {LARGE_H5} ({gib:.2f} GiB logical f32)")
    _write_h5_large(LARGE_H5 / "tensor_large.h5", quiet=quiet)
    if not quiet:
        print(f"Done: {LARGE_H5}")


def generate_large_netcdf(*, quiet: bool) -> None:
    if not quiet:
        gib = LARGE_PER_FORMAT_BYTES / (1024**3)
        print(f"Generating large NetCDF under {LARGE_NC} ({gib:.2f} GiB logical f32)")
    _write_nc_large(LARGE_NC / "tensor_large.nc", quiet=quiet)
    if not quiet:
        print(f"Done: {LARGE_NC}")


def generate_large_zarr(*, quiet: bool) -> None:
    if not quiet:
        gib = LARGE_PER_FORMAT_BYTES / (1024**3)
        print(f"Generating large Zarr under {LARGE_ZARR} ({gib:.2f} GiB logical f32)")
    _write_zarr_large(LARGE_ZARR / "tensor_large", quiet=quiet)
    if not quiet:
        print(f"Done: {LARGE_ZARR}")


def generate_bench_small_h5(*, quiet: bool) -> None:
    if not quiet:
        mib = BENCH_SMALL_LOGICAL_BYTES / (1024**2)
        print(f"Generating bench-small HDF5 under {BENCH_SMALL_H5} ({mib:.0f} MiB logical f32)")
    write_h5_f32_slabs(
        BENCH_SMALL_H5 / "tensor_small.h5",
        logical_bytes=BENCH_SMALL_LOGICAL_BYTES,
        fixture="bench_small_h5",
        suite_total_bytes=None,
        quiet=quiet,
        seed=SEED_BENCH_SMALL,
    )
    if not quiet:
        print(f"Done: {BENCH_SMALL_H5}")


def generate_bench_small_netcdf(*, quiet: bool) -> None:
    if not quiet:
        mib = BENCH_SMALL_LOGICAL_BYTES / (1024**2)
        print(f"Generating bench-small NetCDF under {BENCH_SMALL_NC} ({mib:.0f} MiB logical f32)")
    write_nc_f32_slabs(
        BENCH_SMALL_NC / "tensor_small.nc",
        logical_bytes=BENCH_SMALL_LOGICAL_BYTES,
        fixture="bench_small_nc",
        suite_total_bytes=None,
        quiet=quiet,
        seed=SEED_BENCH_SMALL + 1,
    )
    if not quiet:
        print(f"Done: {BENCH_SMALL_NC}")


def generate_bench_small_zarr(*, quiet: bool) -> None:
    if not quiet:
        mib = BENCH_SMALL_LOGICAL_BYTES / (1024**2)
        print(f"Generating bench-small Zarr under {BENCH_SMALL_ZARR} ({mib:.0f} MiB logical f32)")
    write_zarr_f32_slabs(
        BENCH_SMALL_ZARR / "tensor_small",
        logical_bytes=BENCH_SMALL_LOGICAL_BYTES,
        fixture="bench_small_zarr",
        suite_total_bytes=None,
        quiet=quiet,
        seed=SEED_BENCH_SMALL + 2,
    )
    if not quiet:
        print(f"Done: {BENCH_SMALL_ZARR}")


def generate_bench_small(*, quiet: bool) -> None:
    generate_bench_small_h5(quiet=quiet)
    generate_bench_small_netcdf(quiet=quiet)
    generate_bench_small_zarr(quiet=quiet)

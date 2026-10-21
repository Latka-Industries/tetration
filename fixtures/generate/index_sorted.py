"""Sorted-axis index bench fixtures (THI-613): HDF5 with monotonic time coords.

Footer coord labels cap at 64, so time stays length 64 and **width** carries the
bytes. Tiers:

- **small** — ~256 MiB file, select ≈16 MiB (microbench / fixed-cost)
- **large** — ~6.7 GiB, select ≈0.42 GiB
- **extra** — 20 GiB, select ≈1.25 GiB

One chunk per time step so a label-range select reads a known fraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from generate.constants import (
    EXTRA_LARGE_LOGICAL_BYTES,
    FIXTURES_ROOT,
    FLOAT32_BYTES,
    LARGE_PER_FORMAT_BYTES,
    SEED_LARGE,
)

IndexTier = Literal["small", "large", "extra"]

# Must stay ≤ MetadataLimitsV1.coord_labels_per_axis (64).
INDEX_TIME = 64

# Half-open label range on time (integer labels "0".."63" from Display).
INDEX_SELECT_START = "10"
INDEX_SELECT_STOP = "14"
INDEX_SELECT_LEN = 4  # stop - start → 4/64 of logical bytes

# Fixed width for the small microbench (~4 MiB f32 per time step → 16 MiB select).
SMALL_INDEX_WIDTH = 1_048_576


@dataclass(frozen=True)
class IndexFixture:
    tier: IndexTier
    logical_bytes: int
    width: int
    root: Path
    h5: Path
    tet: Path
    pkg: Path

    @property
    def select_bytes(self) -> int:
        return INDEX_SELECT_LEN * self.width * FLOAT32_BYTES

    @property
    def logical_gib(self) -> float:
        return self.logical_bytes / (1024**3)

    @property
    def select_gib(self) -> float:
        return self.select_bytes / (1024**3)

    @property
    def select_mib(self) -> float:
        return self.select_bytes / (1024**2)


def fixture_for(tier: IndexTier) -> IndexFixture:
    """Paths + geometry for `small` / `large` / `extra`."""
    if tier == "small":
        width = SMALL_INDEX_WIDTH
        logical_bytes = INDEX_TIME * width * FLOAT32_BYTES
        root = FIXTURES_ROOT / "large" / "index_small"
    elif tier == "large":
        logical = LARGE_PER_FORMAT_BYTES
        root = FIXTURES_ROOT / "large" / "index"
        width = logical // (INDEX_TIME * FLOAT32_BYTES)
        if width < 1:
            raise ValueError("logical size too small for INDEX_TIME")
        logical_bytes = INDEX_TIME * width * FLOAT32_BYTES
    elif tier == "extra":
        logical = EXTRA_LARGE_LOGICAL_BYTES
        root = FIXTURES_ROOT / "extra_large" / "index"
        width = logical // (INDEX_TIME * FLOAT32_BYTES)
        if width < 1:
            raise ValueError("logical size too small for INDEX_TIME")
        logical_bytes = INDEX_TIME * width * FLOAT32_BYTES
    else:
        raise ValueError(f"unknown index tier {tier!r}; choose small, large, or extra")
    return IndexFixture(
        tier=tier,
        logical_bytes=logical_bytes,
        width=width,
        root=root,
        h5=root / "tensor_sorted_axis.h5",
        tet=root / "tensor_sorted_axis.tet",
        pkg=root / "tensor_sorted_axis.tet.d",
    )


def write_sorted_axis_h5(fx: IndexFixture, *, quiet: bool = False) -> None:
    """Write 2-D f32 HDF5: shape (time, width), chunk (1, width), CF time coords."""
    fx.h5.parent.mkdir(parents=True, exist_ok=True)
    if fx.h5.is_file():
        fx.h5.unlink()
    seed_off = {"small": 0, "large": 1, "extra": 2}[fx.tier]
    rng = np.random.default_rng(SEED_LARGE + 613 + seed_off)
    # Write x in slabs so peak RAM stays modest on the 20 GiB path.
    slab = min(fx.width, 16 * 1024 * 1024)
    if not quiet:
        print(
            f"HDF5 sorted-axis [{fx.tier}]: {fx.h5} "
            f"({fx.logical_gib:.2f} GiB logical, select≈{fx.select_mib:.1f} MiB, "
            f"time={INDEX_TIME}, width={fx.width}, chunks=(1, {fx.width}))"
        )
    with h5py.File(fx.h5, "w") as f:
        f.attrs["tetration_fixture"] = "sorted_axis_index"
        f.attrs["tetration_tier"] = fx.tier
        f.attrs["tetration_logical_bytes"] = int(fx.logical_bytes)
        coords = f.create_group("coordinates")
        coords.create_dataset("time", data=np.arange(INDEX_TIME, dtype=np.float64))
        coords["time"].attrs["units"] = "seconds"
        dset = f.create_dataset(
            "data",
            shape=(INDEX_TIME, fx.width),
            dtype=np.float32,
            chunks=(1, fx.width),
        )
        dset.attrs["coordinates"] = "time"
        for t in range(INDEX_TIME):
            for x0 in range(0, fx.width, slab):
                x1 = min(x0 + slab, fx.width)
                dset[t, x0:x1] = rng.random(x1 - x0, dtype=np.float32)
            if not quiet and (t + 1) % 8 == 0:
                print(f"  wrote time {t + 1}/{INDEX_TIME}")
    if not quiet:
        print(f"  finished {fx.h5}")


def generate_sorted_axis_index(tier: IndexTier = "large", *, quiet: bool = False) -> IndexFixture:
    """Generate (or overwrite) the sorted-axis HDF5 for `tier`; return fixture paths."""
    fx = fixture_for(tier)
    write_sorted_axis_h5(fx, quiet=quiet)
    return fx

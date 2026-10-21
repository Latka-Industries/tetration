"""Sorted-axis index bench fixture (THI-613): HDF5 with monotonic time coords.

Footer coord labels cap at 64 (`MetadataLimitsV1.coord_labels_per_axis`), so the
time axis is length 64 and width carries the bytes. One chunk per time step.
HDF5 CF `coordinates` import attaches labels without dim_names; seal matches the
unique axis whose label count equals the time extent.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from generate.constants import FIXTURES_ROOT, SEED_LARGE

# Time extent must stay ≤ footer coord label cap (64).
INDEX_TIME = 64
# ~4 MiB f32 per time step → ~256 MiB logical total.
INDEX_WIDTH = 1_048_576
INDEX_LOGICAL_BYTES = INDEX_TIME * INDEX_WIDTH * 4

INDEX_ROOT = FIXTURES_ROOT / "large" / "index"
INDEX_H5 = INDEX_ROOT / "tensor_sorted_axis.h5"
INDEX_TET = INDEX_ROOT / "tensor_sorted_axis.tet"
INDEX_PKG = INDEX_ROOT / "tensor_sorted_axis.tet.d"

# Half-open label range on time (integer labels "0".."63" from Display).
INDEX_SELECT_START = "10"
INDEX_SELECT_STOP = "14"
INDEX_SELECT_LEN = 4  # stop - start


def write_sorted_axis_h5(path: Path = INDEX_H5, *, quiet: bool = False) -> None:
    """Write 2-D f32 HDF5: shape (time, x), chunk (1, width), CF time coords."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        path.unlink()
    rng = np.random.default_rng(SEED_LARGE + 613)
    gib = INDEX_LOGICAL_BYTES / (1024**3)
    if not quiet:
        print(
            f"HDF5 sorted-axis: {path} ({gib:.2f} GiB logical, "
            f"time={INDEX_TIME}, width={INDEX_WIDTH}, chunks=({1}, {INDEX_WIDTH}))"
        )
    with h5py.File(path, "w") as f:
        f.attrs["tetration_fixture"] = "sorted_axis_index"
        f.attrs["tetration_logical_bytes"] = int(INDEX_LOGICAL_BYTES)
        coords = f.create_group("coordinates")
        coords.create_dataset("time", data=np.arange(INDEX_TIME, dtype=np.float64))
        coords["time"].attrs["units"] = "seconds"
        dset = f.create_dataset(
            "data",
            shape=(INDEX_TIME, INDEX_WIDTH),
            dtype=np.float32,
            chunks=(1, INDEX_WIDTH),
        )
        dset.attrs["coordinates"] = "time"
        for t in range(INDEX_TIME):
            dset[t, :] = rng.random(INDEX_WIDTH, dtype=np.float32)
    if not quiet:
        print(f"  finished {path}")


def generate_sorted_axis_index(*, quiet: bool = False) -> None:
    """Generate the sorted-axis index bench HDF5 (gitignored under large/index/)."""
    write_sorted_axis_h5(INDEX_H5, quiet=quiet)

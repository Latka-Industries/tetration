"""Align + group_by bench fixtures: 2-D dual-dataset HDF5 with repeated time labels.

Geometry matches the sorted-axis index tiers (time=64, width carries bytes) so
small/large/extra stay comparable. `data` and `ref` share shape/coords for
`align`; time labels repeat (`LABEL_REPEAT` copies of each distinct id) so
label `group_by` collapses below the index-form group count.
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
from generate.index_sorted import INDEX_TIME, SMALL_INDEX_WIDTH

AlignGroupTier = Literal["small", "large", "extra"]

# Distinct labels along time; INDEX_TIME must be divisible by this.
LABEL_DISTINCT = 8
LABEL_REPEAT = INDEX_TIME // LABEL_DISTINCT  # 8


@dataclass(frozen=True)
class AlignGroupFixture:
    tier: AlignGroupTier
    logical_bytes: int
    width: int
    root: Path
    h5: Path
    tet: Path

    @property
    def logical_gib(self) -> float:
        return self.logical_bytes / (1024**3)

    @property
    def group_count_index(self) -> int:
        """Array-form `group_by` groups (one per time index)."""
        return INDEX_TIME

    @property
    def group_count_labels(self) -> int:
        """Label-form `group_by` groups (collapsed repeated coords)."""
        return LABEL_DISTINCT


def fixture_for(tier: AlignGroupTier) -> AlignGroupFixture:
    """Paths + geometry for `small` / `large` / `extra`.

    `data` and `ref` are the same shape, so width is halved vs a single-array
    tier so the HDF5 payload stays ≈ the named tier size (not 2×).
    """
    if tier == "small":
        # ~256 MiB total across data+ref (each ~128 MiB).
        width = SMALL_INDEX_WIDTH // 2
        root = FIXTURES_ROOT / "large" / "align_group_small"
    elif tier == "large":
        root = FIXTURES_ROOT / "large" / "align_group"
        # Split LARGE_PER_FORMAT_BYTES across the two datasets.
        width = (LARGE_PER_FORMAT_BYTES // 2) // (INDEX_TIME * FLOAT32_BYTES)
    elif tier == "extra":
        root = FIXTURES_ROOT / "extra_large" / "align_group"
        width = (EXTRA_LARGE_LOGICAL_BYTES // 2) // (INDEX_TIME * FLOAT32_BYTES)
    else:
        raise ValueError(f"unknown align_group tier {tier!r}; choose small, large, or extra")
    if width < 1:
        raise ValueError("logical size too small for INDEX_TIME")
    # Per-dataset logical bytes (report / BenchCase use the primary `data` size).
    logical_bytes = INDEX_TIME * width * FLOAT32_BYTES
    return AlignGroupFixture(
        tier=tier,
        logical_bytes=logical_bytes,
        width=width,
        root=root,
        h5=root / "tensor_align_group.h5",
        tet=root / "tensor_align_group.tet",
    )


def _time_labels() -> np.ndarray:
    """Repeated i32 labels: 0×8, 1×8, … → Display strings collapse under label group_by."""
    if INDEX_TIME % LABEL_DISTINCT != 0:
        raise ValueError("INDEX_TIME must be divisible by LABEL_DISTINCT")
    return np.repeat(np.arange(LABEL_DISTINCT, dtype=np.int32), LABEL_REPEAT)


def write_align_group_h5(fx: AlignGroupFixture, *, quiet: bool = False) -> None:
    """Write 2-D f32 HDF5 with `data` + `ref` and CF time coords (repeated labels)."""
    fx.h5.parent.mkdir(parents=True, exist_ok=True)
    if fx.h5.is_file():
        fx.h5.unlink()
    seed_off = {"small": 0, "large": 1, "extra": 2}[fx.tier]
    rng = np.random.default_rng(SEED_LARGE + 594 + seed_off)
    slab = min(fx.width, 16 * 1024 * 1024)
    labels = _time_labels()
    if not quiet:
        print(
            f"HDF5 align/group_by [{fx.tier}]: {fx.h5} "
            f"({fx.logical_gib:.2f} GiB logical, time={INDEX_TIME}, width={fx.width}, "
            f"label_groups={LABEL_DISTINCT}, chunks=(1, {fx.width}))"
        )
    with h5py.File(fx.h5, "w") as f:
        f.attrs["tetration_fixture"] = "align_group"
        f.attrs["tetration_tier"] = fx.tier
        f.attrs["tetration_logical_bytes"] = int(fx.logical_bytes)
        coords = f.create_group("coordinates")
        coords.create_dataset("time", data=labels)
        coords["time"].attrs["units"] = "group_id"
        for name in ("data", "ref"):
            dset = f.create_dataset(
                name,
                shape=(INDEX_TIME, fx.width),
                dtype=np.float32,
                chunks=(1, fx.width),
            )
            dset.attrs["coordinates"] = "time"
        src = f["data"]
        for t in range(INDEX_TIME):
            for x0 in range(0, fx.width, slab):
                x1 = min(x0 + slab, fx.width)
                src[t, x0:x1] = rng.random(x1 - x0, dtype=np.float32)
            if not quiet and (t + 1) % 8 == 0:
                print(f"  wrote data time {t + 1}/{INDEX_TIME}")
        # Byte-identical `ref` so align checks cleanly (metadata only; still pays convert).
        dst = f["ref"]
        for t in range(INDEX_TIME):
            for x0 in range(0, fx.width, slab):
                x1 = min(x0 + slab, fx.width)
                dst[t, x0:x1] = src[t, x0:x1]
            if not quiet and (t + 1) % 8 == 0:
                print(f"  copied ref time {t + 1}/{INDEX_TIME}")
    if not quiet:
        print(f"  finished {fx.h5}")


def generate_align_group(tier: AlignGroupTier = "large", *, quiet: bool = False) -> AlignGroupFixture:
    """Generate (or overwrite) the align/group_by HDF5 for `tier`; return paths."""
    fx = fixture_for(tier)
    write_align_group_h5(fx, quiet=quiet)
    return fx

"""Benchmark case definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark.constants import FormatName, TierName
from generate.constants import (
    BENCH_SMALL_LOGICAL_BYTES,
    EXTRA_LARGE_LOGICAL_BYTES,
    FIXTURES_ROOT,
    LARGE_PER_FORMAT_BYTES,
)


@dataclass(frozen=True)
class BenchCase:
    format: FormatName
    tier: TierName
    src: Path
    tet_out: Path
    gen_target: str
    logical_bytes: int

    @property
    def logical_gib(self) -> str:
        return f"{self.logical_bytes / (1024**3):.2f}"


def _large_basename(format: FormatName) -> str:
    if format == "zarr":
        return "tensor_large"
    return f"tensor_large.{format if format != 'netcdf' else 'nc'}"


def _extra_basename(format: FormatName) -> str:
    if format == "zarr":
        return "tensor_20gb"
    return f"tensor_20gb.{format if format != 'netcdf' else 'nc'}"


def _small_basename(format: FormatName) -> str:
    if format == "zarr":
        return "tensor_small"
    return f"tensor_small.{format if format != 'netcdf' else 'nc'}"


def cases_for(
    format: FormatName,
    *,
    tiers: tuple[TierName, ...] = ("small", "large", "extra"),
) -> tuple[BenchCase, ...]:
    by_tier: dict[TierName, BenchCase] = {
        "small": BenchCase(
            format=format,
            tier="small",
            src=FIXTURES_ROOT / "bench_small" / format / _small_basename(format),
            tet_out=FIXTURES_ROOT / "bench_small" / format / "tensor_small.tet",
            gen_target=f"bench-small-{format}",
            logical_bytes=BENCH_SMALL_LOGICAL_BYTES,
        ),
        "large": BenchCase(
            format=format,
            tier="large",
            src=FIXTURES_ROOT / "large" / format / _large_basename(format),
            tet_out=FIXTURES_ROOT / "large" / format / "tensor_large.tet",
            gen_target=f"large-{format}",
            logical_bytes=LARGE_PER_FORMAT_BYTES,
        ),
        "extra": BenchCase(
            format=format,
            tier="extra",
            src=FIXTURES_ROOT / "extra_large" / format / _extra_basename(format),
            tet_out=FIXTURES_ROOT / "extra_large" / format / "tensor_20gb.tet",
            gen_target=f"extra-large-{format}",
            logical_bytes=EXTRA_LARGE_LOGICAL_BYTES,
        ),
    }
    return tuple(by_tier[t] for t in tiers)

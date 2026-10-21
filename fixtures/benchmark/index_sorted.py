"""THI-613 bench: tet sealed range sum vs HDF5+NumPy on large / extra tiers.

Primary question: is tet’s built-in selection + `sum` accurate and competitive
vs h5py + NumPy on the same slab? Tiers match the main stress sizes (~6.7 GiB
large, 20 GiB extra). Index present/missing is secondary correctness only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from benchmark.log import format_elapsed, log
from benchmark.tet import tet_bin
from generate.constants import FIXTURES_ROOT
from generate.index_sorted import (
    INDEX_SELECT_LEN,
    INDEX_SELECT_START,
    INDEX_SELECT_STOP,
    INDEX_TIME,
    IndexFixture,
    IndexTier,
    fixture_for,
    generate_sorted_axis_index,
)

RESULTS_DIR = FIXTURES_ROOT / "bench_results"
SUM_TOL_REL = 1e-2


@dataclass
class Row:
    backend: str
    seconds: float
    value: float | None
    chunk_count: int | None
    note: str


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "TET_NO_QUERY_HISTORY": "1"}
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def _must_ok(proc: subprocess.CompletedProcess[str], what: str) -> None:
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed:\n{proc.stderr or proc.stdout}")


def query_body() -> str:
    return json.dumps(
        {
            "dataset": "data",
            "layout_version": 1,
            "selection": [
                {"start_label": INDEX_SELECT_START, "stop_label": INDEX_SELECT_STOP},
                {},
            ],
            "sum": [],
            "execution": {"device": "cpu"},
        }
    )


def run_tet_query(tet: Path, pkg: Path) -> tuple[float, dict]:
    cmd = [
        str(tet),
        "query",
        query_body(),
        "-t",
        str(pkg),
        "-x",
        "--preview",
        "0",
        "--format",
        "stats",
        "--device",
        "cpu",
    ]

    def once() -> dict:
        proc = _run(cmd)
        _must_ok(proc, "tet query")
        return json.loads(proc.stdout)

    once()
    t0 = time.perf_counter()
    payload = once()
    return time.perf_counter() - t0, payload


def source_h5_numpy_sum(fx: IndexFixture) -> tuple[float, float]:
    start = int(INDEX_SELECT_START)
    stop = int(INDEX_SELECT_STOP)

    def once() -> float:
        with h5py.File(fx.h5, "r") as f:
            arr = np.asarray(f["data"][start:stop, :], dtype=np.float64)
            return float(arr.sum())

    once()
    t0 = time.perf_counter()
    total = once()
    return time.perf_counter() - t0, total


def build_package(tet: Path, fx: IndexFixture) -> None:
    if fx.pkg.exists():
        shutil.rmtree(fx.pkg)
    if fx.tet.exists():
        fx.tet.unlink()
    log(f"convert → {fx.tet.name} ({fx.logical_gib:.2f} GiB)")
    t0 = time.perf_counter()
    proc = _run([str(tet), "convert", str(fx.h5), str(fx.tet), "--jobs", "0"])
    _must_ok(proc, "tet convert")
    log(f"convert done in {format_elapsed(time.perf_counter() - t0)}")

    log(f"ingest + publish → {fx.pkg.name}")
    t0 = time.perf_counter()
    proc = _run([str(tet), "ingest", str(fx.tet), "--out", str(fx.pkg), "--jobs", "1"])
    _must_ok(proc, "tet ingest")
    proc = _run([str(tet), "publish", str(fx.pkg)])
    _must_ok(proc, "tet publish")
    log(f"publish done in {format_elapsed(time.perf_counter() - t0)}")


def index_path(pkg: Path) -> Path:
    gen = pkg / "generations"
    dirs = sorted(p for p in gen.iterdir() if p.is_dir()) if gen.is_dir() else []
    if not dirs:
        raise RuntimeError(f"no generations under {pkg}")
    return dirs[0] / "INDEX.json"


def values_match(a: float, b: float) -> bool:
    scale = max(abs(a), abs(b), 1e-9)
    return abs(a - b) <= SUM_TOL_REL * scale


def results_path(tier: IndexTier) -> Path:
    return RESULTS_DIR / f"index_sorted_axis_{tier}.md"


def write_report(
    fx: IndexFixture,
    *,
    git_rev: str,
    primary: list[Row],
    secondary: list[Row],
    accurate: bool,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = results_path(fx.tier)
    lines = [
        f"# Sorted-axis range query bench (THI-613) — {fx.tier}",
        "",
        "**Question:** is tet’s built-in range selection + `sum` accurate and "
        "competitive vs loading the same HDF5 slab with NumPy?",
        "",
        f"- git: `{git_rev}`",
        f"- tier: **{fx.tier}** ({fx.logical_gib:.2f} GiB logical; "
        f"select ≈ {fx.select_gib:.2f} GiB = {INDEX_SELECT_LEN}/{INDEX_TIME} time steps)",
        f"- shape: time={INDEX_TIME} × width={fx.width}, chunk (1, {fx.width})",
        f"- op: `sum` over time labels `[{INDEX_SELECT_START}, {INDEX_SELECT_STOP})` × full x",
        f"- accuracy (tet vs h5+numpy): **{'pass' if accurate else 'FAIL'}**",
        "",
        "## Primary: tet vs HDF5 + NumPy",
        "",
        "| backend | seconds | sum | chunks | note |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in primary:
        lines.append(
            f"| {row.backend} | {row.seconds:.4f} | "
            f"{'-' if row.value is None else f'{row.value:.6g}'} | "
            f"{'-' if row.chunk_count is None else row.chunk_count} | "
            f"{row.note} |"
        )
    lines.extend(
        [
            "",
            "## Secondary: index present vs missing (same tet query)",
            "",
            "Answers must match. Not the product speed claim.",
            "",
            "| backend | seconds | sum | chunks | note |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in secondary:
        lines.append(
            f"| {row.backend} | {row.seconds:.4f} | "
            f"{'-' if row.value is None else f'{row.value:.6g}'} | "
            f"{'-' if row.chunk_count is None else row.chunk_count} | "
            f"{row.note} |"
        )
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {out}")


def run_index_bench(tier: IndexTier = "large", *, regenerate: bool = True) -> int:
    tet = tet_bin()
    fx = fixture_for(tier)
    log(
        f"=== THI-613: tet range sum vs HDF5+NumPy [{tier}] "
        f"({fx.logical_gib:.2f} GiB, select≈{fx.select_gib:.2f} GiB) ==="
    )
    if regenerate or not fx.h5.is_file():
        generate_sorted_axis_index(tier, quiet=False)
    else:
        log(f"reusing existing {fx.h5}")

    build_package(tet, fx)

    idx = index_path(fx.pkg)
    if not idx.is_file():
        raise RuntimeError(f"expected INDEX.json at {idx} after publish")
    doc = json.loads(idx.read_text(encoding="utf-8"))
    kinds = [e.get("kind") for e in doc.get("indexes", [])]
    if "sorted_axis_range" not in kinds:
        raise RuntimeError(f"INDEX.json missing sorted_axis_range; kinds={kinds}")

    log("tet: sealed package, selection + sum")
    tet_s, payload = run_tet_query(tet, fx.pkg)
    plan = payload.get("read_plan") or {}
    exe = payload.get("execution") or {}
    tet_sum = exe.get("operation_sum")
    tet_chunks = plan.get("chunk_count")
    primary = [
        Row(
            backend="tet",
            seconds=tet_s,
            value=tet_sum,
            chunk_count=tet_chunks,
            note="query on package (sorted_axis_range sealed)",
        )
    ]

    log("source: HDF5 + NumPy same slab sum")
    h5_s, h5_sum = source_h5_numpy_sum(fx)
    primary.append(
        Row(
            backend="hdf5+numpy",
            seconds=h5_s,
            value=h5_sum,
            chunk_count=INDEX_SELECT_LEN,
            note="h5py read + np.sum (fair peer)",
        )
    )

    backup = idx.with_suffix(".json.bak")
    shutil.move(idx, backup)
    log("tet: same query with INDEX.json removed (fallback)")
    miss_s, miss_payload = run_tet_query(tet, fx.pkg)
    shutil.move(backup, idx)
    miss_exe = miss_payload.get("execution") or {}
    miss_plan = miss_payload.get("read_plan") or {}
    miss_sum = miss_exe.get("operation_sum")
    secondary = [
        Row(
            backend="tet (index present)",
            seconds=tet_s,
            value=tet_sum,
            chunk_count=tet_chunks,
            note="from primary row",
        ),
        Row(
            backend="tet (index missing)",
            seconds=miss_s,
            value=miss_sum,
            chunk_count=miss_plan.get("chunk_count"),
            note="geometry fallback; answer must match",
        ),
    ]

    if tet_sum is None or miss_sum is None:
        log("error: missing sum values")
        return 1

    accurate = values_match(float(tet_sum), h5_sum)
    if not accurate:
        log(f"error: tet vs hdf5+numpy sum mismatch ({tet_sum} vs {h5_sum})")
    if not values_match(float(tet_sum), float(miss_sum)):
        log(f"error: index present vs missing sum mismatch ({tet_sum} vs {miss_sum})")
        accurate = False

    try:
        git_rev = subprocess.check_output(
            ["git", "-C", str(FIXTURES_ROOT.parent), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_rev = "unknown"

    write_report(
        fx,
        git_rev=git_rev,
        primary=primary,
        secondary=secondary,
        accurate=accurate,
    )
    for row in primary + secondary:
        log(
            f"{row.backend}: {format_elapsed(row.seconds)}  sum={row.value}  "
            f"chunks={row.chunk_count}"
        )
    if accurate and primary[0].seconds > 0:
        ratio = primary[1].seconds / primary[0].seconds
        log(
            f"primary [{tier}]: tet {primary[0].seconds:.4f}s vs hdf5+numpy "
            f"{primary[1].seconds:.4f}s (numpy/tet = {ratio:.2f}x)"
        )
    return 0 if accurate else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="THI-613: tet range sum vs HDF5+NumPy (large ≈6.7 GiB, extra 20 GiB)",
    )
    parser.add_argument(
        "--tier",
        choices=("large", "extra"),
        default="large",
        help="fixture size: large ≈6.7 GiB (default), extra = 20 GiB",
    )
    parser.add_argument(
        "--reuse-h5",
        action="store_true",
        help="skip regenerating HDF5 when the file already exists",
    )
    args = parser.parse_args(argv)
    try:
        return run_index_bench(args.tier, regenerate=not args.reuse_h5)
    except Exception as exc:  # noqa: BLE001
        print(f"bench-index failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

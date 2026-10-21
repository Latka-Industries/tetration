"""THI-613 bench: tet sealed query vs HDF5+NumPy on the same time-range sum.

Primary question: is the built-in tet op accurate and performant against the
fair source peer (h5py slab + NumPy sum)? Index present/missing is a secondary
correctness check only — not the headline comparison.
"""

from __future__ import annotations

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
    INDEX_H5,
    INDEX_PKG,
    INDEX_SELECT_LEN,
    INDEX_SELECT_START,
    INDEX_SELECT_STOP,
    INDEX_TET,
    INDEX_TIME,
    INDEX_WIDTH,
    generate_sorted_axis_index,
)

RESULTS_DIR = FIXTURES_ROOT / "bench_results"
INDEX_RESULTS = RESULTS_DIR / "index_sorted_axis.md"

# Same relative tolerance style as fixtures/benchmark/constants.py sum ops.
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
    """Closed wire: label range on time + full x, streaming sum."""
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

    once()  # warm
    t0 = time.perf_counter()
    payload = once()
    return time.perf_counter() - t0, payload


def source_h5_numpy_sum() -> tuple[float, float]:
    """Fair peer: open HDF5, read the same half-open time slab, NumPy sum."""
    start = int(INDEX_SELECT_START)
    stop = int(INDEX_SELECT_STOP)

    def once() -> float:
        with h5py.File(INDEX_H5, "r") as f:
            arr = np.asarray(f["data"][start:stop, :], dtype=np.float64)
            return float(arr.sum())

    once()  # warm (open + first read)
    t0 = time.perf_counter()
    total = once()
    return time.perf_counter() - t0, total


def build_package(tet: Path) -> None:
    if INDEX_PKG.exists():
        shutil.rmtree(INDEX_PKG)
    if INDEX_TET.exists():
        INDEX_TET.unlink()
    log(f"convert → {INDEX_TET.name}")
    t0 = time.perf_counter()
    proc = _run([str(tet), "convert", str(INDEX_H5), str(INDEX_TET), "--jobs", "0"])
    _must_ok(proc, "tet convert")
    log(f"convert done in {format_elapsed(time.perf_counter() - t0)}")

    log(f"ingest + publish → {INDEX_PKG.name}")
    proc = _run(
        [str(tet), "ingest", str(INDEX_TET), "--out", str(INDEX_PKG), "--jobs", "1"]
    )
    _must_ok(proc, "tet ingest")
    proc = _run([str(tet), "publish", str(INDEX_PKG)])
    _must_ok(proc, "tet publish")


def index_path(pkg: Path) -> Path:
    gen = pkg / "generations"
    dirs = sorted(p for p in gen.iterdir() if p.is_dir()) if gen.is_dir() else []
    if not dirs:
        raise RuntimeError(f"no generations under {pkg}")
    return dirs[0] / "INDEX.json"


def values_match(a: float, b: float) -> bool:
    scale = max(abs(a), abs(b), 1e-9)
    return abs(a - b) <= SUM_TOL_REL * scale


def write_report(
    *,
    git_rev: str,
    primary: list[Row],
    secondary: list[Row],
    accurate: bool,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sorted-axis range query bench (THI-613)",
        "",
        "**Question:** is tet’s built-in range selection + `sum` accurate and "
        "competitive vs loading the same HDF5 slab with NumPy?",
        "",
        f"- git: `{git_rev}`",
        f"- fixture: time={INDEX_TIME} × width={INDEX_WIDTH} f32 "
        f"(~{INDEX_TIME * INDEX_WIDTH * 4 / (1024**2):.0f} MiB), "
        f"chunk (1, {INDEX_WIDTH})",
        f"- op: `sum` over time labels `[{INDEX_SELECT_START}, {INDEX_SELECT_STOP})` "
        f"× full x ({INDEX_SELECT_LEN} time chunks)",
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
            "Answers must match. Timing here is not the product claim.",
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
    INDEX_RESULTS.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {INDEX_RESULTS}")


def run_index_bench() -> int:
    tet = tet_bin()
    log("=== THI-613: tet range sum vs HDF5+NumPy ===")
    generate_sorted_axis_index(quiet=False)
    build_package(tet)

    idx = index_path(INDEX_PKG)
    if not idx.is_file():
        raise RuntimeError(f"expected INDEX.json at {idx} after publish")
    doc = json.loads(idx.read_text(encoding="utf-8"))
    kinds = [e.get("kind") for e in doc.get("indexes", [])]
    if "sorted_axis_range" not in kinds:
        raise RuntimeError(f"INDEX.json missing sorted_axis_range; kinds={kinds}")

    log("tet: sealed package, selection + sum (index present)")
    tet_s, payload = run_tet_query(tet, INDEX_PKG)
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
    h5_s, h5_sum = source_h5_numpy_sum()
    primary.append(
        Row(
            backend="hdf5+numpy",
            seconds=h5_s,
            value=h5_sum,
            chunk_count=INDEX_SELECT_LEN,
            note="h5py read + np.sum (fair peer)",
        )
    )

    # Secondary: drop INDEX.json; answers must still match.
    backup = idx.with_suffix(".json.bak")
    shutil.move(idx, backup)
    log("tet: same query with INDEX.json removed (fallback)")
    miss_s, miss_payload = run_tet_query(tet, INDEX_PKG)
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

    if tet_sum is None or h5_sum is None or miss_sum is None:
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
            f"primary: tet {primary[0].seconds:.4f}s vs hdf5+numpy "
            f"{primary[1].seconds:.4f}s (numpy/tet = {ratio:.2f}x)"
        )
    return 0 if accurate else 1


def main(argv: list[str] | None = None) -> int:
    _ = argv
    try:
        return run_index_bench()
    except Exception as exc:  # noqa: BLE001
        print(f"bench-index failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

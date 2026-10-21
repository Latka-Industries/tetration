"""Skip bench for sorted_axis_range (THI-613): present vs missing vs HDF5 peer."""

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


@dataclass
class Row:
    label: str
    seconds: float
    value: float | None
    chunk_count: int | None
    index_skipped: int | None
    note: str | None = None


def _run(cmd: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "TET_NO_QUERY_HISTORY": "1"}
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def _must_ok(proc: subprocess.CompletedProcess[str], what: str) -> None:
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed:\n{proc.stderr or proc.stdout}")


def query_body(*, with_selection: bool) -> str:
    doc: dict = {"dataset": "data", "layout_version": 1, "sum": []}
    if with_selection:
        doc["selection"] = [
            {"start_label": INDEX_SELECT_START, "stop_label": INDEX_SELECT_STOP},
            {},
        ]
    doc["execution"] = {"device": "cpu"}
    return json.dumps(doc)


def run_package_query(tet: Path, pkg: Path, *, with_selection: bool) -> tuple[float, dict]:
    body = query_body(with_selection=with_selection)
    cmd = [
        str(tet),
        "query",
        body,
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


def source_sum_slice() -> tuple[float, float]:
    start = int(INDEX_SELECT_START)
    stop = int(INDEX_SELECT_STOP)
    t0 = time.perf_counter()
    with h5py.File(INDEX_H5, "r") as f:
        arr = np.asarray(f["data"][start:stop, :], dtype=np.float64)
        total = float(arr.sum())
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

    log(f"ingest → {INDEX_PKG.name}")
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


def write_report(rows: list[Row], *, git_rev: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sorted-axis range index bench (THI-613)",
        "",
        f"- git: `{git_rev}`",
        f"- fixture: time={INDEX_TIME} × width={INDEX_WIDTH} f32 "
        f"(~{INDEX_TIME * INDEX_WIDTH * 4 / (1024**2):.0f} MiB), "
        f"chunk (1, {INDEX_WIDTH})",
        f"- selection: time labels `[{INDEX_SELECT_START}, {INDEX_SELECT_STOP})` "
        f"→ {INDEX_SELECT_LEN} chunks",
        "",
        "| case | seconds | sum | chunks | index_skipped | note |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.label} | {row.seconds:.4f} | "
            f"{'-' if row.value is None else f'{row.value:.6g}'} | "
            f"{'-' if row.chunk_count is None else row.chunk_count} | "
            f"{'-' if row.index_skipped is None else row.index_skipped} | "
            f"{row.note or ''} |"
        )
    lines.append("")
    INDEX_RESULTS.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {INDEX_RESULTS}")


def run_index_bench() -> int:
    tet = tet_bin()
    log("=== sorted_axis_range skip bench (THI-613) ===")
    generate_sorted_axis_index(quiet=False)
    build_package(tet)

    idx = index_path(INDEX_PKG)
    if not idx.is_file():
        raise RuntimeError(f"expected INDEX.json at {idx} after publish")

    # Confirm sorted_axis_range is in the sealed index.
    doc = json.loads(idx.read_text(encoding="utf-8"))
    kinds = [e.get("kind") for e in doc.get("indexes", [])]
    if "sorted_axis_range" not in kinds:
        raise RuntimeError(f"INDEX.json missing sorted_axis_range; kinds={kinds}")

    rows: list[Row] = []

    log("query: index present + narrow selection")
    sec, payload = run_package_query(tet, INDEX_PKG, with_selection=True)
    plan = payload.get("read_plan") or {}
    exe = payload.get("execution") or {}
    rows.append(
        Row(
            label="tet index present (select)",
            seconds=sec,
            value=exe.get("operation_sum"),
            chunk_count=plan.get("chunk_count"),
            index_skipped=plan.get("index_filter_skipped_chunks", 0),
            note="sorted_axis_range usable",
        )
    )

    backup = idx.with_suffix(".json.bak")
    shutil.move(idx, backup)
    log("query: index missing + narrow selection")
    sec, payload = run_package_query(tet, INDEX_PKG, with_selection=True)
    plan = payload.get("read_plan") or {}
    exe = payload.get("execution") or {}
    rows.append(
        Row(
            label="tet index missing (select)",
            seconds=sec,
            value=exe.get("operation_sum"),
            chunk_count=plan.get("chunk_count"),
            index_skipped=plan.get("index_filter_skipped_chunks", 0),
            note="geometry only",
        )
    )
    shutil.move(backup, idx)

    log("query: full tensor (I/O baseline)")
    sec, payload = run_package_query(tet, INDEX_PKG, with_selection=False)
    plan = payload.get("read_plan") or {}
    exe = payload.get("execution") or {}
    rows.append(
        Row(
            label="tet full scan",
            seconds=sec,
            value=exe.get("operation_sum"),
            chunk_count=plan.get("chunk_count"),
            index_skipped=plan.get("index_filter_skipped_chunks", 0),
            note=f"all {INDEX_TIME} time chunks",
        )
    )

    log("source: HDF5 slice sum")
    sec, value = source_sum_slice()
    rows.append(
        Row(
            label="hdf5 peer slice",
            seconds=sec,
            value=value,
            chunk_count=INDEX_SELECT_LEN,
            index_skipped=None,
            note="h5py slab",
        )
    )

    present = rows[0].value
    missing = rows[1].value
    peer = rows[3].value
    if present is None or missing is None or peer is None:
        log("error: missing sum values")
        return 1
    if abs(present - missing) > 1e-3 * max(abs(present), 1.0):
        log(f"error: present vs missing sum mismatch ({present} vs {missing})")
        return 1
    if abs(present - peer) > 1e-2 * max(abs(present), 1.0):
        log(f"warning: tet vs hdf5 sum |Δ| large ({present} vs {peer})")

    try:
        git_rev = subprocess.check_output(
            ["git", "-C", str(FIXTURES_ROOT.parent), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_rev = "unknown"

    write_report(rows, git_rev=git_rev)
    for row in rows:
        log(
            f"{row.label}: {format_elapsed(row.seconds)}  sum={row.value}  "
            f"chunks={row.chunk_count}  index_skipped={row.index_skipped}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    _ = argv
    try:
        return run_index_bench()
    except Exception as exc:  # noqa: BLE001
        print(f"bench-index failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

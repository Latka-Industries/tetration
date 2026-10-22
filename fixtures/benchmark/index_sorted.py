"""THI-613 / THI-614 bench: tet sealed sum vs HDF5+NumPy on small / large / extra.

Primary question: is tet’s built-in selection + `sum` accurate and competitive
vs h5py + NumPy on the same slab?

Also runs a thin `label_filter` eq case (THI-614) on the same fixture — present vs
missing index — without a second HDF5 harness.

- **small** — select ≈16 MiB (fixed-cost / CLI microbench)
- **large** / **extra** — stress sizes (CLI spawn is noise)

Times the release `tet` binary only — not tet-py (bindings lag the crate).
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
from typing import Any

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


def query_doc(*, fold_parallel: bool | None = None) -> dict[str, Any]:
    execution: dict[str, Any] = {"device": "cpu"}
    if fold_parallel is not None:
        execution["fold_parallel"] = fold_parallel
    return {
        "dataset": "data",
        "layout_version": 1,
        "selection": [
            {"start_label": INDEX_SELECT_START, "stop_label": INDEX_SELECT_STOP},
            {},
        ],
        "sum": [],
        "execution": execution,
    }


# Single time-label equality (THI-614); one chunk of the 64-step axis.
LABEL_EQ_VALUE = INDEX_SELECT_START


def label_eq_query_doc(*, fold_parallel: bool | None = None) -> dict[str, Any]:
    execution: dict[str, Any] = {"device": "cpu"}
    if fold_parallel is not None:
        execution["fold_parallel"] = fold_parallel
    return {
        "dataset": "data",
        "layout_version": 1,
        # Decimal axis: CF imports may omit dim_names; coords still resolve by extent.
        "label_filter": {"axis": "0", "eq": LABEL_EQ_VALUE},
        "sum": [],
        "execution": execution,
    }


def query_body(*, fold_parallel: bool | None = None) -> str:
    return json.dumps(query_doc(fold_parallel=fold_parallel))


def run_tet_query(
    tet: Path,
    pkg: Path,
    body: str | None = None,
    *,
    warm: bool = True,
) -> tuple[float, dict]:
    cmd = [
        str(tet),
        "query",
        body if body is not None else query_body(),
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

    if warm:
        once()
    t0 = time.perf_counter()
    payload = once()
    return time.perf_counter() - t0, payload


def warm_package(tet: Path, pkg: Path) -> None:
    """Pull sealed bytes into page cache before timed runs (CLI spawn still paid)."""
    log("warm: two throwaway range sums (page-cache packed.tet)")
    body = query_body(fold_parallel=False)
    run_tet_query(tet, pkg, body, warm=False)
    run_tet_query(tet, pkg, body, warm=False)


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
    label_eq: list[Row],
    accurate: bool,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = results_path(fx.tier)
    select_note = (
        f"select ≈ {fx.select_mib:.1f} MiB"
        if fx.tier == "small"
        else f"select ≈ {fx.select_gib:.2f} GiB"
    )
    tet_st = next((r for r in primary if r.backend == "tet ST"), None)
    tet_par = next((r for r in primary if r.backend == "tet parallel"), None)
    h5 = next((r for r in primary if r.backend == "hdf5+numpy"), None)
    lines = [
        f"# Sorted-axis range query bench (THI-613) — {fx.tier}",
        "",
        "**Question:** is tet’s built-in range selection + `sum` accurate and "
        "competitive vs loading the same HDF5 slab with NumPy?",
        "",
        f"- git: `{git_rev}`",
        f"- tier: **{fx.tier}** ({fx.logical_gib:.2f} GiB logical; "
        f"{select_note} = {INDEX_SELECT_LEN}/{INDEX_TIME} time steps)",
        f"- shape: time={INDEX_TIME} × width={fx.width}, chunk (1, {fx.width})",
        f"- op: `sum` over time labels `[{INDEX_SELECT_START}, {INDEX_SELECT_STOP})` × full x",
        f"- accuracy (tet vs h5+numpy): **{'pass' if accurate else 'FAIL'}**",
        "- method: warm page-cache, then time; tet ST = `fold_parallel:false`, "
        "tet parallel = `fold_parallel:true`; hdf5+numpy is in-process",
        "",
        "## Plain English",
        "",
    ]
    if tet_st and tet_par and h5:
        st_vs_h5 = h5.seconds / tet_st.seconds if tet_st.seconds > 0 else float("inf")
        par_vs_h5 = h5.seconds / tet_par.seconds if tet_par.seconds > 0 else float("inf")
        par_vs_st = tet_st.seconds / tet_par.seconds if tet_par.seconds > 0 else float("inf")
        lines.extend(
            [
                f"- Select size ≈ **{select_note}** of a **{fx.logical_gib:.2f} GiB** file.",
                f"- Single-thread tet: **{tet_st.seconds:.4f}s**. "
                f"Parallel tet: **{tet_par.seconds:.4f}s** "
                f"(parallel is {par_vs_st:.2f}× vs ST).",
                f"- HDF5+NumPy (in-process): **{h5.seconds:.4f}s**.",
                f"- vs NumPy: ST is {st_vs_h5:.2f}×, parallel is {par_vs_h5:.2f}× "
                f"(>1 means tet faster).",
                "",
            ]
        )
    lines.extend(
        [
            "## Primary: tet ST / parallel vs HDF5 + NumPy",
            "",
            "| backend | seconds | sum | chunks | note |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
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
    lines.extend(
        [
            "",
            f"## Label-eq (`label_filter` eq={LABEL_EQ_VALUE!r}, THI-614)",
            "",
            "Same package; present vs missing index. Answers must match.",
            "",
            "| backend | seconds | sum | chunks | note |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in label_eq:
        lines.append(
            f"| {row.backend} | {row.seconds:.4f} | "
            f"{'-' if row.value is None else f'{row.value:.6g}'} | "
            f"{'-' if row.chunk_count is None else row.chunk_count} | "
            f"{row.note} |"
        )
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {out}")


def run_index_bench(tier: IndexTier, *, regenerate: bool = True) -> int:
    tet = tet_bin()
    fx = fixture_for(tier)
    select_note = (
        f"select≈{fx.select_mib:.1f} MiB"
        if tier == "small"
        else f"select≈{fx.select_gib:.2f} GiB"
    )
    log(
        f"=== THI-613: tet range sum vs HDF5+NumPy [{tier}] "
        f"({fx.logical_gib:.2f} GiB, {select_note}) ==="
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
    if "coord_label_postings" not in kinds:
        raise RuntimeError(f"INDEX.json missing coord_label_postings; kinds={kinds}")

    warm_package(tet, fx.pkg)

    log("tet ST: fold_parallel=false, range sum")
    st_body = query_body(fold_parallel=False)
    tet_st_s, st_payload = run_tet_query(tet, fx.pkg, st_body, warm=False)
    st_plan = st_payload.get("read_plan") or {}
    st_exe = st_payload.get("execution") or {}
    tet_sum = st_exe.get("operation_sum")
    tet_chunks = st_plan.get("chunk_count")

    log("tet parallel: fold_parallel=true, range sum")
    par_body = query_body(fold_parallel=True)
    tet_par_s, par_payload = run_tet_query(tet, fx.pkg, par_body, warm=False)
    par_exe = par_payload.get("execution") or {}
    par_sum = par_exe.get("operation_sum")
    par_chunks = (par_payload.get("read_plan") or {}).get("chunk_count")

    log("source: HDF5 + NumPy same slab sum")
    h5_s, h5_sum = source_h5_numpy_sum(fx)
    primary = [
        Row(
            backend="tet ST",
            seconds=tet_st_s,
            value=tet_sum,
            chunk_count=tet_chunks,
            note=f"CLI; fold_parallel={st_exe.get('fold_parallel')}",
        ),
        Row(
            backend="tet parallel",
            seconds=tet_par_s,
            value=par_sum,
            chunk_count=par_chunks,
            note=f"CLI; fold_parallel={par_exe.get('fold_parallel')}",
        ),
        Row(
            backend="hdf5+numpy",
            seconds=h5_s,
            value=h5_sum,
            chunk_count=INDEX_SELECT_LEN,
            note="h5py read + np.sum (in-process)",
        ),
    ]

    backup = idx.with_suffix(".json.bak")
    shutil.move(idx, backup)
    log("tet: same query with INDEX.json removed (fallback, ST)")
    miss_s, miss_payload = run_tet_query(tet, fx.pkg, st_body, warm=False)
    shutil.move(backup, idx)
    miss_exe = miss_payload.get("execution") or {}
    miss_plan = miss_payload.get("read_plan") or {}
    miss_sum = miss_exe.get("operation_sum")
    secondary = [
        Row(
            backend="tet ST (index present)",
            seconds=tet_st_s,
            value=tet_sum,
            chunk_count=tet_chunks,
            note="from primary ST row",
        ),
        Row(
            backend="tet ST (index missing)",
            seconds=miss_s,
            value=miss_sum,
            chunk_count=miss_plan.get("chunk_count"),
            note="geometry fallback; answer must match",
        ),
    ]

    if tet_sum is None or miss_sum is None or par_sum is None:
        log("error: missing sum values")
        return 1

    accurate = values_match(float(tet_sum), h5_sum)
    if not accurate:
        log(f"error: tet vs hdf5+numpy sum mismatch ({tet_sum} vs {h5_sum})")
    if not values_match(float(tet_sum), float(miss_sum)):
        log(f"error: index present vs missing sum mismatch ({tet_sum} vs {miss_sum})")
        accurate = False
    if not values_match(float(tet_sum), float(par_sum)):
        log(f"error: tet ST vs parallel sum mismatch ({tet_sum} vs {par_sum})")
        accurate = False

    label_body = json.dumps(label_eq_query_doc(fold_parallel=False))
    log(f"tet: label_filter eq={LABEL_EQ_VALUE!r} (THI-614)")
    le_s, le_payload = run_tet_query(tet, fx.pkg, label_body, warm=False)
    le_exe = le_payload.get("execution") or {}
    le_plan = le_payload.get("read_plan") or {}
    le_sum = le_exe.get("operation_sum")
    le_skip = le_plan.get("index_filter_skipped_chunks")
    backup = idx.with_suffix(".json.bak")
    shutil.move(idx, backup)
    log("tet: label_filter with INDEX.json removed")
    le_miss_s, le_miss_payload = run_tet_query(tet, fx.pkg, label_body, warm=False)
    shutil.move(backup, idx)
    le_miss_exe = le_miss_payload.get("execution") or {}
    le_miss_plan = le_miss_payload.get("read_plan") or {}
    le_miss_sum = le_miss_exe.get("operation_sum")
    label_eq_rows = [
        Row(
            backend="tet label_eq (index present)",
            seconds=le_s,
            value=le_sum,
            chunk_count=le_plan.get("chunk_count"),
            note=f"index_filter_skipped={le_skip}",
        ),
        Row(
            backend="tet label_eq (index missing)",
            seconds=le_miss_s,
            value=le_miss_sum,
            chunk_count=le_miss_plan.get("chunk_count"),
            note="footer scan; answer must match",
        ),
    ]
    if le_sum is None or le_miss_sum is None:
        log("error: missing label_eq sum values")
        return 1
    if not values_match(float(le_sum), float(le_miss_sum)):
        log(f"error: label_eq present vs missing sum mismatch ({le_sum} vs {le_miss_sum})")
        accurate = False
    # Pushdown gate: unique time label → one chunk when INDEX loads via package path.
    le_chunks = le_plan.get("chunk_count")
    le_miss_chunks = le_miss_plan.get("chunk_count")
    if le_skip is None or int(le_skip) < 1:
        log(f"error: label_eq index present did not skip chunks (skipped={le_skip})")
        accurate = False
    if le_chunks is None or int(le_chunks) != 1:
        log(f"error: label_eq index present expected chunk_count=1, got {le_chunks}")
        accurate = False
    if le_miss_chunks is None or int(le_miss_chunks) < 2:
        log(
            f"error: label_eq index missing expected full scan chunk_count>=2, "
            f"got {le_miss_chunks}"
        )
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
        label_eq=label_eq_rows,
        accurate=accurate,
    )
    for row in primary + secondary + label_eq_rows:
        log(
            f"{row.backend}: {format_elapsed(row.seconds)}  sum={row.value}  "
            f"chunks={row.chunk_count}"
        )
    if accurate and len(primary) >= 3 and primary[0].seconds > 0 and primary[1].seconds > 0:
        st, par, h5 = primary[0], primary[1], primary[2]
        log(
            f"primary [{tier}]: ST {st.seconds:.4f}s | parallel {par.seconds:.4f}s | "
            f"hdf5+numpy {h5.seconds:.4f}s "
            f"(numpy/ST={h5.seconds / st.seconds:.2f}x, "
            f"numpy/par={h5.seconds / par.seconds:.2f}x)"
        )
    return 0 if accurate else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "THI-613: tet range sum vs HDF5+NumPy "
            "(small ≈16 MiB select, large ≈0.4 GiB, extra ≈1.25 GiB)"
        ),
    )
    parser.add_argument(
        "--tier",
        choices=("small", "large", "extra", "both", "all"),
        default="both",
        help="fixture size; 'both'=small+large, 'all'=small+large+extra",
    )
    parser.add_argument(
        "--reuse-h5",
        action="store_true",
        help="skip regenerating HDF5 when the file already exists",
    )
    args = parser.parse_args(argv)
    tiers: tuple[IndexTier, ...]
    if args.tier == "both":
        tiers = ("small", "large")
    elif args.tier == "all":
        tiers = ("small", "large", "extra")
    else:
        tiers = (args.tier,)  # type: ignore[assignment]
    try:
        rc = 0
        for tier in tiers:
            rc = run_index_bench(tier, regenerate=not args.reuse_h5) or rc
        return rc
    except Exception as exc:  # noqa: BLE001
        print(f"bench-index failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

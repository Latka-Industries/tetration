"""`.tet` convert and query helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from benchmark.case import BenchCase
from benchmark.constants import DEFAULT_TET_DEVICE, OpName, REPO_ROOT, is_align_group_op, op_meta
from benchmark.log import format_elapsed, log


def tet_bin() -> Path:
    path = Path(os.environ.get("TET_BIN", REPO_ROOT / "target/release/tet"))
    if not path.is_file() or not os.access(path, os.X_OK):
        print(f"missing release binary: {path} (run: cargo build --release)", file=sys.stderr)
        sys.exit(1)
    return path


def git_short_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def query_json(
    op: OpName,
    *,
    device: str | None,
    fold_parallel: bool | None = None,
) -> str:
    """Build closed query JSON for `op` (scalar / transform+write / window fields)."""
    meta = op_meta(op)
    doc: dict = {
        "dataset": "data",
        "layout_version": 1,
    }
    if meta.query_fields:
        doc.update(meta.query_fields)
    elif meta.is_transform:
        doc["transform"] = {"method": meta.transform_method}
        # switch: RAM when budget allows, else spill (needed for multi-GiB tensors).
        doc["write"] = "switch"
    else:
        doc[op] = []
    execution: dict = {}
    if device:
        execution["device"] = device
    if fold_parallel is not None:
        execution["fold_parallel"] = fold_parallel
    if execution:
        doc["execution"] = execution
    return json.dumps(doc)


def _device_label(execution: dict) -> str:
    used = execution.get("device_used") or "-"
    fb = execution.get("device_fallback_reason")
    if fb:
        return f"{used} ({fb})"
    req = execution.get("device_requested")
    if req and req != used:
        return f"{used} (req {req})"
    return str(used)


def extract_tet_value(op: OpName, execution: dict) -> float | None:
    """Pull the comparable scalar from tet execution JSON using `OpMeta.value_mode`."""
    meta = op_meta(op)
    if meta.value_mode == "window_count":
        raw = execution.get("window_count")
        return None if raw is None else float(raw)
    if meta.value_mode == "group_count":
        raw = execution.get("group_count")
        return None if raw is None else float(raw)
    if meta.value_mode == "aligned":
        alignment = execution.get("alignment")
        if not isinstance(alignment, dict):
            return None
        return 1.0 if alignment.get("aligned") else 0.0
    if meta.value_mode == "reduced_sum_total":
        raw = execution.get("operation_reduced_sum")
        if raw is None:
            return None
        if isinstance(raw, list):
            return float(sum(float(x) for x in raw))
        return float(raw)
    raw = execution.get(meta.result_key)
    if raw is None:
        return None
    if meta.is_bool:
        return 1.0 if raw else 0.0
    return float(raw)


@dataclass(frozen=True)
class TetOpTiming:
    seconds: float
    value: float | None
    chunks: int | None
    device: str | None
    fold_parallel: bool | None
    io_regime: str | None
    fold_workers: int | None
    memory_strategy: str | None


def _run_query_once(
    tet: Path,
    tet_path: Path,
    body: str,
    *,
    device: str | None,
) -> dict:
    env = {**os.environ, "TET_NO_QUERY_HISTORY": "1"}
    cmd = [
        str(tet),
        "query",
        "--tet",
        str(tet_path),
        "--execute",
        "--preview",
        "0",
        "--format",
        "stats",
    ]
    if device:
        cmd.extend(["--device", device])
    proc = subprocess.run(
        cmd,
        input=body,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "tet query failed")
    return json.loads(proc.stdout)


def run_tet_op(
    tet: Path,
    tet_path: Path,
    case: BenchCase,
    op: OpName,
    *,
    device: str | None = None,
    fold_parallel: bool | None = None,
    warm_passes: int = 1,
) -> TetOpTiming:
    """Time one `tet query` after optional warm passes. Returns fold-path metadata."""
    tet_device = device if device is not None else os.environ.get(
        "TET_BENCH_DEVICE", DEFAULT_TET_DEVICE
    )
    body = query_json(op, device=tet_device, fold_parallel=fold_parallel)
    meta = op_meta(op)
    label = meta.transform_method or op
    par_tag = (
        ""
        if fold_parallel is None
        else (" parallel" if fold_parallel else " ST")
    )

    for i in range(max(0, warm_passes)):
        log(f".tet {label}{par_tag}: warm {i + 1}/{warm_passes} over ~{case.logical_gib} GiB …")
        try:
            _run_query_once(tet, tet_path, body, device=tet_device)
        except RuntimeError:
            pass

    log(f".tet {label}{par_tag}: timed pass …")
    t0 = time.perf_counter()
    payload = _run_query_once(tet, tet_path, body, device=tet_device)
    elapsed = time.perf_counter() - t0

    execution = payload.get("execution") or {}
    read_plan = payload.get("read_plan") or {}
    value = extract_tet_value(op, execution)
    chunks = read_plan.get("chunk_count")
    device_label = _device_label(execution)
    strategy = execution.get("memory_strategy")
    fold_par = execution.get("fold_parallel")
    io_regime = execution.get("io_regime")
    fold_workers = execution.get("fold_workers")
    log(
        f".tet {label}{par_tag}: done in {format_elapsed(elapsed)}  value={value}  "
        f"strategy={strategy}  fold_parallel={fold_par}  io_regime={io_regime}  "
        f"device={device_label}"
    )
    return TetOpTiming(
        seconds=elapsed,
        value=value,
        chunks=chunks if isinstance(chunks, int) else None,
        device=device_label,
        fold_parallel=fold_par if isinstance(fold_par, bool) else None,
        io_regime=str(io_regime) if io_regime is not None else None,
        fold_workers=fold_workers if isinstance(fold_workers, int) else None,
        memory_strategy=str(strategy) if strategy is not None else None,
    )


def run_tet_op_st_and_parallel(
    tet: Path,
    tet_path: Path,
    case: BenchCase,
    op: OpName,
    *,
    device: str | None = None,
) -> tuple[TetOpTiming, TetOpTiming]:
    """Double page-cache warm, then time ST and parallel (align/group_by style)."""
    tet_device = device if device is not None else os.environ.get(
        "TET_BENCH_DEVICE", DEFAULT_TET_DEVICE
    )
    body = query_json(op, device=tet_device, fold_parallel=False)
    meta = op_meta(op)
    label = meta.transform_method or op
    for i in range(2):
        log(f".tet {label}: page-cache warm {i + 1}/2 over ~{case.logical_gib} GiB …")
        try:
            _run_query_once(tet, tet_path, body, device=tet_device)
        except RuntimeError:
            pass
    st = run_tet_op(
        tet, tet_path, case, op, device=device, fold_parallel=False, warm_passes=0
    )
    par = run_tet_op(
        tet, tet_path, case, op, device=device, fold_parallel=True, warm_passes=0
    )
    return st, par


def run_scalar_sum_sanity(
    tet: Path,
    tet_path: Path,
    case: BenchCase,
    *,
    device: str | None = None,
) -> TetOpTiming:
    """Scalar `sum: []` on the same `.tet` (ceiling vs group_by)."""
    return run_tet_op(
        tet,
        tet_path,
        case,
        "sum",
        device=device,
        fold_parallel=True,
        warm_passes=1,
    )


def wants_st_parallel(op: OpName) -> bool:
    """True for group_by ops that should report ST + parallel like the index bench."""
    return is_align_group_op(op) and op.startswith("group_by")


def run_convert(tet: Path, src: Path, out: Path, jobs: int) -> float:
    env = {**os.environ, "TET_NO_QUERY_HISTORY": "1"}
    t0 = time.perf_counter()
    subprocess.run(
        [str(tet), "convert", str(src), str(out), "--jobs", str(jobs)],
        check=True,
        env=env,
    )
    return time.perf_counter() - t0

"""`.tet` convert and query helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmark.case import BenchCase
from benchmark.constants import DEFAULT_TET_DEVICE, OpName, REPO_ROOT, op_meta
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


def query_json(op: OpName, *, device: str | None) -> str:
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
    if device:
        doc["execution"] = {"device": device}
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


def run_tet_op(
    tet: Path,
    tet_path: Path,
    case: BenchCase,
    op: OpName,
    *,
    device: str | None = None,
) -> tuple[float, float | None, int | None, str | None]:
    tet_device = device if device is not None else os.environ.get(
        "TET_BENCH_DEVICE", DEFAULT_TET_DEVICE
    )
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
    if tet_device:
        cmd.extend(["--device", tet_device])
    body = query_json(op, device=tet_device)
    meta = op_meta(op)
    label = meta.transform_method or op

    def once() -> dict:
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

    log(f".tet {label}: warm pass over ~{case.logical_gib} GiB …")
    try:
        once()
    except RuntimeError:
        pass

    log(f".tet {label}: timed pass …")
    t0 = time.perf_counter()
    payload = once()
    elapsed = time.perf_counter() - t0

    execution = payload.get("execution") or {}
    read_plan = payload.get("read_plan") or {}
    value = extract_tet_value(op, execution)
    chunks = read_plan.get("chunk_count")
    device_label = _device_label(execution)
    strategy = execution.get("memory_strategy")
    log(
        f".tet {label}: done in {format_elapsed(elapsed)}  value={value}  "
        f"strategy={strategy}  device={device_label}"
    )
    return elapsed, value, chunks, device_label


def run_convert(tet: Path, src: Path, out: Path, jobs: int) -> float:
    env = {**os.environ, "TET_NO_QUERY_HISTORY": "1"}
    t0 = time.perf_counter()
    subprocess.run(
        [str(tet), "convert", str(src), str(out), "--jobs", str(jobs)],
        check=True,
        env=env,
    )
    return time.perf_counter() - t0

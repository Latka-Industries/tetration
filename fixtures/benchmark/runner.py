"""Per-case benchmark orchestration."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from benchmark.case import BenchCase
from benchmark.constants import FormatName, OpName, op_meta, op_compat
from benchmark.log import format_elapsed, log
from benchmark.source import reduce_source_op
from benchmark.spec import verify_case
from benchmark.tet import run_convert, run_tet_op
from generate.constants import FIXTURES_ROOT
from generate.dispatch import run_target


def generate_for_bench(target: str, case: BenchCase) -> None:
    log(f"{case.format}/{case.tier}: generating ~{case.logical_gib} GiB ({target})")
    run_target(target, quiet=False)  # type: ignore[arg-type]


def delete_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def delete_path_logged(path: Path, label: str) -> None:
    if not path.exists():
        log(f"{label}: nothing to delete")
        return
    log(f"{label}: deleting {path.name} …")
    t0 = time.perf_counter()
    delete_path(path)
    log(f"{label}: deleted in {format_elapsed(time.perf_counter() - t0)}")


def cleanup_format_tree(format: FormatName) -> None:
    for tier_root in (FIXTURES_ROOT / "large", FIXTURES_ROOT / "extra_large"):
        fmt_dir = tier_root / format
        if fmt_dir.exists():
            shutil.rmtree(fmt_dir)
        try:
            tier_root.rmdir()
        except OSError:
            pass


def values_match(op: OpName, source: float, tet: float) -> bool:
    meta = op_meta(op)
    tol = meta.tolerance
    if meta.is_bool:
        return (source != 0.0) == (tet != 0.0)
    if op == "count" or op.endswith("_count"):
        return abs(source - tet) <= tol
    scale = max(abs(source), abs(tet), 1e-9)
    relative = op in ("sum", "var", "transform_l1") or meta.value_mode == "reduced_sum_total"
    return abs(source - tet) <= tol * scale if relative else abs(source - tet) <= tol


@dataclass
class OpResult:
    op: OpName
    source_s: float | None = None
    source_value: float | None = None
    tet_s: float | None = None
    tet_value: float | None = None
    tet_device: str | None = None
    skip_source: bool = False
    note: str | None = None
    mismatch: bool = False


@dataclass
class BenchRow:
    case: BenchCase
    convert_s: float
    chunks: int | None
    ops: dict[OpName, OpResult] = field(default_factory=dict)


def bench_case(
    tet: Path,
    case: BenchCase,
    *,
    jobs: int,
    ops: tuple[OpName, ...],
    skip_ops: bool,
    tet_device: str | None = None,
) -> BenchRow:
    log(f"=== {case.format} / {case.tier} (~{case.logical_gib} GiB) ops={','.join(ops)} ===")
    generate_for_bench(case.gen_target, case)
    try:
        verify_case(case)
        log(f"{case.format}/{case.tier}: workload verified against benchmark/spec.json")
    except Exception as exc:  # noqa: BLE001
        log(f"warning: {case.format}/{case.tier} spec verify failed — {exc}")

    op_results: dict[OpName, OpResult] = {}

    if not skip_ops:
        for op in ops:
            compat = op_compat(case.format, op)
            result = OpResult(op=op, skip_source=not compat.bench_source, note=compat.note)
            if compat.bench_source:
                try:
                    result.source_s, result.source_value = reduce_source_op(case, op)
                except Exception as exc:  # noqa: BLE001 — bench should continue
                    result.skip_source = True
                    result.note = f"source skipped: {exc}"
                    log(f"warning: {case.format}/{case.tier} {op} source skipped — {exc}")
            else:
                log(f"{case.format}/{case.tier} {op}: source n/a — {compat.note}")
            op_results[op] = result

    log(f"convert → {case.tet_out.name} (jobs={jobs})")
    convert_s = run_convert(tet, case.src, case.tet_out, jobs)
    log(f"convert done in {format_elapsed(convert_s)}")

    delete_path_logged(case.src, "delete source")

    chunks: int | None = None
    if not skip_ops:
        for op in ops:
            result = op_results[op]
            try:
                result.tet_s, result.tet_value, op_chunks, result.tet_device = run_tet_op(
                    tet, case.tet_out, case, op, device=tet_device
                )
                chunks = chunks or op_chunks
            except Exception as exc:  # noqa: BLE001
                result.note = (result.note or "") + f"; tet failed: {exc}"
                log(f"warning: {case.format}/{case.tier} {op} .tet failed — {exc}")
                continue
            if (
                result.source_value is not None
                and result.tet_value is not None
                and not values_match(op, result.source_value, result.tet_value)
            ):
                result.mismatch = True
                delta = abs(result.source_value - result.tet_value)
                log(
                    f"warning: {op} mismatch source vs .tet (|Δ|={delta:.6g}) "
                    f"— check note if format semantics differ",
                )

    delete_path_logged(case.tet_out, "delete .tet")
    return BenchRow(case=case, convert_s=convert_s, chunks=chunks, ops=op_results)

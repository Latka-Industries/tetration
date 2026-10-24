"""Per-case benchmark orchestration."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from benchmark.case import BenchCase
from benchmark.constants import FormatName, OpName, is_align_group_op, op_compat, op_meta
from benchmark.log import format_elapsed, log
from benchmark.source import reduce_source_op
from benchmark.spec import verify_case
from benchmark.tet import (
    run_convert,
    run_scalar_sum_sanity,
    run_tet_op,
    run_tet_op_st_and_parallel,
    wants_st_parallel,
)
from generate.align_group import AlignGroupTier, fixture_for, generate_align_group
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
    for tier_root in (
        FIXTURES_ROOT / "bench_small",
        FIXTURES_ROOT / "large",
        FIXTURES_ROOT / "extra_large",
    ):
        fmt_dir = tier_root / format
        if fmt_dir.exists():
            shutil.rmtree(fmt_dir)
        try:
            tier_root.rmdir()
        except OSError:
            pass
    # Align/group_by dual-dataset trees (h5-only; not under {format}/).
    if format == "h5":
        for path in (
            FIXTURES_ROOT / "large" / "align_group_small",
            FIXTURES_ROOT / "large" / "align_group",
            FIXTURES_ROOT / "extra_large" / "align_group",
        ):
            if path.exists():
                shutil.rmtree(path)


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
    tet_s_st: float | None = None
    tet_s_par: float | None = None
    fold_parallel: bool | None = None
    io_regime: str | None = None
    fold_workers: int | None = None
    skip_source: bool = False
    note: str | None = None
    mismatch: bool = False


@dataclass
class BenchRow:
    case: BenchCase
    convert_s: float
    chunks: int | None
    ops: dict[OpName, OpResult] = field(default_factory=dict)


def _align_group_case(tier: AlignGroupTier) -> BenchCase:
    fx = fixture_for(tier)
    return BenchCase(
        format="h5",
        tier=tier,  # type: ignore[arg-type]
        src=fx.h5,
        tet_out=fx.tet,
        gen_target=f"align-group-{tier}",
        logical_bytes=fx.logical_bytes,
    )


def _bench_align_group(
    tet: Path,
    tier: AlignGroupTier,
    ops: tuple[OpName, ...],
    *,
    jobs: int,
    tet_device: str | None,
    op_results: dict[OpName, OpResult],
) -> tuple[float, int | None]:
    """Generate dual-dataset fixture, convert, time align/group_by ops, cleanup."""
    case = _align_group_case(tier)
    log(
        f"=== h5 / {tier} align_group (~{case.logical_gib} GiB) "
        f"ops={','.join(ops)} ==="
    )
    log(f"h5/{tier}: generating align/group_by fixture")
    generate_align_group(tier, quiet=False)

    for op in ops:
        compat = op_compat(case.format, op)
        result = OpResult(op=op, skip_source=not compat.bench_source, note=compat.note)
        if compat.bench_source:
            try:
                result.source_s, result.source_value = reduce_source_op(case, op)
            except Exception as exc:  # noqa: BLE001
                result.skip_source = True
                result.note = f"source skipped: {exc}"
                log(f"warning: h5/{tier} {op} source skipped — {exc}")
        else:
            log(f"h5/{tier} {op}: source n/a — {compat.note}")
        op_results[op] = result

    log(f"convert → {case.tet_out.name} (jobs={jobs})")
    convert_s = run_convert(tet, case.src, case.tet_out, jobs)
    log(f"convert done in {format_elapsed(convert_s)}")
    delete_path_logged(case.src, "delete align_group source")

    chunks: int | None = None
    # Scalar sum ceiling on the same .tet (once per align_group run).
    try:
        sanity = run_scalar_sum_sanity(tet, case.tet_out, case, device=tet_device)
        log(
            f".tet scalar_sum sanity: {format_elapsed(sanity.seconds)}  "
            f"value={sanity.value}  fold_parallel={sanity.fold_parallel}  "
            f"io_regime={sanity.io_regime}"
        )
        chunks = chunks or sanity.chunks
    except Exception as exc:  # noqa: BLE001
        log(f"warning: h5/{tier} scalar_sum sanity failed — {exc}")

    for op in ops:
        result = op_results[op]
        try:
            if wants_st_parallel(op):
                st, par = run_tet_op_st_and_parallel(
                    tet, case.tet_out, case, op, device=tet_device
                )
                result.tet_s_st = st.seconds
                result.tet_s_par = par.seconds
                result.tet_s = par.seconds
                result.tet_value = par.value
                result.tet_device = par.device
                result.fold_parallel = par.fold_parallel
                result.io_regime = par.io_regime
                result.fold_workers = par.fold_workers
                chunks = chunks or par.chunks
                extra = (
                    f"ST={st.seconds:.3f}s par={par.seconds:.3f}s "
                    f"fold_parallel={par.fold_parallel} io_regime={par.io_regime}"
                )
                result.note = f"{result.note}; {extra}" if result.note else extra
                # Prefer parallel value for mismatch; also check ST agrees.
                if (
                    st.value is not None
                    and par.value is not None
                    and not values_match(op, st.value, par.value)
                ):
                    result.mismatch = True
                    log(
                        f"warning: {op} ST vs parallel value mismatch "
                        f"(|Δ|={abs(st.value - par.value):.6g})"
                    )
            else:
                timing = run_tet_op(
                    tet, case.tet_out, case, op, device=tet_device, warm_passes=2
                )
                result.tet_s = timing.seconds
                result.tet_value = timing.value
                result.tet_device = timing.device
                result.fold_parallel = timing.fold_parallel
                result.io_regime = timing.io_regime
                result.fold_workers = timing.fold_workers
                chunks = chunks or timing.chunks
                if timing.io_regime or timing.fold_parallel is not None:
                    extra = (
                        f"fold_parallel={timing.fold_parallel} io_regime={timing.io_regime}"
                    )
                    result.note = f"{result.note}; {extra}" if result.note else extra
        except Exception as exc:  # noqa: BLE001
            result.note = (result.note or "") + f"; tet failed: {exc}"
            log(f"warning: h5/{tier} {op} .tet failed — {exc}")
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

    delete_path_logged(case.tet_out, "delete align_group .tet")
    pkg = Path(str(case.tet_out) + ".d")
    delete_path_logged(pkg, "delete align_group package")
    return convert_s, chunks


def bench_case(
    tet: Path,
    case: BenchCase,
    *,
    jobs: int,
    ops: tuple[OpName, ...],
    skip_ops: bool,
    tet_device: str | None = None,
) -> BenchRow:
    plain = tuple(op for op in ops if not is_align_group_op(op))
    rich = tuple(op for op in ops if is_align_group_op(op))
    op_results: dict[OpName, OpResult] = {}
    convert_s = 0.0
    chunks: int | None = None

    if skip_ops:
        log(f"=== {case.format} / {case.tier} (~{case.logical_gib} GiB) ops skipped ===")
        if plain or not rich:
            generate_for_bench(case.gen_target, case)
            log(f"convert → {case.tet_out.name} (jobs={jobs})")
            convert_s = run_convert(tet, case.src, case.tet_out, jobs)
            log(f"convert done in {format_elapsed(convert_s)}")
            delete_path_logged(case.src, "delete source")
            delete_path_logged(case.tet_out, "delete .tet")
        return BenchRow(case=case, convert_s=convert_s, chunks=None, ops={})

    if plain:
        log(f"=== {case.format} / {case.tier} (~{case.logical_gib} GiB) ops={','.join(plain)} ===")
        generate_for_bench(case.gen_target, case)
        try:
            verify_case(case)
            log(f"{case.format}/{case.tier}: workload verified against benchmark/spec.json")
        except Exception as exc:  # noqa: BLE001
            log(f"warning: {case.format}/{case.tier} spec verify failed — {exc}")

        # Source peers need the native file; time them before convert deletes it.
        for op in plain:
            compat = op_compat(case.format, op)
            result = OpResult(op=op, skip_source=not compat.bench_source, note=compat.note)
            if compat.bench_source:
                try:
                    result.source_s, result.source_value = reduce_source_op(case, op)
                except Exception as exc:  # noqa: BLE001
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

        for op in plain:
            result = op_results[op]
            try:
                timing = run_tet_op(tet, case.tet_out, case, op, device=tet_device)
                result.tet_s = timing.seconds
                result.tet_value = timing.value
                result.tet_device = timing.device
                result.fold_parallel = timing.fold_parallel
                result.io_regime = timing.io_regime
                result.fold_workers = timing.fold_workers
                chunks = chunks or timing.chunks
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

    report_case = case
    if rich:
        if case.format != "h5":
            for op in rich:
                note = "align/group_by bench fixtures are h5-only (dual-dataset 2-D)"
                op_results[op] = OpResult(op=op, skip_source=True, note=note)
                log(f"{case.format}/{case.tier} {op}: source n/a — {note}")
        else:
            rich_convert, rich_chunks = _bench_align_group(
                tet,
                case.tier,  # type: ignore[arg-type]
                rich,
                jobs=jobs,
                tet_device=tet_device,
                op_results=op_results,
            )
            if not plain:
                convert_s = rich_convert
                report_case = _align_group_case(case.tier)  # type: ignore[arg-type]
            chunks = chunks or rich_chunks

    return BenchRow(case=report_case, convert_s=convert_s, chunks=chunks, ops=op_results)

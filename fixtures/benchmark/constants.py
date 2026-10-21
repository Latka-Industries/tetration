"""Benchmark constants, op registry, and `--ops` / `--tier` parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from generate.constants import FIXTURES_ROOT

REPO_ROOT = FIXTURES_ROOT.parent
RESULTS_DIR = FIXTURES_ROOT / "bench_results"
RUNS_DIR = RESULTS_DIR / "runs"
RESULTS_FILE = RESULTS_DIR / "latest.md"

FormatName = Literal["h5", "netcdf", "zarr"]
TierName = Literal["large", "extra"]
OpName = str
ValueMode = Literal["scalar", "reduced_sum_total", "window_count"]

DEFAULT_FORMATS: tuple[FormatName, ...] = ("h5", "netcdf", "zarr")

# Passed as query `execution.device` (and `tet query --device` when set). Use `auto` for
# Phase 10 routing; `cpu` to match pre-GPU benches.
DEFAULT_TET_DEVICE = "auto"

POP_DDOF = 0

# Window size for `window_*` bench ops. Must stay ≤ QueryLimits::DEFAULT.max_window_size.
WINDOW_BENCH_SIZE = 1_048_576

TRANSFORM_METHODS: tuple[str, ...] = (
    "zscore",
    "minmax",
    "l1",
    "l2",
    "center",
    "scale",
    "log1p",
    "sqrt",
    "softmax",
)


@dataclass(frozen=True)
class OpMeta:
    """Per-op bench metadata used by harness + report rows."""

    result_key: str
    tolerance: float
    bench_source: bool = True
    is_bool: bool = False
    is_transform: bool = False
    transform_method: str | None = None
    note: str | None = None
    # Extra closed query keys merged into the document (window / align / group_by).
    query_fields: dict[str, Any] = field(default_factory=dict)
    # How to pull a comparable scalar from tet execution JSON / source peer.
    value_mode: ValueMode = "scalar"


def _scalar(key: str, result_key: str, tol: float, **kw) -> OpMeta:
    """Register a streaming scalar / QC fold op."""
    return OpMeta(result_key=result_key, tolerance=tol, **kw)


def _transform(method: str, result_key: str, tol: float, **kw) -> OpMeta:
    """Register a two-pass transform (+ write) op; `result_key` is the pass-1 stat."""
    return OpMeta(
        result_key=result_key,
        tolerance=tol,
        is_transform=True,
        transform_method=method,
        **kw,
    )


def _window(mode: str, *, size: int = WINDOW_BENCH_SIZE, step: int | None = None) -> OpMeta:
    """Register a 1-D window sum op (`bins` or `rolling`) along axis 0."""
    window: dict[str, Any] = {"axis": 0, "size": size, "mode": mode}
    if step is not None:
        window["step"] = step
    note = f"0.6.x window {mode} (axis 0, size={size}"
    if step is not None:
        note += f", step={step}"
    note += "); source = NumPy slab fold matching wire geometry"
    return OpMeta(
        result_key="operation_reduced_sum",
        tolerance=1e-2,
        bench_source=True,
        query_fields={"window": window, "sum": []},
        value_mode="reduced_sum_total",
        note=note,
    )


# Registry of ops accepted by `bench --ops` (and groups below).
OP_REGISTRY: dict[OpName, OpMeta] = {
    "mean": _scalar("mean", "operation_mean", 1e-4),
    "sum": _scalar("sum", "operation_sum", 1e-2),
    "min": _scalar("min", "operation_min", 1e-5),
    "max": _scalar("max", "operation_max", 1e-5),
    "count": _scalar("count", "operation_element_count", 0.5),
    "std": _scalar("std", "operation_std", 1e-4),
    "var": _scalar("var", "operation_var", 1e-4),
    "nan_mean": _scalar("nan_mean", "operation_nan_mean", 1e-4),
    "nan_std": _scalar("nan_std", "operation_nan_std", 1e-4),
    "any_inf": _scalar(
        "any_inf",
        "operation_any_inf",
        0.0,
        is_bool=True,
        note="clean uniform f32 workload; expect false",
    ),
    "inf_count": _scalar(
        "inf_count",
        "operation_inf_count",
        0.5,
        note="clean uniform f32 workload; expect 0",
    ),
    "nan_count": _scalar(
        "nan_count",
        "operation_nan_count",
        0.5,
        note="clean uniform f32 workload; expect 0",
    ),
    "transform_zscore": _transform(
        "zscore",
        "operation_mean",
        1e-4,
        note="two-pass+write; value = pass-1 mean (≈0.5 on uniform [0,1])",
    ),
    "transform_minmax": _transform(
        "minmax",
        "operation_min",
        1e-5,
        note="two-pass+write; value = pass-1 min (≈0 on uniform [0,1])",
    ),
    "transform_center": _transform(
        "center",
        "operation_mean",
        1e-4,
        note="two-pass+write; value = pass-1 mean",
    ),
    "transform_scale": _transform(
        "scale",
        "operation_std",
        1e-4,
        note="two-pass+write; value = pass-1 std",
    ),
    "transform_l1": _transform(
        "l1",
        "operation_norm_l1",
        1e-2,
        note="two-pass+write; value = pass-1 L1 norm",
    ),
    "transform_l2": _transform(
        "l2",
        "operation_norm_l2",
        1e-4,
        note="two-pass+write; value = pass-1 L2 norm",
    ),
    "transform_log1p": _transform(
        "log1p",
        "operation_min",
        1e-5,
        note="two-pass+write; value = pass-1 min (shift for log1p)",
    ),
    "transform_sqrt": _transform(
        "sqrt",
        "operation_min",
        1e-5,
        note="two-pass+write; value = pass-1 min (shift for sqrt)",
    ),
    "transform_softmax": _transform(
        "softmax",
        "operation_max",
        1e-5,
        note="two-pass+write; value = pass-1 max (stabilization)",
    ),
    # 0.6.x query surfaces (THI-608). Large fixtures are 1-D `data` without coords /
    # second datasets, so align + label group_by stay registered but need richer fixtures.
    "window_bins_sum": _window("bins"),
    "window_rolling_sum": _window("rolling", step=WINDOW_BENCH_SIZE),
}

SCALAR_OPS: tuple[OpName, ...] = (
    "mean",
    "sum",
    "min",
    "max",
    "count",
    "std",
    "var",
)

QC_OPS: tuple[OpName, ...] = ("nan_mean", "nan_std", "any_inf", "inf_count", "nan_count")

TRANSFORM_OPS: tuple[OpName, ...] = tuple(
    f"transform_{m}" for m in TRANSFORM_METHODS
)

WINDOW_OPS: tuple[OpName, ...] = ("window_bins_sum", "window_rolling_sum")
# Alias kept for THI-608 / docs that say query06.
QUERY06_OPS: tuple[OpName, ...] = WINDOW_OPS

DEFAULT_OPS: tuple[OpName, ...] = SCALAR_OPS + QC_OPS

ALL_OPS: tuple[OpName, ...] = DEFAULT_OPS + TRANSFORM_OPS + WINDOW_OPS

OP_GROUPS: dict[str, tuple[OpName, ...]] = {
    "default": DEFAULT_OPS,
    "scalar": SCALAR_OPS,
    "qc": QC_OPS,
    "transforms": TRANSFORM_OPS,
    "windows": WINDOW_OPS,
    "query06": QUERY06_OPS,
    "all": ALL_OPS,
}

TierFilter = Literal["large", "extra", "all"]
DEFAULT_TIERS: tuple[TierName, ...] = ("large", "extra")


def parse_tiers(raw: str | None) -> tuple[TierName, ...]:
    """Parse `--tier large|extra|all` (default all)."""
    token = (raw or "all").strip().lower()
    if token == "all":
        return DEFAULT_TIERS
    if token in ("large", "extra"):
        return (token,)  # type: ignore[return-value]
    raise ValueError(f"unknown tier {raw!r}; choose large, extra, or all")


@dataclass(frozen=True)
class OpCompat:
    bench_source: bool
    note: str | None = None


def op_meta(op: OpName) -> OpMeta:
    try:
        return OP_REGISTRY[op]
    except KeyError as exc:
        known = ", ".join(OP_REGISTRY)
        raise ValueError(f"unknown op {op!r}; choose from {known}") from exc


def op_compat(format: FormatName, op: OpName) -> OpCompat:
    meta = op_meta(op)
    notes: list[str] = []
    if meta.note:
        notes.append(meta.note)
    if format == "zarr":
        notes.append(
            "zarr: Python dir-store raw f32 chunks (still slower than h5/netcdf C libs)"
        )
    if format == "netcdf" and op in ("std", "var", "nan_std"):
        notes.append("netcdf: population var/std ddof=0 via numpy slabs")
    if meta.is_transform:
        notes.append("transform: source two-pass+write (NumPy) vs tet write:switch spill")
    return OpCompat(
        bench_source=meta.bench_source,
        note="; ".join(notes) if notes else None,
    )

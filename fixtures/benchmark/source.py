"""Native source peers for benchmark comparisons.

Scalar/QC ops stream 16 MiB slabs into running stats. Window ops fold NumPy slabs
to match tetration bins/rolling wire. Transform ops run a fair two-pass+write peer
(pass-1 stats, then apply + spill f32 LE) so source times include the same I/O class
as `.tet` `write: switch`.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass

import numpy as np
from generate.constants import CHUNK_ELEMS
from generate.util import iter_slabs

from benchmark.case import BenchCase
from benchmark.constants import OpName, op_meta
from benchmark.log import time_warm_logged


@dataclass
class RunningStats:
    count: int = 0
    sum: float = 0.0
    min: float = math.inf
    max: float = -math.inf
    m2: float = 0.0
    any_inf: bool = False
    inf_count: int = 0
    nan_count: int = 0

    def push_slab(self, arr: np.ndarray) -> None:
        flat = np.asarray(arr, dtype=np.float64).ravel()
        n = int(flat.size)
        if n == 0:
            return
        if np.isinf(flat).any():
            self.any_inf = True
            self.inf_count += int(np.isinf(flat).sum())
        if np.isnan(flat).any():
            self.nan_count += int(np.isnan(flat).sum())

        finite = flat[np.isfinite(flat)]
        n_fin = int(finite.size)
        if n_fin == 0:
            return

        slab_sum = float(finite.sum())
        slab_mean = slab_sum / n_fin
        slab_m2 = float(((finite - slab_mean) ** 2).sum())
        slab_min = float(finite.min())
        slab_max = float(finite.max())

        if self.count == 0:
            self.count = n_fin
            self.sum = slab_sum
            self.min = slab_min
            self.max = slab_max
            self.m2 = slab_m2
            return

        n_a = self.count
        n_b = n_fin
        total = n_a + n_b
        mean_a = self.sum / n_a
        delta = slab_mean - mean_a
        self.m2 += slab_m2 + delta * delta * n_a * n_b / total
        self.count = total
        self.sum += slab_sum
        self.min = min(self.min, slab_min)
        self.max = max(self.max, slab_max)

    def finish_scalar(self, op: OpName) -> float:
        if self.count == 0:
            raise ValueError("empty selection")
        match op:
            case "mean" | "nan_mean":
                return self.sum / self.count
            case "sum":
                return self.sum
            case "min":
                return self.min
            case "max":
                return self.max
            case "count":
                return float(self.count)
            case "var":
                return self.m2 / self.count if self.count else 0.0
            case "std" | "nan_std":
                return math.sqrt(self.m2 / self.count) if self.count else 0.0
            case "any_inf":
                return 1.0 if self.any_inf else 0.0
            case "inf_count":
                return float(self.inf_count)
            case "nan_count":
                return float(self.nan_count)
        raise ValueError(op)


def _iter_source_slabs(case: BenchCase):
    slab_mib = CHUNK_ELEMS // (1024 * 1024)

    def run_h5():
        import h5py

        with h5py.File(case.src, "r") as f:
            d = f["data"]
            length = int(d.shape[0])
            for start, end in iter_slabs(length):
                yield np.asarray(d[start:end], dtype=np.float64)

    def run_netcdf():
        import netCDF4 as nc

        with nc.Dataset(case.src, "r") as ds:
            var = ds.variables["data"]
            length = int(var.shape[0])
            for start, end in iter_slabs(length):
                yield np.asarray(var[start:end], dtype=np.float64)

    def run_zarr():
        import zarr

        root = zarr.open_group(str(case.src), mode="r")
        arr = root["data"]
        length = int(arr.shape[0])
        chunk_elems = int(arr.chunks[0])
        for start in range(0, length, chunk_elems):
            end = min(start + chunk_elems, length)
            yield np.asarray(arr[start:end], dtype=np.float64)

    match case.format:
        case "h5":
            return run_h5(), f"source ({slab_mib} MiB slabs, h5py)"
        case "netcdf":
            return run_netcdf(), f"source ({slab_mib} MiB slabs, netCDF4)"
        case "zarr":
            chunk_mib = CHUNK_ELEMS * 4 // (1024 * 1024)
            return run_zarr(), f"source ({chunk_mib} MiB zarr chunks, raw)"


def _fold_source(case: BenchCase) -> RunningStats:
    slabs, _ = _iter_source_slabs(case)
    stats = RunningStats()
    for slab in slabs:
        stats.push_slab(slab)
    return stats


def reduce_source_op(case: BenchCase, op: OpName) -> tuple[float, float]:
    """Warm+timed source peer for `op`. Returns `(seconds, comparable_value)`."""
    meta = op_meta(op)
    if not meta.bench_source:
        raise ValueError(meta.note or "source not supported")

    if meta.value_mode == "aligned":
        return time_warm_logged(f"source {op} (h5 metadata)", lambda: _align_check_source(case))

    if meta.query_fields and "group_by" in meta.query_fields:
        by_labels = isinstance(meta.query_fields["group_by"], dict)

        def compute_group_by() -> float:
            return _group_by_sum_total(case, by_labels=by_labels)

        return time_warm_logged(f"source {op} (h5py+numpy group_by)", compute_group_by)

    _, slab_label = _iter_source_slabs(case)

    if meta.query_fields and "window" in meta.query_fields:
        window = meta.query_fields["window"]

        def compute_window() -> float:
            return _window_sum_total(
                case,
                size=int(window["size"]),
                mode=str(window["mode"]),
                step=int(window["step"]) if window.get("step") is not None else None,
            )

        return time_warm_logged(f"source {op} ({slab_label})", compute_window)

    if meta.is_transform:
        method = meta.transform_method
        assert method is not None

        def compute() -> float:
            # Full two-pass peer matching tet `write: switch` spill: pass-1 stats,
            # then stream-apply and write f32 LE to a temp file (same I/O class).
            return _transform_two_pass(case, method)

        return time_warm_logged(
            f"source transform/{method} two-pass+write ({slab_label})",
            compute,
        )

    def compute_scalar() -> float:
        return _fold_source(case).finish_scalar(op)

    return time_warm_logged(f"source {op} ({slab_label})", compute_scalar)


def _source_length(case: BenchCase) -> int:
    """Leading-axis extent of the native `data` array (window peer geometry)."""
    match case.format:
        case "h5":
            import h5py

            with h5py.File(case.src, "r") as f:
                return int(f["data"].shape[0])
        case "netcdf":
            import netCDF4 as nc

            with nc.Dataset(case.src, "r") as ds:
                return int(ds.variables["data"].shape[0])
        case "zarr":
            import zarr

            root = zarr.open_group(str(case.src), mode="r")
            return int(root["data"].shape[0])


def _align_check_source(case: BenchCase) -> float:
    """1.0 when `data` and `ref` agree on shape (align peer; metadata only)."""
    if case.format != "h5":
        raise ValueError("align source peer is h5-only")
    import h5py

    with h5py.File(case.src, "r") as f:
        if "data" not in f or "ref" not in f:
            raise ValueError("align fixture missing data/ref datasets")
        left = tuple(int(x) for x in f["data"].shape)
        right = tuple(int(x) for x in f["ref"].shape)
    return 1.0 if left == right else 0.0


def _group_by_sum_total(case: BenchCase, *, by_labels: bool) -> float:
    """Sum of per-group totals along axis 0 (index or collapsed labels)."""
    if case.format != "h5":
        raise ValueError("group_by source peer is h5-only")
    import h5py

    with h5py.File(case.src, "r") as f:
        data = f["data"]
        time_len, width = (int(data.shape[0]), int(data.shape[1]))
        if by_labels:
            labels = np.asarray(f["coordinates/time"][:], dtype=np.int64)
            if labels.shape != (time_len,):
                raise ValueError("time coords length mismatch")
            # First-seen order matches tetration label_group index_to_group.
            order: list[int] = []
            first: dict[int, int] = {}
            for lab in labels.tolist():
                if lab not in first:
                    first[lab] = len(order)
                    order.append(lab)
            totals = np.zeros(len(order), dtype=np.float64)
            slab = min(width, 16 * 1024 * 1024)
            for t, lab in enumerate(labels.tolist()):
                gid = first[int(lab)]
                for x0 in range(0, width, slab):
                    x1 = min(x0 + slab, width)
                    totals[gid] += float(np.asarray(data[t, x0:x1], dtype=np.float64).sum())
            return float(totals.sum())

        totals = np.zeros(time_len, dtype=np.float64)
        slab = min(width, 16 * 1024 * 1024)
        for t in range(time_len):
            for x0 in range(0, width, slab):
                x1 = min(x0 + slab, width)
                totals[t] += float(np.asarray(data[t, x0:x1], dtype=np.float64).sum())
        return float(totals.sum())


def _window_sum_total(
    case: BenchCase,
    *,
    size: int,
    mode: str,
    step: int | None,
) -> float:
    """1-D windowed sum total matching tetration bins/rolling wire (pad none, align start)."""
    if size < 1:
        raise ValueError("window size must be >= 1")
    stride = step if step is not None else 1
    if stride < 1:
        raise ValueError("window step must be >= 1")
    if mode == "bins" and step is not None:
        raise ValueError("bins rejects step")

    length = _source_length(case)
    if length == 0:
        raise ValueError("empty selection")
    if size > length:
        raise ValueError(f"window size {size} exceeds extent {length}")

    slabs, _ = _iter_source_slabs(case)

    if mode == "bins":
        window_count = (length + size - 1) // size
        totals = np.zeros(window_count, dtype=np.float64)
        offset = 0
        for slab in slabs:
            flat = np.asarray(slab, dtype=np.float64).ravel()
            n = int(flat.size)
            idx = np.arange(offset, offset + n, dtype=np.int64) // size
            np.add.at(totals, idx, flat)
            offset += n
        return float(totals.sum())

    if mode == "rolling":
        # Fully in-bounds starts (pad none): 0, stride, … while start + size <= length.
        last = length - size
        starts = list(range(0, last + 1, stride))
        if not starts:
            raise ValueError("window produces zero output positions")
        # Fast path: non-overlapping rolling (step == size) — stream like bins, drop short tail.
        if stride == size:
            window_count = length // size
            totals = np.zeros(window_count, dtype=np.float64)
            offset = 0
            for slab in slabs:
                flat = np.asarray(slab, dtype=np.float64).ravel()
                n = int(flat.size)
                idx = np.arange(offset, offset + n, dtype=np.int64) // size
                mask = idx < window_count
                if mask.any():
                    np.add.at(totals, idx[mask], flat[mask])
                offset += n
            return float(totals.sum())

        # Overlapping rolling: one streaming cumsum via slab prefix (O(n) RAM for prefix only).
        totals = np.zeros(len(starts), dtype=np.float64)
        # Build full cumsum in one pass into a compact buffer — unavoidable for arbitrary step.
        c = np.empty(length + 1, dtype=np.float64)
        c[0] = 0.0
        offset = 0
        for slab in slabs:
            flat = np.asarray(slab, dtype=np.float64).ravel()
            n = int(flat.size)
            np.cumsum(flat, out=c[offset + 1 : offset + 1 + n])
            if offset > 0:
                c[offset + 1 : offset + 1 + n] += c[offset]
            offset += n
        for wi, start in enumerate(starts):
            totals[wi] = c[start + size] - c[start]
        return float(totals.sum())

    raise ValueError(f"unknown window mode {mode!r}")


def _l1_norm(case: BenchCase) -> float:
    """Streaming L1 over source slabs (transform pass-1 peer)."""
    total = 0.0
    slabs, _ = _iter_source_slabs(case)
    for slab in slabs:
        total += float(np.abs(slab).sum())
    return total


def _l2_norm(case: BenchCase) -> float:
    """Streaming L2 over source slabs (transform pass-1 peer)."""
    total = 0.0
    slabs, _ = _iter_source_slabs(case)
    for slab in slabs:
        total += float((slab * slab).sum())
    return math.sqrt(total)


def _transform_two_pass(case: BenchCase, method: str) -> float:
    """Fair transform peer: pass-1 stats, then apply+write f32 LE; return pass-1 scalar.

    Timed path includes the write so source seconds match `.tet` spill I/O class.
    """
    if method == "l1":
        pass1 = _l1_norm(case)
        apply = lambda x, n1=pass1: x / n1 if n1 != 0.0 else np.nan
    elif method == "l2":
        pass1 = _l2_norm(case)
        apply = lambda x, n2=pass1: x / n2 if n2 != 0.0 else np.nan
    elif method in ("zscore", "center", "minmax", "scale", "log1p", "sqrt", "softmax"):
        stats = _fold_source(case)
        if method in ("zscore", "center"):
            mean = stats.finish_scalar("mean")
            std = stats.finish_scalar("std") if method == "zscore" else 1.0
            pass1 = mean
            apply = (
                (lambda x, mean=mean, std=std: (x - mean) / std)
                if method == "zscore"
                else (lambda x, mean=mean: x - mean)
            )
        elif method == "minmax":
            lo = stats.finish_scalar("min")
            hi = stats.finish_scalar("max")
            rng = hi - lo if hi != lo else 1.0
            pass1 = lo
            apply = lambda x, lo=lo, rng=rng: (x - lo) / rng
        elif method == "scale":
            std = stats.finish_scalar("std")
            pass1 = std
            apply = lambda x, std=std: x / std if std != 0.0 else np.nan
        elif method in ("log1p", "sqrt"):
            lo = stats.finish_scalar("min")
            pass1 = lo
            apply = (
                (lambda x, lo=lo: np.log1p(x - lo))
                if method == "log1p"
                else (lambda x, lo=lo: np.sqrt(np.maximum(x - lo, 0.0)))
            )
        else:  # softmax — extra pass for sum(exp(x-max))
            mx = stats.finish_scalar("max")
            sum_exp = 0.0
            slabs, _ = _iter_source_slabs(case)
            for slab in slabs:
                flat = np.asarray(slab, dtype=np.float64).ravel()
                finite = flat[np.isfinite(flat)]
                if finite.size:
                    sum_exp += float(np.exp(finite - mx).sum())
            pass1 = mx
            apply = (
                (lambda x, mx=mx, se=sum_exp: np.exp(x - mx) / se)
                if sum_exp != 0.0
                else (lambda x: np.full_like(x, np.nan, dtype=np.float64))
            )
    else:
        raise ValueError(method)

    fd, path = tempfile.mkstemp(prefix="bench-transform-", suffix=".f32.bin")
    os.close(fd)
    try:
        with open(path, "wb") as out:
            slabs, _ = _iter_source_slabs(case)
            for slab in slabs:
                flat = np.asarray(slab, dtype=np.float64).ravel()
                y = np.asarray(apply(flat), dtype=np.float32)
                out.write(y.tobytes(order="C"))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return float(pass1)

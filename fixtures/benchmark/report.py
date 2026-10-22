"""Markdown report rendering."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from benchmark.constants import (
    DEFAULT_OPS,
    DEFAULT_TET_DEVICE,
    OP_GROUPS,
    OP_REGISTRY,
    OpName,
    RESULTS_DIR,
    RESULTS_FILE,
    RUNS_DIR,
)
from benchmark.runner import BenchRow
from benchmark.spec import spec_summary
from benchmark.system import probe_system, report_metadata_lines
from benchmark.tet import git_short_rev
from generate.constants import CHUNK_ELEMS


def _cell(seconds: float | None) -> str:
    return f"{seconds:.3f}" if seconds is not None else "-"


def _val(value: float | None) -> str:
    if value is None:
        return "-"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


def format_md_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    aligns: list[str] | None = None,
) -> list[str]:
    if not headers:
        return []
    aligns = aligns or ["left"] * len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def pad(cell: str, width: int, align: str) -> str:
        if align == "right":
            return cell.rjust(width)
        if align == "center":
            return cell.center(width)
        return cell.ljust(width)

    def row_line(cells: list[str]) -> str:
        parts = [pad(c, widths[i], aligns[i]) for i, c in enumerate(cells)]
        return "| " + " | ".join(parts) + " |"

    sep_cells: list[str] = []
    for i, align in enumerate(aligns):
        w = widths[i]
        if align == "right":
            sep_cells.append("-" * max(3, w) + ":")
        elif align == "center":
            sep_cells.append(":" + "-" * max(1, w - 2) + ":")
        else:
            sep_cells.append(":" + "-" * max(2, w - 1))

    return [
        row_line(headers),
        "| " + " | ".join(sep_cells) + " |",
        *(row_line(r) for r in rows),
    ]


def make_run_id(git_rev: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{git_rev}_{ts}"


def _build_markdown_lines(
    *,
    jobs: int,
    ops: tuple[OpName, ...],
    skip_ops: bool,
    tet_device: str | None,
    rows: list[BenchRow],
    git_rev: str,
    run_id: str,
) -> list[str]:
    slab_mib = CHUNK_ELEMS // (1024 * 1024)
    spec = spec_summary()
    lines = [
        "# Small / large / extra_large benchmark",
        "",
        f"- **Run ID:** `{run_id}`",
        f"- **Date (UTC):** {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- **Git:** {git_rev}",
        f"- **Bench spec:** `{spec['spec_sha256'][:12]}…` (schema v{spec['schema_version']})",
        *report_metadata_lines(convert_jobs_requested=jobs),
        f"- **Ops:** {', '.join(ops)}",
        "- **Default ops:** "
        f"{', '.join(DEFAULT_OPS)} (scalar + QC); add `transforms` or `all` via `--ops`",
        "- **Scalar / QC:** tier-A/B streaming fold; population var/std ddof=0",
        "- **Transforms:** two-pass `transform` + `write: switch` (RAM or spill); "
        "values compare pass-1 stats unless noted",
        f"- **Source:** Python {slab_mib} MiB slabs, warm 2nd pass (skipped when not comparable)",
        f"- **`.tet`:** `tet query --execute --device {tet_device or DEFAULT_TET_DEVICE}` "
        "(stats JSON), warm 2nd pass",
        f"- **Ops skipped:** {skip_ops}",
        "",
        "## Notes",
        "",
        "- **zarr** source times use Python dir-store raw f32 chunks; compare `.tet` vs h5/netcdf, not zarr source as a ceiling.",
        "- Rows with `n/a` source skipped ops that do not line up for that format.",
        "- `⚠` = source vs `.tet` value mismatch beyond tolerance.",
        "- Workload verified against committed `benchmark/spec.json` after each generate.",
        "",
        "## convert",
        "",
    ]

    convert_rows: list[list[str]] = []
    for row in rows:
        chunks = str(row.chunks) if row.chunks is not None else "-"
        convert_rows.append(
            [
                row.case.format,
                row.case.tier,
                row.case.logical_gib,
                f"{row.convert_s:.3f}",
                chunks,
            ]
        )
    lines.extend(
        format_md_table(
            ["format", "tier", "GiB", "convert (s)", "chunks"],
            convert_rows,
            aligns=["left", "left", "right", "right", "right"],
        )
    )
    lines.append("")

    op_headers = [
        "format",
        "tier",
        "GiB",
        "source (s)",
        ".tet (s)",
        "source",
        ".tet",
        "device",
        "note",
    ]
    op_aligns = ["left", "left", "right", "right", "right", "right", "right", "left", "left"]

    for op in ops:
        lines.extend([f"## {op}", ""])
        op_rows: list[list[str]] = []
        for row in rows:
            r = row.ops.get(op)
            if r is None:
                continue
            note = r.note or ""
            if r.skip_source and r.source_s is None:
                src_s = "n/a"
                src_v = "n/a"
            else:
                src_s = _cell(r.source_s)
                src_v = _val(r.source_value)
            flag = " ⚠" if r.mismatch else ""
            tet_v = _val(r.tet_value) + flag
            op_rows.append(
                [
                    row.case.format,
                    row.case.tier,
                    row.case.logical_gib,
                    src_s,
                    _cell(r.tet_s),
                    src_v,
                    tet_v,
                    r.tet_device or "-",
                    note,
                ]
            )
        lines.extend(format_md_table(op_headers, op_rows, aligns=op_aligns))
        lines.append("")
    return lines


def _build_json_payload(
    *,
    run_id: str,
    git_rev: str,
    jobs: int,
    ops: tuple[OpName, ...],
    skip_ops: bool,
    tet_device: str | None,
    rows: list[BenchRow],
) -> dict:
    sys_info = probe_system(convert_jobs_requested=jobs)
    convert = [
        {
            "format": row.case.format,
            "tier": row.case.tier,
            "gib": row.case.logical_gib,
            "convert_s": row.convert_s,
            "chunks": row.chunks,
        }
        for row in rows
    ]
    operations: dict[str, list[dict]] = {}
    for op in ops:
        operations[op] = []
        for row in rows:
            r = row.ops.get(op)
            if r is None:
                continue
            operations[op].append(
                {
                    "format": row.case.format,
                    "tier": row.case.tier,
                    "gib": row.case.logical_gib,
                    "source_s": r.source_s,
                    "tet_s": r.tet_s,
                    "source_value": r.source_value,
                    "tet_value": r.tet_value,
                    "tet_device": r.tet_device,
                    "mismatch": r.mismatch,
                    "note": r.note,
                }
            )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "date_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": git_rev,
        "bench_spec": spec_summary(),
        "system": {
            "cpu": sys_info.cpu,
            "logical_cpus": sys_info.logical_cpus,
            "physical_cpus": sys_info.physical_cpus,
            "ram_total_gib": sys_info.ram_total_gib,
            "ram_available_gib": sys_info.ram_available_gib,
            "query_workers": sys_info.query_workers,
            "convert_workers": sys_info.convert_jobs,
            "gpu": sys_info.gpu,
            "gpu_vram_gib": sys_info.gpu_vram_gib,
        },
        "convert_jobs_requested": jobs,
        "tet_device": tet_device or DEFAULT_TET_DEVICE,
        "ops": list(ops),
        "skip_ops": skip_ops,
        "convert": convert,
        "operations": operations,
    }


def write_report(
    *,
    jobs: int,
    ops: tuple[OpName, ...],
    skip_ops: bool,
    tet_device: str | None = None,
    rows: list[BenchRow],
    run_id: str | None = None,
    no_clobber: bool = False,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    git_rev = git_short_rev()
    run_id = run_id or make_run_id(git_rev)
    run_dir = RUNS_DIR / run_id
    if run_dir.exists() and no_clobber:
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    lines = _build_markdown_lines(
        jobs=jobs,
        ops=ops,
        skip_ops=skip_ops,
        tet_device=tet_device,
        rows=rows,
        git_rev=git_rev,
        run_id=run_id,
    )
    text = "\n".join(lines) + "\n"
    md_path = run_dir / "report.md"
    json_path = run_dir / "report.json"
    md_path.write_text(text, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            _build_json_payload(
                run_id=run_id,
                git_rev=git_rev,
                jobs=jobs,
                ops=ops,
                skip_ops=skip_ops,
                tet_device=tet_device,
                rows=rows,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    RESULTS_FILE.write_text(text, encoding="utf-8")
    _format_latest_markdown()
    text = RESULTS_FILE.read_text(encoding="utf-8")
    print(f"\nWrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"Updated {RESULTS_FILE}")
    print(text)
    return run_dir


def _format_latest_markdown() -> None:
    import mdformat

    mdformat.file(str(RESULTS_FILE), extensions={"gfm"})


def parse_ops(raw: list[str] | None) -> tuple[OpName, ...]:
    if not raw:
        return DEFAULT_OPS
    out: list[OpName] = []
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            if part in OP_GROUPS:
                out.extend(OP_GROUPS[part])
                continue
            if part in OP_REGISTRY:
                out.append(part)
            else:
                groups = ", ".join(sorted(OP_GROUPS))
                known = ", ".join(OP_REGISTRY)
                raise ValueError(
                    f"unknown op {part!r}; choose from {known} or groups {groups}"
                )
    # preserve order, dedupe
    seen: set[str] = set()
    deduped: list[OpName] = []
    for op in out:
        if op not in seen:
            seen.add(op)
            deduped.append(op)
    return tuple(deduped)

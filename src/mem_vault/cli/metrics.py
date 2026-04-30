"""``mem-vault metrics`` — read the JSONL metrics sink and aggregate it.

The sink (``mem_vault.metrics.MetricsSink``) writes one JSON line per
tool call to ``<state_dir>/metrics.jsonl``. This subcommand reads that
file and produces a human-readable summary: per-tool count, error rate,
p50/p95/p99 duration, plus the slowest individual calls.

Why this exists: until now the sink was write-only — useful for piping
into external tools, but the user couldn't see their own perf data
without leaving mem-vault. Now ``mem-vault metrics`` answers "is
``memory_search`` getting slower?" or "which tool errors most?"
directly.

Filters:
- ``--since 7d`` / ``--since 24h`` / ``--since 30m`` / ``--since 2026-04-01T00:00``
- ``--tool memory_search`` (repeatable)
- ``--ok-only`` / ``--errors-only``

Output:
- Default: human table (per-tool counts + p50/p95/p99 + error rate)
- ``--json``: machine-readable summary

The metrics file is **append-only and process-safe**: reading it while
a long-lived MCP server is writing is fine — we open with a fresh fd,
read everything that's already flushed, and close. No locks needed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from mem_vault.config import load_config


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "metrics",
        help="Aggregate the JSONL metrics sink (per-tool p50/p95 + error rate).",
        description=(
            "Read <state_dir>/metrics.jsonl and print per-tool counts, "
            "error rate, p50/p95/p99 duration, and the slowest individual "
            "calls. The sink itself must be enabled (set MEM_VAULT_METRICS=1 "
            "or `metrics_enabled = true` in config)."
        ),
    )
    p.add_argument(
        "--since",
        default=None,
        help=(
            "Only include calls newer than this. Accepts shorthands like "
            "'24h', '7d', '30m', '2w', or an ISO-8601 timestamp. Default: "
            "all rows."
        ),
    )
    p.add_argument(
        "--tool",
        action="append",
        default=None,
        help=(
            "Filter to a specific tool name (repeatable). Without this flag every tool is reported."
        ),
    )
    p.add_argument(
        "--errors-only",
        action="store_true",
        help="Only aggregate calls where ok=false.",
    )
    p.add_argument(
        "--ok-only",
        action="store_true",
        help="Only aggregate calls where ok=true.",
    )
    p.add_argument(
        "--top-slow",
        type=int,
        default=5,
        help="How many slowest individual calls to surface (default: 5).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="json_out",
        help="Emit machine-readable JSON instead of the human table.",
    )
    p.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Override the metrics file location (defaults to <state_dir>/metrics.jsonl).",
    )


# ---------------------------------------------------------------------------
# Pure helpers (testable without disk / config)
# ---------------------------------------------------------------------------

# Why a hand-rolled regex instead of e.g. ``humanize.parse_time``: ``humanize``
# isn't a project dep and we want zero additional install surface for a CLI
# that's "just nice to have". The 4-letter alphabet (m / h / d / w) covers
# every shorthand a user would expect; longer windows can be expressed in
# days or via an explicit ISO timestamp.
_SHORTHAND_RE = re.compile(r"^(\d+)\s*([mhdw])$")
_SHORTHAND_TO_DELTA = {
    "m": "minutes",
    "h": "hours",
    "d": "days",
    "w": "weeks",
}


def parse_since(s: str | None) -> datetime | None:
    """Resolve a ``--since`` value to a tz-aware datetime cutoff.

    Accepts:
    - ``None`` → no cutoff (returns ``None``)
    - ``"7d"`` / ``"24h"`` / ``"30m"`` / ``"2w"`` → relative to "now"
    - ``"2026-04-01T00:00:00"`` → ISO-8601 absolute timestamp
    """
    if not s:
        return None
    m = _SHORTHAND_RE.match(s.strip())
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = timedelta(**{_SHORTHAND_TO_DELTA[unit]: n})
        return datetime.now().astimezone() - delta
    # Try ISO-8601. ``fromisoformat`` accepts both naive and aware; we
    # promote naive to local time so comparisons against the line's tz-aware
    # timestamps don't raise.
    parsed = datetime.fromisoformat(s)
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


def iter_lines(path: Path) -> Iterable[dict[str, Any]]:
    """Yield each parsed JSON line. Bad lines are skipped with a warning."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for n, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                # Don't kill the whole report over a single corrupt line —
                # one truncated write (process kill mid-flush) shouldn't
                # mask 30 days of clean data.
                print(
                    f"  ⚠ {path}:{n}: skipping malformed line ({exc})",
                    file=sys.stderr,
                )


def filter_lines(
    lines: Iterable[dict[str, Any]],
    *,
    since: datetime | None = None,
    tools: list[str] | None = None,
    errors_only: bool = False,
    ok_only: bool = False,
) -> list[dict[str, Any]]:
    """Apply the CLI filters to an iterable of metric lines.

    Pure function: takes any iterable, returns a list, no I/O.
    """
    if errors_only and ok_only:
        raise ValueError("--errors-only and --ok-only are mutually exclusive")
    out: list[dict[str, Any]] = []
    tool_filter = set(tools) if tools else None
    for line in lines:
        if tool_filter and line.get("tool") not in tool_filter:
            continue
        if errors_only and line.get("ok") is True:
            continue
        if ok_only and line.get("ok") is False:
            continue
        if since is not None:
            ts_str = line.get("ts")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.astimezone()
            except (TypeError, ValueError):
                continue
            if ts < since:
                continue
        out.append(line)
    return out


def percentile(values: list[float], pct: float) -> float | None:
    """Return the given percentile of a sorted-or-unsorted list (0-100).

    Uses the linear-interpolation method (numpy default), which is the
    one most "p95" articles describe. Returns ``None`` if the input is
    empty (no data → no number).
    """
    if not values:
        return None
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    sorted_v = sorted(values)
    rank = (pct / 100.0) * (len(sorted_v) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_v) - 1)
    frac = rank - lower
    return sorted_v[lower] + frac * (sorted_v[upper] - sorted_v[lower])


def aggregate(lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up filtered lines into a per-tool summary + a global row.

    Returns a dict shaped like::

        {
            "total": 123,
            "errors": 4,
            "by_tool": {
                "memory_search": {
                    "count": 80, "errors": 2, "error_rate": 0.025,
                    "p50": 142.0, "p95": 410.7, "p99": 612.3,
                    "min": 30.1, "max": 712.0,
                },
                ...
            },
        }
    """
    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in lines:
        tool = line.get("tool")
        if not tool:
            continue
        by_tool[tool].append(line)

    summary: dict[str, Any] = {
        "total": len(lines),
        "errors": sum(1 for line in lines if line.get("ok") is False),
        "by_tool": {},
    }
    for tool, rows in sorted(by_tool.items()):
        durations = [
            float(line["duration_ms"])
            for line in rows
            if isinstance(line.get("duration_ms"), int | float)
        ]
        errors = sum(1 for line in rows if line.get("ok") is False)
        summary["by_tool"][tool] = {
            "count": len(rows),
            "errors": errors,
            "error_rate": (errors / len(rows)) if rows else 0.0,
            "p50": percentile(durations, 50),
            "p95": percentile(durations, 95),
            "p99": percentile(durations, 99),
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
        }
    return summary


def top_slow_calls(lines: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Return the ``k`` slowest individual calls, newest-first as tie-break."""
    with_dur = [line for line in lines if isinstance(line.get("duration_ms"), int | float)]
    return sorted(
        with_dur,
        key=lambda line: (-float(line["duration_ms"]), line.get("ts", "")),
    )[:k]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_ms(v: float | None) -> str:
    """Compact format for ms values: '142' / '1.2s' / '—' for None."""
    if v is None:
        return "—"
    if v >= 1000:
        return f"{v / 1000:.2f}s"
    return f"{v:.0f}"


def _render_human(summary: dict[str, Any], top_slow: list[dict[str, Any]]) -> str:
    if summary["total"] == 0:
        return (
            "no metrics rows after filters\n"
            "(set MEM_VAULT_METRICS=1 to enable the sink — or check "
            "your --since / --tool flags)"
        )
    lines: list[str] = []
    lines.append("")
    lines.append("mem-vault metrics")
    lines.append("─" * 67)
    lines.append(
        f"  total: {summary['total']}    errors: {summary['errors']}"
        f"    error rate: {(summary['errors'] / summary['total']) * 100:5.2f}%"
    )
    lines.append("")
    lines.append(f"  {'tool':<26} {'count':>6}  {'err':>4}  {'p50':>6} {'p95':>7} {'p99':>7}")
    lines.append("  " + "─" * 65)
    for tool, row in summary["by_tool"].items():
        err_marker = ""
        if row["error_rate"] >= 0.1:
            err_marker = "!"
        lines.append(
            f"  {tool:<26} {row['count']:>6}  "
            f"{row['errors']:>3}{err_marker:<1}  "
            f"{_fmt_ms(row['p50']):>6} {_fmt_ms(row['p95']):>7} "
            f"{_fmt_ms(row['p99']):>7}"
        )
    if top_slow:
        lines.append("")
        lines.append("  slowest calls")
        lines.append("  " + "─" * 65)
        for r in top_slow:
            mark = "✗" if r.get("ok") is False else " "
            ts = (r.get("ts") or "")[:19]  # drop timezone for compactness
            lines.append(
                f"  {mark} {ts}  {r.get('tool', '?'):<26}  {_fmt_ms(float(r['duration_ms'])):>7}"
            )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    if args.path is not None:
        path = Path(args.path)
    else:
        cfg = load_config()
        path = Path(cfg.state_dir) / "metrics.jsonl"
    since = parse_since(args.since)
    rows = filter_lines(
        iter_lines(path),
        since=since,
        tools=args.tool,
        errors_only=args.errors_only,
        ok_only=args.ok_only,
    )
    summary = aggregate(rows)
    slow = top_slow_calls(rows, args.top_slow)
    if args.json_out:
        out = {**summary, "slowest": slow, "path": str(path)}
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        if not path.exists():
            print(
                f"metrics file does not exist yet: {path}\n"
                "(set MEM_VAULT_METRICS=1 to enable the sink, then call "
                "any tool to populate it)",
                file=sys.stderr,
            )
            return 0
        print(_render_human(summary, slow))
    return 0

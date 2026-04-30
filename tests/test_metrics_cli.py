"""Tests for the ``mem-vault metrics`` aggregator CLI.

The pure helpers (``parse_since``, ``filter_lines``, ``percentile``,
``aggregate``, ``top_slow_calls``) are exercised here without disk or
config — they take any iterable in and return data structures out, so
the test surface stays focused on the math, not on filesystem layout.

The end-to-end paths (``iter_lines`` reading a real ``metrics.jsonl``
and the ``run()`` dispatcher) get one happy-path test each.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from mem_vault.cli import metrics as metrics_mod

# ---------------------------------------------------------------------------
# parse_since
# ---------------------------------------------------------------------------


def test_parse_since_returns_none_for_none_or_empty():
    assert metrics_mod.parse_since(None) is None
    assert metrics_mod.parse_since("") is None


def test_parse_since_shorthand_minutes():
    cutoff = metrics_mod.parse_since("30m")
    assert cutoff is not None
    delta = datetime.now().astimezone() - cutoff
    # Loose bounds: the actual delta should be ~30 min ± some seconds for
    # whatever the test machine took between calls.
    assert timedelta(minutes=29) <= delta <= timedelta(minutes=31)


def test_parse_since_shorthand_hours_days_weeks():
    now = datetime.now().astimezone()
    assert (now - metrics_mod.parse_since("2h")) >= timedelta(hours=1, minutes=59)
    assert (now - metrics_mod.parse_since("3d")) >= timedelta(days=2, hours=23)
    assert (now - metrics_mod.parse_since("1w")) >= timedelta(days=6, hours=23)


def test_parse_since_iso_aware():
    iso = "2026-04-01T12:00:00+00:00"
    cutoff = metrics_mod.parse_since(iso)
    assert cutoff is not None
    assert cutoff.tzinfo is not None
    assert cutoff.year == 2026 and cutoff.month == 4


def test_parse_since_iso_naive_promoted_to_local():
    """Naive ISO must come back tz-aware so comparisons don't crash."""
    cutoff = metrics_mod.parse_since("2026-04-01T12:00:00")
    assert cutoff is not None
    assert cutoff.tzinfo is not None


def test_parse_since_invalid_raises():
    with pytest.raises(ValueError):
        metrics_mod.parse_since("not a date")


# ---------------------------------------------------------------------------
# percentile
# ---------------------------------------------------------------------------


def test_percentile_empty_returns_none():
    assert metrics_mod.percentile([], 50) is None


def test_percentile_p50_matches_median_for_odd_count():
    assert metrics_mod.percentile([1, 2, 3], 50) == 2


def test_percentile_p50_interpolated_for_even_count():
    # Standard linear interpolation: between 2 and 3 → 2.5
    assert metrics_mod.percentile([1, 2, 3, 4], 50) == 2.5


def test_percentile_extremes_are_min_and_max():
    assert metrics_mod.percentile([10, 20, 30], 0) == 10
    assert metrics_mod.percentile([10, 20, 30], 100) == 30


def test_percentile_does_not_mutate_input():
    """Caller passes a list; we sort a copy, never the original."""
    inp = [3, 1, 2]
    metrics_mod.percentile(inp, 50)
    assert inp == [3, 1, 2]


# ---------------------------------------------------------------------------
# filter_lines
# ---------------------------------------------------------------------------


def _line(tool="memory_search", duration_ms=100.0, ok=True, ts=None):
    if ts is None:
        ts = datetime.now().astimezone().isoformat(timespec="seconds")
    return {"ts": ts, "tool": tool, "duration_ms": duration_ms, "ok": ok}


def test_filter_lines_no_filters_returns_all():
    lines = [_line(), _line(), _line()]
    assert len(metrics_mod.filter_lines(lines)) == 3


def test_filter_lines_by_tool():
    lines = [
        _line(tool="memory_search"),
        _line(tool="memory_save"),
        _line(tool="memory_search"),
    ]
    out = metrics_mod.filter_lines(lines, tools=["memory_search"])
    assert len(out) == 2
    assert all(line["tool"] == "memory_search" for line in out)


def test_filter_lines_errors_only():
    lines = [_line(ok=True), _line(ok=False), _line(ok=True), _line(ok=False)]
    out = metrics_mod.filter_lines(lines, errors_only=True)
    assert len(out) == 2
    assert all(line["ok"] is False for line in out)


def test_filter_lines_ok_only():
    lines = [_line(ok=True), _line(ok=False), _line(ok=True)]
    out = metrics_mod.filter_lines(lines, ok_only=True)
    assert len(out) == 2
    assert all(line["ok"] is True for line in out)


def test_filter_lines_errors_only_and_ok_only_mutually_exclusive():
    with pytest.raises(ValueError):
        metrics_mod.filter_lines([_line()], errors_only=True, ok_only=True)


def test_filter_lines_since_drops_older_rows():
    now = datetime.now().astimezone()
    old_ts = (now - timedelta(days=10)).isoformat(timespec="seconds")
    new_ts = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    lines = [_line(ts=old_ts), _line(ts=new_ts)]
    cutoff = now - timedelta(days=1)
    out = metrics_mod.filter_lines(lines, since=cutoff)
    assert len(out) == 1
    assert out[0]["ts"] == new_ts


def test_filter_lines_skips_lines_with_unparseable_ts_when_filtering_by_since():
    """A row with no/invalid ts shouldn't crash the filter — drop and continue."""
    cutoff = datetime.now(UTC) - timedelta(days=1)
    lines = [
        {"tool": "memory_search", "duration_ms": 10, "ok": True, "ts": "garbage"},
        {"tool": "memory_search", "duration_ms": 10, "ok": True},  # no ts
        _line(),
    ]
    out = metrics_mod.filter_lines(lines, since=cutoff)
    # Only the one with a valid recent ts survives.
    assert len(out) == 1


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def test_aggregate_empty():
    s = metrics_mod.aggregate([])
    assert s == {"total": 0, "errors": 0, "by_tool": {}}


def test_aggregate_groups_by_tool_and_computes_percentiles():
    lines = [
        _line(tool="memory_search", duration_ms=100, ok=True),
        _line(tool="memory_search", duration_ms=200, ok=True),
        _line(tool="memory_search", duration_ms=300, ok=False),
        _line(tool="memory_save", duration_ms=50, ok=True),
    ]
    s = metrics_mod.aggregate(lines)
    assert s["total"] == 4
    assert s["errors"] == 1
    assert set(s["by_tool"].keys()) == {"memory_search", "memory_save"}

    search = s["by_tool"]["memory_search"]
    assert search["count"] == 3
    assert search["errors"] == 1
    assert search["error_rate"] == pytest.approx(1 / 3)
    assert search["min"] == 100
    assert search["max"] == 300
    # p50 of [100, 200, 300] = 200
    assert search["p50"] == 200

    save = s["by_tool"]["memory_save"]
    assert save["count"] == 1
    assert save["errors"] == 0
    assert save["min"] == 50 and save["max"] == 50


def test_aggregate_skips_lines_without_tool_key():
    """A malformed row (no tool field) shouldn't end up in any bucket."""
    lines = [_line(), {"duration_ms": 10, "ok": True}]
    s = metrics_mod.aggregate(lines)
    assert s["total"] == 2  # global counts include malformed rows
    assert sum(t["count"] for t in s["by_tool"].values()) == 1


def test_aggregate_handles_missing_duration_field():
    """A row without ``duration_ms`` doesn't poison the percentile math."""
    lines = [
        _line(duration_ms=100),
        {"tool": "memory_search", "ok": True},  # no duration
    ]
    s = metrics_mod.aggregate(lines)
    assert s["by_tool"]["memory_search"]["count"] == 2
    # percentile drops the rowless one and computes over [100]
    assert s["by_tool"]["memory_search"]["p50"] == 100


# ---------------------------------------------------------------------------
# top_slow_calls
# ---------------------------------------------------------------------------


def test_top_slow_returns_k_slowest_descending():
    lines = [
        _line(duration_ms=10),
        _line(duration_ms=500),
        _line(duration_ms=200),
        _line(duration_ms=999),
    ]
    out = metrics_mod.top_slow_calls(lines, k=2)
    durations = [line["duration_ms"] for line in out]
    assert durations == [999, 500]


def test_top_slow_k_zero_returns_empty():
    lines = [_line(duration_ms=100)]
    assert metrics_mod.top_slow_calls(lines, k=0) == []


def test_top_slow_skips_lines_without_duration():
    lines = [
        _line(duration_ms=100),
        {"tool": "memory_search", "ok": True},  # no duration
    ]
    out = metrics_mod.top_slow_calls(lines, k=5)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# iter_lines + run() — happy path E2E (touches disk via tmp_path)
# ---------------------------------------------------------------------------


def test_iter_lines_skips_blank_and_malformed(tmp_path, capsys):
    p = tmp_path / "metrics.jsonl"
    p.write_text(
        json.dumps(_line(tool="memory_search")) + "\n"
        "\n"  # blank line
        "{not json}\n" + json.dumps(_line(tool="memory_save")) + "\n",
        encoding="utf-8",
    )
    rows = list(metrics_mod.iter_lines(p))
    assert len(rows) == 2
    # The malformed line shows up as a stderr warning — test pytest's capsys
    # to avoid noisy CI without losing the diagnostic.
    captured = capsys.readouterr()
    assert "malformed line" in captured.err


def test_iter_lines_missing_path_yields_nothing(tmp_path):
    rows = list(metrics_mod.iter_lines(tmp_path / "nope.jsonl"))
    assert rows == []


def test_run_human_output_e2e(tmp_path, capsys, monkeypatch):
    p = tmp_path / "metrics.jsonl"
    payload = [
        _line(tool="memory_search", duration_ms=100, ok=True),
        _line(tool="memory_search", duration_ms=300, ok=False),
        _line(tool="memory_save", duration_ms=50, ok=True),
    ]
    p.write_text("\n".join(json.dumps(line) for line in payload) + "\n", encoding="utf-8")

    import argparse as _ap

    args = _ap.Namespace(
        path=p,
        since=None,
        tool=None,
        errors_only=False,
        ok_only=False,
        top_slow=3,
        json_out=False,
    )
    rc = metrics_mod.run(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "memory_search" in out
    assert "memory_save" in out
    assert "p50" in out


def test_run_json_output_e2e(tmp_path, capsys):
    p = tmp_path / "metrics.jsonl"
    p.write_text(json.dumps(_line(duration_ms=42)) + "\n", encoding="utf-8")

    import argparse as _ap

    args = _ap.Namespace(
        path=p,
        since=None,
        tool=None,
        errors_only=False,
        ok_only=False,
        top_slow=1,
        json_out=True,
    )
    rc = metrics_mod.run(args)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 1
    assert "by_tool" in payload
    assert "slowest" in payload


def test_run_missing_file_returns_friendly_message(tmp_path, capsys):
    """Pointing at a missing path must NOT raise — print a hint and exit 0."""
    import argparse as _ap

    args = _ap.Namespace(
        path=tmp_path / "nope.jsonl",
        since=None,
        tool=None,
        errors_only=False,
        ok_only=False,
        top_slow=0,
        json_out=False,
    )
    rc = metrics_mod.run(args)
    assert rc == 0
    err = capsys.readouterr().err
    assert "does not exist" in err

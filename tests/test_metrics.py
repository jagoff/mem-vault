"""Tests for the JSONL metrics sink + the ``time_async_call`` wrapper.

We exercise the sink in isolation against a temp file, then verify the
wrapper produces correct ``ok`` / ``error`` payloads for the three
common cases (clean success, envelope-level failure, raised exception).
"""

from __future__ import annotations

import json

import pytest

from mem_vault.metrics import MetricsSink, time_async_call

# ---------------------------------------------------------------------------
# MetricsSink — file output + disabled short-circuit
# ---------------------------------------------------------------------------


def _read_lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_disabled_sink_writes_nothing(tmp_path):
    sink = MetricsSink(tmp_path / "metrics.jsonl", enabled=False)
    sink.record(tool="memory_save", duration_ms=12.3, ok=True)
    assert not (tmp_path / "metrics.jsonl").exists()


def test_enabled_sink_writes_one_line_per_record(tmp_path):
    path = tmp_path / "metrics.jsonl"
    sink = MetricsSink(path, enabled=True)
    sink.record(tool="memory_save", duration_ms=12.3, ok=True)
    sink.record(tool="memory_search", duration_ms=200.0, ok=True)
    sink.close()

    lines = _read_lines(path)
    assert len(lines) == 2
    assert lines[0]["tool"] == "memory_save"
    assert lines[0]["duration_ms"] == 12.3
    assert lines[0]["ok"] is True
    assert "ts" in lines[0]
    assert lines[1]["tool"] == "memory_search"


def test_sink_appends_across_multiple_runs(tmp_path):
    """Each run on the same path should append, never overwrite."""
    path = tmp_path / "metrics.jsonl"

    s1 = MetricsSink(path, enabled=True)
    s1.record(tool="a", duration_ms=1.0, ok=True)
    s1.close()

    s2 = MetricsSink(path, enabled=True)
    s2.record(tool="b", duration_ms=2.0, ok=True)
    s2.close()

    lines = _read_lines(path)
    assert [line["tool"] for line in lines] == ["a", "b"]


def test_sink_records_error_field_when_present(tmp_path):
    path = tmp_path / "metrics.jsonl"
    sink = MetricsSink(path, enabled=True)
    sink.record(tool="memory_get", duration_ms=5.0, ok=False, error="not found")
    sink.close()

    lines = _read_lines(path)
    assert lines[0]["ok"] is False
    assert lines[0]["error"] == "not found"


def test_sink_extra_fields_merged_but_cannot_clobber_standards(tmp_path):
    path = tmp_path / "metrics.jsonl"
    sink = MetricsSink(path, enabled=True)
    sink.record(
        tool="memory_save",
        duration_ms=10.0,
        ok=True,
        extra={"index_entries": 3, "tool": "WRONG", "ok": "should-not-leak"},
    )
    sink.close()

    line = _read_lines(path)[0]
    # User extra is merged.
    assert line["index_entries"] == 3
    # But standard keys are protected.
    assert line["tool"] == "memory_save"
    assert line["ok"] is True


def test_sink_creates_parent_directory(tmp_path):
    path = tmp_path / "deep" / "nested" / "metrics.jsonl"
    sink = MetricsSink(path, enabled=True)
    sink.record(tool="x", duration_ms=1.0, ok=True)
    sink.close()
    assert path.exists()


def test_sink_enters_cooldown_on_io_error_then_drops_writes(tmp_path):
    """A failed write trips the cooldown; subsequent writes drop silently."""

    sink = MetricsSink(tmp_path / "metrics.jsonl", enabled=True, recovery_window_s=300.0)

    # Drive the sink's internal clock manually so we don't sleep in tests.
    fake_now = [1000.0]
    sink._now = lambda: fake_now[0]

    open_calls = {"n": 0}

    class _Exploder:
        def open(self, *args, **kwargs):
            open_calls["n"] += 1
            raise OSError("disk full")

        @property
        def parent(self):
            class _NoOp:
                def mkdir(self, **kwargs):
                    pass

            return _NoOp()

    sink.path = _Exploder()  # type: ignore[assignment]

    # First write fails → counter ticks, cooldown opens.
    sink.record(tool="x", duration_ms=1.0, ok=True)
    assert sink.enabled is True  # NOT permanently disabled anymore
    assert sink.failure_count == 1
    assert sink.last_failure_ts == fake_now[0]
    assert open_calls["n"] == 1

    # The next 5 writes inside the cooldown window must be dropped
    # silently — no FS attempt, no extra failure ticks, no log spam.
    for _ in range(5):
        sink.record(tool="x", duration_ms=1.0, ok=True)
    assert sink.failure_count == 1, "cooldown drops shouldn't increment failure_count"
    assert open_calls["n"] == 1, "no further open() attempts during cooldown"


def test_sink_retries_after_recovery_window_and_resets_on_success(tmp_path):
    """Once the cooldown window passes, the sink retries the write.

    On a successful retry the failure counter resets to 0 and the sink
    keeps writing as if the transient blip never happened.
    """
    real_path = tmp_path / "metrics.jsonl"
    sink = MetricsSink(real_path, enabled=True, recovery_window_s=300.0)

    fake_now = [1000.0]
    sink._now = lambda: fake_now[0]

    # Stage 1: first write fails (FS is "broken").
    class _Exploder:
        def open(self, *args, **kwargs):
            raise OSError("disk full")

        @property
        def parent(self):
            class _NoOp:
                def mkdir(self, **kwargs):
                    pass

            return _NoOp()

    sink.path = _Exploder()  # type: ignore[assignment]
    sink.record(tool="x", duration_ms=1.0, ok=True)
    assert sink.failure_count == 1
    assert sink.last_failure_ts == 1000.0

    # Stage 2: still inside the cooldown — even though we "fix" the FS,
    # we don't try yet (cheap drop).
    sink.path = real_path  # FS recovered
    fake_now[0] = 1000.0 + 100.0  # only 100 s passed, < 300 s window
    sink.record(tool="x", duration_ms=1.0, ok=True)
    assert not real_path.exists(), "must not write while still in cooldown"
    assert sink.failure_count == 1

    # Stage 3: window elapsed → retry. This one succeeds → counter
    # resets to 0, last_failure_ts cleared, file gets the line.
    fake_now[0] = 1000.0 + 301.0
    sink.record(tool="x", duration_ms=2.5, ok=True)
    sink.close()

    assert real_path.exists()
    lines = _read_lines(real_path)
    assert len(lines) == 1
    assert lines[0]["tool"] == "x"
    assert sink.failure_count == 0
    assert sink.last_failure_ts is None


def test_sink_repeated_failures_extend_cooldown(tmp_path):
    """If retry also fails, ``failure_count`` keeps climbing and the
    cooldown timer restarts from the latest failure."""

    sink = MetricsSink(tmp_path / "metrics.jsonl", enabled=True, recovery_window_s=300.0)
    fake_now = [1000.0]
    sink._now = lambda: fake_now[0]

    class _Exploder:
        def open(self, *args, **kwargs):
            raise OSError("disk still full")

        @property
        def parent(self):
            class _NoOp:
                def mkdir(self, **kwargs):
                    pass

            return _NoOp()

    sink.path = _Exploder()  # type: ignore[assignment]

    sink.record(tool="x", duration_ms=1.0, ok=True)
    assert sink.failure_count == 1
    first_ts = sink.last_failure_ts

    # Window elapsed — sink retries, retry also fails.
    fake_now[0] = 1000.0 + 400.0
    sink.record(tool="x", duration_ms=1.0, ok=True)
    assert sink.failure_count == 2
    assert sink.last_failure_ts == 1400.0
    assert sink.last_failure_ts != first_ts


# ---------------------------------------------------------------------------
# time_async_call — wraps a coroutine and records duration + ok
# ---------------------------------------------------------------------------


async def test_time_async_call_records_clean_success(tmp_path):
    sink = MetricsSink(tmp_path / "metrics.jsonl", enabled=True)

    async def _fn():
        return {"ok": True, "value": 42}

    result = await time_async_call(sink, "memory_search", _fn)
    sink.close()

    assert result == {"ok": True, "value": 42}
    line = _read_lines(tmp_path / "metrics.jsonl")[0]
    assert line["tool"] == "memory_search"
    assert line["ok"] is True
    assert line.get("error") is None
    assert line["duration_ms"] >= 0


async def test_time_async_call_records_envelope_failure(tmp_path):
    """Service returning ``ok: false`` should land in the JSONL as ok:false."""
    sink = MetricsSink(tmp_path / "metrics.jsonl", enabled=True)

    async def _fn():
        return {"ok": False, "error": "memory not found"}

    result = await time_async_call(sink, "memory_get", _fn)
    sink.close()

    assert result["ok"] is False
    line = _read_lines(tmp_path / "metrics.jsonl")[0]
    assert line["ok"] is False
    assert line["error"] == "memory not found"


async def test_time_async_call_records_raised_exception_then_reraises(tmp_path):
    sink = MetricsSink(tmp_path / "metrics.jsonl", enabled=True)

    async def _fn():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await time_async_call(sink, "memory_save", _fn)
    sink.close()

    line = _read_lines(tmp_path / "metrics.jsonl")[0]
    assert line["ok"] is False
    assert "ValueError" in line["error"]
    assert "boom" in line["error"]


async def test_time_async_call_disabled_sink_is_a_passthrough(tmp_path):
    sink = MetricsSink(tmp_path / "metrics.jsonl", enabled=False)

    async def _fn():
        return {"ok": True}

    result = await time_async_call(sink, "memory_list", _fn)
    assert result == {"ok": True}
    assert not (tmp_path / "metrics.jsonl").exists()

"""Append-only JSONL metrics sink for MCP tool calls.

We record one structured line per tool invocation. The file lives under
``<state_dir>/metrics.jsonl`` by default and is opened in append mode the
first time something is recorded. Each line shape:

```json
{"ts": "2026-04-29T20:55:01-03:00", "tool": "memory_search",
 "duration_ms": 234.7, "ok": true}
```

Failed calls add an ``error`` field (``"<ExceptionClass>: message"``) and
``ok: false``. The raw ``ok`` value is taken from the service envelope when
the handler returned cleanly, so a handler that returns
``{"ok": false, "error": "..."}`` (e.g. content-too-large) is recorded as
``ok: false`` even though no exception bubbled up.

Why JSONL: tail-friendly with ``tail -f``, append-safe across processes,
zero schema migration cost. A future ``mem-vault metrics summary`` could
aggregate it; for now the file is the analytical primitive.

Default: **disabled**. Set ``MEM_VAULT_METRICS=1`` or
``Config.metrics_enabled=True`` to opt in. Overhead when disabled is one
attribute lookup per call.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)


class MetricsSink:
    """Thread-safe writer for ``metrics.jsonl``. Re-uses the file handle.

    The sink is cheap to construct — the underlying file isn't opened
    until ``record()`` is called for the first time, so we can attach it
    unconditionally and toggle ``enabled`` at runtime without paying the
    open-time cost when off.

    **Transient FS failures are forgiven.** The previous behaviour was to
    flip ``enabled = False`` permanently on any write error, which means
    a brief iCloud blip / disk-full / permission glitch silently killed
    metrics until process restart. The current behaviour:

    - On a write error we increment ``failure_count`` and remember
      ``last_failure_ts``. While inside the cooldown window
      (``recovery_window`` seconds, default 300 s) further ``record()``
      calls return immediately without touching the FS — we don't want
      to hammer a stressed filesystem on every tool call.
    - Once the window has elapsed since ``last_failure_ts``, the next
      ``record()`` call retries the write. On success the failure
      counter resets and the sink is fully healthy again. On another
      failure the cooldown timer simply restarts.
    - The sink's ``failure_count`` is in-memory only — process restart
      always begins healthy.
    """

    def __init__(
        self,
        path: Path,
        *,
        enabled: bool = True,
        recovery_window_s: float = 300.0,
    ) -> None:
        self.path: Path = Path(path)
        self.enabled = enabled
        self.recovery_window_s = recovery_window_s
        self.failure_count = 0
        self.last_failure_ts: float | None = None
        self._fh: IO[str] | None = None
        self._lock = threading.Lock()
        # Indirection so tests can monkeypatch a deterministic clock
        # without having to mutate ``time.monotonic`` at module level.
        self._now: Callable[[], float] = time.monotonic

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None

    def _in_cooldown(self) -> bool:
        """Are we currently sitting out a transient-failure window?"""
        if self.last_failure_ts is None:
            return False
        return (self._now() - self.last_failure_ts) < self.recovery_window_s

    def record(
        self,
        *,
        tool: str,
        duration_ms: float,
        ok: bool,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append one line to the JSONL file. Best-effort: never raises.

        ``extra`` is a free-form dict merged into the line (useful for
        per-tool counters like ``index_entries``). Keys conflicting with
        the standard fields are dropped — we don't let user payload
        clobber ``ts`` / ``tool`` / ``duration_ms`` / ``ok`` / ``error``.
        """
        if not self.enabled:
            return
        if self._in_cooldown():
            # Drop silently — we already logged the underlying failure
            # once, no point re-spamming the logger every 50 ms while
            # the FS is unhappy.
            return
        line: dict[str, Any] = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "tool": tool,
            "duration_ms": round(duration_ms, 2),
            "ok": bool(ok),
        }
        if error:
            line["error"] = error
        if extra:
            for k, v in extra.items():
                if k not in line:
                    line[k] = v
        try:
            with self._lock:
                if self._fh is None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self._fh = self.path.open("a", encoding="utf-8")
                self._fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                self._fh.flush()
            # Healthy write → forget any prior transient failure.
            if self.failure_count:
                logger.info("metrics sink recovered after %d failure(s)", self.failure_count)
            self.failure_count = 0
            self.last_failure_ts = None
        except Exception as exc:
            # Metrics failure must never break the tool call. We bump
            # the counter, drop the file handle (it may be poisoned —
            # e.g. underlying file removed by user), and stay quiet for
            # the next ``recovery_window_s``. After that we'll retry
            # automatically on the next ``record()`` call.
            self.failure_count += 1
            self.last_failure_ts = self._now()
            with self._lock:
                if self._fh is not None:
                    try:
                        self._fh.close()
                    except Exception:
                        pass
                    self._fh = None
            logger.warning(
                "metrics sink write failed (%s); cooling down for %.0fs (failure #%d)",
                exc,
                self.recovery_window_s,
                self.failure_count,
            )


async def time_async_call(
    sink: MetricsSink,
    tool: str,
    fn: Callable[[], Awaitable[Any]],
) -> Any:
    """Run ``fn`` and record a metrics line for the call.

    Designed to be called from the MCP ``call_tool`` wrapper. The
    function returns whatever ``fn`` produced; the sink picks up
    ``payload["ok"]`` if it's a dict (typical service envelope) and
    falls back to ``True`` for any other return shape.
    """
    started = time.monotonic()
    error: str | None = None
    ok = True
    try:
        payload = await fn()
        if isinstance(payload, dict) and "ok" in payload:
            ok = bool(payload.get("ok"))
            if not ok and "error" in payload:
                error = str(payload["error"])
        return payload
    except Exception as exc:
        ok = False
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sink.record(
            tool=tool,
            duration_ms=(time.monotonic() - started) * 1000,
            ok=ok,
            error=error,
        )

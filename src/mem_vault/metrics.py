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
    """

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path: Path = Path(path)
        self.enabled = enabled
        self._fh: IO[str] | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None

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
        except Exception as exc:
            # Metrics failure must never break the tool call. Log once
            # and disable the sink to stop spamming the logger if the
            # filesystem stays unhappy.
            logger.warning("metrics sink write failed (%s); disabling sink", exc)
            self.enabled = False


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

"""Robustness tests for ``MemVaultService``: timeouts, oversized content, breaker.

These tests stub the ``VectorIndex`` so we don't need Ollama or Qdrant. The
storage layer runs against a real on-disk temp vault — that part is fast and
already deeply unit-tested in ``test_storage.py``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mem_vault.config import Config
from mem_vault.index import CircuitBreakerOpenError, _CircuitBreaker
from mem_vault.server import MemVaultService


class _StubIndex:
    """Drop-in replacement for ``VectorIndex`` that records calls and can fail on demand."""

    def __init__(self) -> None:
        self.add_calls: list[tuple[str, dict]] = []
        self.search_calls: list[tuple[str, dict]] = []
        self.delete_calls: list[tuple[str, str, str]] = []
        self.add_behavior: str = "ok"  # ok | hang | raise | breaker_open
        self._breaker = _CircuitBreaker(threshold=3, cooldown_s=30.0)

    @property
    def breaker(self) -> _CircuitBreaker:
        return self._breaker

    def add(self, content, **kwargs):
        self.add_calls.append((content, kwargs))
        if self.add_behavior == "hang":
            # Sleep way past the test's expected timeout so asyncio.wait_for trips.
            time.sleep(5.0)
            return []
        if self.add_behavior == "raise":
            raise RuntimeError("simulated mem0 failure")
        if self.add_behavior == "breaker_open":
            raise CircuitBreakerOpenError("simulated breaker open")
        return [{"id": "stub", "content": content}]

    def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return []

    def delete_by_metadata(self, key, value, user_id):
        self.delete_calls.append((key, value, user_id))
        return 0


@pytest.fixture
def service_factory(tmp_path: Path):
    """Build a ``MemVaultService`` with a real storage layer and a stub index."""

    def _make(**overrides) -> tuple[MemVaultService, _StubIndex]:
        cfg_kwargs = {
            "vault_path": str(tmp_path),
            "memory_subdir": "memory",
            "state_dir": str(tmp_path / "state"),
            "user_id": "tester",
            "auto_extract_default": False,
            "llm_timeout_s": 60.0,
            "max_content_size": 1_000_000,
            **overrides,
        }
        config = Config(**cfg_kwargs)
        config.qdrant_collection = "test"
        # Make sure the dirs exist (load_config normally does this).
        config.state_dir.mkdir(parents=True, exist_ok=True)
        config.memory_dir.mkdir(parents=True, exist_ok=True)
        service = MemVaultService(config)
        stub = _StubIndex()
        service.index = stub  # type: ignore[assignment]
        return service, stub

    return _make


# ---------------------------------------------------------------------------
# max_content_size
# ---------------------------------------------------------------------------


async def test_save_rejects_oversized_content(service_factory) -> None:
    service, stub = service_factory(max_content_size=100)
    res = await service.save({"content": "x" * 200, "type": "note"})
    assert res["ok"] is False
    assert res["code"] == "content_too_large"
    assert "200" in res["error"] and "100" in res["error"]
    assert stub.add_calls == [], "must reject before reaching the index"
    assert list(service.config.memory_dir.glob("*.md")) == [], "no .md should be written"


async def test_save_accepts_content_at_limit(service_factory) -> None:
    service, stub = service_factory(max_content_size=100)
    res = await service.save({"content": "x" * 100, "type": "note"})
    assert res["ok"] is True
    assert res["indexed"] is True
    assert len(stub.add_calls) == 1


async def test_update_rejects_oversized_content(service_factory) -> None:
    service, stub = service_factory(max_content_size=50)
    seed = await service.save({"content": "seed body", "type": "note"})
    mem_id = seed["memory"]["id"]

    res = await service.update({"id": mem_id, "content": "y" * 200})
    assert res["ok"] is False
    assert res["code"] == "content_too_large"


async def test_max_content_size_zero_disables_limit(service_factory) -> None:
    service, stub = service_factory(max_content_size=0)
    huge = "x" * 5_000_000
    res = await service.save({"content": huge, "type": "note"})
    assert res["ok"] is True
    assert res["indexed"] is True


# ---------------------------------------------------------------------------
# llm_timeout_s
# ---------------------------------------------------------------------------


async def test_save_returns_indexing_error_on_timeout(service_factory) -> None:
    service, stub = service_factory(llm_timeout_s=0.05)  # 50 ms
    stub.add_behavior = "hang"
    res = await service.save({"content": "anything", "type": "note"})
    # The .md file must still be on disk — only indexing degraded.
    assert res["ok"] is True
    assert res["indexed"] is False
    assert res["indexing_error_code"] == "llm_timeout"
    assert "timeout" in res["indexing_error"].lower()
    # The timeout should also tick the breaker.
    assert stub.breaker._consecutive_failures == 1


async def test_save_returns_breaker_error_when_breaker_already_open(service_factory) -> None:
    service, stub = service_factory(llm_timeout_s=10.0)
    stub.add_behavior = "breaker_open"
    res = await service.save({"content": "x", "type": "note"})
    assert res["ok"] is True
    assert res["indexed"] is False
    assert res["indexing_error_code"] == "circuit_breaker_open"


async def test_timeout_zero_disables_wrapper(service_factory) -> None:
    """``llm_timeout_s=0`` must skip ``asyncio.wait_for`` entirely (legacy mode)."""
    service, stub = service_factory(llm_timeout_s=0.0)
    res = await service.save({"content": "x", "type": "note"})
    assert res["ok"] is True
    assert res["indexed"] is True


async def test_search_returns_empty_on_timeout(service_factory) -> None:
    service, stub = service_factory(llm_timeout_s=0.05)

    def _hanging_search(query, **kwargs):
        time.sleep(5.0)
        return []

    stub.search = _hanging_search  # type: ignore[assignment]
    res = await service.search({"query": "anything"})
    assert res["ok"] is True
    assert res["count"] == 0
    assert "warning" in res
    assert "timeout" in res["warning"].lower()

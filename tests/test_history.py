"""Tests for memory edit history (JSONL sidecar + memory_history MCP tool).

Three layers:

1. ``VaultStorage._snapshot_to_history`` / ``read_history`` — the
   persistence primitives.
2. ``VaultStorage.update`` — only snapshots when something meaningful
   changed (no spurious history on no-op writes).
3. ``MemVaultService.history`` — the MCP tool envelope (validation,
   not_found, empty list vs missing sidecar).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mem_vault.config import Config
from mem_vault.index import _CircuitBreaker
from mem_vault.server import MemVaultService
from mem_vault.storage import VaultStorage

# ---------------------------------------------------------------------------
# VaultStorage — snapshot + read_history
# ---------------------------------------------------------------------------


def test_history_path_is_sibling_jsonl(tmp_path):
    storage = VaultStorage(tmp_path)
    path = storage.history_path_for("foo_bar")
    assert path.parent == tmp_path
    assert path.name == "foo_bar.history.jsonl"


def test_new_memory_has_no_history(tmp_path):
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="initial body", title="t")
    assert storage.read_history(mem.id) == []


def test_update_with_new_body_creates_snapshot(tmp_path):
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="v1 body", title="t", tags=["a"])
    storage.update(mem.id, content="v2 body")

    history = storage.read_history(mem.id)
    assert len(history) == 1
    entry = history[0]
    # Pre-update state: body was v1
    assert entry["body"] == "v1 body"
    assert entry["name"] == "t"
    assert entry["tags"] == ["a"]
    assert entry["reason"] == "update"
    assert entry["ts"]


def test_multiple_updates_produce_newest_first_history(tmp_path):
    import time

    storage = VaultStorage(tmp_path)
    mem = storage.save(content="v1", title="t")
    storage.update(mem.id, content="v2")
    time.sleep(0.01)  # distinct timestamp
    storage.update(mem.id, content="v3")

    history = storage.read_history(mem.id)
    assert len(history) == 2
    # Newest first: the snapshot BEFORE the last update had body="v2".
    assert history[0]["body"] == "v2"
    assert history[1]["body"] == "v1"


def test_update_with_no_meaningful_change_skips_snapshot(tmp_path):
    """Passing the same body/tags that are already set shouldn't leave a
    history entry — that's spurious noise."""
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="body", title="t", tags=["a"])
    # Identical update — no history.
    storage.update(mem.id, content="body", tags=["a"])
    assert storage.read_history(mem.id) == []


def test_update_with_only_related_change_snapshots(tmp_path):
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="body", title="t")
    storage.update(mem.id, related=["other_id"])
    history = storage.read_history(mem.id)
    assert len(history) == 1
    assert history[0]["related"] == []


def test_update_with_only_contradicts_change_snapshots(tmp_path):
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="body", title="t")
    storage.update(mem.id, contradicts=["x", "y"])
    history = storage.read_history(mem.id)
    assert len(history) == 1
    assert history[0]["contradicts"] == []


def test_read_history_limit_caps_entries(tmp_path):
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="v0", title="t")
    for i in range(1, 10):
        storage.update(mem.id, content=f"v{i}")
    entries = storage.read_history(mem.id, limit=3)
    assert len(entries) == 3


def test_delete_removes_history_sidecar(tmp_path):
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="v1", title="t")
    storage.update(mem.id, content="v2")
    assert storage.history_path_for(mem.id).exists()
    storage.delete(mem.id)
    assert not storage.history_path_for(mem.id).exists()


def test_read_history_handles_corrupt_lines_gracefully(tmp_path):
    """A partial line (crash mid-append) should be skipped, not raise."""
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="body", title="t")
    storage.update(mem.id, content="body2")
    # Append a garbage line
    path = storage.history_path_for(mem.id)
    with path.open("a", encoding="utf-8") as f:
        f.write("{not valid json\n")
    entries = storage.read_history(mem.id)
    # The legit entry still shows up, the garbage is dropped.
    assert len(entries) == 1
    assert entries[0]["body"] == "body"


def test_history_file_is_ignored_by_list(tmp_path):
    """The ``.history.jsonl`` sidecar must not show up as a pseudo-memory."""
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="body", title="t")
    storage.update(mem.id, content="body2")
    # list() globs *.md — sidecar should not count.
    memorias = storage.list(limit=100)
    assert len(memorias) == 1
    assert memorias[0].id == mem.id


# ---------------------------------------------------------------------------
# MemVaultService.history — MCP tool envelope
# ---------------------------------------------------------------------------


class _StubIndex:
    def __init__(self, hits=None):
        self.hits = hits or []
        self._breaker = _CircuitBreaker(threshold=3, cooldown_s=30.0)

    @property
    def breaker(self):
        return self._breaker

    def add(self, content, **kwargs):
        return [{"id": "stub"}]

    def search(self, query, **kwargs):
        return list(self.hits)

    def delete_by_metadata(self, *args):
        return 0


@pytest.fixture
def history_service(tmp_path: Path):
    def _make() -> MemVaultService:
        cfg = Config(
            vault_path=str(tmp_path),
            memory_subdir="memory",
            state_dir=str(tmp_path / "state"),
            user_id="tester",
            auto_extract_default=False,
            llm_timeout_s=0,
            max_content_size=0,
            auto_link_default=False,
        )
        cfg.qdrant_collection = "test"
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        cfg.memory_dir.mkdir(parents=True, exist_ok=True)
        service = MemVaultService(cfg)
        service.index = _StubIndex()
        return service

    return _make


async def test_history_tool_missing_id_returns_validation_error(history_service):
    service = history_service()
    res = await service.history({})
    assert res["ok"] is False
    assert res["code"] == "validation_failed"


async def test_history_tool_unknown_memory_returns_not_found(history_service):
    service = history_service()
    res = await service.history({"id": "nonexistent"})
    assert res["ok"] is False
    assert res["code"] == "not_found"


async def test_history_tool_memory_without_updates_returns_empty(history_service):
    service = history_service()
    saved = await service.save({"content": "body", "title": "t"})
    res = await service.history({"id": saved["memory"]["id"]})
    assert res["ok"] is True
    assert res["count"] == 0
    assert res["entries"] == []


async def test_history_tool_after_update_returns_snapshot(history_service):
    service = history_service()
    saved = await service.save({"content": "v1", "title": "t"})
    await service.update({"id": saved["memory"]["id"], "content": "v2"})
    res = await service.history({"id": saved["memory"]["id"]})
    assert res["count"] == 1
    assert res["entries"][0]["body"] == "v1"


async def test_history_tool_respects_limit(history_service):
    service = history_service()
    saved = await service.save({"content": "v0", "title": "t"})
    for i in range(1, 5):
        await service.update({"id": saved["memory"]["id"], "content": f"v{i}"})
    res = await service.history({"id": saved["memory"]["id"], "limit": 2})
    assert res["count"] == 2

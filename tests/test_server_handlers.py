"""Tests for the read-side ``MemVaultService`` handlers: search, list, get, delete.

The save/update timeout + content-size paths live in ``test_robustness.py``.
This file zeroes in on the visibility filtering, the not-found envelopes,
and the index cleanup behavior on delete.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mem_vault.config import Config
from mem_vault.index import _CircuitBreaker
from mem_vault.server import MemVaultService


class _StubIndex:
    """Configurable stub returning canned hits / canned delete counts."""

    def __init__(self) -> None:
        self.hits: list[dict] = []
        self.delete_log: list[tuple[str, str, str]] = []
        self.delete_count: int = 0
        self._breaker = _CircuitBreaker(threshold=3, cooldown_s=30.0)

    @property
    def breaker(self) -> _CircuitBreaker:
        return self._breaker

    def add(self, *args, **kwargs):
        return [{"id": "stub"}]

    def search(self, query, **kwargs):
        return list(self.hits)

    def delete_by_metadata(self, key, value, user_id):
        self.delete_log.append((key, value, user_id))
        return self.delete_count


@pytest.fixture
def service_factory(tmp_path: Path):
    def _make(**overrides) -> tuple[MemVaultService, _StubIndex]:
        cfg_kwargs = {
            "vault_path": str(tmp_path),
            "memory_subdir": "memory",
            "state_dir": str(tmp_path / "state"),
            "user_id": "tester",
            "agent_id": None,
            "auto_extract_default": False,
            "llm_timeout_s": 0,  # disabled — tests focus on handler logic, not timeouts
            "max_content_size": 0,
            **overrides,
        }
        config = Config(**cfg_kwargs)
        config.qdrant_collection = "test"
        config.state_dir.mkdir(parents=True, exist_ok=True)
        config.memory_dir.mkdir(parents=True, exist_ok=True)
        service = MemVaultService(config)
        stub = _StubIndex()
        service.index = stub  # type: ignore[assignment]
        return service, stub

    return _make


# ---------------------------------------------------------------------------
# search — visibility filtering + memory body resolution
# ---------------------------------------------------------------------------


async def test_search_resolves_memory_body_from_storage(service_factory):
    service, stub = service_factory()
    seed = await service.save({"content": "Decision body", "title": "Decision A", "type": "fact"})
    mem_id = seed["memory"]["id"]
    stub.hits = [{"score": 0.9, "metadata": {"memory_id": mem_id}, "memory": "Decision body"}]

    res = await service.search({"query": "anything", "k": 3})
    assert res["ok"] is True
    assert res["count"] == 1
    assert res["results"][0]["id"] == mem_id
    assert res["results"][0]["memory"]["body"] == "Decision body"
    assert res["results"][0]["score"] == pytest.approx(0.9)


async def test_search_filters_private_memories_from_other_agents(service_factory):
    """A private memory owned by agent ``alice`` must not show up to ``bob``."""
    service, stub = service_factory(agent_id="alice")
    seed = await service.save(
        {
            "content": "Top secret",
            "type": "fact",
            "agent_id": "alice",
            "visible_to": "private",  # only alice can read
        }
    )
    mem_id = seed["memory"]["id"]
    stub.hits = [{"score": 0.9, "metadata": {"memory_id": mem_id}}]

    # alice sees it
    alice_res = await service.search({"query": "x", "viewer_agent_id": "alice"})
    assert alice_res["count"] == 1

    # bob doesn't
    bob_res = await service.search({"query": "x", "viewer_agent_id": "bob"})
    assert bob_res["count"] == 0


async def test_search_dedupes_repeated_memory_id(service_factory):
    service, stub = service_factory()
    seed = await service.save({"content": "x", "title": "x", "type": "note"})
    mem_id = seed["memory"]["id"]
    stub.hits = [
        {"score": 0.9, "metadata": {"memory_id": mem_id}},
        {"score": 0.85, "metadata": {"memory_id": mem_id}},  # duplicate
    ]
    res = await service.search({"query": "x", "k": 5})
    assert res["count"] == 1


async def test_search_respects_top_k(service_factory):
    service, stub = service_factory()
    ids = []
    for i in range(5):
        seed = await service.save({"content": f"body {i}", "title": f"t{i}", "type": "note"})
        ids.append(seed["memory"]["id"])
    stub.hits = [
        {"score": 0.9 - 0.05 * i, "metadata": {"memory_id": mid}} for i, mid in enumerate(ids)
    ]

    res = await service.search({"query": "x", "k": 2})
    assert res["count"] == 2


async def test_search_with_empty_index_returns_zero_results(service_factory):
    service, stub = service_factory()
    stub.hits = []
    res = await service.search({"query": "x"})
    assert res["ok"] is True
    assert res["count"] == 0
    assert res["results"] == []


# ---------------------------------------------------------------------------
# list — filters + visibility
# ---------------------------------------------------------------------------


async def test_list_filters_by_type(service_factory):
    service, _ = service_factory()
    await service.save({"content": "A", "title": "A", "type": "decision"})
    await service.save({"content": "B", "title": "B", "type": "fact"})
    await service.save({"content": "C", "title": "C", "type": "decision"})

    res = await service.list_({"type": "decision"})
    assert res["ok"] is True
    assert res["count"] == 2
    assert {m["type"] for m in res["memories"]} == {"decision"}


async def test_list_filters_by_tags(service_factory):
    service, _ = service_factory()
    await service.save({"content": "A", "title": "A", "type": "note", "tags": ["alpha", "beta"]})
    await service.save({"content": "B", "title": "B", "type": "note", "tags": ["beta"]})
    await service.save({"content": "C", "title": "C", "type": "note", "tags": ["alpha"]})

    res = await service.list_({"tags": ["alpha", "beta"]})
    # Only memory "A" has BOTH alpha and beta
    assert res["count"] == 1
    assert res["memories"][0]["name"] == "A"


async def test_list_respects_limit(service_factory):
    service, _ = service_factory()
    for i in range(8):
        await service.save({"content": f"m{i}", "title": f"m{i}", "type": "note"})
    res = await service.list_({"limit": 3})
    assert res["count"] == 3


async def test_list_filters_private_memories_for_other_viewer(service_factory):
    service, _ = service_factory(agent_id="alice")
    await service.save(
        {"content": "Public", "title": "Public", "type": "note", "visible_to": "public"}
    )
    await service.save(
        {
            "content": "Secret",
            "title": "Secret",
            "type": "note",
            "agent_id": "alice",
            "visible_to": "private",
        }
    )

    res_alice = await service.list_({"viewer_agent_id": "alice"})
    res_bob = await service.list_({"viewer_agent_id": "bob"})

    alice_names = {m["name"] for m in res_alice["memories"]}
    bob_names = {m["name"] for m in res_bob["memories"]}
    assert "Secret" in alice_names
    assert "Public" in alice_names
    assert "Secret" not in bob_names
    assert "Public" in bob_names


# ---------------------------------------------------------------------------
# get — round-trip + not-found envelope
# ---------------------------------------------------------------------------


async def test_get_returns_full_memory(service_factory):
    service, _ = service_factory()
    seed = await service.save(
        {"content": "Body content here", "title": "T", "type": "note", "tags": ["x"]}
    )
    mem_id = seed["memory"]["id"]

    res = await service.get({"id": mem_id})
    assert res["ok"] is True
    assert res["memory"]["id"] == mem_id
    assert res["memory"]["body"] == "Body content here"
    assert res["memory"]["tags"] == ["x"]


async def test_get_missing_returns_not_found(service_factory):
    service, _ = service_factory()
    res = await service.get({"id": "does-not-exist"})
    assert res["ok"] is False
    assert "not found" in res["error"].lower()


# ---------------------------------------------------------------------------
# delete — file removal + index cleanup, idempotency
# ---------------------------------------------------------------------------


async def test_delete_removes_file_and_calls_index_cleanup(service_factory):
    service, stub = service_factory()
    seed = await service.save({"content": "x", "title": "x", "type": "note"})
    mem_id = seed["memory"]["id"]
    stub.delete_count = 3

    res = await service.delete({"id": mem_id})
    assert res["ok"] is True
    assert res["deleted_file"] is True
    assert res["deleted_index_entries"] == 3
    # The .md should be gone
    assert not (service.config.memory_dir / f"{mem_id}.md").exists()
    # And the stub should have been called with the right metadata key/value
    assert stub.delete_log == [("memory_id", mem_id, service.config.user_id)]


async def test_delete_missing_returns_not_found(service_factory):
    service, stub = service_factory()
    res = await service.delete({"id": "ghost"})
    assert res["ok"] is False
    assert "not found" in res["error"].lower()
    # Index cleanup should NOT run when the file doesn't exist
    assert stub.delete_log == []


async def test_delete_handles_index_cleanup_failure_gracefully(service_factory):
    """If index.delete_by_metadata raises, the file still gets removed and ok stays True."""
    service, stub = service_factory()
    seed = await service.save({"content": "x", "title": "x", "type": "note"})
    mem_id = seed["memory"]["id"]

    def _boom(*args, **kwargs):
        raise RuntimeError("qdrant down")

    stub.delete_by_metadata = _boom  # type: ignore[assignment]

    res = await service.delete({"id": mem_id})
    assert res["ok"] is True
    assert res["deleted_file"] is True
    assert res["deleted_index_entries"] == 0
    assert not (service.config.memory_dir / f"{mem_id}.md").exists()


# ---------------------------------------------------------------------------
# update — re-index pipeline
# ---------------------------------------------------------------------------


async def test_update_reindexes_when_content_changes(service_factory):
    service, stub = service_factory()
    seed = await service.save({"content": "old body", "title": "T", "type": "note"})
    mem_id = seed["memory"]["id"]

    add_calls: list[tuple] = []
    original_add = stub.add

    def _record_add(content, **kwargs):
        add_calls.append((content, kwargs))
        return original_add(content, **kwargs)

    stub.add = _record_add  # type: ignore[assignment]

    res = await service.update({"id": mem_id, "content": "new body"})
    assert res["ok"] is True
    assert res["memory"]["body"] == "new body"
    # index.delete_by_metadata + index.add must have run
    assert any(call[0] == "memory_id" for call in stub.delete_log)
    assert add_calls and add_calls[0][0] == "new body"


async def test_update_without_content_does_not_reindex(service_factory):
    service, stub = service_factory()
    seed = await service.save({"content": "body", "title": "T", "type": "note"})
    mem_id = seed["memory"]["id"]
    stub.delete_log.clear()

    res = await service.update({"id": mem_id, "title": "new title only"})
    assert res["ok"] is True
    assert res["memory"]["name"] == "new title only"
    # No content change → no re-embed pass
    assert stub.delete_log == []


async def test_update_missing_id_returns_not_found(service_factory):
    service, _ = service_factory()
    res = await service.update({"id": "ghost", "content": "anything"})
    assert res["ok"] is False
    assert "not found" in res["error"].lower()

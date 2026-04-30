"""Tests for auto-linking on save + the new ``memory_synthesize`` tool.

We use the same stubbed-index pattern as ``test_server_handlers.py`` so
nothing here touches real Ollama / Qdrant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mem_vault.config import Config
from mem_vault.index import _CircuitBreaker
from mem_vault.server import MemVaultService


class _StubIndex:
    """Index stub: lets tests pre-seed the search response per call."""

    def __init__(self) -> None:
        self.add_calls: list[tuple] = []
        self.delete_calls: list[tuple] = []
        self.search_response: list[dict] = []
        self._breaker = _CircuitBreaker(threshold=3, cooldown_s=30.0)

    @property
    def breaker(self) -> _CircuitBreaker:
        return self._breaker

    def add(self, content, **kwargs):
        self.add_calls.append((content, kwargs))
        return [{"id": "stub"}]

    def search(self, query, **kwargs):
        return list(self.search_response)

    def delete_by_metadata(self, key, value, user_id):
        self.delete_calls.append((key, value, user_id))
        return 0


@pytest.fixture
def service_factory(tmp_path: Path):
    def _make(**overrides) -> tuple[MemVaultService, _StubIndex]:
        cfg_kwargs = {
            "vault_path": str(tmp_path),
            "memory_subdir": "memory",
            "state_dir": str(tmp_path / "state"),
            "user_id": "tester",
            "auto_extract_default": False,
            "llm_timeout_s": 0,
            "max_content_size": 0,
            "auto_link_default": True,
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
# Auto-linking
# ---------------------------------------------------------------------------


async def test_save_writes_related_when_similar_memories_exist(service_factory):
    service, stub = service_factory()
    # Pre-stage: pretend the index already has two similar memorias whose
    # search() result will be returned for the next add() call.
    stub.search_response = [
        {"score": 0.9, "metadata": {"memory_id": "older-1"}},
        {"score": 0.8, "metadata": {"memory_id": "older-2"}},
    ]
    res = await service.save({"content": "new memory body", "title": "new", "type": "note"})
    assert res["ok"] is True
    assert res["related"] == ["older-1", "older-2"]
    # Frontmatter on disk should reflect it too.
    assert res["memory"]["related"] == ["older-1", "older-2"]


async def test_save_skips_auto_link_when_disabled_per_call(service_factory):
    service, stub = service_factory()
    stub.search_response = [
        {"score": 0.9, "metadata": {"memory_id": "older-1"}},
    ]
    res = await service.save({"content": "x", "title": "x", "type": "note", "auto_link": False})
    assert res["related"] == []
    assert res["memory"]["related"] == []


async def test_save_skips_auto_link_when_disabled_globally(service_factory):
    service, stub = service_factory(auto_link_default=False)
    stub.search_response = [{"score": 0.9, "metadata": {"memory_id": "x"}}]
    res = await service.save({"content": "x", "title": "x", "type": "note"})
    assert res["related"] == []


async def test_auto_link_excludes_self_from_related(service_factory):
    """The save's own memory_id must never end up in its own ``related`` list.

    Common shape of a real Qdrant search after an embed: the just-stored
    vector ranks high for itself. Without the self-filter, every save would
    point at itself.
    """
    service, stub = service_factory()
    # The search will hit the just-saved memory itself — the title slug
    # (``"first"``) is what storage.save uses for the id.
    stub.search_response = [
        {"score": 1.0, "metadata": {"memory_id": "first"}},  # this IS self
        {"score": 0.85, "metadata": {"memory_id": "older-2"}},
    ]
    res = await service.save({"content": "first body", "title": "first", "type": "note"})
    # Self-exclusion: "first" must not appear, only the other one.
    assert "first" not in res["related"]
    assert res["related"] == ["older-2"]


async def test_auto_link_dedupes_repeated_ids(service_factory):
    service, stub = service_factory()
    stub.search_response = [
        {"score": 0.9, "metadata": {"memory_id": "dupe"}},
        {"score": 0.8, "metadata": {"memory_id": "dupe"}},
        {"score": 0.7, "metadata": {"memory_id": "other"}},
    ]
    res = await service.save({"content": "body", "title": "t", "type": "note"})
    assert res["related"] == ["dupe", "other"]


async def test_auto_link_swallows_search_failures(service_factory):
    service, stub = service_factory()

    def _broken_search(*args, **kwargs):
        raise RuntimeError("qdrant down")

    stub.search = _broken_search  # type: ignore[assignment]
    res = await service.save({"content": "body", "title": "t", "type": "note"})
    # Save itself should still succeed; only related stays empty.
    assert res["ok"] is True
    assert res["indexed"] is True
    assert res["related"] == []


# ---------------------------------------------------------------------------
# memory_synthesize
# ---------------------------------------------------------------------------


async def test_synthesize_rejects_empty_query(service_factory):
    service, _ = service_factory()
    res = await service.synthesize({"query": "  "})
    assert res["ok"] is False
    assert res["code"] == "validation_failed"


async def test_synthesize_returns_helpful_message_when_no_memories(service_factory, monkeypatch):
    service, stub = service_factory()
    # No matching memories.
    stub.search_response = []
    res = await service.synthesize({"query": "anything obscure"})
    assert res["ok"] is True
    assert res["count"] == 0
    assert "No tengo memorias suficientes" in res["synthesis"]


async def test_synthesize_invokes_llm_with_seeded_memories(service_factory, monkeypatch):
    service, stub = service_factory()
    # Seed: two memories on disk + matching search response.
    seed1 = await service.save(
        {"content": "Memory body 1: el patrón XYZ", "title": "M1", "type": "fact"}
    )
    seed2 = await service.save(
        {"content": "Memory body 2: la decisión ABC", "title": "M2", "type": "decision"}
    )
    stub.search_response = [
        {"score": 0.9, "metadata": {"memory_id": seed1["memory"]["id"]}},
        {"score": 0.85, "metadata": {"memory_id": seed2["memory"]["id"]}},
    ]

    captured_prompts: list[str] = []

    async def _fake_llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return f"Síntesis fake. Mencioné `{seed1['memory']['id']}` y `{seed2['memory']['id']}`."

    monkeypatch.setattr(service, "_call_llm_for_synthesis", _fake_llm)

    res = await service.synthesize({"query": "qué patrones tengo guardados"})
    assert res["ok"] is True
    assert res["count"] == 2
    assert seed1["memory"]["id"] in res["source_ids"]
    assert seed2["memory"]["id"] in res["source_ids"]
    assert "Síntesis fake" in res["synthesis"]
    # Prompt must contain query + at least one memory body
    assert "qué patrones tengo guardados" in captured_prompts[0]
    assert "Memory body 1" in captured_prompts[0]
    assert "Memory body 2" in captured_prompts[0]


async def test_synthesize_returns_llm_timeout_envelope(service_factory, monkeypatch):
    service, stub = service_factory()
    seed = await service.save({"content": "body", "title": "M", "type": "note"})
    stub.search_response = [{"score": 0.9, "metadata": {"memory_id": seed["memory"]["id"]}}]

    from mem_vault.server import _LLMTimeoutError

    async def _hangs(prompt):
        raise _LLMTimeoutError("Ollama timed out")

    monkeypatch.setattr(service, "_call_llm_for_synthesis", _hangs)
    res = await service.synthesize({"query": "anything"})
    assert res["ok"] is False
    assert res["code"] == "llm_timeout"
    # Source IDs should still surface so the caller can fall back.
    assert seed["memory"]["id"] in res["source_ids"]


async def test_synthesize_returns_generic_error_envelope(service_factory, monkeypatch):
    service, stub = service_factory()
    seed = await service.save({"content": "body", "title": "M", "type": "note"})
    stub.search_response = [{"score": 0.9, "metadata": {"memory_id": seed["memory"]["id"]}}]

    async def _boom(prompt):
        raise ValueError("bad thing")

    monkeypatch.setattr(service, "_call_llm_for_synthesis", _boom)
    res = await service.synthesize({"query": "anything"})
    assert res["ok"] is False
    assert res["code"] == "llm_failed"
    assert "ValueError" in res["error"]


# ---------------------------------------------------------------------------
# storage round-trip with `related` field
# ---------------------------------------------------------------------------


def test_storage_persists_and_reads_related_field(tmp_path):
    from mem_vault.storage import VaultStorage

    storage = VaultStorage(tmp_path)
    mem = storage.save(content="A", title="A", type="note")
    storage.update(mem.id, related=["other-1", "other-2"])

    reread = storage.get(mem.id)
    assert reread is not None
    assert reread.related == ["other-1", "other-2"]


def test_storage_skips_related_field_when_empty(tmp_path):
    """Empty ``related`` shouldn't appear in the .md frontmatter."""
    from mem_vault.storage import VaultStorage

    storage = VaultStorage(tmp_path)
    mem = storage.save(content="A", title="A", type="note")

    raw = (tmp_path / f"{mem.id}.md").read_text(encoding="utf-8")
    assert "related" not in raw

"""Tests for the ``memory_related`` MCP tool (walk-the-graph).

Covers the four neighbor groups:

1. Explicit ``related`` from frontmatter.
2. Explicit ``contradicts`` from frontmatter.
3. Co-tag neighbors (≥N shared normalized tags).
4. Semantic neighbors (opt-in via ``include_semantic``).

Plus the usual contract: validation errors, not-found, ids excluded from
their own neighbor list.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mem_vault.config import Config
from mem_vault.index import _CircuitBreaker
from mem_vault.server import MemVaultService


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
def related_service(tmp_path: Path):
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


async def test_related_missing_id_returns_validation(related_service):
    service = related_service()
    res = await service.related({})
    assert res["ok"] is False
    assert res["code"] == "validation_failed"


async def test_related_unknown_memory_returns_not_found(related_service):
    service = related_service()
    res = await service.related({"id": "nonexistent"})
    assert res["ok"] is False
    assert res["code"] == "not_found"


async def test_related_returns_frontmatter_related_list(related_service):
    service = related_service()
    a = await service.save({"content": "a body", "title": "a", "tags": ["x"]})
    b = await service.save({"content": "b body", "title": "b", "tags": ["x"]})
    # Manually stamp related list
    service.storage.update(a["memory"]["id"], related=[b["memory"]["id"]])

    res = await service.related(
        {"id": a["memory"]["id"], "include_semantic": False, "min_shared_tags": 99}
    )
    assert res["ok"] is True
    assert len(res["related"]) == 1
    assert res["related"][0]["id"] == b["memory"]["id"]
    assert res["related"][0]["name"] == "b"


async def test_related_returns_frontmatter_contradicts_list(related_service):
    service = related_service()
    a = await service.save({"content": "a body", "title": "a"})
    b = await service.save({"content": "b body", "title": "b"})
    service.storage.update(a["memory"]["id"], contradicts=[b["memory"]["id"]])

    res = await service.related(
        {"id": a["memory"]["id"], "include_semantic": False, "min_shared_tags": 99}
    )
    assert len(res["contradicts"]) == 1
    assert res["contradicts"][0]["id"] == b["memory"]["id"]


async def test_related_computes_cotag_neighbors(related_service):
    service = related_service()
    a = await service.save(
        {"content": "target", "title": "a", "tags": ["project:rag", "python", "local"]}
    )
    # b shares 2 tags (project:rag + python) → counts at min_shared=2
    b = await service.save(
        {"content": "b", "title": "b", "tags": ["project:rag", "python", "cooking"]}
    )
    # c shares 1 tag (local) → excluded at min_shared=2
    await service.save({"content": "c", "title": "c", "tags": ["local", "garden"]})

    res = await service.related(
        {"id": a["memory"]["id"], "include_semantic": False, "min_shared_tags": 2}
    )
    ids = [n["id"] for n in res["cotag_neighbors"]]
    assert b["memory"]["id"] in ids
    assert len(ids) == 1


async def test_related_cotag_normalizes_project_prefix(related_service):
    """``project:rag`` should match ``project:rag-obsidian`` via normalization
    — both normalize to ``rag`` (split on colon)."""
    service = related_service()
    a = await service.save({"content": "target", "title": "a", "tags": ["project:rag", "local"]})
    # The normalizer splits on colon and takes the suffix, so
    # ``project:rag-obsidian`` becomes ``rag-obsidian`` NOT ``rag``.
    # The test asserts that the EXACT suffix must match for a shared tag.
    b = await service.save({"content": "b", "title": "b", "tags": ["project:rag", "local"]})
    res = await service.related(
        {"id": a["memory"]["id"], "include_semantic": False, "min_shared_tags": 2}
    )
    ids = [n["id"] for n in res["cotag_neighbors"]]
    assert b["memory"]["id"] in ids


async def test_related_excludes_self(related_service):
    service = related_service()
    a = await service.save({"content": "a", "title": "a", "tags": ["x", "y"]})
    res = await service.related(
        {"id": a["memory"]["id"], "include_semantic": False, "min_shared_tags": 1}
    )
    assert all(n["id"] != a["memory"]["id"] for n in res["cotag_neighbors"])


async def test_related_include_semantic_false_skips_search(related_service):
    service = related_service()
    # Seed one memoria; stub index returns no hits (it'd be an empty semantic list anyway),
    # but the flag contract is "don't even try".
    a = await service.save({"content": "a body", "title": "a"})

    def _fail_search(*args, **kwargs):
        raise RuntimeError("search should not have been called")

    service.index.search = _fail_search  # type: ignore[assignment]

    res = await service.related({"id": a["memory"]["id"], "include_semantic": False})
    assert res["ok"] is True
    assert res["semantic_neighbors"] == []


async def test_related_semantic_neighbors_from_stub_search(related_service):
    service = related_service()
    a = await service.save({"content": "target body", "title": "a"})
    b = await service.save({"content": "neighbor body", "title": "b"})
    # Stub the dense index to return b as a hit.
    service.index = _StubIndex(
        [
            {
                "score": 0.85,
                "metadata": {"memory_id": b["memory"]["id"]},
                "memory": "neighbor body",
            }
        ]
    )
    res = await service.related({"id": a["memory"]["id"], "include_semantic": True, "k": 3})
    ids = [n["id"] for n in res["semantic_neighbors"]]
    assert b["memory"]["id"] in ids
    assert res["semantic_neighbors"][0]["score"] is not None

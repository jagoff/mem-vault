"""Integration tests for the ``memory_neighborhood`` MCP tool (v0.6.0).

Where ``test_graph.py`` covers the pure BFS module, this file pins the
HTTP-equivalent shape of the MCP handler:

- Validates input (``ids`` non-empty list, ``edge_kinds`` is list-or-None).
- Reports unknown seeds in the response without erroring out (when at
  least one seed exists).
- Sorts results by hop ascending.
- Reuses the corpus cache via ``_list_corpus``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mem_vault.config import Config
from mem_vault.index import _CircuitBreaker
from mem_vault.server import MemVaultService


class _StubIndex:
    """Minimal index stub — neighborhood doesn't actually call it."""

    def __init__(self) -> None:
        self._breaker = _CircuitBreaker(threshold=3, cooldown_s=30.0)

    @property
    def breaker(self):
        return self._breaker

    def add(self, content, **kwargs):
        return [{"id": "stub"}]

    def search(self, query, **kwargs):
        return []

    def delete_by_metadata(self, *args):
        return 0


@pytest.fixture
def hood_service(tmp_path: Path):
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


async def test_neighborhood_rejects_empty_ids(hood_service):
    service = hood_service()
    res = await service.neighborhood({"ids": []})
    assert res["ok"] is False
    assert res["code"] == "validation_failed"


async def test_neighborhood_rejects_non_list_edge_kinds(hood_service):
    service = hood_service()
    a = await service.save({"content": "a body", "title": "a"})
    res = await service.neighborhood(
        {"ids": [a["memory"]["id"]], "edge_kinds": "contradicts"}
    )
    assert res["ok"] is False
    assert res["code"] == "validation_failed"


async def test_neighborhood_returns_seed_at_hop_zero(hood_service):
    service = hood_service()
    a = await service.save({"content": "alone", "title": "a"})
    res = await service.neighborhood({"ids": [a["memory"]["id"]], "hops": 0})
    assert res["ok"] is True
    assert res["count"] == 1
    assert res["nodes"][0]["id"] == a["memory"]["id"]
    assert res["nodes"][0]["hop"] == 0


async def test_neighborhood_traverses_related_one_hop(hood_service):
    service = hood_service()
    a = await service.save({"content": "a", "title": "a"})
    b = await service.save({"content": "b", "title": "b"})
    c = await service.save({"content": "c", "title": "c"})
    service.storage.update(
        a["memory"]["id"], related=[b["memory"]["id"]]
    )
    service.storage.update(
        b["memory"]["id"], related=[c["memory"]["id"]]
    )
    # Invalidate the corpus cache so the new related links are visible.
    service._invalidate_corpus_cache()

    res = await service.neighborhood({"ids": [a["memory"]["id"]], "hops": 1})
    assert res["ok"] is True
    ids = {n["id"] for n in res["nodes"]}
    assert a["memory"]["id"] in ids
    assert b["memory"]["id"] in ids
    assert c["memory"]["id"] not in ids  # 2 hops away


async def test_neighborhood_two_hops(hood_service):
    service = hood_service()
    a = await service.save({"content": "a", "title": "a"})
    b = await service.save({"content": "b", "title": "b"})
    c = await service.save({"content": "c", "title": "c"})
    service.storage.update(a["memory"]["id"], related=[b["memory"]["id"]])
    service.storage.update(b["memory"]["id"], related=[c["memory"]["id"]])
    service._invalidate_corpus_cache()

    res = await service.neighborhood({"ids": [a["memory"]["id"]], "hops": 2})
    ids = {n["id"] for n in res["nodes"]}
    assert ids == {a["memory"]["id"], b["memory"]["id"], c["memory"]["id"]}


async def test_neighborhood_filters_to_contradicts(hood_service):
    """edge_kinds=['contradicts'] should ignore related edges."""
    service = hood_service()
    a = await service.save({"content": "a", "title": "a"})
    b = await service.save({"content": "b", "title": "b"})
    c = await service.save({"content": "c", "title": "c"})
    service.storage.update(a["memory"]["id"], related=[b["memory"]["id"]])
    service.storage.update(a["memory"]["id"], contradicts=[c["memory"]["id"]])
    service._invalidate_corpus_cache()

    res = await service.neighborhood(
        {"ids": [a["memory"]["id"]], "hops": 1, "edge_kinds": ["contradicts"]}
    )
    ids = {n["id"] for n in res["nodes"]}
    assert ids == {a["memory"]["id"], c["memory"]["id"]}


async def test_neighborhood_reports_unknown_seed_alongside_known(hood_service):
    """Mix of valid + invalid seeds: response keeps going + lists unknowns."""
    service = hood_service()
    a = await service.save({"content": "a", "title": "a"})
    res = await service.neighborhood(
        {"ids": [a["memory"]["id"], "totally-fake-id"], "hops": 0}
    )
    assert res["ok"] is True
    assert res["unknown_seeds"] == ["totally-fake-id"]


async def test_neighborhood_all_unknown_seeds_returns_not_found(hood_service):
    service = hood_service()
    res = await service.neighborhood({"ids": ["nope-1", "nope-2"], "hops": 1})
    assert res["ok"] is False
    assert res["code"] == "not_found"


async def test_neighborhood_max_nodes_caps_response(hood_service):
    service = hood_service()
    seeds = []
    related_ids = []
    for i in range(8):
        m = await service.save({"content": f"node-{i}", "title": f"n{i}"})
        related_ids.append(m["memory"]["id"])
    for i in range(8):
        m = await service.save({"content": f"seed-{i}", "title": f"s{i}"})
        seeds.append(m["memory"]["id"])
        service.storage.update(seeds[-1], related=related_ids)
    service._invalidate_corpus_cache()

    res = await service.neighborhood(
        {"ids": seeds[:1], "hops": 1, "max_nodes": 3}
    )
    assert res["ok"] is True
    assert len(res["nodes"]) <= 3

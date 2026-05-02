"""Tests for the v0.6.0 ``/api/graph`` endpoint with real ``kind`` edges.

Pre-v0.6.0 this endpoint only emitted co-tag edges. Now it returns
related / contradicts / cotag edges with a ``primary`` kind for the
frontend to style. This file pins the new contract:

- ``edges_filter`` query param: all / related / contradicts / cotag / explicit.
- Each edge carries ``kinds: [...]`` and a single ``primary`` (priority:
  contradicts > related > cotag) so the frontend can color/style.
- ``explicit`` filter excludes co-tag noise (the agent-friendly default
  for "show me the real knowledge graph").
"""

from __future__ import annotations

import types
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mem_vault.storage import Memory
from mem_vault.ui.server import create_app


def _make_graph_service(memories: list[Memory]):
    """A stub service whose ``storage.list`` returns the supplied memorias.

    ``/api/graph`` only touches storage.list — no Ollama / Qdrant — so this
    is enough surface to drive the endpoint.
    """
    cfg = types.SimpleNamespace(
        http_token=None,
        memory_dir="/tmp/fake",
        agent_id=None,
        user_id="default",
        qdrant_collection="test",
    )

    class _Storage:
        def list(self, **_kw):
            return memories

    class _Stub:
        def __init__(self) -> None:
            self.config = cfg
            self.storage = _Storage()

        async def search(self, args: dict[str, Any]):
            return {"ok": True, "results": []}

        async def list_(self, args: dict[str, Any]):
            return {"ok": True, "memories": []}

        async def get(self, args: dict[str, Any]):
            return {"ok": True, "memory": {}}

        async def save(self, args: dict[str, Any]):
            return {"ok": True}

        async def update(self, args: dict[str, Any]):
            return {"ok": True}

        async def delete(self, args: dict[str, Any]):
            return {"ok": True}

    return _Stub()


def _m(
    mid: str,
    *,
    related: list[str] | None = None,
    contradicts: list[str] | None = None,
    tags: list[str] | None = None,
) -> Memory:
    return Memory(
        id=mid,
        name=f"Memory {mid}",
        description=f"desc {mid}",
        type="note",
        tags=tags or [],
        related=related or [],
        contradicts=contradicts or [],
    )


@pytest.fixture
def client_with_graph_corpus():
    # Three memories: a -> b (related), a -> c (contradicts), b ↔ c (cotag).
    corpus = [
        _m("a", related=["b"], contradicts=["c"], tags=["x", "y"]),
        _m("b", tags=["x", "y", "z"]),
        _m("c", tags=["x", "y", "z"]),
    ]
    service = _make_graph_service(corpus)
    return TestClient(create_app(service=service))


def test_graph_default_returns_nodes_and_edges(client_with_graph_corpus):
    resp = client_with_graph_corpus.get("/api/graph")
    assert resp.status_code == 200
    body = resp.json()
    assert {n["data"]["id"] for n in body["nodes"]} == {"a", "b", "c"}
    # Default filter is "all" — every kind is returned.
    primaries = {e["data"]["primary"] for e in body["edges"]}
    assert {"contradicts", "related", "cotag"} <= primaries


def test_graph_explicit_filter_drops_cotag_edges(client_with_graph_corpus):
    resp = client_with_graph_corpus.get("/api/graph?edges_filter=explicit")
    body = resp.json()
    primaries = {e["data"]["primary"] for e in body["edges"]}
    assert primaries <= {"related", "contradicts"}
    assert "cotag" not in primaries


def test_graph_contradicts_filter_only_returns_contradiction_edges(
    client_with_graph_corpus,
):
    resp = client_with_graph_corpus.get("/api/graph?edges_filter=contradicts")
    body = resp.json()
    assert len(body["edges"]) >= 1
    for e in body["edges"]:
        assert e["data"]["primary"] == "contradicts"


def test_graph_edge_primary_priority_contradicts_over_related(client_with_graph_corpus):
    """If a pair is both contradicts and related, primary must be 'contradicts'."""
    # Build a custom corpus where a has both a related AND contradicts to b.
    corpus = [
        _m("a", related=["b"], contradicts=["b"]),
        _m("b"),
    ]
    service = _make_graph_service(corpus)
    client = TestClient(create_app(service=service))
    resp = client.get("/api/graph")
    body = resp.json()
    assert len(body["edges"]) == 1
    edge = body["edges"][0]["data"]
    assert sorted(edge["kinds"]) == ["contradicts", "related"]
    # Priority: contradicts wins
    assert edge["primary"] == "contradicts"


def test_graph_invalid_filter_returns_422(client_with_graph_corpus):
    resp = client_with_graph_corpus.get("/api/graph?edges_filter=does-not-exist")
    assert resp.status_code == 422


def test_graph_node_data_includes_v6_metrics(client_with_graph_corpus):
    """Each node carries usage_count + helpful_ratio for size/halo styling."""
    resp = client_with_graph_corpus.get("/api/graph")
    body = resp.json()
    for n in body["nodes"]:
        assert "usage_count" in n["data"]
        assert "helpful_ratio" in n["data"]

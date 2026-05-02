"""Tests para el endpoint ``/dashboard`` y ``/api/dashboard`` (v0.6.0).

El dashboard agrega TODA la información cross-cutting en una sola página:
hero KPIs, telemetry, ranker meta, antagonist queue, daemon status,
corpus distribution, top usadas / helpful / zombies, duplicados, lint
issues, tensiones, reflections recientes.

Estos tests pinean el contract:

- /api/dashboard returns 200 con todas las claves esperadas.
- /dashboard renderiza HTML con 200 y trae las secciones principales.
- Best-effort: cada subsistema (telemetry, ranker, antagonist, daemon)
  degrada a None / dict con error sin tirar el endpoint cuando le falta
  data.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mem_vault.storage import Memory
from mem_vault.ui.server import create_app


def _make_dashboard_service(memorias: list[Memory] | None = None):
    cfg = types.SimpleNamespace(
        http_token=None,
        memory_dir="/tmp/fake-vault",
        agent_id=None,
        user_id="default",
        qdrant_collection="test",
        # state_dir solo se toca al leer pickles / SQLite — apuntamos a
        # un dir vacio asi todo degrada a default sin warnings ruidosos.
        state_dir=__import__("pathlib").Path("/tmp/fake-state-dashboard"),
    )

    class _Storage:
        def __init__(self, items: list[Memory]) -> None:
            self._items = items

        def list(self, **_kw):
            return list(self._items)

        def get(self, mid: str):
            return next((m for m in self._items if m.id == mid), None)

    class _Stub:
        def __init__(self, mems: list[Memory]) -> None:
            self.config = cfg
            self.storage = _Storage(mems)

        async def search(self, args: dict[str, Any]):
            return {"ok": True, "results": []}

        async def list_(self, args: dict[str, Any]):
            return {"ok": True, "memories": [m.to_dict() for m in self.storage._items]}

        async def get(self, args: dict[str, Any]):
            return {"ok": True, "memory": {}}

        async def save(self, args: dict[str, Any]):
            return {"ok": True}

        async def update(self, args: dict[str, Any]):
            return {"ok": True}

        async def delete(self, args: dict[str, Any]):
            return {"ok": True}

        async def lint(self, args: dict[str, Any]):
            return {"ok": True, "problems": []}

    return _Stub(memorias or [])


def _m(
    mid: str,
    *,
    type: str = "note",
    contradicts: list[str] | None = None,
    related: list[str] | None = None,
    tags: list[str] | None = None,
    usage_count: int = 0,
    helpful_count: int = 0,
) -> Memory:
    return Memory(
        id=mid,
        name=f"Memory {mid}",
        description=f"desc {mid}",
        type=type,
        tags=tags or [],
        related=related or [],
        contradicts=contradicts or [],
        usage_count=usage_count,
        helpful_count=helpful_count,
    )


@pytest.fixture
def dashboard_client():
    corpus = [
        _m("a", type="decision", related=["b"]),
        _m("b", type="bug", contradicts=["a"]),
        _m("c", type="note", tags=["project:x", "lang:py"], usage_count=5, helpful_count=3),
        _m("d", type="preference", tags=["project:x", "lang:py"]),
    ]
    service = _make_dashboard_service(corpus)
    return TestClient(create_app(service=service))


def test_api_dashboard_returns_200_with_all_keys(dashboard_client):
    resp = dashboard_client.get("/api/dashboard")
    assert resp.status_code == 200
    body = resp.json()
    expected_keys = {
        "version",
        "vault_path",
        "agent_id",
        "user_id",
        "collection",
        "totals",
        "activity",
        "graph",
        "top_used",
        "top_helpful",
        "zombies",
        "contradictions",
        "lint",
        "duplicates",
        "telemetry",
        "ranker",
        "antagonist",
        "daemon",
        "reflections",
    }
    assert expected_keys <= set(body.keys())


def test_api_dashboard_totals_include_by_type(dashboard_client):
    body = dashboard_client.get("/api/dashboard").json()
    assert body["totals"]["memorias"] == 4
    by_type = body["totals"]["by_type"]
    assert by_type["decision"] == 1
    assert by_type["bug"] == 1
    assert by_type["note"] == 1
    assert by_type["preference"] == 1


def test_api_dashboard_includes_graph_edge_counts(dashboard_client):
    body = dashboard_client.get("/api/dashboard").json()
    g = body["graph"]
    assert g["nodes"] == 4
    # a-b carries both related and contradicts
    assert g["related"] >= 1
    assert g["contradicts"] >= 1
    # c and d share project:x + lang:py → one cotag edge
    assert g["cotag"] >= 1


def test_api_dashboard_top_used_lists_only_used_memorias(dashboard_client):
    body = dashboard_client.get("/api/dashboard").json()
    used = body["top_used"]
    assert len(used) == 1
    assert used[0]["id"] == "c"
    assert used[0]["usage_count"] == 5


def test_api_dashboard_contradictions_list_includes_target_ids(dashboard_client):
    body = dashboard_client.get("/api/dashboard").json()
    contras = body["contradictions"]
    assert len(contras) == 1
    assert contras[0]["id"] == "b"
    assert contras[0]["contradicts"] == ["a"]


def test_api_dashboard_by_project_breakdown(dashboard_client):
    body = dashboard_client.get("/api/dashboard").json()
    proj = body["totals"]["by_project"]
    assert proj.get("x") == 2  # c + d both tagged project:x


def test_api_dashboard_telemetry_ranker_antagonist_have_subsystem_dicts(dashboard_client):
    body = dashboard_client.get("/api/dashboard").json()
    # Cada subsistema retorna un dict (con o sin error/empty data).
    assert isinstance(body["telemetry"], dict)
    assert isinstance(body["ranker"], dict)
    assert isinstance(body["antagonist"], dict)
    assert isinstance(body["daemon"], dict)
    # Defaults razonables cuando no hay state populado:
    assert body["antagonist"]["pending_count"] == 0
    assert body["antagonist"]["enabled"] in (True, False)


def test_dashboard_html_renders_with_200(dashboard_client):
    resp = dashboard_client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    html = resp.text
    # Sanity: estructura basica del template presente.
    assert "memorias" in html
    assert "Knowledge graph" in html
    assert "Telemetry" in html
    assert "Antagonist" in html
    # Navlinks coherentes
    assert "/dashboard" in html
    assert "/graph" in html


def test_dashboard_html_navlinks_present_in_other_pages(dashboard_client):
    """index.html y graph.html deberian tener el link al dashboard tambien."""
    home = dashboard_client.get("/").text
    graph = dashboard_client.get("/graph").text
    for page in (home, graph):
        assert ">dashboard<" in page or "/dashboard" in page

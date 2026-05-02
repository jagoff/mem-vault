"""Tests para la pestaña ``overview`` y el endpoint JSON ``/api/dashboard`` (v0.6.0).

A partir de v0.6.0 el dashboard NO es una página separada — vive como
una pestaña ``overview`` dentro del panel principal de mem-vault (``/``)
y se expone también como JSON en ``/api/dashboard``. Estos tests pinean
ambos contracts:

- ``GET /api/overview`` retorna el HTML fragment para HTMX.
- ``GET /api/dashboard`` retorna el JSON con todas las claves esperadas.
- ``GET /`` (la página principal) incluye la pestaña overview como
  primera y default.
- ``GET /dashboard`` (la URL legacy) ya NO existe — devuelve 404.
- Cada subsistema (telemetry / ranker / antagonist / daemon) degrada a
  un dict con datos por defecto cuando no hay state populado.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mem_vault.storage import Memory
from mem_vault.ui.server import create_app


def _make_overview_service(memorias: list[Memory] | None = None):
    cfg = types.SimpleNamespace(
        http_token=None,
        memory_dir="/tmp/fake-vault",
        agent_id=None,
        user_id="default",
        qdrant_collection="test",
        # state_dir solo se toca al leer pickles / SQLite — apuntamos a
        # un dir vacio asi todo degrada a default sin warnings ruidosos.
        state_dir=__import__("pathlib").Path("/tmp/fake-state-overview"),
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
def overview_client():
    corpus = [
        _m("a", type="decision", related=["b"]),
        _m("b", type="bug", contradicts=["a"]),
        _m("c", type="note", tags=["project:x", "lang:py"], usage_count=5, helpful_count=3),
        _m("d", type="preference", tags=["project:x", "lang:py"]),
    ]
    service = _make_overview_service(corpus)
    return TestClient(create_app(service=service))


def test_api_dashboard_returns_200_with_all_keys(overview_client):
    resp = overview_client.get("/api/dashboard")
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


def test_api_dashboard_totals_include_by_type(overview_client):
    body = overview_client.get("/api/dashboard").json()
    assert body["totals"]["memorias"] == 4
    by_type = body["totals"]["by_type"]
    assert by_type["decision"] == 1
    assert by_type["bug"] == 1
    assert by_type["note"] == 1
    assert by_type["preference"] == 1


def test_api_dashboard_includes_graph_edge_counts(overview_client):
    body = overview_client.get("/api/dashboard").json()
    g = body["graph"]
    assert g["nodes"] == 4
    assert g["related"] >= 1
    assert g["contradicts"] >= 1
    assert g["cotag"] >= 1


def test_api_dashboard_top_used_lists_only_used_memorias(overview_client):
    body = overview_client.get("/api/dashboard").json()
    used = body["top_used"]
    assert len(used) == 1
    assert used[0]["id"] == "c"
    assert used[0]["usage_count"] == 5


def test_api_dashboard_contradictions_list_includes_target_ids(overview_client):
    body = overview_client.get("/api/dashboard").json()
    contras = body["contradictions"]
    assert len(contras) == 1
    assert contras[0]["id"] == "b"
    assert contras[0]["contradicts"] == ["a"]


def test_api_dashboard_by_project_breakdown(overview_client):
    body = overview_client.get("/api/dashboard").json()
    proj = body["totals"]["by_project"]
    assert proj.get("x") == 2


def test_api_dashboard_subsystem_dicts_present(overview_client):
    body = overview_client.get("/api/dashboard").json()
    assert isinstance(body["telemetry"], dict)
    assert isinstance(body["ranker"], dict)
    assert isinstance(body["antagonist"], dict)
    assert isinstance(body["daemon"], dict)
    assert body["antagonist"]["pending_count"] == 0
    assert body["antagonist"]["enabled"] in (True, False)


def test_api_overview_renders_html_fragment(overview_client):
    """``/api/overview`` returns an HTMX fragment (no <html> wrapper)."""
    resp = overview_client.get("/api/overview")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    # Fragment shape: starts with the dashboard <div>, no <html>/<body>.
    assert "<html" not in body.lower()
    assert "<body" not in body.lower()
    assert 'class="dashboard"' in body
    # Sanity: las secciones principales del overview estan presentes.
    assert "Telemetry" in body
    assert "Knowledge graph" in body
    assert "Antagonist" in body


def test_index_page_includes_overview_tab_first(overview_client):
    """La página principal debe tener la tab ``overview`` como primera + active."""
    resp = overview_client.get("/")
    assert resp.status_code == 200
    html = resp.text
    # Tab overview presente con class active.
    assert 'data-tab="overview"' in html
    assert "/api/overview" in html
    # El boot fetch ahora apunta a /api/overview, no a /api/memories.
    assert 'hx-get="/api/overview"' in html or "hx-get=\"/api/overview\"" in html


def test_index_page_navlinks_drop_dashboard_link(overview_client):
    """El link ``dashboard`` debe haberse retirado de las navlinks.

    Excepción: ``dashboard.css`` sigue cargando porque los estilos del
    overview viven ahí. Lo que importa es que NO haya un ``<a href=
    "/dashboard">`` en el topbar y que el link visible "dashboard" no
    aparezca como entrada de navegación.
    """
    html = overview_client.get("/").text
    assert ">dashboard<" not in html  # ningún link visible "dashboard"
    assert 'href="/dashboard"' not in html  # ni el href en navlinks
    assert "mem-vault" in html  # el link nuevo está


def test_legacy_dashboard_route_returns_404(overview_client):
    """La página standalone ``/dashboard`` ya no existe."""
    resp = overview_client.get("/dashboard")
    assert resp.status_code == 404


def test_graph_page_navlinks_drop_dashboard_link(overview_client):
    """``/graph`` también limpió el link al dashboard ahora obsoleto."""
    html = overview_client.get("/graph").text
    assert ">dashboard<" not in html
    assert 'href="/dashboard"' not in html
    assert "mem-vault" in html

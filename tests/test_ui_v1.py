"""Tests for the Pydantic-validated ``/api/v1/*`` JSON endpoints.

These cover the API hygiene wrapper introduced for ``api_v1_save`` and
``api_v1_update``:

* Bad payloads (missing required field, wrong type, unknown extra key) must
  bounce back a clean 422 with FastAPI's standard error envelope, not a 500
  from somewhere deep inside the service.
* Good payloads must reach the service untouched (after Pydantic strips
  ``None`` defaults via ``exclude_none=True``).

The setup mirrors ``tests/test_ui_auth.py``: a stub service exposing the
small surface ``ui.server`` actually calls, wired through ``create_app`` and
driven by FastAPI's ``TestClient``. No Qdrant / Ollama / disk I/O.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mem_vault.ui.server import create_app


def _make_fake_service():
    """Mirror of the stub in ``test_ui_auth.py``, plus call recording.

    We capture the args that reach ``save`` / ``update`` so the "good payload"
    tests can assert that ``model_dump(exclude_none=True)`` was applied
    (no spurious ``None`` keys leaking into the service).
    """

    cfg = types.SimpleNamespace(
        http_token=None,
        memory_dir="/tmp/fake",
        agent_id=None,
        user_id="default",
        qdrant_collection="test",
    )

    class _Stub:
        def __init__(self) -> None:
            self.config = cfg
            self.calls: dict[str, list[dict[str, Any]]] = {"save": [], "update": []}

        async def search(self, args):
            return {"ok": True, "query": args.get("query", ""), "count": 0, "results": []}

        async def list_(self, args):
            return {"ok": True, "count": 0, "memories": []}

        async def get(self, args):
            return {"ok": True, "memory": {"id": args["id"], "name": "stub"}}

        async def save(self, args):
            self.calls["save"].append(dict(args))
            return {
                "ok": True,
                "memory": {"id": "stub", "content": args.get("content", "")},
                "indexed": True,
            }

        async def update(self, args):
            self.calls["update"].append(dict(args))
            return {"ok": True, "memory": {"id": args["id"], "name": "stub"}}

        async def delete(self, args):
            return {"ok": True, "deleted_file": True, "deleted_index_entries": 0}

        @property
        def storage(self):
            class _S:
                def list(self, **_kw):
                    return []

            return _S()

    return _Stub()


@pytest.fixture
def client_and_service():
    service = _make_fake_service()
    client = TestClient(create_app(service=service))
    return client, service


# ---------------------------------------------------------------------------
# POST /api/v1/memories — MemoryCreate
# ---------------------------------------------------------------------------


def test_save_missing_content_returns_422(client_and_service):
    client, service = client_and_service
    resp = client.post("/api/v1/memories", json={"tags": ["x"]})
    assert resp.status_code == 422
    body = resp.json()
    # FastAPI's default validation envelope keys the errors under ``detail``.
    assert "detail" in body
    assert any(err["loc"][-1] == "content" for err in body["detail"])
    # Service must not have been called when validation fails.
    assert service.calls["save"] == []


def test_save_wrong_type_for_tags_returns_422(client_and_service):
    client, service = client_and_service
    resp = client.post(
        "/api/v1/memories",
        json={"content": "hi", "tags": "not-a-list"},
    )
    assert resp.status_code == 422
    assert any(err["loc"][-1] == "tags" for err in resp.json()["detail"])
    assert service.calls["save"] == []


def test_save_invalid_type_literal_returns_422(client_and_service):
    client, _ = client_and_service
    resp = client.post(
        "/api/v1/memories",
        json={"content": "hi", "type": "not-a-real-type"},
    )
    assert resp.status_code == 422
    assert any(err["loc"][-1] == "type" for err in resp.json()["detail"])


def test_save_empty_content_returns_422(client_and_service):
    """``content`` has ``min_length=1`` — the empty string is invalid."""
    client, _ = client_and_service
    resp = client.post("/api/v1/memories", json={"content": ""})
    assert resp.status_code == 422


def test_save_unknown_field_returns_422(client_and_service):
    """``extra="forbid"`` should reject typos before they reach the service."""
    client, _ = client_and_service
    resp = client.post(
        "/api/v1/memories",
        json={"content": "hi", "tagz": ["typo"]},
    )
    assert resp.status_code == 422


def test_save_valid_payload_reaches_service(client_and_service):
    client, service = client_and_service
    resp = client.post(
        "/api/v1/memories",
        json={
            "content": "hello world",
            "tags": ["a", "b"],
            "type": "note",
            "auto_extract": False,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert len(service.calls["save"]) == 1
    args = service.calls["save"][0]
    # exclude_none=True must drop the optional-unset fields.
    assert args == {
        "content": "hello world",
        "tags": ["a", "b"],
        "type": "note",
        "auto_extract": False,
    }


def test_save_visible_to_accepts_list_or_string(client_and_service):
    client, service = client_and_service
    resp_a = client.post(
        "/api/v1/memories",
        json={"content": "x", "visible_to": "private"},
    )
    resp_b = client.post(
        "/api/v1/memories",
        json={"content": "x", "visible_to": ["agent-1", "agent-2"]},
    )
    assert resp_a.status_code == 200
    assert resp_b.status_code == 200
    assert service.calls["save"][0]["visible_to"] == "private"
    assert service.calls["save"][1]["visible_to"] == ["agent-1", "agent-2"]


# ---------------------------------------------------------------------------
# PATCH /api/v1/memories/{id} — MemoryUpdate
# ---------------------------------------------------------------------------


def test_update_wrong_type_for_tags_returns_422(client_and_service):
    client, service = client_and_service
    resp = client.patch(
        "/api/v1/memories/abc",
        json={"tags": "not-a-list"},
    )
    assert resp.status_code == 422
    assert service.calls["update"] == []


def test_update_unknown_field_returns_422(client_and_service):
    """Updates can't carry e.g. ``content_hash`` or ``id`` in the body — the
    URL is the source of truth for the id, and arbitrary extras are typos."""
    client, _ = client_and_service
    resp = client.patch(
        "/api/v1/memories/abc",
        json={"content": "x", "id": "spoofed"},
    )
    assert resp.status_code == 422


def test_update_empty_payload_is_valid(client_and_service):
    """All fields are optional — an empty body just becomes a no-op update."""
    client, service = client_and_service
    resp = client.patch("/api/v1/memories/abc", json={})
    assert resp.status_code == 200
    assert service.calls["update"] == [{"id": "abc"}]


def test_update_valid_payload_reaches_service(client_and_service):
    client, service = client_and_service
    resp = client.patch(
        "/api/v1/memories/abc",
        json={"content": "edited", "tags": ["x"], "title": "T"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert service.calls["update"] == [
        {"id": "abc", "content": "edited", "tags": ["x"], "title": "T"}
    ]


def test_update_url_id_overrides_body(client_and_service):
    """The id is set from the URL after ``model_dump`` — even if a future
    schema accepted ``id`` in the body, the path param wins. With
    ``extra='forbid'`` today, ``id`` in the body would 422 first; this guards
    against regressions if someone relaxes ``extra``.
    """
    client, service = client_and_service
    resp = client.patch("/api/v1/memories/from-url", json={"title": "ok"})
    assert resp.status_code == 200
    assert service.calls["update"][0]["id"] == "from-url"

"""Tests for the HTTP-backed RemoteMemVaultService.

Strategy: spin up a real FastAPI app with a mocked MemVaultService, then
point a RemoteMemVaultService at the in-process TestClient transport. We
exercise every public method to confirm the JSON contract matches what
``MemVaultService`` returns in-process.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from mem_vault.remote import RemoteMemVaultService


class _FakeService:
    """Minimal MemVaultService stub recording calls and returning canned data."""

    def __init__(self):
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def save(self, args):
        self.calls.append(("save", args))
        return {"ok": True, "memory": {"id": "stub", **args}, "indexed": True}

    async def search(self, args):
        self.calls.append(("search", args))
        return {
            "ok": True,
            "query": args["query"],
            "count": 1,
            "results": [
                {"id": "stub", "score": 0.9, "memory": {"id": "stub", "name": "x"}, "snippet": "x"},
            ],
        }

    async def list_(self, args):
        self.calls.append(("list", args))
        return {"ok": True, "count": 0, "memories": []}

    async def get(self, args):
        self.calls.append(("get", args))
        return {"ok": True, "memory": {"id": args["id"], "name": "got"}}

    async def update(self, args):
        self.calls.append(("update", args))
        return {"ok": True, "memory": {"id": args["id"], "name": "updated"}}

    async def delete(self, args):
        self.calls.append(("delete", args))
        return {"ok": True, "deleted_file": True, "deleted_index_entries": 1}

    @property
    def storage(self):
        m = AsyncMock()
        m.list = lambda **k: []  # graceful default for /api/stats
        return m


@pytest.fixture
def client_factory():
    from fastapi.testclient import TestClient

    from mem_vault.ui.server import create_app

    fake = _FakeService()
    # Bypass the real load_config / Qdrant setup by passing an instance.
    app = create_app(service=fake)
    return fake, TestClient(app)


@pytest.fixture
async def remote(client_factory):
    """A RemoteMemVaultService routed through the in-process TestClient."""
    fake, client = client_factory
    transport = httpx.ASGITransport(app=client.app)
    svc = RemoteMemVaultService("http://test")
    # Replace the bare httpx client with one that hits our ASGI app.
    await svc.close()
    svc._client = httpx.AsyncClient(transport=transport, base_url="http://test")
    yield fake, svc
    await svc.close()


# ---------------------------------------------------------------------------


async def test_remote_save(remote):
    fake, svc = remote
    res = await svc.save({"content": "hello world", "type": "note"})
    assert res["ok"] is True
    assert res["memory"]["id"] == "stub"
    assert fake.calls[-1][0] == "save"
    assert fake.calls[-1][1]["content"] == "hello world"


async def test_remote_search(remote):
    fake, svc = remote
    res = await svc.search({"query": "X", "k": 3, "threshold": 0.2, "type": "note"})
    assert res["ok"] is True
    assert res["count"] == 1
    assert fake.calls[-1][0] == "search"
    assert fake.calls[-1][1]["query"] == "X"


async def test_remote_list_with_tags(remote):
    fake, svc = remote
    res = await svc.list_({"tags": ["a", "b"], "limit": 50})
    assert res["ok"] is True
    args = fake.calls[-1][1]
    assert args["tags"] == ["a", "b"]
    assert args["limit"] == 50


async def test_remote_get(remote):
    fake, svc = remote
    res = await svc.get({"id": "abc"})
    assert res["ok"] is True
    assert res["memory"]["id"] == "abc"


async def test_remote_update(remote):
    fake, svc = remote
    res = await svc.update({"id": "abc", "title": "new"})
    assert res["ok"] is True
    args = fake.calls[-1][1]
    assert args["id"] == "abc"
    assert args["title"] == "new"


async def test_remote_delete(remote):
    fake, svc = remote
    res = await svc.delete({"id": "abc"})
    assert res["ok"] is True
    assert res["deleted_file"] is True


async def test_remote_unreachable_returns_friendly_error():
    """When the server is offline we should NOT raise — return ok:false with a hint."""
    svc = RemoteMemVaultService("http://127.0.0.1:1", timeout=0.1)  # port that's not listening
    try:
        res = await svc.search({"query": "anything"})
        assert res["ok"] is False
        assert "unreachable" in res["error"].lower() or "connect" in res["error"].lower()
    finally:
        await svc.close()

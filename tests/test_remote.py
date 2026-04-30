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
        import types

        self.calls: list[tuple[str, dict[str, Any]]] = []
        # Bare-bones config so the bearer-auth middleware can read
        # ``service.config.http_token`` (None ⇒ auth disabled by default).
        self.config = types.SimpleNamespace(
            http_token=None,
            memory_dir="/tmp/fake",
            agent_id=None,
            user_id="default",
            qdrant_collection="test",
        )

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
    # max_retries=0 para que el test no demore por el backoff (0.5+1.0=1.5s)
    # de la nueva lógica de retry. Lo que verificamos acá es el mensaje, no
    # los retries (eso lo cubren los tests de abajo con MockTransport).
    svc = RemoteMemVaultService("http://127.0.0.1:1", timeout=0.1, max_retries=0)
    try:
        res = await svc.search({"query": "anything"})
        assert res["ok"] is False
        assert "unreachable" in res["error"].lower() or "connect" in res["error"].lower()
    finally:
        await svc.close()


# ---------------------------------------------------------------------------
# Retry + HTTPS-warning tests (added together with the hardening of
# `_request`). Strategy: drive the client through `httpx.MockTransport` so
# we can inject 5xx / 4xx / network errors deterministically and assert the
# call count + final result, without hitting any real server.
# ---------------------------------------------------------------------------


def _mk_remote_with_transport(
    handler, *, max_retries: int = 2, timeout: float = 1.0
) -> RemoteMemVaultService:
    """Build a RemoteMemVaultService whose AsyncClient is backed by MockTransport.

    Bypasses the constructor's httpx client by replacing `_client` after init.
    Tests own closing the service.
    """
    svc = RemoteMemVaultService("http://test", max_retries=max_retries, timeout=timeout)
    # Replace the client with one routed through the mock transport. We
    # don't `await svc.close()` here — the original client wasn't used and
    # leaving it un-closed in tests is harmless (it doesn't open sockets
    # until first request). Closing it would require an event loop.
    svc._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://test",
        timeout=timeout,
    )
    return svc


async def test_remote_retries_on_5xx():
    """503 twice, then 200 — expect success with 3 total calls."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(503, text="upstream restarting")
        return httpx.Response(200, json={"ok": True, "memory": {"id": "z"}, "indexed": True})

    # Patch sleep so backoff doesn't actually delay the test (we only care
    # that retries happened, not how long they slept).
    import mem_vault.remote as remote_mod

    orig_sleep = remote_mod.asyncio.sleep

    async def _no_sleep(_s):
        return None

    remote_mod.asyncio.sleep = _no_sleep  # type: ignore[assignment]
    try:
        svc = _mk_remote_with_transport(handler, max_retries=2)
        try:
            res = await svc.save({"content": "hi"})
            assert res["ok"] is True
            assert res["memory"]["id"] == "z"
            assert calls["n"] == 3, f"expected 3 attempts (1 + 2 retries), got {calls['n']}"
        finally:
            await svc.close()
    finally:
        remote_mod.asyncio.sleep = orig_sleep  # type: ignore[assignment]


async def test_remote_does_not_retry_on_400():
    """400 is the caller's fault — return immediately without retrying."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        # 400 con body JSON `{ok: false, ...}` — el cliente lo devuelve tal cual.
        return httpx.Response(400, json={"ok": False, "error": "bad request"})

    svc = _mk_remote_with_transport(handler, max_retries=5)
    try:
        res = await svc.save({"content": "hi"})
        assert res["ok"] is False
        assert calls["n"] == 1, f"expected 1 attempt (no retry on 4xx), got {calls['n']}"
    finally:
        await svc.close()


async def test_remote_does_not_retry_on_401():
    """401 means auth failure — never retry, return 'unauthorized' code."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="missing token")

    svc = _mk_remote_with_transport(handler, max_retries=5)
    try:
        res = await svc.save({"content": "hi"})
        assert res["ok"] is False
        assert res.get("code") == "unauthorized"
        assert calls["n"] == 1, f"expected 1 attempt (no retry on 401), got {calls['n']}"
    finally:
        await svc.close()


async def test_remote_warns_on_http_with_token(caplog):
    """http:// + token must emit a logger.warning about plaintext credentials."""
    import logging as _logging

    caplog.set_level(_logging.WARNING, logger="mem_vault.remote")
    svc = RemoteMemVaultService("http://insecure.example.com", token="secret-abc")
    try:
        warnings = [
            r
            for r in caplog.records
            if r.levelno == _logging.WARNING and "plaintext" in r.getMessage().lower()
        ]
        assert warnings, (
            f"expected a plaintext warning, got: {[r.getMessage() for r in caplog.records]}"
        )
        # Defensa adicional: el token literal no debería aparecer en el mensaje
        # (queremos avisar, no echar el secreto al log).
        for w in warnings:
            assert "secret-abc" not in w.getMessage()
    finally:
        await svc.close()


async def test_remote_no_warning_on_https(caplog):
    """https:// + token must NOT emit a plaintext warning."""
    import logging as _logging

    caplog.set_level(_logging.WARNING, logger="mem_vault.remote")
    svc = RemoteMemVaultService("https://secure.example.com", token="secret-abc")
    try:
        warnings = [
            r
            for r in caplog.records
            if r.levelno == _logging.WARNING and "plaintext" in r.getMessage().lower()
        ]
        assert not warnings, (
            f"unexpected plaintext warning over https: {[r.getMessage() for r in warnings]}"
        )
    finally:
        await svc.close()


async def test_remote_no_warning_when_no_token(caplog, monkeypatch):
    """http:// without any token (no arg, no config, no env) must NOT warn."""
    import logging as _logging

    # Aseguramos que no haya un token suelto en el env del test runner.
    monkeypatch.delenv("MEM_VAULT_HTTP_TOKEN", raising=False)
    caplog.set_level(_logging.WARNING, logger="mem_vault.remote")
    svc = RemoteMemVaultService("http://insecure.example.com")
    try:
        warnings = [
            r
            for r in caplog.records
            if r.levelno == _logging.WARNING and "plaintext" in r.getMessage().lower()
        ]
        assert not warnings, (
            f"unexpected plaintext warning without a token: {[r.getMessage() for r in warnings]}"
        )
    finally:
        await svc.close()

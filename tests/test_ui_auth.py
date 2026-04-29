"""Tests for the optional bearer-token auth on the UI / JSON HTTP server.

The token is read from ``service.config.http_token``. When set, every endpoint
except ``/healthz`` and ``/static/*`` requires ``Authorization: Bearer <token>``.
We exercise the middleware directly with a TestClient so we don't need uvicorn.

We also assert the startup guard in ``serve()``: binding to a non-loopback host
without a token must raise ``SystemExit``.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mem_vault.ui.server import _is_loopback_host, create_app, serve


def _make_fake_service(*, http_token: str | None = None):
    """Build a minimal stub that mimics the ``MemVaultService`` surface used by the UI."""

    cfg = types.SimpleNamespace(
        http_token=http_token,
        memory_dir="/tmp/fake",
        agent_id=None,
        user_id="default",
        qdrant_collection="test",
    )

    class _Stub:
        def __init__(self):
            self.config = cfg

        async def search(self, args):
            return {"ok": True, "query": args.get("query", ""), "count": 0, "results": []}

        async def list_(self, args):
            return {"ok": True, "count": 0, "memories": []}

        async def get(self, args):
            return {"ok": True, "memory": {"id": args["id"], "name": "stub"}}

        async def save(self, args):
            return {"ok": True, "memory": {"id": "stub", **args}, "indexed": True}

        async def update(self, args):
            return {"ok": True, "memory": {"id": args["id"], "name": "stub"}}

        async def delete(self, args):
            return {"ok": True, "deleted_file": True, "deleted_index_entries": 0}

        @property
        def storage(self):
            class _S:
                def list(self, **kw):
                    return []

            return _S()

    return _Stub()


# ---------------------------------------------------------------------------
# Token disabled (default) — every route is open
# ---------------------------------------------------------------------------


def test_no_token_means_open_access():
    service = _make_fake_service(http_token=None)
    client = TestClient(create_app(service=service))
    assert client.get("/healthz").status_code == 200
    assert client.get("/api/v1/list").status_code == 200


# ---------------------------------------------------------------------------
# Token enabled — auth enforced everywhere except /healthz
# ---------------------------------------------------------------------------


def test_healthz_always_open_even_with_token():
    service = _make_fake_service(http_token="s3cret")
    client = TestClient(create_app(service=service))
    assert client.get("/healthz").status_code == 200


def test_protected_endpoint_rejects_missing_header():
    service = _make_fake_service(http_token="s3cret")
    client = TestClient(create_app(service=service))
    resp = client.get("/api/v1/list")
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Unauthorized"
    assert resp.headers.get("www-authenticate", "").lower().startswith("bearer")


def test_protected_endpoint_rejects_wrong_scheme():
    service = _make_fake_service(http_token="s3cret")
    client = TestClient(create_app(service=service))
    resp = client.get("/api/v1/list", headers={"Authorization": "Basic abcd"})
    assert resp.status_code == 401


def test_protected_endpoint_rejects_wrong_token():
    service = _make_fake_service(http_token="s3cret")
    client = TestClient(create_app(service=service))
    resp = client.get("/api/v1/list", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_protected_endpoint_accepts_valid_token():
    service = _make_fake_service(http_token="s3cret")
    client = TestClient(create_app(service=service))
    resp = client.get("/api/v1/list", headers={"Authorization": "Bearer s3cret"})
    assert resp.status_code == 200


def test_token_rotation_at_runtime():
    """Mutating ``config.http_token`` should take effect on the next request."""
    service = _make_fake_service(http_token="old")
    client = TestClient(create_app(service=service))
    assert client.get("/api/v1/list", headers={"Authorization": "Bearer old"}).status_code == 200
    service.config.http_token = "new"
    assert client.get("/api/v1/list", headers={"Authorization": "Bearer old"}).status_code == 401
    assert client.get("/api/v1/list", headers={"Authorization": "Bearer new"}).status_code == 200


# ---------------------------------------------------------------------------
# _is_loopback_host helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host,expected",
    [
        ("127.0.0.1", True),
        ("127.255.0.1", True),
        ("::1", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("192.168.1.10", False),
        ("example.com", False),
        ("", False),
    ],
)
def test_is_loopback_host(host: str, expected: bool) -> None:
    assert _is_loopback_host(host) is expected


# ---------------------------------------------------------------------------
# serve() startup guard
# ---------------------------------------------------------------------------


def test_serve_refuses_non_loopback_without_token():
    """``serve(host='0.0.0.0')`` without a token must abort with SystemExit."""
    fake_cfg = types.SimpleNamespace(http_token=None, memory_dir="/tmp/fake")
    with patch("mem_vault.ui.server.load_config", return_value=fake_cfg):
        with pytest.raises(SystemExit) as exc_info:
            serve(host="0.0.0.0", port=7880)
    assert "MEM_VAULT_HTTP_TOKEN" in str(exc_info.value)


def test_serve_allows_loopback_without_token(monkeypatch):
    """Binding to localhost without a token must NOT abort (legacy default)."""
    fake_cfg = types.SimpleNamespace(http_token=None, memory_dir="/tmp/fake")
    called: dict = {}

    def _fake_run(app, **kw):
        called.update(kw)

    with patch("mem_vault.ui.server.load_config", return_value=fake_cfg):
        with patch("mem_vault.ui.server.MemVaultService", lambda cfg: _make_fake_service()):
            with patch("uvicorn.run", _fake_run):
                serve(host="127.0.0.1", port=9999)
    assert called.get("host") == "127.0.0.1"
    assert called.get("port") == 9999


def test_serve_allows_non_loopback_with_token():
    """Token + non-loopback is the explicit opt-in for LAN exposure."""
    fake_cfg = types.SimpleNamespace(http_token="s3cret", memory_dir="/tmp/fake")
    called: dict = {}

    def _fake_run(app, **kw):
        called.update(kw)

    with patch("mem_vault.ui.server.load_config", return_value=fake_cfg):
        with patch("mem_vault.ui.server.MemVaultService", lambda cfg: _make_fake_service()):
            with patch("uvicorn.run", _fake_run):
                serve(host="0.0.0.0", port=9999)
    assert called.get("host") == "0.0.0.0"


# ---------------------------------------------------------------------------
# Remote client picks up token from env / config
# ---------------------------------------------------------------------------


def test_remote_client_sends_bearer_from_explicit_token():
    """``RemoteMemVaultService(token=...)`` must attach the Authorization header."""
    from mem_vault.remote import RemoteMemVaultService

    svc = RemoteMemVaultService("http://test", token="s3cret")
    try:
        assert svc._client.headers["authorization"] == "Bearer s3cret"
    finally:
        # Synchronous close — we never opened a real connection.
        import asyncio

        asyncio.get_event_loop().run_until_complete(svc.close()) if False else None


def test_remote_client_sends_bearer_from_env(monkeypatch):
    monkeypatch.setenv("MEM_VAULT_HTTP_TOKEN", "from-env")
    from mem_vault.remote import RemoteMemVaultService

    svc = RemoteMemVaultService("http://test")
    try:
        assert svc._client.headers["authorization"] == "Bearer from-env"
    finally:
        pass


def test_remote_client_no_header_when_no_token(monkeypatch):
    monkeypatch.delenv("MEM_VAULT_HTTP_TOKEN", raising=False)
    from mem_vault.remote import RemoteMemVaultService

    svc = RemoteMemVaultService("http://test")
    try:
        assert "authorization" not in {k.lower() for k in svc._client.headers}
    finally:
        pass

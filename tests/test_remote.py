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

    # ---- discovery / introspection stubs (mirror MemVaultService) --------

    async def briefing(self, args):
        self.calls.append(("briefing", args))
        return {
            "ok": True,
            "cwd": args.get("cwd"),
            "project_tag": "project:fake" if args.get("cwd") else None,
            "total_global": 42,
            "project_total": 7 if args.get("cwd") else 0,
            "recent_3": [],
            "top_tags": [],
            "lint_summary": {"few_tags": 0, "no_aprendido": 0, "short_body": 0},
        }

    async def derive_metadata(self, args):
        self.calls.append(("derive_metadata", args))
        return {
            "ok": True,
            "title": "auto-derived",
            "type": "note",
            "tags": ["a", "b", "c"],
            "tag_count": 3,
            "missing_tags": 0,
        }

    async def stats(self, args):
        self.calls.append(("stats", args))
        return {
            "ok": True,
            "scope": "global",
            "by_type": {"note": 1},
            "by_agent": {},
            "top_tags": [],
            "age_buckets": {"today": 0, "week": 0, "month": 0, "older": 0},
        }

    async def duplicates(self, args):
        self.calls.append(("duplicates", args))
        return {
            "ok": True,
            "threshold": args.get("threshold", 0.7),
            "scope": "global",
            "count": 0,
            "pairs": [],
        }

    async def lint(self, args):
        self.calls.append(("lint", args))
        return {
            "ok": True,
            "scope": "global",
            "total_scanned": 0,
            "with_issues": 0,
            "problems": [],
        }

    async def related(self, args):
        self.calls.append(("related", args))
        return {
            "ok": True,
            "id": args["id"],
            "related": [],
            "contradicts": [],
            "cotag_neighbors": [],
            "semantic_neighbors": [],
        }

    async def history(self, args):
        self.calls.append(("history", args))
        return {"ok": True, "id": args["id"], "count": 0, "entries": []}

    async def feedback(self, args):
        self.calls.append(("feedback", args))
        return {
            "ok": True,
            "id": args["id"],
            "helpful_count": 1 if args.get("helpful") is True else 0,
            "unhelpful_count": 1 if args.get("helpful") is False else 0,
            "usage_count": 1,
            "last_used": "2026-04-30T00:00:00",
        }

    async def synthesize(self, args):
        self.calls.append(("synthesize", args))
        return {
            "ok": True,
            "query": args["query"],
            "synthesis": "stub synthesis",
            "source_ids": ["stub"],
            "count": 1,
        }

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


# ---------------------------------------------------------------------------
# Discovery / introspection round-trips. Each test goes through the full
# stack: RemoteMemVaultService → ASGI app (mounted /api/v1/*) → fake service.
# Asserting both ``ok`` and the args the fake recorded confirms the params
# survive serialization in both directions.
# ---------------------------------------------------------------------------


async def test_remote_briefing(remote):
    fake, svc = remote
    res = await svc.briefing({"cwd": "/Users/fer/repositories/mem-vault"})
    assert res["ok"] is True
    assert fake.calls[-1][0] == "briefing"
    assert fake.calls[-1][1]["cwd"] == "/Users/fer/repositories/mem-vault"


async def test_remote_briefing_no_cwd(remote):
    """Omitting ``cwd`` must not send a stray ``cwd=None`` query param."""
    fake, svc = remote
    res = await svc.briefing({})
    assert res["ok"] is True
    # Server side sees ``cwd=None`` as the FastAPI default — what we care
    # about is the round-trip semantics, not the wire shape. The ``project_tag``
    # branch in the fake confirms ``cwd`` arrived empty.
    assert fake.calls[-1][1].get("cwd") in (None, "")


async def test_remote_derive_metadata(remote):
    fake, svc = remote
    res = await svc.derive_metadata({"content": "una decisión sobre X", "cwd": "/x"})
    assert res["ok"] is True
    assert res["tag_count"] == 3
    assert fake.calls[-1][0] == "derive_metadata"
    assert fake.calls[-1][1]["content"] == "una decisión sobre X"
    assert fake.calls[-1][1]["cwd"] == "/x"


async def test_remote_derive_metadata_empty_content_422(remote):
    """``content`` has ``min_length=1`` server-side — empty must 422."""
    fake, svc = remote
    res = await svc.derive_metadata({"content": ""})
    # 422 round-trips as a JSON body with ``detail`` (FastAPI default).
    assert res.get("ok") is not True
    assert "detail" in res or res.get("ok") is False


async def test_remote_stats(remote):
    fake, svc = remote
    res = await svc.stats({"cwd": "/x"})
    assert res["ok"] is True
    assert "by_type" in res
    assert fake.calls[-1][0] == "stats"


async def test_remote_duplicates(remote):
    fake, svc = remote
    res = await svc.duplicates({"threshold": 0.5, "cwd": "/x"})
    assert res["ok"] is True
    assert res["threshold"] == 0.5
    assert fake.calls[-1][0] == "duplicates"
    assert fake.calls[-1][1]["threshold"] == 0.5


async def test_remote_lint(remote):
    fake, svc = remote
    res = await svc.lint({"cwd": "/x"})
    assert res["ok"] is True
    assert "problems" in res
    assert fake.calls[-1][0] == "lint"


async def test_remote_related(remote):
    fake, svc = remote
    res = await svc.related({"id": "abc", "min_shared_tags": 3, "k": 7, "include_semantic": False})
    assert res["ok"] is True
    assert res["id"] == "abc"
    args = fake.calls[-1][1]
    assert args["id"] == "abc"
    assert args["min_shared_tags"] == 3
    assert args["k"] == 7
    assert args["include_semantic"] is False


async def test_remote_related_missing_id_does_not_hit_server(remote):
    """Validation short-circuits client-side — fake.calls must not record it."""
    fake, svc = remote
    pre = list(fake.calls)
    res = await svc.related({})
    assert res["ok"] is False
    assert res.get("code") == "validation_failed"
    assert fake.calls == pre  # no HTTP round-trip happened


async def test_remote_history(remote):
    fake, svc = remote
    res = await svc.history({"id": "abc", "limit": 5})
    assert res["ok"] is True
    assert res["count"] == 0
    args = fake.calls[-1][1]
    assert args["id"] == "abc"
    assert args["limit"] == 5


async def test_remote_history_missing_id_does_not_hit_server(remote):
    fake, svc = remote
    pre = list(fake.calls)
    res = await svc.history({})
    assert res["ok"] is False
    assert res.get("code") == "validation_failed"
    assert fake.calls == pre


async def test_remote_feedback_thumbs_up(remote):
    fake, svc = remote
    res = await svc.feedback({"id": "abc", "helpful": True})
    assert res["ok"] is True
    assert res["helpful_count"] == 1
    args = fake.calls[-1][1]
    assert args["id"] == "abc"
    assert args["helpful"] is True


async def test_remote_feedback_null_helpful_just_records_usage(remote):
    """``helpful=None`` must round-trip as null (NOT be stripped)."""
    fake, svc = remote
    res = await svc.feedback({"id": "abc", "helpful": None})
    assert res["ok"] is True
    args = fake.calls[-1][1]
    assert args["id"] == "abc"
    assert args["helpful"] is None
    assert res["usage_count"] == 1


async def test_remote_feedback_missing_id(remote):
    fake, svc = remote
    pre = list(fake.calls)
    res = await svc.feedback({})
    assert res["ok"] is False
    assert res.get("code") == "validation_failed"
    assert fake.calls == pre


async def test_remote_synthesize(remote):
    fake, svc = remote
    res = await svc.synthesize({"query": "qué sé sobre X", "k": 5, "threshold": 0.2})
    assert res["ok"] is True
    assert res["synthesis"] == "stub synthesis"
    assert res["source_ids"] == ["stub"]
    args = fake.calls[-1][1]
    assert args["query"] == "qué sé sobre X"
    assert args["k"] == 5
    assert args["threshold"] == 0.2


def test_remote_service_satisfies_build_handlers_symmetry():
    """All tools declared in ``server._TOOLS`` must resolve to a method on
    ``RemoteMemVaultService`` — otherwise ``_build_handlers`` fails at boot.

    This is a regression guard. The MCP server enforces this symmetry at
    startup: when a new tool is added to ``_TOOLS`` and the matching method
    isn't added on the remote service, the entire MCP boots with an
    ``AttributeError`` instead of returning a clean error to clients. This
    test catches that gap before it ships.
    """
    from mem_vault.server import _build_handlers

    svc = RemoteMemVaultService("http://test")
    handlers = _build_handlers(svc)
    # Every ``memory_*`` tool must have a callable handler.
    assert handlers, "expected at least one handler"
    for name, fn in handlers.items():
        assert callable(fn), f"handler for {name} is not callable"


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

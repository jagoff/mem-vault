"""FastAPI app powering the browser UI.

Routes:
- ``GET /``                 → index page (filters + memory list)
- ``GET /api/memories``     → list/search HTMX fragment (returns rows HTML)
- ``GET /api/memories/{id}``→ memory detail HTMX fragment (modal body)
- ``PATCH /api/memories/{id}`` → save edits (returns refreshed row)
- ``DELETE /api/memories/{id}``→ delete (HTMX swaps the row out)
- ``GET /api/stats``        → header stats badges
- ``GET /healthz``          → liveness probe

The server keeps a single ``MemVaultService`` instance — Qdrant + mem0 are
expensive to spin up, so we boot once and reuse. All filesystem ops are
async-safe via ``asyncio.to_thread``.

The server binds to ``127.0.0.1`` by default, never to ``0.0.0.0``. Pass
``--host 0.0.0.0`` if you really want network exposure (no auth — local
use only).
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from datetime import UTC
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from mem_vault import __version__
from mem_vault.config import load_config
from mem_vault.server import MemVaultService

logger = logging.getLogger("mem_vault.ui")

_HERE = Path(__file__).resolve().parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"


def _format_relative(iso_str: str) -> str:
    """Best-effort '2h ago' / '3d ago' formatting from an ISO timestamp."""
    if not iso_str:
        return ""
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        now = datetime.now(tz=dt.tzinfo)
        delta = now - dt
        s = int(delta.total_seconds())
    except Exception:
        return iso_str[:10]
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    if s < 86400 * 30:
        return f"{s // 86400}d ago"
    return iso_str[:10]


def create_app(service: MemVaultService | None = None) -> FastAPI:
    """Build the FastAPI application. Accepts an optional pre-built service for tests."""
    if service is None:
        config = load_config()
        service = MemVaultService(config)

    app = FastAPI(
        title="mem-vault UI",
        version=__version__,
        docs_url=None,  # disable /docs by default — local-only tool
        redoc_url=None,
    )
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.filters["relative_time"] = _format_relative

    app.state.service = service

    # ----- Pages ------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        cfg = service.config
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "version": __version__,
                "vault_path": str(cfg.memory_dir),
                "agent_id": cfg.agent_id or "—",
                "user_id": cfg.user_id,
                "collection": cfg.qdrant_collection,
            },
        )

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "version": __version__}

    # ----- API: list / search ----------------------------------------------

    @app.get("/api/memories", response_class=HTMLResponse)
    async def list_memories(
        request: Request,
        q: str = Query("", description="Optional semantic search query."),
        type: str = Query("", description="Filter by memory type."),
        tag: str = Query("", description="Filter by a single tag."),
        limit: int = Query(100, ge=1, le=500),
    ):
        results: list[dict[str, Any]]
        searched = bool(q.strip())
        if searched:
            payload = await service.search({"query": q.strip(), "k": limit, "threshold": 0.1})
            hits = payload.get("results", []) if payload.get("ok") else []
            results = []
            for h in hits:
                mem = h.get("memory")
                if mem:
                    mem = dict(mem)
                    mem["_score"] = h.get("score")
                    results.append(mem)
        else:
            payload = await service.list_(
                {
                    "type": type or None,
                    "tags": [tag] if tag else None,
                    "limit": limit,
                }
            )
            results = payload.get("memories", [])
            if type or tag:
                results = [
                    m
                    for m in results
                    if (not type or m.get("type") == type)
                    and (not tag or tag in (m.get("tags") or []))
                ]

        return templates.TemplateResponse(
            request,
            "_rows.html",
            {"memories": results, "searched": searched, "q": q, "type": type, "tag": tag},
        )

    # ----- API: detail / edit / delete -------------------------------------

    @app.get("/api/memories/{mem_id}", response_class=HTMLResponse)
    async def get_memory(request: Request, mem_id: str):
        payload = await service.get({"id": mem_id})
        if not payload.get("ok"):
            raise HTTPException(404, payload.get("error", "not found"))
        return templates.TemplateResponse(
            request,
            "_detail.html",
            {"m": payload["memory"]},
        )

    @app.patch("/api/memories/{mem_id}", response_class=HTMLResponse)
    async def update_memory(
        request: Request,
        mem_id: str,
        title: str = Form(""),
        description: str = Form(""),
        tags: str = Form(""),
        content: str = Form(""),
    ):
        kwargs = {"id": mem_id}
        if title.strip():
            kwargs["title"] = title.strip()
        if description.strip():
            kwargs["description"] = description.strip()
        if content.strip():
            kwargs["content"] = content.strip()
        if tags.strip():
            kwargs["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
        payload = await service.update(kwargs)
        if not payload.get("ok"):
            raise HTTPException(404, payload.get("error", "not found"))
        return templates.TemplateResponse(
            request,
            "_row.html",
            {"m": payload["memory"]},
        )

    @app.delete("/api/memories/{mem_id}")
    async def delete_memory(mem_id: str):
        payload = await service.delete({"id": mem_id})
        if not payload.get("ok"):
            raise HTTPException(404, payload.get("error", "not found"))
        # HTMX expects empty 200 → swaps the row out via hx-swap="outerHTML"
        return Response(status_code=200)

    # ----- API: stats -------------------------------------------------------

    @app.get("/api/stats", response_class=HTMLResponse)
    async def stats(request: Request):
        memories = await asyncio.to_thread(
            service.storage.list, type=None, tags=None, user_id=None, limit=10**9
        )
        total = len(memories)
        by_type = Counter(m.type for m in memories)
        by_agent = Counter((m.agent_id or "—") for m in memories)
        return templates.TemplateResponse(
            request,
            "_stats.html",
            {"total": total, "by_type": dict(by_type), "by_agent": dict(by_agent)},
        )

    @app.get("/api/types")
    async def types_endpoint():
        memories = await asyncio.to_thread(
            service.storage.list, type=None, tags=None, user_id=None, limit=10**9
        )
        types = sorted({m.type for m in memories})
        tags: set[str] = set()
        for m in memories:
            tags.update(m.tags)
        return JSONResponse({"types": types, "tags": sorted(tags)})

    return app


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def serve(host: str = "127.0.0.1", port: int = 7880, log_level: str = "info") -> None:
    """Run the UI server with uvicorn. Blocks until killed."""
    import uvicorn

    app = create_app()
    print(f"\n  mem-vault UI · http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level=log_level)

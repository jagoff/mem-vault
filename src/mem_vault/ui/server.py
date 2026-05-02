"""FastAPI app powering the browser UI.

Routes:
- ``GET /``                 → index page (filters + memory list)
- ``GET /api/memories``     → list/search HTMX fragment (returns rows HTML)
- ``GET /api/memories/{id}``→ memory detail HTMX fragment (modal body)
- ``PATCH /api/memories/{id}`` → save edits (returns refreshed row)
- ``DELETE /api/memories/{id}``→ delete (HTMX swaps the row out)
- ``GET /api/stats``        → header stats badges
- ``GET /healthz``          → liveness probe (always unauthenticated)

The server keeps a single ``MemVaultService`` instance — Qdrant + mem0 are
expensive to spin up, so we boot once and reuse. All filesystem ops are
async-safe via ``asyncio.to_thread``.

The server binds to ``127.0.0.1`` by default. When you bind to a non-
loopback address (``--host 0.0.0.0`` or a LAN IP) you **must** set
``MEM_VAULT_HTTP_TOKEN``: the startup helper :func:`serve` refuses to
launch otherwise. Authenticated requests pass ``Authorization: Bearer
<token>``; ``/healthz`` is exempt so external monitors keep working.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import secrets
import time
from collections import Counter
from datetime import UTC
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

from mem_vault import __version__
from mem_vault.config import load_config
from mem_vault.server import MemVaultService

logger = logging.getLogger("mem_vault.ui")

# Endpoints that bypass auth even when a token is configured. Keep the list
# tight — every entry is a chunk of attack surface someone can hit anonymously.
_AUTH_EXEMPT_PATHS = {"/healthz"}


def _is_loopback_host(host: str) -> bool:
    """Returns True iff ``host`` resolves to the loopback interface.

    Accepts ``127.0.0.1``, ``::1``, ``localhost``, and anything else inside
    the IPv4/IPv6 loopback ranges (``127.0.0.0/8``, ``::1/128``). Treats
    unknown hostnames as non-loopback (safe default — auth required).
    """
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


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


# ---------------------------------------------------------------------------
# Pydantic v2 schemas for /api/v1 JSON endpoints
# ---------------------------------------------------------------------------
#
# Why these exist: ``api_v1_save`` and ``api_v1_update`` used to take a raw
# ``dict[str, Any]`` and pass it straight to the service. A buggy/malicious
# caller sending ``{"content": null, "tags": "not a list"}`` would crash with
# a 500 instead of a clean 422. These models give us:
#   * input validation at the FastAPI boundary (typed bodies → 422 with the
#     Pydantic error envelope on bad shapes),
#   * an OpenAPI doc that reflects the actual contract,
#   * dict-passthrough into the service via ``model_dump(exclude_none=True)``
#     so the existing ``args.get(...)`` lookups inside ``MemVaultService``
#     keep working without any change to the service layer.
#
# The set of literal types mirrors ``mem_vault.storage._VALID_TYPES``. Keep
# them in sync — if ``storage.py`` ever grows a new type, add it here too or
# the API will reject the payload before the storage layer can validate.

_MemoryType = Literal["feedback", "preference", "decision", "fact", "note", "bug", "todo"]


class MemoryCreate(BaseModel):
    """Request body for ``POST /api/v1/memories``.

    Mirrors the kwargs accepted by ``MemVaultService.save``. Only ``content``
    is required; everything else has sane server-side defaults. ``visible_to``
    accepts either a list of agent IDs or one of the shorthands ``"public"`` /
    ``"private"`` (handled inside the service).
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1)
    title: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    type: _MemoryType | None = None
    agent_id: str | None = None
    user_id: str | None = None
    visible_to: list[str] | str | None = None
    auto_extract: bool | None = None
    auto_link: bool | None = None
    project: str | None = None
    auto_contradict: bool | None = None


class MemoryUpdate(BaseModel):
    """Request body for ``PATCH /api/v1/memories/{mem_id}``.

    All fields optional — the URL param carries the id. Only fields the caller
    actually wants to mutate should be present; ``model_dump(exclude_none=True)``
    in the route handler ensures we never send a spurious ``None`` to the
    service that would be misread as "clear this field".
    """

    model_config = ConfigDict(extra="forbid")

    content: str | None = None
    title: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    visible_to: list[str] | str | None = None
    project: str | None = None


class DeriveMetadataRequest(BaseModel):
    """Request body for ``POST /api/v1/derive_metadata``.

    The body is a tiny JSON wrapper around the in-process call so the remote
    client doesn't have to URL-encode multi-KB ``content`` payloads.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1)
    cwd: str | None = None


class FeedbackRequest(BaseModel):
    """Request body for ``POST /api/v1/memories/{id}/feedback``.

    ``helpful`` is a tri-state: ``true`` (thumbs up), ``false`` (thumbs down),
    ``null``/omitted (just record a 'used' event). Mirrors the in-process
    ``MemVaultService.feedback`` contract.
    """

    model_config = ConfigDict(extra="forbid")

    helpful: bool | None = None


class SynthesizeRequest(BaseModel):
    """Request body for ``POST /api/v1/synthesize``.

    Accepts the same ``{query, k, threshold}`` shape that ``memory_synthesize``
    takes. POST instead of GET because the LLM can chew on multi-paragraph
    questions that would be ugly as a query string.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    k: int = Field(10, ge=1, le=30)
    threshold: float = Field(0.1, ge=0.0, le=1.0)


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

    # Bearer-token auth is **opt-in** via ``MEM_VAULT_HTTP_TOKEN``. We attach
    # the middleware unconditionally so the token can be rotated at runtime
    # (mutate ``service.config.http_token`` and the next request honors it),
    # but the middleware short-circuits when the token is empty.
    @app.middleware("http")
    async def _bearer_auth(request: Request, call_next):
        token = (service.config.http_token or "").strip()
        if not token:
            return await call_next(request)
        if request.url.path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)
        # Static assets shouldn't require auth either — they're public CSS/JS,
        # leaking nothing about the user's memories. Keeping them open also
        # means a browser can fetch them with a plain ``<script src>`` after
        # the user authenticates the main page (e.g. via a ``token`` query
        # param wired up in templates if you go that route).
        if request.url.path.startswith("/static/"):
            return await call_next(request)
        header = request.headers.get("authorization", "")
        scheme, _, value = header.partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(value, token):
            return JSONResponse(
                {"detail": "Unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="mem-vault"'},
            )
        return await call_next(request)

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.filters["relative_time"] = _format_relative

    app.state.service = service

    # In-process TTL cache for the stats / types endpoints. Without this,
    # every dashboard refresh walks the entire vault (parses N markdown
    # files, hits stat() for each one). On iCloud-mounted Obsidian vaults
    # with thousands of memorias that's 3-5 s per request — turns the UI
    # into a slideshow. The cache is invalidated automatically when the TTL
    # lapses; UI refreshes inherit the lag at most once per ``_STATS_TTL_S``.
    _STATS_TTL_S = 30.0
    _stats_cache: dict[str, Any] = {
        "ts": 0.0,
        "by_type": None,
        "by_agent": None,
        "total": 0,
        "with_issues": 0,
        "duplicates": 0,
    }
    _types_cache: dict[str, Any] = {"ts": 0.0, "types": None, "tags": None}
    _cache_lock = asyncio.Lock()

    def _stats_fresh() -> bool:
        return (time.monotonic() - _stats_cache["ts"]) < _STATS_TTL_S

    def _types_fresh() -> bool:
        return (time.monotonic() - _types_cache["ts"]) < _STATS_TTL_S

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

    async def _lint_set() -> set[str]:
        """Returns the set of memory IDs that ``service.lint`` flagged.

        The full lint payload is paginated to 100 entries server-side, but for
        the UI badge we only need the IDs — anything beyond the cap shows up
        unflagged, which is fine: the user already gets a yellow pill in the
        header pointing them at the dedicated Quality tab.
        """
        payload = await service.lint({})
        if not payload.get("ok"):
            return set()
        return {p["id"] for p in payload.get("problems", [])}

    def _sort_memories(memories: list[dict[str, Any]], sort: str) -> list[dict[str, Any]]:
        """Apply a stable sort over the in-memory list.

        Sort keys are lifted from the row counters that ``Memory.to_frontmatter``
        emits when non-zero. ``zombie`` is the inverse signal — memorias never
        retrieved AND old enough to plausibly be dead weight; we sort by oldest
        ``updated`` so the list reads top→bottom as "first candidates to prune".
        """
        if not sort:
            return memories
        if sort == "usage_desc":
            return sorted(memories, key=lambda m: int(m.get("usage_count") or 0), reverse=True)
        if sort == "helpful_desc":
            return sorted(memories, key=lambda m: int(m.get("helpful_count") or 0), reverse=True)
        if sort == "unhelpful_desc":
            return sorted(
                memories, key=lambda m: int(m.get("unhelpful_count") or 0), reverse=True
            )
        if sort == "zombie":
            zombies = [
                m
                for m in memories
                if int(m.get("usage_count") or 0) == 0
                and int(m.get("helpful_count") or 0) == 0
                and int(m.get("unhelpful_count") or 0) == 0
            ]
            return sorted(zombies, key=lambda m: m.get("updated") or "")
        if sort == "recent":
            return sorted(memories, key=lambda m: m.get("updated") or "", reverse=True)
        return memories

    @app.get("/api/memories", response_class=HTMLResponse)
    async def list_memories(
        request: Request,
        q: str = Query("", description="Optional semantic search query."),
        type: str = Query("", description="Filter by memory type."),
        tag: str = Query("", description="Filter by a single tag."),
        sort: str = Query(
            "",
            description="Optional sort: usage_desc / helpful_desc / unhelpful_desc / zombie / recent.",
        ),
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
            # When sorting by usage / zombie we need the FULL corpus, not the
            # default 100-most-recent slice — otherwise "top usado" only ranks
            # within the most recent window and silently drops older winners.
            effective_limit = 500 if sort in {"usage_desc", "helpful_desc", "unhelpful_desc", "zombie"} else limit
            payload = await service.list_(
                {
                    "type": type or None,
                    "tags": [tag] if tag else None,
                    "limit": effective_limit,
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
            if sort:
                results = _sort_memories(results, sort)[:limit]

        lint_ids = await _lint_set() if results else set()
        return templates.TemplateResponse(
            request,
            "_rows.html",
            {
                "memories": results,
                "searched": searched,
                "q": q,
                "type": type,
                "tag": tag,
                "sort": sort,
                "lint_ids": lint_ids,
            },
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
        # Heterogeneous: ``tags`` is a list, the rest are strings — annotate
        # explicitly so mypy doesn't infer ``dict[str, str]`` from the seed.
        kwargs: dict[str, Any] = {"id": mem_id}
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

    @app.post("/api/memories/{mem_id}/feedback", response_class=HTMLResponse)
    async def feedback_memory(
        request: Request,
        mem_id: str,
        helpful: str = Form(""),
        inline: str = Form(""),
    ):
        """Record a thumbs up/down on a memory from the UI.

        ``helpful`` comes from the form as the string ``"true"`` / ``"false"``
        / ``""``. We coerce to bool | None and delegate to
        ``MemVaultService.feedback``; the response re-renders the feedback
        chunk so HTMX can swap it in place without a page reload.

        When ``inline=1`` the response is the FULL row (``_row.html``)
        instead of just the modal feedback chunk — that's what the new tab
        UI uses to update counters in place after a thumbs click without
        reopening the modal.
        """
        helpful_value: bool | None
        val = helpful.strip().lower()
        if val in {"true", "1", "yes", "up"}:
            helpful_value = True
        elif val in {"false", "0", "no", "down"}:
            helpful_value = False
        else:
            helpful_value = None
        payload = await service.feedback({"id": mem_id, "helpful": helpful_value})
        if not payload.get("ok"):
            raise HTTPException(
                404 if payload.get("code") == "not_found" else 400,
                payload.get("error", "feedback failed"),
            )
        # Fetch the memory again so the template has the full record —
        # cheap (single .md read) and keeps the chunk self-contained.
        mem_payload = await service.get({"id": mem_id})
        if not mem_payload.get("ok"):
            raise HTTPException(404, "memory vanished")
        memory = mem_payload["memory"]
        if inline.strip() in {"1", "true", "yes"}:
            lint_ids = await _lint_set()
            return templates.TemplateResponse(
                request,
                "_row.html",
                {"m": memory, "lint_ids": lint_ids},
            )
        return templates.TemplateResponse(
            request,
            "_feedback.html",
            {"m": memory},
        )

    # ----- API: tab views (Quality / Duplicates / Top / By project) --------
    #
    # Each tab is its own HTMX fragment endpoint that swaps into ``#rows``.
    # Putting them behind dedicated URLs (instead of overloading
    # ``/api/memories`` with a giant ``view`` enum) keeps the templates
    # self-contained and the URL hash routing on the client stays simple.

    @app.get("/api/quality", response_class=HTMLResponse)
    async def quality_view(request: Request):
        """Render the Quality tab — every memory ``service.lint`` flagged.

        Pre-fetches the full memory dict for each problem so the row card can
        render its title / description / counters without a second roundtrip.
        Service.lint is capped at 100 problems server-side; we honor that here.
        """
        payload = await service.lint({})
        problems = payload.get("problems", []) if payload.get("ok") else []
        # Pull the corpus once and index by id — N reads from disk is cheaper
        # than N roundtrips to ``service.get`` for large lint lists.
        corpus = await service.list_({"limit": 500})
        by_id = {m["id"]: m for m in corpus.get("memories", [])}
        items = []
        for p in problems:
            mem = by_id.get(p["id"])
            if mem:
                items.append({"memory": mem, "issues": p.get("issues", [])})
        return templates.TemplateResponse(
            request,
            "_tab_quality.html",
            {
                "items": items,
                "total_scanned": payload.get("total_scanned", 0),
                "with_issues": payload.get("with_issues", 0),
            },
        )

    @app.get("/api/duplicates", response_class=HTMLResponse)
    async def duplicates_view(
        request: Request,
        threshold: float = Query(0.7, ge=0.0, le=1.0),
    ):
        """Render the Duplicates tab — pairs with tag-jaccard ≥ threshold."""
        payload = await service.duplicates({"threshold": threshold})
        pairs_raw = payload.get("pairs", []) if payload.get("ok") else []
        corpus = await service.list_({"limit": 500})
        by_id = {m["id"]: m for m in corpus.get("memories", [])}
        pairs = []
        for p in pairs_raw:
            a, b = by_id.get(p["a"]), by_id.get(p["b"])
            if a and b:
                pairs.append({"a": a, "b": b, "jaccard": p.get("jaccard")})
        return templates.TemplateResponse(
            request,
            "_tab_duplicates.html",
            {
                "pairs": pairs,
                "threshold": threshold,
                "count": payload.get("count", 0),
            },
        )

    @app.get("/api/top", response_class=HTMLResponse)
    async def top_view(request: Request, limit: int = Query(8, ge=1, le=20)):
        """Render the Top tab — 4 sub-rankings side-by-side.

        Each ranking pulls from the same in-RAM corpus snapshot to minimize
        disk IO. ``zombie`` is the only ranking that filters before sorting;
        the other three sort the full corpus and slice the head.
        """
        corpus = await service.list_({"limit": 500})
        memories = corpus.get("memories", [])
        rankings = {
            "usage": _sort_memories(list(memories), "usage_desc")[:limit],
            "helpful": _sort_memories(list(memories), "helpful_desc")[:limit],
            "unhelpful": _sort_memories(list(memories), "unhelpful_desc")[:limit],
            "zombie": _sort_memories(list(memories), "zombie")[:limit],
        }
        # Filter out zero-count entries from helpful/unhelpful — showing all
        # zeros is noise; better to render an "all 0 — no signal yet" empty
        # state for those two columns and let usage + zombies carry the tab
        # until the feedback hooks accumulate signal.
        rankings["usage"] = [m for m in rankings["usage"] if int(m.get("usage_count") or 0) > 0]
        rankings["helpful"] = [
            m for m in rankings["helpful"] if int(m.get("helpful_count") or 0) > 0
        ]
        rankings["unhelpful"] = [
            m for m in rankings["unhelpful"] if int(m.get("unhelpful_count") or 0) > 0
        ]
        lint_ids = await _lint_set()
        return templates.TemplateResponse(
            request,
            "_tab_top.html",
            {"rankings": rankings, "limit": limit, "lint_ids": lint_ids},
        )

    @app.get("/api/by-project", response_class=HTMLResponse)
    async def by_project_view(request: Request):
        """Render the "By project" tab — group memorias by their primary tag.

        We group by the first tag the memory has under a small whitelist of
        "project-shaped" tags (lowercase, hyphenated, no whitespace). Any
        memoria without a project tag goes into a catch-all bucket so it's
        not invisible. Projects are sorted by count descending so the most
        active appears first.
        """
        corpus = await service.list_({"limit": 500})
        memories = corpus.get("memories", [])
        # Count tag frequency to surface the top N project tags. We then
        # bucket each memory by its first tag that hits the project list,
        # which usually matches the user's mental model ("this is for
        # obsidian-rag", "this is for whatsapp-listener", ...).
        tag_counts: Counter = Counter()
        for m in memories:
            for t in (m.get("tags") or []):
                tag_counts[t] += 1
        # Heuristic: a "project" tag is one of the top 20 tags AND looks
        # project-shaped (no spaces, no slashes, not a generic technique
        # word like "rag" / "frontend"). The user's vault has obsidian-rag,
        # mem-vault, whatsapp-listener, etc. as the dominant projects.
        GENERIC_TAGS = {
            "rag", "frontend", "backend", "launchd", "ollama", "feedback",
            "bug", "fix", "test", "refactor", "decision", "performance",
            "ui", "cli", "mcp", "web", "python", "config", "cache",
        }
        project_candidates = [
            t for t, _ in tag_counts.most_common(40)
            if "-" in t and t not in GENERIC_TAGS
        ]
        buckets: dict[str, list[dict[str, Any]]] = {}
        unsorted: list[dict[str, Any]] = []
        for m in memories:
            tags = m.get("tags") or []
            project = next((t for t in tags if t in project_candidates), None)
            if project:
                buckets.setdefault(project, []).append(m)
            else:
                unsorted.append(m)
        groups = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
        if unsorted:
            groups.append(("(other)", unsorted))
        lint_ids = await _lint_set()
        return templates.TemplateResponse(
            request,
            "_tab_by_project.html",
            {"groups": groups, "lint_ids": lint_ids},
        )

    # ----- API: stats -------------------------------------------------------

    def _compute_stats() -> dict[str, Any]:
        """Single-pass walk over the vault to populate both caches.

        Iterates with ``iter_memories`` so we never hold the full corpus
        in RAM — important for vaults with 10k+ memorias on iCloud.
        """
        total = 0
        by_type: Counter = Counter()
        by_agent: Counter = Counter()
        types: set[str] = set()
        tags: set[str] = set()
        for m in service.storage.iter_memories():
            total += 1
            by_type[m.type] += 1
            by_agent[m.agent_id or "—"] += 1
            types.add(m.type)
            tags.update(m.tags)
        return {
            "total": total,
            "by_type": dict(by_type),
            "by_agent": dict(by_agent),
            "types": sorted(types),
            "tags": sorted(tags),
        }

    async def _refresh_stats_cache() -> dict[str, Any]:
        async with _cache_lock:
            if _stats_fresh() and _types_fresh():
                # Another request beat us into the lock — reuse its data.
                return {
                    "total": _stats_cache["total"],
                    "by_type": _stats_cache["by_type"] or {},
                    "by_agent": _stats_cache["by_agent"] or {},
                    "types": _types_cache["types"] or [],
                    "tags": _types_cache["tags"] or [],
                }
            data = await asyncio.to_thread(_compute_stats)
            now = time.monotonic()
            _stats_cache.update(
                {
                    "ts": now,
                    "total": data["total"],
                    "by_type": data["by_type"],
                    "by_agent": data["by_agent"],
                }
            )
            _types_cache.update({"ts": now, "types": data["types"], "tags": data["tags"]})
            return data

    async def _refresh_health_cache() -> tuple[int, int]:
        """Refresh with_issues + duplicates counters in the stats cache.

        These two counters power the "Quality · N" / "Duplicates · N" pills
        in the header. They're computed via dedicated service calls (lint +
        duplicates) and cached on the same TTL as the corpus stats so a
        single header refresh isn't a triple disk walk.
        """
        lint_payload = await service.lint({})
        with_issues = int(lint_payload.get("with_issues", 0) or 0)
        dup_payload = await service.duplicates({"threshold": 0.7})
        duplicates = int(dup_payload.get("count", 0) or 0)
        async with _cache_lock:
            _stats_cache["with_issues"] = with_issues
            _stats_cache["duplicates"] = duplicates
        return with_issues, duplicates

    @app.get("/api/stats", response_class=HTMLResponse)
    async def stats(request: Request):
        if _stats_fresh():
            data = {
                "total": _stats_cache["total"],
                "by_type": _stats_cache["by_type"] or {},
                "by_agent": _stats_cache["by_agent"] or {},
                "with_issues": _stats_cache.get("with_issues", 0),
                "duplicates": _stats_cache.get("duplicates", 0),
            }
        else:
            base = await _refresh_stats_cache()
            with_issues, duplicates = await _refresh_health_cache()
            data = {
                "total": base["total"],
                "by_type": base["by_type"],
                "by_agent": base["by_agent"],
                "with_issues": with_issues,
                "duplicates": duplicates,
            }
        return templates.TemplateResponse(
            request,
            "_stats.html",
            {
                "total": data["total"],
                "by_type": data["by_type"],
                "by_agent": data["by_agent"],
                "with_issues": data["with_issues"],
                "duplicates": data["duplicates"],
            },
        )

    @app.get("/api/types")
    async def types_endpoint():
        if _types_fresh():
            return JSONResponse(
                {"types": _types_cache["types"] or [], "tags": _types_cache["tags"] or []}
            )
        data = await _refresh_stats_cache()
        return JSONResponse({"types": data["types"], "tags": data["tags"]})

    # ----- API v1: JSON endpoints for the MCP / CLI remote client ----------
    # These exist alongside the HTMX endpoints above. They expose the exact
    # same shape that ``MemVaultService`` returns when called in-process, so
    # ``mem_vault.remote.RemoteMemVaultService`` can be a drop-in HTTP-backed
    # replacement when the MCP server has its own Qdrant locked out by the
    # web server.

    @app.get("/api/v1/search")
    async def api_v1_search(
        query: str,
        k: int = Query(5, ge=1, le=50),
        type: str | None = None,
        threshold: float = Query(0.1, ge=0.0, le=1.0),
        user_id: str | None = None,
        viewer_agent_id: str | None = None,
    ):
        return JSONResponse(
            await service.search(
                {
                    "query": query,
                    "k": k,
                    "type": type,
                    "threshold": threshold,
                    "user_id": user_id,
                    "viewer_agent_id": viewer_agent_id,
                }
            )
        )

    @app.get("/api/v1/list")
    async def api_v1_list(
        type: str | None = None,
        tag: list[str] | None = Query(default=None),
        user_id: str | None = None,
        viewer_agent_id: str | None = None,
        limit: int = Query(20, ge=1, le=200),
    ):
        return JSONResponse(
            await service.list_(
                {
                    "type": type,
                    "tags": tag,
                    "user_id": user_id,
                    "viewer_agent_id": viewer_agent_id,
                    "limit": limit,
                }
            )
        )

    @app.get("/api/v1/memories/{mem_id}")
    async def api_v1_get(mem_id: str):
        return JSONResponse(await service.get({"id": mem_id}))

    @app.post("/api/v1/memories")
    async def api_v1_save(payload: MemoryCreate):
        # ``exclude_none=True`` keeps optional-but-unset fields out of the dict
        # so the service falls back to its own defaults (``config.user_id``,
        # ``auto_extract_default``, etc) instead of seeing an explicit ``None``
        # and treating it as "the caller really wants null here".
        return JSONResponse(await service.save(payload.model_dump(exclude_none=True)))

    @app.patch("/api/v1/memories/{mem_id}")
    async def api_v1_update(mem_id: str, payload: MemoryUpdate):
        args = payload.model_dump(exclude_none=True)
        args["id"] = mem_id
        return JSONResponse(await service.update(args))

    @app.delete("/api/v1/memories/{mem_id}")
    async def api_v1_delete(mem_id: str):
        return JSONResponse(await service.delete({"id": mem_id}))

    # ----- API v1: discovery / introspection tools -------------------------
    # Mirror the 9 ``MemVaultService`` methods that aren't basic CRUD so
    # ``RemoteMemVaultService`` can route every MCP tool through HTTP. Each
    # endpoint forwards its query/path/body params into a plain ``dict`` and
    # returns whatever the service returns — the JSON shape is the same the
    # in-process call would produce, by design.

    @app.get("/api/v1/briefing")
    async def api_v1_briefing(cwd: str | None = None):
        return JSONResponse(await service.briefing({"cwd": cwd}))

    @app.post("/api/v1/derive_metadata")
    async def api_v1_derive_metadata(payload: DeriveMetadataRequest):
        return JSONResponse(await service.derive_metadata(payload.model_dump(exclude_none=True)))

    @app.get("/api/v1/stats")
    async def api_v1_stats(cwd: str | None = None):
        return JSONResponse(await service.stats({"cwd": cwd}))

    @app.get("/api/v1/duplicates")
    async def api_v1_duplicates(
        threshold: float = Query(0.7, ge=0.0, le=1.0),
        cwd: str | None = None,
    ):
        return JSONResponse(await service.duplicates({"threshold": threshold, "cwd": cwd}))

    @app.get("/api/v1/lint")
    async def api_v1_lint(cwd: str | None = None):
        return JSONResponse(await service.lint({"cwd": cwd}))

    @app.get("/api/v1/memories/{mem_id}/related")
    async def api_v1_related(
        mem_id: str,
        min_shared_tags: int = Query(2, ge=1, le=10),
        k: int = Query(5, ge=1, le=50),
        include_semantic: bool = True,
    ):
        return JSONResponse(
            await service.related(
                {
                    "id": mem_id,
                    "min_shared_tags": min_shared_tags,
                    "k": k,
                    "include_semantic": include_semantic,
                }
            )
        )

    @app.get("/api/v1/memories/{mem_id}/history")
    async def api_v1_history(
        mem_id: str,
        limit: int = Query(20, ge=1, le=500),
    ):
        return JSONResponse(await service.history({"id": mem_id, "limit": limit}))

    @app.post("/api/v1/memories/{mem_id}/feedback")
    async def api_v1_feedback(mem_id: str, payload: FeedbackRequest):
        # Use ``model_dump`` (NOT ``exclude_none``): ``helpful=None`` is a
        # legitimate "just record usage" signal, not a missing field.
        return JSONResponse(await service.feedback({"id": mem_id, "helpful": payload.helpful}))

    @app.post("/api/v1/synthesize")
    async def api_v1_synthesize(payload: SynthesizeRequest):
        return JSONResponse(await service.synthesize(payload.model_dump()))

    # ----- Graph view -------------------------------------------------------

    @app.get("/graph", response_class=HTMLResponse)
    async def graph_page(request: Request):
        cfg = service.config
        return templates.TemplateResponse(
            request,
            "graph.html",
            {
                "version": __version__,
                "vault_path": str(cfg.memory_dir),
                "agent_id": cfg.agent_id or "—",
                "collection": cfg.qdrant_collection,
            },
        )

    @app.get("/api/graph")
    async def graph_data(
        min_shared_tags: int = Query(2, ge=1, le=10),
        max_nodes: int = Query(200, ge=10, le=1000),
        edges_filter: str = Query(
            "all",
            pattern=r"^(all|related|contradicts|cotag|explicit)$",
            description=(
                "Which edge kinds to render: ``all`` (default — every kind), "
                "``related``, ``contradicts``, ``cotag`` (single kind), or "
                "``explicit`` (related ∪ contradicts, no co-tag noise)."
            ),
        ),
    ):
        """Build a node/edge graph for cytoscape.js.

        v0.6.0: edges now carry a ``kind`` field — ``related`` (auto-link
        wikilinks from frontmatter), ``contradicts`` (auto-detected
        tensions), or ``cotag`` (≥ ``min_shared_tags`` shared after the
        normalization split). The frontend styles each kind differently
        so the graph stops being decorative and starts being a real
        knowledge-graph visualization.

        Nodes  = memories (limited to ``max_nodes`` most recent).
        Edges  = the union of related / contradicts / cotag (each pair
                 may carry multiple kinds, joined into one edge with
                 ``kinds=[...]``).
        """
        from mem_vault import graph as _graph

        memories = await asyncio.to_thread(
            service.storage.list, type=None, tags=None, user_id=None, limit=max_nodes
        )

        nodes = []
        for m in memories:
            nodes.append(
                {
                    "data": {
                        "id": m.id,
                        "label": m.name,
                        "type": m.type,
                        "tags": m.tags,
                        "agent_id": m.agent_id,
                        "updated": m.updated,
                        "description": m.description,
                        "usage_count": m.usage_count or 0,
                        "helpful_ratio": round(m.helpful_ratio, 2),
                    }
                }
            )

        # Pick edge kinds to render. ``explicit`` is the agent-friendly
        # default for "show me the real knowledge graph": related +
        # contradicts only, no co-tag noise from large project tags.
        if edges_filter == "all":
            include_cotag = True
            allowed: set[str] | None = None
        elif edges_filter == "related":
            include_cotag = False
            allowed = {"related"}
        elif edges_filter == "contradicts":
            include_cotag = False
            allowed = {"contradicts"}
        elif edges_filter == "cotag":
            include_cotag = True
            allowed = {"cotag"}
        else:  # explicit
            include_cotag = False
            allowed = {"related", "contradicts"}

        adj = await asyncio.to_thread(
            _graph.build_adjacency,
            memories,
            min_shared_tags=min_shared_tags,
            include_cotag=include_cotag,
        )

        # Flatten the symmetric adjacency to undirected edges (each pair
        # appears once). The kind set is preserved on the edge so the
        # frontend can style by primary kind (contradicts > related > cotag).
        edges = []
        seen_pairs: set[tuple[str, str]] = set()
        for a, neigh in adj.items():
            for b, kinds in neigh.items():
                key = (a, b) if a < b else (b, a)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                kept = kinds if allowed is None else (kinds & allowed)
                if not kept:
                    continue
                edges.append(
                    {
                        "data": {
                            "id": f"{key[0]}__{key[1]}",
                            "source": key[0],
                            "target": key[1],
                            "kinds": sorted(kept),
                            # ``primary`` defines the rendering color/style.
                            # Priority: contradicts > related > cotag.
                            "primary": (
                                "contradicts"
                                if "contradicts" in kept
                                else ("related" if "related" in kept else "cotag")
                            ),
                            "weight": len(kept),
                        }
                    }
                )

        return JSONResponse({"nodes": nodes, "edges": edges})

    # ----- Dashboard (v0.6.0) -----------------------------------------------
    #
    # ``/dashboard`` is the new landing for power users — a single page that
    # summarizes every signal mem-vault tracks across the corpus + the v0.6
    # subsystems (telemetry, ranker, antagonist queue, reflection daemon).
    # The page is server-rendered jinja2 (no HTMX islands) — a 30 s manual
    # reload is more than enough cadence for these aggregates and keeps the
    # template legible.
    #
    # ``/api/dashboard`` returns the same payload as JSON so external
    # tooling (CLI watch, external monitors) can poll it without scraping.

    async def _dashboard_payload() -> dict[str, Any]:
        """Aggregate every cross-cutting signal into one payload.

        Best-effort throughout: each subsystem (telemetry, ranker,
        antagonist, daemon) is wrapped in its own try/except so a missing
        module / corrupt state file degrades the corresponding card to
        ``None`` rather than blanking the whole dashboard.
        """
        from datetime import datetime, timedelta

        cfg = service.config

        # --- Corpus --------------------------------------------------------
        memorias = await asyncio.to_thread(
            service.storage.list, type=None, tags=None, user_id=None, limit=10**9
        )
        total = len(memorias)
        by_type: Counter = Counter()
        by_agent: Counter = Counter()
        by_project: Counter = Counter()
        tag_counts: Counter = Counter()
        now_utc = datetime.now(UTC)
        cutoff_24h = now_utc - timedelta(hours=24)
        cutoff_7d = now_utc - timedelta(days=7)
        cutoff_30d = now_utc - timedelta(days=30)
        cutoff_60d = now_utc - timedelta(days=60)
        new_24h = updated_24h = new_7d = updated_7d = new_30d = updated_30d = 0

        def _to_dt(iso: str | None):
            if not iso:
                return None
            try:
                if iso.endswith("Z"):
                    iso = iso[:-1] + "+00:00"
                dt = datetime.fromisoformat(iso)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return dt
            except Exception:
                return None

        zombies: list[dict[str, Any]] = []
        contradictions: list[dict[str, Any]] = []
        for m in memorias:
            by_type[m.type] += 1
            by_agent[m.agent_id or "—"] += 1
            for t in m.tags or []:
                tag_counts[t] += 1
                if t.startswith("project:"):
                    by_project[t.split(":", 1)[1].lower()] += 1
            created = _to_dt(m.created)
            updated = _to_dt(m.updated)
            if created and created >= cutoff_24h:
                new_24h += 1
            elif updated and updated >= cutoff_24h:
                updated_24h += 1
            if created and created >= cutoff_7d:
                new_7d += 1
            elif updated and updated >= cutoff_7d:
                updated_7d += 1
            if created and created >= cutoff_30d:
                new_30d += 1
            elif updated and updated >= cutoff_30d:
                updated_30d += 1
            if (m.usage_count or 0) == 0 and updated and updated < cutoff_60d:
                age_days = (now_utc - updated).days
                zombies.append(
                    {
                        "id": m.id,
                        "name": m.name,
                        "description": m.description,
                        "age_days": age_days,
                    }
                )
            if m.contradicts:
                contradictions.append(
                    {
                        "id": m.id,
                        "name": m.name,
                        "description": m.description,
                        "contradicts": list(m.contradicts),
                    }
                )

        zombies.sort(key=lambda z: z["age_days"], reverse=True)

        # --- Top usadas / helpful / unhelpful ------------------------------
        top_used_pairs: list[tuple[int, dict[str, Any]]] = [
            (
                m.usage_count or 0,
                {
                    "id": m.id,
                    "name": m.name,
                    "type": m.type,
                    "usage_count": m.usage_count or 0,
                    "helpful_count": m.helpful_count or 0,
                    "unhelpful_count": m.unhelpful_count or 0,
                },
            )
            for m in memorias
            if (m.usage_count or 0) > 0
        ]
        top_used_pairs.sort(key=lambda p: p[0], reverse=True)
        top_used = [d for _, d in top_used_pairs[:10]]

        top_helpful_pairs: list[tuple[int, dict[str, Any]]] = [
            (
                m.helpful_count or 0,
                {
                    "id": m.id,
                    "name": m.name,
                    "type": m.type,
                    "helpful_count": m.helpful_count or 0,
                    "ratio": round(m.helpful_ratio, 2),
                },
            )
            for m in memorias
            if (m.helpful_count or 0) > 0
        ]
        top_helpful_pairs.sort(key=lambda p: p[0], reverse=True)
        top_helpful = [d for _, d in top_helpful_pairs[:10]]

        # --- Lint + duplicates ---------------------------------------------
        lint_payload = {}
        try:
            lint_payload = await service.lint({})
        except Exception as exc:
            logger.warning("dashboard: lint failed: %s", exc)
        lint_problems = (lint_payload or {}).get("problems") or []

        dup_pairs: list[dict[str, Any]] = []
        try:
            from mem_vault.discovery import find_duplicate_pairs_by_tag_overlap

            for a, b, score in find_duplicate_pairs_by_tag_overlap(
                memorias, threshold=0.7
            )[:10]:
                dup_pairs.append({"a": a, "b": b, "score": round(score, 3)})
        except Exception as exc:
            logger.warning("dashboard: duplicates failed: %s", exc)

        # --- Knowledge graph -----------------------------------------------
        graph_stats = {
            "nodes": total,
            "related": 0,
            "contradicts": 0,
            "cotag": 0,
        }
        try:
            from mem_vault import graph as _graph

            adj = await asyncio.to_thread(
                _graph.build_adjacency, memorias, min_shared_tags=2, include_cotag=True
            )
            seen: set[tuple[str, str]] = set()
            for a, neigh in adj.items():
                for b, kinds in neigh.items():
                    key = (a, b) if a < b else (b, a)
                    if key in seen:
                        continue
                    seen.add(key)
                    if "contradicts" in kinds:
                        graph_stats["contradicts"] += 1
                    if "related" in kinds:
                        graph_stats["related"] += 1
                    if "cotag" in kinds:
                        graph_stats["cotag"] += 1
        except Exception as exc:
            logger.warning("dashboard: graph failed: %s", exc)

        # --- Telemetry -----------------------------------------------------
        telemetry_snap: dict[str, Any] | None = None
        try:
            from mem_vault import telemetry as _telemetry

            telemetry_snap = _telemetry.stats(cfg.state_dir)
        except Exception as exc:
            logger.warning("dashboard: telemetry stats failed: %s", exc)
            telemetry_snap = {"error": str(exc)}

        # --- Ranker meta ---------------------------------------------------
        ranker_snap: dict[str, Any] | None = None
        try:
            import json

            from mem_vault import ranker as _ranker

            active = _ranker.active_pickle_path(cfg.state_dir)
            meta = _ranker.meta_path(cfg.state_dir)
            ranker_snap = {
                "enabled": _ranker.is_enabled(),
                "active_pickle_exists": active.exists(),
                "active_pickle_path": str(active),
            }
            if meta.exists():
                try:
                    blob = json.loads(meta.read_text(encoding="utf-8"))
                    ranker_snap.update(
                        {
                            "version": blob.get("version"),
                            "n_train": blob.get("n_train"),
                            "n_positive": blob.get("n_positive"),
                            "n_negative": blob.get("n_negative"),
                            "auc": blob.get("auc"),
                            "trained_at": blob.get("trained_at"),
                        }
                    )
                except Exception as exc:
                    logger.warning("dashboard: ranker meta parse failed: %s", exc)
        except Exception as exc:
            logger.warning("dashboard: ranker meta failed: %s", exc)
            ranker_snap = {"error": str(exc)}

        # --- Antagonist queue ----------------------------------------------
        antagonist_snap: dict[str, Any] | None = None
        try:
            from mem_vault import antagonist as _antagonist

            pending = _antagonist.read_pending(cfg.state_dir, ttl_s=10**9)  # show all, even expired
            antagonist_snap = {
                "enabled": _antagonist.is_enabled(),
                "pending_count": len(pending),
                "items": [p.to_dict() for p in pending[:5]],
            }
        except Exception as exc:
            logger.warning("dashboard: antagonist failed: %s", exc)
            antagonist_snap = {"error": str(exc)}

        # --- Reflection daemon + last reflections --------------------------
        reflections: list[dict[str, Any]] = []
        for m in memorias:
            if not m.id.startswith("reflection_"):
                continue
            reflections.append(
                {
                    "id": m.id,
                    "name": m.name,
                    "description": m.description,
                    "updated": m.updated,
                }
            )
        reflections.sort(key=lambda r: r.get("updated") or "", reverse=True)
        reflections = reflections[:7]

        daemon_snap: dict[str, Any] | None = None
        try:
            import platform
            import subprocess

            label = "com.mem-vault.reflect"
            sys_name = platform.system()
            loaded = None
            detail = ""
            if sys_name == "Darwin":
                try:
                    cp = subprocess.run(
                        ["launchctl", "print", f"gui/{__import__('os').getuid()}/{label}"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    loaded = cp.returncode == 0
                    detail = "launchd loaded" if loaded else "launchd not loaded"
                except Exception as exc:
                    detail = f"launchctl probe failed: {exc}"
            elif sys_name == "Linux":
                try:
                    cp = subprocess.run(
                        ["systemctl", "--user", "is-active", "mem-vault-reflect.timer"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    loaded = cp.returncode == 0
                    detail = cp.stdout.strip() or detail
                except Exception as exc:
                    detail = f"systemctl probe failed: {exc}"
            daemon_snap = {
                "system": sys_name,
                "label": label,
                "loaded": loaded,
                "detail": detail,
                "last_reflection": reflections[0]["id"] if reflections else None,
                "last_reflection_at": reflections[0]["updated"] if reflections else None,
            }
        except Exception as exc:
            logger.warning("dashboard: daemon probe failed: %s", exc)
            daemon_snap = {"error": str(exc)}

        return {
            "version": __version__,
            "vault_path": str(cfg.memory_dir),
            "agent_id": cfg.agent_id,
            "user_id": cfg.user_id,
            "collection": cfg.qdrant_collection,
            "totals": {
                "memorias": total,
                "by_type": dict(by_type.most_common()),
                "by_agent": dict(by_agent.most_common()),
                "by_project": dict(by_project.most_common(10)),
                "top_tags": tag_counts.most_common(15),
            },
            "activity": {
                "new_24h": new_24h,
                "updated_24h": updated_24h,
                "new_7d": new_7d,
                "updated_7d": updated_7d,
                "new_30d": new_30d,
                "updated_30d": updated_30d,
            },
            "graph": graph_stats,
            "top_used": top_used,
            "top_helpful": top_helpful,
            "zombies": zombies[:10],
            "contradictions": contradictions[:10],
            "lint": {
                "count": len(lint_problems),
                "problems": [
                    {"id": p.get("id"), "issues": p.get("issues") or p}
                    for p in lint_problems[:10]
                ],
            },
            "duplicates": {
                "count": len(dup_pairs),
                "pairs": dup_pairs,
            },
            "telemetry": telemetry_snap,
            "ranker": ranker_snap,
            "antagonist": antagonist_snap,
            "daemon": daemon_snap,
            "reflections": reflections,
        }

    @app.get("/api/overview", response_class=HTMLResponse)
    async def overview_fragment(request: Request):
        """HTMX fragment with the dashboard content (rendered into ``#rows``).

        v0.6.0: the standalone ``/dashboard`` page is gone; the same
        information lives as a tab inside the main ``/`` page so all
        mem-vault info stays at a single URL.
        """
        payload = await _dashboard_payload()
        return templates.TemplateResponse(
            request,
            "_tab_overview.html",
            {"d": payload},
        )

    @app.get("/api/dashboard")
    async def dashboard_api():
        """JSON payload of the overview tab — for CLI watch / external monitors.

        Same data the ``overview`` tab consumes; exposed for tooling that
        wants programmatic access without scraping HTML.
        """
        return JSONResponse(await _dashboard_payload())

    return app


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def serve(host: str = "127.0.0.1", port: int = 7880, log_level: str = "info") -> None:
    """Run the UI server with uvicorn. Blocks until killed.

    Refuses to start when binding to a non-loopback host without
    ``MEM_VAULT_HTTP_TOKEN`` set — that combination would expose an
    unauthenticated CRUD API to whatever network the address reaches.
    Set the env var (and pass the same token in the ``Authorization``
    header on every request) to opt in to LAN/remote exposure.
    """
    import uvicorn

    config = load_config()
    if not _is_loopback_host(host) and not (config.http_token or "").strip():
        raise SystemExit(
            f"Refusing to bind to non-loopback host {host!r} without "
            "MEM_VAULT_HTTP_TOKEN set. Set the env var or pass --host 127.0.0.1."
        )
    service = MemVaultService(config)
    app = create_app(service)
    banner = f"\n  mem-vault UI · http://{host}:{port}"
    if (config.http_token or "").strip():
        banner += "  [auth: bearer token required]"
    print(banner + "\n")
    uvicorn.run(app, host=host, port=port, log_level=log_level)

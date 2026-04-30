"""MCP stdio server exposing the mem-vault memory tools.

Tools (all writes go to the Obsidian vault as `.md`; reads come from a mix of
filesystem + the local Qdrant index):

- ``memory_save``      — persist a new memory (literal or auto-extracted)
- ``memory_search``    — semantic search across memories
- ``memory_list``      — list memories with metadata filters
- ``memory_get``       — read one memory by id
- ``memory_update``    — replace body/title/tags of an existing memory
- ``memory_delete``    — remove a memory (file + index)

Run with ``mem-vault-mcp`` (after ``uv tool install --editable .``) or
``uv run python -m mem_vault.server`` from inside this repo.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

from mem_vault.config import Config, load_config
from mem_vault.index import CircuitBreakerOpenError, VectorIndex, compute_content_hash
from mem_vault.storage import VaultStorage, slugify

logger = logging.getLogger("mem_vault.server")

SERVER_NAME = "mem-vault"
SERVER_VERSION = "0.2.0"


class _ContentTooLargeError(ValueError):
    """Raised when ``content`` exceeds ``Config.max_content_size``."""


class _LLMTimeoutError(TimeoutError):
    """Raised when an Ollama-backed mem0 call exceeded ``Config.llm_timeout_s``.

    Distinct from :class:`asyncio.TimeoutError` so handlers can render a
    user-friendly message instead of a bare ``TimeoutError`` on the wire.
    """


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_TOOLS: list[types.Tool] = [
    types.Tool(
        name="memory_save",
        description=(
            "Persist a memory in the user's Obsidian vault as a markdown file with "
            "YAML frontmatter, then index it locally for semantic search. Use this "
            "to remember decisions, preferences, bug fixes, conventions, or any "
            "piece of context that should survive the current session.\n\n"
            "Set auto_extract=true to let an Ollama LLM extract canonical facts "
            "and dedupe against existing memories (slower, smarter). Default "
            "(false) saves the literal content (faster, deterministic)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The full memory body in markdown.",
                },
                "title": {
                    "type": "string",
                    "description": "Short human-readable title (defaults to first line of content).",
                },
                "description": {
                    "type": "string",
                    "description": "One-line synopsis (defaults to first ~200 chars of content).",
                },
                "type": {
                    "type": "string",
                    "enum": ["feedback", "preference", "decision", "fact", "note", "bug", "todo"],
                    "default": "note",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for filtering later.",
                },
                "auto_extract": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, run the LLM extractor + dedup. If false, save literally.",
                },
                "visible_to": {
                    "description": (
                        "Which agents can see this memory. Defaults to public. Pass "
                        "the string 'private' to scope it to the saving agent only, "
                        "'public' for everyone (default), or a list of agent ids "
                        "(['claude-code', 'cursor']) for explicit allowlist. The "
                        "saving agent is always implicitly included."
                    ),
                    "oneOf": [
                        {"type": "string", "enum": ["public", "private"]},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                },
                "user_id": {"type": "string"},
                "agent_id": {"type": "string"},
            },
            "required": ["content"],
        },
    ),
    types.Tool(
        name="memory_search",
        description=(
            "Semantic search across all memories using local embeddings. Returns "
            "the top-k most relevant memories with their full content. Useful at "
            "the start of a session to recover relevant context, or before "
            "answering questions that depend on past decisions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query."},
                "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                "type": {
                    "type": "string",
                    "description": "Optional type filter.",
                    "enum": ["feedback", "preference", "decision", "fact", "note", "bug", "todo"],
                },
                "user_id": {"type": "string"},
                "threshold": {
                    "type": "number",
                    "default": 0.1,
                    "description": "Minimum similarity score (0-1). Lower = more lenient.",
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="memory_list",
        description=(
            "List memories sorted by most-recently-modified. Optional filters by "
            "type, tags, or user_id. Use this to browse without a specific query."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["feedback", "preference", "decision", "fact", "note", "bug", "todo"],
                },
                "tags": {"type": "array", "items": {"type": "string"}},
                "user_id": {"type": "string"},
                "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 200},
            },
        },
    ),
    types.Tool(
        name="memory_get",
        description="Read a single memory by id (the file slug, e.g. `feedback_local_free_stack`).",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Memory id (filename without .md)."}
            },
            "required": ["id"],
        },
    ),
    types.Tool(
        name="memory_update",
        description=(
            "Replace fields on an existing memory. Any field omitted is left "
            "unchanged. The `updated` timestamp is bumped automatically."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "content": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["id"],
        },
    ),
    types.Tool(
        name="memory_delete",
        description=(
            "Permanently delete a memory (its .md file + every embedding pointing "
            "to it). This is irreversible — confirm with the user before calling "
            "in agent flows."
        ),
        inputSchema={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


class MemVaultService:
    """Glue between MCP tool calls, vault storage, and the vector index."""

    def __init__(self, config: Config):
        self.config = config
        self.storage = VaultStorage(config.memory_dir)
        self.index = VectorIndex(config)

    async def _to_thread(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def _index_call(self, fn, *args, **kwargs):
        """Run an Ollama-backed index call with the configured wall-clock timeout.

        ``Config.llm_timeout_s == 0`` disables the timeout (legacy behavior).
        On timeout we raise :class:`_LLMTimeoutError` so handlers can return
        a structured error rather than letting the MCP call hang on a dead
        Ollama. The underlying thread will keep running in the background
        until Ollama responds or dies — Python can't kill threads — but the
        caller is unblocked immediately.
        """
        timeout = float(self.config.llm_timeout_s or 0)
        if timeout <= 0:
            return await asyncio.to_thread(fn, *args, **kwargs)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(fn, *args, **kwargs),
                timeout=timeout,
            )
        except TimeoutError as exc:
            # Tick the breaker so a dead Ollama trips it after a few attempts.
            self.index.breaker.record_failure()
            raise _LLMTimeoutError(
                f"Ollama call exceeded {timeout:.0f}s timeout. "
                "Check `ollama serve` is running and the model is loaded."
            ) from exc

    def _check_content_size(self, content: str | None) -> None:
        """Reject oversized content before it reaches the vault or the index."""
        if content is None:
            return
        limit = int(self.config.max_content_size or 0)
        if limit <= 0:
            return
        if len(content) > limit:
            raise _ContentTooLargeError(
                f"content too large: {len(content)} chars exceeds limit of {limit}. "
                "Raise MEM_VAULT_MAX_CONTENT_SIZE or split the memory."
            )

    async def save(self, args: dict[str, Any]) -> dict[str, Any]:
        content: str = args["content"]
        try:
            self._check_content_size(content)
        except _ContentTooLargeError as exc:
            return {"ok": False, "error": str(exc), "code": "content_too_large"}
        title = args.get("title")
        description = args.get("description")
        mtype = args.get("type", "note")
        tags = args.get("tags") or []
        user_id = args.get("user_id") or self.config.user_id
        agent_id = args.get("agent_id") or self.config.agent_id
        auto_extract = bool(args.get("auto_extract", self.config.auto_extract_default))
        # Visibility: caller can pass an explicit list, or one of the shorthands.
        # When ``visible_to`` is omitted, default to public ("*"). Mostly we want
        # multi-agent setups to opt-in to privacy, not the other way around.
        visible_to = args.get("visible_to")
        if visible_to == "private":
            visible_to = []
        elif visible_to == "public" or visible_to is None:
            visible_to = ["*"]

        mem = await self._to_thread(
            self.storage.save,
            content=content,
            title=title,
            description=description,
            type=mtype,
            tags=tags,
            agent_id=agent_id,
            user_id=user_id,
            visible_to=visible_to,
        )

        index_results: list[dict[str, Any]] = []
        try:
            index_results = await self._index_call(
                self.index.add,
                content,
                user_id=user_id,
                agent_id=agent_id,
                metadata={
                    "memory_id": mem.id,
                    "type": mtype,
                    "tags": tags,
                    "content_hash": compute_content_hash(content),
                },
                auto_extract=auto_extract,
            )
        except _LLMTimeoutError as exc:
            logger.warning("indexing timed out for memory %s: %s", mem.id, exc)
            return {
                "ok": True,
                "indexed": False,
                "indexing_error": str(exc),
                "indexing_error_code": "llm_timeout",
                "memory": mem.to_dict(),
                "path": str(self.storage.path_for(mem.id)),
            }
        except CircuitBreakerOpenError as exc:
            logger.warning("indexing short-circuited for memory %s: %s", mem.id, exc)
            return {
                "ok": True,
                "indexed": False,
                "indexing_error": str(exc),
                "indexing_error_code": "circuit_breaker_open",
                "memory": mem.to_dict(),
                "path": str(self.storage.path_for(mem.id)),
            }
        except Exception as exc:
            logger.exception("indexing failed for memory %s: %s", mem.id, exc)
            return {
                "ok": True,
                "indexed": False,
                "indexing_error": str(exc),
                "memory": mem.to_dict(),
                "path": str(self.storage.path_for(mem.id)),
            }

        return {
            "ok": True,
            "indexed": True,
            "memory": mem.to_dict(),
            "path": str(self.storage.path_for(mem.id)),
            "index_entries": len(index_results),
            "auto_extract": auto_extract,
        }

    async def search(self, args: dict[str, Any]) -> dict[str, Any]:
        query: str = args["query"]
        k = int(args.get("k", 5))
        mtype = args.get("type")
        user_id = args.get("user_id") or self.config.user_id
        threshold = float(args.get("threshold", 0.1))
        # Visibility filtering happens AFTER the vector search, on the
        # filesystem side. Reason: the embedding metadata stored by mem0 is
        # a flat dict, and post-filtering against the canonical .md file is
        # cheap (we already need to load the body for the response).
        viewer_agent_id = args.get("viewer_agent_id")
        if viewer_agent_id is None:
            viewer_agent_id = self.config.agent_id

        filters: dict[str, Any] = {}
        if mtype:
            filters["type"] = mtype

        # Over-fetch so visibility filtering doesn't leave us short.
        raw_k = max(k, k * 3, 20)
        try:
            hits = await self._index_call(
                self.index.search,
                query,
                user_id=user_id,
                top_k=raw_k,
                filters=filters or None,
                threshold=threshold,
            )
        except _LLMTimeoutError as exc:
            # search() catches its own breaker errors and returns []; we
            # only end up here if the asyncio.wait_for tripped first.
            logger.warning("search timed out: %s", exc)
            return {"ok": True, "query": query, "count": 0, "results": [], "warning": str(exc)}

        # Resolve hits → full memory bodies from the vault.
        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for hit in hits:
            if len(results) >= k:
                break
            md = (hit.get("metadata") or {}) if isinstance(hit, dict) else {}
            mem_id = md.get("memory_id")
            if not mem_id:
                # Fall back: derive from content slug (best-effort)
                txt = hit.get("memory") or hit.get("text") or ""
                mem_id = slugify(txt[:80]) if txt else None
            if not mem_id or mem_id in seen_ids:
                continue
            seen_ids.add(mem_id)
            mem = await self._to_thread(self.storage.get, mem_id)
            if mem is not None and not mem.is_visible_to(viewer_agent_id):
                continue
            results.append(
                {
                    "id": mem_id,
                    "score": hit.get("score") if isinstance(hit, dict) else None,
                    "memory": mem.to_dict() if mem else None,
                    "snippet": hit.get("memory") or hit.get("text")
                    if isinstance(hit, dict)
                    else None,
                }
            )

        return {"ok": True, "query": query, "count": len(results), "results": results}

    async def list_(self, args: dict[str, Any]) -> dict[str, Any]:
        mtype = args.get("type")
        tags = args.get("tags")
        user_id = args.get("user_id")
        limit = int(args.get("limit", 20))
        viewer_agent_id = args.get("viewer_agent_id")
        if viewer_agent_id is None:
            viewer_agent_id = self.config.agent_id
        memories = await self._to_thread(
            self.storage.list,
            type=mtype,
            tags=tags,
            user_id=user_id,
            viewer_agent_id=viewer_agent_id,
            limit=limit,
        )
        return {
            "ok": True,
            "count": len(memories),
            "memories": [m.to_dict() for m in memories],
        }

    async def get(self, args: dict[str, Any]) -> dict[str, Any]:
        mem = await self._to_thread(self.storage.get, args["id"])
        if mem is None:
            return {"ok": False, "error": f"Memory not found: {args['id']}"}
        return {"ok": True, "memory": mem.to_dict()}

    async def update(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            self._check_content_size(args.get("content"))
        except _ContentTooLargeError as exc:
            return {"ok": False, "error": str(exc), "code": "content_too_large"}
        try:
            mem = await self._to_thread(
                self.storage.update,
                args["id"],
                content=args.get("content"),
                title=args.get("title"),
                description=args.get("description"),
                tags=args.get("tags"),
            )
        except FileNotFoundError as exc:
            return {"ok": False, "error": str(exc)}

        # Re-index the new body so search reflects the update.
        if args.get("content") is not None:
            user_id = self.config.user_id
            try:
                await self._to_thread(self.index.delete_by_metadata, "memory_id", mem.id, user_id)
                await self._index_call(
                    self.index.add,
                    mem.body,
                    user_id=user_id,
                    agent_id=self.config.agent_id,
                    metadata={
                        "memory_id": mem.id,
                        "type": mem.type,
                        "tags": mem.tags,
                        "content_hash": compute_content_hash(mem.body),
                    },
                    auto_extract=False,
                )
            except (_LLMTimeoutError, CircuitBreakerOpenError) as exc:
                # The .md file is updated; only the index is stale. The
                # caller can `mem-vault reindex` later, or the next save
                # will trip the breaker shut anyway.
                logger.warning("re-index after update degraded: %s", exc)
            except Exception as exc:
                logger.warning("re-index after update failed: %s", exc)

        return {"ok": True, "memory": mem.to_dict()}

    async def delete(self, args: dict[str, Any]) -> dict[str, Any]:
        mem_id = args["id"]
        existed = await self._to_thread(self.storage.delete, mem_id)
        if not existed:
            return {"ok": False, "error": f"Memory not found: {mem_id}"}
        try:
            removed = await self._to_thread(
                self.index.delete_by_metadata, "memory_id", mem_id, self.config.user_id
            )
        except Exception as exc:
            logger.warning("index cleanup for %s failed: %s", mem_id, exc)
            removed = 0
        return {"ok": True, "deleted_file": True, "deleted_index_entries": removed}


# ---------------------------------------------------------------------------
# MCP wiring
# ---------------------------------------------------------------------------


def _build_server(service: Any) -> Server:
    server: Server = Server(SERVER_NAME, version=SERVER_VERSION)

    handlers = {
        "memory_save": service.save,
        "memory_search": service.search,
        "memory_list": service.list_,
        "memory_get": service.get,
        "memory_update": service.update,
        "memory_delete": service.delete,
    }

    # Metrics sink: disabled by default, opt-in via Config.metrics_enabled.
    # We attach it unconditionally so the call_tool wrapper has one less
    # branch — the sink itself short-circuits when disabled.
    sink = _build_metrics_sink(service)

    @server.list_tools()
    async def _list() -> list[types.Tool]:
        return _TOOLS

    @server.call_tool()
    async def _call(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        args = arguments or {}
        handler = handlers.get(name)

        async def _invoke() -> dict[str, Any]:
            if handler is None:
                return {"ok": False, "error": f"Unknown tool: {name}"}
            try:
                return await handler(args)
            except Exception as exc:
                logger.exception("tool %s failed", name)
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        from mem_vault.metrics import time_async_call

        payload = await time_async_call(sink, name, _invoke)
        return [
            types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
        ]

    return server


def _build_metrics_sink(service: Any):
    """Create a ``MetricsSink`` from ``service.config`` when present.

    Falls back to a disabled sink when the service doesn't expose a
    config (remote stubs in tests, etc.) — this keeps ``_build_server``
    a no-op for non-local services.
    """
    from mem_vault.metrics import MetricsSink

    cfg = getattr(service, "config", None)
    if cfg is None or not getattr(cfg, "metrics_enabled", False):
        # Path won't be touched while disabled; supply a benign default
        # so type checkers stay happy.
        return MetricsSink(Path("/dev/null"), enabled=False)
    return MetricsSink(cfg.metrics_path, enabled=True)


def build_service(config: Any | None = None):
    """Pick between in-process MemVaultService and the HTTP RemoteMemVaultService.

    When ``MEM_VAULT_REMOTE_URL`` is set we route every call through HTTP to
    a long-lived web server that owns the Qdrant lock. This is what makes
    the MCP server (spawned per tool call by Devin) coexist peacefully with
    the obsidian-rag web server's mounted ``/memory`` UI.

    The function is the single dispatch point for the CLI, the MCP server,
    and the lifecycle hooks — keep it that way.
    """
    remote = os.environ.get("MEM_VAULT_REMOTE_URL", "").strip()
    if remote:
        from mem_vault.remote import RemoteMemVaultService  # local import: optional dep

        return RemoteMemVaultService(remote, config=config)
    if config is None:
        config = load_config()
    return MemVaultService(config)


async def _amain() -> None:
    logging.basicConfig(
        level=os.environ.get("MEM_VAULT_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        stream=sys.stderr,
    )
    remote = os.environ.get("MEM_VAULT_REMOTE_URL", "").strip()
    if remote:
        # Remote mode: we don't need vault config at all — the remote
        # server is the one that holds it. We still try to ``load_config``
        # so the user gets a clear error if something is missing, but we
        # tolerate the failure (vault path is irrelevant to remote calls).
        logger.info("mem-vault starting (remote mode) · base_url=%s", remote)
        try:
            config = load_config()
        except Exception:
            config = None
        service = build_service(config)
    else:
        config = load_config()
        logger.info(
            "mem-vault starting (local mode) · vault=%s · ollama=%s · llm=%s · embedder=%s · collection=%s",
            config.vault_path,
            config.ollama_host,
            config.llm_model,
            config.embedder_model,
            config.qdrant_collection,
        )
        service = build_service(config)
    server = _build_server(service)
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

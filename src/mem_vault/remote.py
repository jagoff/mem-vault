"""HTTP client for a remote mem-vault instance.

Why this exists: Qdrant in embedded mode (the default for mem-vault) holds an
exclusive lock on its storage directory. If two processes try to open the
same directory — for example, the obsidian-rag web server with the UI
mounted at ``/memory`` AND the MCP server spawned by Devin per tool call —
only one wins, the other silently returns empty results.

The fix: make the long-lived process (the web server) own the Qdrant lock,
and have every other client (MCP server, CLI) talk to it via HTTP. This
module is the HTTP client. It mirrors the public surface of
:class:`mem_vault.server.MemVaultService` (``save``, ``search``, ``list_``,
``get``, ``update``, ``delete``) so callers can swap implementations behind
an env var (``MEM_VAULT_REMOTE_URL``) without changing any other code.

The endpoints it talks to live in :mod:`mem_vault.ui.server` under
``/api/v1/*`` and return the same JSON shape that ``MemVaultService``
methods produce in-process.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 10.0


class RemoteMemVaultService:
    """Drop-in replacement for ``MemVaultService`` that hits a remote HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        config: Any | None = None,
    ):
        # Strip trailing slash so we can join endpoints with f-strings safely.
        self.base_url = base_url.rstrip("/")
        # Some callers (the SessionStart hook, the smoke test) reach for
        # ``service.config.memory_dir`` even though the remote service
        # doesn't really need a vault path. We keep an optional reference
        # to the config so those code paths still work.
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            follow_redirects=False,
        )

    # ---- shape-compatible "service" API ----------------------------------

    async def save(self, args: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/api/v1/memories", json=args)

    async def search(self, args: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "query": args["query"],
            "k": int(args.get("k", 5)),
            "threshold": float(args.get("threshold", 0.1)),
        }
        if args.get("type"):
            params["type"] = args["type"]
        if args.get("user_id"):
            params["user_id"] = args["user_id"]
        if args.get("viewer_agent_id"):
            params["viewer_agent_id"] = args["viewer_agent_id"]
        return await self._get("/api/v1/search", params=params)

    async def list_(self, args: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": int(args.get("limit", 20))}
        if args.get("type"):
            params["type"] = args["type"]
        if args.get("user_id"):
            params["user_id"] = args["user_id"]
        if args.get("viewer_agent_id"):
            params["viewer_agent_id"] = args["viewer_agent_id"]
        if args.get("tags"):
            # FastAPI's Query(default=None) accepts repeated ?tag=foo&tag=bar
            params["tag"] = list(args["tags"])
        return await self._get("/api/v1/list", params=params)

    async def get(self, args: dict[str, Any]) -> dict[str, Any]:
        mem_id = args["id"]
        return await self._get(f"/api/v1/memories/{mem_id}")

    async def update(self, args: dict[str, Any]) -> dict[str, Any]:
        mem_id = args["id"]
        body = {k: v for k, v in args.items() if k != "id"}
        return await self._patch(f"/api/v1/memories/{mem_id}", json=body)

    async def delete(self, args: dict[str, Any]) -> dict[str, Any]:
        mem_id = args["id"]
        return await self._delete(f"/api/v1/memories/{mem_id}")

    # ---- low-level helpers -----------------------------------------------

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._request("GET", path, params=params)

    async def _post(self, path: str, *, json: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._request("POST", path, json=json)

    async def _patch(self, path: str, *, json: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._request("PATCH", path, json=json)

    async def _delete(self, path: str) -> dict[str, Any]:
        return await self._request("DELETE", path)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            resp = await self._client.request(method, path, params=params, json=json)
        except httpx.ConnectError as exc:
            return {
                "ok": False,
                "error": (
                    f"mem-vault remote server unreachable at {self.base_url}: {exc}. "
                    "Is the obsidian-rag web server running? "
                    "Check `lsof -i :8765` or unset MEM_VAULT_REMOTE_URL to fall back to local mode."
                ),
            }
        except httpx.TimeoutException as exc:
            return {"ok": False, "error": f"mem-vault remote timeout: {exc}"}
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception("remote request failed: %s %s", method, path)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        if resp.status_code >= 500:
            return {
                "ok": False,
                "error": f"mem-vault remote returned {resp.status_code}: {resp.text[:200]}",
            }
        try:
            data = resp.json()
        except ValueError:
            return {"ok": False, "error": f"non-JSON response from {path}"}
        return data

    async def close(self) -> None:
        await self._client.aclose()

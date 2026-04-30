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

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Status codes que sí re-tryeamos. 502/503/504 son los típicos de un upstream
# transitorio (gateway down, servicio reiniciando, timeout entre proxies).
# 500 NO se re-trya — suele ser un bug en el server, no algo transitorio.
_RETRYABLE_STATUS = {502, 503, 504}

# Backoff entre intento N y N+1 (segundos). Lista corta = max 3 intentos
# (initial + 2 retries). Si max_retries=2: backoff[0]=0.5s antes del 1er
# retry, backoff[1]=1.0s antes del 2do retry. Si max_retries=0 ⇒ no se
# tocan estos delays porque no hay retries.
_RETRY_BACKOFF_S = (0.5, 1.0)

# Default timeout para read+write+connect cubriendo cualquier endpoint del
# servidor remoto. ANTES era 10s, lo cual rompía `save({auto_extract: true})`
# casi siempre: cuando el server delega al LLM extractor (Ollama qwen2.5:7b
# u otro), una sola llamada de extract+dedup contra contenido de varios KB
# tarda 30-120s; el timeout disparaba antes y devolvía
# `{"ok": false, "error": "mem-vault remote timeout: "}` aunque el server
# del otro lado SÍ estuviera trabajando bien.
#
# 180s cubre el caso del LLM con margen sin colgarnos eternamente si algo
# realmente falla. Operaciones rápidas (search/list/get/delete = <1s
# normalmente) NO se ven afectadas — el timeout es un MAX, no un MIN.
#
# Para operaciones que SÍ querramos rápidas a propósito (ej. health checks)
# se puede pasar `timeout=N` explícito al constructor del cliente.
DEFAULT_TIMEOUT_S = 180.0


class RemoteMemVaultService:
    """Drop-in replacement for ``MemVaultService`` that hits a remote HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        config: Any | None = None,
        token: str | None = None,
        max_retries: int = 2,
    ):
        # Strip trailing slash so we can join endpoints with f-strings safely.
        self.base_url = base_url.rstrip("/")
        # Some callers (the SessionStart hook, the smoke test) reach for
        # ``service.config.memory_dir`` even though the remote service
        # doesn't really need a vault path. We keep an optional reference
        # to the config so those code paths still work.
        self.config = config
        # Cap retries a algo razonable y NO permitir negativos (un usuario
        # que pase -1 esperando "infinito" sería un foot-gun).
        self.max_retries = max(0, int(max_retries))
        # Bearer token resolution order: explicit ``token`` arg > Config
        # field > env var. The env-var fallback exists so the hooks (which
        # build the client without seeing the user's local config) can
        # still authenticate against a token-protected web server.
        resolved_token = token
        if resolved_token is None and config is not None:
            resolved_token = getattr(config, "http_token", None)
        if resolved_token is None:
            resolved_token = os.environ.get("MEM_VAULT_HTTP_TOKEN")
        headers: dict[str, str] = {}
        if resolved_token:
            headers["Authorization"] = f"Bearer {resolved_token.strip()}"
            # Plain-HTTP + bearer token = el token viaja en plaintext en cada
            # request. No fail (a veces tiene sentido en una red privada o
            # en localhost dev), pero queremos que el usuario lo vea en logs.
            if self.base_url.lower().startswith("http://"):
                logger.warning(
                    "RemoteMemVaultService: bearer token will be transmitted in "
                    "PLAINTEXT because base_url uses http:// (got %r). Use https:// "
                    "or unset the token (MEM_VAULT_HTTP_TOKEN / config.http_token / "
                    "token=...) to avoid leaking credentials on the wire.",
                    self.base_url,
                )
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            follow_redirects=False,
            headers=headers,
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

    # ---- discovery / introspection (mirror MemVaultService) --------------
    #
    # Each of these maps 1:1 to a tool declared in ``server._TOOLS`` so the
    # ``_build_handlers`` symmetry check passes when the MCP server runs in
    # remote mode. The endpoints they hit are defined in
    # ``mem_vault/ui/server.py`` under the same ``/api/v1/*`` prefix as the
    # CRUD endpoints above.

    async def briefing(self, args: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if args.get("cwd"):
            params["cwd"] = args["cwd"]
        return await self._get("/api/v1/briefing", params=params)

    async def derive_metadata(self, args: dict[str, Any]) -> dict[str, Any]:
        # ``content`` can be multi-KB, so this is a POST body — mirror
        # the in-process arg shape and let the server's Pydantic model
        # do the validation.
        body: dict[str, Any] = {"content": args.get("content", "")}
        if args.get("cwd"):
            body["cwd"] = args["cwd"]
        return await self._post("/api/v1/derive_metadata", json=body)

    async def stats(self, args: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if args.get("cwd"):
            params["cwd"] = args["cwd"]
        return await self._get("/api/v1/stats", params=params)

    async def duplicates(self, args: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {"threshold": float(args.get("threshold", 0.7))}
        if args.get("cwd"):
            params["cwd"] = args["cwd"]
        return await self._get("/api/v1/duplicates", params=params)

    async def lint(self, args: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if args.get("cwd"):
            params["cwd"] = args["cwd"]
        return await self._get("/api/v1/lint", params=params)

    async def related(self, args: dict[str, Any]) -> dict[str, Any]:
        mem_id = args.get("id")
        if not mem_id:
            # Mirror the in-process service's validation envelope so the
            # MCP dispatcher gets a consistent error shape regardless of
            # which side of the HTTP boundary the check fires on.
            return {"ok": False, "error": "id is required", "code": "validation_failed"}
        params: dict[str, Any] = {
            "min_shared_tags": int(args.get("min_shared_tags", 2)),
            "k": int(args.get("k", 5)),
            "include_semantic": bool(args.get("include_semantic", True)),
        }
        return await self._get(f"/api/v1/memories/{mem_id}/related", params=params)

    async def history(self, args: dict[str, Any]) -> dict[str, Any]:
        mem_id = args.get("id")
        if not mem_id:
            return {"ok": False, "error": "id is required", "code": "validation_failed"}
        params: dict[str, Any] = {"limit": int(args.get("limit", 20))}
        return await self._get(f"/api/v1/memories/{mem_id}/history", params=params)

    async def feedback(self, args: dict[str, Any]) -> dict[str, Any]:
        mem_id = args.get("id")
        if not mem_id:
            return {"ok": False, "error": "id is required", "code": "validation_failed"}
        # ``helpful`` may be ``None`` ("just record usage") — preserve it
        # explicitly (don't strip with ``exclude_none``).
        body: dict[str, Any] = {"helpful": args.get("helpful")}
        return await self._post(f"/api/v1/memories/{mem_id}/feedback", json=body)

    async def synthesize(self, args: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": args.get("query", ""),
            "k": int(args.get("k", 10)),
            "threshold": float(args.get("threshold", 0.1)),
        }
        return await self._post("/api/v1/synthesize", json=body)

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
        # Total de intentos = 1 (inicial) + max_retries. Si max_retries=0
        # esto es un solo intento y el loop se comporta como antes.
        total_attempts = 1 + self.max_retries
        last_connect_err: httpx.ConnectError | None = None
        last_timeout_err: httpx.TimeoutException | None = None
        last_5xx_resp: httpx.Response | None = None

        for attempt in range(total_attempts):
            # ¿Es el último intento? Si sí, no dormimos después.
            is_last = attempt == total_attempts - 1
            try:
                resp = await self._client.request(method, path, params=params, json=json)
            except httpx.ConnectError as exc:
                last_connect_err = exc
                if is_last:
                    break
                # Sleep antes del próximo intento — backoff exponencial fijo.
                await asyncio.sleep(_RETRY_BACKOFF_S[min(attempt, len(_RETRY_BACKOFF_S) - 1)])
                logger.info(
                    "remote %s %s: ConnectError, retrying (%d/%d)",
                    method,
                    path,
                    attempt + 1,
                    self.max_retries,
                )
                continue
            except httpx.ReadTimeout as exc:
                # ReadTimeout sí re-tryeamos (servidor tarda un toque pero
                # eventualmente responde). ConnectTimeout/WriteTimeout caen
                # acá también porque heredan de TimeoutException; ConnectError
                # ya está manejado arriba — el resto se trata como timeout.
                last_timeout_err = exc
                if is_last:
                    break
                await asyncio.sleep(_RETRY_BACKOFF_S[min(attempt, len(_RETRY_BACKOFF_S) - 1)])
                logger.info(
                    "remote %s %s: ReadTimeout, retrying (%d/%d)",
                    method,
                    path,
                    attempt + 1,
                    self.max_retries,
                )
                continue
            except httpx.TimeoutException as exc:
                # Otros timeouts (connect/write/pool) — NO retry: si no podemos
                # ni conectar/escribir el request, retry corto raramente ayuda
                # y enmascararía un problema de red más serio. Devolvemos
                # como antes (preservando el comportamiento previo del cliente).
                return {"ok": False, "error": f"mem-vault remote timeout: {exc}"}
            except Exception as exc:  # pragma: no cover — defensive
                # Cualquier otra excepción NO es transitoria → no retry.
                logger.exception("remote request failed: %s %s", method, path)
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

            # Llegamos acá ⇒ tenemos `resp`. Decidir si retry según status.
            if resp.status_code in (401, 403):
                # Auth failure — NO retry (el token no se va a "arreglar" solo).
                return {
                    "ok": False,
                    "error": (
                        f"mem-vault remote rejected request ({resp.status_code}). "
                        "The server requires a bearer token — set MEM_VAULT_HTTP_TOKEN "
                        "or pass `token=...` to RemoteMemVaultService."
                    ),
                    "code": "unauthorized",
                }
            if resp.status_code in _RETRYABLE_STATUS:
                last_5xx_resp = resp
                if is_last:
                    break
                await asyncio.sleep(_RETRY_BACKOFF_S[min(attempt, len(_RETRY_BACKOFF_S) - 1)])
                logger.info(
                    "remote %s %s: %d, retrying (%d/%d)",
                    method,
                    path,
                    resp.status_code,
                    attempt + 1,
                    self.max_retries,
                )
                continue
            if resp.status_code >= 500:
                # 5xx no-retryable (ej. 500, 501, 505) — error definitivo.
                return {
                    "ok": False,
                    "error": f"mem-vault remote returned {resp.status_code}: {resp.text[:200]}",
                }
            # 2xx/3xx/4xx no-auth: parsear JSON y devolver. 4xx llega acá y
            # se devuelve como respuesta JSON normal — el server YA codifica
            # errores como `{ok: false, ...}`, así que el caller ya sabe.
            try:
                data = resp.json()
            except ValueError:
                return {"ok": False, "error": f"non-JSON response from {path}"}
            return data

        # Salimos del loop sin éxito → devolver el último error que vimos.
        # Prioridad: 5xx (servidor respondió pero con error) > timeout > connect.
        if last_5xx_resp is not None:
            return {
                "ok": False,
                "error": (
                    f"mem-vault remote returned {last_5xx_resp.status_code} "
                    f"after {total_attempts} attempts: {last_5xx_resp.text[:200]}"
                ),
            }
        if last_timeout_err is not None:
            return {"ok": False, "error": f"mem-vault remote timeout: {last_timeout_err}"}
        if last_connect_err is not None:
            return {
                "ok": False,
                "error": (
                    f"mem-vault remote server unreachable at {self.base_url}: {last_connect_err}. "
                    "Is the obsidian-rag web server running? "
                    "Check `lsof -i :8765` or unset MEM_VAULT_REMOTE_URL to fall back to local mode."
                ),
            }
        # No deberíamos llegar acá nunca — defensive fallback.
        return {"ok": False, "error": "mem-vault remote: unknown error after retries"}

    async def close(self) -> None:
        await self._client.aclose()

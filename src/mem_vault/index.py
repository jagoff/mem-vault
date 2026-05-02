"""Vector index + optional LLM extractor, both backed by mem0ai.

Two responsibilities:

1. **Indexing**: every memory written to the vault is also embedded and stored
   in a local Qdrant collection so we can do semantic search (`memory_search`).
   Embeddings come from Ollama (default ``bge-m3``).

2. **Auto-extraction (opt-in)**: when ``memory_save(auto_extract=True)`` is
   called, we delegate to ``mem0.add(infer=True)`` so the LLM (default
   ``qwen2.5:7b`` via Ollama) extracts canonical facts, deduplicates against
   existing memories, and decides whether to ADD/UPDATE/NOOP. Pure save mode
   (``auto_extract=False``) bypasses the LLM entirely (``infer=False``).

Both paths run on localhost: zero API keys, zero outbound calls.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import UTC, datetime
from typing import Any

from mem0 import Memory as Mem0Memory

from mem_vault.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Content hashing (used for incremental reindex skipping)
# ---------------------------------------------------------------------------


def compute_content_hash(content: str) -> str:
    """Stable digest used to detect whether a memory body has changed.

    We use the first 16 hex chars of SHA-256 — short enough to compare
    quickly in metadata payloads, long enough to make collisions a non-
    issue for a single user's memory corpus (~64 bits of entropy is
    far above the ~10⁵ memory ceiling we'd plausibly see).

    Whitespace at the edges is stripped before hashing so a trailing
    newline added by an editor doesn't cause an unnecessary re-embed.
    """
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()[:16]


class CircuitBreakerOpenError(RuntimeError):
    """Raised when too many consecutive Ollama calls failed; we're failing fast.

    Distinct from a transport timeout — this means we *chose* to short-circuit
    so the MCP server doesn't keep stacking 60-second hangs on a dead Ollama.
    """


class _CircuitBreaker:
    """Tiny in-process circuit breaker around the Ollama-backed mem0 calls.

    State machine:
    - **closed**: every call goes through, failures increment a counter.
    - **open**: after ``threshold`` consecutive failures, every call short-
      circuits with :class:`CircuitBreakerOpenError` for ``cooldown_s``.
    - **half-open**: once the cooldown lapses, the next call is allowed
      through. Success → closed. Failure → open again, cooldown restarts.

    Thread-safety: the counter and the timestamp are read/written from
    ``asyncio.to_thread`` workers, but the worst-case race is two extra
    calls slipping through right at the edge — harmless. We don't lock.
    """

    def __init__(
        self,
        *,
        threshold: int = 3,
        cooldown_s: float = 30.0,
        failure_decay_s: float = 300.0,
    ):
        self.threshold = max(1, threshold)
        self.cooldown_s = max(0.0, cooldown_s)
        # If no new failure happens within ``failure_decay_s`` of the last
        # one, treat the breaker as healed: reset the counter on the next
        # ``record_failure`` call. Without this, three transient failures
        # spread across hours would still pile up and trip the breaker on
        # an otherwise-healthy Ollama. 5 minutes default is long enough to
        # group genuine flakiness and short enough to forget noise.
        self.failure_decay_s = max(0.0, failure_decay_s)
        self._consecutive_failures = 0
        self._open_until = 0.0
        self._last_failure_ts = 0.0

    def is_open(self) -> bool:
        return self._open_until > time.monotonic()

    def cooldown_remaining(self) -> float:
        return max(0.0, self._open_until - time.monotonic())

    def check(self) -> None:
        """Raise :class:`CircuitBreakerOpenError` if the breaker is open."""
        if self.is_open():
            raise CircuitBreakerOpenError(
                f"LLM circuit breaker is open after {self._consecutive_failures} "
                f"consecutive failures. Retry in ~{self.cooldown_remaining():.0f}s "
                "(or set MEM_VAULT_LLM_TIMEOUT_S=0 to disable timeouts)."
            )

    def record_success(self) -> None:
        if self._consecutive_failures or self._open_until:
            logger.info("LLM circuit breaker: closed (success after failures)")
        self._consecutive_failures = 0
        self._open_until = 0.0
        self._last_failure_ts = 0.0

    def record_failure(self) -> None:
        now = time.monotonic()
        # Time-decay reset: stale failures don't compound into a trip.
        if (
            self.failure_decay_s > 0
            and self._last_failure_ts > 0
            and (now - self._last_failure_ts) > self.failure_decay_s
        ):
            self._consecutive_failures = 0
        self._last_failure_ts = now
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.threshold:
            self._open_until = now + self.cooldown_s
            logger.warning(
                "LLM circuit breaker: OPEN — %d consecutive failures, cooling down %.0fs",
                self._consecutive_failures,
                self.cooldown_s,
            )


def _parse_iso(ts: str | None) -> datetime | None:
    """ISO8601 → aware datetime; None on missing/unparseable."""
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except Exception:
        return None


def time_decay_factor(
    updated_iso: str | None,
    half_life_days: float,
    last_used_iso: str | None = None,
) -> float:
    """Multiplier in (0, 1] applied to a search score given the memory's age.

    v0.6.0 update: when ``last_used_iso`` is supplied (the
    ``last_used`` frontmatter that the Stop hook bumps every time the
    agent cites the memory), we use the **more recent** of
    ``updated_iso`` and ``last_used_iso`` to compute the age. This
    converts the global half-life into a *per-memory effective* decay:
    a 6-month-old memory that the agent cited yesterday looks fresh,
    while one that's never been cited decays at the full rate.

    Returns 1.0 when:

    - ``half_life_days`` is 0 or negative (decay disabled)
    - Both ``updated_iso`` and ``last_used_iso`` are missing/unparseable
      (we don't punish memories that simply lack a timestamp — better
      to fall back to pure semantic score).

    Otherwise: ``2 ** (-age_days / half_life_days)``.
    """
    if half_life_days is None or half_life_days <= 0:
        return 1.0
    candidates = [_parse_iso(updated_iso), _parse_iso(last_used_iso)]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return 1.0
    # Most recent timestamp wins — citing an old memory keeps it warm.
    dt = max(candidates)
    age_seconds = (datetime.now(tz=dt.tzinfo) - dt).total_seconds()
    if age_seconds <= 0:
        return 1.0
    age_days = age_seconds / 86400.0
    return 2.0 ** (-age_days / half_life_days)


class VectorIndex:
    """Thin wrapper over a `mem0.Memory` configured for fully-local Ollama + Qdrant."""

    def __init__(self, config: Config):
        self.config = config
        self._mem0: Mem0Memory | None = None
        # Default thresholds keep behavior gentle: 3 failures in a row open
        # the breaker for 30 s. The wall-clock timeout itself comes from
        # ``server.py`` (``asyncio.wait_for``); the breaker is the second
        # layer that prevents stacking hangs once Ollama is unreachable.
        self._breaker = _CircuitBreaker(threshold=3, cooldown_s=30.0)
        # Silent failure observability: the design choice in ``search`` is to
        # swallow runtime errors (RemoteProtocolError, ConnectionError, model
        # eviction mid-flight, …) and return ``[]`` so a missed lookup
        # doesn't abort the agent's turn. That's friendly for the happy path
        # but invisible to the caller — under memory pressure the agent
        # silently loses access to past context. We expose the *last* error
        # captured by ``search`` (cleared on success) so higher-level code
        # can render a ``warning: dense_error: ...`` and decide whether to
        # surface a fallback result. See ``server.search`` for the consumer.
        self._last_search_error: Exception | None = None

    @property
    def breaker(self) -> _CircuitBreaker:
        """Expose the breaker so tests / callers can inspect or reset it."""
        return self._breaker

    @property
    def last_search_error(self) -> Exception | None:
        """The last exception captured by :meth:`search` and turned into an
        empty-list return. ``None`` after a successful search.

        Read once per search call from ``server.py`` to detect silent
        degraded mode. Writes happen only inside :meth:`search` (and the
        circuit breaker short-circuit), so there's no cross-thread
        synchronization concern beyond the GIL — same instance is owned
        by a single :class:`MemVaultService`.
        """
        return self._last_search_error

    @property
    def mem0(self) -> Mem0Memory:
        if self._mem0 is None:
            self._mem0 = Mem0Memory.from_config(self._build_mem0_config())
        return self._mem0

    def _build_mem0_config(self) -> dict[str, Any]:
        cfg = self.config
        # Qdrant connection — server mode (url) takes precedence over
        # embedded mode (path). Server mode is required when multiple
        # agents / sessions / tools share the same vault: the embedded
        # qdrant-client takes an exclusive file lock and silently fails
        # the second concurrent caller, which surfaces as `memory_search`
        # returning empty lists with no visible error to the agent.
        # See `mem-vault Qdrant cleanup` memory (2026-04-30) for the
        # full diagnosis.
        vector_store_config: dict[str, Any] = {
            "collection_name": cfg.qdrant_collection,
            "embedding_model_dims": cfg.embedder_dims,
        }
        if cfg.qdrant_url:
            vector_store_config["url"] = cfg.qdrant_url
        else:
            # Embedded mode: passing `path` makes qdrant-client use a
            # local file-backed collection — no daemon, single writer.
            # Each Config instance writes to its own state_dir/qdrant.
            vector_store_config["path"] = str(cfg.qdrant_path)
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": vector_store_config,
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": cfg.llm_model,
                    "ollama_base_url": cfg.ollama_host,
                    "temperature": 0.1,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": cfg.embedder_model,
                    "ollama_base_url": cfg.ollama_host,
                    "embedding_dims": cfg.embedder_dims,
                },
            },
            "history_db_path": str(cfg.history_db),
        }

    def add(
        self,
        content: str,
        *,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        auto_extract: bool = False,
    ) -> list[dict[str, Any]]:
        """Embed + store one memory. Returns mem0's structured result.

        With ``auto_extract=False`` (default), this calls ``infer=False`` so
        mem0 saves the literal text without LLM rewriting/dedup. With
        ``auto_extract=True``, the LLM runs and may emit multiple facts or
        skip duplicates entirely.

        Raises:
            CircuitBreakerOpenError: when the breaker is open after recent
                consecutive Ollama failures. The caller should retry later.
            Exception: any error from mem0 / Ollama propagates after the
                failure is recorded against the breaker.
        """
        self._breaker.check()
        try:
            result = self.mem0.add(
                messages=content,
                user_id=user_id,
                agent_id=agent_id,
                metadata=metadata or {},
                infer=auto_extract,
            )
        except Exception:
            self._breaker.record_failure()
            raise
        self._breaker.record_success()
        # mem0 returns a dict like {"results": [...]} since v0.1
        if isinstance(result, dict):
            return list(result.get("results", []))
        if isinstance(result, list):
            return result
        return []

    def search(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Semantic search against the local Qdrant collection."""
        merged_filters: dict[str, Any] = {"user_id": user_id}
        if filters:
            merged_filters.update(filters)
        # Pull a wider set than the caller asked for when decay is enabled —
        # the rerank below may shuffle older items down past the cut.
        decay = self.config.decay_half_life_days
        oversample = max(top_k * 3, 20) if decay > 0 else top_k
        # Search short-circuits to an empty list on failure (the breaker
        # still ticks though, so repeated failures will trip it). Returning
        # ``[]`` is friendlier than raising for the agent: a failed search
        # is a missed lookup, not a hard error worth aborting the turn for.
        # Reset the silent-failure marker before each call. If the search
        # succeeds, ``last_search_error`` stays ``None`` and callers know
        # an empty list means "no semantic matches", not "the embedder
        # crashed". If it fails, we stash the exception so ``server.py``
        # can render a degraded-mode warning without losing the friendly
        # empty-list contract.
        self._last_search_error = None
        try:
            self._breaker.check()
            res = self.mem0.search(
                query=query,
                top_k=oversample,
                filters=merged_filters,
                threshold=threshold,
            )
        except CircuitBreakerOpenError as exc:
            logger.warning("mem0 search short-circuited: %s", exc)
            self._last_search_error = exc
            return []
        except Exception as exc:
            self._breaker.record_failure()
            logger.warning("mem0 search failed (%s); returning empty list", exc)
            self._last_search_error = exc
            return []
        self._breaker.record_success()

        if isinstance(res, dict):
            hits = list(res.get("results", []))
        elif isinstance(res, list):
            hits = list(res)
        else:
            hits = []

        if decay > 0 and hits:
            for h in hits:
                if not isinstance(h, dict):
                    continue
                base = h.get("score") or 0.0
                # Prefer ``updated_at`` (mem0 native), fall back to
                # metadata.updated, then memory.updated.
                ts = (
                    h.get("updated_at")
                    or h.get("created_at")
                    or (h.get("metadata") or {}).get("updated")
                    or (h.get("metadata") or {}).get("created")
                )
                # v0.6.0: per-memory effective decay. ``last_used`` is
                # the Stop-hook-maintained "last citation" timestamp on
                # the .md frontmatter — when newer than ``updated``, it
                # keeps the memory warm. mem0 doesn't surface our
                # custom field directly, so the canonical source is
                # ``metadata.last_used`` (stamped via vault payload).
                last_used = (
                    (h.get("metadata") or {}).get("last_used")
                    or h.get("last_used")
                )
                factor = time_decay_factor(ts, decay, last_used_iso=last_used)
                h["score_raw"] = base
                h["decay_factor"] = factor
                h["score"] = base * factor
            hits.sort(
                key=lambda h: (h.get("score") or 0.0) if isinstance(h, dict) else 0.0, reverse=True
            )
            hits = hits[:top_k]

        return hits

    def get_by_metadata(self, key: str, value: str, user_id: str) -> list[dict[str, Any]]:
        """Fetch every mem0 entry whose metadata[key] == value.

        Returned entries are mem0's raw dicts (with ``id``, ``metadata``,
        possibly ``memory`` body, etc.). Errors are swallowed — the caller
        gets ``[]`` and a warning in the log instead of an exception, so
        higher-level code can fall back to a no-cache codepath gracefully.
        """
        try:
            entries = self.mem0.get_all(filters={"user_id": user_id, key: value})
        except Exception as exc:
            logger.warning("mem0 get_all failed (%s); returning empty list", exc)
            return []
        if isinstance(entries, dict):
            return list(entries.get("results", []))
        if isinstance(entries, list):
            return list(entries)
        return []

    def delete_by_metadata(self, key: str, value: str, user_id: str) -> int:
        """Delete every mem0 entry whose metadata[key] == value. Returns count."""
        items = self.get_by_metadata(key, value, user_id)
        deleted = 0
        for item in items:
            mid = item.get("id") if isinstance(item, dict) else None
            if not mid:
                continue
            try:
                self.mem0.delete(memory_id=mid)
                deleted += 1
            except Exception as exc:
                logger.warning("mem0 delete %s failed: %s", mid, exc)
        return deleted

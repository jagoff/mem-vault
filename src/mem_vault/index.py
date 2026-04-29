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

import logging
from datetime import UTC, datetime
from typing import Any

from mem0 import Memory as Mem0Memory

from mem_vault.config import Config

logger = logging.getLogger(__name__)


def time_decay_factor(updated_iso: str | None, half_life_days: float) -> float:
    """Multiplier in (0, 1] applied to a search score given an ``updated`` ISO timestamp.

    Returns 1.0 when:
    - ``half_life_days`` is 0 or negative (decay disabled)
    - ``updated_iso`` is missing/unparseable (we don't punish memories that
      simply lack a timestamp — better to fall back to pure semantic score)

    Otherwise: ``2 ** (-age_days / half_life_days)`` — true half-life decay.
    With a 90-day half-life, a memory updated 90 days ago has its score
    halved; 180 days ago, quartered; one updated yesterday is barely
    affected.
    """
    if half_life_days is None or half_life_days <= 0:
        return 1.0
    if not updated_iso:
        return 1.0
    try:
        dt = datetime.fromisoformat(updated_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
    except Exception:
        return 1.0
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

    @property
    def mem0(self) -> Mem0Memory:
        if self._mem0 is None:
            self._mem0 = Mem0Memory.from_config(self._build_mem0_config())
        return self._mem0

    def _build_mem0_config(self) -> dict[str, Any]:
        cfg = self.config
        # Qdrant embedded mode: passing `path` makes qdrant-client use a local
        # file-backed collection — no Docker, no daemon. Each Config instance
        # writes to its own state_dir/qdrant directory.
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": cfg.qdrant_collection,
                    "path": str(cfg.qdrant_path),
                    "embedding_model_dims": cfg.embedder_dims,
                },
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
        """
        result = self.mem0.add(
            messages=content,
            user_id=user_id,
            agent_id=agent_id,
            metadata=metadata or {},
            infer=auto_extract,
        )
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
        try:
            res = self.mem0.search(
                query=query,
                top_k=oversample,
                filters=merged_filters,
                threshold=threshold,
            )
        except Exception as exc:
            logger.warning("mem0 search failed (%s); returning empty list", exc)
            return []

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
                factor = time_decay_factor(ts, decay)
                h["score_raw"] = base
                h["decay_factor"] = factor
                h["score"] = base * factor
            hits.sort(
                key=lambda h: (h.get("score") or 0.0) if isinstance(h, dict) else 0.0, reverse=True
            )
            hits = hits[:top_k]

        return hits

    def delete_by_metadata(self, key: str, value: str, user_id: str) -> int:
        """Delete every mem0 entry whose metadata[key] == value. Returns count."""
        try:
            entries = self.mem0.get_all(filters={"user_id": user_id, key: value})
        except Exception as exc:
            logger.warning("mem0 get_all failed (%s); skipping delete", exc)
            return 0
        items: list[dict[str, Any]]
        if isinstance(entries, dict):
            items = list(entries.get("results", []))
        elif isinstance(entries, list):
            items = entries
        else:
            items = []
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

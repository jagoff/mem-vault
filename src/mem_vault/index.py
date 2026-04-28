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
from typing import Any

from mem0 import Memory as Mem0Memory

from mem_vault.config import Config

logger = logging.getLogger(__name__)


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
        try:
            res = self.mem0.search(
                query=query,
                top_k=top_k,
                filters=merged_filters,
                threshold=threshold,
            )
        except Exception as exc:
            logger.warning("mem0 search failed (%s); returning empty list", exc)
            return []
        if isinstance(res, dict):
            return list(res.get("results", []))
        if isinstance(res, list):
            return res
        return []

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

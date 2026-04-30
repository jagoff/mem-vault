"""Local reranking layer for ``memory_search``.

The dense bi-encoder (``bge-m3`` via Ollama) is fast but coarse — it
encodes query and candidates into the same vector space and returns the
top-k by cosine similarity. A cross-encoder reranker re-scores each
``(query, candidate)`` pair through a single model that sees both
together, which captures interactions the bi-encoder misses (negation,
specificity, entity matching).

We use [`fastembed`](https://github.com/qdrant/fastembed)'s
``TextCrossEncoder`` because:

1. It's already in the optional ``[hybrid]`` extra of ``mem-vault`` — no
   new top-level dependency.
2. It runs locally on CPU (no API calls), keeping the project's
   "100% local stack" promise intact.
3. The default model, ``jinaai/jina-reranker-v1-tiny-en`` (~130 MB),
   reranks 20 candidates in ~30-60 ms on a modern laptop.

The ``LocalReranker`` is **opt-in** via ``Config.reranker_enabled``
(default off) so the base ``mem-vault`` install stays lightweight.
When the extra isn't installed, ``LocalReranker.available`` returns
``False`` and the reranker becomes a no-op pass-through — calling code
doesn't have to branch.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LocalReranker:
    """Cross-encoder reranker wrapping ``fastembed.TextCrossEncoder``.

    The underlying model is downloaded once on first use (cached at
    ``~/.cache/fastembed/``). Subsequent invocations are pure CPU.

    Failure modes are quiet by design:
    - If ``fastembed`` isn't installed, ``available`` flips to ``False``
      after the first lookup and stays there.
    - If the model download fails (no internet, disk full), same behavior.
    - If a single rerank call raises mid-flight (rare), the reranker
      logs a warning and returns the candidates in their original order.

    All three paths preserve search functionality — the user just doesn't
    benefit from the rerank quality boost.
    """

    def __init__(self, model: str = "jinaai/jina-reranker-v1-tiny-en") -> None:
        self.model = model
        self._encoder: Any | None = None
        self._import_error: str | None = None

    @property
    def available(self) -> bool:
        """Lazily import + instantiate the cross-encoder. Cached after first hit."""
        if self._encoder is not None:
            return True
        if self._import_error is not None:
            return False
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._encoder = TextCrossEncoder(self.model)
            return True
        except Exception as exc:
            self._import_error = f"{type(exc).__name__}: {exc}"
            logger.info(
                "LocalReranker unavailable (%s) — falling back to bi-encoder order. "
                "Install with: uv pip install 'mem-vault[hybrid]'",
                self._import_error,
            )
            return False

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Re-order ``candidates`` by cross-encoder score, keep top-``k``.

        ``candidates`` should be the raw mem0 hit dicts — we pull the
        memory body out of ``memory.body`` / ``memory`` (str) / ``snippet``
        / ``text`` in that order. Hits with no extractable body are kept
        in their original position (we can't score them).

        Each returned hit gets a ``rerank_score`` field tacked on so
        downstream callers can decide whether to surface it. The original
        ``score`` (bi-encoder cosine) is preserved.
        """
        if top_k <= 0 or not candidates:
            return candidates[:top_k]
        if not self.available:
            return candidates[:top_k]

        bodies: list[str] = []
        scorable: list[dict[str, Any]] = []
        unscorable: list[dict[str, Any]] = []
        for c in candidates:
            body = _extract_body(c)
            if body:
                bodies.append(body)
                scorable.append(c)
            else:
                unscorable.append(c)

        if not bodies:
            return candidates[:top_k]

        try:
            assert self._encoder is not None  # narrowed by ``available``
            scores = list(self._encoder.rerank(query, bodies, batch_size=64))
        except Exception as exc:
            logger.warning("rerank scoring failed (%s); using bi-encoder order", exc)
            return candidates[:top_k]

        # Pair, sort by rerank score desc, stamp the score onto the hit.
        ranked = sorted(zip(scorable, scores), key=lambda pair: float(pair[1]), reverse=True)
        out: list[dict[str, Any]] = []
        for hit, s in ranked:
            with_score = dict(hit)
            with_score["rerank_score"] = float(s)
            out.append(with_score)

        # Unscorable candidates go to the tail — better than dropping them
        # when the rerank set is smaller than ``top_k``.
        out.extend(unscorable)
        return out[:top_k]


def _extract_body(hit: dict[str, Any]) -> str:
    """Pull a body string out of a mem0 search hit, tolerant of shape variants.

    mem0's hit dict has historically used different keys across versions:
    sometimes ``memory`` is a string (the body itself), sometimes a dict
    with ``body``/``content``, sometimes neither and ``snippet``/``text``
    is the only carrier. We try them in order.
    """
    mem = hit.get("memory")
    if isinstance(mem, str) and mem.strip():
        return mem
    if isinstance(mem, dict):
        for key in ("body", "content", "text"):
            v = mem.get(key)
            if isinstance(v, str) and v.strip():
                return v
    for key in ("snippet", "text", "body"):
        v = hit.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""

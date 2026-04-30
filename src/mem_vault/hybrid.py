"""Hybrid retrieval: BM25 sparse + dense vector + Reciprocal Rank Fusion.

The default ``memory_search`` pipeline is pure dense (Ollama's ``bge-m3``
via Qdrant). It's excellent at semantic similarity and cross-lingual
queries, and weak at exact keyword matches (error strings, command
names, file paths, identifiers). BM25, a classical sparse retriever, is
the opposite: blind to semantics, surgical with keywords.

We combine them with [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf),
a parameter-free algorithm that takes multiple ranked lists and produces
one fused ranking: ``score(d) = Σ 1 / (k + rank_i(d))`` where ``k`` is a
smoothing constant (60 is the community-standard default) and ``rank_i``
is the doc's position in the *i*-th list. RRF has two nice properties
for our case:

1. It only needs rank positions (not comparable scores), so we can fuse
   a dense cosine with a BM25 Okapi score without normalizing.
2. It's deterministic and parameter-minimal — no tuning knobs beyond
   ``k``, which is robust across corpora.

The BM25 implementation is inline (no ``rank_bm25`` dep) — the standard
Okapi formula with stop-word-free tokenization. For single-user vaults
(≤1-2k memorias) a rebuild is <10 ms; the retriever caches the index
and invalidates only on write events (``memory_save`` / ``memory_update``
/ ``memory_delete``).

Enable via ``Config.hybrid_enabled`` (default ``False``) or
``MEM_VAULT_HYBRID=1``. When disabled, this module is never imported.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections import Counter
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenization — cheap, language-agnostic
# ---------------------------------------------------------------------------

# We keep anything that looks like a word (letters + digits + underscore).
# No stemming / no stop-word list: the corpus is small, over-aggressive
# preprocessing costs more (forgets domain terms) than it saves (dedupes
# a handful of tokens). ``re.UNICODE`` means "word" includes é/á/ñ/… so
# Spanish + English + CJK mentions all tokenize consistently.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase + extract word tokens. Empty input → ``[]``."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# BM25 — classic Okapi BM25, rebuilt in-memory on demand
# ---------------------------------------------------------------------------


class BM25Index:
    """In-memory BM25 index over a list of ``(id, text)`` pairs.

    Standard Okapi formula:

        score(D, Q) = Σ idf(q) * tf(q, D) * (k1 + 1) /
                       (tf(q, D) + k1 * (1 - b + b * |D| / avgdl))

    with ``k1 = 1.5`` and ``b = 0.75`` (the classical defaults — they
    generalize well and our corpus is too small to tune empirically).

    Build is O(Σ|D|); search is O(|Q| × N) where N is the number of
    docs. For N ≤ 2000 the linear scan takes under 10 ms on a laptop.
    Above that, Qdrant's native sparse support or a proper inverted
    index is worth the dep — this is the "good enough" baseline for
    single-user vaults.
    """

    def __init__(
        self,
        pairs: Iterable[tuple[str, str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Build the index from ``(id, text)`` pairs.

        ``pairs`` is consumed once; the caller is responsible for
        rebuilding when memories change (see :class:`HybridRetriever`).
        """
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self.docs: list[list[str]] = []
        doc_freq: Counter[str] = Counter()
        for mem_id, text in pairs:
            tokens = tokenize(text)
            self.ids.append(mem_id)
            self.docs.append(tokens)
            for term in set(tokens):
                doc_freq[term] += 1
        self.n = len(self.docs)
        self.doc_lens = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_lens) / self.n) if self.n else 0.0
        # BM25 IDF (``log((N - df + 0.5) / (df + 0.5) + 1)``), which stays
        # non-negative and plays well with the score sum. The ``+1`` before
        # the outer log avoids the negative-idf edge case when ``df`` is
        # a large fraction of ``N``.
        self.idf: dict[str, float] = {
            term: math.log(((self.n - freq + 0.5) / (freq + 0.5)) + 1.0)
            for term, freq in doc_freq.items()
        }
        self.term_freqs: list[Counter[str]] = [Counter(d) for d in self.docs]

    def search(self, query: str, *, top_k: int = 20) -> list[tuple[str, float]]:
        """Return the top-``k`` ``(id, score)`` pairs for ``query``.

        Documents with score 0 (no query term appears) are dropped —
        returning them in arbitrary order adds noise without helping
        the fusion step.
        """
        q_tokens = tokenize(query)
        if not q_tokens or self.n == 0:
            return []
        scored: list[tuple[str, float]] = []
        for i in range(self.n):
            score = self._score_doc(i, q_tokens)
            if score > 0:
                scored.append((self.ids[i], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(0, top_k)]

    def _score_doc(self, doc_idx: int, q_tokens: list[str]) -> float:
        tf = self.term_freqs[doc_idx]
        dl = self.doc_lens[doc_idx]
        if dl == 0 or self.avgdl == 0:
            return 0.0
        total = 0.0
        for term in q_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            term_tf = tf.get(term, 0)
            if term_tf == 0:
                continue
            denom = term_tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += idf * term_tf * (self.k1 + 1) / denom
        return total


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion — parameter-free list merger
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    *,
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists into one via RRF.

    ``rankings`` is a list of lists of document ids, each list sorted
    descending by that ranker's relevance. ``k`` is the smoothing
    constant (the 2009 paper uses 60; higher = flatter contribution).
    ``top_n`` caps the output; ``None`` returns everything.

    Returns ``(id, rrf_score)`` sorted by score desc, ties broken by id
    (lexicographic) for determinism — useful for tests.

    Duplicate ids within a single ranking use the first (best) position
    only; a duplicate in a ranker shouldn't inflate its own score.

    ``k`` must be non-negative. Negative ``k`` values would make the
    contribution ``1/(k + position + 1)`` go negative or explode near
    zero (when ``k + pos + 1 == 0``), silently inverting the fusion or
    producing nonsense scores. We fail loudly instead — typo in config
    or a bad caller is the most likely cause.
    """
    if k < 0:
        raise ValueError(f"RRF k must be non-negative, got {k}")
    scores: dict[str, float] = {}
    for rank_list in rankings:
        seen: set[str] = set()
        for position, doc_id in enumerate(rank_list):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            contribution = 1.0 / (k + position + 1)
            scores[doc_id] = scores.get(doc_id, 0.0) + contribution
    ordered = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    if top_n is not None and top_n >= 0:
        ordered = ordered[:top_n]
    return ordered


# ---------------------------------------------------------------------------
# HybridRetriever — glue that owns the BM25 cache + invalidation
# ---------------------------------------------------------------------------


class HybridRetriever:
    """BM25 index cached in-memory, rebuilt lazily on the next search after
    an ``invalidate()`` call. Thread-safe across the search + write paths.

    Lifecycle:

    1. The service instantiates one ``HybridRetriever`` per
       ``MemVaultService`` (or ``None`` when ``hybrid_enabled=False``).
    2. Every ``memory_save`` / ``memory_update`` / ``memory_delete`` calls
       ``invalidate()`` — cheap (just flips a dirty flag).
    3. The next ``search`` call triggers ``_rebuild()``, re-reading every
       ``.md`` from the vault and constructing the BM25 index fresh.
    4. Rebuilds are serialized via an ``RLock``; reads can happen
       concurrently in principle but are cheap enough we don't worry.

    The rebuild time grows linearly with the corpus; the overhead is
    one-time per write event. For vaults below ~2000 memorias this is
    negligible (single-digit ms).
    """

    def __init__(
        self,
        vault_storage: Any,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.storage = vault_storage
        self.k1 = k1
        self.b = b
        self._index: BM25Index | None = None
        self._dirty = True
        self._lock = threading.RLock()

    def invalidate(self) -> None:
        """Mark the cached index stale. O(1) — the rebuild happens at the
        next ``search`` call."""
        with self._lock:
            self._dirty = True

    def _rebuild(self) -> None:
        """Read every .md in the vault and build a fresh BM25 index.

        The BM25 corpus is ``name + body + tags`` concatenated per
        memory — this way a search for the tag ``rag-obsidian`` also
        hits the body text where the tag appears in a wikilink. Tags
        alone would miss the body mentions.
        """
        # Stream through ``iter_memories`` instead of loading the full list
        # at once — for vaults with thousands of memorias this avoids holding
        # the entire corpus in RAM at peak. The BM25 index itself is the
        # only persistent structure (~tokenized strings + counters per doc).
        pairs = []
        for m in self.storage.iter_memories():
            parts = [m.name or "", m.body or ""]
            if m.tags:
                parts.extend(m.tags)
            pairs.append((m.id, "\n".join(parts)))
        self._index = BM25Index(pairs, k1=self.k1, b=self.b)
        self._dirty = False

    def search(self, query: str, *, top_k: int = 20) -> list[tuple[str, float]]:
        with self._lock:
            if self._dirty or self._index is None:
                self._rebuild()
            assert self._index is not None
            return self._index.search(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Fusion adapter — dense mem0 hits + BM25 (id, score) → unified hit list
# ---------------------------------------------------------------------------


def fuse_dense_and_bm25(
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[tuple[str, float]],
    *,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Merge a dense mem0 result list with a BM25 id-score list into one
    unified hit list ordered by RRF score.

    Each output dict carries:
    - ``id`` — memory id (the service already knows to resolve these).
    - ``score`` — the RRF score (pipeline-compatible with the rest of
      ``memory_search``: multiplied by the usage boost downstream).
    - ``rrf_score`` — the same value, exposed for debugging.
    - ``metadata`` — from the dense hit when present, else a minimal
      ``{"memory_id": id}`` stub (the downstream code extracts
      ``memory_id`` and looks up the full ``.md`` anyway).

    Ids appearing only in BM25 keep the stub metadata; ids in both get
    the dense hit's richer metadata + snippet. Ties in RRF score break
    by ``id`` asc for determinism.
    """
    # Keep only the first (best-ranked) occurrence of each id in dense_hits
    # — mem0 occasionally returns dup ids across variants.
    dense_first: dict[str, tuple[int, dict[str, Any]]] = {}
    for rank, hit in enumerate(dense_hits):
        if not isinstance(hit, dict):
            continue
        meta = hit.get("metadata") or {}
        mem_id = meta.get("memory_id")
        if not mem_id or mem_id in dense_first:
            continue
        dense_first[mem_id] = (rank, hit)

    bm25_rank: dict[str, int] = {}
    for rank, (mem_id, _score) in enumerate(bm25_hits):
        if mem_id not in bm25_rank:
            bm25_rank[mem_id] = rank

    # ``reverse=False`` is explicit on purpose: RRF expects the input lists
    # ordered by rank position ASCENDING (best = position 0). The default
    # ``sorted()`` is already ascending, but if a future refactor flips this
    # to ``reverse=True`` the entire fusion silently inverts (worst hits
    # become "best"). The keyword arg here + the regression test
    # ``test_fuse_dense_and_bm25_known_good_ordering`` together guard
    # against that drift.
    dense_ids = [
        mid for mid, _ in sorted(dense_first.items(), key=lambda x: x[1][0], reverse=False)
    ]
    bm25_ids = [mid for mid, _ in sorted(bm25_rank.items(), key=lambda x: x[1], reverse=False)]

    fused = reciprocal_rank_fusion([dense_ids, bm25_ids], k=rrf_k)

    output: list[dict[str, Any]] = []
    for mem_id, rrf_score in fused:
        if mem_id in dense_first:
            base = dict(dense_first[mem_id][1])
        else:
            # BM25-only hit — synthesize a minimal shape. The rest of
            # the pipeline calls storage.get(memory_id), which fills in
            # the body.
            base = {"id": mem_id, "metadata": {"memory_id": mem_id}}
        base["score"] = float(rrf_score)
        base["rrf_score"] = float(rrf_score)
        output.append(base)
    return output

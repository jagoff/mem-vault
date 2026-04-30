"""Tests for hybrid retrieval (BM25 + dense + RRF).

Three layers:

1. **Pure helpers** (``tokenize``, ``BM25Index``, ``reciprocal_rank_fusion``,
   ``fuse_dense_and_bm25``) — tested without touching storage.
2. **HybridRetriever** — cache rebuild + invalidation against a real
   ``VaultStorage`` on a tmp dir.
3. **Integration** — ``MemVaultService.search`` with ``hybrid_enabled=True``
   flips ranking to favor a BM25-unique match the dense stub misses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mem_vault.config import Config
from mem_vault.hybrid import (
    BM25Index,
    HybridRetriever,
    fuse_dense_and_bm25,
    reciprocal_rank_fusion,
    tokenize,
)
from mem_vault.index import _CircuitBreaker
from mem_vault.server import MemVaultService
from mem_vault.storage import VaultStorage

# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("hello world", ["hello", "world"]),
        ("CamelCase API_KEY", ["camelcase", "api_key"]),
        (
            "ninguna — palabra con ñ y tildes áéíóú",
            ["ninguna", "palabra", "con", "ñ", "y", "tildes", "áéíóú"],
        ),
        ("", []),
        ("!@#$%", []),
        ("rate_limit 60/min", ["rate_limit", "60", "min"]),
    ],
)
def test_tokenize(text, expected):
    assert tokenize(text) == expected


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------


def test_bm25_empty_corpus_returns_empty():
    idx = BM25Index([])
    assert idx.search("anything") == []


def test_bm25_single_doc_exact_match():
    idx = BM25Index([("a", "hello world")])
    results = idx.search("hello")
    assert len(results) == 1
    assert results[0][0] == "a"
    assert results[0][1] > 0


def test_bm25_ranks_keyword_match_above_unrelated_doc():
    """The doc with more query-term hits should outrank the unrelated one."""
    idx = BM25Index(
        [
            ("match", "rate_limit configuration for the api endpoint"),
            ("other", "cooking recipes and garden tips"),
        ]
    )
    results = idx.search("rate_limit")
    assert results[0][0] == "match"


def test_bm25_tf_saturation():
    """With k1=1.5 (default), a second occurrence of a term shouldn't
    dramatically outrank a single occurrence."""
    idx = BM25Index(
        [
            ("once", "apple banana"),
            ("twice", "apple apple banana"),
        ]
    )
    results = idx.search("apple")
    # The doubled-occurrence doc wins, but TF saturation keeps the gap modest.
    assert results[0][0] == "twice"
    # If we'd been doing raw TF, twice would be ~2x once. With k1=1.5 saturation
    # it should be closer to 1.3-1.5x — assert a ceiling to catch regressions.
    twice_score = results[0][1]
    once_score = next(s for i, s in results if i == "once")
    assert twice_score > once_score
    assert twice_score < 2.0 * once_score


def test_bm25_skips_zero_score_docs():
    """Docs that don't contain any query token must not appear in the output."""
    idx = BM25Index(
        [
            ("alpha", "hello world"),
            ("beta", "fizz buzz"),
        ]
    )
    results = idx.search("hello")
    assert [i for i, _ in results] == ["alpha"]


def test_bm25_top_k_caps_output():
    pairs = [(f"d{i}", f"apple doc {i}") for i in range(10)]
    idx = BM25Index(pairs)
    results = idx.search("apple", top_k=3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------


def test_rrf_single_ranking_passthrough():
    """With one input list the RRF ordering matches the input."""
    out = reciprocal_rank_fusion([["a", "b", "c"]])
    ids = [i for i, _ in out]
    assert ids == ["a", "b", "c"]


def test_rrf_prefers_doc_present_in_both_rankings():
    """A doc ranked high in both should beat docs ranked high in one only."""
    dense = ["alpha", "beta", "gamma"]
    bm25 = ["delta", "alpha", "epsilon"]
    out = reciprocal_rank_fusion([dense, bm25])
    # alpha at rank 0 (dense) + rank 1 (bm25) should outrank delta/beta/etc.
    assert out[0][0] == "alpha"


def test_rrf_dedupes_within_single_ranking():
    """A doc appearing twice in the same ranking gets its duplicate ignored,
    but subsequent docs keep their original positions (no compaction).

    With input ``["a", "a", "b"]``:
    - "a" at rank 0 → score = 1/(k+1)
    - "a" at rank 1 → skipped (already seen)
    - "b" at rank 2 → score = 1/(k+3)

    The alternative (compact positions after dedup) would let a ranker
    inflate a trailing doc by earlier-in-list duplication — we prefer
    position-stable semantics: the duplicated rank "is burned", not
    erased. The regression assertion is that "a" is not double-counted.
    """
    out = dict(reciprocal_rank_fusion([["a", "a", "b"]]))
    baseline = dict(reciprocal_rank_fusion([["a", "b"]]))
    # "a" is not double-counted — it gets the same contribution as a clean list.
    assert out["a"] == pytest.approx(baseline["a"])
    # "b" stayed at its original rank 2 position (NOT compacted to rank 1).
    assert out["b"] < baseline["b"]


def test_rrf_k_constant_shapes_contribution():
    """Higher k flattens the rank contribution curve."""
    # With k=10, the difference between rank 0 and rank 5 is larger than with k=100.
    low_k = reciprocal_rank_fusion([["a", "b", "c", "d", "e", "f"]], k=10)
    high_k = reciprocal_rank_fusion([["a", "b", "c", "d", "e", "f"]], k=100)
    ratio_low = low_k[0][1] / low_k[5][1]
    ratio_high = high_k[0][1] / high_k[5][1]
    assert ratio_low > ratio_high


def test_rrf_top_n_caps_output():
    out = reciprocal_rank_fusion([["a", "b", "c", "d"]], top_n=2)
    assert [i for i, _ in out] == ["a", "b"]


def test_rrf_deterministic_on_tied_scores():
    """Ties break by id lexicographically — essential for test determinism."""
    out = reciprocal_rank_fusion([["x", "y"], ["y", "x"]])
    # Both x and y appear at positions (0,1) and (1,0) → identical scores.
    ids = [i for i, _ in out]
    assert ids == ["x", "y"]  # lexicographic tiebreak


# ---------------------------------------------------------------------------
# fuse_dense_and_bm25
# ---------------------------------------------------------------------------


def test_fuse_both_sources_combines_rankings():
    dense = [
        {"metadata": {"memory_id": "a"}, "score": 0.9, "memory": "dense body a"},
        {"metadata": {"memory_id": "b"}, "score": 0.8},
    ]
    bm25 = [("c", 5.0), ("a", 3.0)]  # 'a' in both
    out = fuse_dense_and_bm25(dense, bm25)
    ids = [h["id"] if "id" in h else h["metadata"]["memory_id"] for h in out]
    # 'a' wins because it's ranked well in both.
    assert ids[0] == "a"
    # 'c' (BM25-only) still appears; 'b' (dense-only) too.
    assert set(ids) == {"a", "b", "c"}


def test_fuse_preserves_dense_metadata_when_available():
    dense = [{"metadata": {"memory_id": "a", "tags": ["foo"]}, "score": 0.9, "memory": "body"}]
    bm25 = [("b", 1.0)]
    out = fuse_dense_and_bm25(dense, bm25)
    by_id = {(h.get("metadata") or {}).get("memory_id") or h.get("id"): h for h in out}
    # 'a' keeps its rich metadata + snippet from dense
    assert by_id["a"]["metadata"]["tags"] == ["foo"]
    assert by_id["a"]["memory"] == "body"
    # 'b' gets a stub shape
    assert by_id["b"]["metadata"]["memory_id"] == "b"


def test_fuse_empty_inputs():
    assert fuse_dense_and_bm25([], []) == []
    out = fuse_dense_and_bm25([], [("a", 1.0)])
    assert len(out) == 1
    assert out[0]["metadata"]["memory_id"] == "a"


# ---------------------------------------------------------------------------
# HybridRetriever — cache + invalidation against a real storage
# ---------------------------------------------------------------------------


def test_hybrid_retriever_builds_on_first_search(tmp_path):
    storage = VaultStorage(tmp_path)
    storage.save(content="hello rate_limit world", title="alpha")
    storage.save(content="other body with no match", title="beta")

    retriever = HybridRetriever(storage)
    results = retriever.search("rate_limit")
    assert len(results) >= 1
    assert results[0][0] == "alpha"


def test_hybrid_retriever_invalidate_picks_up_new_memory(tmp_path):
    storage = VaultStorage(tmp_path)
    storage.save(content="old body", title="alpha")
    retriever = HybridRetriever(storage)
    assert retriever.search("new_term") == []

    storage.save(content="fresh body with new_term included", title="beta")
    # Without invalidate, the search still sees the old index
    # (deterministic: the cached index doesn't know about beta yet).
    assert retriever.search("new_term") == []

    retriever.invalidate()
    results = retriever.search("new_term")
    assert [i for i, _ in results] == ["beta"]


# ---------------------------------------------------------------------------
# Integration — MemVaultService.search with hybrid_enabled=True
# ---------------------------------------------------------------------------


class _DenseStubIndex:
    def __init__(self, hits):
        self.hits = hits
        self._breaker = _CircuitBreaker(threshold=3, cooldown_s=30.0)

    @property
    def breaker(self):
        return self._breaker

    def add(self, content, **kwargs):
        return [{"id": "stub"}]

    def search(self, query, **kwargs):
        return list(self.hits)

    def delete_by_metadata(self, *args):
        return 0


@pytest.fixture
def hybrid_service(tmp_path: Path):
    def _make(**overrides) -> MemVaultService:
        cfg_kwargs = {
            "vault_path": str(tmp_path),
            "memory_subdir": "memory",
            "state_dir": str(tmp_path / "state"),
            "user_id": "tester",
            "auto_extract_default": False,
            "llm_timeout_s": 0,
            "max_content_size": 0,
            "auto_link_default": False,
            "hybrid_enabled": True,
            **overrides,
        }
        config = Config(**cfg_kwargs)
        config.qdrant_collection = "test"
        config.state_dir.mkdir(parents=True, exist_ok=True)
        config.memory_dir.mkdir(parents=True, exist_ok=True)
        service = MemVaultService(config)
        return service

    return _make


async def test_hybrid_search_surfaces_bm25_only_match(hybrid_service):
    """A memory the dense stub doesn't return but BM25 catches should
    still appear in the fused top-k."""
    service = hybrid_service()
    # Seed two memorias. The agent searches for "rate_limit" which appears
    # in alpha's body. The dense stub deliberately returns ONLY beta (the
    # irrelevant one) — BM25 must rescue alpha via RRF.
    alpha = await service.save({"content": "endpoint: rate_limit=60/min", "title": "alpha"})
    beta = await service.save({"content": "cooking recipes and garden tips", "title": "beta"})
    service.index = _DenseStubIndex(
        [
            {
                "score": 0.4,
                "metadata": {"memory_id": beta["memory"]["id"]},
                "memory": "cooking recipes and garden tips",
            },
        ]
    )

    res = await service.search({"query": "rate_limit", "k": 5})
    ids = [r["id"] for r in res["results"]]
    assert alpha["memory"]["id"] in ids  # BM25 rescued alpha
    # And alpha outranks beta (RRF favors the keyword match).
    alpha_pos = ids.index(alpha["memory"]["id"])
    beta_pos = ids.index(beta["memory"]["id"]) if beta["memory"]["id"] in ids else len(ids)
    assert alpha_pos < beta_pos


async def test_hybrid_disabled_service_has_no_retriever(hybrid_service):
    """Sanity: with hybrid_enabled=False the service skips instantiation."""
    service = hybrid_service(hybrid_enabled=False)
    assert service._hybrid is None


async def test_hybrid_invalidate_is_called_on_save(hybrid_service, monkeypatch):
    service = hybrid_service()
    calls = []
    monkeypatch.setattr(service._hybrid, "invalidate", lambda: calls.append(1))
    await service.save({"content": "body one", "title": "alpha"})
    assert calls, "save should invalidate the BM25 cache"


async def test_hybrid_invalidate_is_called_on_delete(hybrid_service, monkeypatch):
    service = hybrid_service()
    seed = await service.save({"content": "body", "title": "alpha"})
    # Reset the fake index to a no-op that deletes smoothly
    service.index = _DenseStubIndex([])  # type: ignore[assignment]
    calls = []
    monkeypatch.setattr(service._hybrid, "invalidate", lambda: calls.append(1))
    await service.delete({"id": seed["memory"]["id"]})
    assert calls, "delete should invalidate the BM25 cache"

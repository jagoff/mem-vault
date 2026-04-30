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


# ---------------------------------------------------------------------------
# Determinism guards — reciprocal_rank_fusion + fuse_dense_and_bm25
# ---------------------------------------------------------------------------


def test_rrf_rejects_negative_k():
    """Negative ``k`` values would make the RRF contribution
    ``1/(k + position + 1)`` go negative or blow up near zero. The
    function must fail loudly instead of silently producing nonsense
    scores. Common cause: a typo in config (``rrf_k=-1``) or a bad
    caller passing through user input unchecked.
    """
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion([["a", "b"]], k=-1)
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion([["a", "b"]], k=-60)


def test_rrf_accepts_k_zero():
    """``k=0`` is a degenerate-but-valid case: contribution becomes
    ``1/(position+1)``. The validator must allow it (only negative is
    nonsense)."""
    out = reciprocal_rank_fusion([["a", "b"]], k=0)
    # First doc: 1/1 = 1.0; second doc: 1/2 = 0.5.
    scores = dict(out)
    assert scores["a"] == pytest.approx(1.0)
    assert scores["b"] == pytest.approx(0.5)


def test_fuse_dense_and_bm25_known_good_ordering():
    """Regression guard for the implicit-ascending sort in
    ``fuse_dense_and_bm25``.

    Inputs:
    - dense_hits ordered by mem0 score desc → ``[a, b, c]`` (a is best).
    - bm25_hits  ordered by BM25 score desc → ``[c, b, a]`` (c is best).

    Expected RRF scores with k=60:
    - a: 1/(60+1) [dense rank 0] + 1/(60+3) [bm25 rank 2] = 1/61 + 1/63
    - b: 1/(60+2) [dense rank 1] + 1/(60+2) [bm25 rank 1] = 2/62
    - c: 1/(60+3) [dense rank 2] + 1/(60+1) [bm25 rank 0] = 1/63 + 1/61

    Numerically a == c by construction (same pair of denominators, just
    swapped) and they're slightly above b (the convexity of ``1/x``
    means the asymmetric pair ``(1/61 + 1/63)`` beats the symmetric
    ``(1/62 + 1/62)`` — Jensen's inequality, in tiny). So the
    deterministic order is ``[a, c, b]``: ``a`` and ``c`` tied at the
    top with lexicographic tiebreak, ``b`` last.

    If anyone refactors and accidentally flips one of the sorts to
    ``reverse=True``, the dense list would be re-read as ``[c, b, a]``
    → ``c``'s score becomes ``2 * 1/61`` (best in both) and ``a``'s
    becomes ``2 * 1/63`` (worst in both). Order would flip to
    ``[c, b, a]``. This test catches that drift.
    """
    dense_hits = [
        {"metadata": {"memory_id": "a"}, "score": 0.9},
        {"metadata": {"memory_id": "b"}, "score": 0.7},
        {"metadata": {"memory_id": "c"}, "score": 0.5},
    ]
    bm25_hits = [("c", 9.0), ("b", 5.0), ("a", 1.0)]
    out = fuse_dense_and_bm25(dense_hits, bm25_hits, rrf_k=60)
    ids = [(h.get("metadata") or {}).get("memory_id") or h.get("id") for h in out]
    # a and c tied at top (lex tiebreak puts a first), b last.
    assert ids == ["a", "c", "b"]
    by_id = {(h.get("metadata") or {}).get("memory_id") or h.get("id"): h for h in out}
    expected_outer = 1.0 / 61 + 1.0 / 63
    expected_middle = 2.0 / 62
    assert by_id["a"]["rrf_score"] == pytest.approx(expected_outer)
    assert by_id["c"]["rrf_score"] == pytest.approx(expected_outer)
    assert by_id["b"]["rrf_score"] == pytest.approx(expected_middle)
    # The whole point of the regression guard: outer > middle (Jensen).
    assert by_id["a"]["rrf_score"] > by_id["b"]["rrf_score"]


def test_fuse_dense_and_bm25_asymmetric_known_good():
    """A second known-good check with asymmetric inputs to catch any
    sort regression that happens to preserve symmetric ties."""
    # 'a' is BEST in dense (rank 0) and ABSENT in BM25.
    # 'b' is ABSENT in dense and BEST in BM25 (rank 0).
    # 'c' is in both but at rank 1 in each.
    dense_hits = [
        {"metadata": {"memory_id": "a"}, "score": 0.9},
        {"metadata": {"memory_id": "c"}, "score": 0.7},
    ]
    bm25_hits = [("b", 9.0), ("c", 5.0)]
    out = fuse_dense_and_bm25(dense_hits, bm25_hits, rrf_k=60)
    scores = {
        (h.get("metadata") or {}).get("memory_id") or h.get("id"): h["rrf_score"] for h in out
    }
    # 'c' appears in BOTH at rank 1 → contribution 2/62 ≈ 0.03226
    # 'a' and 'b' each appear once at rank 0 → contribution 1/61 ≈ 0.01639
    assert scores["c"] == pytest.approx(2.0 / 62)
    assert scores["a"] == pytest.approx(1.0 / 61)
    assert scores["b"] == pytest.approx(1.0 / 61)
    # 'c' must outrank both 'a' and 'b' (it's in both lists at the same
    # near-top rank). If a sort flipped, 'c' would drop to last position.
    ids = [(h.get("metadata") or {}).get("memory_id") or h.get("id") for h in out]
    assert ids[0] == "c"


# ---------------------------------------------------------------------------
# HybridRetriever — invalidate cycle stress (regression for iter_memories)
# ---------------------------------------------------------------------------


def test_hybrid_retriever_multiple_invalidate_search_cycles(tmp_path):
    """Regression: ensure ``_rebuild`` reads the full corpus through
    ``iter_memories()`` correctly across multiple invalidate→search
    cycles. A previous audit pass swapped ``storage.list(limit=10**9)``
    for ``storage.iter_memories()`` — this test exercises the new
    streaming path under repeated saves to make sure the BM25 corpus
    stays in sync after every invalidation.

    The historic foot-gun: a generator-based source that gets exhausted
    on the first read and yields nothing on the second. Our
    ``iter_memories`` re-walks the directory each call, so this passes —
    the test is the canary that locks that contract.
    """
    storage = VaultStorage(tmp_path)
    retriever = HybridRetriever(storage)

    # Cycle 1: empty corpus → nothing matches.
    assert retriever.search("alpha_term") == []

    # Cycle 2: save one + invalidate → it shows up.
    storage.save(content="body with alpha_term included", title="alpha")
    retriever.invalidate()
    res = retriever.search("alpha_term")
    assert [i for i, _ in res] == ["alpha"]

    # Cycle 3: save a second + invalidate → both queryable, BM25 corpus
    # contains both docs (proves _rebuild didn't only see the new one
    # or only the old one).
    storage.save(content="body with beta_term and alpha_term", title="beta")
    retriever.invalidate()
    res_alpha = retriever.search("alpha_term")
    res_beta = retriever.search("beta_term")
    assert {i for i, _ in res_alpha} == {"alpha", "beta"}
    assert [i for i, _ in res_beta] == ["beta"]

    # Cycle 4: a third invalidate without any change must yield the
    # same corpus (idempotent rebuild).
    retriever.invalidate()
    res_alpha_again = retriever.search("alpha_term")
    assert {i for i, _ in res_alpha_again} == {"alpha", "beta"}

    # Cycle 5: save a third + invalidate → confirms the streaming path
    # still works after multiple back-to-back rebuilds.
    storage.save(content="gamma_term only here", title="gamma")
    retriever.invalidate()
    res_gamma = retriever.search("gamma_term")
    assert [i for i, _ in res_gamma] == ["gamma"]
    res_alpha_final = retriever.search("alpha_term")
    assert {i for i, _ in res_alpha_final} == {"alpha", "beta"}

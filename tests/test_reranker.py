"""Tests for ``mem_vault.retrieval.LocalReranker`` and the search integration.

The cross-encoder model itself isn't loaded — we mock ``fastembed`` at
the import boundary so tests stay offline + fast. The behavioral
contracts we're checking:

1. When fastembed is unavailable, the reranker becomes a no-op
   pass-through (returns top-k of the input order).
2. When fastembed scores are higher for hit B than hit A, the output
   reflects that swap.
3. Each returned hit carries a ``rerank_score`` field.
4. Hits with no extractable body are kept (tail position) instead of
   dropped, so we never silently lose candidates.
5. ``MemVaultService.search`` integrates the reranker only when
   ``Config.reranker_enabled=True``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from mem_vault.retrieval import LocalReranker, _extract_body


# ---------------------------------------------------------------------------
# _extract_body — tolerant of mem0 hit shape variants
# ---------------------------------------------------------------------------


def test_extract_body_from_string_memory():
    assert _extract_body({"memory": "the body"}) == "the body"


def test_extract_body_from_dict_memory_with_body_key():
    assert _extract_body({"memory": {"body": "abc"}}) == "abc"


def test_extract_body_from_dict_memory_with_content_key():
    assert _extract_body({"memory": {"content": "xyz"}}) == "xyz"


def test_extract_body_from_snippet_fallback():
    assert _extract_body({"snippet": "from snippet"}) == "from snippet"


def test_extract_body_returns_empty_when_no_body():
    assert _extract_body({"id": "no-body"}) == ""
    assert _extract_body({"memory": ""}) == ""
    assert _extract_body({"memory": {"body": "   "}}) == ""


# ---------------------------------------------------------------------------
# LocalReranker — fastembed unavailable path
# ---------------------------------------------------------------------------


def test_reranker_falls_back_to_passthrough_when_import_fails(monkeypatch):
    """If fastembed isn't installed, ``available`` stays False and rerank passes through."""

    def _fake_import(name, *args, **kwargs):
        if name.startswith("fastembed"):
            raise ImportError("fastembed not installed")
        return _orig_import(name, *args, **kwargs)

    _orig_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )
    monkeypatch.setattr("builtins.__import__", _fake_import)

    reranker = LocalReranker()
    assert reranker.available is False
    candidates = [
        {"memory": "first body"},
        {"memory": "second body"},
        {"memory": "third body"},
    ]
    out = reranker.rerank("query", candidates, top_k=2)
    # Pass-through: order preserved, only top_k taken.
    assert len(out) == 2
    assert out[0]["memory"] == "first body"
    assert out[1]["memory"] == "second body"


def test_reranker_zero_top_k_returns_empty():
    reranker = LocalReranker()
    assert reranker.rerank("q", [{"memory": "x"}], top_k=0) == []


def test_reranker_empty_candidates_returns_empty():
    reranker = LocalReranker()
    assert reranker.rerank("q", [], top_k=5) == []


# ---------------------------------------------------------------------------
# LocalReranker — with stubbed fastembed encoder
# ---------------------------------------------------------------------------


class _FakeEncoder:
    """Pretends to be fastembed.TextCrossEncoder with deterministic scores."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.calls: list[tuple[str, list[str]]] = []
        self._scores: list[float] = []

    def set_scores(self, scores: list[float]) -> None:
        self._scores = scores

    def rerank(self, query, documents, batch_size=64):
        self.calls.append((query, list(documents)))
        # Return one score per document (or 0.0 if not pre-set).
        if not self._scores:
            return [0.0] * len(documents)
        if len(self._scores) < len(documents):
            return self._scores + [0.0] * (len(documents) - len(self._scores))
        return self._scores[: len(documents)]


@pytest.fixture
def stub_encoder(monkeypatch):
    """Inject a fake fastembed.rerank.cross_encoder.TextCrossEncoder."""
    fake_encoder_instance = _FakeEncoder("stub-model")

    fake_module = types.ModuleType("fastembed.rerank.cross_encoder")
    fake_module.TextCrossEncoder = lambda model: fake_encoder_instance  # type: ignore[attr-defined]
    fake_parent = types.ModuleType("fastembed.rerank")
    fake_parent.cross_encoder = fake_module  # type: ignore[attr-defined]
    fake_root = types.ModuleType("fastembed")
    fake_root.rerank = fake_parent  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "fastembed", fake_root)
    monkeypatch.setitem(sys.modules, "fastembed.rerank", fake_parent)
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", fake_module)

    return fake_encoder_instance


def test_reranker_reorders_by_score(stub_encoder):
    reranker = LocalReranker()
    # B should rank higher than A and C.
    stub_encoder.set_scores([0.2, 0.9, 0.5])
    candidates = [
        {"memory": "first body", "id": "a"},
        {"memory": "second body", "id": "b"},
        {"memory": "third body", "id": "c"},
    ]
    out = reranker.rerank("query", candidates, top_k=3)
    assert [h["id"] for h in out] == ["b", "c", "a"]
    assert out[0]["rerank_score"] == pytest.approx(0.9)
    assert out[1]["rerank_score"] == pytest.approx(0.5)
    assert out[2]["rerank_score"] == pytest.approx(0.2)


def test_reranker_caps_to_top_k(stub_encoder):
    reranker = LocalReranker()
    stub_encoder.set_scores([0.9, 0.5, 0.1])
    candidates = [
        {"memory": "a", "id": "a"},
        {"memory": "b", "id": "b"},
        {"memory": "c", "id": "c"},
    ]
    out = reranker.rerank("q", candidates, top_k=2)
    assert len(out) == 2
    assert out[0]["id"] == "a"
    assert out[1]["id"] == "b"


def test_reranker_keeps_unscorable_candidates_at_tail(stub_encoder):
    """Hits with no body shouldn't be dropped — better in the tail than gone."""
    reranker = LocalReranker()
    stub_encoder.set_scores([0.7])  # only 1 score because only 1 body
    candidates = [
        {"id": "no-body-1"},  # no body → unscorable
        {"memory": "real body", "id": "scored"},
        {"id": "no-body-2"},
    ]
    out = reranker.rerank("q", candidates, top_k=5)
    # Scored hit first, then the unscorable ones (in original order).
    assert out[0]["id"] == "scored"
    assert {h["id"] for h in out[1:]} == {"no-body-1", "no-body-2"}


def test_reranker_swallows_score_errors(stub_encoder):
    """If the encoder raises, fall back to bi-encoder order, don't crash."""
    reranker = LocalReranker()

    def _boom(query, docs, batch_size=64):
        raise RuntimeError("model OOM")

    stub_encoder.rerank = _boom  # type: ignore[assignment]
    candidates = [{"memory": "a"}, {"memory": "b"}]
    out = reranker.rerank("q", candidates, top_k=2)
    assert len(out) == 2
    # No rerank_score because rerank failed — original dicts back.
    assert "rerank_score" not in out[0]


# ---------------------------------------------------------------------------
# MemVaultService.search integration with reranker
# ---------------------------------------------------------------------------


from mem_vault.config import Config
from mem_vault.index import _CircuitBreaker
from mem_vault.server import MemVaultService


class _IntegStubIndex:
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
def service_factory(tmp_path: Path):
    def _make(**overrides) -> MemVaultService:
        cfg_kwargs = {
            "vault_path": str(tmp_path),
            "memory_subdir": "memory",
            "state_dir": str(tmp_path / "state"),
            "user_id": "tester",
            "auto_extract_default": False,
            "llm_timeout_s": 0,
            "max_content_size": 0,
            "auto_link_default": False,  # keep search tests focused
            **overrides,
        }
        config = Config(**cfg_kwargs)
        config.qdrant_collection = "test"
        config.state_dir.mkdir(parents=True, exist_ok=True)
        config.memory_dir.mkdir(parents=True, exist_ok=True)
        return MemVaultService(config)

    return _make


async def test_search_does_not_invoke_reranker_when_disabled(service_factory):
    """Default config → reranker_enabled=False → reranker untouched."""
    service = service_factory()
    seed = await service.save({"content": "body", "title": "T", "type": "note"})
    service.index = _IntegStubIndex(  # type: ignore[assignment]
        [{"score": 0.9, "metadata": {"memory_id": seed["memory"]["id"]}}]
    )
    # Force the reranker to be a sentinel — if it gets called we'll know.
    service._reranker = "should-not-be-touched"  # type: ignore[assignment]
    res = await service.search({"query": "x"})
    assert res["ok"] is True
    # Sentinel survived → rerank() was never called.
    assert service._reranker == "should-not-be-touched"


async def test_search_calls_reranker_when_enabled(service_factory, stub_encoder):
    service = service_factory(reranker_enabled=True)

    # Two seeded memories; the index returns them with bi-encoder scores
    # putting "low" first, but the rerank flips them so "high" wins.
    s_low = await service.save({"content": "low body", "title": "low", "type": "note"})
    s_high = await service.save({"content": "high body", "title": "high", "type": "note"})

    service.index = _IntegStubIndex(  # type: ignore[assignment]
        [
            {"score": 0.6, "metadata": {"memory_id": s_low["memory"]["id"]}, "memory": "low body"},
            {
                "score": 0.5,
                "metadata": {"memory_id": s_high["memory"]["id"]},
                "memory": "high body",
            },
        ]
    )
    # Pretend the cross-encoder thinks "high body" is more relevant.
    stub_encoder.set_scores([0.2, 0.95])

    res = await service.search({"query": "search query", "k": 2})
    assert res["ok"] is True
    assert res["count"] == 2
    # Order should now reflect the rerank: "high" first, "low" second.
    assert res["results"][0]["id"] == s_high["memory"]["id"]
    assert res["results"][1]["id"] == s_low["memory"]["id"]

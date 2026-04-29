"""Tests for the consolidation module — pure unit tests with mocked LLM/index."""

from __future__ import annotations

from typing import Any

from mem_vault.consolidate import (
    Pair,
    Resolution,
    apply_resolution,
    find_candidate_pairs,
)
from mem_vault.storage import VaultStorage


class FakeIndex:
    """A minimal VectorIndex stub that returns canned hits per query string."""

    def __init__(self, canned: dict[str, list[dict[str, Any]]]):
        self.canned = canned
        self.deletes: list[tuple[str, str, str]] = []
        self.adds: list[tuple[str, dict[str, Any]]] = []

    def search(self, query, *, user_id, top_k, threshold=0.0, filters=None):
        # Match by prefix — the consolidate code passes the full body but our
        # canned keys are body[:N] for clarity in tests.
        for key, hits in self.canned.items():
            if query.startswith(key):
                return [h for h in hits if (h.get("score") or 0) >= threshold]
        return []

    def delete_by_metadata(self, key, value, user_id):
        self.deletes.append((key, value, user_id))
        return 1

    def add(self, content, *, user_id, agent_id=None, metadata=None, auto_extract=False):
        self.adds.append((content, metadata or {}))
        return [{"id": "fake", "memory": content[:80], "event": "ADD"}]


# ---------------------------------------------------------------------------
# find_candidate_pairs
# ---------------------------------------------------------------------------


def test_find_pairs_dedupes_and_sorts(tmp_path):
    storage = VaultStorage(tmp_path)
    a = storage.save(content="Apple is red.", title="apple")
    b = storage.save(content="Banana is yellow.", title="banana")
    c = storage.save(content="Cherry is red.", title="cherry")

    fake = FakeIndex(
        {
            "Apple is red.": [
                {"metadata": {"memory_id": c.id}, "score": 0.91},
                {"metadata": {"memory_id": b.id}, "score": 0.45},
            ],
            "Banana is yellow.": [
                {"metadata": {"memory_id": a.id}, "score": 0.45},
            ],
            "Cherry is red.": [
                {"metadata": {"memory_id": a.id}, "score": 0.91},  # mirror of apple→cherry
                {"metadata": {"memory_id": b.id}, "score": 0.30},
            ],
        }
    )

    pairs = find_candidate_pairs(storage, fake, threshold=0.8, user_id="default")

    # Only one pair (apple/cherry) should survive — the b<->a edge is below threshold.
    assert len(pairs) == 1
    assert {pairs[0].a.id, pairs[0].b.id} == {a.id, c.id}
    assert pairs[0].score >= 0.8


def test_find_pairs_skips_self_hits(tmp_path):
    storage = VaultStorage(tmp_path)
    a = storage.save(content="Solo memory.", title="solo")
    fake = FakeIndex(
        {
            "Solo memory.": [
                {"metadata": {"memory_id": a.id}, "score": 1.0},  # self-hit
            ],
        }
    )
    pairs = find_candidate_pairs(storage, fake, threshold=0.5)
    assert pairs == []


def test_find_pairs_below_threshold_returns_empty(tmp_path):
    storage = VaultStorage(tmp_path)
    storage.save(content="A.", title="a")
    storage.save(content="B.", title="b")
    fake = FakeIndex({"A.": [{"metadata": {"memory_id": "b"}, "score": 0.4}]})
    pairs = find_candidate_pairs(storage, fake, threshold=0.85)
    assert pairs == []


# ---------------------------------------------------------------------------
# apply_resolution
# ---------------------------------------------------------------------------


def _seeded(tmp_path) -> tuple[VaultStorage, FakeIndex, Pair]:
    storage = VaultStorage(tmp_path)
    a = storage.save(content="A body", title="a")
    b = storage.save(content="B body", title="b")
    fake = FakeIndex({})
    pair = Pair(a=a, b=b, score=0.95)
    return storage, fake, pair


def test_apply_keep_both_does_nothing(tmp_path):
    storage, fake, pair = _seeded(tmp_path)
    res = Resolution(action="KEEP_BOTH", rationale="different topics")
    out = apply_resolution(storage, fake, pair, res)
    assert out["action"] == "KEEP_BOTH"
    assert storage.get(pair.a.id) is not None
    assert storage.get(pair.b.id) is not None
    assert fake.deletes == []
    assert fake.adds == []


def test_apply_keep_first_deletes_b(tmp_path):
    storage, fake, pair = _seeded(tmp_path)
    res = Resolution(action="KEEP_FIRST", rationale="A subsumes B")
    out = apply_resolution(storage, fake, pair, res)
    assert out["action"] == "KEEP_FIRST"
    assert storage.get(pair.a.id) is not None
    assert storage.get(pair.b.id) is None
    assert any(d[1] == pair.b.id for d in fake.deletes)


def test_apply_keep_second_deletes_a(tmp_path):
    storage, fake, pair = _seeded(tmp_path)
    res = Resolution(action="KEEP_SECOND", rationale="B subsumes A")
    out = apply_resolution(storage, fake, pair, res)
    assert out["action"] == "KEEP_SECOND"
    assert storage.get(pair.a.id) is None
    assert storage.get(pair.b.id) is not None
    assert any(d[1] == pair.a.id for d in fake.deletes)


def test_apply_merge_rewrites_older_keeps_id(tmp_path):
    storage, fake, _ = _seeded(tmp_path)
    # Force a known temporal ordering: a is older than b.
    a = storage.get("a")
    b = storage.get("b")
    a.created = "2026-01-01T00:00:00-03:00"
    b.created = "2026-02-01T00:00:00-03:00"
    storage._write(a)
    storage._write(b)

    pair = Pair(a=storage.get("a"), b=storage.get("b"), score=0.97)
    res = Resolution(
        action="MERGE",
        rationale="overlap",
        merged_body="Fused body content",
        merged_title="Fused",
    )
    out = apply_resolution(storage, fake, pair, res)
    assert out["action"] == "MERGE"

    # Older id (a) survives, newer (b) is gone.
    assert storage.get("a") is not None
    assert storage.get("b") is None
    assert "Fused" in (storage.get("a").body + storage.get("a").name)

    # Index got both deletes + one add for the merged content.
    deleted_ids = [d[1] for d in fake.deletes]
    assert "a" in deleted_ids and "b" in deleted_ids
    assert len(fake.adds) == 1
    assert fake.adds[0][1].get("memory_id") == "a"
    assert "merged" in (fake.adds[0][1].get("tags") or [])

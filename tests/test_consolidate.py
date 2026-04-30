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

    def get_by_metadata(self, key, value, user_id):
        # Stubbed enough for the merge rollback path. Real consolidate uses
        # this only as best-effort snapshot; returning [] is safe.
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


# ---------------------------------------------------------------------------
# MERGE crash-safety: a mid-merge index failure must not leave the system
# with both memories invisible to search. The .md gets rolled back, and the
# newer.md is left untouched on disk.
# ---------------------------------------------------------------------------


class _CrashingIndex(FakeIndex):
    """Like FakeIndex but ``add`` raises the first time after the delete.

    We let rollback's add succeed so the test can verify the recovered
    state — that's the realistic flake (Ollama hiccup on one call, then
    recovers).
    """

    def __init__(self):
        super().__init__(canned={})
        self._add_calls = 0

    def add(self, content, *, user_id, agent_id=None, metadata=None, auto_extract=False):
        self._add_calls += 1
        if self._add_calls == 1:
            raise RuntimeError("ollama crashed mid-merge")
        return super().add(content, user_id=user_id, agent_id=agent_id, metadata=metadata)


def test_apply_merge_crash_rolls_back_older_and_keeps_newer(tmp_path):
    storage = VaultStorage(tmp_path)
    a = storage.save(content="A original body", title="a")
    b = storage.save(content="B original body", title="b")
    a.created = "2026-01-01T00:00:00-03:00"
    b.created = "2026-02-01T00:00:00-03:00"
    storage._write(a)
    storage._write(b)

    crashing = _CrashingIndex()
    pair = Pair(a=storage.get("a"), b=storage.get("b"), score=0.97)
    res = Resolution(
        action="MERGE",
        rationale="overlap",
        merged_body="Fused body",
        merged_title="Fused",
    )

    import pytest

    with pytest.raises(RuntimeError):
        apply_resolution(storage, crashing, pair, res)

    # older.md must have its original body back (rollback succeeded)
    a_after = storage.get("a")
    assert a_after is not None
    assert a_after.body == "A original body"
    # newer.md must still exist — we never delete it before the embed succeeds
    b_after = storage.get("b")
    assert b_after is not None
    assert b_after.body == "B original body"
    # Index recovered: rollback added back the previous body so search finds it
    assert any(
        meta.get("memory_id") == "a" and "A original body" in content
        for content, meta in crashing.adds
    )


class _AddAlwaysCrashingIndex(FakeIndex):
    """Hard failure mode: every ``add`` raises. Rollback re-add also fails.

    This proves the contract: even when the embed pipeline is fully dead,
    the .md state stays consistent (older keeps its old body, newer keeps
    its file). Recovery is then via ``mem-vault reindex``.
    """

    def __init__(self):
        super().__init__(canned={})

    def add(self, *args, **kwargs):
        raise RuntimeError("ollama is dead, all calls fail")


def test_apply_merge_total_failure_still_keeps_disk_consistent(tmp_path):
    storage = VaultStorage(tmp_path)
    a = storage.save(content="A original body", title="a")
    b = storage.save(content="B original body", title="b")
    a.created = "2026-01-01T00:00:00-03:00"
    b.created = "2026-02-01T00:00:00-03:00"
    storage._write(a)
    storage._write(b)

    dead = _AddAlwaysCrashingIndex()
    pair = Pair(a=storage.get("a"), b=storage.get("b"), score=0.97)
    res = Resolution(
        action="MERGE",
        rationale="overlap",
        merged_body="Fused body",
        merged_title="Fused",
    )

    import pytest

    with pytest.raises(RuntimeError):
        apply_resolution(storage, dead, pair, res)

    # Both .md files must still be on disk with their original bodies.
    # The user can run ``mem-vault reindex`` to restore the index.
    a_after = storage.get("a")
    b_after = storage.get("b")
    assert a_after is not None and a_after.body == "A original body"
    assert b_after is not None and b_after.body == "B original body"


def test_apply_keep_first_orders_storage_delete_before_index(tmp_path):
    """KEEP_FIRST must delete the .md before the index entry. A crash in
    between is recoverable via reindex's orphan sweep; the reverse order
    would leave a file with no embedding (silent search miss).
    """
    storage, fake, pair = _seeded(tmp_path)

    call_order: list[str] = []
    real_delete_file = storage.delete

    def spy_delete(mid):
        call_order.append(f"storage.delete({mid})")
        return real_delete_file(mid)

    def spy_delete_idx(key, value, user_id):
        call_order.append(f"index.delete({value})")
        return 1

    storage.delete = spy_delete  # type: ignore[assignment]
    fake.delete_by_metadata = spy_delete_idx  # type: ignore[assignment]

    res = Resolution(action="KEEP_FIRST", rationale="A subsumes B")
    apply_resolution(storage, fake, pair, res)

    assert call_order == [f"storage.delete({pair.b.id})", f"index.delete({pair.b.id})"]

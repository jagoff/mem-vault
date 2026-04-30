"""Tests for the feedback loop (usage tracking + helpful/unhelpful + search boost).

Covers:

1. ``Memory`` dataclass: new fields, ``helpful_ratio`` property, round-trip
   through frontmatter (presence vs absence when counters are 0).
2. ``VaultStorage.record_usage`` / ``record_feedback``: increment without
   touching ``updated`` so the hash-based incremental reindex doesn't
   spuriously re-embed on a plain thumbs event.
3. ``MemVaultService.search``: the boost multiplies ``score`` by
   ``1 + usage_boost * helpful_ratio`` (clamped to [0, 1]) and reorders
   the top-k. Tracking auto-increments ``usage_count`` on every returned
   hit when ``usage_tracking_enabled=True``.
4. ``MemVaultService.feedback``: tool wrapper, validation errors, return
   envelope shape.
5. Backward compatibility: legacy memorias without the new fields load
   cleanly (all counters = 0, ``helpful_ratio = 0.0``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mem_vault.config import Config
from mem_vault.index import _CircuitBreaker
from mem_vault.server import MemVaultService
from mem_vault.storage import Memory, VaultStorage

# ---------------------------------------------------------------------------
# Memory dataclass — new fields + helpful_ratio property
# ---------------------------------------------------------------------------


def test_memory_defaults_all_counters_zero():
    mem = Memory(id="x", name="x", description="")
    assert mem.usage_count == 0
    assert mem.helpful_count == 0
    assert mem.unhelpful_count == 0
    assert mem.last_used == ""
    assert mem.helpful_ratio == 0.0


@pytest.mark.parametrize(
    "helpful,unhelpful,expected",
    [
        (0, 0, 0.0),
        (3, 0, 1.0),
        (0, 3, -1.0),
        (3, 1, 0.5),
        (1, 3, -0.5),
        (10, 10, 0.0),
    ],
)
def test_helpful_ratio_formula(helpful, unhelpful, expected):
    mem = Memory(
        id="x",
        name="x",
        description="",
        helpful_count=helpful,
        unhelpful_count=unhelpful,
    )
    assert mem.helpful_ratio == pytest.approx(expected, abs=1e-6)


def test_to_frontmatter_omits_zero_counters():
    """Pristine memorias keep their frontmatter tight (back-compat)."""
    mem = Memory(id="x", name="x", description="")
    fm = mem.to_frontmatter()
    assert "usage_count" not in fm
    assert "helpful_count" not in fm
    assert "unhelpful_count" not in fm
    assert "last_used" not in fm


def test_to_frontmatter_includes_nonzero_counters():
    mem = Memory(
        id="x",
        name="x",
        description="",
        usage_count=5,
        helpful_count=3,
        unhelpful_count=1,
        last_used="2026-04-29T20:00:00-03:00",
    )
    fm = mem.to_frontmatter()
    assert fm["usage_count"] == 5
    assert fm["helpful_count"] == 3
    assert fm["unhelpful_count"] == 1
    assert fm["last_used"] == "2026-04-29T20:00:00-03:00"


# ---------------------------------------------------------------------------
# VaultStorage.record_usage / record_feedback
# ---------------------------------------------------------------------------


def test_record_usage_increments_count_and_bumps_last_used(tmp_path):
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="x", title="t")
    assert mem.usage_count == 0

    updated = storage.record_usage(mem.id)
    assert updated is not None
    assert updated.usage_count == 1
    assert updated.last_used != ""

    again = storage.record_usage(mem.id)
    assert again is not None
    assert again.usage_count == 2


def test_record_usage_does_not_bump_updated_timestamp(tmp_path):
    """Usage is ambient signal, not content — ``updated`` must not change."""
    import time

    storage = VaultStorage(tmp_path)
    mem = storage.save(content="x", title="t")
    original_updated = mem.updated

    time.sleep(1.1)  # ensure timestamp granularity
    bumped = storage.record_usage(mem.id)
    assert bumped is not None
    assert bumped.updated == original_updated, (
        "record_usage should NOT touch `updated` (only last_used + counters)"
    )


def test_record_usage_on_missing_memory_returns_none(tmp_path):
    storage = VaultStorage(tmp_path)
    assert storage.record_usage("does_not_exist") is None


def test_record_feedback_helpful_increments_helpful_count(tmp_path):
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="x", title="t")
    updated = storage.record_feedback(mem.id, helpful=True)
    assert updated is not None
    assert updated.helpful_count == 1
    assert updated.unhelpful_count == 0
    assert updated.last_used != ""


def test_record_feedback_unhelpful_increments_unhelpful_count(tmp_path):
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="x", title="t")
    updated = storage.record_feedback(mem.id, helpful=False)
    assert updated is not None
    assert updated.helpful_count == 0
    assert updated.unhelpful_count == 1


def test_record_feedback_none_only_bumps_last_used(tmp_path):
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="x", title="t")
    updated = storage.record_feedback(mem.id, helpful=None)
    assert updated is not None
    assert updated.helpful_count == 0
    assert updated.unhelpful_count == 0
    assert updated.last_used != ""


def test_record_feedback_missing_returns_none(tmp_path):
    storage = VaultStorage(tmp_path)
    assert storage.record_feedback("does_not_exist", helpful=True) is None


# ---------------------------------------------------------------------------
# Round-trip: legacy files (no counter fields) must load cleanly
# ---------------------------------------------------------------------------


def test_legacy_memory_without_counters_loads_with_zeros(tmp_path):
    """A .md from before the feedback loop should not need migration."""
    legacy = """---
name: Legacy
description: pre-feedback-loop memo
type: note
tags: [x]
created: 2026-01-01T00:00:00-03:00
updated: 2026-01-01T00:00:00-03:00
---
body
"""
    storage = VaultStorage(tmp_path)
    (tmp_path / "legacy.md").write_text(legacy, encoding="utf-8")
    mem = storage.get("legacy")
    assert mem is not None
    assert mem.usage_count == 0
    assert mem.helpful_count == 0
    assert mem.unhelpful_count == 0
    assert mem.last_used == ""


def test_counter_round_trip_through_disk(tmp_path):
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="x", title="t")
    storage.record_feedback(mem.id, helpful=True)
    storage.record_feedback(mem.id, helpful=True)
    storage.record_feedback(mem.id, helpful=False)
    storage.record_usage(mem.id)

    reloaded = storage.get(mem.id)
    assert reloaded is not None
    assert reloaded.helpful_count == 2
    assert reloaded.unhelpful_count == 1
    assert reloaded.usage_count == 1
    assert reloaded.helpful_ratio == pytest.approx(1 / 3, abs=1e-6)


def test_corrupt_counter_value_falls_back_to_zero(tmp_path):
    """A hand-edited file with ``usage_count: abc`` must not crash the reader."""
    storage = VaultStorage(tmp_path)
    (tmp_path / "bad.md").write_text(
        """---
name: bad
description: ""
type: note
usage_count: abc
helpful_count: "nope"
---
body
""",
        encoding="utf-8",
    )
    mem = storage.get("bad")
    assert mem is not None
    assert mem.usage_count == 0
    assert mem.helpful_count == 0


# ---------------------------------------------------------------------------
# MemVaultService.feedback — MCP tool wrapper
# ---------------------------------------------------------------------------


class _StubIndex:
    def __init__(self, hits=None):
        self.hits = hits or []
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
            "auto_link_default": False,
            **overrides,
        }
        config = Config(**cfg_kwargs)
        config.qdrant_collection = "test"
        config.state_dir.mkdir(parents=True, exist_ok=True)
        config.memory_dir.mkdir(parents=True, exist_ok=True)
        service = MemVaultService(config)
        service.index = _StubIndex()  # type: ignore[assignment]
        return service

    return _make


async def test_feedback_tool_thumbs_up(service_factory):
    service = service_factory()
    seed = await service.save({"content": "body", "title": "T"})
    mid = seed["memory"]["id"]

    res = await service.feedback({"id": mid, "helpful": True})
    assert res["ok"] is True
    assert res["helpful_count"] == 1
    assert res["unhelpful_count"] == 0
    assert res["last_used"]
    assert res["helpful_ratio"] == 1.0


async def test_feedback_tool_thumbs_down_then_up(service_factory):
    service = service_factory()
    seed = await service.save({"content": "body", "title": "T"})
    mid = seed["memory"]["id"]

    await service.feedback({"id": mid, "helpful": False})
    res = await service.feedback({"id": mid, "helpful": True})
    assert res["helpful_count"] == 1
    assert res["unhelpful_count"] == 1
    assert res["helpful_ratio"] == 0.0


async def test_feedback_tool_null_helpful_is_plain_usage(service_factory):
    service = service_factory()
    seed = await service.save({"content": "body", "title": "T"})
    mid = seed["memory"]["id"]

    res = await service.feedback({"id": mid, "helpful": None})
    assert res["ok"] is True
    assert res["helpful_count"] == 0
    assert res["unhelpful_count"] == 0
    assert res["last_used"]


async def test_feedback_tool_missing_id_returns_validation_error(service_factory):
    service = service_factory()
    res = await service.feedback({})
    assert res["ok"] is False
    assert res["code"] == "validation_failed"


async def test_feedback_tool_missing_memory_returns_not_found(service_factory):
    service = service_factory()
    res = await service.feedback({"id": "nonexistent"})
    assert res["ok"] is False
    assert res["code"] == "not_found"


async def test_feedback_tool_invalid_helpful_type_returns_validation_error(service_factory):
    service = service_factory()
    seed = await service.save({"content": "body", "title": "T"})
    res = await service.feedback({"id": seed["memory"]["id"], "helpful": "yes"})
    assert res["ok"] is False
    assert res["code"] == "validation_failed"


# ---------------------------------------------------------------------------
# MemVaultService.search — usage boost reorders top-k
# ---------------------------------------------------------------------------


async def test_search_without_feedback_preserves_base_order(service_factory):
    """With no feedback, the boost is 1.0 for every hit; order matches scores."""
    service = service_factory()
    a = await service.save({"content": "alpha body", "title": "A"})
    b = await service.save({"content": "beta body", "title": "B"})

    service.index = _StubIndex(
        [
            {"score": 0.9, "metadata": {"memory_id": a["memory"]["id"]}, "memory": "alpha body"},
            {"score": 0.6, "metadata": {"memory_id": b["memory"]["id"]}, "memory": "beta body"},
        ]
    )
    res = await service.search({"query": "q", "k": 2})
    assert res["count"] == 2
    assert res["results"][0]["id"] == a["memory"]["id"]
    assert res["results"][1]["id"] == b["memory"]["id"]
    # boost factor is 1.0 (no feedback yet)
    for r in res["results"]:
        assert r["usage_boost"] == 1.0


async def test_search_lifts_memory_with_positive_feedback(service_factory):
    """A memory with thumbs-up should outrank a semantically-closer peer when
    the margin is small enough that the boost can flip them."""
    service = service_factory(usage_boost=0.5)
    a = await service.save({"content": "alpha body", "title": "A"})
    b = await service.save({"content": "beta body", "title": "B"})

    # Stamp helpful feedback on B so it gets boost factor 1.5, and leave A neutral.
    # Base scores: A=0.70, B=0.60. With boost 0.5, B becomes 0.60 * 1.5 = 0.90 > 0.70.
    await service.feedback({"id": b["memory"]["id"], "helpful": True})

    service.index = _StubIndex(
        [
            {"score": 0.70, "metadata": {"memory_id": a["memory"]["id"]}, "memory": "alpha body"},
            {"score": 0.60, "metadata": {"memory_id": b["memory"]["id"]}, "memory": "beta body"},
        ]
    )
    res = await service.search({"query": "q", "k": 2})
    assert res["count"] == 2
    # B (with boost) wins despite lower raw score
    assert res["results"][0]["id"] == b["memory"]["id"]
    assert res["results"][0]["usage_boost"] == 1.5
    # Composed score is raw × boost
    assert res["results"][0]["score"] == pytest.approx(0.90, abs=1e-6)
    assert res["results"][0]["score_raw"] == pytest.approx(0.60, abs=1e-6)


async def test_search_boost_disabled_preserves_raw_order(service_factory):
    """``usage_boost_enabled=False`` → boost does not apply even with feedback."""
    service = service_factory(usage_boost_enabled=False, usage_boost=0.5)
    a = await service.save({"content": "alpha body", "title": "A"})
    b = await service.save({"content": "beta body", "title": "B"})

    await service.feedback({"id": b["memory"]["id"], "helpful": True})

    service.index = _StubIndex(
        [
            {"score": 0.70, "metadata": {"memory_id": a["memory"]["id"]}, "memory": "alpha body"},
            {"score": 0.60, "metadata": {"memory_id": b["memory"]["id"]}, "memory": "beta body"},
        ]
    )
    res = await service.search({"query": "q", "k": 2})
    assert res["results"][0]["id"] == a["memory"]["id"]  # A wins on raw score
    for r in res["results"]:
        assert r["usage_boost"] == 1.0


async def test_search_negative_feedback_does_not_bury_memory(service_factory):
    """A single thumbs-down must not actively demote a memory (clamp at 0)."""
    service = service_factory(usage_boost=0.5)
    a = await service.save({"content": "alpha body", "title": "A"})
    b = await service.save({"content": "beta body", "title": "B"})

    await service.feedback({"id": a["memory"]["id"], "helpful": False})

    service.index = _StubIndex(
        [
            {"score": 0.70, "metadata": {"memory_id": a["memory"]["id"]}, "memory": "alpha body"},
            {"score": 0.60, "metadata": {"memory_id": b["memory"]["id"]}, "memory": "beta body"},
        ]
    )
    res = await service.search({"query": "q", "k": 2})
    # A still wins — its boost is 1.0 (negative ratio clamped to 0).
    assert res["results"][0]["id"] == a["memory"]["id"]
    assert res["results"][0]["usage_boost"] == 1.0


async def test_search_auto_increments_usage_count(service_factory):
    """Every returned hit should have its ``usage_count`` incremented by 1."""
    service = service_factory()
    a = await service.save({"content": "alpha body", "title": "A"})

    service.index = _StubIndex(
        [
            {"score": 0.9, "metadata": {"memory_id": a["memory"]["id"]}, "memory": "alpha body"},
        ]
    )
    # First search: usage_count should go 0 → 1
    await service.search({"query": "q", "k": 1})
    mem = service.storage.get(a["memory"]["id"])
    assert mem is not None
    assert mem.usage_count == 1

    # Second search: 1 → 2
    await service.search({"query": "q", "k": 1})
    mem = service.storage.get(a["memory"]["id"])
    assert mem is not None
    assert mem.usage_count == 2


async def test_search_usage_tracking_disabled_skips_increment(service_factory):
    service = service_factory(usage_tracking_enabled=False)
    a = await service.save({"content": "alpha body", "title": "A"})

    service.index = _StubIndex(
        [
            {"score": 0.9, "metadata": {"memory_id": a["memory"]["id"]}, "memory": "alpha body"},
        ]
    )
    await service.search({"query": "q", "k": 1})
    mem = service.storage.get(a["memory"]["id"])
    assert mem is not None
    assert mem.usage_count == 0

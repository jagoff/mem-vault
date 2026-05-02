"""Tests for the antagonist module + Stop/SessionStart/UserPrompt wires.

Coverage:

- ``is_enabled`` honors MEM_VAULT_ANTAGONIST env var.
- ``detect_from_citations`` returns pending items only when the cited
  memory has non-empty ``contradicts:``.
- ``write_pending`` + ``read_pending`` round-trip + dedupe by cited_id +
  TTL drop expired items.
- ``render_warning_block`` produces a markdown block readable by the
  agent.
- ``clear_pending`` is idempotent.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from mem_vault import antagonist
from mem_vault.storage import Memory


class _FakeStorage:
    """In-memory stand-in for VaultStorage used by detect_from_citations.

    We only need ``.get(id)`` for the antagonist; everything else stays
    out of the test surface.
    """

    def __init__(self, memories: list[Memory]):
        self._by_id = {m.id: m for m in memories}

    def get(self, mem_id: str) -> Memory | None:
        return self._by_id.get(mem_id)


def _m(
    mid: str,
    *,
    contradicts: list[str] | None = None,
    name: str | None = None,
) -> Memory:
    return Memory(
        id=mid,
        name=name or f"Memory {mid}",
        description=f"desc {mid}",
        type="decision",
        tags=[],
        contradicts=contradicts or [],
    )


def test_is_enabled_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEM_VAULT_ANTAGONIST", raising=False)
    assert antagonist.is_enabled() is False
    monkeypatch.setenv("MEM_VAULT_ANTAGONIST", "1")
    assert antagonist.is_enabled() is True
    monkeypatch.setenv("MEM_VAULT_ANTAGONIST", "off")
    assert antagonist.is_enabled() is False


def test_detect_returns_empty_for_memory_without_contradictions() -> None:
    storage = _FakeStorage(
        [
            _m("clean"),
            _m("alone"),
        ]
    )
    items = antagonist.detect_from_citations(["clean", "alone"], storage=storage)
    assert items == []


def test_detect_returns_pending_for_cited_with_contradicts() -> None:
    storage = _FakeStorage(
        [
            _m("a", contradicts=["b"], name="Decision A"),
            _m("b", name="Decision B"),
            _m("c"),
        ]
    )
    items = antagonist.detect_from_citations(["a"], storage=storage)
    assert len(items) == 1
    item = items[0]
    assert item.cited_id == "a"
    assert item.cited_name == "Decision A"
    assert item.contradicts_ids == ["b"]
    assert item.contradicts_summaries[0]["name"] == "Decision B"


def test_detect_skips_dangling_contradiction_ids() -> None:
    """A memory listing contradicts: [deleted] but the deleted .md is gone."""
    storage = _FakeStorage(
        [
            _m("a", contradicts=["nonexistent"]),
        ]
    )
    items = antagonist.detect_from_citations(["a"], storage=storage)
    assert items == []


def test_detect_skips_unknown_cited_ids() -> None:
    storage = _FakeStorage([_m("real")])
    items = antagonist.detect_from_citations(
        ["does-not-exist"], storage=storage
    )
    assert items == []


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    item = antagonist.PendingContradiction(
        cited_id="a",
        cited_name="A",
        cited_description=None,
        contradicts_ids=["b"],
        contradicts_summaries=[{"id": "b", "name": "B", "description": None}],
    )
    assert antagonist.write_pending(tmp_path, [item])
    pending = antagonist.read_pending(tmp_path)
    assert len(pending) == 1
    assert pending[0].cited_id == "a"
    assert pending[0].contradicts_ids == ["b"]


def test_write_pending_dedupes_by_cited_id(tmp_path: Path) -> None:
    item1 = antagonist.PendingContradiction(
        cited_id="a",
        cited_name="A",
        cited_description=None,
        contradicts_ids=["b"],
        contradicts_summaries=[{"id": "b", "name": "B", "description": None}],
    )
    item1_again = antagonist.PendingContradiction(
        cited_id="a",
        cited_name="A",
        cited_description=None,
        contradicts_ids=["b", "c"],
        contradicts_summaries=[
            {"id": "b", "name": "B", "description": None},
            {"id": "c", "name": "C", "description": None},
        ],
    )
    antagonist.write_pending(tmp_path, [item1])
    antagonist.write_pending(tmp_path, [item1_again])
    pending = antagonist.read_pending(tmp_path)
    assert len(pending) == 1
    # Updated to the latest summaries, not duplicated.
    assert sorted(pending[0].contradicts_ids) == ["b", "c"]


def test_read_pending_drops_expired_items(tmp_path: Path) -> None:
    """Items older than ttl_s are filtered out at read time."""
    blob = {
        "version": 1,
        "items": [
            {
                "cited_id": "old",
                "cited_name": "Old",
                "cited_description": None,
                "contradicts_ids": ["x"],
                "contradicts_summaries": [
                    {"id": "x", "name": "X", "description": None}
                ],
                "detected_at": time.time() - 10_000,  # 2.7 hours ago
            },
            {
                "cited_id": "new",
                "cited_name": "New",
                "cited_description": None,
                "contradicts_ids": ["y"],
                "contradicts_summaries": [
                    {"id": "y", "name": "Y", "description": None}
                ],
                "detected_at": time.time() - 60,
            },
        ],
    }
    antagonist.pending_path(tmp_path).write_text(json.dumps(blob), encoding="utf-8")
    # Cutoff at 5 minutes — drops "old", keeps "new".
    pending = antagonist.read_pending(tmp_path, ttl_s=300)
    assert {p.cited_id for p in pending} == {"new"}


def test_read_pending_caps_to_max_items(tmp_path: Path) -> None:
    items = [
        antagonist.PendingContradiction(
            cited_id=f"id_{i}",
            cited_name=f"M{i}",
            cited_description=None,
            contradicts_ids=[f"x{i}"],
            contradicts_summaries=[{"id": f"x{i}", "name": "x", "description": None}],
            detected_at=time.time() + i,  # newer first when sorted
        )
        for i in range(10)
    ]
    antagonist.write_pending(tmp_path, items)
    pending = antagonist.read_pending(tmp_path, max_items=3)
    assert len(pending) == 3


def test_clear_pending_idempotent(tmp_path: Path) -> None:
    antagonist.clear_pending(tmp_path)  # no file: ok
    antagonist.write_pending(
        tmp_path,
        [
            antagonist.PendingContradiction(
                cited_id="a",
                cited_name="A",
                cited_description=None,
                contradicts_ids=["b"],
                contradicts_summaries=[{"id": "b", "name": "B", "description": None}],
            )
        ],
    )
    assert antagonist.pending_path(tmp_path).exists()
    antagonist.clear_pending(tmp_path)
    assert not antagonist.pending_path(tmp_path).exists()
    antagonist.clear_pending(tmp_path)  # idempotent


def test_render_warning_block_includes_cited_and_contradictions() -> None:
    item = antagonist.PendingContradiction(
        cited_id="a_decision",
        cited_name="Decided to use Rust",
        cited_description="2025: Rust over Go for the daemon.",
        contradicts_ids=["b_old_decision"],
        contradicts_summaries=[
            {
                "id": "b_old_decision",
                "name": "Decided to use Go",
                "description": "2024: Go for the daemon, easy ops.",
            }
        ],
    )
    block = antagonist.render_warning_block([item])
    assert "Antagonist" in block
    assert "a_decision" in block
    assert "b_old_decision" in block
    assert "Reconciliá" in block


def test_render_warning_block_empty_input_returns_empty() -> None:
    assert antagonist.render_warning_block([]) == ""

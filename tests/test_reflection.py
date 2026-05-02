"""Tests for the nightly reflection module.

The reflection pass is deterministic when ``apply_consolidate=False``
(no Ollama/Qdrant calls), so we can drive it end-to-end with a tmp
vault and assert on the structural output.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mem_vault.config import Config
from mem_vault.reflection import (
    DEFAULT_LOOKBACK_HOURS,
    ReflectionReport,
    _format_body,
    _today_slug,
    run_reflection,
)
from mem_vault.storage import VaultStorage


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    c = Config(
        vault_path=str(tmp_path),
        memory_subdir="memory",
        state_dir=str(tmp_path / "state"),
        user_id="tester",
        auto_extract_default=False,
        llm_timeout_s=0,
        max_content_size=0,
        auto_link_default=False,
    )
    c.qdrant_collection = "test"
    c.state_dir.mkdir(parents=True, exist_ok=True)
    c.memory_dir.mkdir(parents=True, exist_ok=True)
    return c


def _seed_memories(storage: VaultStorage, count_new: int, count_old: int) -> None:
    """Create ``count_new`` memorias (updated now) + ``count_old`` (updated 100 d ago)."""
    for i in range(count_new):
        storage.save(content=f"new {i}", title=f"new{i}", type="note")
    for i in range(count_old):
        # Backdate the file in-place so the zombie detector triggers.
        mem = storage.save(content=f"old {i}", title=f"old{i}", type="note")
        old_iso = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        # Rewrite frontmatter timestamps directly: no public API to
        # backdate a memory, and that's on purpose — in prod nothing
        # should. Tests reach past the API boundary to simulate age.
        file = storage.memory_dir / f"{mem.id}.md"
        raw = file.read_text(encoding="utf-8")
        raw = raw.replace(mem.updated, old_iso)
        raw = raw.replace(mem.created, old_iso)
        file.write_text(raw, encoding="utf-8")


def test_run_reflection_writes_daily_memory(cfg: Config) -> None:
    storage = VaultStorage(cfg.memory_dir)
    _seed_memories(storage, count_new=3, count_old=0)
    report = run_reflection(cfg, apply_consolidate=False)
    assert report.memory_id == _today_slug()
    written = storage.get(report.memory_id)
    assert written is not None
    assert "Reflexión" in written.body
    assert report.created_in_window >= 3


def test_run_reflection_counts_zombies(cfg: Config) -> None:
    storage = VaultStorage(cfg.memory_dir)
    _seed_memories(storage, count_new=0, count_old=2)
    # Zombies: 0 usage_count + updated 100 d ago, threshold 60 d.
    report = run_reflection(
        cfg, apply_consolidate=False, zombie_age_days=60.0
    )
    assert len(report.zombies) >= 2


def test_run_reflection_idempotent_same_day(cfg: Config) -> None:
    storage = VaultStorage(cfg.memory_dir)
    _seed_memories(storage, count_new=2, count_old=0)

    first = run_reflection(cfg, apply_consolidate=False)
    # Snapshot the body; a second run should update-in-place, not create
    # a second reflection_YYYY_MM_DD sibling.
    count_before = sum(1 for _ in cfg.memory_dir.glob("reflection_*.md"))

    # Seed one more memory and run again.
    storage.save(content="extra", title="extra", type="note")
    second = run_reflection(cfg, apply_consolidate=False)

    count_after = sum(1 for _ in cfg.memory_dir.glob("reflection_*.md"))
    assert first.memory_id == second.memory_id
    assert count_before == count_after  # idempotent


def test_run_reflection_detects_contradictions(cfg: Config) -> None:
    storage = VaultStorage(cfg.memory_dir)
    m_a = storage.save(content="a", title="a")
    m_b = storage.save(content="b", title="b")
    storage.update(m_a.id, contradicts=[m_b.id])

    report = run_reflection(cfg, apply_consolidate=False)
    ids_with_contra = {c["id"] for c in report.contradictions}
    assert m_a.id in ids_with_contra


def test_report_body_rendering() -> None:
    report = ReflectionReport(
        day="2026-05-01",
        lookback_hours=24,
        total_memorias=42,
        created_in_window=3,
        updated_in_window=1,
        consolidated_pairs=0,
        pending_dup_pairs=5,
        new_decisions=[{"id": "d1", "name": "Decisión 1", "description": "x"}],
        zombies=[{"id": "z1", "description": "unused", "age_days": 120}],
        contradictions=[{"id": "c1", "contradicts": ["c2"]}],
        knowledge_gaps=["`project:foo` — sin writes hace 30 días"],
    )
    body = _format_body(report)
    assert "Reflexión del día — 2026-05-01" in body
    assert "[[d1]]" in body
    assert "[[c1]]" in body
    assert "[[z1]]" in body
    assert "Knowledge gaps" in body
    assert "project:foo" in body

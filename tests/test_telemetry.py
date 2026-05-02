"""Unit tests for ``mem_vault.telemetry``.

The telemetry module is the foundation for the closed-loop ranker — every
``memory_search`` call writes one row per surfaced hit, the Stop hook flips
``was_cited=1`` when the agent's response references a memory id, and the
ranker fits on the resulting (features, label) pairs.

These tests cover:

- Schema creation is idempotent.
- ``record_search`` persists every event in the batch.
- ``mark_cited`` flips the *most recent* un-cited row per memory id, scoped
  by session_id when given.
- ``fetch_training_rows`` returns rows in the expected shape, with optional
  time + limit filters.
- ``stats`` reflects counts, citation rate, and avg cited rank correctly.
- Best-effort: every helper swallows DB errors and returns a safe default.
"""

from __future__ import annotations

import time
from pathlib import Path

from mem_vault import telemetry


def _mem(mid: str, **fm: object) -> dict[str, object]:
    """Tiny helper to build a memory dict shaped like ``Memory.to_dict()``."""
    base = {"id": mid, "tags": [], "agent_id": None, "updated": None}
    base.update(fm)
    return base


def test_ensure_schema_is_idempotent(tmp_path: Path) -> None:
    telemetry.ensure_schema(tmp_path)
    telemetry.ensure_schema(tmp_path)  # second call must not raise
    assert telemetry.db_path(tmp_path).exists()


def test_record_search_persists_every_event(tmp_path: Path) -> None:
    events = [
        telemetry.build_event(
            query="rate limits API auth",
            rank=i,
            memory=_mem(f"id_{i}", helpful_ratio=0.5, usage_count=2, updated="2026-04-30T12:00:00+00:00"),
            score_dense=0.9 - i * 0.1,
            score_bm25=None,
            score_rerank=None,
            score_final=0.9 - i * 0.1,
            usage_boost=1.15,
            user_id="default",
            agent_id="devin",
            project="mem-vault",
        )
        for i in range(3)
    ]
    inserted = telemetry.record_search(tmp_path, events)
    assert inserted == 3
    rows = telemetry.fetch_training_rows(tmp_path)
    assert len(rows) == 3
    assert {r["memory_id"] for r in rows} == {"id_0", "id_1", "id_2"}
    assert all(r["was_cited"] == 0 for r in rows)


def test_mark_cited_flips_most_recent_uncited_row_per_memory(tmp_path: Path) -> None:
    # Two different searches surface "id_a" — the second one is what the
    # agent actually saw, so the citation should land there.
    older = telemetry.build_event(
        query="q1",
        rank=0,
        memory=_mem("id_a"),
        score_dense=0.5,
        score_bm25=None,
        score_rerank=None,
        score_final=0.5,
        usage_boost=1.0,
        user_id=None,
        agent_id=None,
        project=None,
    )
    older.ts = time.time() - 3600  # 1h ago
    newer = telemetry.build_event(
        query="q2",
        rank=2,
        memory=_mem("id_a"),
        score_dense=0.7,
        score_bm25=None,
        score_rerank=None,
        score_final=0.7,
        usage_boost=1.0,
        user_id=None,
        agent_id=None,
        project=None,
    )
    telemetry.record_search(tmp_path, [older, newer])

    flipped = telemetry.mark_cited(tmp_path, ["id_a"])
    assert flipped == 1

    rows = sorted(
        telemetry.fetch_training_rows(tmp_path), key=lambda r: r["ts"]
    )
    # Older row stays un-cited, newer row gets the citation.
    assert rows[0]["was_cited"] == 0
    assert rows[1]["was_cited"] == 1
    assert rows[1]["cited_at"] is not None


def test_mark_cited_respects_session_scope(tmp_path: Path) -> None:
    e_a = telemetry.build_event(
        query="q",
        rank=0,
        memory=_mem("id_x"),
        score_dense=0.6,
        score_bm25=None,
        score_rerank=None,
        score_final=0.6,
        usage_boost=1.0,
        user_id=None,
        agent_id=None,
        project=None,
        session_id="session-A",
    )
    e_b = telemetry.build_event(
        query="q",
        rank=0,
        memory=_mem("id_x"),
        score_dense=0.6,
        score_bm25=None,
        score_rerank=None,
        score_final=0.6,
        usage_boost=1.0,
        user_id=None,
        agent_id=None,
        project=None,
        session_id="session-B",
    )
    telemetry.record_search(tmp_path, [e_a, e_b])

    # Cite within session A — only the row tagged session-A flips.
    flipped = telemetry.mark_cited(tmp_path, ["id_x"], session_id="session-A")
    assert flipped == 1
    rows = telemetry.fetch_training_rows(tmp_path)
    by_session = {r["session_id"]: r["was_cited"] for r in rows}
    assert by_session["session-A"] == 1
    assert by_session["session-B"] == 0


def test_mark_cited_idempotent(tmp_path: Path) -> None:
    e = telemetry.build_event(
        query="q",
        rank=0,
        memory=_mem("id_y"),
        score_dense=0.5,
        score_bm25=None,
        score_rerank=None,
        score_final=0.5,
        usage_boost=1.0,
        user_id=None,
        agent_id=None,
        project=None,
    )
    telemetry.record_search(tmp_path, [e])
    assert telemetry.mark_cited(tmp_path, ["id_y"]) == 1
    # Second call: there's no un-cited row left → 0 flipped.
    assert telemetry.mark_cited(tmp_path, ["id_y"]) == 0


def test_build_event_computes_recency_days(tmp_path: Path) -> None:
    # 3 days ago in UTC
    from datetime import datetime, timedelta, timezone

    three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
    e = telemetry.build_event(
        query="q",
        rank=0,
        memory=_mem("id", updated=three_days_ago.isoformat()),
        score_dense=0.5,
        score_bm25=None,
        score_rerank=None,
        score_final=0.5,
        usage_boost=1.0,
        user_id=None,
        agent_id=None,
        project=None,
    )
    assert e.recency_days is not None
    assert 2.5 <= e.recency_days <= 3.5


def test_build_event_project_match_via_tag_or_payload() -> None:
    e_tag = telemetry.build_event(
        query="q",
        rank=0,
        memory=_mem("id", tags=["project:mem-vault"]),
        score_dense=0.5,
        score_bm25=None,
        score_rerank=None,
        score_final=0.5,
        usage_boost=1.0,
        user_id=None,
        agent_id=None,
        project="mem-vault",
    )
    assert e_tag.project_match == 1

    e_no_match = telemetry.build_event(
        query="q",
        rank=0,
        memory=_mem("id", tags=["project:other"]),
        score_dense=0.5,
        score_bm25=None,
        score_rerank=None,
        score_final=0.5,
        usage_boost=1.0,
        user_id=None,
        agent_id=None,
        project="mem-vault",
    )
    assert e_no_match.project_match == 0


def test_stats_reports_citation_rate(tmp_path: Path) -> None:
    events = [
        telemetry.build_event(
            query="q",
            rank=i,
            memory=_mem(f"m{i}"),
            score_dense=0.5,
            score_bm25=None,
            score_rerank=None,
            score_final=0.5,
            usage_boost=1.0,
            user_id=None,
            agent_id=None,
            project=None,
        )
        for i in range(5)
    ]
    telemetry.record_search(tmp_path, events)
    telemetry.mark_cited(tmp_path, ["m1", "m3"])

    s = telemetry.stats(tmp_path)
    assert s["total_events"] == 5
    assert s["citations"] == 2
    assert abs(s["citation_rate"] - 0.4) < 1e-9
    # The cited rows were rank 1 and rank 3 → avg = 2.0
    assert s["avg_cited_rank"] is not None
    assert abs(s["avg_cited_rank"] - 2.0) < 1e-9


def test_record_search_swallows_errors(monkeypatch, tmp_path: Path) -> None:
    """A broken DB path must NOT raise; record_search returns 0."""
    bogus = tmp_path / "does" / "not" / "exist.db"  # parent will be created — switch to a path we make read-only

    # Make the parent directory non-writable so the schema bootstrap fails.
    parent = tmp_path / "ro"
    parent.mkdir()
    parent.chmod(0o555)
    try:
        e = telemetry.build_event(
            query="q",
            rank=0,
            memory=_mem("id"),
            score_dense=0.5,
            score_bm25=None,
            score_rerank=None,
            score_final=0.5,
            usage_boost=1.0,
            user_id=None,
            agent_id=None,
            project=None,
        )
        # Use the read-only directory as state_dir; the DB path will be
        # ``ro/search_events.db`` — opening / writing should fail; we
        # expect 0 inserted rows, no exception.
        result = telemetry.record_search(parent, [e])
        assert result == 0
    finally:
        parent.chmod(0o755)


def test_fetch_training_rows_filters_and_limits(tmp_path: Path) -> None:
    now = time.time()
    events = []
    for i in range(5):
        e = telemetry.build_event(
            query="q",
            rank=i,
            memory=_mem(f"m{i}"),
            score_dense=0.5,
            score_bm25=None,
            score_rerank=None,
            score_final=0.5,
            usage_boost=1.0,
            user_id=None,
            agent_id=None,
            project=None,
        )
        e.ts = now - (5 - i) * 100
        events.append(e)
    telemetry.record_search(tmp_path, events)

    # ts schedule: i=0 → now-500, i=1 → now-400, i=2 → now-300,
    # i=3 → now-200, i=4 → now-100. Cutoff at now-350 keeps i=2,3,4.
    recent = telemetry.fetch_training_rows(tmp_path, since_ts=now - 350)
    assert len(recent) == 3
    assert {r["memory_id"] for r in recent} == {"m2", "m3", "m4"}

    # Limit caps the result set
    limited = telemetry.fetch_training_rows(tmp_path, limit=2)
    assert len(limited) == 2

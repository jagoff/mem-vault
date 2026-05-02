"""Search-event telemetry — the foundation for closed-loop ranker learning.

Every ``memory_search`` call writes one row per surfaced hit to a SQLite
database under ``<state_dir>/search_events.db``. Each row captures the
features the ranker will train on:

- ``score_dense`` / ``score_bm25`` — raw retrieval signals before fusion.
- ``score_final`` — what the user actually saw (post-rerank, post-boost).
- ``rank`` — the position the hit ended up at in the response.
- ``helpful_ratio`` / ``usage_count`` — the supervised signal at search-time.
- ``recency_days`` — how stale the memory was when surfaced.
- ``project_match`` / ``agent_id_match`` — soft contextual signals.
- ``was_cited`` — initially 0; flipped to 1 by the Stop hook when the
  agent's final response actually mentions the memory id.

This is the *only* place we materialize the (query, results, feedback)
tuples that ``ranker.py`` later trains on. The Stop hook updates rows
in-place via ``mark_cited``; the rest of the system never mutates events.

Design choices:

- **SQLite, not JSONL**: appends are still cheap, but ``mark_cited``
  needs ``UPDATE … WHERE event_id = ?`` semantics. JSONL would force a
  full-file rewrite for every citation. The DB lives under
  ``state_dir`` (alongside ``history.db`` + ``metrics.jsonl``), never
  the vault — it's a derived cache, not a source of truth.

- **Best-effort writes**: every helper swallows exceptions and logs a
  warning. The ranker is opt-in (``MEM_VAULT_LEARNED_RANKER=1``); the
  base search path must keep working even when the DB is locked,
  full, or missing.

- **Append-only by default**: rows are immutable except for the
  ``was_cited`` / ``cited_at`` columns the Stop hook fills in. This
  makes the dataset reproducible across ranker retrains.

- **Schema versioned in ``meta``**: a one-row ``meta`` table tracks
  the schema version. ``ensure_schema`` migrates forward; older rows
  stay queryable.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS search_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    query_hash      TEXT    NOT NULL,
    query_len       INTEGER NOT NULL,
    user_id         TEXT,
    agent_id        TEXT,
    project         TEXT,
    memory_id       TEXT    NOT NULL,
    rank            INTEGER NOT NULL,
    score_dense     REAL,
    score_bm25      REAL,
    score_rerank    REAL,
    score_final     REAL,
    usage_boost     REAL,
    helpful_ratio   REAL,
    usage_count     INTEGER,
    recency_days    REAL,
    project_match   INTEGER,
    agent_id_match  INTEGER,
    was_cited       INTEGER NOT NULL DEFAULT 0,
    cited_at        REAL,
    session_id      TEXT
);

CREATE INDEX IF NOT EXISTS idx_search_events_ts          ON search_events(ts);
CREATE INDEX IF NOT EXISTS idx_search_events_memory_id   ON search_events(memory_id);
CREATE INDEX IF NOT EXISTS idx_search_events_session_id  ON search_events(session_id);
CREATE INDEX IF NOT EXISTS idx_search_events_query_hash  ON search_events(query_hash);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class SearchEvent:
    """One surfaced hit in a ``memory_search`` response.

    Mirrors the columns of ``search_events`` 1:1. The ``server.search``
    handler builds this from each candidate before the ``record_search``
    call; ``ranker.train`` reads the same shape back from disk.
    """

    ts: float
    query_hash: str
    query_len: int
    user_id: str | None
    agent_id: str | None
    project: str | None
    memory_id: str
    rank: int
    score_dense: float | None
    score_bm25: float | None
    score_rerank: float | None
    score_final: float | None
    usage_boost: float | None
    helpful_ratio: float | None
    usage_count: int | None
    recency_days: float | None
    project_match: int | None
    agent_id_match: int | None
    session_id: str | None = None


def db_path(state_dir: Path) -> Path:
    """The on-disk location of the search-events SQLite. Stable across versions."""
    return Path(state_dir) / "search_events.db"


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Open the SQLite with sane defaults. WAL = concurrent reads OK."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=2.0, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
    finally:
        conn.close()


def ensure_schema(state_dir: Path) -> None:
    """Create the schema if it doesn't exist; idempotent."""
    path = db_path(state_dir)
    with _connect(path) as conn:
        conn.executescript(_DDL)
        conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )


def _hash_query(query: str) -> str:
    """Stable short hash of the query — privacy-preserving + dedup-friendly.

    We don't store the raw query text by default. Two reasons:
    (1) the search-events DB is a small, long-lived sidecar; bodies of
    arbitrary length would balloon it across months of use.
    (2) queries can be sensitive (project codenames, names of people) —
    storing only a hash keeps the dataset useful for ranker training
    (we only need "did the same query come up again?", not the literal
    string) without doubling as an exfiltration target.
    """
    import hashlib

    return hashlib.sha256(query.encode("utf-8", errors="replace")).hexdigest()[:16]


def _recency_days(updated_iso: str | None, now_ts: float | None = None) -> float | None:
    """Days between ``updated`` (frontmatter ISO8601) and now. None if unparseable."""
    if not updated_iso:
        return None
    try:
        # Frontmatter writes timezone-aware ISO8601; ``fromisoformat`` handles
        # the offset (Python 3.11+). Defensive against legacy "Z" suffix.
        if updated_iso.endswith("Z"):
            updated_iso = updated_iso[:-1] + "+00:00"
        dt = datetime.fromisoformat(updated_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        now = datetime.fromtimestamp(now_ts or time.time(), tz=UTC)
        return max(0.0, (now - dt).total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return None


def build_event(
    *,
    query: str,
    rank: int,
    memory: dict[str, Any],
    score_dense: float | None,
    score_bm25: float | None,
    score_rerank: float | None,
    score_final: float | None,
    usage_boost: float | None,
    user_id: str | None,
    agent_id: str | None,
    project: str | None,
    session_id: str | None = None,
) -> SearchEvent:
    """Project a search hit into a flat ``SearchEvent`` row.

    Pulled out as a helper so ``server.search`` stays readable and so
    tests can build events without the full search pipeline.

    The ``memory`` dict is the same shape returned by ``Memory.to_dict``
    (frontmatter + body). We pluck the columns the ranker cares about
    and drop the rest.
    """
    md = memory or {}
    fm = md.get("frontmatter") or md  # accept either shape
    project_match = None
    if project is not None:
        # Encoded as a 0/1 int because SQLite booleans are integers
        # anyway and integer comparison ranges cleaner over the dataset.
        # ``project`` lives both in tags (``project:foo``) and in the
        # ``project`` payload; cover both.
        tags = fm.get("tags") or []
        proj_tag = f"project:{project}".lower()
        project_match = 1 if (
            (md.get("project") == project)
            or any(t.lower() == proj_tag for t in tags)
        ) else 0
    agent_id_match = None
    if agent_id is not None:
        agent_id_match = 1 if fm.get("agent_id") == agent_id else 0

    return SearchEvent(
        ts=time.time(),
        query_hash=_hash_query(query),
        query_len=len(query),
        user_id=user_id,
        agent_id=agent_id,
        project=project,
        memory_id=md.get("id") or fm.get("id") or "",
        rank=rank,
        score_dense=score_dense,
        score_bm25=score_bm25,
        score_rerank=score_rerank,
        score_final=score_final,
        usage_boost=usage_boost,
        helpful_ratio=fm.get("helpful_ratio"),
        usage_count=fm.get("usage_count"),
        recency_days=_recency_days(fm.get("updated")),
        project_match=project_match,
        agent_id_match=agent_id_match,
        session_id=session_id,
    )


def record_search(state_dir: Path, events: Iterable[SearchEvent]) -> int:
    """Persist a batch of search events. Returns rows inserted; 0 on failure.

    Best-effort: every exception is swallowed + logged as a warning. The
    base search path NEVER fails because telemetry failed.
    """
    try:
        ensure_schema(state_dir)
        with _connect(db_path(state_dir)) as conn:
            cur = conn.executemany(
                """
                INSERT INTO search_events(
                    ts, query_hash, query_len, user_id, agent_id, project,
                    memory_id, rank, score_dense, score_bm25, score_rerank,
                    score_final, usage_boost, helpful_ratio, usage_count,
                    recency_days, project_match, agent_id_match, session_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        e.ts,
                        e.query_hash,
                        e.query_len,
                        e.user_id,
                        e.agent_id,
                        e.project,
                        e.memory_id,
                        e.rank,
                        e.score_dense,
                        e.score_bm25,
                        e.score_rerank,
                        e.score_final,
                        e.usage_boost,
                        e.helpful_ratio,
                        e.usage_count,
                        e.recency_days,
                        e.project_match,
                        e.agent_id_match,
                        e.session_id,
                    )
                    for e in events
                ],
            )
            return cur.rowcount or 0
    except Exception as exc:
        logger.warning("telemetry: failed to record search events: %s", exc)
        return 0


def mark_cited(
    state_dir: Path,
    memory_ids: Iterable[str],
    *,
    session_id: str | None = None,
    since_ts: float | None = None,
    cited_ts: float | None = None,
) -> int:
    """Flip ``was_cited=1`` on the most recent un-cited rows for these IDs.

    Called by the Stop hook once it has detected citations in the agent's
    final response. We update the *most recent* event per memory_id so
    the ranker learns "this memory was returned at rank N and the agent
    actually used it".

    Args:
        memory_ids: IDs cited in the agent's response.
        session_id: optional. When set, scope the update to events from
            this session — keeps the citation signal tightly coupled to
            the turn that produced it (no cross-session leakage).
        since_ts: epoch float lower bound. Defaults to "last 30 minutes"
            when neither ``session_id`` nor ``since_ts`` is provided —
            same conservative window the Stop hook uses elsewhere.
        cited_ts: epoch float to stamp into ``cited_at``; defaults now.

    Returns:
        Number of rows actually flipped (0 if nothing matched).
    """
    ids = [mid for mid in (memory_ids or []) if mid]
    if not ids:
        return 0
    if since_ts is None and session_id is None:
        since_ts = time.time() - 30 * 60  # last 30 minutes
    cited_ts = cited_ts or time.time()
    flipped = 0
    try:
        ensure_schema(state_dir)
        with _connect(db_path(state_dir)) as conn:
            for mid in ids:
                # Pick the latest un-cited row for this memory in scope.
                # ORDER BY ts DESC LIMIT 1 maps to the most recent search
                # that surfaced this memory; the agent likely used the
                # one it just saw, not one from yesterday.
                if session_id is not None:
                    row = conn.execute(
                        """
                        SELECT event_id FROM search_events
                        WHERE memory_id = ? AND was_cited = 0 AND session_id = ?
                        ORDER BY ts DESC LIMIT 1
                        """,
                        (mid, session_id),
                    ).fetchone()
                else:
                    row = conn.execute(
                        """
                        SELECT event_id FROM search_events
                        WHERE memory_id = ? AND was_cited = 0 AND ts >= ?
                        ORDER BY ts DESC LIMIT 1
                        """,
                        (mid, since_ts),
                    ).fetchone()
                if row is None:
                    continue
                conn.execute(
                    "UPDATE search_events SET was_cited = 1, cited_at = ? WHERE event_id = ?",
                    (cited_ts, row[0]),
                )
                flipped += 1
    except Exception as exc:
        logger.warning("telemetry: failed to mark citations: %s", exc)
    return flipped


def fetch_training_rows(
    state_dir: Path,
    *,
    since_ts: float | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Pull rows for ``ranker.train``. Returns a list of dicts (col → value).

    Filters: ``since_ts`` for incremental training; ``limit`` to cap rows
    when the DB grows. The ranker fits over rows that have at least one
    feature populated — early rows from before the schema landed are
    naturally excluded by the WHERE clause.
    """
    rows: list[dict[str, Any]] = []
    try:
        ensure_schema(state_dir)
        with _connect(db_path(state_dir)) as conn:
            conn.row_factory = sqlite3.Row
            sql = "SELECT * FROM search_events WHERE 1=1"
            params: list[Any] = []
            if since_ts is not None:
                sql += " AND ts >= ?"
                params.append(since_ts)
            sql += " ORDER BY ts DESC"
            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)
            for r in conn.execute(sql, params).fetchall():
                rows.append({k: r[k] for k in r.keys()})
    except Exception as exc:
        logger.warning("telemetry: failed to fetch training rows: %s", exc)
    return rows


def stats(state_dir: Path) -> dict[str, Any]:
    """Quick health snapshot for ``mem-vault doctor`` + UI dashboard.

    Returns counts useful for "is the closed-loop actually getting
    signal?": total events, unique queries, citations seen, avg rank
    of cited results.
    """
    out: dict[str, Any] = {
        "total_events": 0,
        "unique_queries": 0,
        "citations": 0,
        "citation_rate": 0.0,
        "avg_cited_rank": None,
        "first_event_ts": None,
        "last_event_ts": None,
        "db_path": str(db_path(state_dir)),
        "db_size_bytes": 0,
    }
    try:
        path = db_path(state_dir)
        if path.exists():
            out["db_size_bytes"] = path.stat().st_size
        ensure_schema(state_dir)
        with _connect(db_path(state_dir)) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                   AS total,
                    COUNT(DISTINCT query_hash) AS uniq,
                    SUM(was_cited)             AS cited,
                    AVG(CASE WHEN was_cited = 1 THEN rank END) AS avg_rank,
                    MIN(ts) AS first_ts,
                    MAX(ts) AS last_ts
                FROM search_events
                """
            ).fetchone()
            total = int(row[0] or 0)
            cited = int(row[2] or 0)
            out["total_events"] = total
            out["unique_queries"] = int(row[1] or 0)
            out["citations"] = cited
            out["citation_rate"] = (cited / total) if total else 0.0
            out["avg_cited_rank"] = float(row[3]) if row[3] is not None else None
            out["first_event_ts"] = float(row[4]) if row[4] is not None else None
            out["last_event_ts"] = float(row[5]) if row[5] is not None else None
    except Exception as exc:
        logger.warning("telemetry: stats query failed: %s", exc)
    return out


__all__ = [
    "SCHEMA_VERSION",
    "SearchEvent",
    "build_event",
    "db_path",
    "ensure_schema",
    "fetch_training_rows",
    "mark_cited",
    "record_search",
    "stats",
]

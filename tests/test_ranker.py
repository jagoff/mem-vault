"""Unit tests for ``mem_vault.ranker`` — the closed-loop adaptive ranker.

Coverage:

- ``featurize`` is robust to missing columns + uses the documented order.
- ``train`` refuses to fit below ``min_rows`` / ``min_positives`` and writes
  a versioned pickle + an ``active.pkl`` symlink-equivalent on success.
- ``load_active`` round-trips a fitted model that scores in [0, 1] and
  returns higher probability for the kind of row the model was trained on.
- ``rollback`` swaps the active pointer to the previous version.
- ``is_enabled`` reads the env var.

Tests use synthetic data so they run without Ollama / Qdrant. The
sklearn dependency is hard for this file (we test the actual training
loop), so each test imports it inside the function and skips on
ImportError. CI installs ``[learning]`` so the skip is only relevant
to a developer working without the extra.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mem_vault import ranker, telemetry


def _seed_events(state_dir: Path, n_pos: int = 30, n_neg: int = 70) -> None:
    """Populate the search-events DB with separable positive/negative rows.

    Positives have high score_dense + high helpful_ratio + project_match=1;
    negatives have low score_dense + low helpful_ratio + project_match=0.
    A logreg should fit AUC ≥ 0.85 on this kind of dataset.
    """
    events: list[telemetry.SearchEvent] = []
    for i in range(n_pos):
        e = telemetry.SearchEvent(
            ts=1000.0 + i,
            query_hash=f"qh_{i:04d}",
            query_len=20,
            user_id="default",
            agent_id="devin",
            project="mem-vault",
            memory_id=f"mem_pos_{i}",
            rank=i % 5,
            score_dense=0.8 + 0.01 * (i % 5),
            score_bm25=None,
            score_rerank=None,
            score_final=0.85 + 0.01 * (i % 5),
            usage_boost=1.2,
            helpful_ratio=0.9,
            usage_count=5 + i,
            recency_days=7.0,
            project_match=1,
            agent_id_match=1,
        )
        events.append(e)
    for i in range(n_neg):
        e = telemetry.SearchEvent(
            ts=2000.0 + i,
            query_hash=f"qh_neg_{i:04d}",
            query_len=20,
            user_id="default",
            agent_id="devin",
            project="other",
            memory_id=f"mem_neg_{i}",
            rank=i % 5,
            score_dense=0.3 - 0.01 * (i % 5),
            score_bm25=None,
            score_rerank=None,
            score_final=0.32 - 0.01 * (i % 5),
            usage_boost=1.0,
            helpful_ratio=0.0,
            usage_count=0,
            recency_days=120.0,
            project_match=0,
            agent_id_match=0,
        )
        events.append(e)
    telemetry.record_search(state_dir, events)
    # Mark the positive ones as cited (the supervised label).
    telemetry.mark_cited(
        state_dir,
        [f"mem_pos_{i}" for i in range(n_pos)],
        since_ts=0.0,
    )


def test_featurize_uses_documented_column_order() -> None:
    row = {
        "score_dense": 0.7,
        "score_final": 0.75,
        "rank": 2,
        "helpful_ratio": 0.5,
        "usage_count": 4,
        "recency_days": 10.0,
        "project_match": 1,
        "agent_id_match": 0,
    }
    feats = ranker.featurize(row)
    assert len(feats) == len(ranker.FEATURE_COLUMNS)
    # Spot-check the values landed in the right slot.
    assert feats[0] == 0.7   # score_dense
    assert feats[2] == 2.0   # rank
    assert feats[3] == 0.5   # helpful_ratio
    assert feats[6] == 1.0   # project_match


def test_featurize_handles_missing_values() -> None:
    feats = ranker.featurize({})
    assert len(feats) == len(ranker.FEATURE_COLUMNS)
    # recency_days defaults to 365 (older = no boost), every other column = 0.
    recency_idx = ranker.FEATURE_COLUMNS.index("recency_days")
    assert feats[recency_idx] == 365.0
    for idx, _name in enumerate(ranker.FEATURE_COLUMNS):
        if idx == recency_idx:
            continue
        assert feats[idx] == 0.0


def test_featurize_caps_recency_at_365() -> None:
    feats = ranker.featurize({"recency_days": 9999.0})
    recency_idx = ranker.FEATURE_COLUMNS.index("recency_days")
    assert feats[recency_idx] == 365.0


def test_featurize_log1p_usage_count() -> None:
    import math

    row = {"usage_count": 99}
    feats = ranker.featurize(row)
    usage_idx = ranker.FEATURE_COLUMNS.index("usage_count_log")
    assert feats[usage_idx] == pytest.approx(math.log1p(99))


def test_train_rejects_insufficient_data(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    # 5 events, 1 positive — way below thresholds.
    _seed_events(tmp_path, n_pos=1, n_neg=4)
    with pytest.raises(ValueError, match=r"not enough"):
        ranker.train(tmp_path)


def test_train_rejects_few_positives(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    _seed_events(tmp_path, n_pos=2, n_neg=60)
    with pytest.raises(ValueError, match=r"not enough positive"):
        ranker.train(tmp_path)


def test_train_writes_versioned_and_active_pickles(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    _seed_events(tmp_path, n_pos=30, n_neg=70)
    result = ranker.train(tmp_path)

    assert result.version == 1
    assert result.n_train == 100
    assert result.n_positive == 30
    assert result.n_negative == 70
    assert Path(result.pickle_path).exists()
    assert ranker.active_pickle_path(tmp_path).exists()
    # Meta JSON sidecar:
    meta = ranker.meta_path(tmp_path)
    assert meta.exists()
    import json
    payload = json.loads(meta.read_text())
    assert payload["version"] == 1
    assert payload["n_train"] == 100


def test_train_separable_data_hits_high_auc(tmp_path: Path) -> None:
    """Sanity check: a perfectly separable synthetic dataset should fit AUC ≥ 0.95."""
    pytest.importorskip("sklearn")
    _seed_events(tmp_path, n_pos=40, n_neg=80)
    result = ranker.train(tmp_path)
    assert result.auc is not None
    assert result.auc >= 0.95


def test_load_active_round_trip_predicts_in_unit_interval(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    _seed_events(tmp_path, n_pos=30, n_neg=70)
    ranker.train(tmp_path)
    loaded = ranker.load_active(tmp_path)
    assert loaded is not None
    score = loaded.score(
        {
            "score_dense": 0.85,
            "score_final": 0.88,
            "rank": 0,
            "helpful_ratio": 0.9,
            "usage_count": 7,
            "recency_days": 5.0,
            "project_match": 1,
            "agent_id_match": 1,
        }
    )
    assert 0.0 <= score <= 1.0


def test_load_active_returns_none_when_no_pickle(tmp_path: Path) -> None:
    assert ranker.load_active(tmp_path) is None


def test_load_active_returns_none_on_corrupt_pickle(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    rd = ranker.ranker_dir(tmp_path)
    rd.mkdir(parents=True, exist_ok=True)
    ranker.active_pickle_path(tmp_path).write_bytes(b"not-a-real-pickle")
    assert ranker.load_active(tmp_path) is None


def test_score_higher_for_positive_shaped_row(tmp_path: Path) -> None:
    """The fitted model should put positive-shaped rows above negative-shaped ones."""
    pytest.importorskip("sklearn")
    _seed_events(tmp_path, n_pos=40, n_neg=80)
    ranker.train(tmp_path)
    loaded = ranker.load_active(tmp_path)
    assert loaded is not None
    pos_row = {
        "score_dense": 0.82,
        "score_final": 0.86,
        "rank": 0,
        "helpful_ratio": 0.9,
        "usage_count": 6,
        "recency_days": 5.0,
        "project_match": 1,
        "agent_id_match": 1,
    }
    neg_row = {
        "score_dense": 0.28,
        "score_final": 0.30,
        "rank": 4,
        "helpful_ratio": 0.0,
        "usage_count": 0,
        "recency_days": 200.0,
        "project_match": 0,
        "agent_id_match": 0,
    }
    assert loaded.score(pos_row) > loaded.score(neg_row)


def test_rollback_restores_previous_version(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    _seed_events(tmp_path, n_pos=30, n_neg=70)
    ranker.train(tmp_path)  # v1
    # Add more events, retrain → v2 active.
    _seed_events(tmp_path, n_pos=10, n_neg=10)
    r2 = ranker.train(tmp_path)
    assert r2.version == 2

    target = ranker.rollback(tmp_path)
    assert target == 1
    # The active pickle should now match the v1 file's contents.
    v1_bytes = (ranker.ranker_dir(tmp_path) / "ranker_v1.pkl").read_bytes()
    active_bytes = ranker.active_pickle_path(tmp_path).read_bytes()
    assert v1_bytes == active_bytes


def test_rollback_no_op_when_only_one_version(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    _seed_events(tmp_path, n_pos=30, n_neg=70)
    ranker.train(tmp_path)
    assert ranker.rollback(tmp_path) is None


def test_is_enabled_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEM_VAULT_LEARNED_RANKER", raising=False)
    assert ranker.is_enabled() is False
    monkeypatch.setenv("MEM_VAULT_LEARNED_RANKER", "1")
    assert ranker.is_enabled() is True
    monkeypatch.setenv("MEM_VAULT_LEARNED_RANKER", "off")
    assert ranker.is_enabled() is False

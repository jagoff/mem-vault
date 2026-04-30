"""Unit tests for helper functions in :mod:`mem_vault.index`.

We exercise ``time_decay_factor`` in isolation (no Ollama / Qdrant). The
``VectorIndex`` integration with mem0 is covered in ``test_robustness.py``
and ``test_breaker.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from mem_vault.index import time_decay_factor

# ---------------------------------------------------------------------------
# Disabled / no-op cases
# ---------------------------------------------------------------------------


def test_decay_disabled_returns_one():
    assert time_decay_factor("2026-04-29T10:00:00+00:00", half_life_days=0) == 1.0
    assert time_decay_factor("2026-04-29T10:00:00+00:00", half_life_days=-1) == 1.0


def test_decay_missing_timestamp_returns_one():
    assert time_decay_factor(None, half_life_days=30) == 1.0
    assert time_decay_factor("", half_life_days=30) == 1.0


def test_decay_unparseable_timestamp_returns_one():
    assert time_decay_factor("not-an-iso-string", half_life_days=30) == 1.0
    assert time_decay_factor("2026-13-99T99:99:99", half_life_days=30) == 1.0


# ---------------------------------------------------------------------------
# Decay math
# ---------------------------------------------------------------------------


def test_decay_factor_just_now_is_close_to_one():
    """A memory updated just now should hardly be decayed."""
    now_iso = datetime.now(tz=UTC).isoformat()
    assert time_decay_factor(now_iso, half_life_days=30) > 0.99


def test_decay_factor_at_half_life_equals_half():
    """A memory exactly half_life days old should score ~0.5×."""
    half_life = 90.0
    iso = (datetime.now(tz=UTC) - timedelta(days=half_life)).isoformat()
    factor = time_decay_factor(iso, half_life_days=half_life)
    assert 0.49 < factor < 0.51


def test_decay_factor_at_two_half_lives_equals_quarter():
    """Two half-lives → 0.25× (cumulative halving)."""
    half_life = 30.0
    iso = (datetime.now(tz=UTC) - timedelta(days=half_life * 2)).isoformat()
    factor = time_decay_factor(iso, half_life_days=half_life)
    assert 0.24 < factor < 0.26


def test_decay_factor_naive_timestamp_treated_as_utc():
    """Bare ISO without ``+HH:MM`` should be assumed UTC, not crash."""
    naive = (datetime.now(tz=UTC) - timedelta(days=30)).replace(tzinfo=None).isoformat()
    factor = time_decay_factor(naive, half_life_days=30)
    # Approximately 0.5 (within tolerance for the test setup latency).
    assert 0.45 < factor < 0.55


def test_decay_factor_future_timestamp_returns_one():
    """A timestamp in the future shouldn't *boost* the score above 1.0."""
    future = (datetime.now(tz=UTC) + timedelta(days=10)).isoformat()
    assert time_decay_factor(future, half_life_days=30) == 1.0


def test_decay_factor_aggressive_vs_mild_half_life():
    """Same age, smaller half-life → stronger decay."""
    iso = (datetime.now(tz=UTC) - timedelta(days=30)).isoformat()
    aggressive = time_decay_factor(iso, half_life_days=30)  # ~0.5
    mild = time_decay_factor(iso, half_life_days=365)  # ~0.95
    assert aggressive < mild

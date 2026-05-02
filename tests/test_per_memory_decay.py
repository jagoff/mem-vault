"""Per-memory effective decay (v0.6.0).

The legacy ``time_decay_factor`` only looked at ``updated``. As of v0.6.0
it also accepts ``last_used_iso`` and uses the *most recent* of the two
timestamps. This means the global ``decay_half_life_days`` becomes a
per-memory effective half-life: a memory cited yesterday looks fresh
even if it was authored months ago, while one that's never been cited
decays at the full rate.

These tests pin the contract.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mem_vault.index import time_decay_factor


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def test_decay_disabled_returns_one_regardless_of_inputs() -> None:
    assert time_decay_factor(_iso(900), half_life_days=0) == 1.0
    assert time_decay_factor(_iso(900), half_life_days=0, last_used_iso=_iso(0)) == 1.0


def test_decay_falls_back_to_one_when_both_timestamps_missing() -> None:
    assert time_decay_factor(None, half_life_days=30) == 1.0
    assert time_decay_factor(None, half_life_days=30, last_used_iso=None) == 1.0
    assert time_decay_factor("not-a-date", half_life_days=30, last_used_iso="also-bad") == 1.0


def test_legacy_path_still_uses_updated_when_no_last_used() -> None:
    # 90-day half-life × 90 days old = factor 0.5
    factor = time_decay_factor(_iso(90), half_life_days=90)
    assert 0.45 < factor < 0.55


def test_recent_last_used_keeps_old_memory_warm() -> None:
    """A memory authored 6 months ago that was cited yesterday should NOT decay."""
    # updated 180d ago + last_used 1d ago, half-life 30d
    fresh = time_decay_factor(_iso(180), half_life_days=30, last_used_iso=_iso(1))
    stale = time_decay_factor(_iso(180), half_life_days=30, last_used_iso=None)
    assert fresh > stale
    # 1-day-old usage with 30d half-life → factor very close to 1.
    assert fresh > 0.95
    # Without last_used, the 180d-old memory is essentially gone.
    assert stale < 0.02


def test_uncited_memory_decays_at_full_rate() -> None:
    """A 90d-old memory with last_used at the same day shouldn't be reclaimed."""
    factor = time_decay_factor(_iso(90), half_life_days=90, last_used_iso=_iso(90))
    # Both timestamps are 90d old → max() picks one → factor ≈ 0.5.
    assert 0.45 < factor < 0.55


def test_zulu_suffix_normalized() -> None:
    # ``2026-01-01T00:00:00Z`` should parse as UTC.
    z = "2026-01-01T00:00:00Z"
    assert time_decay_factor(z, half_life_days=30) > 0.0


def test_max_of_updated_and_last_used_is_chosen() -> None:
    """If updated is more recent than last_used, the function picks updated."""
    factor_u = time_decay_factor(_iso(1), half_life_days=30, last_used_iso=_iso(60))
    factor_l = time_decay_factor(_iso(60), half_life_days=30, last_used_iso=_iso(1))
    # Both should be near 1 — the max() gets a 1d-old timestamp either way.
    assert abs(factor_u - factor_l) < 0.05
    assert factor_u > 0.95

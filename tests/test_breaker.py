"""Unit tests for the in-process circuit breaker that guards Ollama calls.

These tests exercise the breaker state machine in isolation. The full
integration with ``MemVaultService`` is covered in ``test_robustness.py``
where we stub a fake ``VectorIndex`` and assert the service-level wiring
(timeouts, error envelopes) without spinning up Ollama.
"""

from __future__ import annotations

import time

import pytest

from mem_vault.index import CircuitBreakerOpenError, _CircuitBreaker


def test_breaker_starts_closed() -> None:
    breaker = _CircuitBreaker(threshold=3, cooldown_s=10.0)
    assert breaker.is_open() is False
    breaker.check()  # should not raise


def test_breaker_opens_after_threshold_consecutive_failures() -> None:
    breaker = _CircuitBreaker(threshold=3, cooldown_s=10.0)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open() is False, "below threshold yet"
    breaker.record_failure()
    assert breaker.is_open() is True
    with pytest.raises(CircuitBreakerOpenError):
        breaker.check()


def test_breaker_records_success_resets_counter() -> None:
    breaker = _CircuitBreaker(threshold=3, cooldown_s=10.0)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()  # back to clean
    breaker.record_failure()  # only 1 in a row now
    breaker.record_failure()  # 2
    assert breaker.is_open() is False
    breaker.record_failure()  # 3 → open
    assert breaker.is_open() is True


def test_breaker_cooldown_elapses() -> None:
    breaker = _CircuitBreaker(threshold=2, cooldown_s=0.05)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open() is True
    time.sleep(0.08)
    assert breaker.is_open() is False, "cooldown should have elapsed"
    breaker.check()  # half-open: should allow the call through


def test_breaker_disabled_when_threshold_one_then_success() -> None:
    """One failure trips a threshold=1 breaker; one success heals it."""
    breaker = _CircuitBreaker(threshold=1, cooldown_s=10.0)
    breaker.record_failure()
    assert breaker.is_open() is True
    breaker.record_success()
    assert breaker.is_open() is False


def test_breaker_cooldown_remaining_decreases() -> None:
    breaker = _CircuitBreaker(threshold=1, cooldown_s=1.0)
    breaker.record_failure()
    first = breaker.cooldown_remaining()
    time.sleep(0.05)
    second = breaker.cooldown_remaining()
    assert first > 0
    assert second < first


# ---------------------------------------------------------------------------
# Failure-decay reset added in the audit pass
# ---------------------------------------------------------------------------


def test_failure_counter_decays_after_silence(monkeypatch):
    """Stale failures don't compound — counter resets after ``failure_decay_s``."""
    from mem_vault.index import _CircuitBreaker

    fake_now = [1000.0]

    def fake_monotonic():
        return fake_now[0]

    import mem_vault.index as idx_mod

    monkeypatch.setattr(idx_mod.time, "monotonic", fake_monotonic)
    breaker = _CircuitBreaker(threshold=3, cooldown_s=30.0, failure_decay_s=300.0)

    # Two failures, close together — both count.
    breaker.record_failure()
    fake_now[0] += 5
    breaker.record_failure()
    assert breaker._consecutive_failures == 2
    assert not breaker.is_open()

    # 10 minutes of silence — the next failure should NOT add to the prior count.
    fake_now[0] += 600
    breaker.record_failure()
    assert breaker._consecutive_failures == 1, (
        "stale failures should have decayed before the new one was recorded"
    )
    assert not breaker.is_open()


def test_failure_counter_does_not_decay_when_disabled(monkeypatch):
    """``failure_decay_s=0`` keeps the legacy behavior (no decay)."""
    from mem_vault.index import _CircuitBreaker

    fake_now = [1000.0]
    import mem_vault.index as idx_mod

    monkeypatch.setattr(idx_mod.time, "monotonic", lambda: fake_now[0])
    breaker = _CircuitBreaker(threshold=3, cooldown_s=30.0, failure_decay_s=0.0)

    breaker.record_failure()
    fake_now[0] += 99999  # arbitrarily long gap
    breaker.record_failure()
    assert breaker._consecutive_failures == 2  # no decay


def test_failure_decay_does_not_close_open_breaker(monkeypatch):
    """If the breaker tripped already, decay applies only on the NEXT failure."""
    from mem_vault.index import _CircuitBreaker

    fake_now = [1000.0]
    import mem_vault.index as idx_mod

    monkeypatch.setattr(idx_mod.time, "monotonic", lambda: fake_now[0])
    breaker = _CircuitBreaker(threshold=2, cooldown_s=30.0, failure_decay_s=60.0)

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open()
    # Cooldown lapses (2 min), but the breaker reads the timestamp on
    # is_open() — that path is unchanged by the decay logic.
    fake_now[0] += 120
    assert not breaker.is_open()

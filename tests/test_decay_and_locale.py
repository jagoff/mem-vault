"""Tests for time-decay scoring and locale-aware UserPromptSubmit skip rules.

Pure unit tests — no Ollama, no Qdrant, no network.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from mem_vault.hooks.userprompt import detect_script
from mem_vault.index import time_decay_factor

# ---------------------------------------------------------------------------
# time_decay_factor
# ---------------------------------------------------------------------------


def _ago(days: float) -> str:
    return (datetime.now(tz=UTC) - timedelta(days=days)).isoformat(timespec="seconds")


def test_decay_disabled_returns_one():
    assert time_decay_factor(_ago(30), 0.0) == 1.0
    assert time_decay_factor(_ago(30), -10.0) == 1.0


def test_decay_missing_timestamp_returns_one():
    assert time_decay_factor(None, 90.0) == 1.0
    assert time_decay_factor("", 90.0) == 1.0


def test_decay_unparseable_timestamp_returns_one():
    assert time_decay_factor("not a date", 90.0) == 1.0


def test_decay_at_half_life_is_half():
    f = time_decay_factor(_ago(90), 90.0)
    assert math.isclose(f, 0.5, abs_tol=0.05)


def test_decay_at_zero_age_is_one():
    f = time_decay_factor(datetime.now(tz=UTC).isoformat(), 90.0)
    assert math.isclose(f, 1.0, abs_tol=0.01)


def test_decay_future_timestamp_returns_one():
    """Defensive: a future timestamp shouldn't crash or amplify the score."""
    future = (datetime.now(tz=UTC) + timedelta(days=10)).isoformat()
    assert time_decay_factor(future, 90.0) == 1.0


def test_decay_naive_timestamp_treated_as_utc():
    naive = (datetime.utcnow() - timedelta(days=90)).isoformat()
    assert math.isclose(time_decay_factor(naive, 90.0), 0.5, abs_tol=0.1)


def test_decay_monotonic_in_age():
    """Older memories must decay more than younger ones."""
    young = time_decay_factor(_ago(10), 90.0)
    middle = time_decay_factor(_ago(60), 90.0)
    old = time_decay_factor(_ago(180), 90.0)
    assert young > middle > old


# ---------------------------------------------------------------------------
# detect_script
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hola, qué tal", "latin"),
        ("Hello world from Devin", "latin"),
        ("café résumé naïve", "latin"),  # Latin with diacritics still latin
        ("привет мир как estás", "cyrillic"),  # mostly cyrillic
        ("こんにちは世界", "cjk"),
        ("你好世界", "cjk"),
        ("안녕하세요 세상", "cjk"),
        ("שלום עולם", "hebrew"),
        ("مرحبا بالعالم", "arabic"),
        ("नमस्ते दुनिया", "devanagari"),
    ],
)
def test_detect_script_basics(text, expected):
    assert detect_script(text) == expected


def test_detect_script_unknown_when_too_short():
    assert detect_script("ok") == "unknown"
    assert detect_script("") == "unknown"


def test_detect_script_ignores_emoji_and_digits():
    """A prompt that's mostly emoji + digits returns ``unknown`` (not a script)."""
    assert detect_script("🚀 1234 ✨") == "unknown"


def test_detect_script_picks_dominant_when_mixed():
    # Mostly Latin with one Japanese character
    assert detect_script("Hello world this is mostly latin こ") == "latin"
    # Mostly CJK with one English word
    assert detect_script("こんにちは世界 hi") == "cjk"

"""Tests for the IndexLockedError detection in :mod:`mem_vault.sync`.

The key invariants we want to keep regression-proof:

1. The substring we match against (``_QDRANT_LOCK_SUBSTR``) is still
   present verbatim in the qdrant-client local backend's source. If a
   future qdrant-client release reworks the message we want a loud
   failure here, not a silent regression where ``sync_status`` starts
   re-raising opaque ``RuntimeError`` to the user.

2. ``sync_status`` translates BOTH the typed
   ``portalocker.exceptions.LockException`` (defensive case, in case
   qdrant-client ever stops wrapping it) and the string-matched
   ``RuntimeError`` (current production case) into our own
   ``IndexLockedError``.

3. ``sync_status`` does NOT swallow unrelated ``RuntimeError`` — those
   should bubble up so we don't paint over real bugs as "locked".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mem_vault import sync as sync_mod
from mem_vault.config import Config
from mem_vault.sync import (
    _QDRANT_LOCK_SUBSTR,
    IndexLockedError,
    _is_lock_exception,
    _is_lock_runtime_error,
    sync_status,
)

# ---------------------------------------------------------------------------
# Pin the substring against qdrant-client's local backend source.
# ---------------------------------------------------------------------------


def test_lock_message_substring_is_still_in_qdrant_source():
    """If qdrant-client ever changes its lock-error wording, this fails
    loudly — fix ``_QDRANT_LOCK_SUBSTR`` in step.
    """
    qdrant_local = pytest.importorskip("qdrant_client.local.qdrant_local")
    src = Path(qdrant_local.__file__).read_text(encoding="utf-8")
    assert _QDRANT_LOCK_SUBSTR in src.lower(), (
        f"qdrant-client local backend no longer mentions "
        f"{_QDRANT_LOCK_SUBSTR!r} verbatim — update _QDRANT_LOCK_SUBSTR "
        f"in src/mem_vault/sync.py to match the new message."
    )


# ---------------------------------------------------------------------------
# Predicates.
# ---------------------------------------------------------------------------


def test_is_lock_runtime_error_matches_qdrant_message():
    exc = RuntimeError(
        "Storage folder /tmp/state is already accessed by another instance "
        "of Qdrant client. If you require concurrent access, use Qdrant "
        "server instead."
    )
    assert _is_lock_runtime_error(exc) is True


def test_is_lock_runtime_error_rejects_unrelated_runtime_errors():
    assert _is_lock_runtime_error(RuntimeError("something else broke")) is False


def test_is_lock_exception_matches_portalocker_typed_exception():
    portalocker_exceptions = pytest.importorskip("portalocker.exceptions")
    exc = portalocker_exceptions.LockException("boom")
    assert _is_lock_exception(exc) is True


def test_is_lock_exception_rejects_plain_exceptions():
    assert _is_lock_exception(RuntimeError("nope")) is False
    assert _is_lock_exception(ValueError("nope")) is False


# ---------------------------------------------------------------------------
# sync_status — exception translation.
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path) -> Config:
    """Build a :class:`Config` rooted at a temp dir.

    We don't actually need the index to exist — sync_status will fail at
    the ``VectorIndex(config)`` step which we monkeypatch.
    """
    config = Config(
        vault_path=str(tmp_path),
        memory_subdir="memory",
        state_dir=str(tmp_path / "state"),
        user_id="tester",
    )
    config.qdrant_collection = "test"
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.memory_dir.mkdir(parents=True, exist_ok=True)
    return config


def test_sync_status_translates_qdrant_runtime_error_into_index_locked(monkeypatch, tmp_path):
    """Production case: qdrant-client wraps the lock failure into a
    plain ``RuntimeError`` whose message contains "already accessed"."""
    cfg = _make_config(tmp_path)

    def _boom(_: Any) -> None:
        raise RuntimeError(
            "Storage folder /x is already accessed by another instance of Qdrant client."
        )

    monkeypatch.setattr("mem_vault.index.VectorIndex", _boom)

    with pytest.raises(IndexLockedError):
        sync_status(cfg)


def test_sync_status_translates_portalocker_exception_into_index_locked(monkeypatch, tmp_path):
    """Defensive case: if a future qdrant-client lets the typed
    ``portalocker`` exception propagate, we still surface it as
    ``IndexLockedError`` rather than as an opaque crash."""
    cfg = _make_config(tmp_path)
    portalocker_exceptions = pytest.importorskip("portalocker.exceptions")

    def _boom(_: Any) -> None:
        raise portalocker_exceptions.LockException("could not acquire flock")

    monkeypatch.setattr("mem_vault.index.VectorIndex", _boom)

    with pytest.raises(IndexLockedError):
        sync_status(cfg)


def test_sync_status_lets_unrelated_runtime_error_bubble_up(monkeypatch, tmp_path):
    """Unrelated ``RuntimeError`` must NOT be re-cast as ``IndexLockedError``.

    Otherwise we'd silently paint legit bugs as "locked".
    """
    cfg = _make_config(tmp_path)

    def _boom(_: Any) -> None:
        raise RuntimeError("collection schema mismatch")

    monkeypatch.setattr("mem_vault.index.VectorIndex", _boom)

    with pytest.raises(RuntimeError) as excinfo:
        sync_status(cfg)
    assert not isinstance(excinfo.value, IndexLockedError)
    assert "schema" in str(excinfo.value)


def test_sync_status_translates_blockingio_into_index_locked(monkeypatch, tmp_path):
    """``BlockingIOError`` from a hypothetical future backend is still a lock."""
    cfg = _make_config(tmp_path)

    def _boom(_: Any) -> None:
        raise BlockingIOError("flock: would block")

    monkeypatch.setattr("mem_vault.index.VectorIndex", _boom)

    with pytest.raises(IndexLockedError):
        sync_status(cfg)


def test_sync_status_keeps_lock_msg_helper_text(monkeypatch, tmp_path):
    """The IndexLockedError raised carries the user-facing remediation hint."""
    cfg = _make_config(tmp_path)

    def _boom(_: Any) -> None:
        raise RuntimeError("/x is already accessed by another instance of Qdrant client.")

    monkeypatch.setattr("mem_vault.index.VectorIndex", _boom)

    with pytest.raises(IndexLockedError) as excinfo:
        sync_status(cfg)
    msg = str(excinfo.value)
    assert "locked by another process" in msg
    assert "MCP server" in msg


# Reference the module to keep linters happy if someone trims imports later.
_ = sync_mod

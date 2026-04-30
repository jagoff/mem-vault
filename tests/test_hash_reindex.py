"""Tests for ``compute_content_hash`` and the incremental reindex path.

The hashing helper is exercised in isolation. The reindex skip flow is
driven through ``cli/reindex.py:run`` against a stubbed ``VectorIndex``
that records what got re-embedded vs skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from mem_vault.cli import reindex as reindex_mod
from mem_vault.config import Config
from mem_vault.index import _CircuitBreaker, compute_content_hash
from mem_vault.server import MemVaultService

# ---------------------------------------------------------------------------
# compute_content_hash — pure helper
# ---------------------------------------------------------------------------


def test_hash_is_deterministic_and_short():
    h1 = compute_content_hash("the quick brown fox")
    h2 = compute_content_hash("the quick brown fox")
    assert h1 == h2
    assert len(h1) == 16
    # Hex chars only.
    assert all(c in "0123456789abcdef" for c in h1)


def test_hash_is_sensitive_to_content_changes():
    h_a = compute_content_hash("Hola mundo")
    h_b = compute_content_hash("Hola Mundo")  # capital M
    assert h_a != h_b


def test_hash_strips_surrounding_whitespace():
    """Trailing newlines or padding spaces shouldn't trigger re-embed."""
    assert compute_content_hash("body") == compute_content_hash("  body  \n\n")
    assert compute_content_hash("body\n") == compute_content_hash("body")


def test_hash_keeps_internal_whitespace_significant():
    """Internal whitespace IS part of the content; changing it changes the hash."""
    assert compute_content_hash("a b") != compute_content_hash("a  b")
    assert compute_content_hash("line1\nline2") != compute_content_hash("line1 line2")


# ---------------------------------------------------------------------------
# Incremental reindex flow
# ---------------------------------------------------------------------------


class _StubIndex:
    """VectorIndex stand-in that records add / get_by_metadata / delete calls."""

    def __init__(self) -> None:
        self.add_calls: list[tuple[str, dict]] = []
        self.delete_calls: list[tuple[str, str, str]] = []
        # ``store`` is a tiny in-memory facsimile of the Qdrant collection,
        # keyed by memory_id → list of dicts shaped like mem0 entries.
        self.store: dict[str, list[dict]] = {}
        self._breaker = _CircuitBreaker(threshold=3, cooldown_s=30.0)

    @property
    def breaker(self) -> _CircuitBreaker:
        return self._breaker

    @property
    def mem0(self):  # ``--purge`` reaches into vector_store; tests skip --purge
        raise RuntimeError("mem0 not available in stub")

    def add(self, content, *, user_id, agent_id=None, metadata=None, auto_extract=False):
        meta = dict(metadata or {})
        mem_id = meta.get("memory_id", "?")
        entry = {"id": f"emb-{len(self.store.get(mem_id, []))}", "metadata": meta}
        self.store.setdefault(mem_id, []).append(entry)
        self.add_calls.append((content, meta))
        return [{"id": entry["id"], "memory": content}]

    def get_by_metadata(self, key, value, user_id):
        if key == "memory_id":
            return list(self.store.get(value, []))
        return []

    def delete_by_metadata(self, key, value, user_id):
        before = len(self.store.get(value, []))
        if key == "memory_id":
            self.store.pop(value, None)
        self.delete_calls.append((key, value, user_id))
        return before


@pytest.fixture
def reindex_env(tmp_path: Path):
    """Build a service with the stub index plus a few seeded memories on disk."""
    cfg_kwargs = {
        "vault_path": str(tmp_path),
        "memory_subdir": "memory",
        "state_dir": str(tmp_path / "state"),
        "user_id": "tester",
        "auto_extract_default": False,
        "llm_timeout_s": 0,
        "max_content_size": 0,
    }
    config = Config(**cfg_kwargs)
    config.qdrant_collection = "test"
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.memory_dir.mkdir(parents=True, exist_ok=True)

    service = MemVaultService(config)
    stub = _StubIndex()
    service.index = stub  # type: ignore[assignment]

    # Seed three memories on disk through the real save() path (which also
    # populates the stub with fresh content_hash payloads).
    return service, stub, config


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {"auto_extract": False, "purge": False, "limit": 0, "force": False}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


async def _seed(service, n: int = 3):
    """Save ``n`` memories with predictable content."""
    ids = []
    for i in range(n):
        seed = await service.save(
            {"content": f"seed body {i}", "title": f"seed-{i}", "type": "note"}
        )
        ids.append(seed["memory"]["id"])
    return ids


async def test_reindex_skips_unchanged_memories_by_default(monkeypatch, reindex_env):
    service, stub, config = reindex_env
    await _seed(service)

    # Patch load_config + MemVaultService inside reindex.run so we don't spin up
    # a second service against the real vault.
    monkeypatch.setattr("mem_vault.config.load_config", lambda *a, **kw: config)
    monkeypatch.setattr("mem_vault.server.MemVaultService", lambda cfg: service)

    stub.add_calls.clear()
    stub.delete_calls.clear()

    rc = await reindex_mod.run(_make_args())
    assert rc == 0
    # Every memory had its hash already in the stub from the save() call →
    # skip path → no add() and no delete().
    assert stub.add_calls == []
    assert stub.delete_calls == []


async def test_reindex_re_embeds_when_content_changed(monkeypatch, reindex_env):
    service, stub, config = reindex_env
    ids = await _seed(service)

    # Tamper with one memory body on disk so its hash diverges from the index.
    target = config.memory_dir / f"{ids[0]}.md"
    target.write_text(
        target.read_text(encoding="utf-8").replace("seed body 0", "seed body 0 EDITED"),
        encoding="utf-8",
    )

    monkeypatch.setattr("mem_vault.config.load_config", lambda *a, **kw: config)
    monkeypatch.setattr("mem_vault.server.MemVaultService", lambda cfg: service)

    stub.add_calls.clear()
    stub.delete_calls.clear()

    await reindex_mod.run(_make_args())
    # Only the tampered memory should have been re-embedded.
    assert len(stub.add_calls) == 1
    assert "EDITED" in stub.add_calls[0][0]
    assert stub.delete_calls == [("memory_id", ids[0], service.config.user_id)]


async def test_reindex_force_reembeds_everything(monkeypatch, reindex_env):
    service, stub, config = reindex_env
    await _seed(service, n=4)

    monkeypatch.setattr("mem_vault.config.load_config", lambda *a, **kw: config)
    monkeypatch.setattr("mem_vault.server.MemVaultService", lambda cfg: service)

    stub.add_calls.clear()
    stub.delete_calls.clear()

    await reindex_mod.run(_make_args(force=True))
    assert len(stub.add_calls) == 4
    assert len(stub.delete_calls) == 4


async def test_reindex_treats_missing_hash_metadata_as_stale(monkeypatch, reindex_env):
    """Pre-existing entries without content_hash must be re-embedded once."""
    service, stub, config = reindex_env
    ids = await _seed(service, n=2)

    # Strip content_hash from the stub entries to simulate pre-upgrade state.
    for mid in ids:
        for entry in stub.store.get(mid, []):
            entry["metadata"].pop("content_hash", None)

    monkeypatch.setattr("mem_vault.config.load_config", lambda *a, **kw: config)
    monkeypatch.setattr("mem_vault.server.MemVaultService", lambda cfg: service)

    stub.add_calls.clear()
    stub.delete_calls.clear()

    await reindex_mod.run(_make_args())
    # Without a hash to compare against, every memory should re-embed.
    assert len(stub.add_calls) == 2
    # And the new add() calls should now include a content_hash.
    for _content, meta in stub.add_calls:
        assert "content_hash" in meta and len(meta["content_hash"]) == 16


async def test_save_writes_content_hash_into_metadata(monkeypatch, reindex_env):
    """End-to-end: a fresh save() must stamp content_hash on the index entry."""
    service, stub, _ = reindex_env
    seed = await service.save({"content": "anything", "title": "x", "type": "note"})
    mem_id = seed["memory"]["id"]

    entries = stub.store.get(mem_id, [])
    assert entries, "save() should have populated the index"
    expected_hash = compute_content_hash("anything")
    assert entries[0]["metadata"]["content_hash"] == expected_hash

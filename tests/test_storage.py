"""Filesystem-only tests for ``mem_vault.storage``.

These tests don't touch Ollama, Qdrant, or any network resource — pure I/O
against a temp dir. Should pass on macOS, Linux, and Windows alike.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import frontmatter
import pytest

from mem_vault.storage import (
    VaultStorage,
    atomic_write_bytes,
    slugify,
)

# ---------------------------------------------------------------------------
# slugify — pure function, runs anywhere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("hello world", "hello_world"),
        ("Idioma preferido para agents", "idioma_preferido_para_agents"),
        ("café — résumé", "cafe_resume"),  # NFKD normalize + ascii fold
        ("MIXED Case STRING", "mixed_case_string"),
        ("multiple   spaces", "multiple_spaces"),
        ("with/slashes\\and:colons", "with_slashes_and_colons"),
        ("", "memory"),  # empty → fallback
        ("   ", "memory"),  # whitespace-only → fallback
        ("!!!?@#$%", "memory"),  # only special chars → fallback
    ],
)
def test_slugify(inp, expected):
    assert slugify(inp) == expected


def test_slugify_truncates_long_input():
    long = "a" * 200
    assert len(slugify(long, max_len=64)) == 64


def test_slugify_idempotent():
    s = slugify("Some Title — with bits")
    assert slugify(s) == s


# ---------------------------------------------------------------------------
# atomic_write_bytes — the whole point of this commit
# ---------------------------------------------------------------------------


def test_atomic_write_creates_file(tmp_path):
    target = tmp_path / "memo.md"
    atomic_write_bytes(target, b"hello world")
    assert target.read_bytes() == b"hello world"


def test_atomic_write_replaces_existing(tmp_path):
    target = tmp_path / "memo.md"
    target.write_bytes(b"OLD")
    atomic_write_bytes(target, b"NEW")
    assert target.read_bytes() == b"NEW"


def test_atomic_write_creates_parent_dirs(tmp_path):
    target = tmp_path / "deep" / "nested" / "memo.md"
    atomic_write_bytes(target, b"data")
    assert target.read_bytes() == b"data"


def test_atomic_write_leaves_no_temp_files(tmp_path):
    """The .tmp file used during the rename must not survive a successful write."""
    target = tmp_path / "memo.md"
    atomic_write_bytes(target, b"data")
    leftovers = [p for p in tmp_path.iterdir() if p.name != "memo.md"]
    assert leftovers == [], f"unexpected leftover files: {leftovers}"


def test_atomic_write_temp_in_same_dir(tmp_path, monkeypatch):
    """The temp file must live in the target's directory, not /tmp.

    If the temp file goes to /tmp, the os.replace() at the end becomes a
    cross-filesystem rename, which is not atomic. We assert the invariant.
    """
    seen_dirs: list[str] = []
    import tempfile as _tempfile

    real = _tempfile.mkstemp

    def spy(*args, **kwargs):
        seen_dirs.append(kwargs.get("dir", "<missing>"))
        return real(*args, **kwargs)

    monkeypatch.setattr(_tempfile, "mkstemp", spy)
    target = tmp_path / "subdir" / "memo.md"
    atomic_write_bytes(target, b"x")
    assert seen_dirs, "atomic_write_bytes did not call tempfile.mkstemp"
    assert Path(seen_dirs[0]).resolve() == target.parent.resolve()


def test_atomic_write_no_partial_on_concurrent_reads(tmp_path):
    """A reader sees either the old contents or the new ones — never a mix.

    We don't model true crash semantics here (that needs a separate process),
    but we do assert that the file is never observed in a half-written state
    while atomic_write_bytes is running.
    """
    target = tmp_path / "memo.md"
    target.write_bytes(b"A" * 4096)

    stop = threading.Event()
    bad: list[str] = []

    def reader():
        while not stop.is_set():
            try:
                data = target.read_bytes()
            except FileNotFoundError:
                continue
            # The file must always be fully one of the two states. If we
            # ever see a partial mix (e.g. some "A"s and some "B"s), that's
            # a bug.
            if data and len(set(data)) not in (1,):
                bad.append(data[:64].decode("ascii", errors="replace"))

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        for _ in range(50):
            atomic_write_bytes(target, b"B" * 4096)
            atomic_write_bytes(target, b"A" * 4096)
    finally:
        stop.set()
        t.join(timeout=2)

    assert not bad, f"partial reads observed: {bad[:3]}"


# ---------------------------------------------------------------------------
# VaultStorage — the public API used by the MCP server
# ---------------------------------------------------------------------------


def test_save_round_trip(tmp_path):
    storage = VaultStorage(tmp_path)
    mem = storage.save(
        content="Body of the memory.",
        title="Some Title",
        type="note",
        tags=["a", "b"],
        agent_id="test",
        user_id="alice",
    )
    assert mem.id == "some_title"
    assert mem.body == "Body of the memory."
    assert mem.tags == ["a", "b"]

    loaded = storage.get(mem.id)
    assert loaded is not None
    assert loaded.id == mem.id
    assert loaded.name == "Some Title"
    assert loaded.body == "Body of the memory."
    assert loaded.tags == ["a", "b"]
    assert loaded.agent_id == "test"
    assert loaded.user_id == "alice"
    assert loaded.created
    assert loaded.updated == loaded.created


def test_save_unique_id_when_collision(tmp_path):
    storage = VaultStorage(tmp_path)
    a = storage.save(content="x", title="Same Title")
    b = storage.save(content="y", title="Same Title")
    c = storage.save(content="z", title="Same Title")
    assert a.id == "same_title"
    assert b.id == "same_title_2"
    assert c.id == "same_title_3"


def test_save_invalid_type_falls_back_to_note(tmp_path):
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="x", type="weird-unknown-type")
    assert mem.type == "note"


def test_update_preserves_created_bumps_updated(tmp_path):
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="x", title="t")
    original_created = mem.created
    # The IDE may run faster than 1s — make sure timestamps differ.
    import time

    time.sleep(1.1)
    updated = storage.update(mem.id, content="new body", tags=["x"])
    assert updated.created == original_created
    assert updated.updated > original_created
    assert updated.body == "new body"
    assert updated.tags == ["x"]


def test_update_missing_raises(tmp_path):
    storage = VaultStorage(tmp_path)
    with pytest.raises(FileNotFoundError):
        storage.update("does_not_exist", content="x")


def test_delete_returns_true_when_existed(tmp_path):
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="x", title="del me")
    assert storage.delete(mem.id) is True
    assert storage.get(mem.id) is None
    # second delete is a no-op
    assert storage.delete(mem.id) is False


def test_list_filters(tmp_path):
    storage = VaultStorage(tmp_path)
    storage.save(content="a", title="prefA", type="preference", tags=["x", "y"])
    storage.save(content="b", title="prefB", type="preference", tags=["y"])
    storage.save(content="c", title="factC", type="fact", tags=["x"])

    by_type = storage.list(type="preference")
    assert {m.id for m in by_type} == {"prefa", "prefb"}

    by_tag = storage.list(tags=["x"])
    assert {m.id for m in by_tag} == {"prefa", "factc"}

    by_both = storage.list(type="preference", tags=["y"])
    assert {m.id for m in by_both} == {"prefa", "prefb"}


def test_list_returns_most_recent_first(tmp_path):
    storage = VaultStorage(tmp_path)
    storage.save(content="a", title="first")
    import time

    time.sleep(0.05)
    storage.save(content="b", title="second")
    items = storage.list()
    assert items[0].id == "second"
    assert items[1].id == "first"


def test_list_skips_corrupt_files(tmp_path):
    """A corrupt .md should not crash list() — it just gets skipped."""
    storage = VaultStorage(tmp_path)
    storage.save(content="ok", title="good")
    # Drop a malformed .md that frontmatter can't parse.
    (tmp_path / "broken.md").write_bytes(b"---\nname: bad\n---\nbody\nbad: [unclosed")
    items = storage.list()
    # We accept either "good only" or "good + broken parsed leniently",
    # but list() must not raise.
    assert any(m.id == "good" for m in items)


def test_existing_origin_session_id_camelcase_is_read(tmp_path):
    """Files written by other tools may use ``originSessionId`` (camelCase)."""
    storage = VaultStorage(tmp_path)
    raw = """---
name: legacy
description: legacy memo
type: note
originSessionId: legacy-session-123
---
body
"""
    (tmp_path / "legacy.md").write_text(raw, encoding="utf-8")
    mem = storage.get("legacy")
    assert mem is not None
    assert mem.origin_session_id == "legacy-session-123"


def test_frontmatter_is_yaml_parseable(tmp_path):
    """The file we write must round-trip through python-frontmatter cleanly."""
    storage = VaultStorage(tmp_path)
    mem = storage.save(
        content="body with: colons, [brackets], {braces}",
        title="tricky title",
        tags=["with: colon", "with-dash"],
    )
    raw = storage.path_for(mem.id).read_text(encoding="utf-8")
    parsed = frontmatter.loads(raw)
    assert parsed.metadata["name"] == "tricky title"
    assert parsed.content == "body with: colons, [brackets], {braces}"


def test_atomic_write_used_by_storage(tmp_path, monkeypatch):
    """Sanity check: VaultStorage._write goes through atomic_write_bytes."""
    storage = VaultStorage(tmp_path)
    calls: list[tuple] = []
    real = atomic_write_bytes

    from mem_vault import storage as storage_mod

    def spy(path, data):
        calls.append((Path(path), len(data)))
        return real(path, data)

    monkeypatch.setattr(storage_mod, "atomic_write_bytes", spy)
    storage.save(content="x", title="t")
    assert calls, "VaultStorage._write did not use atomic_write_bytes"
    assert calls[0][0] == storage.path_for("t")

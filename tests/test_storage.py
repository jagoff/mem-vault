"""Filesystem-only tests for ``mem_vault.storage``.

These tests don't touch Ollama, Qdrant, or any network resource — pure I/O
against a temp dir. Should pass on macOS, Linux, and Windows alike.
"""

from __future__ import annotations

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


def test_save_unique_id_under_concurrent_writes(tmp_path):
    """Two threads saving with the same title must NOT clobber each other.

    Pre-fix, ``_unique_id`` did ``while exists(): bump``; both threads
    saw exists()==False at the same time, both picked the same slug,
    one body got overwritten by the other's atomic_write. Now we reserve
    via ``open(O_CREAT|O_EXCL)`` so only one thread wins each slug.
    """
    storage = VaultStorage(tmp_path)
    barrier = threading.Barrier(8)
    bodies_seen: set[str] = set()
    ids_seen: list[str] = []
    lock = threading.Lock()

    def save_one(idx: int) -> None:
        body = f"body-{idx}"
        barrier.wait()
        m = storage.save(content=body, title="Same Title")
        with lock:
            ids_seen.append(m.id)
            bodies_seen.add(body)

    threads = [threading.Thread(target=save_one, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Eight distinct ids, eight distinct files on disk
    assert len(set(ids_seen)) == 8
    files = sorted(p.stem for p in tmp_path.glob("*.md"))
    assert len(files) == 8
    # And every body wrote *some* file (no body lost to a clobber)
    surviving_bodies = {storage.get(mid).body for mid in ids_seen if storage.get(mid)}
    assert surviving_bodies == bodies_seen


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


# ---------------------------------------------------------------------------
# iter_memories + count — streaming corpus walk
# ---------------------------------------------------------------------------


def test_iter_memories_yields_in_mtime_desc_order(tmp_path):
    """``iter_memories`` matches ``list()`` order: newest mtime first."""
    storage = VaultStorage(tmp_path)
    storage.save(content="first body", title="first")
    storage.save(content="second body", title="second")
    storage.save(content="third body", title="third")
    yielded = [m.id for m in storage.iter_memories()]
    listed = [m.id for m in storage.list(limit=10)]
    assert yielded == listed
    assert yielded[0] == "third"  # most recent first


def test_iter_memories_applies_filters(tmp_path):
    storage = VaultStorage(tmp_path)
    storage.save(content="a", title="a", type="bug", tags=["x"])
    storage.save(content="b", title="b", type="note", tags=["x", "y"])
    storage.save(content="c", title="c", type="note", tags=["y"])
    bugs = list(storage.iter_memories(type="bug"))
    assert {m.id for m in bugs} == {"a"}
    tagged_x = list(storage.iter_memories(tags=["x"]))
    assert {m.id for m in tagged_x} == {"a", "b"}


def test_count_is_O_glob(tmp_path):
    """``count`` returns the number of .md files without parsing frontmatter."""
    storage = VaultStorage(tmp_path)
    assert storage.count() == 0
    storage.save(content="one", title="one")
    storage.save(content="two", title="two")
    assert storage.count() == 2
    # A malformed .md should still be counted (count() doesn't parse).
    (tmp_path / "broken.md").write_text("not yaml at all", encoding="utf-8")
    assert storage.count() == 3


def test_iter_memories_skips_unparseable(tmp_path):
    """A broken .md is silently skipped by iter_memories (mirrors list())."""
    storage = VaultStorage(tmp_path)
    storage.save(content="ok", title="ok")
    # Frontmatter starts but never closes → parse fails on some inputs.
    (tmp_path / "broken.md").write_text(
        "---\nname: broken\nthis: is: not: valid:\n: yaml :\n---\n", encoding="utf-8"
    )
    yielded = list(storage.iter_memories())
    assert any(m.id == "ok" for m in yielded)


# ---------------------------------------------------------------------------
# _reserve_unique_id max_attempts guard
# ---------------------------------------------------------------------------


def test_reserve_unique_id_caps_linear_probe(tmp_path, monkeypatch):
    """Pathological collision chains fall back to a random suffix."""
    storage = VaultStorage(tmp_path)
    # Pre-create the linear chain so the next reserve must use the fallback.
    base = "memory"
    (tmp_path / f"{base}.md").write_text("", encoding="utf-8")
    for i in range(2, 12):
        (tmp_path / f"{base}_{i}.md").write_text("", encoding="utf-8")
    # max_attempts=10 → after 10 collisions falls back to ``<base>_<hex>``.
    new_id = storage._reserve_unique_id(base, max_attempts=10)
    assert new_id.startswith(f"{base}_")
    # The fallback hex suffix is 8 chars — never collides with the linear
    # ``_2``..``_11`` shapes the precreation produced.
    suffix = new_id.removeprefix(f"{base}_")
    assert len(suffix) == 8


# ---------------------------------------------------------------------------
# record_usage / record_feedback concurrency — per-file flock fix
# ---------------------------------------------------------------------------


def test_record_usage_concurrent_increments_preserved(tmp_path):
    """N threads bumping the same memoria must preserve all N increments.

    Pre-fix: each thread did ``read → modify → write`` without any lock,
    so two threads reading ``usage_count = K`` simultaneously both wrote
    back ``K + 1`` and one increment was silently lost. Net counter ended
    up well below the number of calls.

    Post-fix: a per-file ``fcntl.flock(LOCK_EX)`` on a sibling
    ``<id>.md.lock`` serializes the read-modify-write window, so every
    increment lands on disk.

    We use a ``threading.Barrier`` so all threads contend on the lock at
    the exact same moment — maximizes the chance of catching a regression
    if the lock is removed.
    """
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="x", title="hot memoria")
    n_threads = 16
    barrier = threading.Barrier(n_threads)

    def bump():
        barrier.wait()
        storage.record_usage(mem.id)

    threads = [threading.Thread(target=bump) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = storage.get(mem.id)
    assert final is not None
    # Hard invariant: no increment can be lost. Every bump must persist.
    assert final.usage_count == n_threads, (
        f"expected {n_threads} increments, got {final.usage_count} — "
        "one or more record_usage calls lost their bump (lock missing/broken?)"
    )


def test_record_feedback_concurrent_with_record_usage(tmp_path):
    """``record_feedback`` shares the lock with ``record_usage``.

    A thumbs-up landing at the same instant as a search-driven usage
    bump must not lose either count: helpful_count and usage_count are
    both touched, and both writes go through ``_write`` (full file
    rewrite). The shared per-file flock guarantees they serialize.
    """
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="x", title="hot memoria 2")
    n_each = 10
    barrier = threading.Barrier(n_each * 2)

    def thumbs():
        barrier.wait()
        storage.record_feedback(mem.id, helpful=True)

    def usage():
        barrier.wait()
        storage.record_usage(mem.id)

    threads = [threading.Thread(target=thumbs) for _ in range(n_each)] + [
        threading.Thread(target=usage) for _ in range(n_each)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = storage.get(mem.id)
    assert final is not None
    assert final.helpful_count == n_each, f"expected {n_each} helpful, got {final.helpful_count}"
    assert final.usage_count == n_each, f"expected {n_each} usages, got {final.usage_count}"


def test_record_usage_lock_sidecar_lifecycle(tmp_path):
    """The ``<id>.md.lock`` sibling exists post-bump and is removed on delete.

    Implementation choice: we keep the lock file on disk between calls
    (avoids a TOCTOU between unlink + reopen for the next caller). On
    ``delete``, the lock sidecar is cleaned up alongside the ``.md`` so
    a deleted memoria leaves no orphan lock file behind.

    We also assert the lock file does NOT show up in ``list()`` /
    ``count()`` — those globs match ``*.md``, but ``foo.md.lock`` ends
    in ``.lock`` and must be invisible to the corpus walkers.
    """
    storage = VaultStorage(tmp_path)
    mem = storage.save(content="x", title="lock lifecycle")
    storage.record_usage(mem.id)

    md_path = storage.path_for(mem.id)
    lock_path = md_path.with_name(f"{md_path.name}.lock")

    # On POSIX the lock file should exist; on Windows it doesn't (no-op
    # path). Don't fail the suite on Windows — just check the corpus
    # invariant (which holds either way).
    import sys

    if sys.platform != "win32":
        assert lock_path.exists(), "expected lock sidecar to persist between calls"

    # Crucially, the corpus walkers don't see the lock file as a memoria.
    assert storage.count() == 1, "lock sidecar must not be counted as a memoria"
    assert {m.id for m in storage.list()} == {mem.id}

    # Delete cleans up the lock sidecar.
    assert storage.delete(mem.id) is True
    assert not md_path.exists()
    assert not lock_path.exists(), "delete() should clean up the lock sidecar"

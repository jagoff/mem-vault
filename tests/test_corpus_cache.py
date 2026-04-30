"""Tests for the corpus-list cache that backs briefing/stats/duplicates/lint/related.

The cache memoizes ``storage.list(type=None, user_id=None, limit=∞)`` for
the discovery verbs that scan the entire vault. Five verbs hit it
back-to-back during a typical ``/mv`` boot — without caching that's 5×
parse-from-disk on every session start.

Invariants we test:

1. Repeated calls within the TTL hit the cache (storage.list called once).
2. Save/update/delete invalidate so the next discovery sees fresh data.
3. Different tag filters bucket separately (``stats(cwd=X)`` vs
   ``stats(cwd=Y)`` don't clobber each other).
4. The cache survives across calls but expires after TTL.
"""

from __future__ import annotations

import pytest

from mem_vault.config import load_config
from mem_vault.server import MemVaultService


@pytest.fixture
def svc(tmp_path, monkeypatch):
    monkeypatch.setenv("MEM_VAULT_PATH", str(tmp_path))
    monkeypatch.delenv("MEM_VAULT_REMOTE_URL", raising=False)
    return MemVaultService(load_config())


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


async def test_list_corpus_caches_within_ttl(svc, monkeypatch):
    """Two calls within TTL → storage.list runs ONCE, second call returns the
    same list object from cache."""
    counter = {"calls": 0}
    real_list = svc.storage.list

    def counting_list(*a, **kw):
        counter["calls"] += 1
        return real_list(*a, **kw)

    monkeypatch.setattr(svc.storage, "list", counting_list)

    a = await svc._list_corpus()
    b = await svc._list_corpus()
    assert counter["calls"] == 1
    # Same object returned (not just equal) — confirms it's the cached snapshot.
    assert a is b


async def test_list_corpus_invalidated_after_save(svc):
    """A save call must blow the cache so the next discovery sees the new memory."""
    # Prime the cache.
    before = await svc._list_corpus()
    assert before == []

    res = await svc.save({"content": "hola", "title": "h", "auto_link": False})
    assert res["ok"] is True

    # Cache should be empty now → next call hits storage.
    after = await svc._list_corpus()
    assert len(after) == 1
    assert after[0].id == res["memory"]["id"]


async def test_list_corpus_invalidated_after_update(svc):
    res = await svc.save({"content": "x", "title": "old title", "auto_link": False})
    mid = res["memory"]["id"]
    await svc._list_corpus()  # prime
    upd = await svc.update({"id": mid, "title": "new title"})
    assert upd["ok"] is True
    fresh = await svc._list_corpus()
    assert fresh[0].name == "new title"


async def test_list_corpus_invalidated_after_delete(svc):
    res = await svc.save({"content": "doomed", "title": "x", "auto_link": False})
    mid = res["memory"]["id"]
    await svc._list_corpus()  # prime
    delete_res = await svc.delete({"id": mid})
    assert delete_res["ok"] is True
    after = await svc._list_corpus()
    assert after == []


async def test_list_corpus_buckets_by_tag_filter(svc, monkeypatch):
    """``stats(cwd=A)`` and ``stats(cwd=B)`` resolve to DIFFERENT tag filters
    and must not share a cache slot — each tag filter caches independently."""
    counter = {"calls": 0}
    real_list = svc.storage.list

    def counting_list(*a, **kw):
        counter["calls"] += 1
        return real_list(*a, **kw)

    monkeypatch.setattr(svc.storage, "list", counting_list)

    await svc._list_corpus(tags=["project:alpha"])
    await svc._list_corpus(tags=["project:beta"])
    await svc._list_corpus()  # no filter — third bucket

    assert counter["calls"] == 3, "each distinct tag filter should hit storage once"

    # Re-asking each bucket within TTL → no extra calls.
    await svc._list_corpus(tags=["project:alpha"])
    await svc._list_corpus(tags=["project:beta"])
    await svc._list_corpus()
    assert counter["calls"] == 3, "within-TTL repeats must hit cache"


async def test_list_corpus_expires_after_ttl(svc, monkeypatch):
    """Past the TTL we re-query storage. We don't actually wait — bump the
    cached entry's timestamp into the past."""
    counter = {"calls": 0}
    real_list = svc.storage.list

    def counting_list(*a, **kw):
        counter["calls"] += 1
        return real_list(*a, **kw)

    monkeypatch.setattr(svc.storage, "list", counting_list)

    await svc._list_corpus()
    assert counter["calls"] == 1

    # Force the entry to look ancient.
    key = next(iter(svc._corpus_cache))
    _ts, payload = svc._corpus_cache[key]
    svc._corpus_cache[key] = (0.0, payload)

    await svc._list_corpus()
    assert counter["calls"] == 2


async def test_briefing_uses_cached_corpus(svc, monkeypatch):
    """The discovery verbs should pull from the cache too — their
    storage.list should only run once even if briefing+stats+lint
    fire back-to-back."""
    # Prime with one memory so there's something to count.
    await svc.save({"content": "x", "title": "x", "auto_link": False})

    counter = {"calls": 0}
    real_list = svc.storage.list

    def counting_list(*a, **kw):
        counter["calls"] += 1
        return real_list(*a, **kw)

    monkeypatch.setattr(svc.storage, "list", counting_list)

    await svc.briefing({})
    await svc.stats({})
    await svc.lint({})
    # All 3 share the same key (no project_tag → empty frozenset).
    assert counter["calls"] == 1, (
        f"briefing+stats+lint should share one cached snapshot, got {counter['calls']}"
    )


async def test_cache_invalidate_clears_all_buckets(svc, monkeypatch):
    """A single save must blow EVERY tag bucket, not just the matching one.
    Otherwise a save with no project tag would leave the project-X bucket
    stale even though the global count changed."""
    counter = {"calls": 0}
    real_list = svc.storage.list

    def counting_list(*a, **kw):
        counter["calls"] += 1
        return real_list(*a, **kw)

    monkeypatch.setattr(svc.storage, "list", counting_list)

    # Prime two buckets.
    await svc._list_corpus()
    await svc._list_corpus(tags=["project:alpha"])
    assert counter["calls"] == 2

    res = await svc.save({"content": "novedad", "title": "n", "auto_link": False})
    assert res["ok"] is True

    # Both buckets should re-fetch now.
    await svc._list_corpus()
    await svc._list_corpus(tags=["project:alpha"])
    assert counter["calls"] == 4, "save must invalidate every bucket"

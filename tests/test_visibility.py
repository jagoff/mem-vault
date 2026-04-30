"""Tests for the per-agent visibility model."""

from __future__ import annotations

from mem_vault.storage import VISIBLE_TO_ALL, Memory, VaultStorage

# ---------------------------------------------------------------------------
# Memory.is_visible_to — pure function tests
# ---------------------------------------------------------------------------


def _mem(visible_to=None, agent_id="alice"):
    return Memory(
        id="m1",
        name="m",
        description="d",
        body="b",
        agent_id=agent_id,
        visible_to=[VISIBLE_TO_ALL] if visible_to is None else visible_to,
    )


def test_default_visible_to_all_is_public():
    m = _mem()
    assert m.is_visible_to("alice") is True
    assert m.is_visible_to("bob") is True
    assert m.is_visible_to(None) is True


def test_empty_visible_to_is_strictly_private_to_owner():
    m = _mem(visible_to=[], agent_id="alice")
    assert m.is_visible_to("alice") is True
    assert m.is_visible_to("bob") is False
    assert m.is_visible_to(None) is False


def test_explicit_allowlist():
    m = _mem(visible_to=["bob", "carol"], agent_id="alice")
    assert m.is_visible_to("alice") is True  # owner always sees their own
    assert m.is_visible_to("bob") is True
    assert m.is_visible_to("carol") is True
    assert m.is_visible_to("dave") is False
    assert m.is_visible_to(None) is False


def test_star_in_list_is_public():
    m = _mem(visible_to=["*"])
    assert m.is_visible_to("anyone") is True


# ---------------------------------------------------------------------------
# VaultStorage.list — viewer_agent_id filter integration
# ---------------------------------------------------------------------------


def test_list_default_no_viewer_returns_all(tmp_path):
    s = VaultStorage(tmp_path)
    s.save(content="public", title="pub", agent_id="alice")
    s.save(content="private", title="priv", agent_id="alice", visible_to=[])
    s.save(content="allowlist", title="allow", agent_id="alice", visible_to=["bob"])
    items = s.list(limit=10)  # viewer_agent_id=None → no filter
    ids = {m.id for m in items}
    assert ids == {"pub", "priv", "allow"}


def test_list_viewer_sees_only_public_and_their_own(tmp_path):
    s = VaultStorage(tmp_path)
    s.save(content="A's public", title="aPub", agent_id="alice")  # default *
    s.save(content="A's private", title="aPriv", agent_id="alice", visible_to=[])
    s.save(
        content="A's allowlist for B",
        title="aAllowB",
        agent_id="alice",
        visible_to=["bob"],
    )
    s.save(content="B's private", title="bPriv", agent_id="bob", visible_to=[])

    # bob's view: should see A's public, A's allowlist (he's in it), B's private (he owns it)
    bob_view = {m.id for m in s.list(limit=10, viewer_agent_id="bob")}
    assert bob_view == {"apub", "aallowb", "bpriv"}

    # carol's view: only A's public
    carol_view = {m.id for m in s.list(limit=10, viewer_agent_id="carol")}
    assert carol_view == {"apub"}

    # alice's view (owner of A): all of A + nothing of B
    alice_view = {m.id for m in s.list(limit=10, viewer_agent_id="alice")}
    assert alice_view == {"apub", "apriv", "aallowb"}


def test_save_default_visible_to_is_star(tmp_path):
    s = VaultStorage(tmp_path)
    m = s.save(content="x", title="t")
    assert m.visible_to == ["*"]


def test_save_explicit_visible_to_persists(tmp_path):
    s = VaultStorage(tmp_path)
    m = s.save(content="x", title="t", visible_to=["bob", "carol"])
    again = s.get(m.id)
    assert again is not None
    assert again.visible_to == ["bob", "carol"]


def test_update_can_change_visibility(tmp_path):
    s = VaultStorage(tmp_path)
    m = s.save(content="x", title="t")
    assert m.visible_to == ["*"]
    upd = s.update(m.id, visible_to=[])
    assert upd.visible_to == []


# ---------------------------------------------------------------------------
# Service-layer visibility update — regression test for a bug caught by
# the contract-symmetry test on 2026-04-30. ``MemVaultService.update`` was
# accepting ``visible_to`` in its args dict (the field is declared in
# ``MemoryUpdate`` for the HTTP endpoint), but never forwarding it to
# ``storage.update`` — so callers got a 200 OK and the visibility silently
# stayed unchanged.
# ---------------------------------------------------------------------------


async def test_service_update_honors_visible_to_list(tmp_path, monkeypatch):
    """Pass an explicit allowlist — service must forward it through to storage."""
    monkeypatch.setenv("MEM_VAULT_PATH", str(tmp_path))
    monkeypatch.delenv("MEM_VAULT_REMOTE_URL", raising=False)
    from mem_vault.config import load_config
    from mem_vault.server import MemVaultService

    svc = MemVaultService(load_config())
    res = await svc.save({"content": "hello", "title": "t", "auto_link": False})
    mid = res["memory"]["id"]
    upd = await svc.update({"id": mid, "visible_to": ["bob"]})
    assert upd["ok"] is True
    assert upd["memory"]["visible_to"] == ["bob"]


async def test_service_update_honors_visible_to_shorthand_private(tmp_path, monkeypatch):
    """The ``"private"`` shorthand must round-trip into an empty list."""
    monkeypatch.setenv("MEM_VAULT_PATH", str(tmp_path))
    monkeypatch.delenv("MEM_VAULT_REMOTE_URL", raising=False)
    from mem_vault.config import load_config
    from mem_vault.server import MemVaultService

    svc = MemVaultService(load_config())
    res = await svc.save({"content": "secret", "title": "s", "auto_link": False})
    mid = res["memory"]["id"]
    upd = await svc.update({"id": mid, "visible_to": "private"})
    assert upd["ok"] is True
    assert upd["memory"]["visible_to"] == []


async def test_service_update_honors_visible_to_shorthand_public(tmp_path, monkeypatch):
    """The ``"public"`` shorthand must round-trip into ``["*"]``."""
    monkeypatch.setenv("MEM_VAULT_PATH", str(tmp_path))
    monkeypatch.delenv("MEM_VAULT_REMOTE_URL", raising=False)
    from mem_vault.config import load_config
    from mem_vault.server import MemVaultService

    svc = MemVaultService(load_config())
    # Save private first so we have something to widen.
    res = await svc.save(
        {"content": "x", "title": "t", "visible_to": "private", "auto_link": False}
    )
    mid = res["memory"]["id"]
    upd = await svc.update({"id": mid, "visible_to": "public"})
    assert upd["ok"] is True
    assert upd["memory"]["visible_to"] == ["*"]


async def test_service_update_without_visible_to_leaves_it_unchanged(tmp_path, monkeypatch):
    """Omitting the field must NOT clobber the existing visibility — that was
    the original failure mode this test family is named after."""
    monkeypatch.setenv("MEM_VAULT_PATH", str(tmp_path))
    monkeypatch.delenv("MEM_VAULT_REMOTE_URL", raising=False)
    from mem_vault.config import load_config
    from mem_vault.server import MemVaultService

    svc = MemVaultService(load_config())
    res = await svc.save(
        {"content": "x", "title": "t", "visible_to": ["alice"], "auto_link": False}
    )
    mid = res["memory"]["id"]
    # Update touching only ``title`` — visibility must survive.
    upd = await svc.update({"id": mid, "title": "new title"})
    assert upd["ok"] is True
    assert upd["memory"]["visible_to"] == ["alice"]


def test_legacy_md_without_visible_to_defaults_to_public(tmp_path):
    """A memory file written before visibility existed must default to public."""
    s = VaultStorage(tmp_path)
    raw = """---
name: legacy memo
description: legacy
type: note
---
body
"""
    (tmp_path / "legacymemo.md").write_text(raw, encoding="utf-8")
    m = s.get("legacymemo")
    assert m is not None
    assert m.visible_to == ["*"]
    assert m.is_visible_to("anyone") is True


def test_visible_to_not_emitted_when_default(tmp_path):
    """Frontmatter shouldn't grow a noisy ``visible_to: ['*']`` on every save."""
    s = VaultStorage(tmp_path)
    m = s.save(content="x", title="t")
    raw = s.path_for(m.id).read_text(encoding="utf-8")
    assert "visible_to" not in raw, f"unexpected visible_to in:\n{raw}"


def test_visible_to_emitted_when_non_default(tmp_path):
    s = VaultStorage(tmp_path)
    m = s.save(content="x", title="t", visible_to=[])
    raw = s.path_for(m.id).read_text(encoding="utf-8")
    assert "visible_to:" in raw

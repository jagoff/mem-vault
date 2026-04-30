"""Tests for the ``mem-vault delete`` shell wrapper.

The other CRUD subcommands are covered indirectly through the MCP-server
tests; here we focus on the surface-area changes we made to ``delete``:
non-TTY invocations now refuse to proceed unless ``--yes`` is set, instead
of hanging on ``input()`` (or silently aborting on EOF) the way the prior
behavior did when stdin was a pipe / closed.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import pytest

from mem_vault.cli import crud as crud_mod


def _make_delete_args(**kwargs: Any) -> argparse.Namespace:
    """Build the Namespace the argparse layer would hand to ``run``."""
    defaults = {"id": "some-memory", "yes": False}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Non-TTY safety
# ---------------------------------------------------------------------------


def test_delete_non_tty_without_yes_exits_2(monkeypatch, capsys):
    """Piped / CI invocations must hard-fail instead of hanging on ``input()``.

    Before the fix, ``input("type 'yes' to confirm: ")`` would either block
    forever (no EOF on the pipe) or trip ``EOFError`` on a closed stdin.
    Both are surprising — CI scripts couldn't tell the command was waiting
    for human input. Now we reject the call up front with a clear message.
    """
    # Force ``sys.stdin.isatty()`` False — same shape as `< /dev/null` or a
    # pipe. We don't need to mock ``input`` because we should never reach it.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    # ``build_service`` is invoked unconditionally at the top of ``run`` —
    # we don't need to short-circuit it, just keep it from booting Qdrant /
    # Ollama. The important assertion is that ``service.delete`` is never
    # reached on the early-exit path.
    class _ExplodingService:
        async def delete(self, _payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("service.delete must not be called on the early-exit path")

    monkeypatch.setattr("mem_vault.server.build_service", lambda: _ExplodingService())

    rc = asyncio.run(crud_mod.run("delete", _make_delete_args(yes=False)))
    captured = capsys.readouterr()

    assert rc == 2
    assert "delete requires --yes" in captured.err
    assert "TTY" in captured.err
    # Nothing should have been printed to stdout.
    assert captured.out == ""


def test_delete_yes_flag_bypasses_tty_check(monkeypatch, capsys):
    """With ``--yes``, the delete should proceed even when stdin isn't a TTY.

    We mock ``build_service`` so we never touch a real Qdrant / vault — the
    fake service just records the delete call and returns the canned ok
    payload. That's enough to prove the CLI took the happy path.
    """
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    delete_calls: list[dict[str, Any]] = []

    class _FakeService:
        async def delete(self, payload: dict[str, Any]) -> dict[str, Any]:
            delete_calls.append(payload)
            return {"ok": True, "deleted_index_entries": 3}

    monkeypatch.setattr("mem_vault.server.build_service", lambda: _FakeService())

    rc = asyncio.run(crud_mod.run("delete", _make_delete_args(id="m-1", yes=True)))
    captured = capsys.readouterr()

    assert rc == 0
    assert delete_calls == [{"id": "m-1"}]
    assert "deleted" in captured.out
    assert "removed_index_entries=3" in captured.out
    # No prompt and no error on stderr.
    assert "type 'yes'" not in captured.out
    assert captured.err == ""


def test_delete_tty_without_yes_still_prompts(monkeypatch, capsys):
    """A real TTY without ``--yes`` keeps the legacy interactive confirm.

    We answer ``yes`` via a stubbed ``input`` so the call should reach
    ``service.delete`` and return ok. This covers the regression risk that
    we accidentally wired the new check too aggressively.
    """
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "yes")

    class _FakeService:
        async def delete(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {"ok": True, "deleted_index_entries": 1}

    monkeypatch.setattr("mem_vault.server.build_service", lambda: _FakeService())

    rc = asyncio.run(crud_mod.run("delete", _make_delete_args(id="m-1", yes=False)))
    captured = capsys.readouterr()

    assert rc == 0
    assert "about to delete" in captured.out
    assert "deleted" in captured.out


def test_delete_tty_user_aborts(monkeypatch, capsys):
    """A real TTY where the user types anything but ``yes`` aborts cleanly."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "no")

    class _FakeService:
        async def delete(self, _payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("delete must not be called when the user aborts")

    monkeypatch.setattr("mem_vault.server.build_service", lambda: _FakeService())

    rc = asyncio.run(crud_mod.run("delete", _make_delete_args(id="m-1", yes=False)))
    captured = capsys.readouterr()

    assert rc == 0
    assert "aborted" in captured.out


# ---------------------------------------------------------------------------
# Smoke test for the parser wiring — argparse should accept the flag combo.
# ---------------------------------------------------------------------------


def test_delete_parser_accepts_yes_flag():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    crud_mod.add_subparsers(sub)
    args = parser.parse_args(["delete", "m-1", "--yes"])
    assert args.cmd == "delete"
    assert args.id == "m-1"
    assert args.yes is True


def test_delete_parser_short_flag_y():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    crud_mod.add_subparsers(sub)
    args = parser.parse_args(["delete", "m-1", "-y"])
    assert args.yes is True


# ---------------------------------------------------------------------------
# Tail import sanity — the helper attribute is reachable as written.
# ---------------------------------------------------------------------------


def test_module_exposes_run_callable():
    assert callable(crud_mod.run)


# Keep pytest from complaining about an unused import on platforms where the
# fixtures aren't auto-imported.
pytest  # noqa: B018

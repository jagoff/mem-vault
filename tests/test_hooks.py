"""Tests for the lifecycle hooks: SessionStart, UserPromptSubmit, Stop.

Strategy: replace ``mem_vault.server.build_service`` with a stub that
returns canned data. We feed stdin via ``monkeypatch.setattr`` and capture
stdout/stderr with ``capsys``. None of these tests require Ollama,
Qdrant, or the real vault.
"""

from __future__ import annotations

import importlib
import io
import json
from unittest.mock import AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_stdin(monkeypatch, payload):
    """Replace ``sys.stdin`` with an in-memory buffer holding ``payload`` (JSON)."""
    raw = json.dumps(payload) if not isinstance(payload, str) else payload
    monkeypatch.setattr("sys.stdin", io.StringIO(raw))


def _patch_build_service(monkeypatch, hook_module: str, service):
    """Replace the ``build_service`` symbol that the hook imports lazily.

    Each hook does ``from mem_vault.server import build_service`` *inside*
    ``_gather_context``. We patch the binding on the source module, so the
    lazy import in the hook resolves to our stub.
    """
    monkeypatch.setattr("mem_vault.server.build_service", lambda *a, **kw: service)


# ---------------------------------------------------------------------------
# SessionStart hook
# ---------------------------------------------------------------------------


def test_sessionstart_emits_context_with_preferences(monkeypatch, capsys):
    service = AsyncMock()

    async def _list(args):
        if args.get("type") == "preference":
            return {
                "ok": True,
                "memories": [
                    {"id": "p1", "name": "Pref uno", "description": "Idioma rioplatense"},
                ],
            }
        if args.get("type") == "feedback":
            return {"ok": True, "memories": []}
        # Tag-filtered listings used by _fetch_project_memories return empty.
        if args.get("tags"):
            return {"ok": True, "memories": []}
        return {"ok": True, "memories": [{"id": "r1", "name": "Reciente", "description": "x"}]}

    service.list_ = _list
    service.search = AsyncMock(return_value={"ok": True, "results": []})

    # Force ``_resolve_cwd`` to return None so this legacy test stays focused
    # on the prefs+recent path. The cwd-aware branch is exercised separately.
    monkeypatch.setattr("mem_vault.hooks.sessionstart._resolve_cwd", lambda payload: None)
    _patch_stdin(monkeypatch, {})
    _patch_build_service(monkeypatch, "sessionstart", service)

    from mem_vault.hooks import sessionstart

    importlib.reload(sessionstart)
    monkeypatch.setattr("mem_vault.hooks.sessionstart._resolve_cwd", lambda payload: None)
    sessionstart.run()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    additional = payload["hookSpecificOutput"]["additionalContext"]
    assert "Preferencias y feedback" in additional
    assert "Pref uno" in additional
    assert "Memorias recientes" in additional
    assert "Reciente" in additional


# ---------------------------------------------------------------------------
# SessionStart cwd-aware (project signals from path)
# ---------------------------------------------------------------------------


def test_project_signals_from_cwd_extracts_leaf():
    from mem_vault.hooks.sessionstart import project_signals_from_cwd

    signals = project_signals_from_cwd("/Users/fer/repositories/mem-vault")
    assert signals[0] == "mem-vault"


def test_project_signals_skips_filesystem_noise():
    from mem_vault.hooks.sessionstart import project_signals_from_cwd

    # ``Users`` and ``repositories`` are noise; ``fer`` is a username-style
    # short component. None of those should appear in the signals output.
    signals = project_signals_from_cwd("/Users/fer/repositories/mem-vault")
    assert "Users" not in signals
    assert "repositories" not in signals
    assert "fer" not in signals


def test_project_signals_handles_empty_or_root():
    from mem_vault.hooks.sessionstart import project_signals_from_cwd

    assert project_signals_from_cwd(None) == []
    assert project_signals_from_cwd("") == []
    assert project_signals_from_cwd("/") == []


def test_project_signals_keeps_compound_names():
    from mem_vault.hooks.sessionstart import project_signals_from_cwd

    # Names with ``-`` or ``_`` or ``.`` always stay (those are real project names).
    signals = project_signals_from_cwd("/home/dev/work/big-data-pipeline")
    assert signals[0] == "big-data-pipeline"


def test_sessionstart_uses_cwd_for_project_memories(monkeypatch, capsys):
    """When cwd resolves to a project, memorias relevantes go to the top section."""
    service = AsyncMock()

    async def _list(args):
        if args.get("type") in ("preference", "feedback"):
            return {"ok": True, "memories": []}
        if args.get("tags") == ["project:mem-vault"]:
            return {
                "ok": True,
                "memories": [
                    {
                        "id": "proj-1",
                        "name": "Decisión arquitectura mem-vault",
                        "description": "...",
                    }
                ],
            }
        return {"ok": True, "memories": []}

    service.list_ = _list
    service.search = AsyncMock(return_value={"ok": True, "results": []})

    monkeypatch.setattr(
        "mem_vault.hooks.sessionstart._resolve_cwd",
        lambda payload: "/Users/fer/repositories/mem-vault",
    )
    _patch_stdin(monkeypatch, {})
    _patch_build_service(monkeypatch, "sessionstart", service)

    from mem_vault.hooks import sessionstart

    importlib.reload(sessionstart)
    monkeypatch.setattr(
        "mem_vault.hooks.sessionstart._resolve_cwd",
        lambda payload: "/Users/fer/repositories/mem-vault",
    )
    sessionstart.run()

    captured = capsys.readouterr()
    additional = json.loads(captured.out)["hookSpecificOutput"]["additionalContext"]
    assert "Memorias del proyecto (`mem-vault`)" in additional
    assert "Decisión arquitectura mem-vault" in additional


def test_sessionstart_falls_back_to_semantic_search_when_no_tag_matches(monkeypatch, capsys):
    """If no project:<leaf> tag exists, a semantic search on the leaf still pulls memorias."""
    service = AsyncMock()

    async def _list(args):
        if args.get("tags"):
            return {"ok": True, "memories": []}
        if args.get("type") in ("preference", "feedback"):
            return {"ok": True, "memories": []}
        return {"ok": True, "memories": []}

    service.list_ = _list
    service.search = AsyncMock(
        return_value={
            "ok": True,
            "results": [
                {
                    "memory": {
                        "id": "sem-1",
                        "name": "Semantic match for foo",
                        "description": "...",
                    }
                }
            ],
        }
    )

    monkeypatch.setattr("mem_vault.hooks.sessionstart._resolve_cwd", lambda p: "/path/to/foo")
    _patch_stdin(monkeypatch, {})
    _patch_build_service(monkeypatch, "sessionstart", service)

    from mem_vault.hooks import sessionstart

    importlib.reload(sessionstart)
    monkeypatch.setattr("mem_vault.hooks.sessionstart._resolve_cwd", lambda p: "/path/to/foo")
    sessionstart.run()

    additional = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "Memorias del proyecto (`foo`)" in additional
    assert "Semantic match for foo" in additional


def test_sessionstart_empty_vault_writes_nothing(monkeypatch, capsys):
    service = AsyncMock()
    service.list_ = AsyncMock(return_value={"ok": True, "memories": []})
    service.search = AsyncMock(return_value={"ok": True, "results": []})

    _patch_stdin(monkeypatch, {})
    _patch_build_service(monkeypatch, "sessionstart", service)

    from mem_vault.hooks import sessionstart

    importlib.reload(sessionstart)
    sessionstart.run()

    captured = capsys.readouterr()
    assert captured.out == ""


def test_sessionstart_service_init_failure_is_swallowed(monkeypatch, capsys):
    """If ``build_service()`` blows up, the hook must exit silently."""

    def _boom(*args, **kwargs):
        raise RuntimeError("config missing")

    monkeypatch.setattr("mem_vault.server.build_service", _boom)
    _patch_stdin(monkeypatch, {})

    from mem_vault.hooks import sessionstart

    importlib.reload(sessionstart)
    sessionstart.run()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "service init failed" in captured.err or "config missing" in captured.err


def test_sessionstart_invalid_stdin_does_not_crash(monkeypatch, capsys):
    """Garbage JSON on stdin must be tolerated — best-effort design."""
    service = AsyncMock()
    service.list_ = AsyncMock(return_value={"ok": True, "memories": []})
    service.search = AsyncMock(return_value={"ok": True, "results": []})

    monkeypatch.setattr("sys.stdin", io.StringIO("not valid json {{"))
    _patch_build_service(monkeypatch, "sessionstart", service)

    from mem_vault.hooks import sessionstart

    importlib.reload(sessionstart)
    sessionstart.run()  # must not raise


# ---------------------------------------------------------------------------
# UserPromptSubmit hook — _should_skip and detect_script (pure helpers)
# ---------------------------------------------------------------------------


def test_userprompt_should_skip_empty():
    from mem_vault.hooks.userprompt import _should_skip

    assert _should_skip("") == "empty"
    assert _should_skip("   ") == "empty"


def test_userprompt_should_skip_too_short():
    from mem_vault.hooks.userprompt import _should_skip

    reason = _should_skip("hi")
    assert reason is not None and reason.startswith("too_short(")


def test_userprompt_should_skip_slash_command():
    from mem_vault.hooks.userprompt import _should_skip

    # ``_should_skip`` checks ``too_short`` before ``slash_command``, so the
    # prompt has to clear the 20-char floor for the slash detection to fire.
    assert _should_skip("/recap please tell me what happened") == "slash_command"
    assert _should_skip("/clar this whole thing right now") == "slash_command"


def test_userprompt_should_skip_normal_prompt_not_skipped():
    from mem_vault.hooks.userprompt import _should_skip

    assert _should_skip("Cómo configurar Ollama y Qdrant en local?") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hola, qué tal", "latin"),
        ("こんにちは世界の人々へ", "cjk"),
        ("Привет мир друзья", "cyrillic"),
        ("نص عربي قصير", "arabic"),
        ("שלום עולם וכו", "hebrew"),
        ("12345", "unknown"),  # no letters
        ("a", "unknown"),  # below 3-letter floor
    ],
)
def test_userprompt_detect_script(text, expected):
    from mem_vault.hooks.userprompt import detect_script

    assert detect_script(text) == expected


# ---------------------------------------------------------------------------
# UserPromptSubmit hook — full run() flow
# ---------------------------------------------------------------------------


def test_userprompt_emits_context_with_search_results(monkeypatch, capsys):
    service = AsyncMock()
    service.search = AsyncMock(
        return_value={
            "ok": True,
            "results": [
                {
                    "id": "mem1",
                    "score": 0.873,
                    "memory": {
                        "id": "mem1",
                        "name": "Ollama setup",
                        "description": "Local install + bge-m3 model",
                    },
                    "snippet": "Ollama es...",
                },
            ],
        }
    )

    _patch_stdin(monkeypatch, {"prompt": "Cómo instalar Ollama localmente paso a paso"})
    _patch_build_service(monkeypatch, "userprompt", service)

    from mem_vault.hooks import userprompt

    importlib.reload(userprompt)
    userprompt.run()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    additional = payload["hookSpecificOutput"]["additionalContext"]
    assert "score 0.87" in additional
    assert "Ollama setup" in additional
    assert "Local install + bge-m3 model" in additional


def test_userprompt_user_prompt_field_fallback(monkeypatch, capsys):
    """Hook must accept ``user_prompt`` as a fallback field for ``prompt``."""
    service = AsyncMock()
    service.search = AsyncMock(return_value={"ok": True, "results": []})

    _patch_stdin(monkeypatch, {"user_prompt": "Pregunta razonablemente larga sobre Qdrant"})
    _patch_build_service(monkeypatch, "userprompt", service)

    from mem_vault.hooks import userprompt

    importlib.reload(userprompt)
    userprompt.run()

    # No results → empty stdout. The important thing is no crash.
    assert capsys.readouterr().out == ""
    # And the search ran with the fallback field.
    service.search.assert_called_once()


def test_userprompt_short_prompt_is_skipped(monkeypatch, capsys):
    _patch_stdin(monkeypatch, {"prompt": "ok"})

    from mem_vault.hooks import userprompt

    importlib.reload(userprompt)
    userprompt.run()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "skip (too_short" in captured.err


def test_userprompt_slash_command_is_skipped(monkeypatch, capsys):
    # Use a slash command long enough to skip the ``too_short`` check that
    # would otherwise win. Real slash commands tend to be short, but for
    # this test we want to assert specifically the ``slash_command`` path.
    _patch_stdin(monkeypatch, {"prompt": "/recap please tell me everything"})

    from mem_vault.hooks import userprompt

    importlib.reload(userprompt)
    userprompt.run()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "skip (slash_command)" in captured.err


def test_userprompt_search_failure_is_swallowed(monkeypatch, capsys):
    """If service.search raises, the hook still exits 0 with empty stdout."""
    service = AsyncMock()

    async def _boom(args):
        raise RuntimeError("ollama unreachable")

    service.search = _boom

    _patch_stdin(monkeypatch, {"prompt": "Una pregunta lo bastante larga para no ser skip"})
    _patch_build_service(monkeypatch, "userprompt", service)

    from mem_vault.hooks import userprompt

    importlib.reload(userprompt)
    userprompt.run()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "search failed" in captured.err


# ---------------------------------------------------------------------------
# Stop hook — appends a line to ~/.local/share/mem-vault/sessions.log
# ---------------------------------------------------------------------------


def test_stop_writes_audit_line(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("DEVIN_PROJECT_DIR", "/some/project")
    monkeypatch.setattr("mem_vault.hooks.stop._vault_memory_count", lambda: 42)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"stop_hook_active": true}'))

    from mem_vault.hooks import stop

    importlib.reload(stop)
    monkeypatch.setattr("mem_vault.hooks.stop._vault_memory_count", lambda: 42)
    stop.run()

    log_file = tmp_path / ".local" / "share" / "mem-vault" / "sessions.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8").strip()
    parts = content.split("\t")
    assert parts[1] == "stop"
    assert parts[2] == "cwd=/some/project"
    assert parts[3] == "memories=42"
    assert parts[4] == "stop_hook_active=True"


def test_stop_invalid_stdin_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("definitely not json"))
    monkeypatch.setattr("mem_vault.hooks.stop._vault_memory_count", lambda: 0)

    from mem_vault.hooks import stop

    importlib.reload(stop)
    monkeypatch.setattr("mem_vault.hooks.stop._vault_memory_count", lambda: 0)
    stop.run()  # must not raise

    log_file = tmp_path / ".local" / "share" / "mem-vault" / "sessions.log"
    assert log_file.exists()


def test_stop_appends_not_overwrites(monkeypatch, tmp_path):
    """Two consecutive runs should produce two lines, not one."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("mem_vault.hooks.stop._vault_memory_count", lambda: 1)

    from mem_vault.hooks import stop

    importlib.reload(stop)
    monkeypatch.setattr("mem_vault.hooks.stop._vault_memory_count", lambda: 1)

    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    stop.run()
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    stop.run()

    log_file = tmp_path / ".local" / "share" / "mem-vault" / "sessions.log"
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_stop_audit_line_includes_auto_feedback_counter(monkeypatch, tmp_path):
    """The new audit line carries ``auto_feedback=N`` so the user can grep it."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("mem_vault.hooks.stop._vault_memory_count", lambda: 5)
    monkeypatch.setattr("mem_vault.hooks.stop._apply_auto_feedback", lambda p: 3)
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    from mem_vault.hooks import stop

    importlib.reload(stop)
    monkeypatch.setattr("mem_vault.hooks.stop._vault_memory_count", lambda: 5)
    monkeypatch.setattr("mem_vault.hooks.stop._apply_auto_feedback", lambda p: 3)
    stop.run()

    log_file = tmp_path / ".local" / "share" / "mem-vault" / "sessions.log"
    line = log_file.read_text(encoding="utf-8").strip()
    assert "auto_feedback=3" in line


def test_stop_auto_feedback_exception_is_swallowed(monkeypatch, tmp_path, capsys):
    """If _apply_auto_feedback raises, the hook still writes the audit line."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("mem_vault.hooks.stop._vault_memory_count", lambda: 1)

    def _boom(payload):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr("mem_vault.hooks.stop._apply_auto_feedback", _boom)
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    from mem_vault.hooks import stop

    importlib.reload(stop)
    monkeypatch.setattr("mem_vault.hooks.stop._vault_memory_count", lambda: 1)
    monkeypatch.setattr("mem_vault.hooks.stop._apply_auto_feedback", _boom)
    stop.run()  # must not raise

    log_file = tmp_path / ".local" / "share" / "mem-vault" / "sessions.log"
    assert log_file.exists()
    assert "auto_feedback=0" in log_file.read_text()
    err = capsys.readouterr().err
    assert "auto-feedback failed" in err


# ---------------------------------------------------------------------------
# find_memory_citations — pure citation detection
# ---------------------------------------------------------------------------


def test_find_citations_wikilinks():
    from mem_vault.hooks.stop import find_memory_citations

    known = {"idioma_preferido", "rate_limit", "note_x"}
    text = "Según [[idioma_preferido]] el user usa rioplatense."
    assert find_memory_citations(text, known) == {"idioma_preferido"}


def test_find_citations_inline_code_spans():
    from mem_vault.hooks.stop import find_memory_citations

    known = {"long_enough_id", "rate_limit", "note_x"}
    text = "La memoria `rate_limit` dice 60/min."
    assert find_memory_citations(text, known) == {"rate_limit"}


def test_find_citations_bare_mentions_require_min_length():
    """Bare mentions (no wikilink, no backticks) only count for ids ≥ 8 chars.

    Short ids like ``note_x`` (6 chars) are too risky — they'd match common
    English/Spanish words. We require 8+ chars before a bare match counts.
    """
    from mem_vault.hooks.stop import find_memory_citations

    known = {"feedback_idioma_preferido", "nx"}
    text = "Esto lo dice feedback_idioma_preferido. Also mentions nx randomly."
    out = find_memory_citations(text, known)
    assert "feedback_idioma_preferido" in out
    assert "nx" not in out  # too short for bare match


def test_find_citations_dedupes_within_response():
    from mem_vault.hooks.stop import find_memory_citations

    known = {"some_long_memo"}
    text = "[[some_long_memo]] y después `some_long_memo` y bare some_long_memo."
    # Multiple matches of the same id should collapse to one entry.
    assert find_memory_citations(text, known) == {"some_long_memo"}


def test_find_citations_skips_unknown_ids():
    from mem_vault.hooks.stop import find_memory_citations

    known = {"known_id_alpha"}
    text = "[[unknown_id_beta]] this should NOT be included."
    assert find_memory_citations(text, known) == set()


def test_find_citations_empty_text_or_known_returns_empty():
    from mem_vault.hooks.stop import find_memory_citations

    assert find_memory_citations("", {"x"}) == set()
    assert find_memory_citations("some text", set()) == set()


# ---------------------------------------------------------------------------
# _apply_auto_feedback — end-to-end wiring with a real storage
# ---------------------------------------------------------------------------


def test_apply_auto_feedback_bumps_cited_memories(monkeypatch, tmp_path):
    """When the payload carries a ``response`` citing a known id, the
    storage counters must move."""
    from mem_vault.config import Config
    from mem_vault.hooks import stop
    from mem_vault.storage import VaultStorage

    # Set up a vault with two memorias
    vault = tmp_path / "vault"
    vault.mkdir()
    memory_dir = vault / "memory"
    memory_dir.mkdir()
    storage = VaultStorage(memory_dir)
    storage.save(content="body one", title="Long Enough Id Alpha")
    storage.save(content="body two", title="Other Long Enough Id")

    cfg = Config(
        vault_path=str(vault),
        memory_subdir="memory",
        state_dir=str(tmp_path / "state"),
        user_id="t",
        max_content_size=0,
        llm_timeout_s=0,
    )
    cfg.qdrant_collection = "test"
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("mem_vault.config.load_config", lambda: cfg)

    payload = {
        "response": "[[long_enough_id_alpha]] dice que sí, pero `other_long_enough_id` no.",
    }
    bumped = stop._apply_auto_feedback(payload)
    assert bumped == 2

    reloaded_a = storage.get("long_enough_id_alpha")
    reloaded_b = storage.get("other_long_enough_id")
    assert reloaded_a is not None and reloaded_a.last_used != ""
    assert reloaded_b is not None and reloaded_b.last_used != ""
    # helpful/unhelpful stay 0 — we bumped usage as neutral signal
    assert reloaded_a.helpful_count == 0
    assert reloaded_a.unhelpful_count == 0


def test_apply_auto_feedback_disabled_by_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MEM_VAULT_STOP_AUTO_FEEDBACK", "0")
    from mem_vault.hooks import stop

    assert stop._apply_auto_feedback({"response": "[[anything]]"}) == 0


def test_apply_auto_feedback_no_response_returns_zero(monkeypatch):
    from mem_vault.hooks import stop

    assert stop._apply_auto_feedback({}) == 0


def test_stop_vault_memory_count_returns_minus_one_when_config_fails(monkeypatch):
    """If load_config raises, _vault_memory_count should return -1, not crash."""

    def _boom():
        raise RuntimeError("vault not configured")

    monkeypatch.setattr("mem_vault.config.load_config", _boom)

    from mem_vault.hooks.stop import _vault_memory_count

    assert _vault_memory_count() == -1


# ---------------------------------------------------------------------------
# UserPromptSubmit hook — dedup of repeated memory ids
# ---------------------------------------------------------------------------


def test_userprompt_dedupes_repeated_memory_ids(monkeypatch, capsys):
    """If memory_search returns the same id twice (rare hybrid+rerank edge
    case), the injected context must list it only once."""
    service = AsyncMock()
    service.search = AsyncMock(
        return_value={
            "ok": True,
            "results": [
                {
                    "id": "dup-mem",
                    "score": 0.91,
                    "memory": {
                        "id": "dup-mem",
                        "name": "Memoria duplicada",
                        "description": "Primer hit",
                    },
                },
                {
                    "id": "dup-mem",
                    "score": 0.88,
                    "memory": {
                        "id": "dup-mem",
                        "name": "Memoria duplicada",
                        "description": "Segundo hit (mismo id)",
                    },
                },
                {
                    "id": "other",
                    "score": 0.71,
                    "memory": {
                        "id": "other",
                        "name": "Otra memoria",
                        "description": "Distinta",
                    },
                },
            ],
        }
    )

    _patch_stdin(monkeypatch, {"prompt": "Una pregunta lo bastante larga para no ser skip"})
    _patch_build_service(monkeypatch, "userprompt", service)

    from mem_vault.hooks import userprompt

    importlib.reload(userprompt)
    userprompt.run()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    additional = payload["hookSpecificOutput"]["additionalContext"]
    # The duplicated memoria should appear exactly once — first occurrence wins.
    assert additional.count("Memoria duplicada") == 1
    assert "Primer hit" in additional
    assert "Segundo hit" not in additional
    # The non-duplicate memoria still shows up.
    assert "Otra memoria" in additional


# ---------------------------------------------------------------------------
# Stop hook — sessions.log rotation + survives rotation failure
# ---------------------------------------------------------------------------


def test_stop_hook_rotates_oversized_log(monkeypatch, tmp_path):
    """When sessions.log exceeds the cap, rotate to sessions.log.1 before append."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("mem_vault.hooks.stop._vault_memory_count", lambda: 7)
    # Tiny cap so the test stays fast — 1 KiB.
    monkeypatch.setenv("MEM_VAULT_SESSIONS_LOG_MAX_BYTES", "1024")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    log_dir = tmp_path / ".local" / "share" / "mem-vault"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "sessions.log"
    # Pre-fill above the 1 KiB cap.
    log_file.write_text("x" * 4096, encoding="utf-8")
    assert log_file.stat().st_size > 1024

    from mem_vault.hooks import stop

    importlib.reload(stop)
    monkeypatch.setattr("mem_vault.hooks.stop._vault_memory_count", lambda: 7)
    stop.run()

    rotated = log_dir / "sessions.log.1"
    assert rotated.exists(), "rotation target should exist after the hook ran"
    # Rotated copy keeps the old content (≥4096 bytes of ``x``).
    assert rotated.stat().st_size >= 4096
    # New live log only has the freshly appended audit line — way under 1 KiB.
    assert log_file.exists()
    new_content = log_file.read_text(encoding="utf-8")
    assert "auto_feedback=" in new_content
    assert len(new_content) < 1024


def test_stop_hook_survives_rotation_failure(monkeypatch, tmp_path, capsys):
    """If rotation raises, the hook still exits cleanly and writes nothing weird."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("mem_vault.hooks.stop._vault_memory_count", lambda: 3)
    monkeypatch.setenv("MEM_VAULT_SESSIONS_LOG_MAX_BYTES", "1024")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    log_dir = tmp_path / ".local" / "share" / "mem-vault"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "sessions.log"
    log_file.write_text("x" * 4096, encoding="utf-8")

    # Force the rotation step to blow up — simulates an unwritable target.
    def _boom(_log_file):
        raise PermissionError("rotation target unwritable")

    monkeypatch.setattr("mem_vault.hooks.stop._maybe_rotate_sessions_log", _boom)

    from mem_vault.hooks import stop

    importlib.reload(stop)
    monkeypatch.setattr("mem_vault.hooks.stop._vault_memory_count", lambda: 3)
    monkeypatch.setattr("mem_vault.hooks.stop._maybe_rotate_sessions_log", _boom)

    # Must NOT raise — hook contract requires exit 0 on any failure path.
    stop.run()

    err = capsys.readouterr().err
    assert "rotate failed" in err
    # The hook still appended the audit line to the existing (oversized) file.
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "auto_feedback=" in content

"""Unit tests for the human-readable printers in :mod:`mem_vault.cli._common`.

These power the default (non-``--json``) output of ``mem-vault search`` /
``mem-vault list`` / ``mem-vault get``. We capture stdout/stderr with
pytest's ``capsys`` fixture and assert on the rendered shape.
"""

from __future__ import annotations

from mem_vault.cli._common import print_human_get, print_human_list, print_human_search

# ---------------------------------------------------------------------------
# print_human_search
# ---------------------------------------------------------------------------


def test_print_human_search_with_results(capsys):
    payload = {
        "ok": True,
        "query": "ollama setup",
        "count": 2,
        "results": [
            {
                "id": "mem-1",
                "score": 0.876,
                "memory": {
                    "id": "mem-1",
                    "name": "Ollama setup notes",
                    "description": "Local install + bge-m3 model",
                },
            },
            {
                "id": "mem-2",
                "score": 0.421,
                "memory": {
                    "id": "mem-2",
                    "name": "Qdrant config",
                    "description": "Embedded mode pitfalls",
                },
            },
        ],
    }
    print_human_search(payload)
    out = capsys.readouterr().out
    assert "found 2 matches for 'ollama setup'" in out
    assert "mem-1" in out
    assert "0.876" in out
    assert "Ollama setup notes" in out
    assert "Local install + bge-m3 model" in out


def test_print_human_search_empty(capsys):
    print_human_search({"ok": True, "query": "asdf", "count": 0, "results": []})
    assert capsys.readouterr().out.strip() == "no matches."


def test_print_human_search_error(capsys):
    print_human_search({"ok": False, "error": "boom"})
    captured = capsys.readouterr()
    assert "boom" in captured.err
    assert captured.out == ""


def test_print_human_search_score_missing(capsys):
    """A missing score must not crash — just omit the score tag."""
    payload = {
        "ok": True,
        "query": "x",
        "count": 1,
        "results": [
            {"id": "m", "memory": {"id": "m", "name": "m", "description": "d"}},
        ],
    }
    print_human_search(payload)
    out = capsys.readouterr().out
    assert "(score" not in out


# ---------------------------------------------------------------------------
# print_human_list
# ---------------------------------------------------------------------------


def test_print_human_list_basic(capsys):
    payload = {
        "ok": True,
        "count": 1,
        "memories": [
            {
                "id": "m-1",
                "type": "decision",
                "tags": ["a", "b"],
                "updated": "2026-04-29T10:00:00-03:00",
                "description": "short desc",
            },
        ],
    }
    print_human_list(payload)
    out = capsys.readouterr().out
    assert "1 memories:" in out
    assert "decision" in out
    assert "m-1" in out
    assert "2026-04-29" in out
    assert "#a,b" in out
    assert "short desc" in out


def test_print_human_list_error(capsys):
    print_human_list({"ok": False, "error": "boom"})
    assert "boom" in capsys.readouterr().err


def test_print_human_list_no_description(capsys):
    """Memories without description shouldn't print an empty line."""
    payload = {
        "ok": True,
        "count": 1,
        "memories": [
            {
                "id": "m-1",
                "type": "note",
                "tags": [],
                "updated": "2026-04-29T10:00:00-03:00",
                "description": "",
            },
        ],
    }
    print_human_list(payload)
    out = capsys.readouterr().out
    # Description is empty → no body line. The ID line should still be there.
    assert "m-1" in out


# ---------------------------------------------------------------------------
# print_human_get
# ---------------------------------------------------------------------------


def test_print_human_get_full_memory(capsys):
    payload = {
        "ok": True,
        "memory": {
            "id": "m-1",
            "name": "Important decision",
            "type": "decision",
            "tags": ["a", "b"],
            "created": "2026-04-29T10:00:00-03:00",
            "updated": "2026-04-29T10:30:00-03:00",
            "agent_id": "devin",
            "user_id": "default",
            "visible_to": ["*"],
            "body": "Body of the memory\nwith multiple lines.",
        },
    }
    print_human_get(payload)
    out = capsys.readouterr().out
    assert "id:" in out and "m-1" in out
    assert "Important decision" in out
    assert "decision" in out
    assert "a,b" in out
    assert "devin" in out
    assert "Body of the memory" in out
    assert "with multiple lines." in out


def test_print_human_get_missing_agent_id_uses_dash(capsys):
    payload = {
        "ok": True,
        "memory": {
            "id": "m",
            "name": "n",
            "type": "note",
            "tags": [],
            "created": "",
            "updated": "",
            "agent_id": None,
            "user_id": "default",
            "visible_to": ["*"],
            "body": "",
        },
    }
    print_human_get(payload)
    out = capsys.readouterr().out
    # The dash placeholder is rendered when agent_id is None.
    assert "agent_id:    —" in out


def test_print_human_get_error(capsys):
    print_human_get({"ok": False, "error": "boom"})
    assert "boom" in capsys.readouterr().err

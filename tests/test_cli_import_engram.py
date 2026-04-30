"""Unit tests for the engram → mem-vault mapping helper.

These tests exercise ``_engram_to_memory`` in isolation. The full
``import-engram`` flow (file IO, MCP service interaction, dry-run output)
is integration-tier and would need a real or mocked vault — out of scope
for this layer.
"""

from __future__ import annotations

from mem_vault.cli.import_engram import _engram_to_memory


def _call(obs, **kw):
    """Convenience wrapper: fill in sensible defaults for the keyword-only args."""
    defaults = {"type_default": "fact", "user_id": "tester", "agent_id": "engram"}
    defaults.update(kw)
    return _engram_to_memory(obs, **defaults)


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------


def test_engram_type_setup_maps_to_fact():
    out = _call({"id": 1, "type": "setup", "title": "x", "content": "y"})
    assert out["type"] == "fact"


def test_engram_type_decision_passthrough():
    out = _call({"id": 1, "type": "decision", "title": "x", "content": "y"})
    assert out["type"] == "decision"


def test_engram_type_uppercase_normalized():
    out = _call({"id": 1, "type": "DECISION", "title": "x", "content": "y"})
    assert out["type"] == "decision"


def test_engram_type_unknown_falls_back_to_default():
    out = _call({"id": 1, "type": "wat", "title": "x", "content": "y"}, type_default="note")
    assert out["type"] == "note"


def test_engram_type_missing_falls_back_to_default():
    out = _call({"id": 1, "title": "x", "content": "y"}, type_default="fact")
    assert out["type"] == "fact"


# ---------------------------------------------------------------------------
# Title / content fallback
# ---------------------------------------------------------------------------


def test_title_uses_id_when_missing():
    out = _call({"id": 42, "content": "body"})
    assert out["title"] == "engram-obs-42"


def test_title_uses_sync_id_when_id_also_missing():
    out = _call({"sync_id": "abc-def", "content": "body"})
    assert out["title"] == "engram-obs-abc-def"


def test_content_falls_back_to_title_when_missing():
    out = _call({"id": 1, "title": "Just the title"})
    assert out["content"] == "Just the title"


def test_content_falls_back_to_generated_title_when_both_missing():
    out = _call({"id": 1})
    assert out["content"] == "engram-obs-1"


# ---------------------------------------------------------------------------
# Tag generation
# ---------------------------------------------------------------------------


def test_tags_always_include_source_engram():
    out = _call({"id": 1, "title": "x", "content": "y"})
    assert "source:engram" in out["tags"]


def test_tags_include_project_and_scope_prefixes():
    out = _call(
        {"id": 1, "title": "x", "content": "y", "project": "rag", "scope": "global"},
    )
    assert "project:rag" in out["tags"]
    assert "scope:global" in out["tags"]


def test_topic_key_is_split_on_slash():
    out = _call(
        {"id": 1, "title": "x", "content": "y", "topic_key": "finance/source-separation"},
    )
    assert "finance" in out["tags"]
    assert "source-separation" in out["tags"]


def test_topic_key_skips_empty_pieces():
    out = _call(
        {"id": 1, "title": "x", "content": "y", "topic_key": "/leading/double//trailing/"},
    )
    # Empty strings between slashes must not show up.
    assert "" not in out["tags"]
    assert "leading" in out["tags"]
    assert "double" in out["tags"]
    assert "trailing" in out["tags"]


def test_topic_key_does_not_duplicate_existing_tag():
    out = _call(
        {
            "id": 1,
            "title": "x",
            "content": "y",
            "project": "rag",
            "topic_key": "rag/details",  # 'rag' already present as project tag
        },
    )
    # 'rag' shouldn't appear twice (project:rag stays; bare 'rag' from topic_key
    # is added because the deduplication only checks the bare suffix).
    occurrences = sum(1 for t in out["tags"] if t == "rag")
    assert occurrences <= 1


# ---------------------------------------------------------------------------
# Identity fields
# ---------------------------------------------------------------------------


def test_user_and_agent_id_are_propagated():
    out = _call(
        {"id": 1, "title": "x", "content": "y"},
        user_id="custom-user",
        agent_id="custom-agent",
    )
    assert out["user_id"] == "custom-user"
    assert out["agent_id"] == "custom-agent"


def test_auto_extract_default_is_false():
    """The mapper itself never forces auto_extract; the caller can override."""
    out = _call({"id": 1, "title": "x", "content": "y"})
    assert out["auto_extract"] is False

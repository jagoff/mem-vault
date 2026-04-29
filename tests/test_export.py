"""Tests for ``mem_vault.export``. Pure I/O on in-memory buffers."""

from __future__ import annotations

import csv
import io
import json

import pytest

from mem_vault.export import export, supported_formats
from mem_vault.storage import Memory


def _sample_memories() -> list[Memory]:
    return [
        Memory(
            id="alpha",
            name="Alpha",
            description="first memory",
            body="alpha body, has some\nnewlines and, commas",
            type="preference",
            tags=["lang", "ts"],
            created="2026-01-01T00:00:00-03:00",
            updated="2026-01-02T00:00:00-03:00",
            agent_id="devin",
            user_id="default",
        ),
        Memory(
            id="beta",
            name="Beta — café résumé",  # unicode round-trip
            description="second memory",
            body="beta body",
            type="fact",
            tags=[],
            created="2026-01-03T00:00:00-03:00",
            updated="2026-01-03T00:00:00-03:00",
            agent_id=None,
            user_id="default",
        ),
    ]


# ---------------------------------------------------------------------------
# format guard
# ---------------------------------------------------------------------------


def test_supported_formats_contains_all_expected():
    assert {"json", "jsonl", "csv", "markdown"} == supported_formats()


def test_unknown_format_raises():
    with pytest.raises(ValueError, match="unknown format"):
        export([], "xml")


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def test_json_round_trip_preserves_unicode_and_newlines():
    buf = io.StringIO()
    export(_sample_memories(), "json", out=buf)
    payload = json.loads(buf.getvalue())
    assert payload["schema"] == "mem-vault.export.v1"
    assert payload["count"] == 2
    assert payload["memories"][0]["body"].startswith("alpha body")
    assert payload["memories"][1]["name"] == "Beta — café résumé"


def test_json_no_body_strips_body():
    buf = io.StringIO()
    export(_sample_memories(), "json", out=buf, include_body=False)
    payload = json.loads(buf.getvalue())
    for m in payload["memories"]:
        assert "body" not in m


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def test_jsonl_one_per_line():
    buf = io.StringIO()
    export(_sample_memories(), "jsonl", out=buf)
    lines = buf.getvalue().splitlines()
    assert len(lines) == 2
    a = json.loads(lines[0])
    b = json.loads(lines[1])
    assert a["id"] == "alpha"
    assert b["id"] == "beta"


def test_jsonl_empty_when_no_memories():
    buf = io.StringIO()
    export([], "jsonl", out=buf)
    assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def test_csv_round_trip_quoting_handles_commas_and_newlines():
    buf = io.StringIO()
    export(_sample_memories(), "csv", out=buf)
    rows = list(csv.DictReader(io.StringIO(buf.getvalue())))
    assert len(rows) == 2
    assert rows[0]["id"] == "alpha"
    assert rows[0]["type"] == "preference"
    assert rows[0]["body"] == "alpha body, has some\nnewlines and, commas"
    assert json.loads(rows[0]["tags"]) == ["lang", "ts"]


def test_csv_no_body_drops_body_column():
    buf = io.StringIO()
    export(_sample_memories(), "csv", out=buf, include_body=False)
    reader = csv.DictReader(io.StringIO(buf.getvalue()))
    assert "body" not in reader.fieldnames


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def test_markdown_includes_headings_and_metadata():
    buf = io.StringIO()
    export(_sample_memories(), "markdown", out=buf)
    text = buf.getvalue()
    assert "# mem-vault export" in text
    assert "## Alpha" in text
    assert "## Beta — café résumé" in text
    assert "type: `preference`" in text
    assert "alpha body, has some" in text  # body present


def test_markdown_no_body_omits_content():
    buf = io.StringIO()
    export(_sample_memories(), "markdown", out=buf, include_body=False)
    text = buf.getvalue()
    assert "## Alpha" in text
    assert "alpha body" not in text  # body suppressed

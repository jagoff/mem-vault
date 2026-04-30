"""Tests for the discovery / introspection MCP tools and cross-linking wikilinks.

Covers:
- ``mem_vault.discovery`` pure helpers (project tag, classifier, tag tables,
  stats, lint, duplicates).
- ``MemVaultService`` handlers: briefing, derive_metadata, stats, duplicates,
  lint.
- ``_insert_wikilinks_section`` body augmentation.
- End-to-end ``save()`` with auto_link writes wikilinks into the body.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mem_vault.config import Config
from mem_vault.discovery import (
    compute_stats,
    derive_domain_tags,
    derive_project_tag,
    derive_technique_tag,
    derive_title_from_content,
    derive_type_from_content,
    find_duplicate_pairs_by_tag_overlap,
    lint_memory,
)
from mem_vault.index import _CircuitBreaker
from mem_vault.server import MemVaultService, _insert_wikilinks_section
from mem_vault.storage import Memory

# ---------------------------------------------------------------------------
# Project tag derivation
# ---------------------------------------------------------------------------


def test_project_tag_from_cwd_leaf():
    assert derive_project_tag("/Users/fer/repositories/mem-vault") == "mem-vault"


def test_project_tag_aliases_repo_name():
    assert derive_project_tag("/Users/fer/repos/rag-obsidian") == "obsidian-rag"


def test_project_tag_returns_none_for_root():
    assert derive_project_tag("/") is None
    assert derive_project_tag(None) is None
    assert derive_project_tag("") is None


def test_project_tag_devin_skills_override_beats_cwd():
    """A memory body about ~/.config/devin/skills tags as devin-config."""
    content = "Edité ~/.config/devin/skills/memory/SKILL.md..."
    assert derive_project_tag("/Users/fer/repos/some-other", content) == "devin-config"


def test_project_tag_claude_skills_override():
    content = "Subí mi nuevo skill a ~/.claude/skills/dream/SKILL.md"
    assert derive_project_tag("/random", content) == "claude-config"


# ---------------------------------------------------------------------------
# Type classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content,expected",
    [
        ("Resolví un bug en el módulo X — root cause: race condition.", "bug"),
        ("Decidimos ir con FastAPI sobre Flask por el async support.", "decision"),
        ("TODO: limpiar el manejo de errores en parser.py", "todo"),
        ("Prefiero rate limits a 60/min, no a 100", "preference"),
        ("El user dijo que prefiere indentación con tabs", "feedback"),
        ("Qdrant es una vector DB", "fact"),  # short + no triggers → fact
        # ``note`` is the default for long-ish content without keyword
        # triggers and without the short-fact heuristic. Multi-paragraph
        # ensures the ``len<300 && no \n\n`` short-fact branch doesn't fire.
        ("Una observación general sin keywords clave.\n\nOtro párrafo similar.", "note"),
    ],
)
def test_type_classifier(content, expected):
    assert derive_type_from_content(content) == expected


def test_type_todo_only_for_short_content():
    """TODO mentioned in a long memory body shouldn't trigger ``todo``."""
    long_content = "TODO " + "x " * 500  # >500 chars
    assert derive_type_from_content(long_content) != "todo"


# ---------------------------------------------------------------------------
# Domain + technique tags
# ---------------------------------------------------------------------------


def test_domain_tags_extract_multiple():
    content = "Configuré launchd con un plist para correr Qdrant + fastembed via ollama"
    tags = derive_domain_tags(content)
    # All three should fire; cap is 3.
    assert "launchd" in tags
    assert "qdrant" in tags
    # Cap respected
    assert len(tags) <= 3


def test_domain_tags_caps_to_three():
    content = "macos launchd qdrant fastembed fastapi sqlite pytest"
    assert len(derive_domain_tags(content, cap=3)) == 3


def test_technique_tag_picks_first_match():
    assert derive_technique_tag("Latency p95 was 120 ms before the fix") == "performance"
    assert derive_technique_tag("Foot-gun cuando no setás el env var") == "gotcha"
    assert derive_technique_tag("Sin keywords reconocibles") is None


# ---------------------------------------------------------------------------
# Title derivation
# ---------------------------------------------------------------------------


def test_title_from_heading():
    assert derive_title_from_content("# Mi título\n\nbody") == "Mi título"


def test_title_from_first_line_when_short():
    assert derive_title_from_content("Una línea corta\nresto del body") == "Una línea corta"


def test_title_truncates_long_first_line():
    long_first = "x" * 200
    out = derive_title_from_content(long_first, max_len=50)
    assert len(out) == 50


# ---------------------------------------------------------------------------
# compute_stats / duplicates / lint
# ---------------------------------------------------------------------------


def _mk(id_, type_="note", tags=None, body="", updated="", description=""):
    return Memory(
        id=id_,
        name=id_,
        description=description,
        body=body,
        type=type_,
        tags=tags or [],
        updated=updated,
    )


def test_compute_stats_counts_by_type_and_tags():
    mems = [
        _mk("a", type_="bug", tags=["x", "y"]),
        _mk("b", type_="bug", tags=["x"]),
        _mk("c", type_="decision", tags=["y", "z"]),
    ]
    stats = compute_stats(mems)
    assert stats["total"] == 3
    assert stats["by_type"] == {"bug": 2, "decision": 1}
    top = dict(stats["top_tags"])
    assert top["x"] == 2
    assert top["y"] == 2


def test_find_duplicates_by_tag_jaccard():
    mems = [
        _mk("a", tags=["x", "y", "z"]),
        _mk("b", tags=["x", "y", "z"]),  # full overlap → jaccard 1.0
        _mk("c", tags=["x", "y", "w"]),  # 2/4 = 0.5
    ]
    pairs = find_duplicate_pairs_by_tag_overlap(mems, threshold=0.7)
    # Only a/b should pass the 0.7 threshold.
    assert len(pairs) == 1
    assert pairs[0][:2] == ("a", "b")
    assert pairs[0][2] == pytest.approx(1.0)


def test_lint_flags_few_tags_and_short_body():
    mem = _mk("a", tags=["only-one"], body="x")
    issues = lint_memory(mem)
    assert any("3 tags" in i for i in issues)
    assert any("body" in i for i in issues)


def test_lint_flags_missing_aprendido_for_long_body():
    mem = _mk(
        "a",
        tags=["t1", "t2", "t3"],
        body="A long-ish body without the convention closer. " * 10,
        description="d",
        updated="2026-04-29T10:00:00-03:00",
    )
    mem.created = "2026-04-29T10:00:00-03:00"
    issues = lint_memory(mem)
    assert any("Aprendido el" in i for i in issues)


def test_lint_clean_memory_returns_empty():
    mem = Memory(
        id="ok",
        name="ok",
        description="all good",
        body="A body with content. " * 20 + "\n## Aprendido el 2026-04-29\nWhatever.",
        type="fact",
        tags=["t1", "t2", "t3"],
        created="2026-04-29T10:00:00-03:00",
        updated="2026-04-29T10:00:00-03:00",
    )
    assert lint_memory(mem) == []


# ---------------------------------------------------------------------------
# _insert_wikilinks_section
# ---------------------------------------------------------------------------


def test_wikilinks_inserted_at_end_when_no_aprendido():
    body = "## Contexto\nfoo bar\n\n## Solución\nbaz"
    out = _insert_wikilinks_section(body, [("mem-1", "primer match"), ("mem-2", "segundo")])
    assert "## Memorias relacionadas" in out
    assert "[[mem-1]] (primer match)" in out
    assert "[[mem-2]] (segundo)" in out
    # End of body
    assert out.rstrip().endswith("[[mem-2]] (segundo)")


def test_wikilinks_inserted_before_aprendido():
    body = "## Contexto\nfoo\n\n## Solución\nbar\n\n## Aprendido el 2026-04-29\nLast learning."
    out = _insert_wikilinks_section(body, [("mem-1", "")])
    # The relacionadas section comes BEFORE Aprendido el …
    related_idx = out.find("## Memorias relacionadas")
    aprendido_idx = out.find("## Aprendido el")
    assert related_idx > 0
    assert aprendido_idx > related_idx


def test_wikilinks_caps_at_three():
    body = "body"
    related = [(f"id-{i}", f"desc {i}") for i in range(10)]
    out = _insert_wikilinks_section(body, related)
    # Only first 3 should appear.
    assert out.count("[[id-") == 3


def test_wikilinks_empty_related_returns_body_unchanged():
    body = "## Contexto\nfoo"
    assert _insert_wikilinks_section(body, []) == body


# ---------------------------------------------------------------------------
# MemVaultService handlers (briefing, derive_metadata, stats, duplicates, lint)
# ---------------------------------------------------------------------------


class _StubIndex:
    def __init__(self, search_response=None):
        self.search_response = search_response or []
        self._breaker = _CircuitBreaker(threshold=3, cooldown_s=30.0)

    @property
    def breaker(self):
        return self._breaker

    def add(self, *args, **kwargs):
        return [{"id": "stub"}]

    def search(self, *args, **kwargs):
        return list(self.search_response)

    def delete_by_metadata(self, *args, **kwargs):
        return 0


@pytest.fixture
def service_factory(tmp_path: Path):
    def _make(**overrides) -> MemVaultService:
        cfg_kwargs = {
            "vault_path": str(tmp_path),
            "memory_subdir": "memory",
            "state_dir": str(tmp_path / "state"),
            "user_id": "tester",
            "auto_extract_default": False,
            "llm_timeout_s": 0,
            "max_content_size": 0,
            "auto_link_default": False,
        }
        cfg_kwargs.update(overrides)
        cfg = Config(**cfg_kwargs)
        cfg.qdrant_collection = "test"
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        cfg.memory_dir.mkdir(parents=True, exist_ok=True)
        service = MemVaultService(cfg)
        service.index = _StubIndex()  # type: ignore[assignment]
        return service

    return _make


async def test_briefing_groups_memorias_by_project(service_factory):
    service = service_factory()
    # Seed two project-tagged memorias and one global.
    await service.save(
        {"content": "A", "title": "A", "type": "fact", "tags": ["mem-vault", "x", "y"]}
    )
    await service.save(
        {"content": "B", "title": "B", "type": "decision", "tags": ["mem-vault", "z", "w"]}
    )
    await service.save(
        {"content": "C", "title": "C", "type": "note", "tags": ["other-project", "u", "v"]}
    )

    res = await service.briefing({"cwd": "/Users/fer/repositories/mem-vault"})
    assert res["ok"] is True
    assert res["project_tag"] == "mem-vault"
    assert res["project_total"] == 2
    assert res["total_global"] == 3
    assert len(res["recent_3"]) == 2
    # top_tags excludes project_tag
    top_tag_names = {t["tag"] for t in res["top_tags"]}
    assert "mem-vault" not in top_tag_names


async def test_briefing_handles_no_cwd(service_factory):
    service = service_factory()
    res = await service.briefing({})
    assert res["ok"] is True
    assert res["project_tag"] is None
    assert res["project_total"] == 0


async def test_derive_metadata_returns_complete_envelope(service_factory):
    service = service_factory()
    res = await service.derive_metadata(
        {
            "content": "# Bug fix\n\nResolví un bug en Qdrant — root cause: el lock.",
            "cwd": "/Users/fer/repositories/mem-vault",
        }
    )
    assert res["ok"] is True
    assert res["title"] == "Bug fix"
    assert res["type"] == "bug"
    assert "mem-vault" in res["tags"]
    assert "qdrant" in res["tags"]
    # bugfix technique should be the technique tag
    assert "bugfix" in res["tags"]
    assert res["missing_tags"] == 0


async def test_derive_metadata_flags_missing_tags_when_under_three(service_factory):
    service = service_factory()
    # No project tag (empty cwd), no domain triggers, no technique.
    res = await service.derive_metadata({"content": "Algo trivial sin keywords reconocibles"})
    assert res["missing_tags"] >= 1


async def test_stats_handler_aggregates(service_factory):
    service = service_factory()
    await service.save({"content": "A", "title": "A", "type": "bug", "tags": ["t1", "t2", "t3"]})
    await service.save({"content": "B", "title": "B", "type": "bug", "tags": ["t1", "t4", "t5"]})
    await service.save(
        {
            "content": "C",
            "title": "C",
            "type": "decision",
            "tags": ["t2", "t3", "t6"],
        }
    )
    res = await service.stats({})
    assert res["ok"] is True
    assert res["total"] == 3
    assert res["by_type"]["bug"] == 2
    assert res["by_type"]["decision"] == 1


async def test_duplicates_handler_returns_pairs(service_factory):
    service = service_factory()
    await service.save({"content": "A", "title": "A", "type": "note", "tags": ["x", "y", "z"]})
    await service.save({"content": "B", "title": "B", "type": "note", "tags": ["x", "y", "z"]})
    await service.save({"content": "C", "title": "C", "type": "note", "tags": ["w", "u", "t"]})
    res = await service.duplicates({"threshold": 0.5})
    assert res["ok"] is True
    assert res["count"] == 1
    assert res["pairs"][0]["jaccard"] >= 0.5


async def test_lint_handler_surfaces_issues(service_factory):
    service = service_factory()
    # Save one bad (1 tag, no description) and one good.
    await service.save({"content": "Bad", "title": "Bad", "type": "note", "tags": ["only-one"]})
    await service.save(
        {
            "content": "Good body. " * 30 + "\n## Aprendido el 2026-04-29\nClosing.",
            "title": "Good",
            "description": "Good description",
            "type": "fact",
            "tags": ["t1", "t2", "t3"],
        }
    )
    res = await service.lint({})
    assert res["ok"] is True
    assert res["with_issues"] >= 1
    bad = next((p for p in res["problems"] if p["name"] == "Bad"), None)
    assert bad is not None
    assert any("3 tags" in issue for issue in bad["issues"])


# ---------------------------------------------------------------------------
# End-to-end: save() with auto_link writes wikilinks into the body
# ---------------------------------------------------------------------------


async def test_save_with_auto_link_writes_wikilinks_in_body(service_factory):
    service = service_factory(auto_link_default=True)
    # Pre-stage the index: pretend we have an existing similar memory.
    service.index.search_response = [  # type: ignore[attr-defined]
        {
            "score": 0.85,
            "metadata": {"memory_id": "older-memory"},
            "memory": "el older memory body resumido",
        },
    ]
    res = await service.save(
        {"content": "## Contexto\nbody nuevo\n", "title": "nuevo", "type": "note"}
    )
    assert res["ok"] is True
    new_body = res["memory"]["body"]
    assert "## Memorias relacionadas" in new_body
    assert "[[older-memory]]" in new_body
    assert "el older memory body resumido" in new_body
    # related field also stamped on frontmatter
    assert res["related"] == ["older-memory"]

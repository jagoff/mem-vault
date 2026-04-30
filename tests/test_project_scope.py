"""Tests for project-scoped metadata + search filter.

``memory_save`` stamps a ``project`` field in Qdrant metadata derived
from (in order): explicit ``project`` arg > first ``project:X`` tag >
``Config.project_default``. ``memory_search`` then uses that field as
a payload filter — faster and more precise than a tag-list scan.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mem_vault.config import Config
from mem_vault.index import _CircuitBreaker
from mem_vault.server import MemVaultService


class _CapturingIndex:
    """Stub that records the metadata / filters it sees."""

    def __init__(self, hits=None):
        self.hits = hits or []
        self.last_add_metadata: dict | None = None
        self.last_search_filters: dict | None = None
        self._breaker = _CircuitBreaker(threshold=3, cooldown_s=30.0)

    @property
    def breaker(self):
        return self._breaker

    def add(self, content, **kwargs):
        self.last_add_metadata = kwargs.get("metadata")
        return [{"id": "stub"}]

    def search(self, query, **kwargs):
        # mem0's index.search merges ``user_id`` into the filters; we
        # capture whatever was passed so tests can assert ``project`` is
        # in there.
        self.last_search_filters = kwargs.get("filters") or {}
        return list(self.hits)

    def delete_by_metadata(self, *args):
        return 0


@pytest.fixture
def project_service(tmp_path: Path):
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
            **overrides,
        }
        config = Config(**cfg_kwargs)
        config.qdrant_collection = "test"
        config.state_dir.mkdir(parents=True, exist_ok=True)
        config.memory_dir.mkdir(parents=True, exist_ok=True)
        service = MemVaultService(config)
        service.index = _CapturingIndex()
        return service

    return _make


# ---------------------------------------------------------------------------
# Save — stamping project metadata
# ---------------------------------------------------------------------------


async def test_save_without_project_leaves_metadata_absent(project_service):
    """No project arg + no project: tag + no default → no project key."""
    service = project_service()
    await service.save({"content": "x", "title": "t"})
    assert "project" not in (service.index.last_add_metadata or {})


async def test_save_with_explicit_project_stamps_it(project_service):
    service = project_service()
    await service.save({"content": "x", "title": "t", "project": "mem-vault"})
    assert service.index.last_add_metadata["project"] == "mem-vault"


async def test_save_derives_project_from_tag(project_service):
    """First ``project:X`` tag wins when no explicit project passed."""
    service = project_service()
    await service.save(
        {"content": "x", "title": "t", "tags": ["foo", "project:obsidian-rag", "bar"]}
    )
    assert service.index.last_add_metadata["project"] == "obsidian-rag"


async def test_save_derives_project_from_config_default(project_service):
    service = project_service(project_default="mem-vault")
    await service.save({"content": "x", "title": "t"})
    assert service.index.last_add_metadata["project"] == "mem-vault"


async def test_save_explicit_project_beats_tag(project_service):
    """Explicit arg takes precedence over derived-from-tag."""
    service = project_service()
    await service.save(
        {
            "content": "x",
            "title": "t",
            "project": "explicit",
            "tags": ["project:from-tag"],
        }
    )
    assert service.index.last_add_metadata["project"] == "explicit"


# ---------------------------------------------------------------------------
# Search — filter composition
# ---------------------------------------------------------------------------


async def test_search_without_project_omits_filter(project_service):
    service = project_service()
    await service.search({"query": "q"})
    assert "project" not in (service.index.last_search_filters or {})


async def test_search_with_explicit_project_adds_filter(project_service):
    service = project_service()
    await service.search({"query": "q", "project": "mem-vault"})
    assert service.index.last_search_filters["project"] == "mem-vault"


async def test_search_config_default_applies_when_no_explicit(project_service):
    service = project_service(project_default="mem-vault")
    await service.search({"query": "q"})
    assert service.index.last_search_filters["project"] == "mem-vault"


async def test_search_wildcard_project_bypasses_default(project_service):
    """Explicit ``project: "*"`` means "search globally" even when the
    config default is set."""
    service = project_service(project_default="mem-vault")
    await service.search({"query": "q", "project": "*"})
    assert "project" not in (service.index.last_search_filters or {})


async def test_search_empty_string_project_bypasses_default(project_service):
    """Explicit empty string also disables the default — same semantics as ``*``."""
    service = project_service(project_default="mem-vault")
    await service.search({"query": "q", "project": ""})
    assert "project" not in (service.index.last_search_filters or {})

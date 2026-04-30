"""Tests for the contradiction-detection path in ``memory_save``.

Unit-tests the two pure helpers (``_build_contradict_prompt`` and
``_parse_contradict_response``) and the integration via
``MemVaultService.save({auto_contradict: True})`` with the Ollama LLM
call stubbed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mem_vault.config import Config
from mem_vault.index import _CircuitBreaker
from mem_vault.server import (
    MemVaultService,
    _build_contradict_prompt,
    _parse_contradict_response,
)

# ---------------------------------------------------------------------------
# _build_contradict_prompt — pure helper
# ---------------------------------------------------------------------------


def test_prompt_includes_new_and_candidate_blocks():
    prompt = _build_contradict_prompt(
        "the new claim",
        [
            ("alpha", "the old claim"),
            ("beta", "another old claim"),
        ],
    )
    assert "<<<NEW>>>" in prompt
    assert "the new claim" in prompt
    assert "<<<MEM id=alpha>>>" in prompt
    assert "the old claim" in prompt
    assert "<<<MEM id=beta>>>" in prompt


def test_prompt_instructs_strict_json_output():
    prompt = _build_contradict_prompt("x", [("a", "y")])
    assert "STRICT JSON" in prompt
    assert '"contradicts"' in prompt


def test_prompt_truncates_long_bodies():
    """Bodies over 2000 chars should be trimmed to keep the context tight."""
    long_body = "paragraph.\n\n" * 500  # ~5000 chars
    prompt = _build_contradict_prompt(long_body, [("a", long_body)])
    # The marker we add on truncation — if we trimmed correctly, both
    # blocks should carry it.
    assert prompt.count("[...truncado...]") >= 1


# ---------------------------------------------------------------------------
# _parse_contradict_response — tolerant parsing
# ---------------------------------------------------------------------------


def test_parse_standard_json_list():
    raw = json.dumps({"contradicts": ["alpha", "beta"]})
    out = _parse_contradict_response(raw, allowed_ids={"alpha", "beta"})
    assert out == ["alpha", "beta"]


def test_parse_empty_list_returns_empty():
    assert _parse_contradict_response(json.dumps({"contradicts": []}), allowed_ids={"alpha"}) == []


def test_parse_comma_separated_string_variant():
    """Some models emit `"a, b"` as a string instead of a list."""
    raw = json.dumps({"contradicts": "alpha, beta"})
    out = _parse_contradict_response(raw, allowed_ids={"alpha", "beta"})
    assert out == ["alpha", "beta"]


def test_parse_filters_hallucinated_ids():
    """LLM makes up an id not in the candidate set → filtered out."""
    raw = json.dumps({"contradicts": ["alpha", "imagined_id"]})
    out = _parse_contradict_response(raw, allowed_ids={"alpha"})
    assert out == ["alpha"]


def test_parse_dedupes_while_preserving_order():
    raw = json.dumps({"contradicts": ["alpha", "beta", "alpha"]})
    out = _parse_contradict_response(raw, allowed_ids={"alpha", "beta"})
    assert out == ["alpha", "beta"]


def test_parse_missing_key_returns_empty():
    raw = json.dumps({"something_else": ["alpha"]})
    assert _parse_contradict_response(raw, allowed_ids={"alpha"}) == []


def test_parse_alternate_key_contradictions():
    """Accept the misspelled-plural variant ``contradictions``."""
    raw = json.dumps({"contradictions": ["alpha"]})
    out = _parse_contradict_response(raw, allowed_ids={"alpha"})
    assert out == ["alpha"]


def test_parse_empty_string_returns_empty():
    assert _parse_contradict_response("", allowed_ids={"alpha"}) == []


def test_parse_non_json_returns_empty():
    assert _parse_contradict_response("just prose", allowed_ids={"alpha"}) == []


def test_parse_non_object_json_returns_empty():
    assert _parse_contradict_response("[1, 2]", allowed_ids={"alpha"}) == []


def test_parse_non_string_items_are_skipped():
    raw = json.dumps({"contradicts": ["alpha", 42, None, "beta"]})
    out = _parse_contradict_response(raw, allowed_ids={"alpha", "beta"})
    assert out == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Integration — service.save({auto_contradict=True})
# ---------------------------------------------------------------------------


class _StubIndex:
    def __init__(self, hits=None):
        self.hits = hits or []
        self._breaker = _CircuitBreaker(threshold=3, cooldown_s=30.0)

    @property
    def breaker(self):
        return self._breaker

    def add(self, content, **kwargs):
        return [{"id": "stub"}]

    def search(self, query, **kwargs):
        return list(self.hits)

    def delete_by_metadata(self, *args):
        return 0


@pytest.fixture
def contradict_service(tmp_path: Path):
    def _make(**overrides) -> MemVaultService:
        cfg_kwargs = {
            "vault_path": str(tmp_path),
            "memory_subdir": "memory",
            "state_dir": str(tmp_path / "state"),
            "user_id": "tester",
            "auto_extract_default": False,
            "llm_timeout_s": 0,
            "max_content_size": 0,
            "auto_link_default": False,  # keep focus on contradict
            **overrides,
        }
        config = Config(**cfg_kwargs)
        config.qdrant_collection = "test"
        config.state_dir.mkdir(parents=True, exist_ok=True)
        config.memory_dir.mkdir(parents=True, exist_ok=True)
        service = MemVaultService(config)
        service.index = _StubIndex()
        return service

    return _make


async def test_save_without_auto_contradict_skips_llm(contradict_service, monkeypatch):
    """Default auto_contradict=False → no LLM call, ``contradicts=[]``."""
    service = contradict_service()
    called = []

    async def _unexpected(*args, **kwargs):
        called.append("llm")
        return "{}"

    monkeypatch.setattr(service, "_call_llm_for_contradict", _unexpected)

    res = await service.save({"content": "new body", "title": "x"})
    assert called == []
    assert res["contradicts"] == []


async def test_save_with_auto_contradict_stamps_flagged_ids(contradict_service, monkeypatch):
    """With auto_contradict=True and the LLM flagging a candidate, the new
    memory's frontmatter must include the contradict id."""
    service = contradict_service()
    # Seed a candidate memoria so ``_detect_contradictions`` has something
    # to return from its search.
    existing = await service.save({"content": "old claim", "title": "old"})
    # Make the dense stub return the seed when the save runs.
    service.index = _StubIndex(
        [
            {
                "score": 0.9,
                "metadata": {"memory_id": existing["memory"]["id"]},
                "memory": "old claim",
            }
        ]
    )

    async def _fake_llm(prompt):
        return json.dumps({"contradicts": [existing["memory"]["id"]]})

    monkeypatch.setattr(service, "_call_llm_for_contradict", _fake_llm)

    res = await service.save(
        {"content": "new claim that contradicts", "title": "new", "auto_contradict": True}
    )
    assert existing["memory"]["id"] in res["contradicts"]
    # And the frontmatter on disk must carry it too.
    reloaded = service.storage.get(res["memory"]["id"])
    assert reloaded is not None
    assert existing["memory"]["id"] in reloaded.contradicts


async def test_save_with_auto_contradict_no_match_leaves_empty(contradict_service, monkeypatch):
    service = contradict_service()
    existing = await service.save({"content": "old claim", "title": "old"})
    service.index = _StubIndex(
        [
            {
                "score": 0.9,
                "metadata": {"memory_id": existing["memory"]["id"]},
                "memory": "old claim",
            }
        ]
    )

    async def _fake_llm(prompt):
        return json.dumps({"contradicts": []})

    monkeypatch.setattr(service, "_call_llm_for_contradict", _fake_llm)

    res = await service.save(
        {"content": "new complementary claim", "title": "new", "auto_contradict": True}
    )
    assert res["contradicts"] == []


async def test_save_with_auto_contradict_llm_failure_returns_empty(contradict_service, monkeypatch):
    """LLM crash should not bubble up — the memory still gets saved, just
    without contradict detection."""
    service = contradict_service()
    existing = await service.save({"content": "old claim", "title": "old"})
    service.index = _StubIndex(
        [
            {
                "score": 0.9,
                "metadata": {"memory_id": existing["memory"]["id"]},
                "memory": "old claim",
            }
        ]
    )

    async def _boom(prompt):
        raise RuntimeError("ollama dead")

    monkeypatch.setattr(service, "_call_llm_for_contradict", _boom)

    res = await service.save({"content": "new body", "title": "new", "auto_contradict": True})
    # Save still succeeds — contradict detection is best-effort.
    assert res["ok"] is True
    assert res["contradicts"] == []


async def test_save_with_auto_contradict_filters_hallucinated_ids(contradict_service, monkeypatch):
    """If the LLM invents an id not among the candidates, we drop it."""
    service = contradict_service()
    existing = await service.save({"content": "old claim", "title": "old"})
    service.index = _StubIndex(
        [
            {
                "score": 0.9,
                "metadata": {"memory_id": existing["memory"]["id"]},
                "memory": "old claim",
            }
        ]
    )

    async def _fake_llm(prompt):
        return json.dumps({"contradicts": [existing["memory"]["id"], "made_up_memoria"]})

    monkeypatch.setattr(service, "_call_llm_for_contradict", _fake_llm)

    res = await service.save({"content": "new claim", "title": "new", "auto_contradict": True})
    assert res["contradicts"] == [existing["memory"]["id"]]
    assert "made_up_memoria" not in res["contradicts"]


async def test_save_contradict_respects_config_default(contradict_service, monkeypatch):
    """``auto_contradict_default=True`` should trigger detection even when
    the caller didn't pass ``auto_contradict`` explicitly."""
    service = contradict_service(auto_contradict_default=True)
    existing = await service.save({"content": "old", "title": "old"})
    service.index = _StubIndex(
        [
            {
                "score": 0.9,
                "metadata": {"memory_id": existing["memory"]["id"]},
                "memory": "old",
            }
        ]
    )

    called = []

    async def _fake_llm(prompt):
        called.append(prompt)
        return json.dumps({"contradicts": []})

    monkeypatch.setattr(service, "_call_llm_for_contradict", _fake_llm)

    await service.save({"content": "new body", "title": "new"})  # no explicit flag
    assert called, "config default auto_contradict_default=True should trigger LLM call"

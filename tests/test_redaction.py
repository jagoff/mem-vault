"""Tests for ``mem_vault.redaction`` and its integration with ``memory_save``.

Pure-function layer is parametrized across every pattern we care about.
Integration test verifies the save handler actually rewrites the ``.md``
body and surfaces a ``redactions`` summary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mem_vault.config import Config
from mem_vault.index import _CircuitBreaker
from mem_vault.redaction import contains_secrets, redact
from mem_vault.server import MemVaultService

# ---------------------------------------------------------------------------
# Pure redaction patterns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected_kind",
    [
        ("AKIAIOSFODNN7EXAMPLE", "aws_access_key"),
        ("ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL", "github_token"),
        ("gho_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL", "github_token"),
        ("sk-ant-abcdefghijklmnop1234567890xyz", "anthropic_key"),
        ("sk-proj-abcdefghijklmnop1234567890", "openai_key"),
        ("xoxb-12345-abcdef-supersecrettoken", "slack_token"),
        ("AIzaSyA-exampleKEYxxxxxxxxxxxxxxxxxxxxx", "google_api_key"),
    ],
)
def test_redact_detects_common_key_shapes(raw, expected_kind):
    text = f"The key is {raw} — keep it secret."
    out, hits = redact(text)
    assert raw not in out, f"raw secret leaked into output: {out!r}"
    assert any(h.kind == expected_kind for h in hits), [(h.kind, h.count) for h in hits]


def test_redact_authorization_bearer():
    text = "Request: Authorization: Bearer abcdef1234567890XYZ0987654321"
    out, hits = redact(text)
    assert "abcdef1234567890XYZ0987654321" not in out
    assert any(h.kind == "bearer_token" for h in hits)
    # The ``Authorization: Bearer `` prefix must survive — we want humans
    # to still see the structure, just not the token.
    assert "Authorization: Bearer" in out
    assert "[REDACTED:bearer_token]" in out


def test_redact_jwt():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.s3cretsignaturePartHere"
    out, hits = redact(f"token={jwt}")
    assert jwt not in out
    assert any(h.kind == "jwt" for h in hits)


def test_redact_pem_private_key_block():
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEAxample...\nmorebase64content==\n"
        "-----END RSA PRIVATE KEY-----"
    )
    text = f"config blob:\n{pem}\n# end"
    out, hits = redact(text)
    assert "MIIEpAIBAAKCAQEA" not in out
    assert any(h.kind == "private_key" for h in hits)


def test_redact_credential_assignments():
    text = "PASSWORD=\"hunter2\"\napi_token: s3cretAF\nSECRET = 'hunter22'"
    out, hits = redact(text)
    assert "hunter2" not in out
    assert "s3cretAF" not in out
    assert "hunter22" not in out
    # Three distinct assignments should produce at least one "credential_assignment" hit.
    total = sum(h.count for h in hits if h.kind == "credential_assignment")
    assert total >= 3


def test_redact_leaves_clean_text_untouched():
    clean = "Hola mundo\nEsto es una memoria normal sin secretos."
    out, hits = redact(clean)
    assert out == clean
    assert hits == []


def test_redact_is_idempotent_on_redacted_output():
    text = "token=abcdefghijklmn123456"
    once, _ = redact(text)
    twice, hits_twice = redact(once)
    assert once == twice
    assert hits_twice == []


def test_redact_empty_input():
    out, hits = redact("")
    assert out == ""
    assert hits == []


def test_contains_secrets_fast_check():
    assert contains_secrets("sk-ant-abcdefghijklmnop1234567890xyz") is True
    assert contains_secrets("plain text no secrets") is False
    assert contains_secrets("") is False


# ---------------------------------------------------------------------------
# Integration — MemVaultService.save rewrites the body
# ---------------------------------------------------------------------------


class _StubIndex:
    def __init__(self, hits=None):
        self.hits = hits or []
        self._breaker = _CircuitBreaker(threshold=3, cooldown_s=30.0)

    @property
    def breaker(self):
        return self._breaker

    def add(self, content, **kwargs):
        # Capture what we were asked to embed — the test asserts the
        # redacted version hit the index, not the raw secret.
        self.last_add_content = content
        return [{"id": "stub"}]

    def search(self, query, **kwargs):
        return list(self.hits)

    def delete_by_metadata(self, *args):
        return 0


@pytest.fixture
def redaction_service(tmp_path: Path):
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
        service.index = _StubIndex()
        return service

    return _make


async def test_save_redacts_aws_key_from_body(redaction_service):
    service = redaction_service()
    raw = "My dev creds:\nAKIAIOSFODNN7EXAMPLE\nShould not leak."
    res = await service.save({"content": raw, "title": "creds"})
    assert res["ok"] is True
    # Summary surfaced
    assert len(res["redactions"]) >= 1
    assert any(h["kind"] == "aws_access_key" for h in res["redactions"])
    # Body on disk no longer contains the raw secret
    reloaded = service.storage.get(res["memory"]["id"])
    assert reloaded is not None
    assert "AKIAIOSFODNN7EXAMPLE" not in reloaded.body
    assert "[REDACTED:aws_access_key]" in reloaded.body
    # And the index saw the redacted version (not the raw)
    assert "AKIAIOSFODNN7EXAMPLE" not in service.index.last_add_content


async def test_save_redact_disabled_keeps_raw_content(redaction_service):
    service = redaction_service(redact_secrets=False)
    raw = "AKIAIOSFODNN7EXAMPLE should stay raw"
    res = await service.save({"content": raw, "title": "x"})
    assert res["redactions"] == []
    reloaded = service.storage.get(res["memory"]["id"])
    assert reloaded is not None
    assert "AKIAIOSFODNN7EXAMPLE" in reloaded.body


async def test_save_clean_content_has_empty_redactions_list(redaction_service):
    service = redaction_service()
    res = await service.save(
        {"content": "Just a note, nothing credential-shaped here.", "title": "clean"}
    )
    assert res["redactions"] == []

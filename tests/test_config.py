"""Config loader tests — env vars, TOML, defaults."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mem_vault.config import Config, _vault_candidates, load_config


def _fresh_env(monkeypatch):
    """Strip every MEM_VAULT_* env var so each test starts clean."""
    for key in list(os.environ):
        if key.startswith("MEM_VAULT_") or key in {"OBSIDIAN_VAULT_PATH"}:
            monkeypatch.delenv(key, raising=False)


def test_explicit_env_vault_path(tmp_path, monkeypatch):
    _fresh_env(monkeypatch)
    monkeypatch.setenv("MEM_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEM_VAULT_STATE_DIR", str(tmp_path / "state"))
    cfg = load_config(config_path=Path("/dev/null"))
    assert cfg.vault_path == tmp_path.resolve()


def test_obsidian_alias_env(tmp_path, monkeypatch):
    _fresh_env(monkeypatch)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEM_VAULT_STATE_DIR", str(tmp_path / "state"))
    cfg = load_config(config_path=Path("/dev/null"))
    assert cfg.vault_path == tmp_path.resolve()


def test_missing_vault_raises_helpful_error(monkeypatch, tmp_path):
    _fresh_env(monkeypatch)
    # Force the candidates list to point at non-existent dirs.
    monkeypatch.setattr("mem_vault.config._vault_candidates", lambda: [tmp_path / "nope"])
    with pytest.raises(ValueError, match="No vault_path configured"):
        load_config(config_path=Path("/dev/null"))


def test_per_agent_collection_default(tmp_path, monkeypatch):
    _fresh_env(monkeypatch)
    monkeypatch.setenv("MEM_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEM_VAULT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEM_VAULT_AGENT_ID", "claude-code")
    cfg = load_config(config_path=Path("/dev/null"))
    assert cfg.qdrant_collection == "mem_vault_claude-code"


def test_explicit_collection_overrides_agent(tmp_path, monkeypatch):
    _fresh_env(monkeypatch)
    monkeypatch.setenv("MEM_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEM_VAULT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEM_VAULT_AGENT_ID", "claude-code")
    monkeypatch.setenv("MEM_VAULT_COLLECTION", "shared_team")
    cfg = load_config(config_path=Path("/dev/null"))
    assert cfg.qdrant_collection == "shared_team"


def test_default_collection_no_agent(tmp_path, monkeypatch):
    _fresh_env(monkeypatch)
    monkeypatch.setenv("MEM_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEM_VAULT_STATE_DIR", str(tmp_path / "state"))
    cfg = load_config(config_path=Path("/dev/null"))
    assert cfg.qdrant_collection == "mem_vault"


def test_toml_overrides_defaults(tmp_path, monkeypatch):
    _fresh_env(monkeypatch)
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text(
        f"""
vault_path = "{tmp_path}"
memory_subdir = "custom-memories"
state_dir = "{tmp_path / "state"}"
llm_model = "qwen2.5:14b"
""",
        encoding="utf-8",
    )
    cfg = load_config(config_path=cfg_file)
    assert cfg.memory_subdir == "custom-memories"
    assert cfg.llm_model == "qwen2.5:14b"


def test_env_overrides_toml(tmp_path, monkeypatch):
    _fresh_env(monkeypatch)
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text(
        f"""
vault_path = "{tmp_path}"
state_dir = "{tmp_path / "state"}"
llm_model = "from_toml"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("MEM_VAULT_LLM_MODEL", "from_env")
    cfg = load_config(config_path=cfg_file)
    assert cfg.llm_model == "from_env"


def test_state_dir_is_platform_appropriate():
    """The default state_dir lives under a platform-appropriate user-data dir."""
    cfg = Config(vault_path=Path("/tmp"))
    s = str(cfg.state_dir)
    # On macOS/Linux the default ends with "mem-vault" (a single component).
    # On Windows platformdirs emits ".../mem-vault/mem-vault" — we accept both.
    assert s.endswith("mem-vault") or s.endswith(os.path.join("mem-vault", "mem-vault"))


def test_vault_candidates_includes_macos_icloud(monkeypatch, tmp_path):
    """The candidate list should include the macOS iCloud-Obsidian path."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    candidates = _vault_candidates()
    assert any("iCloud~md~obsidian" in str(c) for c in candidates)


def test_auto_extract_default_env_parsed_as_bool(tmp_path, monkeypatch):
    _fresh_env(monkeypatch)
    monkeypatch.setenv("MEM_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEM_VAULT_STATE_DIR", str(tmp_path / "state"))

    monkeypatch.setenv("MEM_VAULT_AUTO_EXTRACT", "true")
    assert load_config(config_path=Path("/dev/null")).auto_extract_default is True

    monkeypatch.setenv("MEM_VAULT_AUTO_EXTRACT", "false")
    assert load_config(config_path=Path("/dev/null")).auto_extract_default is False

    monkeypatch.setenv("MEM_VAULT_AUTO_EXTRACT", "1")
    assert load_config(config_path=Path("/dev/null")).auto_extract_default is True

    monkeypatch.setenv("MEM_VAULT_AUTO_EXTRACT", "0")
    assert load_config(config_path=Path("/dev/null")).auto_extract_default is False


# ---------------------------------------------------------------------------
# qdrant_collection sanitization for non-slug agent_ids
# ---------------------------------------------------------------------------


def test_qdrant_collection_sanitizes_agent_id_with_spaces(monkeypatch, tmp_path):
    """Agent ids with spaces / punctuation must yield a Qdrant-legal name."""
    from mem_vault.config import load_config

    monkeypatch.setenv("MEM_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEM_VAULT_AGENT_ID", "claude code")
    monkeypatch.setenv("MEM_VAULT_STATE_DIR", str(tmp_path / "_state"))
    monkeypatch.delenv("MEM_VAULT_COLLECTION", raising=False)

    cfg = load_config()
    assert cfg.qdrant_collection == "mem_vault_claude_code"
    # Qdrant collection regex: ``^[a-zA-Z0-9_-]+$``
    import re

    assert re.match(r"^[a-zA-Z0-9_-]+$", cfg.qdrant_collection or "")


def test_qdrant_collection_falls_back_when_agent_id_is_only_punctuation(monkeypatch, tmp_path):
    """If sanitizing yields an empty string, fall back to ``mem_vault``."""
    from mem_vault.config import load_config

    monkeypatch.setenv("MEM_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("MEM_VAULT_AGENT_ID", "@@@")
    monkeypatch.setenv("MEM_VAULT_STATE_DIR", str(tmp_path / "_state"))
    monkeypatch.delenv("MEM_VAULT_COLLECTION", raising=False)

    cfg = load_config()
    assert cfg.qdrant_collection == "mem_vault"

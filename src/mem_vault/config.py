"""Configuration loader for mem-vault.

Resolution order (highest priority first):
1. Environment variables (``MEM_VAULT_*``)
2. ``$XDG_CONFIG_HOME/mem-vault/config.toml`` (defaults to ``~/.config/mem-vault/config.toml``)
3. Hard-coded defaults

The config never reaches the network: vault stays on disk, Qdrant runs embedded,
Ollama runs on localhost. No API keys are required or read.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class Config(BaseModel):
    """Runtime configuration for the mem-vault MCP server."""

    vault_path: Path = Field(
        ...,
        description=(
            "Absolute path to the Obsidian vault root. Memories will be written "
            "under ``<vault_path>/<memory_subdir>``."
        ),
    )
    memory_subdir: str = Field(
        default="04-Archive/99-obsidian-system/99-AI/memory",
        description="Subdir relative to ``vault_path`` where .md memory files live.",
    )
    state_dir: Path = Field(
        default_factory=lambda: Path.home() / ".local" / "share" / "mem-vault",
        description="Local non-vault state (Qdrant collection, history sqlite, logs).",
    )
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama HTTP endpoint used for both LLM extraction and embeddings.",
    )
    llm_model: str = Field(
        default="qwen2.5:3b",
        description=(
            "Ollama model used to extract/dedupe facts when auto_extract=True. "
            "qwen2.5:3b is the conservative default (fast, ~2GB RAM). Bump to "
            "qwen2.5:7b or larger via MEM_VAULT_LLM_MODEL if you have headroom."
        ),
    )
    embedder_model: str = Field(
        default="bge-m3:latest",
        description="Ollama embedding model. bge-m3 = 1024 dims, multilingual.",
    )
    embedder_dims: int = Field(
        default=1024,
        description="Embedding dimensionality. Must match the embedder model.",
    )
    qdrant_collection: str = Field(
        default="mem_vault",
        description="Qdrant collection name. Useful to namespace per-agent or per-project.",
    )
    user_id: str = Field(
        default="default",
        description="Default user_id passed to mem0 when the caller doesn't override it.",
    )
    agent_id: str | None = Field(
        default=None,
        description="Optional agent_id stamped on every memory (e.g. 'devin', 'claude-code').",
    )
    auto_extract_default: bool = Field(
        default=False,
        description="Whether memory_save uses the LLM by default. Tools can override per-call.",
    )

    @field_validator("vault_path", "state_dir", mode="before")
    @classmethod
    def _expand(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve() if v else v

    @property
    def memory_dir(self) -> Path:
        return self.vault_path / self.memory_subdir

    @property
    def qdrant_path(self) -> Path:
        return self.state_dir / "qdrant"

    @property
    def history_db(self) -> Path:
        return self.state_dir / "history.db"


def _resolve_vault_path(raw: str | None) -> Path | None:
    """Try a few well-known iCloud-Obsidian paths if the user didn't set one."""
    if raw:
        return Path(raw).expanduser()
    home = Path.home()
    candidates = [
        home / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes",
        home / "Obsidian",
        home / "Documents/Obsidian",
        home / "Notes",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def load_config(config_path: Path | None = None) -> Config:
    """Build a Config from env vars + optional TOML file.

    Args:
        config_path: explicit path to a config TOML. Falls back to
            ``$XDG_CONFIG_HOME/mem-vault/config.toml``.

    Raises:
        ValueError: if no vault_path can be resolved.
    """
    xdg = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    default_toml = xdg / "mem-vault" / "config.toml"
    cfg_file = config_path or default_toml
    file_data = _load_toml(cfg_file)

    env_vault = os.environ.get("MEM_VAULT_PATH") or os.environ.get("OBSIDIAN_VAULT_PATH")
    vault_path = _resolve_vault_path(env_vault or file_data.get("vault_path"))
    if vault_path is None:
        raise ValueError(
            "No vault_path configured. Set MEM_VAULT_PATH (or OBSIDIAN_VAULT_PATH) "
            "env var, or create ~/.config/mem-vault/config.toml with `vault_path = \"...\"`."
        )

    merged: dict = {**file_data, "vault_path": str(vault_path)}

    env_map = {
        "MEM_VAULT_MEMORY_SUBDIR": "memory_subdir",
        "MEM_VAULT_STATE_DIR": "state_dir",
        "MEM_VAULT_OLLAMA_HOST": "ollama_host",
        "MEM_VAULT_LLM_MODEL": "llm_model",
        "MEM_VAULT_EMBEDDER_MODEL": "embedder_model",
        "MEM_VAULT_EMBEDDER_DIMS": "embedder_dims",
        "MEM_VAULT_COLLECTION": "qdrant_collection",
        "MEM_VAULT_USER_ID": "user_id",
        "MEM_VAULT_AGENT_ID": "agent_id",
        "MEM_VAULT_AUTO_EXTRACT": "auto_extract_default",
    }
    for env_var, field in env_map.items():
        if env_var in os.environ:
            value: str | int | bool = os.environ[env_var]
            if field == "embedder_dims":
                value = int(value)
            elif field == "auto_extract_default":
                value = str(value).lower() in {"1", "true", "yes", "on"}
            merged[field] = value

    cfg = Config(**merged)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.qdrant_path.mkdir(parents=True, exist_ok=True)
    cfg.memory_dir.mkdir(parents=True, exist_ok=True)
    return cfg

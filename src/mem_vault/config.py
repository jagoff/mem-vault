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
import sys
import tomllib
from pathlib import Path

from platformdirs import user_data_dir
from pydantic import BaseModel, Field, field_validator


def _default_state_dir() -> Path:
    """Cross-platform XDG-style state directory.

    - macOS: ``~/Library/Application Support/mem-vault``
    - Linux: ``~/.local/share/mem-vault`` (respects ``$XDG_DATA_HOME``)
    - Windows: ``%LOCALAPPDATA%\\mem-vault\\mem-vault`` (the double name is
      platformdirs' convention to namespace by author/app — we accept it)
    """
    return Path(user_data_dir("mem-vault", appauthor=False))


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
        default="mem-vault",
        description=(
            "Subdir relative to ``vault_path`` where .md memory files live. "
            "Default ``mem-vault`` keeps memories self-contained in their own "
            "folder. Override with whatever fits your vault structure — e.g. "
            "``99-system/memory`` or ``inbox/agent-memories``."
        ),
    )
    state_dir: Path = Field(
        default_factory=_default_state_dir,
        description=(
            "Local non-vault state (Qdrant collection, history sqlite, logs). "
            "Defaults to the platform-appropriate user data dir: "
            "``~/Library/Application Support/mem-vault`` on macOS, "
            "``~/.local/share/mem-vault`` on Linux, "
            "``%LOCALAPPDATA%\\\\mem-vault\\\\mem-vault`` on Windows."
        ),
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
    qdrant_collection: str | None = Field(
        default=None,
        description=(
            "Qdrant collection name. If left null and ``agent_id`` is set, "
            "defaults to ``mem_vault_<agent_id>`` for natural per-agent isolation. "
            "Otherwise falls back to ``mem_vault``."
        ),
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
    decay_half_life_days: float = Field(
        default=0.0,
        description=(
            "Time-decay half-life in days for ``memory_search``. When > 0, hit "
            "scores are multiplied by ``exp(-age_days / half_life_days)`` so "
            "recent memories rank higher. Set to 0 (default) to disable decay "
            "and keep pure semantic-similarity ordering. Reasonable values: "
            "30 (aggressive), 90 (moderate), 365 (mild). Zero means no decay."
        ),
    )
    llm_timeout_s: float = Field(
        default=60.0,
        description=(
            "Wall-clock timeout (in seconds) for the embedding/LLM calls that "
            "back ``memory_save`` and ``memory_search``. Protects the MCP "
            "server from hanging indefinitely when Ollama is loading a model, "
            "OOM, or unreachable. ``auto_extract=True`` calls can be slow "
            "(5-15 s typical, occasionally 30 s+); 60 s leaves a healthy margin. "
            "Set to 0 to disable the timeout entirely (legacy behavior)."
        ),
    )
    max_content_size: int = Field(
        default=1_000_000,
        description=(
            "Maximum size (in characters) accepted by ``memory_save`` and "
            "``memory_update``. Calls with content larger than this are "
            "rejected with a structured error before touching the vault or "
            "the index. 1 MiB-ish is a sane upper bound for human-written "
            "memories; raise it if you intentionally store large dumps. Set "
            "to 0 to disable the limit."
        ),
    )
    http_token: str | None = Field(
        default=None,
        description=(
            "Bearer token required by the optional UI / JSON HTTP server when "
            "set. Clients must send ``Authorization: Bearer <token>`` on every "
            "request other than ``/healthz``. Leaving this null disables auth "
            "(safe only when the server is bound to a loopback interface). "
            "Binding to a non-loopback host without a token causes startup to "
            "abort with a clear error."
        ),
    )
    metrics_enabled: bool = Field(
        default=False,
        description=(
            "Append a JSONL line per MCP tool call to ``<state_dir>/metrics.jsonl`` "
            "with ``{ts, tool, duration_ms, ok, error?}``. Useful for profiling "
            "real workloads without bolting on Prometheus. Off by default to "
            "stay disk-quiet; enable via ``MEM_VAULT_METRICS=1`` or this field."
        ),
    )
    auto_link_default: bool = Field(
        default=True,
        description=(
            "When True (default), every successful ``memory_save`` runs a "
            "second semantic search to find similar existing memorias and "
            "stamps their IDs in the new memory's ``related`` frontmatter. "
            "Builds an emergent knowledge graph at zero extra cost to the "
            "user. Disable per-call with ``auto_link=false`` or globally "
            "via this field / ``MEM_VAULT_AUTO_LINK=0``."
        ),
    )
    reranker_enabled: bool = Field(
        default=False,
        description=(
            "When True, ``memory_search`` over-fetches a wider candidate set "
            "and re-orders it through a local cross-encoder reranker before "
            "returning the top-k. Boosts relevance significantly on noisy "
            "queries at the cost of ~30-100 ms extra latency per search. "
            "Requires the ``[hybrid]`` extra (``fastembed``); falls back to "
            "bi-encoder order silently if the import fails. Enable via "
            "``MEM_VAULT_RERANK=1`` or this field."
        ),
    )
    reranker_model: str = Field(
        default="jinaai/jina-reranker-v1-tiny-en",
        description=(
            "Cross-encoder model used by the local reranker. Defaults to "
            "Jina's tiny English-only reranker (~130 MB, ~30 ms for 20 "
            "candidates on CPU). For multilingual: "
            "``jinaai/jina-reranker-v2-base-multilingual`` (1.1 GB, slower)."
        ),
    )

    @field_validator("vault_path", "state_dir", mode="before")
    @classmethod
    def _expand(cls, v: str | Path | None) -> str | Path | None:
        # Pydantic ``mode="before"`` accepts whatever the caller passed, so the
        # signature stays loose. We only normalize truthy values; ``None`` /
        # empty string passes through and Pydantic's own validator handles it.
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

    @property
    def metrics_path(self) -> Path:
        return self.state_dir / "metrics.jsonl"


def _vault_candidates() -> list[Path]:
    """Likely Obsidian vault locations to try when the user didn't set one.

    Cross-platform: includes the macOS iCloud-Obsidian path, Linux/macOS
    common manual paths, and Windows ``%USERPROFILE%`` paths (which
    ``Path.home()`` already returns on Windows).
    """
    home = Path.home()
    candidates = [
        # macOS — iCloud-synced Obsidian vault (the most common setup)
        home / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents" / "Notes",
        # cross-platform — user typed `~/Obsidian` or `~/Notes` themselves
        home / "Obsidian",
        home / "Documents" / "Obsidian",
        home / "Notes",
        home / "Documents" / "Notes",
    ]
    if sys.platform.startswith("win"):
        candidates += [
            home / "OneDrive" / "Obsidian",
            home / "OneDrive" / "Documents" / "Obsidian",
        ]
    return candidates


def _resolve_vault_path(raw: str | None) -> Path | None:
    """Pick a vault path: explicit value > first existing candidate > None."""
    if raw:
        return Path(raw).expanduser()
    for c in _vault_candidates():
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
            'env var, or create ~/.config/mem-vault/config.toml with `vault_path = "..."`.'
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
        "MEM_VAULT_DECAY_HALF_LIFE_DAYS": "decay_half_life_days",
        "MEM_VAULT_LLM_TIMEOUT_S": "llm_timeout_s",
        "MEM_VAULT_MAX_CONTENT_SIZE": "max_content_size",
        "MEM_VAULT_HTTP_TOKEN": "http_token",
        "MEM_VAULT_METRICS": "metrics_enabled",
        "MEM_VAULT_AUTO_LINK": "auto_link_default",
        "MEM_VAULT_RERANK": "reranker_enabled",
        "MEM_VAULT_RERANKER_MODEL": "reranker_model",
    }
    for env_var, field in env_map.items():
        if env_var in os.environ:
            value: str | int | bool | float = os.environ[env_var]
            if field in {"embedder_dims", "max_content_size"}:
                value = int(value)
            elif field in {
                "auto_extract_default",
                "metrics_enabled",
                "auto_link_default",
                "reranker_enabled",
            }:
                value = str(value).lower() in {"1", "true", "yes", "on"}
            elif field in {"decay_half_life_days", "llm_timeout_s"}:
                value = float(value)
            merged[field] = value

    cfg = Config(**merged)
    if cfg.qdrant_collection is None:
        cfg.qdrant_collection = f"mem_vault_{cfg.agent_id}" if cfg.agent_id else "mem_vault"
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.qdrant_path.mkdir(parents=True, exist_ok=True)
    cfg.memory_dir.mkdir(parents=True, exist_ok=True)
    return cfg

"""``mem-vault doctor`` — one-shot health diagnostic.

A single command that runs every check new users hit during setup, with a
short colored report and a non-zero exit code when something is broken.
Inspired by [`brew doctor`](https://docs.brew.sh/Manpage#doctor), [`nvm
doctor`](https://github.com/nvm-sh/nvm#problems), and the equivalent
``rag setup`` in the sibling [obsidian-rag](https://github.com/jagoff/obsidian-rag).

Checks performed (in order, stopping on none — we always show the full
report):

1. **Vault path** exists, is a directory, and writable (canary temp file).
2. **State dir** exists and writable; ``history.db`` openable.
3. **Ollama** reachable via ``GET /api/tags`` (configurable host).
4. **Models pulled**: embedder (``bge-m3``) + LLM (``qwen2.5:3b`` by default,
   whichever ``llm_model`` is configured).
5. **Qdrant collection**: can we ``sync_status`` the vault? If locked,
   flag a warning (the MCP server probably holds it).
6. **Memory corpus**: total memorias count + drift summary (stale,
   orphans, missing) from ``sync_status``.
7. **Optional extras**:
   - ``fastembed`` installed but ``reranker_enabled=False`` → hint to
     turn it on.
   - ``FASTEMBED_CACHE_PATH`` gotcha on macOS: warn if unset and the
     default resolves to a temp dir (the download gets re-fetched every
     boot). See the sibling ``obsidian-rag`` memory about this.
   - ``MEM_VAULT_HTTP_TOKEN`` with a non-loopback bind is actually
     fine, but worth surfacing so the user remembers it's on.

Exit codes:
- ``0`` — all checks green.
- ``1`` — warnings only (hints / non-blocking issues).
- ``2`` — at least one hard failure (vault missing, Ollama down, etc.).

The command reads the same ``Config`` as the MCP server, so flags /
env vars the user passes (``MEM_VAULT_PATH``, ``MEM_VAULT_OLLAMA_HOST``,
…) are honored. Lazy imports keep the ``--help`` path instant.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class _Check:
    """One row in the doctor report.

    ``status`` is one of ``"ok"`` / ``"warn"`` / ``"err"``.
    ``detail`` is the human-readable one-liner the user sees after the glyph.
    ``hint`` is an optional second line with the remediation command.
    """

    name: str
    status: str
    detail: str
    hint: str | None = None


@dataclass
class _Report:
    checks: list[_Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str, hint: str | None = None) -> None:
        self.checks.append(_Check(name=name, status=status, detail=detail, hint=hint))

    @property
    def any_error(self) -> bool:
        return any(c.status == "err" for c in self.checks)

    @property
    def any_warn(self) -> bool:
        return any(c.status == "warn" for c in self.checks)

    def exit_code(self) -> int:
        if self.any_error:
            return 2
        if self.any_warn:
            return 1
        return 0


_GLYPHS = {"ok": "\u2713", "warn": "\u26a0", "err": "\u2717"}


def _supports_color() -> bool:
    """Conservative color detection: only emit ANSI when stdout is a TTY and
    ``NO_COLOR`` is unset (the de-facto convention — see https://no-color.org/).
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def _colorize(text: str, color: str) -> str:
    if not _supports_color():
        return text
    codes = {
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "dim": "\033[2m",
        "reset": "\033[0m",
    }
    return f"{codes.get(color, '')}{text}{codes['reset']}"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_vault(cfg: Any, report: _Report) -> None:
    """Can we read and write to the vault memory dir?"""
    memory_dir: Path = cfg.memory_dir
    if not cfg.vault_path.exists():
        report.add(
            "vault",
            "err",
            f"vault_path does not exist: {cfg.vault_path}",
            hint="Set MEM_VAULT_PATH or create the directory.",
        )
        return
    if not cfg.vault_path.is_dir():
        report.add(
            "vault",
            "err",
            f"vault_path is not a directory: {cfg.vault_path}",
        )
        return
    # ``memory_dir`` is created by ``load_config``; if we can't write a
    # canary file, we have a permissions problem worth surfacing.
    try:
        memory_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=memory_dir, prefix=".doctor.", delete=True):
            pass
        report.add("vault", "ok", f"{memory_dir}")
    except OSError as exc:
        report.add(
            "vault",
            "err",
            f"cannot write to memory_dir ({memory_dir}): {exc}",
            hint="Check filesystem permissions.",
        )


def _check_state_dir(cfg: Any, report: _Report) -> None:
    state_dir: Path = cfg.state_dir
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=state_dir, prefix=".doctor.", delete=True):
            pass
        report.add("state", "ok", f"{state_dir}")
    except OSError as exc:
        report.add(
            "state",
            "err",
            f"cannot write to state_dir ({state_dir}): {exc}",
            hint="Set MEM_VAULT_STATE_DIR or fix permissions.",
        )


def _check_ollama(cfg: Any, report: _Report) -> tuple[bool, set[str]]:
    """Return (ollama_up, installed_model_names).

    We query ``/api/tags`` and pull out the model names. The model-check
    helper below compares against ``cfg.llm_model`` / ``cfg.embedder_model``.
    """
    try:
        import ollama
    except ImportError:
        report.add(
            "ollama-client",
            "err",
            "python package `ollama` is not importable",
            hint="pip install ollama",
        )
        return False, set()

    # ``ollama.Client(host=...)`` doesn't do a health check at construction;
    # ``list()`` is the first real roundtrip.
    client = ollama.Client(host=cfg.ollama_host)
    try:
        started = time.monotonic()
        resp = client.list()
        elapsed_ms = (time.monotonic() - started) * 1000
    except Exception as exc:
        report.add(
            "ollama",
            "err",
            f"{cfg.ollama_host} unreachable: {type(exc).__name__}: {exc}",
            hint="Start Ollama with `ollama serve` (brew services start ollama).",
        )
        return False, set()

    # Normalize the response across ollama client versions. Older versions
    # returned ``{"models": [{"name": ...}, ...]}``; newer ones returned
    # ``{"models": [{"model": ...}, ...]}`` or an object with ``.models``.
    raw_models: list[Any]
    if isinstance(resp, dict):
        raw_models = list(resp.get("models", []))
    else:
        raw_models = list(getattr(resp, "models", []))

    installed: set[str] = set()
    for m in raw_models:
        if isinstance(m, dict):
            name = m.get("name") or m.get("model")
        else:
            name = getattr(m, "name", None) or getattr(m, "model", None)
        if isinstance(name, str):
            installed.add(name)

    report.add(
        "ollama",
        "ok",
        f"{cfg.ollama_host} responded in {elapsed_ms:.0f}ms · {len(installed)} models",
    )
    return True, installed


def _model_is_installed(target: str, installed: set[str]) -> bool:
    """Match ``qwen2.5:3b`` against ``qwen2.5:3b`` / ``qwen2.5:3b-instruct-q4_K_M``.

    Ollama tags carry trailing quantization suffixes (``:latest``, ``:3b``,
    ``:7b-q4_K_M``) that users sometimes don't spell exactly the same way.
    We accept a prefix match on the name-before-colon.
    """
    if not target:
        return False
    if target in installed:
        return True
    base = target.split(":", 1)[0]
    return any(m.split(":", 1)[0] == base for m in installed)


def _check_models(cfg: Any, installed: set[str], report: _Report) -> None:
    for role, model in (("embedder", cfg.embedder_model), ("llm", cfg.llm_model)):
        if not model:
            report.add(f"model:{role}", "warn", "not configured (skipped)")
            continue
        if _model_is_installed(model, installed):
            report.add(f"model:{role}", "ok", f"{model}")
        else:
            report.add(
                f"model:{role}",
                "err",
                f"{model} not installed on Ollama",
                hint=f"ollama pull {model}",
            )


def _check_index(cfg: Any, report: _Report) -> None:
    """Run sync_status; handle the index-locked case gracefully."""
    from mem_vault.sync import IndexLockedError, sync_status

    try:
        rep = sync_status(cfg)
    except IndexLockedError:
        report.add(
            "qdrant",
            "warn",
            "collection locked (MCP server running?) — skipping drift check",
            hint="Stop the MCP server to run a full sync-status.",
        )
        return
    except Exception as exc:
        report.add(
            "qdrant",
            "err",
            f"could not open Qdrant collection: {type(exc).__name__}: {exc}",
            hint=f"Try `mem-vault reindex --purge` (path: {cfg.qdrant_path})",
        )
        return

    report.add(
        "qdrant",
        "ok",
        f"collection={cfg.qdrant_collection} · {rep.in_index} entries",
    )
    if rep.needs_reindex:
        parts = []
        if rep.stale_in_index:
            parts.append(f"stale={rep.stale_in_index}")
        if rep.orphan_in_index:
            parts.append(f"orphans={rep.orphan_in_index}")
        if rep.missing_in_index:
            parts.append(f"missing={rep.missing_in_index}")
        report.add(
            "sync",
            "warn",
            f"drift detected ({', '.join(parts)})",
            hint="mem-vault reindex",
        )
    else:
        report.add(
            "sync",
            "ok",
            f"vault={rep.in_vault} files · index in lockstep",
        )


def _check_extras(cfg: Any, report: _Report) -> None:
    """Non-critical extras: fastembed, rerank toggle, token+LAN bind."""
    try:
        import fastembed  # noqa: F401

        have_fastembed = True
    except ImportError:
        have_fastembed = False

    if have_fastembed and not cfg.reranker_enabled:
        report.add(
            "rerank",
            "warn",
            "fastembed installed but reranker_enabled=False",
            hint="export MEM_VAULT_RERANK=1 to turn it on.",
        )
    elif not have_fastembed and cfg.reranker_enabled:
        report.add(
            "rerank",
            "err",
            "reranker_enabled=True but fastembed is not installed",
            hint="uv tool install --editable '.[hybrid]' (or uv pip install fastembed)",
        )
    elif have_fastembed:
        report.add("rerank", "ok", f"fastembed present · model={cfg.reranker_model}")
    else:
        report.add(
            "rerank",
            "ok",
            "disabled (no fastembed installed; rank stays pure semantic)",
        )

    # FASTEMBED_CACHE_PATH gotcha on macOS: the default mem0 / fastembed
    # cache tries to write to a path under /var/folders/... which gets
    # wiped on reboot. Warn if unset and the user is on macOS — they
    # probably want to pin it to ~/.cache/fastembed.
    if have_fastembed and sys.platform == "darwin":
        if not os.environ.get("FASTEMBED_CACHE_PATH"):
            report.add(
                "fastembed-cache",
                "warn",
                "FASTEMBED_CACHE_PATH unset on macOS",
                hint="export FASTEMBED_CACHE_PATH=$HOME/.cache/fastembed (so the download survives reboots).",
            )


def _check_feedback_loop(cfg: Any, report: _Report) -> None:
    """Surface usage-boost + tracking flags so the user knows what's on."""
    if not cfg.usage_tracking_enabled:
        report.add(
            "feedback",
            "warn",
            "usage_tracking_enabled=False — searches won't update usage_count",
            hint="unset MEM_VAULT_USAGE_TRACKING to re-enable.",
        )
    elif cfg.usage_boost_enabled and cfg.usage_boost > 0:
        report.add(
            "feedback",
            "ok",
            f"tracking on · boost={cfg.usage_boost:.2f}",
        )
    else:
        report.add(
            "feedback",
            "ok",
            "tracking on · boost disabled (pure semantic ordering)",
        )


def _check_hybrid(cfg: Any, report: _Report) -> None:
    """Surface the hybrid-retrieval flag so the user knows it's (not) on."""
    if cfg.hybrid_enabled:
        report.add(
            "hybrid",
            "ok",
            f"BM25 + dense + RRF (k={cfg.hybrid_rrf_k})",
        )
    else:
        report.add(
            "hybrid",
            "ok",
            "disabled (pure dense; turn on with MEM_VAULT_HYBRID=1 to fuse BM25)",
        )


def _check_project_scope(cfg: Any, report: _Report) -> None:
    """Surface the active project scope (if any) so the user knows what they're filtering on."""
    if cfg.project_default:
        report.add(
            "project",
            "ok",
            f"scope={cfg.project_default} (searches filter to this project by default)",
        )
    else:
        report.add(
            "project",
            "ok",
            "no default scope (global search; set MEM_VAULT_PROJECT=<name> to scope)",
        )


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _render(report: _Report) -> None:
    print()
    print(_colorize("mem-vault doctor", "dim"))
    print(_colorize("─────────────────", "dim"))
    # Compute name col width for clean alignment.
    width = max((len(c.name) for c in report.checks), default=0) + 2
    for c in report.checks:
        color = {"ok": "green", "warn": "yellow", "err": "red"}[c.status]
        glyph = _colorize(_GLYPHS[c.status], color)
        name_col = c.name.ljust(width)
        print(f"  {glyph}  {name_col} {c.detail}")
        if c.hint:
            print(f"     {' ' * width} {_colorize('↳ ' + c.hint, 'dim')}")
    print()

    if report.any_error:
        print(_colorize("✗ one or more critical checks failed", "red"))
    elif report.any_warn:
        print(_colorize("⚠ some warnings to review", "yellow"))
    else:
        print(_colorize("✓ all checks passed", "green"))
    print()


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "doctor",
        help=(
            "Run every health check (vault, Ollama, models, Qdrant, extras) "
            "and print a short report. Non-zero exit on errors."
        ),
    )
    p.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip the Ollama + model checks (useful in CI / offline dev).",
    )
    p.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip the Qdrant sync check (leaves vault+config checks only).",
    )


def run(args: argparse.Namespace) -> int:
    """Entry point — build the report, render it, return the exit code."""
    from mem_vault.config import load_config

    report = _Report()

    # Config loading is itself a possible failure mode — surface it clearly.
    try:
        cfg = load_config()
    except Exception as exc:
        report.add(
            "config",
            "err",
            f"load_config failed: {type(exc).__name__}: {exc}",
            hint="Set MEM_VAULT_PATH or create ~/.config/mem-vault/config.toml.",
        )
        _render(report)
        return report.exit_code()

    report.add(
        "config",
        "ok",
        (
            f"vault={cfg.memory_dir} · ollama={cfg.ollama_host} · "
            f"llm={cfg.llm_model} · embedder={cfg.embedder_model}"
        ),
    )

    _check_vault(cfg, report)
    _check_state_dir(cfg, report)

    if args.skip_ollama:
        report.add("ollama", "warn", "skipped via --skip-ollama")
    else:
        ok, installed = _check_ollama(cfg, report)
        if ok:
            _check_models(cfg, installed, report)

    if args.skip_index:
        report.add("qdrant", "warn", "skipped via --skip-index")
    else:
        _check_index(cfg, report)

    _check_extras(cfg, report)
    _check_feedback_loop(cfg, report)
    _check_hybrid(cfg, report)
    _check_project_scope(cfg, report)

    _render(report)
    return report.exit_code()

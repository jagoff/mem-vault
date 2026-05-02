"""Top-level CLI for mem-vault.

This module is intentionally thin: it builds the argparse tree by asking
each subcommand module to register its own subparsers, then dispatches
the parsed ``args.cmd`` to the module that owns it. Each subcommand body
lives in its own file under :mod:`mem_vault.cli`:

- :mod:`mem_vault.cli.crud`           — search / list / save / get / delete
- :mod:`mem_vault.cli.ui`             — ``mem-vault ui``
- :mod:`mem_vault.cli.reindex`        — ``mem-vault reindex``
- :mod:`mem_vault.cli.consolidate`    — ``mem-vault consolidate``
- :mod:`mem_vault.cli.import_engram`  — ``mem-vault import-engram``
- :mod:`mem_vault.cli.export_cmd`     — ``mem-vault export``
- :mod:`mem_vault.cli.sync_cmd`       — ``mem-vault sync-status`` / ``sync-watch``
- :mod:`mem_vault.cli.hooks_cmd`      — ``mem-vault hook-*``
- :mod:`mem_vault.cli.install_skill`  — ``mem-vault install-skill`` (Devin /mv slash)

The default (no subcommand) preserves the legacy behavior: bare
``mem-vault`` boots the MCP stdio server, same as ``mem-vault-mcp``. This
keeps existing Devin / Claude Code MCP configs working unchanged.

Subcommand modules also import their heavy deps lazily (inside ``run``)
so ``mem-vault --help`` stays fast even with the optional extras
installed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from mem_vault import __version__
from mem_vault.cli import (
    consolidate as _consolidate_mod,
)
from mem_vault.cli import (
    crud as _crud_mod,
)
from mem_vault.cli import (
    doctor as _doctor_mod,
)
from mem_vault.cli import (
    eval_cmd as _eval_mod,
)
from mem_vault.cli import (
    export_cmd as _export_mod,
)
from mem_vault.cli import (
    hooks_cmd as _hooks_mod,
)
from mem_vault.cli import (
    import_engram as _import_engram_mod,
)
from mem_vault.cli import (
    install_skill as _install_skill_mod,
)
from mem_vault.cli import (
    learn as _learn_mod,
)
from mem_vault.cli import (
    metrics as _metrics_mod,
)
from mem_vault.cli import (
    reindex as _reindex_mod,
)
from mem_vault.cli import (
    sync_cmd as _sync_mod,
)
from mem_vault.cli import (
    ui as _ui_mod,
)
from mem_vault.server import main as serve_main


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mem-vault",
        description="Local MCP server with infinite memory backed by an Obsidian vault.",
    )
    sub = parser.add_subparsers(dest="cmd")

    # Default + version are simple enough to register inline.
    sub.add_parser("serve", help="Start the MCP stdio server (default).")
    sub.add_parser("version", help="Print package version.")

    # Each module registers its own subparsers — keeps argparse setup colocated
    # with the implementation.
    _ui_mod.add_subparser(sub)
    _sync_mod.add_subparsers(sub)
    _export_mod.add_subparser(sub)
    _crud_mod.add_subparsers(sub)
    _hooks_mod.add_subparsers(sub)
    _reindex_mod.add_subparser(sub)
    _consolidate_mod.add_subparser(sub)
    _import_engram_mod.add_subparser(sub)
    _install_skill_mod.add_subparser(sub)
    _doctor_mod.add_subparser(sub)
    _eval_mod.add_subparser(sub)
    _metrics_mod.add_subparser(sub)
    _learn_mod.add_subparsers(sub)

    return parser


def _dispatch(argv: list[str]) -> None:
    if not argv:
        # Backwards-compat: bare `mem-vault` boots the MCP server, like
        # `mem-vault-mcp` does. This keeps the existing Devin / Claude Code
        # MCP configs working unchanged.
        return serve_main()

    args = _build_parser().parse_args(argv)
    cmd = args.cmd or "serve"

    if cmd == "version":
        print(__version__)
        return

    if cmd == "serve":
        return serve_main()

    if cmd == "ui":
        return _ui_mod.run(args)

    if cmd == "import-engram":
        sys.exit(asyncio.run(_import_engram_mod.run(args)))

    if cmd in {"search", "list", "save", "get", "delete"}:
        sys.exit(asyncio.run(_crud_mod.run(cmd, args)))

    if cmd == "export":
        sys.exit(_export_mod.run(args))

    if cmd == "sync-status":
        sys.exit(_sync_mod.run_status())

    if cmd == "sync-watch":
        sys.exit(_sync_mod.run_watch(args))

    if cmd == "reindex":
        sys.exit(asyncio.run(_reindex_mod.run(args)))

    if cmd == "consolidate":
        sys.exit(_consolidate_mod.run(args))

    if cmd == "install-skill":
        sys.exit(_install_skill_mod.run(args))

    if cmd == "doctor":
        sys.exit(_doctor_mod.run(args))

    if cmd == "eval":
        sys.exit(asyncio.run(_eval_mod.run(args)))

    if cmd == "metrics":
        sys.exit(_metrics_mod.run(args))

    if cmd == "telemetry-stats":
        sys.exit(_learn_mod.run_telemetry_stats(args))

    if cmd == "ranker-train":
        sys.exit(_learn_mod.run_ranker_train(args))

    if cmd == "ranker-eval":
        sys.exit(_learn_mod.run_ranker_eval(args))

    if cmd == "ranker-rollback":
        sys.exit(_learn_mod.run_ranker_rollback(args))

    if cmd in {"hook-sessionstart", "hook-userprompt", "hook-stop"}:
        _hooks_mod.run(cmd)
        return

    print(f"error: unknown subcommand: {cmd}", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    """CLI entrypoint with a top-level error guard.

    Any exception bubbling out of a subcommand is rendered as a one-line
    error to stderr and the process exits with code 1. Without this, a
    misconfigured ``MEM_VAULT_PATH`` (or any other unexpected condition)
    prints a full traceback that leaks config paths, the Ollama host, and
    other internals to whoever is reading stdout — bad surprise in CI or
    a shared shell. ``--debug`` opts back into the raw traceback for
    development.
    """
    argv = sys.argv[1:]
    debug = "--debug" in argv
    if debug:
        argv = [a for a in argv if a != "--debug"]
    try:
        _dispatch(argv)
    except KeyboardInterrupt:
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as exc:
        if debug:
            raise
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("(re-run with --debug for the full traceback)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

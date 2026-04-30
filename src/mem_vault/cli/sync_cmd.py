"""``sync-status`` and ``sync-watch`` — keep the Qdrant index in lockstep with the vault.

``sync-status`` is a one-shot diff between the markdown source of truth
and the Qdrant collection (orphans, stale entries, missing entries).

``sync-watch`` runs forever, watching the vault dir with ``watchdog`` and
re-embedding each memory when its file changes (debounced).
"""

from __future__ import annotations

import argparse
import logging
import sys


def add_subparsers(sub: argparse._SubParsersAction) -> None:
    sub.add_parser(
        "sync-status",
        help=(
            "Diff between vault files and the local Qdrant index. Tells you if a reindex is needed."
        ),
    )
    p_watch = sub.add_parser(
        "sync-watch",
        help="Watch the vault dir and reindex memories as their .md files change.",
    )
    p_watch.add_argument(
        "--debounce",
        type=float,
        default=1.0,
        help="Wait N seconds after the last write before reindexing (default 1.0).",
    )


def run_status() -> int:
    from mem_vault.config import load_config
    from mem_vault.sync import IndexLockedError, sync_status

    config = load_config()
    print(f"sync-status · vault={config.memory_dir}")
    print(f"            · collection={config.qdrant_collection}")
    try:
        report = sync_status(config)
    except IndexLockedError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "\nThe MCP server (`mem-vault-mcp`) holds an exclusive lock on the local "
            "Qdrant DB. Stop your agent (Claude Code / Devin) or kill the process and "
            "retry. The vault files themselves are unaffected.",
            file=sys.stderr,
        )
        return 2
    for line in report.to_lines():
        print(line)
    if report.needs_reindex:
        print(
            "\nIndex is out of sync with the vault. Run "
            "`mem-vault reindex` to rebuild, or `mem-vault sync-watch` "
            "to keep them in lockstep automatically (also requires the MCP server to be off)."
        )
        return 1
    print("\nIndex is in sync with the vault.")
    return 0


def run_watch(args: argparse.Namespace) -> int:
    from mem_vault.config import load_config
    from mem_vault.sync import VaultWatcher

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        stream=sys.stderr,
    )
    config = load_config()
    print(f"sync-watch · watching {config.memory_dir}", file=sys.stderr)
    print(f"           · debounce={args.debounce}s", file=sys.stderr)
    watcher = VaultWatcher(config, debounce_seconds=args.debounce)
    try:
        watcher.run()
    except KeyboardInterrupt:
        watcher.stop()
    return 0

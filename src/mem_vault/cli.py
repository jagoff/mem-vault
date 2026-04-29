"""Top-level CLI for mem-vault.

Subcommands:
- ``serve`` (default) — start the MCP stdio server. Same as ``mem-vault-mcp``.
- ``import-engram`` — bulk-import memories from an ``engram export`` JSON file.
- ``hook-sessionstart`` — SessionStart lifecycle hook (reads stdin, prints JSON).
- ``hook-stop`` — Stop lifecycle hook (logs to ~/.local/share/mem-vault/sessions.log).
- ``version`` — print package version.

Usage:
    mem-vault                            # equivalent to `mem-vault serve`
    mem-vault serve
    mem-vault import-engram /tmp/engram-export.json --agent-id engram
    mem-vault hook-sessionstart < event.json
    mem-vault hook-stop < event.json
    mem-vault version
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mem_vault import __version__
from mem_vault.config import load_config
from mem_vault.server import MemVaultService, main as serve_main


_ENGRAM_TYPE_MAP = {
    "setup": "fact",
    "config": "fact",
    "decision": "decision",
    "bug": "bug",
    "todo": "todo",
    "feedback": "feedback",
    "preference": "preference",
    "fact": "fact",
    "note": "note",
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mem-vault",
        description="Local MCP server with infinite memory backed by an Obsidian vault.",
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("serve", help="Start the MCP stdio server (default).")
    sub.add_parser("version", help="Print package version.")
    sub.add_parser(
        "hook-sessionstart",
        help="SessionStart lifecycle hook: inject preferences into agent context.",
    )
    sub.add_parser(
        "hook-stop",
        help="Stop lifecycle hook: append a line to the audit log. Never blocks.",
    )

    p_import = sub.add_parser(
        "import-engram",
        help="Bulk-import memories from an `engram export` JSON file.",
    )
    p_import.add_argument(
        "export_path",
        type=Path,
        help="Path to the engram-export.json file (run `engram export` to produce one).",
    )
    p_import.add_argument(
        "--user-id",
        default=None,
        help="user_id stamped on every imported memory (defaults to config.user_id).",
    )
    p_import.add_argument(
        "--agent-id",
        default="engram",
        help="agent_id stamped on every imported memory (default: 'engram').",
    )
    p_import.add_argument(
        "--type-default",
        default="fact",
        help="Fallback type when the engram observation type doesn't map cleanly.",
    )
    p_import.add_argument(
        "--auto-extract",
        action="store_true",
        help="Run the LLM extractor on each observation (slower, dedupes against vault).",
    )
    p_import.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without writing anything.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# import-engram
# ---------------------------------------------------------------------------


def _engram_to_memory(
    obs: dict[str, Any],
    *,
    type_default: str,
    user_id: str,
    agent_id: str,
) -> dict[str, Any]:
    """Map one engram observation row → mem-vault save kwargs."""
    raw_type = (obs.get("type") or "").lower()
    mapped_type = _ENGRAM_TYPE_MAP.get(raw_type, type_default)

    title = obs.get("title") or f"engram-obs-{obs.get('id') or obs.get('sync_id')}"
    content = obs.get("content") or title
    project = obs.get("project")
    topic_key = obs.get("topic_key")
    scope = obs.get("scope")

    tags: list[str] = []
    if project:
        tags.append(f"project:{project}")
    if scope:
        tags.append(f"scope:{scope}")
    if topic_key:
        # topic_key looks like "finance/source-separation" — split into parts
        for piece in str(topic_key).split("/"):
            piece = piece.strip()
            if piece and piece not in tags:
                tags.append(piece)
    tags.append("source:engram")

    return {
        "content": content,
        "title": title,
        "type": mapped_type,
        "tags": tags,
        "user_id": user_id,
        "agent_id": agent_id,
        "auto_extract": False,
    }


async def _import_engram(args: argparse.Namespace) -> int:
    if not args.export_path.exists():
        print(f"error: file not found: {args.export_path}", file=sys.stderr)
        return 2
    try:
        data = json.loads(args.export_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in {args.export_path}: {exc}", file=sys.stderr)
        return 2

    observations = data.get("observations") or []
    if not isinstance(observations, list):
        print("error: expected 'observations' to be a list", file=sys.stderr)
        return 2

    config = load_config()
    user_id = args.user_id or config.user_id
    agent_id = args.agent_id

    print(
        f"engram-import: {len(observations)} observations from {args.export_path} "
        f"→ vault={config.memory_dir} user_id={user_id} agent_id={agent_id} "
        f"auto_extract={args.auto_extract} dry_run={args.dry_run}",
        flush=True,
    )

    if args.dry_run:
        for obs in observations:
            mapped = _engram_to_memory(
                obs, type_default=args.type_default, user_id=user_id, agent_id=agent_id
            )
            print(
                f"  would save · type={mapped['type']} · title={mapped['title'][:60]!r} "
                f"· tags={mapped['tags']}"
            )
        print("(dry-run — nothing written)")
        return 0

    service = MemVaultService(config)
    saved = 0
    skipped = 0
    errors = 0
    for obs in observations:
        kwargs = _engram_to_memory(
            obs, type_default=args.type_default, user_id=user_id, agent_id=agent_id
        )
        kwargs["auto_extract"] = args.auto_extract
        try:
            res = await service.save(kwargs)
        except Exception as exc:
            errors += 1
            print(f"  ERROR saving {kwargs['title'][:60]!r}: {exc}", file=sys.stderr)
            continue
        if not res.get("ok"):
            skipped += 1
            print(f"  skipped: {res.get('error')}", file=sys.stderr)
            continue
        saved += 1
        print(
            f"  saved [{saved:>3}/{len(observations)}] {res['memory']['id']} "
            f"(indexed={res.get('indexed')})"
        )

    print(f"engram-import done: saved={saved} skipped={skipped} errors={errors}")
    return 0 if errors == 0 else 1


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        # Backwards-compat: bare `mem-vault` boots the MCP server, like
        # `mem-vault-mcp` does. This keeps the existing Devin / Claude Code
        # MCP configs working unchanged.
        return serve_main()

    args = _parse_args(argv)
    cmd = args.cmd or "serve"

    if cmd == "version":
        print(__version__)
        return

    if cmd == "serve":
        return serve_main()

    if cmd == "import-engram":
        sys.exit(asyncio.run(_import_engram(args)))

    if cmd == "hook-sessionstart":
        from mem_vault.hooks import sessionstart  # local import to keep startup fast
        sessionstart.run()
        return

    if cmd == "hook-stop":
        from mem_vault.hooks import stop  # local import to keep startup fast
        stop.run()
        return

    print(f"error: unknown subcommand: {cmd}", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()

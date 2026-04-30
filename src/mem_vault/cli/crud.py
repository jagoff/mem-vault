"""``search`` / ``list`` / ``save`` / ``get`` / ``delete`` — shell wrappers.

These five subcommands hit the same ``MemVaultService`` methods the MCP
server exposes, but from the shell. They respect ``MEM_VAULT_REMOTE_URL``
so the same commands work both with the embedded Qdrant (default) and
against a long-lived web server (when the obsidian-rag server holds the
lock).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from mem_vault.cli._common import print_human_get, print_human_list, print_human_search


def add_subparsers(sub: argparse._SubParsersAction) -> None:
    """Register the five CRUD subparsers on the top-level parser."""
    p_search = sub.add_parser(
        "search",
        help="Semantic search across memories (same as memory_search MCP tool).",
    )
    p_search.add_argument("query", help="Natural-language query.")
    p_search.add_argument("-k", "--k", type=int, default=5)
    p_search.add_argument("--type", default=None, help="Filter by memory type.")
    p_search.add_argument("--threshold", type=float, default=0.1)
    p_search.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of the human-readable summary.",
    )

    p_list = sub.add_parser(
        "list",
        help="List memories sorted by mtime (same as memory_list MCP tool).",
    )
    p_list.add_argument("--type", default=None)
    p_list.add_argument("--tag", action="append", default=None, help="Repeatable.")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--json", action="store_true")

    p_save = sub.add_parser(
        "save",
        help=(
            "Save a new memory (same as memory_save MCP tool). "
            "Reads body from stdin if --content is omitted."
        ),
    )
    p_save.add_argument("--content", default=None, help="Memory body (default: stdin).")
    p_save.add_argument("--title", default=None)
    p_save.add_argument(
        "--type",
        default="note",
        choices=["feedback", "preference", "decision", "fact", "note", "bug", "todo"],
    )
    p_save.add_argument("--tag", action="append", default=None, help="Repeatable.")
    p_save.add_argument(
        "--visibility",
        default=None,
        help="'public' (default), 'private', or comma-separated agent ids.",
    )
    p_save.add_argument(
        "--auto-extract",
        action="store_true",
        help="Run the LLM to extract / dedupe canonical facts.",
    )
    p_save.add_argument("--json", action="store_true")

    p_get = sub.add_parser(
        "get",
        help="Read one memory by id (the file slug, e.g. 'feedback_local_free_stack').",
    )
    p_get.add_argument("id")
    p_get.add_argument("--json", action="store_true")

    p_delete = sub.add_parser(
        "delete",
        help="Delete a memory (file + index). Asks for confirmation unless --yes.",
    )
    p_delete.add_argument("id")
    p_delete.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt.",
    )


async def run(cmd: str, args: argparse.Namespace) -> int:
    """Dispatch search / list / save / get / delete from the shell.

    Respects ``MEM_VAULT_REMOTE_URL`` — when set, talks to the remote web
    server instead of opening Qdrant locally. That means the same commands
    work even when the obsidian-rag web server is holding the Qdrant lock.
    """
    from mem_vault.server import build_service

    service = build_service()

    if cmd == "search":
        payload = await service.search(
            {
                "query": args.query,
                "k": args.k,
                "type": args.type,
                "threshold": args.threshold,
            }
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print_human_search(payload)
        return 0 if payload.get("ok") else 1

    if cmd == "list":
        payload = await service.list_(
            {
                "type": args.type,
                "tags": args.tag,
                "limit": args.limit,
            }
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print_human_list(payload)
        return 0 if payload.get("ok") else 1

    if cmd == "save":
        content = args.content
        if content is None:
            if sys.stdin.isatty():
                print(
                    "error: pass --content '...' or pipe the body via stdin.",
                    file=sys.stderr,
                )
                return 2
            content = sys.stdin.read()
        if not content.strip():
            print("error: empty content.", file=sys.stderr)
            return 2

        visibility: Any = args.visibility
        if visibility and visibility not in {"public", "private"}:
            visibility = [v.strip() for v in visibility.split(",") if v.strip()]

        payload = await service.save(
            {
                "content": content,
                "title": args.title,
                "type": args.type,
                "tags": args.tag,
                "auto_extract": args.auto_extract,
                "visible_to": visibility,
            }
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            mem = payload.get("memory") or {}
            print(
                f"saved · id={mem.get('id')} · type={mem.get('type')} · "
                f"indexed={payload.get('indexed')}"
            )
            print(f"path: {payload.get('path')}")
        return 0 if payload.get("ok") else 1

    if cmd == "get":
        payload = await service.get({"id": args.id})
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print_human_get(payload)
        return 0 if payload.get("ok") else 1

    if cmd == "delete":
        if not args.yes:
            # Reject non-TTY without --yes outright. Otherwise ``input()``
            # either hangs forever (no EOF on the pipe) or returns immediately
            # with EOF and silently aborts — both confusing for CI / piped
            # invocations where the user can't actually type a response.
            if not sys.stdin.isatty():
                print(
                    "error: delete requires --yes when stdin is not a TTY (CI / piped input).",
                    file=sys.stderr,
                )
                return 2
            print(f"about to delete memory {args.id!r} (file + index entry).")
            ans = input("type 'yes' to confirm: ").strip().lower()
            if ans != "yes":
                print("aborted.")
                return 0
        payload = await service.delete({"id": args.id})
        if not payload.get("ok"):
            print(f"error: {payload.get('error')}", file=sys.stderr)
            return 1
        print(f"deleted · removed_index_entries={payload.get('deleted_index_entries')}")
        return 0

    return 2

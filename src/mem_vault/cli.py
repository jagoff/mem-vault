"""Top-level CLI for mem-vault.

Subcommands:
- ``serve`` (default) — start the MCP stdio server. Same as ``mem-vault-mcp``.
- ``import-engram`` — bulk-import memories from an ``engram export`` JSON file.
- ``hook-sessionstart`` — SessionStart lifecycle hook (reads stdin, prints JSON).
- ``hook-userprompt`` — UserPromptSubmit lifecycle hook (semantic search per prompt).
- ``hook-stop`` — Stop lifecycle hook (logs to ~/.local/share/mem-vault/sessions.log).
- ``version`` — print package version.

Usage:
    mem-vault                            # equivalent to `mem-vault serve`
    mem-vault serve
    mem-vault import-engram /tmp/engram-export.json --agent-id engram
    mem-vault hook-sessionstart < event.json
    mem-vault hook-userprompt < event.json
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
        "hook-userprompt",
        help="UserPromptSubmit lifecycle hook: inject per-prompt memories via semantic search.",
    )
    sub.add_parser(
        "hook-stop",
        help="Stop lifecycle hook: append a line to the audit log. Never blocks.",
    )

    p_reindex = sub.add_parser(
        "reindex",
        help="Walk the vault directory and (re)embed every .md memory into Qdrant.",
    )
    p_reindex.add_argument(
        "--auto-extract",
        action="store_true",
        help="Run the LLM extractor while reindexing (slower, dedupes against vault).",
    )
    p_reindex.add_argument(
        "--purge",
        action="store_true",
        help="Delete the existing Qdrant collection first to start clean.",
    )
    p_reindex.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many memories (0 = no limit, default).",
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
# reindex
# ---------------------------------------------------------------------------


async def _reindex(args: argparse.Namespace) -> int:
    """Walk the vault and re-embed every memory into Qdrant.

    Useful when:
    - you've been editing memories by hand in Obsidian (the file is updated
      but the embedding pointed to the old text);
    - you imported memories from an external source (engram, …) and want a
      clean index;
    - the local Qdrant collection got corrupted or out of sync.

    Idempotent: re-embeds the same content into the same memory_id slot.
    """
    config = load_config()
    service = MemVaultService(config)

    # List every memory directly from disk (no filter) so we hit every file,
    # not just the most-recent N.
    memories = await service._to_thread(
        service.storage.list,
        type=None,
        tags=None,
        user_id=None,
        limit=10**9,  # effectively unbounded
    )
    total = len(memories)
    print(
        f"reindex: {total} memories under {config.memory_dir} "
        f"→ collection={config.qdrant_collection} "
        f"auto_extract={args.auto_extract} purge={args.purge}"
    )

    if args.purge:
        # Recreate the collection in-place: delete + create with the correct
        # embedding dims. Without re-creating, mem0.add() fails with
        # "Collection not found".
        try:
            vs = service.index.mem0.vector_store
            try:
                vs.delete_col()
            except Exception as exc:
                print(f"  delete_col warning: {exc}", file=sys.stderr)
            # Qdrant.create_col(vector_size, on_disk, distance=Cosine).
            vs.create_col(vector_size=config.embedder_dims, on_disk=True)
            print(
                f"  purged + recreated collection ({config.qdrant_collection}, "
                f"{config.embedder_dims} dims)."
            )
        except Exception as exc:
            print(f"  WARNING: purge failed (continuing): {exc}", file=sys.stderr)

    indexed = 0
    failed = 0
    for i, mem in enumerate(memories, start=1):
        if args.limit and indexed >= args.limit:
            print(f"  --limit {args.limit} reached, stopping early.")
            break
        try:
            # Drop any stale entry for this memory_id before re-embedding.
            await service._to_thread(
                service.index.delete_by_metadata, "memory_id", mem.id, mem.user_id
            )
            await service._to_thread(
                service.index.add,
                mem.body or mem.description or mem.name,
                user_id=mem.user_id,
                agent_id=mem.agent_id,
                metadata={"memory_id": mem.id, "type": mem.type, "tags": mem.tags},
                auto_extract=args.auto_extract,
            )
            indexed += 1
            if i % 10 == 0 or i == total:
                print(f"  [{i:>3}/{total}] indexed={indexed} failed={failed}")
        except Exception as exc:
            failed += 1
            print(f"  FAILED on {mem.id}: {exc}", file=sys.stderr)

    print(f"reindex done: indexed={indexed} failed={failed} total={total}")
    return 0 if failed == 0 else 1


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

    if cmd == "reindex":
        sys.exit(asyncio.run(_reindex(args)))

    if cmd == "hook-sessionstart":
        from mem_vault.hooks import sessionstart  # local import to keep startup fast
        sessionstart.run()
        return

    if cmd == "hook-userprompt":
        from mem_vault.hooks import userprompt  # local import to keep startup fast
        userprompt.run()
        return

    if cmd == "hook-stop":
        from mem_vault.hooks import stop  # local import to keep startup fast
        stop.run()
        return

    print(f"error: unknown subcommand: {cmd}", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()

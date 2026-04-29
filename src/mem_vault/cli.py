"""Top-level CLI for mem-vault.

Subcommands:
- ``serve`` (default) — start the MCP stdio server. Same as ``mem-vault-mcp``.
- ``ui`` — start a browser UI (localhost) to browse / search / edit / delete.
- ``reindex`` — re-embed every memory into Qdrant (after hand-edits / imports).
- ``consolidate`` — detect + merge near-duplicate memories with the LLM.
- ``import-engram`` — bulk-import memories from an ``engram export`` JSON file.
- ``export`` — dump every memory to JSON / JSONL / CSV / Markdown for backup.
- ``sync-status`` — diff between the markdown vault and the Qdrant index.
- ``sync-watch`` — watch the vault dir and reindex memories on change (cross-vault sync).
- ``hook-sessionstart`` — SessionStart lifecycle hook (reads stdin, prints JSON).
- ``hook-userprompt`` — UserPromptSubmit lifecycle hook (semantic search per prompt).
- ``hook-stop`` — Stop lifecycle hook (logs to <state_dir>/sessions.log).
- ``version`` — print package version.

Usage:
    mem-vault                            # equivalent to `mem-vault serve`
    mem-vault serve
    mem-vault ui --port 7880
    mem-vault reindex --purge
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
import logging
import sys
from pathlib import Path
from typing import Any

from mem_vault import __version__
from mem_vault.config import load_config
from mem_vault.server import MemVaultService
from mem_vault.server import main as serve_main

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

    p_ui = sub.add_parser(
        "ui",
        help="Start a local browser UI to browse, search, edit, and delete memories.",
    )
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=7880)
    p_ui.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )

    sub.add_parser(
        "sync-status",
        help="Diff between vault files and the local Qdrant index. Tells you if a reindex is needed.",
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

    p_export = sub.add_parser(
        "export",
        help="Export every memory to a portable file (json/jsonl/csv/markdown).",
    )
    p_export.add_argument(
        "format",
        choices=["json", "jsonl", "csv", "markdown"],
        help="Output format.",
    )
    p_export.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write to this file. Defaults to stdout.",
    )
    p_export.add_argument(
        "--no-body",
        action="store_true",
        help="Omit memory bodies (smaller / faster, useful for CSV inspection).",
    )
    p_export.add_argument(
        "--type",
        default=None,
        help="Filter by memory type (e.g. preference, decision, fact).",
    )
    p_export.add_argument(
        "--tag",
        default=None,
        help="Filter by a single tag.",
    )
    # ── direct CRUD: same calls the MCP server exposes, available from the shell ──
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
        help="Save a new memory (same as memory_save MCP tool). Reads body from stdin if --content is omitted.",
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

    p_consolidate = sub.add_parser(
        "consolidate",
        help="Detect near-duplicate memories and merge them with the LLM.",
    )
    p_consolidate.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help=(
            "Cosine similarity threshold for candidate pairs (default 0.85). "
            "Tune to your embedder: bge-m3 rarely exceeds 0.92 even on near-"
            "duplicates; OpenAI text-embedding-3-large can. The LLM is the "
            "second filter that catches false positives."
        ),
    )
    p_consolidate.add_argument(
        "--max-pairs",
        type=int,
        default=20,
        help="Process at most this many pairs per run (default 20).",
    )
    p_consolidate.add_argument(
        "--apply",
        action="store_true",
        help="Actually merge/delete. Without this flag, only print the plan.",
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
# direct CRUD (search / list / save / get / delete) — shell-friendly wrappers
# ---------------------------------------------------------------------------


def _print_human_search(payload: dict[str, Any]) -> None:
    if not payload.get("ok"):
        print(f"error: {payload.get('error')}", file=sys.stderr)
        return
    results = payload.get("results") or []
    if not results:
        print("no matches.")
        return
    print(f"found {payload.get('count', len(results))} matches for {payload.get('query')!r}:\n")
    for r in results:
        score = r.get("score")
        score_tag = f"  (score {score:.3f})" if isinstance(score, (int, float)) else ""
        mem = r.get("memory") or {}
        print(f"· {mem.get('id') or r.get('id')}{score_tag}")
        print(f"    {mem.get('name', '')}")
        if mem.get("description"):
            print(f"    {mem['description'][:160]}")
        print()


def _print_human_list(payload: dict[str, Any]) -> None:
    if not payload.get("ok"):
        print(f"error: {payload.get('error')}", file=sys.stderr)
        return
    memories = payload.get("memories") or []
    print(f"{payload.get('count', len(memories))} memories:\n")
    for m in memories:
        tags = ",".join(m.get("tags") or [])
        print(
            f"· [{m.get('type', 'note'):>10}]  {m.get('id')}  ·  {m.get('updated', '')[:10]}  ·  #{tags}"
        )
        if m.get("description"):
            print(f"    {m['description'][:120]}")


def _print_human_get(payload: dict[str, Any]) -> None:
    if not payload.get("ok"):
        print(f"error: {payload.get('error')}", file=sys.stderr)
        return
    m = payload.get("memory") or {}
    print(f"id:          {m.get('id')}")
    print(f"name:        {m.get('name')}")
    print(f"type:        {m.get('type')}")
    print(f"tags:        {','.join(m.get('tags') or [])}")
    print(f"created:     {m.get('created')}")
    print(f"updated:     {m.get('updated')}")
    print(f"agent_id:    {m.get('agent_id') or '—'}")
    print(f"user_id:     {m.get('user_id')}")
    print(f"visible_to:  {m.get('visible_to')}")
    print()
    print(m.get("body") or "")


async def _crud(cmd: str, args: argparse.Namespace) -> int:
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
            _print_human_search(payload)
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
            _print_human_list(payload)
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
                f"saved · id={mem.get('id')} · type={mem.get('type')} · indexed={payload.get('indexed')}"
            )
            print(f"path: {payload.get('path')}")
        return 0 if payload.get("ok") else 1

    if cmd == "get":
        payload = await service.get({"id": args.id})
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _print_human_get(payload)
        return 0 if payload.get("ok") else 1

    if cmd == "delete":
        if not args.yes:
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


# ---------------------------------------------------------------------------
# consolidate
# ---------------------------------------------------------------------------


def _consolidate(args: argparse.Namespace) -> int:
    """Detect + merge near-duplicate memories.

    Two-pass design:
    1. ``find_candidate_pairs`` (pure embedding similarity, no LLM).
    2. For each pair, ``_ask_llm`` decides MERGE / KEEP_BOTH / KEEP_FIRST /
       KEEP_SECOND. The LLM call is the slow part; we cap it to
       ``--max-pairs`` so a wild run doesn't burn through 400 LLM calls.
    """
    import ollama

    from mem_vault.consolidate import (
        _ask_llm,
        apply_resolution,
        find_candidate_pairs,
    )

    config = load_config()
    service = MemVaultService(config)
    storage = service.storage
    index = service.index

    print(
        f"consolidate: scanning {config.memory_dir} "
        f"threshold={args.threshold} max_pairs={args.max_pairs} "
        f"apply={args.apply}"
    )

    pairs = find_candidate_pairs(storage, index, threshold=args.threshold, user_id=config.user_id)
    if not pairs:
        print("  no near-duplicate pairs found.")
        return 0
    print(f"  found {len(pairs)} candidate pairs (showing top {args.max_pairs}):")

    ollama_client = ollama.Client(host=config.ollama_host)
    summary = {"MERGE": 0, "KEEP_BOTH": 0, "KEEP_FIRST": 0, "KEEP_SECOND": 0}

    for i, pair in enumerate(pairs[: args.max_pairs], start=1):
        print(
            f"\n  [{i}/{min(len(pairs), args.max_pairs)}] "
            f"score={pair.score:.3f}\n"
            f"    A: {pair.a.id} ({pair.a.type}) — {pair.a.name[:60]}\n"
            f"    B: {pair.b.id} ({pair.b.type}) — {pair.b.name[:60]}"
        )
        try:
            res = _ask_llm(config, pair, ollama_client=ollama_client)
        except Exception as exc:
            print(f"    LLM call failed: {exc}", file=sys.stderr)
            continue

        print(f"    -> {res.action}: {res.rationale[:100]}")
        if not args.apply:
            continue

        try:
            outcome = apply_resolution(storage, index, pair, res, user_id=config.user_id)
            summary[res.action] = summary.get(res.action, 0) + 1
            print(f"       applied: {outcome}")
        except Exception as exc:
            print(f"    apply failed: {exc}", file=sys.stderr)

    print()
    if args.apply:
        print(f"consolidate done: {summary}")
    else:
        print("consolidate done (dry-run, no changes written). Re-run with --apply to commit.")
    return 0


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def _export(args: argparse.Namespace) -> int:
    """Walk the vault, optionally filter, write to stdout or a file."""
    from mem_vault.export import export as do_export

    config = load_config()
    storage = MemVaultService(config).storage

    memories = storage.list(
        type=args.type,
        tags=[args.tag] if args.tag else None,
        user_id=None,
        limit=10**9,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            do_export(memories, args.format, out=f, include_body=not args.no_body)
        size = args.output.stat().st_size
        print(
            f"export done: {len(memories)} memor"
            f"{'ies' if len(memories) != 1 else 'y'} → {args.output} ({size:,} bytes)",
            file=sys.stderr,
        )
    else:
        do_export(memories, args.format, out=sys.stdout, include_body=not args.no_body)
    return 0


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


def _sync_status() -> int:
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


def _sync_watch(args: argparse.Namespace) -> int:
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

    if cmd == "ui":
        try:
            from mem_vault.ui.server import serve as ui_serve
        except ImportError as exc:
            print(
                f"error: ui dependencies not installed ({exc}). "
                "Install with: uv tool install --editable '.[ui]'",
                file=sys.stderr,
            )
            sys.exit(2)
        return ui_serve(host=args.host, port=args.port, log_level=args.log_level)

    if cmd == "import-engram":
        sys.exit(asyncio.run(_import_engram(args)))

    if cmd in {"search", "list", "save", "get", "delete"}:
        sys.exit(asyncio.run(_crud(cmd, args)))

    if cmd == "export":
        sys.exit(_export(args))

    if cmd == "sync-status":
        sys.exit(_sync_status())

    if cmd == "sync-watch":
        sys.exit(_sync_watch(args))

    if cmd == "reindex":
        sys.exit(asyncio.run(_reindex(args)))

    if cmd == "consolidate":
        sys.exit(_consolidate(args))

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

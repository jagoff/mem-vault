"""``mem-vault reindex`` — re-embed every memory into Qdrant.

Useful when:
- you've been editing memories by hand in Obsidian (the file is updated
  but the embedding pointed to the old text);
- you imported memories from an external source (engram, …) and want a
  clean index;
- the local Qdrant collection got corrupted or out of sync.

Idempotent: re-embeds the same content into the same memory_id slot.
"""

from __future__ import annotations

import argparse
import sys


def add_subparser(sub: argparse._SubParsersAction) -> None:
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


async def run(args: argparse.Namespace) -> int:
    from mem_vault.config import load_config
    from mem_vault.server import MemVaultService

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

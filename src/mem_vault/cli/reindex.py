"""``mem-vault reindex`` — re-embed every memory into Qdrant.

Useful when:
- you've been editing memories by hand in Obsidian (the file is updated
  but the embedding pointed to the old text);
- you imported memories from an external source (engram, …) and want a
  clean index;
- the local Qdrant collection got corrupted or out of sync.

Idempotent: re-embeds the same content into the same memory_id slot.

Incremental by default
----------------------
Each indexed memory carries a ``content_hash`` (truncated SHA-256 of the
body) in its Qdrant metadata. On subsequent runs, memories whose current
hash already matches the indexed one are *skipped* — no embedding call,
no LLM call, no I/O against Ollama. The first run after upgrading from
a pre-hash version will re-embed everything (since the existing entries
have no ``content_hash`` yet) but each later run only touches the diff.

Pass ``--force`` to disable the skip and re-embed every memory
unconditionally — useful after changing the embedder model or upgrading
mem0.

Orphan cleanup
--------------
After the walk completes, ``reindex`` also looks for entries inside
Qdrant whose ``memory_id`` no longer has a backing ``.md`` file (e.g.
because the user deleted the file in Obsidian, or a previous
``consolidate`` run crashed mid-merge). Those orphans are removed so
``memory_search`` doesn't return ghost results. Skipped automatically
when ``--purge`` (the collection is fresh) or ``--limit`` (we didn't
walk every memory, so we can't tell what's orphaned vs unwalked) is set.
"""

from __future__ import annotations

import argparse
import asyncio
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
    p_reindex.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-embed every memory unconditionally. Without this flag, "
            "memories whose content_hash already matches the indexed one "
            "are skipped (incremental reindex)."
        ),
    )
    p_reindex.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help=(
            "Max concurrent embed calls to Ollama (default 4). Each memory "
            "triggers one embedding roundtrip; parallelism speeds up "
            "cold reindex roughly linearly with value. Set to 1 for "
            "strictly-sequential behavior (debug, low-memory). Higher "
            "than ~8 usually starts bottlenecking on Ollama itself."
        ),
    )


async def run(args: argparse.Namespace) -> int:
    from mem_vault.config import load_config
    from mem_vault.index import compute_content_hash
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
        f"auto_extract={args.auto_extract} purge={args.purge} force={args.force}"
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

    # Concurrency: bound the number of in-flight embed calls to Ollama.
    # Ollama happily serves several embedding requests in parallel, but too
    # many just bottleneck on the model backend. Default 4 is the empirical
    # sweet spot on a laptop with bge-m3; raise via ``--concurrency``.
    # ``concurrency=1`` degrades to strictly-sequential (the old behavior).
    concurrency = max(1, int(args.concurrency))
    sem = asyncio.Semaphore(concurrency)
    counters = {"indexed": 0, "skipped": 0, "failed": 0, "stopped": False}
    counters_lock = asyncio.Lock()

    async def _process_one(i: int, mem) -> None:
        # Under --limit, stop once we've reached the cap. Concurrent workers
        # coordinate via the ``stopped`` flag so pending tasks don't keep
        # going after the cap was hit. Accurate bookkeeping: we count the
        # *indexed* items under lock, not just the attempted ones.
        async with counters_lock:
            if args.limit and counters["indexed"] >= args.limit:
                counters["stopped"] = True
                return
            if counters["stopped"]:
                return

        body = mem.body or mem.description or mem.name
        current_hash = compute_content_hash(body)

        async with sem:
            # Incremental skip.
            if not args.force and not args.purge:
                try:
                    existing = await service._to_thread(
                        service.index.get_by_metadata, "memory_id", mem.id, mem.user_id
                    )
                except Exception as exc:
                    print(
                        f"  hash lookup failed for {mem.id}: {exc} (will re-embed)",
                        file=sys.stderr,
                    )
                    existing = []
                if existing and any(
                    (e.get("metadata") or {}).get("content_hash") == current_hash for e in existing
                ):
                    async with counters_lock:
                        counters["skipped"] += 1
                        if i % 25 == 0 or i == total:
                            print(
                                f"  [{i:>3}/{total}] "
                                f"indexed={counters['indexed']} "
                                f"skipped={counters['skipped']} "
                                f"failed={counters['failed']}"
                            )
                    return

            try:
                await service._to_thread(
                    service.index.delete_by_metadata, "memory_id", mem.id, mem.user_id
                )
                metadata = {
                    "memory_id": mem.id,
                    "type": mem.type,
                    "tags": mem.tags,
                    "content_hash": current_hash,
                }
                results = await service._to_thread(
                    service.index.add,
                    body,
                    user_id=mem.user_id,
                    agent_id=mem.agent_id,
                    metadata=metadata,
                    auto_extract=args.auto_extract,
                )
                if args.auto_extract:
                    from mem_vault.server import _has_added_event

                    if not _has_added_event(list(results) if results else []):
                        await service._to_thread(
                            service.index.add,
                            body,
                            user_id=mem.user_id,
                            agent_id=mem.agent_id,
                            metadata=metadata,
                            auto_extract=False,
                        )
                async with counters_lock:
                    counters["indexed"] += 1
                    if i % 10 == 0 or i == total:
                        print(
                            f"  [{i:>3}/{total}] "
                            f"indexed={counters['indexed']} "
                            f"skipped={counters['skipped']} "
                            f"failed={counters['failed']}"
                        )
            except Exception as exc:
                async with counters_lock:
                    counters["failed"] += 1
                print(f"  FAILED on {mem.id}: {exc}", file=sys.stderr)

    print(f"  concurrency={concurrency}")
    await asyncio.gather(*(_process_one(i, m) for i, m in enumerate(memories, start=1)))
    if counters["stopped"]:
        print(f"  --limit {args.limit} reached, stopping early.")

    indexed = counters["indexed"]
    skipped = counters["skipped"]
    failed = counters["failed"]

    # Orphan sweep: remove index entries whose memory_id has no .md anymore.
    # We only run it when we've actually walked every file (no --limit) and
    # weren't told to start clean (--purge already nuked the collection).
    orphans_removed = 0
    if not args.purge and not args.limit:
        orphans_removed = await _sweep_orphans(service, memories)

    print(
        f"reindex done: indexed={indexed} skipped={skipped} failed={failed} "
        f"orphans_removed={orphans_removed} total={total}"
    )
    return 0 if failed == 0 else 1


async def _sweep_orphans(service, memories) -> int:
    """Remove Qdrant entries whose memory_id has no ``.md`` in the vault.

    Returns the count of removed entries. Best-effort: any failure is
    logged to stderr and treated as zero — we don't want a flaky orphan
    sweep to mask the otherwise-successful reindex.
    """
    vault_ids = {m.id for m in memories}
    user_id = service.config.user_id

    try:
        entries = await service._to_thread(
            service.index.mem0.get_all,
            filters={"user_id": user_id},
            limit=10**6,
        )
    except TypeError:
        try:
            entries = await service._to_thread(
                service.index.mem0.get_all, filters={"user_id": user_id}
            )
        except Exception as exc:
            print(f"  orphan sweep skipped (get_all failed: {exc})", file=sys.stderr)
            return 0
    except Exception as exc:
        print(f"  orphan sweep skipped (get_all failed: {exc})", file=sys.stderr)
        return 0

    if isinstance(entries, dict):
        items = list(entries.get("results", []))
    elif isinstance(entries, list):
        items = entries
    else:
        items = []

    indexed_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = (item.get("metadata") or {}).get("memory_id")
        if isinstance(mid, str) and mid:
            indexed_ids.add(mid)

    orphan_ids = indexed_ids - vault_ids
    if not orphan_ids:
        return 0

    removed_total = 0
    for orphan_id in sorted(orphan_ids):
        try:
            removed = await service._to_thread(
                service.index.delete_by_metadata, "memory_id", orphan_id, user_id
            )
            removed_total += int(removed or 0)
            print(f"  removed orphan: {orphan_id} ({removed} entries)")
        except Exception as exc:
            print(f"  orphan {orphan_id} cleanup failed: {exc}", file=sys.stderr)

    return removed_total

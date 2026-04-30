"""``mem-vault consolidate`` — detect + merge near-duplicate memories.

Two-pass design:
1. ``find_candidate_pairs`` (pure embedding similarity, no LLM).
2. For each pair, ``_ask_llm`` decides MERGE / KEEP_BOTH / KEEP_FIRST /
   KEEP_SECOND. The LLM call is the slow part; we cap it to
   ``--max-pairs`` so a wild run doesn't burn through 400 LLM calls.
"""

from __future__ import annotations

import argparse
import sys


def add_subparser(sub: argparse._SubParsersAction) -> None:
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


def run(args: argparse.Namespace) -> int:
    import ollama

    from mem_vault.config import load_config
    from mem_vault.consolidate import (
        _ask_llm,
        apply_resolution,
        find_candidate_pairs,
    )
    from mem_vault.server import MemVaultService

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

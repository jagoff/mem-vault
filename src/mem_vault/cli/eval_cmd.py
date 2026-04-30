"""``mem-vault eval`` — run a labeled query set through ``memory_search``.

Ships no default fixture (the corpus is the user's own vault), but
accepts ``--queries <path>`` pointing at a JSON file of
``[{query, expected, tag?}, ...]`` entries. For each query we call
the real service with the user's live Qdrant index, collect the
returned ids, and compute hit@k + MRR via :mod:`mem_vault.eval`.

Useful for:

- Verifying a retrieval tweak (reranker, hybrid, decay) improves a
  labeled set before shipping.
- CI regression — drop a ``queries.json`` in a fixture vault, run
  ``mem-vault eval --queries queries.json --threshold 0.6``; non-zero
  exit when the hit@5 falls below the threshold.

Kept outside ``memory_search`` on purpose: this is an evaluation tool,
not a retrieval backend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "eval",
        help=(
            "Run a labeled query set through memory_search and report "
            "hit@k + MRR. Useful for regression tests on retrieval changes."
        ),
    )
    p.add_argument(
        "--queries",
        type=str,
        required=True,
        help=(
            "Path to a JSON file of ``[{query, expected, tag?}]`` entries. "
            "``expected`` is the list of memory ids that should appear in the "
            "top-k results."
        ),
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k cap requested from memory_search (default 10).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help=(
            "Fail with non-zero exit code when hit@5 < threshold. Default 0 "
            "(report only). Typical CI value: 0.6-0.8 depending on corpus."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON on stdout (machine-readable).",
    )


async def run(args: argparse.Namespace) -> int:
    from mem_vault.config import load_config
    from mem_vault.eval import EvalRun, compute_metrics, load_queries
    from mem_vault.server import MemVaultService

    queries_path = Path(args.queries).expanduser()
    if not queries_path.exists():
        print(f"error: queries file not found: {queries_path}", file=sys.stderr)
        return 2
    try:
        queries = load_queries(queries_path)
    except Exception as exc:
        print(f"error: failed to parse queries: {exc}", file=sys.stderr)
        return 2
    if not queries:
        print(f"error: {queries_path} contained zero valid queries", file=sys.stderr)
        return 2

    config = load_config()
    service = MemVaultService(config)

    runs: list[EvalRun] = []
    for q in queries:
        payload = await service.search({"query": q.query, "k": args.k})
        if not payload.get("ok"):
            print(f"  WARN: search failed for {q.query!r}: {payload.get('error')}", file=sys.stderr)
            runs.append(EvalRun(query=q, returned=[]))
            continue
        returned_ids = [str(r.get("id")) for r in payload.get("results", []) if r.get("id")]
        runs.append(EvalRun(query=q, returned=returned_ids))

    report = compute_metrics(runs)

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print()
        print("mem-vault eval")
        print("─" * 50)
        print(report.render())
        # Per-query preview: which queries missed? Useful context.
        missed = [row for row in report.per_query if not row["hit@5"]]
        if missed:
            print(f"\n  missed at hit@5: {len(missed)} / {len(report.per_query)}")
            for row in missed[:5]:
                print(f"    - {row['query'][:60]!r} — expected {row['expected']}")
                print(f"      returned top 5: {row['returned'][:5]}")
        print()

    if args.threshold > 0 and report.hit_at_5 < args.threshold:
        print(
            f"FAIL: hit@5 {report.hit_at_5:.2%} < threshold {args.threshold:.2%}",
            file=sys.stderr,
        )
        return 1
    return 0

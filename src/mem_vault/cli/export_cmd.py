"""``mem-vault export`` — dump every memory to a portable file.

Wraps :func:`mem_vault.export.export` with the CLI surface (filters,
output path, ``--no-body``). Default destination is stdout so the
command pipes nicely (e.g. ``mem-vault export jsonl | jq`` ).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_subparser(sub: argparse._SubParsersAction) -> None:
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


def run(args: argparse.Namespace) -> int:
    from mem_vault.config import load_config
    from mem_vault.export import export as do_export
    from mem_vault.server import MemVaultService

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

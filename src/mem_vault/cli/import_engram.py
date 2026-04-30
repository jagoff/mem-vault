"""``mem-vault import-engram`` — bulk import from an ``engram export`` JSON file.

We map engram's observation rows to mem-vault save kwargs (preserving the
project / scope / topic_key as tags, marking origin with ``source:engram``),
then call ``MemVaultService.save`` for each one. The mapping table lives
inline at the top of this module so it's easy to extend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


def add_subparser(sub: argparse._SubParsersAction) -> None:
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


async def run(args: argparse.Namespace) -> int:
    from mem_vault.config import load_config
    from mem_vault.server import MemVaultService

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

"""Shared helpers used by multiple CLI subcommands.

Pretty-printers for the human-readable output of the CRUD subcommands. The
``--json`` flag bypasses these and dumps the raw service envelope instead.
"""

from __future__ import annotations

import sys
from typing import Any


def print_human_search(payload: dict[str, Any]) -> None:
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


def print_human_list(payload: dict[str, Any]) -> None:
    if not payload.get("ok"):
        print(f"error: {payload.get('error')}", file=sys.stderr)
        return
    memories = payload.get("memories") or []
    print(f"{payload.get('count', len(memories))} memories:\n")
    for m in memories:
        tags = ",".join(m.get("tags") or [])
        print(
            f"· [{m.get('type', 'note'):>10}]  {m.get('id')}  ·  "
            f"{m.get('updated', '')[:10]}  ·  #{tags}"
        )
        if m.get("description"):
            print(f"    {m['description'][:120]}")


def print_human_get(payload: dict[str, Any]) -> None:
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

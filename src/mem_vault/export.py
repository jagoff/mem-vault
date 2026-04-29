"""Export memories to portable formats for backup, migration, or analysis.

Supported formats:

- ``json``       — single JSON object with metadata + ``memories: [...]``
- ``jsonl``      — one memory per line, streamable, friendly to ``jq`` and
                   tools like ``rg --json``
- ``csv``        — flat columns (id, name, type, tags, created, updated, agent_id);
                   body excluded by default to keep CSV scannable
- ``markdown``   — single concatenated `.md` file (one memory per ``---`` block);
                   useful to paste into a document or feed to an LLM as one prompt

All formats include enough metadata to round-trip back into mem-vault via
the JSON/JSONL importers (TODO — currently ``import-engram`` is the only
importer; importing a ``mem-vault export`` is on the roadmap).
"""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Iterable
from datetime import datetime
from typing import IO

from mem_vault import __version__
from mem_vault.storage import Memory

_SUPPORTED = {"json", "jsonl", "csv", "markdown"}


def supported_formats() -> set[str]:
    return set(_SUPPORTED)


def export(
    memories: Iterable[Memory],
    fmt: str,
    *,
    out: IO[str] | None = None,
    include_body: bool = True,
) -> None:
    """Stream-write ``memories`` in the requested ``fmt`` to ``out``.

    ``out`` defaults to stdout when not provided. ``include_body`` drops the
    ``body`` field for compactness — useful for CSV exports that you want to
    open in a spreadsheet without 10 KB cells everywhere.
    """
    if fmt not in _SUPPORTED:
        raise ValueError(f"unknown format: {fmt!r} (supported: {sorted(_SUPPORTED)})")
    sink = out if out is not None else sys.stdout

    materialised = list(memories)

    if fmt == "json":
        _write_json(materialised, sink, include_body)
    elif fmt == "jsonl":
        _write_jsonl(materialised, sink, include_body)
    elif fmt == "csv":
        _write_csv(materialised, sink, include_body)
    elif fmt == "markdown":
        _write_markdown(materialised, sink, include_body)


def _memory_dict(mem: Memory, include_body: bool) -> dict:
    """Serialise one Memory to a dict, omitting body if ``include_body`` is False."""
    d = mem.to_dict()
    if not include_body:
        d.pop("body", None)
    return d


def _write_json(memories: list[Memory], out: IO[str], include_body: bool) -> None:
    payload = {
        "schema": "mem-vault.export.v1",
        "exporter_version": __version__,
        "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "count": len(memories),
        "memories": [_memory_dict(m, include_body) for m in memories],
    }
    json.dump(payload, out, ensure_ascii=False, indent=2)
    out.write("\n")


def _write_jsonl(memories: list[Memory], out: IO[str], include_body: bool) -> None:
    for mem in memories:
        out.write(json.dumps(_memory_dict(mem, include_body), ensure_ascii=False))
        out.write("\n")


def _write_csv(memories: list[Memory], out: IO[str], include_body: bool) -> None:
    # CSV gets a flat schema. Lists become JSON-encoded strings so the file
    # round-trips cleanly without quoting hell.
    columns = [
        "id",
        "name",
        "type",
        "description",
        "tags",
        "created",
        "updated",
        "agent_id",
        "user_id",
        "origin_session_id",
    ]
    if include_body:
        columns.append("body")

    writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for mem in memories:
        d = mem.to_dict()
        row = {
            "id": d.get("id", ""),
            "name": d.get("name", ""),
            "type": d.get("type", ""),
            "description": d.get("description", ""),
            "tags": json.dumps(d.get("tags") or [], ensure_ascii=False),
            "created": d.get("created") or "",
            "updated": d.get("updated") or "",
            "agent_id": d.get("agent_id") or "",
            "user_id": d.get("user_id") or "",
            "origin_session_id": d.get("origin_session_id") or "",
        }
        if include_body:
            row["body"] = d.get("body", "")
        writer.writerow(row)


def _write_markdown(memories: list[Memory], out: IO[str], include_body: bool) -> None:
    out.write("# mem-vault export\n\n")
    out.write(f"_Exported {datetime.now().astimezone().isoformat(timespec='seconds')} · ")
    out.write(f"{len(memories)} memor{'ies' if len(memories) != 1 else 'y'}_\n\n")
    for mem in memories:
        out.write("---\n\n")
        out.write(f"## {mem.name}\n\n")
        bits = [
            f"`{mem.id}`",
            f"type: `{mem.type}`",
        ]
        if mem.tags:
            bits.append("tags: " + ", ".join(f"`#{t}`" for t in mem.tags))
        if mem.agent_id:
            bits.append(f"agent: `@{mem.agent_id}`")
        if mem.updated:
            bits.append(f"updated: `{mem.updated}`")
        out.write(" · ".join(bits) + "\n\n")
        if mem.description:
            out.write(f"> {mem.description}\n\n")
        if include_body and mem.body:
            out.write(mem.body.rstrip() + "\n\n")

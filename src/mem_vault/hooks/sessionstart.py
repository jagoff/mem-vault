"""SessionStart hook: inject the user's stored preferences into agent context.

Devin and Claude Code support a ``hookSpecificOutput.additionalContext`` field
on the SessionStart event — anything we put there shows up to the agent as
extra system context for the rest of the session. We use it to surface the
user's preferences and the most recently updated memories, so the agent
doesn't have to call ``memory_search`` on every cold start.

Robust by design: if mem-vault config is missing, the vault is empty, or
Ollama is unreachable, the hook emits no context and exits 0. It never blocks
the session.
"""

from __future__ import annotations

import asyncio
import json
import sys


async def _gather_context() -> str:
    try:
        from mem_vault.server import build_service
    except Exception as exc:
        print(f"mem-vault: import failed: {exc}", file=sys.stderr)
        return ""

    try:
        service = build_service()
    except Exception as exc:
        print(f"mem-vault: service init failed: {exc}", file=sys.stderr)
        return ""

    sections: dict[str, list[dict]] = {"preferences": [], "recent": []}
    try:
        prefs = await service.list_({"type": "preference", "limit": 10})
        sections["preferences"].extend(prefs.get("memories", []))
        feedback = await service.list_({"type": "feedback", "limit": 5})
        sections["preferences"].extend(feedback.get("memories", []))
        recent = await service.list_({"limit": 5})
        sections["recent"].extend(recent.get("memories", []))
    except Exception as exc:
        print(f"mem-vault: list failed: {exc}", file=sys.stderr)
        return ""

    seen: set[str] = set()
    lines: list[str] = ["## Memorias relevantes (mem-vault)\n"]

    def emit(label: str, memories: list[dict], cap: int) -> None:
        bucket: list[str] = []
        for m in memories:
            mid = m.get("id")
            if not mid or mid in seen:
                continue
            seen.add(mid)
            name = m.get("name") or mid
            descr = (m.get("description") or "").strip().replace("\n", " ")
            bucket.append(f"- **{name}** — {descr[:240]}")
            if len(bucket) >= cap:
                break
        if bucket:
            lines.append(f"### {label}")
            lines.extend(bucket)
            lines.append("")

    emit("Preferencias y feedback", sections["preferences"], cap=8)
    emit("Memorias recientes", sections["recent"], cap=5)

    if len(lines) <= 1:
        return ""
    lines.append(
        "_Estas memorias vienen de tu vault Obsidian (`99-AI/memory/`). "
        "Si necesitás más detalle, llamá a `memory_search` o `memory_get`._"
    )
    return "\n".join(lines)


def run() -> None:
    try:
        json.load(sys.stdin)  # consume stdin payload (we don't need any field)
    except Exception:
        pass

    context = ""
    try:
        context = asyncio.run(_gather_context())
    except Exception as exc:
        print(f"mem-vault: SessionStart hook failed: {exc}", file=sys.stderr)

    if not context:
        return  # empty stdout + exit 0 = no-op hook

    payload = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }
    json.dump(payload, sys.stdout, ensure_ascii=False)

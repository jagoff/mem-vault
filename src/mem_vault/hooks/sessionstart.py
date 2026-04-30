"""SessionStart hook: inject preferences + project-aware memories on cold start.

Devin and Claude Code support a ``hookSpecificOutput.additionalContext`` field
on the SessionStart event — anything we put there shows up to the agent as
extra system context for the rest of the session. We use it to surface, in
order:

1. **Memorias del proyecto** — memorias relacionadas al ``cwd`` actual
   (typically the leaf of the path becomes a project signal). When the user
   re-opens Devin in a repo they've worked on before, the agent comes
   pre-loaded with the relevant decisions, gotchas, and bug fixes for
   that project — no cold start, no re-explaining.

2. **Preferencias y feedback** — global memorias ``type=preference`` /
   ``type=feedback`` so the agent honors universal style choices
   (idioma, conventions, escalation policies).

3. **Memorias recientes** — ``type``-agnóstic top recent so the agent
   has weak signal on what was being worked on lately.

Robust by design: if mem-vault config is missing, the vault is empty, or
Ollama is unreachable, the hook emits no context and exits 0. It never
blocks the session.

Project-signal extraction: ``cwd`` comes from the SessionStart stdin
payload (preferred — Devin documents this), falling back to the
``DEVIN_PROJECT_DIR`` env var, and ultimately to ``os.getcwd()``. The
leaf component (e.g. ``mem-vault`` for ``/Users/fer/repositories/mem-vault``)
is used as the primary semantic + tag signal. Path prefixes specific to
the user's filesystem (``/Users/<name>/``, ``/home/<name>/``) are stripped
so they don't pollute the embedding query.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Component fragments we always strip when deriving project signals from a
# cwd: filesystem-layout boilerplate that adds no semantic value.
_CWD_NOISE_COMPONENTS: frozenset[str] = frozenset(
    {
        "users",
        "home",
        "repositories",
        "repos",
        "code",
        "src",
        "projects",
        "documents",
        "workspace",
        "work",
        "dev",
    }
)


def project_signals_from_cwd(cwd: str | None) -> list[str]:
    """Return ordered list of project-name candidates derived from ``cwd``.

    The first element is the leaf (most specific) — that's the strongest
    signal and what we use as the search query / tag filter. Following
    elements are progressively more general (parent → grandparent → …),
    skipping noise components like ``/Users/<me>/``.

    Returns an empty list when ``cwd`` is None / empty / "/" so callers
    can short-circuit cleanly.
    """
    if not cwd:
        return []
    parts = [p for p in Path(cwd).parts if p and p != "/"]
    signals: list[str] = []
    for part in reversed(parts):
        bare = part.lower()
        if bare in _CWD_NOISE_COMPONENTS:
            continue
        # Skip the home dir name itself ("fer", "ubuntu", etc.) by
        # heuristic: short enough to be a username, no separators.
        if len(part) <= 12 and "-" not in part and "_" not in part and "." not in part:
            # First leaf is always kept; later "username-like" components are dropped.
            if signals:
                continue
        signals.append(part)
    return signals


async def _fetch_project_memories(service: Any, signals: list[str]) -> list[dict]:
    """Pull memorias relevantes al proyecto, deduped by ``id``.

    Strategy: combine three cheap queries.

    1. Tag filter on ``project:<leaf>`` (the convention used by hooks that
       auto-tag from cwd).
    2. Tag filter on the bare leaf name (``project:rag`` style schema is
       common, but freeform tags happen).
    3. Semantic search using the leaf as the query (catches memorias whose
       body mentions the project name even if the tag is missing).

    Each step is best-effort — failures are swallowed so a single dead
    Ollama doesn't blank out the whole hook.
    """
    if not signals:
        return []
    leaf = signals[0]

    seen: set[str] = set()
    out: list[dict] = []

    async def _absorb(memories: list[dict]) -> None:
        for m in memories:
            mid = m.get("id")
            if mid and mid not in seen:
                seen.add(mid)
                out.append(m)

    # 1. project:<leaf>
    try:
        by_proj_tag = await service.list_({"tags": [f"project:{leaf}"], "limit": 8})
        await _absorb(by_proj_tag.get("memories", []))
    except Exception:
        pass

    # 2. bare leaf
    try:
        by_bare_tag = await service.list_({"tags": [leaf], "limit": 8})
        await _absorb(by_bare_tag.get("memories", []))
    except Exception:
        pass

    # 3. semantic search using the leaf (skip if we already have a healthy bucket)
    if len(out) < 5:
        try:
            sem = await service.search({"query": leaf, "k": 5, "threshold": 0.25})
            sem_results = sem.get("results", []) if sem.get("ok") else []
            sem_memories = [r.get("memory") for r in sem_results if r.get("memory")]
            await _absorb(sem_memories)
        except Exception:
            pass

    return out


async def _gather_context(cwd: str | None = None) -> str:
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

    signals = project_signals_from_cwd(cwd)
    project_memories: list[dict] = []
    if signals:
        project_memories = await _fetch_project_memories(service, signals)

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

    # Project memorias go FIRST — those are the most contextually relevant
    # at this exact `cwd`. Prefs and recent provide global / lateral signal.
    if signals:
        leaf = signals[0]
        emit(f"Memorias del proyecto (`{leaf}`)", project_memories, cap=8)
    emit("Preferencias y feedback", sections["preferences"], cap=5)
    emit("Memorias recientes", sections["recent"], cap=3)

    if len(lines) <= 1:
        return ""
    lines.append(
        "_Estas memorias vienen de tu vault Obsidian. "
        "Si necesitás más detalle, llamá a `memory_search` o `memory_get`._"
    )
    return "\n".join(lines)


def _resolve_cwd(payload: dict[str, Any]) -> str | None:
    """Pick the best-available cwd: payload field > DEVIN_PROJECT_DIR > getcwd."""
    raw = payload.get("cwd") or payload.get("workingDirectory") or payload.get("project_dir")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    env_dir = os.environ.get("DEVIN_PROJECT_DIR")
    if env_dir and env_dir.strip():
        return env_dir.strip()
    try:
        return os.getcwd()
    except OSError:
        return None


def run() -> None:
    payload: dict[str, Any] = {}
    try:
        loaded = json.load(sys.stdin)
        if isinstance(loaded, dict):
            payload = loaded
    except Exception:
        pass

    cwd = _resolve_cwd(payload)

    context = ""
    try:
        context = asyncio.run(_gather_context(cwd=cwd))
    except Exception as exc:
        print(f"mem-vault: SessionStart hook failed: {exc}", file=sys.stderr)

    if not context:
        return  # empty stdout + exit 0 = no-op hook

    out = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }
    json.dump(out, sys.stdout, ensure_ascii=False)

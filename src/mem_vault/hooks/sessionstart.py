"""SessionStart hook: inject preferences + project-aware memories on cold start.

Devin and Claude Code support a ``hookSpecificOutput.additionalContext`` field
on the SessionStart event — anything we put there shows up to the agent as
extra system context for the rest of the session. We use it to surface, in
order:

1. **Estadísticas del vault** — banner con totales, recientes, top tags,
   actividad temporal (1d/7d/30d), salud del corpus (lint flags +
   duplicates pendientes) y métricas operativas (p50/p95 + error rate de
   las últimas 24 h, si el JSONL sink está habilitado). Es lo primero que
   ve el agente en una sesión nueva: un health check del estado del vault
   antes de cualquier query.

2. **Memorias del proyecto** — memorias relacionadas al ``cwd`` actual
   (typically the leaf of the path becomes a project signal). When the user
   re-opens Devin in a repo they've worked on before, the agent comes
   pre-loaded with the relevant decisions, gotchas, and bug fixes for
   that project — no cold start, no re-explaining.

3. **Preferencias y feedback** — global memorias ``type=preference`` /
   ``type=feedback`` so the agent honors universal style choices
   (idioma, conventions, escalation policies).

4. **Memorias recientes** — ``type``-agnóstic top recent so the agent
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
from datetime import datetime, timedelta
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


def _parse_iso(ts: str | None) -> datetime | None:
    """Tolerant ISO-8601 parse → tz-aware datetime, or ``None`` on failure.

    The vault stores ``created`` / ``updated`` as ISO strings via
    ``datetime.isoformat``; some legacy memorias may have naive timestamps,
    which we promote to local tz so comparisons against ``datetime.now()``
    don't blow up.
    """
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.astimezone()


def _format_temporal_activity(memories: list[dict]) -> str | None:
    """Bucket memorias by ``created`` / ``updated`` into 24h/7d/30d windows.

    Returns a single-line summary like
    ``24h: +2 nuevas, ~1 editadas · 7d: +5/3 · 30d: +12/8``, or ``None``
    when nothing falls inside any window (e.g. fresh-empty vault).

    A memory counts as "editada" only when ``updated`` is materially newer
    than ``created`` — otherwise every save would double-count itself
    (each save sets both fields to "now" by default).
    """
    now = datetime.now().astimezone()
    bins: list[tuple[str, timedelta]] = [
        ("24h", timedelta(days=1)),
        ("7d", timedelta(days=7)),
        ("30d", timedelta(days=30)),
    ]
    created_counts = {label: 0 for label, _ in bins}
    edited_counts = {label: 0 for label, _ in bins}

    for m in memories:
        c = _parse_iso(m.get("created"))
        u = _parse_iso(m.get("updated"))
        # An "edit" is only counted when the update timestamp is strictly
        # later than create — saving a brand-new memoria sets both to the
        # same value, which would otherwise inflate edited counts.
        is_edit = bool(c and u and (u - c) > timedelta(seconds=1))
        for label, span in bins:
            cutoff = now - span
            if c is not None and c >= cutoff:
                created_counts[label] += 1
            if is_edit and u is not None and u >= cutoff:
                edited_counts[label] += 1

    if not any(created_counts.values()) and not any(edited_counts.values()):
        return None
    parts = [
        f"{label}: +{created_counts[label]} nuevas · ~{edited_counts[label]} editadas"
        for label, _ in bins
    ]
    return " · ".join(parts)


def _format_metrics_24h(metrics_path: Path) -> str | None:
    """Aggregate the JSONL metrics sink for the last 24 h.

    Returns a one-liner summary (call count, p50/p95, error rate) or
    ``None`` when the sink is disabled, the file doesn't exist yet, or
    no rows fall inside the window. Re-uses the pure helpers in
    ``mem_vault.cli.metrics`` so the rendering stays in lockstep with
    ``mem-vault metrics`` — same percentile method, same filter logic,
    no drift between CLI and boot banner.
    """
    if not metrics_path.exists():
        return None
    try:
        from mem_vault.cli.metrics import (
            aggregate,
            filter_lines,
            iter_lines,
            parse_since,
            percentile,
        )
    except Exception:
        return None

    since = parse_since("24h")
    rows = filter_lines(iter_lines(metrics_path), since=since)
    if not rows:
        return None
    summary = aggregate(rows)
    total = summary["total"]
    if total <= 0:
        return None
    errors = summary["errors"]
    err_rate = (errors / total) * 100 if total else 0.0
    durations = [
        float(r["duration_ms"]) for r in rows if isinstance(r.get("duration_ms"), int | float)
    ]
    p50 = percentile(durations, 50)
    p95 = percentile(durations, 95)

    def _fmt(v: float | None) -> str:
        if v is None:
            return "—"
        return f"{v / 1000:.2f}s" if v >= 1000 else f"{v:.0f}ms"

    return (
        f"{total} llamadas · p50 {_fmt(p50)} · p95 {_fmt(p95)} · errores {errors} ({err_rate:.1f}%)"
    )


async def _gather_stats(service: Any, cwd: str | None) -> str:
    """Render the boot-banner "## Estadísticas del vault" block.

    Pulls four signals in order, each best-effort: every step is wrapped
    in its own ``try`` so one failure doesn't blank the whole block —
    mirrors the resilience contract of the Memorias section.

    1. **Totales + lint + top tags** via ``service.briefing(cwd)`` — the
       same dict the ``/mv`` boot briefing already consumes, so we share
       the corpus walk with any subsequent skill call inside the same
       process (``_corpus_cache`` TTL 30 s).
    2. **Actividad temporal** by walking ``service.list_({limit: ∞})`` and
       bucketing by ``created`` / ``updated``. Cheap because the corpus
       cache from step 1 is already warm.
    3. **Duplicates pendientes** via ``service.duplicates(threshold=0.7)``
       — tag-overlap Jaccard, no Qdrant. Surfaced only when ``count > 0``
       so a clean vault doesn't add noise.
    4. **Métricas operativas 24 h** via the JSONL sink, if enabled. We
       resolve the path through ``service.config.metrics_path`` rather
       than re-loading config so the hook honours whatever overrides the
       service was built with.

    Returns the empty string when nothing useful surfaced (fresh-empty
    vault, all queries failed, etc.) so ``_gather_context`` can decide
    not to emit a section header at all.
    """
    sections: list[str] = []

    # 1. Totals + lint + top tags ------------------------------------------------
    try:
        brief = await service.briefing({"cwd": cwd or ""})
        if isinstance(brief, dict) and brief.get("ok"):
            total_global = int(brief.get("total_global") or 0)
            project_total = int(brief.get("project_total") or 0)
            project_tag = brief.get("project_tag")
            top_tags = brief.get("top_tags") or []
            lint = brief.get("lint_summary") or {}

            total_line = f"- **Total**: {total_global} memorias"
            if project_tag:
                total_line += f" · `{project_tag}`: {project_total}"
            sections.append(total_line)

            if top_tags:
                top_str = ", ".join(f"`{t.get('tag')}` ({t.get('count')})" for t in top_tags[:5])
                sections.append(f"- **Top tags**: {top_str}")

            lint_issues: list[str] = []
            if lint.get("few_tags"):
                lint_issues.append(f"{lint['few_tags']} con <3 tags")
            if lint.get("no_aprendido"):
                lint_issues.append(f"{lint['no_aprendido']} sin 'Aprendido el'")
            if lint.get("short_body"):
                lint_issues.append(f"{lint['short_body']} con body corto")
            if lint_issues:
                sections.append(f"- **Lint**: {', '.join(lint_issues)}")
    except Exception as exc:
        print(f"mem-vault: stats briefing failed: {exc}", file=sys.stderr)

    # 2. Actividad temporal ------------------------------------------------------
    try:
        listing = await service.list_({"limit": 10**9})
        if isinstance(listing, dict) and listing.get("ok"):
            activity_line = _format_temporal_activity(listing.get("memories") or [])
            if activity_line:
                sections.append(f"- **Actividad**: {activity_line}")
    except Exception as exc:
        print(f"mem-vault: stats activity failed: {exc}", file=sys.stderr)

    # 3. Duplicates pendientes ---------------------------------------------------
    try:
        dups = await service.duplicates({"threshold": 0.7})
        if isinstance(dups, dict) and dups.get("ok"):
            count = int(dups.get("count") or 0)
            if count > 0:
                sections.append(
                    f"- **Duplicates** (jaccard ≥ 0.7): {count} pares pendientes — "
                    "`memory_duplicates` para revisar"
                )
    except Exception as exc:
        print(f"mem-vault: stats duplicates failed: {exc}", file=sys.stderr)

    # 4. Métricas operativas 24 h -----------------------------------------------
    try:
        cfg = getattr(service, "config", None)
        metrics_path = getattr(cfg, "metrics_path", None) if cfg is not None else None
        if metrics_path is not None:
            metrics_line = _format_metrics_24h(Path(metrics_path))
            if metrics_line:
                sections.append(f"- **Métricas 24h**: {metrics_line}")
    except Exception as exc:
        print(f"mem-vault: stats metrics failed: {exc}", file=sys.stderr)

    if not sections:
        return ""

    out: list[str] = ["## Estadísticas del vault (mem-vault)\n"]
    out.extend(sections)
    out.append("")
    return "\n".join(out)


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

    # Stats banner first — health check of the vault before any query.
    # Built independently (its own try blocks) so a failure here doesn't
    # blank the Memorias relevantes section that follows.
    stats_block = await _gather_stats(service, cwd)

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
        # No memorias section, but if stats succeeded we still want to
        # emit it — losing the recents shouldn't blank out the banner.
        return stats_block

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

    memorias_block = ""
    if len(lines) > 1:
        lines.append(
            "_Estas memorias vienen de tu vault Obsidian. "
            "Si necesitás más detalle, llamá a `memory_search` o `memory_get`._"
        )
        memorias_block = "\n".join(lines)

    blocks = [b for b in (stats_block, memorias_block) if b]
    return "\n\n".join(blocks)


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

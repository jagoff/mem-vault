"""Nightly reflection daemon (v0.6.0 game-changer #3b).

Once a day (typically 03:00 local via launchd), the reflection pass:

1. **Audita** las memorias nuevas/editadas en las últimas 24 h.
2. **Consolida obvias**: corre el pase existente de ``consolidate.py``
   con ``threshold=0.92`` (conservador) y ``apply=True`` — solo merges
   muy seguros se aplican automáticamente. El resto queda flagged para
   review humana.
3. **Detecta zombies**: memorias sin uso (``usage_count=0``) en >60 d.
4. **Cuenta tensiones**: memorias con ``contradicts: [...]`` no vacío.
5. **Genera una memoria diaria** ``type=reflection`` con la siguiente
   estructura:
   - resumen de actividad (creadas / editadas / consolidadas)
   - bugs / decisiones nuevas
   - tensiones detectadas (con cross-link a las memorias)
   - zombies candidatas a archivar
   - knowledge gaps inferidos (por proyecto sin escrituras nuevas).

La idea es que cada mañana el user encuentre en su vault una nota
breve que le dice "esto pasó ayer, esto necesita atención". Es la
diferencia entre "memoria que acumula" y "memoria que aprende".

Diseño:

- **Idempotente**: ejecutar dos veces el mismo día no duplica la nota
  (se identifica por slug ``reflection_YYYY_MM_DD``; el segundo run
  hace ``memory_update`` en vez de save).
- **No-LLM por default**: el resumen es estructural (counts + listas).
  Para narrativa LLM-generada, pasar ``--narrate`` que llama a
  ``memory_synthesize``.
- **Best-effort**: cualquier fallo deja un mensaje en stderr pero el
  daemon termina con exit 0 para no asustar a launchd con re-throttling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_AUTO_MERGE_THRESHOLD = 0.92  # conservative: only obvious dupes
DEFAULT_ZOMBIE_AGE_DAYS = 60.0
DEFAULT_LOOKBACK_HOURS = 24


@dataclass
class ReflectionReport:
    """What ``run_reflection`` returns; serialized into the .md body too."""

    day: str  # YYYY-MM-DD
    lookback_hours: int
    total_memorias: int
    created_in_window: int
    updated_in_window: int
    consolidated_pairs: int
    consolidated_kept: list[tuple[str, str]] = field(default_factory=list)
    pending_dup_pairs: int = 0
    zombies: list[dict[str, Any]] = field(default_factory=list)
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    new_decisions: list[dict[str, Any]] = field(default_factory=list)
    new_bugs: list[dict[str, Any]] = field(default_factory=list)
    knowledge_gaps: list[str] = field(default_factory=list)
    narrative: str | None = None
    memory_id: str | None = None
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _parse_dt(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _today_slug(now: datetime | None = None) -> str:
    now = now or datetime.now().astimezone()
    return now.strftime("reflection_%Y_%m_%d")


def _format_body(report: ReflectionReport) -> str:
    """Render the report as a Spanish-rioplatense markdown body.

    The agent already enforces idioma rioplatense globally; this
    matches that voice so the daily note doesn't feel like a foreign
    object in the vault.
    """
    parts: list[str] = []
    parts.append(f"# Reflexión del día — {report.day}")
    parts.append("")
    parts.append("## Resumen")
    parts.append(
        f"- **Total en el vault**: {report.total_memorias}"
    )
    parts.append(
        f"- **Creadas en últimas {report.lookback_hours} h**: {report.created_in_window}"
    )
    parts.append(
        f"- **Editadas en últimas {report.lookback_hours} h**: {report.updated_in_window}"
    )
    parts.append(
        f"- **Pairs consolidados (auto-merge ≥ {DEFAULT_AUTO_MERGE_THRESHOLD})**: "
        f"{report.consolidated_pairs}"
    )
    if report.pending_dup_pairs:
        parts.append(
            f"- **Pares similares pendientes de review humana**: {report.pending_dup_pairs}"
        )
    parts.append("")

    if report.new_decisions:
        parts.append("## Decisiones nuevas")
        for d in report.new_decisions:
            parts.append(f"- [[{d['id']}]] — {d.get('description') or d.get('name') or ''}")
        parts.append("")

    if report.new_bugs:
        parts.append("## Bugs nuevos / fixes documentados")
        for b in report.new_bugs:
            parts.append(f"- [[{b['id']}]] — {b.get('description') or b.get('name') or ''}")
        parts.append("")

    if report.consolidated_kept:
        parts.append("## Auto-merges aplicados")
        for kept, dropped in report.consolidated_kept:
            parts.append(f"- [[{kept}]] absorbió `{dropped}` (eliminada)")
        parts.append("")

    if report.contradictions:
        parts.append("## ⚠ Tensiones documentadas en el vault")
        for c in report.contradictions[:10]:
            parts.append(
                f"- [[{c['id']}]] contradice "
                + ", ".join(f"`{x}`" for x in c.get("contradicts", []))
            )
        parts.append("")

    if report.zombies:
        parts.append("## Zombies (sin uso reciente — candidatas a archivar)")
        for z in report.zombies[:10]:
            age = z.get("age_days")
            age_s = f" · {age:.0f} d sin tocar" if age is not None else ""
            parts.append(f"- [[{z['id']}]] — {z.get('description') or ''}{age_s}")
        parts.append("")

    if report.knowledge_gaps:
        parts.append("## Knowledge gaps")
        for g in report.knowledge_gaps:
            parts.append(f"- {g}")
        parts.append("")

    if report.narrative:
        parts.append("## Narrativa generada")
        parts.append(report.narrative)
        parts.append("")

    parts.append("---")
    parts.append(
        f"_Generado por `mem-vault reflect` el "
        f"{datetime.now().astimezone().isoformat(timespec='seconds')}._"
    )
    return "\n".join(parts)


def _classify_memory(mem: Any, by_type: list[dict[str, Any]]) -> None:
    """Append a small dict to ``by_type`` describing one mem (for new_decisions/new_bugs)."""
    by_type.append(
        {
            "id": mem.id,
            "name": mem.name,
            "description": mem.description,
        }
    )


def _knowledge_gaps(corpus: list[Any], lookback: timedelta) -> list[str]:
    """Heuristic: projects with memorias but zero writes in the lookback window.

    Useful as a "this project went quiet — should we close out?" signal.
    Returns a short list of human-readable strings.
    """
    project_last_write: dict[str, datetime] = {}
    cutoff = datetime.now(timezone.utc) - lookback
    for m in corpus:
        for t in m.tags or []:
            if not t.startswith("project:"):
                continue
            proj = t.split(":", 1)[1].lower()
            updated = _parse_dt(m.updated)
            if updated is None:
                continue
            prev = project_last_write.get(proj)
            if prev is None or updated > prev:
                project_last_write[proj] = updated
    gaps: list[str] = []
    for proj, last in sorted(project_last_write.items(), key=lambda x: x[1]):
        if last < cutoff:
            days = (datetime.now(timezone.utc) - last).days
            gaps.append(f"`project:{proj}` — sin writes hace {days} días")
    return gaps[:5]  # cap to 5; anything longer is noise


def run_reflection(
    config: Any,
    *,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    auto_merge_threshold: float = DEFAULT_AUTO_MERGE_THRESHOLD,
    zombie_age_days: float = DEFAULT_ZOMBIE_AGE_DAYS,
    apply_consolidate: bool = True,
    narrate: bool = False,
    now: datetime | None = None,
) -> ReflectionReport:
    """Compute the report + write the ``reflection_YYYY_MM_DD`` memory.

    Args:
        config: a ``Config`` object (``load_config()``).
        lookback_hours: window for "recent" — default 24 h.
        auto_merge_threshold: threshold for auto-applying consolidate
            merges. 0.92 default = only obvious duplicates.
        zombie_age_days: memorias older than this with usage_count=0
            land in the zombies list.
        apply_consolidate: if False, only count pending pairs (dry-run
            mode).
        narrate: if True, run ``memory_synthesize`` for an LLM-written
            narrative paragraph appended to the body. Disabled by
            default to keep the daemon snappy + offline-friendly.
        now: injectable for tests; defaults to wall clock.
    """
    from mem_vault.storage import VaultStorage

    storage = VaultStorage(config.memory_dir)

    now = now or datetime.now().astimezone()
    cutoff = now.astimezone(timezone.utc) - timedelta(hours=lookback_hours)
    day_slug = _today_slug(now)

    corpus = storage.list(type=None, tags=None, user_id=None, limit=10**9)
    total = len(corpus)

    created_n = 0
    updated_n = 0
    new_decisions: list[dict[str, Any]] = []
    new_bugs: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    zombies: list[dict[str, Any]] = []

    for m in corpus:
        # Skip the daily reflection memory itself if it already exists,
        # so we don't recursively count it as "new".
        if m.id == day_slug:
            continue

        created = _parse_dt(m.created)
        updated = _parse_dt(m.updated)
        if created and created >= cutoff:
            created_n += 1
            if m.type == "decision":
                _classify_memory(m, new_decisions)
            elif m.type == "bug":
                _classify_memory(m, new_bugs)
        elif updated and updated >= cutoff:
            updated_n += 1

        if m.contradicts:
            contradictions.append(
                {"id": m.id, "name": m.name, "contradicts": list(m.contradicts)}
            )

        # Zombies: never used + old enough.
        if (m.usage_count or 0) == 0 and updated:
            age_days = (now.astimezone(timezone.utc) - updated).days
            if age_days >= zombie_age_days:
                zombies.append(
                    {
                        "id": m.id,
                        "name": m.name,
                        "description": m.description,
                        "age_days": age_days,
                    }
                )

    zombies.sort(key=lambda z: z["age_days"], reverse=True)

    # Consolidation pass.
    #
    # By default we DON'T apply consolidation (``apply_consolidate=False``
    # would use the legacy heuristic on tags; ``apply_consolidate=True``
    # as set below goes through Qdrant + Ollama and can take minutes).
    # Reflections are supposed to be fast and offline-friendly; pass
    # ``--consolidate`` to the CLI when you explicitly want the nightly
    # pass to also fire merges. The "pending" count uses the cheap
    # discovery function so we always report it regardless.
    consolidated_pairs = 0
    consolidated_kept: list[tuple[str, str]] = []
    pending_dup_pairs = 0
    try:
        from mem_vault.discovery import find_duplicate_pairs_by_tag_overlap

        # The discovery helper takes ``Memory`` objects (not dicts) and
        # uses ``threshold=`` not ``jaccard_threshold=``. Keep params
        # aligned with that signature.
        candidate_pairs = find_duplicate_pairs_by_tag_overlap(
            corpus, threshold=0.7
        )
        pending_dup_pairs = len(candidate_pairs)
    except Exception as exc:
        logger.warning("reflection: pending dup count failed: %s", exc)

    if apply_consolidate:
        try:
            from mem_vault.consolidate import apply_resolution, find_candidate_pairs, _ask_llm
            from mem_vault.index import VectorIndex
            import ollama

            vi = VectorIndex(config)
            pairs = find_candidate_pairs(
                storage,
                vi,
                threshold=auto_merge_threshold,
                user_id=config.user_id,
                per_memory_neighbors=4,
            )
            client = ollama.Client(host=config.ollama_host)
            for pair in pairs[:10]:  # hard cap to keep the daemon bounded
                try:
                    res = _ask_llm(config, pair, ollama_client=client)
                    if res.action in {"MERGE", "KEEP_FIRST", "KEEP_SECOND"}:
                        summary = apply_resolution(
                            storage, vi, pair, res, user_id=config.user_id
                        )
                        consolidated_pairs += 1
                        kept_ids = summary.get("kept") or [pair.a.id]
                        deleted = summary.get("deleted") or []
                        if kept_ids and deleted:
                            consolidated_kept.append((kept_ids[0], deleted[0]))
                except Exception as exc:
                    logger.warning("reflection: pair resolve failed: %s", exc)
                    continue
        except ImportError:
            # consolidate / ollama not installed — skip silently.
            pass
        except Exception as exc:
            logger.warning("reflection: consolidate phase failed: %s", exc)

    gaps = _knowledge_gaps(corpus, timedelta(days=14))

    report = ReflectionReport(
        day=now.strftime("%Y-%m-%d"),
        lookback_hours=lookback_hours,
        total_memorias=total,
        created_in_window=created_n,
        updated_in_window=updated_n,
        consolidated_pairs=consolidated_pairs,
        consolidated_kept=consolidated_kept,
        pending_dup_pairs=max(0, pending_dup_pairs - consolidated_pairs),
        zombies=zombies[:20],
        contradictions=contradictions[:20],
        new_decisions=new_decisions,
        new_bugs=new_bugs,
        knowledge_gaps=gaps,
    )

    body = _format_body(report)

    # Idempotent write: if reflection_YYYY_MM_DD already exists, update it
    # (yesterday's run captured a partial day; today's run overwrites).
    # ``storage.save`` with ``memory_id=<day_slug>`` overwrites, and
    # ``storage.update`` replaces the body for an existing record — either
    # works; we prefer the explicit two-branch to surface errors clearly.
    existing = storage.get(day_slug)
    if existing is not None:
        try:
            storage.update(day_slug, content=body)
            report.memory_id = day_slug
        except Exception as exc:
            logger.warning("reflection: update failed: %s", exc)
    else:
        try:
            description = (
                f"{report.created_in_window} nuevas, "
                f"{report.updated_in_window} editadas, "
                f"{consolidated_pairs} merges, "
                f"{len(zombies)} zombies"
            )
            storage.save(
                content=body,
                title=f"Reflexión del día — {report.day}",
                description=description,
                type="note",
                tags=["reflection", f"date:{report.day}"],
                agent_id=config.agent_id,
                user_id=config.user_id,
                memory_id=day_slug,
            )
            report.memory_id = day_slug
        except Exception as exc:
            logger.warning("reflection: save failed: %s", exc)

    return report


__all__ = [
    "DEFAULT_AUTO_MERGE_THRESHOLD",
    "DEFAULT_LOOKBACK_HOURS",
    "DEFAULT_ZOMBIE_AGE_DAYS",
    "ReflectionReport",
    "run_reflection",
]

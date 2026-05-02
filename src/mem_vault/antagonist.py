"""Antagonist mode (v0.6.0 game-changer #3): the vault interrupts when
the agent is about to repeat something it already disagreed with.

How it works:

1. **At Stop time**: after the existing citation-detection step in
   ``hooks.stop``, the antagonist inspects the agent's last response
   and looks for memorias that the agent cited (or that the response
   touches semantically) which carry a non-empty ``contradicts: [...]``
   frontmatter list. Each such pair becomes a "pending warning" item:
   "you leaned on X, but X has documented tensions with Y".

2. **Persistence**: pending items are written to
   ``<state_dir>/pending_contradictions.json`` with a timestamp. Best-
   effort: a write failure is logged + ignored.

3. **At SessionStart / UserPromptSubmit**: the next hook reads the JSON,
   injects each pending item as an ``additionalContext`` block for the
   agent, then clears the file. The warning surfaces *exactly once* —
   right at the start of the next turn after the one that introduced
   the conflict.

Why deterministic instead of LLM-based:

We deliberately don't call Ollama from the Stop hook (latency adds up
to the close-of-turn delay the user already perceives). Instead the
detector relies on the structure already in the vault: ``contradicts``
field on each memory's frontmatter (populated at ``memory_save`` time
via ``auto_contradict`` when enabled). The agent saved the tension; the
antagonist just surfaces it at the moment it's relevant.

Knobs:

- ``MEM_VAULT_ANTAGONIST=1`` — enable (default off, opt-in feature).
- ``MEM_VAULT_ANTAGONIST_TTL_S`` — pending TTL in seconds; older items
  are dropped on read. Default 24 h.
- ``MEM_VAULT_ANTAGONIST_MAX`` — cap on items injected per turn. Default 3.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PENDING_FILENAME = "pending_contradictions.json"
DEFAULT_TTL_S = 24 * 3600  # 24 hours
DEFAULT_MAX_ITEMS = 3


def is_enabled() -> bool:
    return os.environ.get("MEM_VAULT_ANTAGONIST", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ttl_s() -> int:
    raw = os.environ.get("MEM_VAULT_ANTAGONIST_TTL_S", "").strip()
    if not raw:
        return DEFAULT_TTL_S
    try:
        return max(60, int(raw))
    except ValueError:
        return DEFAULT_TTL_S


def _max_items() -> int:
    raw = os.environ.get("MEM_VAULT_ANTAGONIST_MAX", "").strip()
    if not raw:
        return DEFAULT_MAX_ITEMS
    try:
        return max(1, min(10, int(raw)))
    except ValueError:
        return DEFAULT_MAX_ITEMS


@dataclass
class PendingContradiction:
    """One pending warning to inject at the next turn.

    ``cited_id`` is the memory the agent leaned on; ``contradicts_ids``
    are the memorias listed under ``cited.contradicts``. The render hook
    formats this into a human-readable block.
    """

    cited_id: str
    cited_name: str | None
    cited_description: str | None
    contradicts_ids: list[str]
    contradicts_summaries: list[dict[str, str | None]]
    detected_at: float = field(default_factory=time.time)
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pending_path(state_dir: Path) -> Path:
    return Path(state_dir) / PENDING_FILENAME


def _load_storage_for_corpus():
    """Lazy import of config + storage. Returns ``(cfg, storage)`` or ``(None, None)``.

    Kept inside a helper because the antagonist runs from the Stop /
    SessionStart hooks, which exit 0 even when the import fails.
    """
    try:
        from mem_vault.config import load_config
        from mem_vault.storage import VaultStorage
    except Exception:
        return None, None
    try:
        cfg = load_config()
    except Exception:
        return None, None
    return cfg, VaultStorage(cfg.memory_dir)


def detect_from_citations(
    cited_ids: list[str],
    *,
    storage: Any | None = None,
) -> list[PendingContradiction]:
    """Look up each cited memory and return pending items for those with contradictions.

    ``storage`` is injected for tests; in production the helper loads
    the config + storage itself.
    """
    if not cited_ids:
        return []
    if storage is None:
        _, storage = _load_storage_for_corpus()
        if storage is None:
            return []

    out: list[PendingContradiction] = []
    for mid in cited_ids:
        try:
            mem = storage.get(mid)
        except Exception as exc:
            logger.warning("antagonist: storage.get(%s) failed: %s", mid, exc)
            continue
        if mem is None or not mem.contradicts:
            continue
        # Resolve each contradicting id to a (name, description) summary
        # so the warning has something to render. Missing memorias
        # (deleted out-of-band) are skipped silently.
        summaries: list[dict[str, str | None]] = []
        for cid in mem.contradicts:
            try:
                cmem = storage.get(cid)
            except Exception:
                continue
            if cmem is None:
                continue
            summaries.append(
                {
                    "id": cmem.id,
                    "name": cmem.name,
                    "description": cmem.description,
                }
            )
        if not summaries:
            continue
        out.append(
            PendingContradiction(
                cited_id=mem.id,
                cited_name=mem.name,
                cited_description=mem.description,
                contradicts_ids=[s["id"] or "" for s in summaries],
                contradicts_summaries=summaries,
            )
        )
    return out


def write_pending(
    state_dir: Path, items: list[PendingContradiction]
) -> bool:
    """Persist ``items`` to the pending file. Best-effort; returns success."""
    if not items:
        return True
    try:
        path = pending_path(state_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Append-merge: when there's already a pending block we don't
        # blow it away (the previous turn might still be unread). We
        # union by cited_id so a repeated detection doesn't multiply.
        existing: list[dict[str, Any]] = []
        if path.exists():
            try:
                blob = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(blob, dict):
                    existing = list(blob.get("items") or [])
                elif isinstance(blob, list):
                    existing = list(blob)
            except Exception:
                existing = []
        seen = {e.get("cited_id") for e in existing}
        merged = list(existing)
        for item in items:
            d = item.to_dict()
            if d["cited_id"] in seen:
                # Update the timestamp rather than duplicating.
                for e in merged:
                    if e.get("cited_id") == d["cited_id"]:
                        e["detected_at"] = d["detected_at"]
                        e["contradicts_summaries"] = d["contradicts_summaries"]
                        e["contradicts_ids"] = d["contradicts_ids"]
                        break
            else:
                merged.append(d)
                seen.add(d["cited_id"])
        path.write_text(
            json.dumps(
                {"version": 1, "items": merged}, indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )
        return True
    except Exception as exc:
        logger.warning("antagonist: write_pending failed: %s", exc)
        return False


def read_pending(
    state_dir: Path,
    *,
    ttl_s: int | None = None,
    max_items: int | None = None,
) -> list[PendingContradiction]:
    """Load + filter pending items. Returns empty list when nothing pending.

    Items older than ``ttl_s`` are dropped (the warning is no longer
    timely — the conversation has moved on). Cap to ``max_items`` to
    avoid drowning the next turn.
    """
    ttl = ttl_s if ttl_s is not None else _ttl_s()
    cap = max_items if max_items is not None else _max_items()
    path = pending_path(state_dir)
    if not path.exists():
        return []
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("antagonist: read_pending parse failed: %s", exc)
        return []
    raw_items: list[dict[str, Any]] = []
    if isinstance(blob, dict):
        raw_items = list(blob.get("items") or [])
    elif isinstance(blob, list):
        raw_items = list(blob)

    now = time.time()
    out: list[PendingContradiction] = []
    for d in raw_items:
        if not isinstance(d, dict):
            continue
        try:
            ts = float(d.get("detected_at") or 0)
        except (TypeError, ValueError):
            ts = 0
        if ttl > 0 and (now - ts) > ttl:
            continue
        out.append(
            PendingContradiction(
                cited_id=str(d.get("cited_id") or ""),
                cited_name=d.get("cited_name"),
                cited_description=d.get("cited_description"),
                contradicts_ids=list(d.get("contradicts_ids") or []),
                contradicts_summaries=list(d.get("contradicts_summaries") or []),
                detected_at=ts,
                session_id=d.get("session_id"),
            )
        )
    # Newest first — the most recent tension is the one most likely
    # still on the user's mind.
    out.sort(key=lambda p: p.detected_at, reverse=True)
    return out[:cap] if cap > 0 else out


def clear_pending(state_dir: Path) -> None:
    """Remove the pending file after rendering. Idempotent."""
    try:
        path = pending_path(state_dir)
        if path.exists():
            path.unlink()
    except Exception as exc:
        logger.warning("antagonist: clear_pending failed: %s", exc)


def render_warning_block(items: list[PendingContradiction]) -> str:
    """Format pending items into a human-readable additionalContext block.

    The output is meant to be appended to the SessionStart /
    UserPromptSubmit additionalContext so the agent sees it as a
    system message at the top of the next turn.
    """
    if not items:
        return ""
    lines = [
        "",
        "## ⚠ Antagonist — contradicciones pendientes del turno anterior",
        "",
        "El turno anterior te apoyaste en memorias que ya tienen tensiones documentadas:",
        "",
    ]
    for item in items:
        cited_label = (
            f"`{item.cited_id}`" + (f" — {item.cited_name}" if item.cited_name else "")
        )
        lines.append(f"- **{cited_label}** contradice:")
        for s in item.contradicts_summaries:
            sid = s.get("id") or ""
            sname = s.get("name") or ""
            sdesc = s.get("description") or ""
            label = f"`{sid}`" + (f" — {sname}" if sname else "")
            if sdesc:
                lines.append(f"  - {label}: {sdesc}")
            else:
                lines.append(f"  - {label}")
        lines.append("")
    lines.append(
        "Reconciliá explícitamente antes de continuar: confirmá la memoria vigente o "
        "actualizá la antigua. No repitas la decisión sin abordar la tensión."
    )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MAX_ITEMS",
    "DEFAULT_TTL_S",
    "PENDING_FILENAME",
    "PendingContradiction",
    "clear_pending",
    "detect_from_citations",
    "is_enabled",
    "pending_path",
    "read_pending",
    "render_warning_block",
    "write_pending",
]

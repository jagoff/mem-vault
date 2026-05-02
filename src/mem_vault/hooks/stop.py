"""Stop hook: session-close bookkeeping + auto-detect memory citations.

Two responsibilities, both best-effort + non-blocking:

1. **Audit line**: append tab-separated breadcrumb to
   ``~/.local/share/mem-vault/sessions.log`` so the user can later ask
   "when did I last close a session in this cwd, and how many memorias
   were in the vault then?".

2. **Auto-feedback via citation**: scan the agent's last response for
   mentions of memory ids (``[[id]]`` wikilinks, backticked ``` `id` ```
   mentions, plain word-bounded references). Every match bumps the
   cited memory's ``usage_count`` + ``last_used`` via
   ``VaultStorage.record_feedback(helpful=None)`` — a neutral "this was
   used" signal. Explicit thumbs (``helpful=True/False``) remain the
   domain of the ``memory_feedback`` MCP tool.

The citation detection is conservative: we pre-filter candidate ids to
the set that actually exists in ``memory_dir``, then scan the response
with a compiled-once regex. False positives are negligible because
memory ids are specific slugs (``feedback_idioma_preferido``), not
common English words. Disable the whole auto-feedback path by setting
``MEM_VAULT_STOP_AUTO_FEEDBACK=0``.

Never calls the LLM. Never blocks the agent. Exits 0 on any failure.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Default soft cap for sessions.log: 50 MiB. Configurable via env so tests can
# use a tiny cap. We rotate to ``sessions.log.1`` (single rotation, no compression)
# when the live file exceeds the cap. Best-effort — failure to rotate must NOT
# break the hook (contract: hooks always exit 0).
_DEFAULT_SESSIONS_LOG_MAX_BYTES = 52_428_800  # 50 MiB


def _sessions_log_max_bytes() -> int:
    raw = os.environ.get("MEM_VAULT_SESSIONS_LOG_MAX_BYTES", "").strip()
    if not raw:
        return _DEFAULT_SESSIONS_LOG_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_SESSIONS_LOG_MAX_BYTES
    # Negative or zero disables the cap (no rotation).
    return value if value > 0 else 0


def _maybe_rotate_sessions_log(log_file: Path) -> None:
    """Rotate ``log_file`` to ``log_file.1`` once it exceeds the soft cap.

    Single rotation, no compression. Wrapped fully in try/except by the caller
    so a rotation failure (permissions, FS full, etc.) is a no-op rather than
    a crash — the contract for the Stop hook is "always exit 0".
    """
    cap = _sessions_log_max_bytes()
    if cap <= 0:
        return
    try:
        size = log_file.stat().st_size
    except FileNotFoundError:
        return
    if size <= cap:
        return
    rotated = log_file.with_suffix(log_file.suffix + ".1")
    # ``Path.replace`` is atomic on POSIX (rename(2)). Overwrites an existing
    # ``.1`` if present — we keep only one historical file by design.
    log_file.replace(rotated)


def _vault_memory_count() -> int:
    try:
        from mem_vault.config import load_config
    except Exception:
        return -1
    try:
        cfg = load_config()
    except Exception:
        return -1
    try:
        return sum(1 for _ in cfg.memory_dir.glob("*.md"))
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Citation detection — pure helpers
# ---------------------------------------------------------------------------


def find_memory_citations(text: str, known_ids: set[str]) -> set[str]:
    """Return the subset of ``known_ids`` that appear as citations in ``text``.

    Recognized citation forms (all case-sensitive — slugs are lowercase):

    - ``[[<id>]]`` — Obsidian wikilink. Strongest signal.
    - ``` `<id>` ``` — Markdown inline code span. The agent often uses
      this when naming a memory without linking.
    - Bare word-bounded ``<id>``. Cheapest catch, but risky if the slug
      is short; we mitigate by pre-filtering against ``known_ids``.

    We return a set (dedup within a response). An empty input or no
    matches returns ``set()``.
    """
    if not text or not known_ids:
        return set()
    hits: set[str] = set()
    # ``[[id]]`` wikilinks — first pass
    for match in re.finditer(r"\[\[([a-z0-9_][a-z0-9_\-]{1,80})\]\]", text):
        mid = match.group(1)
        if mid in known_ids:
            hits.add(mid)
    # Inline code spans ``...`` — second pass
    for match in re.finditer(r"`([a-z0-9_][a-z0-9_\-]{1,80})`", text):
        mid = match.group(1)
        if mid in known_ids:
            hits.add(mid)
    # Bare word-bounded mentions — only for ids that survived the filter
    # and are reasonably specific (≥8 chars, reduces false positives like
    # matching ``body`` or ``notes`` if a memory happens to be named that).
    specific = {mid for mid in known_ids if len(mid) >= 8}
    for mid in specific - hits:  # skip ids we already matched
        if re.search(rf"\b{re.escape(mid)}\b", text):
            hits.add(mid)
    return hits


def _load_response_text(payload: dict[str, Any]) -> str:
    """Best-effort extraction of the agent's last response from the payload.

    Supports:
    - ``payload["response"]`` / ``payload["assistant_message"]``: direct.
    - ``payload["transcript_path"]`` (Claude Code convention): jsonl file
      where each line is a ``{"role": "...", "content": "..."}`` turn.
      We read the last ``assistant`` entry.

    Returns an empty string on any failure or missing data — the hook
    degrades to the legacy audit-line-only behavior.
    """
    for key in ("response", "assistant_message", "message"):
        v = payload.get(key)
        if isinstance(v, str) and v.strip():
            return v

    transcript = payload.get("transcript_path") or payload.get("transcript")
    if not transcript:
        return ""
    path = Path(str(transcript)).expanduser()
    if not path.exists() or not path.is_file():
        return ""
    try:
        last_assistant = ""
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    turn = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(turn, dict):
                    continue
                role = turn.get("role") or (turn.get("message") or {}).get("role")
                if role != "assistant":
                    continue
                # Different SDKs nest content differently — try the common shapes.
                content = turn.get("content")
                if isinstance(content, list):
                    # Anthropic style: content is a list of {type, text} blocks.
                    parts = [b.get("text", "") for b in content if isinstance(b, dict)]
                    last_assistant = "\n".join(p for p in parts if p)
                elif isinstance(content, str):
                    last_assistant = content
                else:
                    msg = turn.get("message") or {}
                    msg_content = msg.get("content")
                    if isinstance(msg_content, str):
                        last_assistant = msg_content
                    elif isinstance(msg_content, list):
                        parts = [b.get("text", "") for b in msg_content if isinstance(b, dict)]
                        last_assistant = "\n".join(p for p in parts if p)
        return last_assistant
    except Exception:
        return ""


def _apply_auto_feedback(payload: dict[str, Any]) -> int:
    """Scan the response for memory citations; bump counters on matches.

    Returns the number of memories that had their ``last_used`` bumped.
    Safe to call with any payload shape — returns 0 on any missing piece.

    Controlled by ``MEM_VAULT_STOP_AUTO_FEEDBACK`` (default on); set to
    ``0`` to disable without removing the hook.
    """
    if os.environ.get("MEM_VAULT_STOP_AUTO_FEEDBACK", "1").lower() in {"0", "false", "no", "off"}:
        return 0

    response = _load_response_text(payload)
    if not response:
        return 0

    try:
        from mem_vault.config import load_config
        from mem_vault.storage import VaultStorage
    except Exception:
        return 0
    try:
        cfg = load_config()
    except Exception:
        return 0

    try:
        known_ids = {p.stem for p in cfg.memory_dir.glob("*.md")}
    except Exception:
        return 0
    if not known_ids:
        return 0

    cited = find_memory_citations(response, known_ids)
    if not cited:
        return 0

    storage = VaultStorage(cfg.memory_dir)
    bumped = 0
    for mid in cited:
        try:
            mem = storage.record_feedback(mid, helpful=None)
            if mem is not None:
                bumped += 1
        except Exception:
            # Storage errors (disk full, corrupt file) must not break
            # the Stop hook. Silent skip — the user can always re-run.
            continue

    # Telemetry: flip ``was_cited=1`` on the most-recent search-event row
    # for each cited memory id. This is the supervised signal that
    # ``mem-vault ranker train`` fits on — every time the agent's final
    # response actually references a memory, the row that surfaced it
    # gets stamped as a positive example.
    #
    # Best-effort: a missing telemetry DB (telemetry disabled, brand-new
    # vault, mid-migration) just returns 0. Never raises.
    if os.environ.get("MEM_VAULT_TELEMETRY", "1").lower() not in {"0", "false", "no", "off"}:
        try:
            from mem_vault import telemetry as _telemetry

            session_id = (
                payload.get("session_id")
                or os.environ.get("MEM_VAULT_SESSION_ID")
            )
            _telemetry.mark_cited(
                cfg.state_dir,
                cited,
                session_id=session_id if isinstance(session_id, str) else None,
            )
        except Exception:
            # Telemetry failures must never bubble — the Stop hook's
            # contract is "always exit 0".
            pass
    return bumped


def run() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    # Session audit line (same as before — keeps sessions.log stable)
    log_dir = Path.home() / ".local" / "share" / "mem-vault"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "sessions.log"

    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    cwd = os.environ.get("DEVIN_PROJECT_DIR") or os.getcwd()
    stop_active = bool(payload.get("stop_hook_active"))
    count = _vault_memory_count()

    # Auto-feedback via citation detection
    bumped = 0
    try:
        bumped = _apply_auto_feedback(payload)
    except Exception as exc:
        print(f"mem-vault: auto-feedback failed: {exc}", file=sys.stderr)

    line = (
        f"{now}\tstop\tcwd={cwd}\tmemories={count}"
        f"\tstop_hook_active={stop_active}\tauto_feedback={bumped}\n"
    )

    # Best-effort rotation BEFORE the write so the new line lands in the fresh
    # file. Wrapped in its own try/except — if rotation fails (e.g. the target
    # is unwritable) we still try to append to the existing file. Hook contract
    # forbids any non-zero exit here.
    try:
        _maybe_rotate_sessions_log(log_file)
    except Exception as exc:
        print(f"mem-vault: stop_log rotate failed: {exc}", file=sys.stderr)

    # POSIX-atomic append: ``open(path, "a")`` opens with ``O_APPEND``, which
    # the kernel guarantees atomic for a single ``write(2)`` call up to
    # ``PIPE_BUF`` bytes (≥512 by POSIX, 4096 on Linux/macOS). Our line is
    # well under that ceiling, so concurrent Claude Code + Devin sessions
    # cannot interleave bytes within a single line. We deliberately do ONE
    # ``f.write(line)`` with a complete, newline-terminated string.
    try:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:
        print(f"mem-vault: stop_log write failed: {exc}", file=sys.stderr)

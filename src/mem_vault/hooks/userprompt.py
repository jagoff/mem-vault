"""UserPromptSubmit hook: inject memories relevant to the user's current prompt.

Whenever the user types a message, this hook does a semantic search against
the local mem-vault index, picks the top-k most relevant memories, and emits
them as ``hookSpecificOutput.additionalContext`` so the agent sees them
*before* processing the prompt. Functionally similar to what the mem0 plugin
for Claude Code does on every prompt.

Conservative by default to avoid bloating the context window:

- Prompts shorter than ``MIN_PROMPT_CHARS`` (20 by default, override with
  ``MEM_VAULT_USERPROMPT_MIN_CHARS``) are skipped — short replies like "ok"
  or "sí" don't carry enough signal to retrieve anything useful.
- Slash-commands (``/recap``, ``/clar``, etc.) are skipped — those activate
  a different skill flow that already brings its own context.
- Locale-aware skip: when ``MEM_VAULT_USERPROMPT_SCRIPTS`` is set (e.g.
  ``"latin"`` or ``"latin,cyrillic"``), prompts that are predominantly
  written in a non-listed Unicode script are skipped. Useful when your
  vault is Spanish/English and you don't want the embedder to spend
  cycles on accidentally-pasted CJK / Arabic / Hebrew / Devanagari text.
- Default top-k is 3 (override with ``MEM_VAULT_USERPROMPT_K``); default
  similarity threshold is 0.35 (override with ``MEM_VAULT_USERPROMPT_THRESHOLD``).
- Hook timeout should be ≥10 s to comfortably absorb the embedding call.

Like every other hook in this package, it is best-effort: any failure is
printed to stderr and produces an empty stdout (no-op).

**Canary log** (added 2026-05-01): every invocation appends one JSON line
to ``<state_dir>/hooks/userprompt.log`` (override path with
``MEM_VAULT_USERPROMPT_LOG``) recording the timestamp, prompt preview,
skip reason, results count and latency. Without this we couldn't tell
whether the hook fires at all, fires and skips, or fires and times out
silently — every observable failure mode for the auto-search-first feature
looks the same from the agent's POV (no ``additionalContext`` injected).
The log is plain JSONL, append-only, best-effort: write failures never
abort the hook.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

MIN_PROMPT_CHARS = int(os.environ.get("MEM_VAULT_USERPROMPT_MIN_CHARS", "20"))
TOP_K = int(os.environ.get("MEM_VAULT_USERPROMPT_K", "3"))
THRESHOLD = float(os.environ.get("MEM_VAULT_USERPROMPT_THRESHOLD", "0.35"))
MAX_SNIPPET = int(os.environ.get("MEM_VAULT_USERPROMPT_MAX_SNIPPET", "240"))
SCRIPTS_RAW = os.environ.get("MEM_VAULT_USERPROMPT_SCRIPTS", "").strip()
SCRIPT_DOMINANCE_THRESHOLD = float(os.environ.get("MEM_VAULT_USERPROMPT_SCRIPT_RATIO", "0.6"))


def _default_log_path() -> Path:
    """Resolve the default canary log path lazily (avoids importing Config
    at module top — keeps the hook startup fast on cold cache)."""
    override = os.environ.get("MEM_VAULT_USERPROMPT_LOG", "").strip()
    if override:
        return Path(override).expanduser()
    # Mirror Config._default_state_dir without importing Config (faster).
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "mem-vault"
    elif sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) / "mem-vault" if local else Path.home() / "mem-vault"
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) / "mem-vault" if xdg else Path.home() / ".local" / "share" / "mem-vault"
    return base / "hooks" / "userprompt.log"


def _log_canary(
    *,
    prompt: str,
    skip_reason: str | None,
    results_count: int,
    latency_ms: float,
    exc: str | None = None,
) -> None:
    """Append one JSON line per invocation. Best-effort: never raises.

    The log accumulates; rotate manually or via launchd's log rotation if
    it grows beyond a few MB. JSONL keeps it `jq`-friendly for ad-hoc
    audits like ``jq 'select(.skip_reason==null and .results_count==0)'
    < userprompt.log | tail`` to find prompts that triggered an embed
    call but matched nothing.
    """
    try:
        path = _default_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        entry: dict[str, object] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "prompt_preview": prompt[:80],
            "prompt_len": len(prompt),
            "skip_reason": skip_reason,
            "results_count": results_count,
            "latency_ms": round(latency_ms, 1),
        }
        if exc:
            entry["exc"] = exc
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as inner_exc:
        # Last-resort stderr breadcrumb so the failure isn't completely
        # silent. Printing to stderr is what other failure paths in this
        # hook already do (see ``run`` below) — same channel, same UX.
        print(f"mem-vault: canary log failed: {inner_exc}", file=sys.stderr)


# We use unicodedata.name() to bucket each letter into a coarse script. The
# names are stable across Python versions and cover everything we care about
# without an extra dep like ``regex`` or ``langid``.
_SCRIPT_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("latin", ("LATIN ",)),
    ("cyrillic", ("CYRILLIC ",)),
    ("greek", ("GREEK ",)),
    ("arabic", ("ARABIC ",)),
    ("hebrew", ("HEBREW ",)),
    ("devanagari", ("DEVANAGARI ",)),
    (
        "cjk",
        (
            "CJK ",
            "HIRAGANA ",
            "KATAKANA ",
            "HANGUL ",
            "BOPOMOFO ",
            "YI ",
        ),
    ),
    ("thai", ("THAI ",)),
    ("ethiopic", ("ETHIOPIC ",)),
)


def detect_script(text: str) -> str:
    """Return the dominant Unicode script bucket of ``text`` ('latin', 'cjk', …).

    Only counts letters (skips digits, whitespace, punctuation, emoji). The
    result is the bucket with the largest letter count; ``"unknown"`` when
    fewer than three letters are present overall.
    """
    counts: dict[str, int] = {}
    total_letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        total_letters += 1
        for bucket, prefixes in _SCRIPT_BUCKETS:
            if any(name.startswith(p) for p in prefixes):
                counts[bucket] = counts.get(bucket, 0) + 1
                break
    if total_letters < 3:
        return "unknown"
    if not counts:
        return "unknown"
    # Wrap ``counts.get`` so mypy sees a non-Optional return — ``dict.get``
    # alone is typed as returning ``int | None`` and confuses ``max(key=...)``.
    return max(counts, key=lambda k: counts[k])


def _allowed_scripts() -> set[str]:
    return {s.strip().lower() for s in SCRIPTS_RAW.split(",") if s.strip()}


def _should_skip(prompt: str) -> str | None:
    """Return a short reason string if the prompt should not trigger a search."""
    stripped = prompt.strip()
    if not stripped:
        return "empty"
    if len(stripped) < MIN_PROMPT_CHARS:
        return f"too_short(<{MIN_PROMPT_CHARS})"
    if stripped.startswith("/"):
        # /recap, /clar, /ship, etc. — handled by their own skill flows
        return "slash_command"

    allowed = _allowed_scripts()
    if allowed:
        script = detect_script(stripped)
        # ``unknown`` (mostly numbers, emoji, or very short alphabetic text)
        # is allowed through — better to over-search than miss a real query.
        if script != "unknown" and script not in allowed:
            return f"script={script}_not_in_{sorted(allowed)}"
    return None


async def _gather_context(prompt: str) -> tuple[str, int]:
    """Return ``(injected_context, results_count)``. Empty string + 0 on
    any failure path so the canary log can still record what happened.
    """
    try:
        from mem_vault.server import build_service
    except Exception as exc:
        print(f"mem-vault: import failed: {exc}", file=sys.stderr)
        return "", 0

    try:
        service = build_service()
    except Exception as exc:
        print(f"mem-vault: service init failed: {exc}", file=sys.stderr)
        return "", 0
    try:
        result = await service.search(
            {
                "query": prompt,
                "k": TOP_K,
                "threshold": THRESHOLD,
            }
        )
    except Exception as exc:
        print(f"mem-vault: search failed: {exc}", file=sys.stderr)
        return "", 0

    if not result.get("ok") or not result.get("results"):
        return "", 0

    lines: list[str] = ["## Memorias relevantes al mensaje actual (mem-vault)\n"]
    # Dedup by memory id — ``memory_search`` should already collapse duplicates
    # via the hybrid+rerank path, but we belt-and-suspender it here so a buggy
    # ranker can never inject the same memoria twice into the prompt.
    seen: set[str] = set()
    for hit in result["results"]:
        mem = hit.get("memory") or {}
        mid = hit.get("id") or mem.get("id") or "?"
        if mid != "?" and mid in seen:
            continue
        seen.add(mid)
        name = mem.get("name") or mid
        descr = (mem.get("description") or hit.get("snippet") or "").strip().replace("\n", " ")
        score = hit.get("score")
        score_tag = f" (score {score:.2f})" if isinstance(score, (int, float)) else ""
        lines.append(f"- **{name}**{score_tag} — {descr[:MAX_SNIPPET]}")

    count = len(seen)
    if count == 0:
        return "", 0
    lines.append("\n_Si necesitás el cuerpo completo de alguna, llamá a `memory_get` con su id._")
    return "\n".join(lines), count


def run() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        payload = {}

    prompt = str(payload.get("prompt") or payload.get("user_prompt") or "")
    started = time.monotonic()
    skip_reason = _should_skip(prompt)
    if skip_reason:
        # No-op: empty stdout + exit 0. We log to stderr so users can debug
        # via Devin's hook output if a prompt unexpectedly didn't trigger,
        # and append to the canary log so we can audit skip rates over time.
        latency_ms = (time.monotonic() - started) * 1000
        _log_canary(
            prompt=prompt,
            skip_reason=skip_reason,
            results_count=0,
            latency_ms=latency_ms,
        )
        print(f"mem-vault: skip ({skip_reason})", file=sys.stderr)
        return

    context = ""
    results_count = 0
    exc_str: str | None = None
    try:
        context, results_count = asyncio.run(_gather_context(prompt))
    except Exception as exc:
        exc_str = f"{type(exc).__name__}: {exc}"
        print(f"mem-vault: UserPromptSubmit hook failed: {exc}", file=sys.stderr)

    # Antagonist warnings (v0.6.0): drain pending contradictions and
    # prepend them to the additionalContext. The Stop hook of the
    # previous turn enqueues these when the agent leans on a memory
    # that has documented contradictions; the user's next prompt is
    # the right moment to surface them. Best-effort: failure here
    # never blocks the hook (contract: exit 0).
    antagonist_block = ""
    try:
        from mem_vault import antagonist as _antagonist
        from mem_vault.config import load_config

        if _antagonist.is_enabled():
            cfg = load_config()
            pending = _antagonist.read_pending(cfg.state_dir)
            if pending:
                antagonist_block = _antagonist.render_warning_block(pending)
                _antagonist.clear_pending(cfg.state_dir)
    except Exception as exc:
        print(f"mem-vault: antagonist render failed: {exc}", file=sys.stderr)

    latency_ms = (time.monotonic() - started) * 1000
    _log_canary(
        prompt=prompt,
        skip_reason=None,
        results_count=results_count,
        latency_ms=latency_ms,
        exc=exc_str,
    )

    # Combine antagonist + memorias context. Antagonist goes FIRST so
    # the warning is the first thing the agent reads.
    parts = [b for b in (antagonist_block, context) if b]
    if not parts:
        return
    full_context = "\n\n".join(parts)

    out = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": full_context,
        }
    }
    json.dump(out, sys.stdout, ensure_ascii=False)

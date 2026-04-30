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
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unicodedata

MIN_PROMPT_CHARS = int(os.environ.get("MEM_VAULT_USERPROMPT_MIN_CHARS", "20"))
TOP_K = int(os.environ.get("MEM_VAULT_USERPROMPT_K", "3"))
THRESHOLD = float(os.environ.get("MEM_VAULT_USERPROMPT_THRESHOLD", "0.35"))
MAX_SNIPPET = int(os.environ.get("MEM_VAULT_USERPROMPT_MAX_SNIPPET", "240"))
SCRIPTS_RAW = os.environ.get("MEM_VAULT_USERPROMPT_SCRIPTS", "").strip()
SCRIPT_DOMINANCE_THRESHOLD = float(os.environ.get("MEM_VAULT_USERPROMPT_SCRIPT_RATIO", "0.6"))


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


async def _gather_context(prompt: str) -> str:
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
        return ""

    if not result.get("ok") or not result.get("results"):
        return ""

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

    if len(lines) <= 1:
        return ""
    lines.append("\n_Si necesitás el cuerpo completo de alguna, llamá a `memory_get` con su id._")
    return "\n".join(lines)


def run() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        payload = {}

    prompt = str(payload.get("prompt") or payload.get("user_prompt") or "")
    skip_reason = _should_skip(prompt)
    if skip_reason:
        # No-op: empty stdout + exit 0. We log to stderr so users can debug
        # via Devin's hook output if a prompt unexpectedly didn't trigger.
        print(f"mem-vault: skip ({skip_reason})", file=sys.stderr)
        return

    context = ""
    try:
        context = asyncio.run(_gather_context(prompt))
    except Exception as exc:
        print(f"mem-vault: UserPromptSubmit hook failed: {exc}", file=sys.stderr)

    if not context:
        return

    out = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }
    json.dump(out, sys.stdout, ensure_ascii=False)

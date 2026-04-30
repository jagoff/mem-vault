"""Pure helpers used by the discovery + derivation MCP tools.

This module is the single source of truth for:

- ``derive_project_tag(cwd, content)`` — picks a project-level tag from
  the cwd plus optional content hints (overrides for agent-config paths
  like ``~/.config/devin/skills``).
- ``derive_type_from_content(content)`` — keyword-priority classifier
  that maps a memory body to one of the canonical ``Memory.type`` values.
- ``derive_domain_tags(content)`` / ``derive_technique_tag(content)`` —
  regex tables that extract orthogonal taxonomy tags from the body.
- ``compute_stats(memories)`` — counts by type / agent + top tags +
  age histogram. No side effects.
- ``find_duplicate_pairs_by_tag_overlap(memories, threshold)`` — cheap
  duplicate detection without hitting Qdrant. Used as the offline
  fallback for ``memory_duplicates``.
- ``lint_memory(mem)`` — list of human-readable issues (``[]`` when
  the memory is well-formed).

Everything here is import-light and side-effect-free so it can be
unit-tested without touching Ollama / Qdrant / the filesystem.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mem_vault.storage import Memory

# ---------------------------------------------------------------------------
# Project tag derivation (cwd + content overrides)
# ---------------------------------------------------------------------------

# Filesystem-layout boilerplate that is never useful as a project tag. The
# leaf component beats every parent, so this list is only consulted when
# scanning *upward* past the leaf — which we don't currently do, but it
# keeps the door open if we later allow multi-tier tags.
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

# Content-based overrides — when the memory body explicitly references one
# of these paths, the project tag is the override regardless of where the
# user ran the agent from. Order matters: more-specific patterns first.
_PROJECT_OVERRIDES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"~/\.config/devin/skills|\.devin/skills"), "devin-config"),
    (re.compile(r"~/\.claude/skills|~/\.claude/agents|~/\.claude/CLAUDE\.md"), "claude-config"),
    (re.compile(r"~/\.config/devin/|\.devin/config\.json"), "devin-config"),
)

# Aliases for known repo names that the user prefers as canonical tags.
_REPO_ALIASES: dict[str, str] = {
    "rag-obsidian": "obsidian-rag",
}


def derive_project_tag(cwd: str | None, content: str = "") -> str | None:
    """Pick the canonical project tag for a given cwd, with content overrides.

    Priority:
    1. ``content`` matches one of ``_PROJECT_OVERRIDES`` (so a memory about
       a global skill always tags as ``devin-config`` even if the agent
       ran the save from a project repo).
    2. The cwd leaf, lowercased and kebab-cased, optionally aliased through
       ``_REPO_ALIASES``.
    3. ``None`` when neither rule fires (caller decides whether to error
       or fall back).
    """
    if content:
        for pattern, tag in _PROJECT_OVERRIDES:
            if pattern.search(content):
                return tag

    if not cwd:
        return None

    leaf = Path(cwd).name
    if not leaf or leaf.lower() in _CWD_NOISE_COMPONENTS:
        return None
    bare = leaf.lower()
    return _REPO_ALIASES.get(bare, bare)


# ---------------------------------------------------------------------------
# Type classifier
# ---------------------------------------------------------------------------


# Regex table — first match wins. Each pattern is matched with re.IGNORECASE
# and a word-boundary on the alternatives where it makes sense.
_TYPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "bug",
        re.compile(
            r"\b(bug|fix|broken|crash|leak|regression|rompi[óo]|fall[óo]|gotcha|"
            r"foot.?gun|root cause|causa raíz)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "decision",
        re.compile(
            r"\b(decid[íi]m?os?|elegim?os?|vamos con|opted for|chose|decision|"
            r"trade.?off|arquitectur)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "todo",
        re.compile(r"\b(TODO|pending|pendiente|por hacer|hay que|need to)\b", re.IGNORECASE),
    ),
    (
        "preference",
        re.compile(
            r"\b(prefiero|me gusta|always use|never use|preferencia)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "feedback",
        re.compile(
            r"\b(el user|the user|user)\s+(dijo|said|prefer|wants?|told|asked|"
            r"report[óo]|pidi[óo]|quiere|quiso)\b",
            re.IGNORECASE,
        ),
    ),
)


def derive_type_from_content(content: str) -> str:
    """Run the keyword-priority classifier; return one of ``Memory.type``.

    Special case for ``todo``: only triggers when content is short
    (<500 chars) — long memorias that mention "TODO" in passing are
    almost always notes/decisions/bugs, not actual todos.

    Falls back to ``"fact"`` for short definition-style content
    (<300 chars without paragraph breaks) and ``"note"`` for everything
    else (the safe default).
    """
    if not content:
        return "note"
    for label, pattern in _TYPE_PATTERNS:
        if not pattern.search(content):
            continue
        if label == "todo" and len(content) >= 500:
            continue
        return label
    if len(content) < 300 and "\n\n" not in content:
        return "fact"
    return "note"


# ---------------------------------------------------------------------------
# Domain tags
# ---------------------------------------------------------------------------

# Each entry: (label, pattern). Multiple patterns can fire — we cap the
# total at 3 in ``derive_domain_tags`` (preferring the order below).
_DOMAIN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("macos", re.compile(r"\b(macos|darwin|/var/folders|Path\.home|\$HOME)\b", re.IGNORECASE)),
    ("launchd", re.compile(r"\b(launchd|launchctl|plist|LaunchAgents|LaunchDaemons)\b")),
    ("qdrant", re.compile(r"\b(qdrant|bge-m3)\b", re.IGNORECASE)),
    ("fastembed", re.compile(r"\b(fastembed)\b", re.IGNORECASE)),
    ("fastapi", re.compile(r"\b(fastapi|uvicorn|Starlette)\b")),
    ("sqlite", re.compile(r"\b(sqlite|sqlite-vec)\b", re.IGNORECASE)),
    ("tests", re.compile(r"(\bpytest\b|@pytest|conftest)", re.IGNORECASE)),
    ("memory-system", re.compile(r"\b(mem0|mem-vault)\b", re.IGNORECASE)),
    ("llm", re.compile(r"\b(LLM|ollama|bge-m3|bge-reranker)\b")),
    ("rag", re.compile(r"\b(RAG|retrieve|rerank|paraphrase|hybrid search)\b", re.IGNORECASE)),
    ("env-vars", re.compile(r"\b(env.?var|environment|HF_HUB|FASTEMBED_)\b", re.IGNORECASE)),
    ("git", re.compile(r"\b(commit|git push|git rebase|git pull)\b", re.IGNORECASE)),
    ("streaming", re.compile(r"\b(SSE|streaming|/api/chat)\b")),
    ("frontend", re.compile(r"\b(CSS|HTML|Chart\.js|dashboard)\b")),
    ("whatsapp", re.compile(r"\b(WhatsApp|WA bridge)\b", re.IGNORECASE)),
    ("fine-tune", re.compile(r"\b(LoRA|fine.?tune|reranker)\b", re.IGNORECASE)),
    ("devin-skills", re.compile(r"(skill|SKILL\.md|\.devin/skills|\.claude/skills)")),
    ("mcp", re.compile(r"\b(MCP\b|MCP server|mcp_call_tool|mcp__)")),
    ("agent-tooling", re.compile(r"\b(Devin|Claude Code|agent profile|run_subagent)\b")),
    ("obsidian", re.compile(r"\b(frontmatter|wikilink|obsidian://|Obsidian vault)\b")),
)


def derive_domain_tags(content: str, *, cap: int = 3) -> list[str]:
    """Extract up to ``cap`` domain tags from the content.

    Tags fire in declaration order; the cap is applied after collection
    so the most-specific tags win when many patterns match (the table is
    arranged with concrete platforms first, abstract concepts last).
    """
    out: list[str] = []
    for tag, pattern in _DOMAIN_PATTERNS:
        if pattern.search(content) and tag not in out:
            out.append(tag)
            if len(out) >= cap:
                break
    return out


# ---------------------------------------------------------------------------
# Technique tag
# ---------------------------------------------------------------------------

_TECHNIQUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("performance", re.compile(r"\b(p50|p95|p99|ms|latency|perf|benchmark)\b", re.IGNORECASE)),
    ("gotcha", re.compile(r"\b(gotcha|foot.?gun)\b", re.IGNORECASE)),
    ("refactor", re.compile(r"\b(refactor)\b", re.IGNORECASE)),
    (
        "bugfix",
        re.compile(r"\b(root cause|causa raíz|fix|patch|rompi[óo]|fall[óo])\b", re.IGNORECASE),
    ),
    ("setup", re.compile(r"\b(setup|configurar|configuraci[óo]n|env var)\b", re.IGNORECASE)),
    ("eval", re.compile(r"\b(eval gate|baseline|hit@5)\b", re.IGNORECASE)),
    ("architecture", re.compile(r"\b(architectur|arquitectur|tradeoff)\b", re.IGNORECASE)),
)


def derive_technique_tag(content: str) -> str | None:
    """Single-tag technique classifier — returns ``None`` if nothing fires."""
    for tag, pattern in _TECHNIQUE_PATTERNS:
        if pattern.search(content):
            return tag
    return None


# ---------------------------------------------------------------------------
# Title derivation
# ---------------------------------------------------------------------------


def derive_title_from_content(content: str, *, max_len: int = 80) -> str:
    """Pull a title out of the body. Prefer a ``# Heading`` if present."""
    stripped = content.strip()
    if not stripped:
        return "memory"
    first_line = stripped.splitlines()[0].strip()
    # ``# Heading`` form
    if first_line.startswith("#"):
        return first_line.lstrip("#").strip()[:max_len] or "memory"
    if len(first_line) <= 100:
        return first_line[:max_len] or "memory"
    return first_line[:max_len]


# ---------------------------------------------------------------------------
# Stats over a corpus
# ---------------------------------------------------------------------------


def _age_bucket(updated: str) -> str:
    """Bucket an ISO timestamp into 'today', 'week', 'month', 'older'."""
    if not updated:
        return "older"
    try:
        dt = datetime.fromisoformat(updated)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
    except Exception:
        return "older"
    age_days = (datetime.now(tz=dt.tzinfo) - dt).total_seconds() / 86400.0
    if age_days < 1:
        return "today"
    if age_days < 7:
        return "week"
    if age_days < 30:
        return "month"
    return "older"


def compute_stats(memories: Iterable[Memory], *, top_tags_n: int = 5) -> dict[str, Any]:
    """Produce a structured summary of a memory corpus.

    Output shape::

        {
          "total": 42,
          "by_type": {"fact": 14, "decision": 9, ...},
          "by_agent": {"devin": 22, "claude-code": 8, ...},
          "top_tags": [("rag", 18), ("obsidian-rag", 14), ...],  # tuples
          "age": {"today": 1, "week": 5, "month": 15, "older": 21},
        }
    """
    mems = list(memories)
    by_type: Counter[str] = Counter(m.type for m in mems)
    by_agent: Counter[str] = Counter((m.agent_id or "—") for m in mems)
    tag_counts: Counter[str] = Counter(t for m in mems for t in (m.tags or []))
    age_counts: Counter[str] = Counter(_age_bucket(m.updated) for m in mems)
    return {
        "total": len(mems),
        "by_type": dict(by_type),
        "by_agent": dict(by_agent),
        "top_tags": tag_counts.most_common(top_tags_n),
        "age": dict(age_counts),
    }


# ---------------------------------------------------------------------------
# Duplicate detection (offline, tag-overlap based)
# ---------------------------------------------------------------------------


def find_duplicate_pairs_by_tag_overlap(
    memories: Iterable[Memory], *, threshold: float = 0.7
) -> list[tuple[str, str, float]]:
    """Cheap pairwise overlap by tag Jaccard similarity.

    Trades off precision for the ability to run without Qdrant. For real
    semantic dedup, ``mem_vault.consolidate.find_candidate_pairs`` is the
    right tool — it uses the embedding index and is much more accurate.
    This function is the fallback that ``memory_duplicates`` falls back
    to when Qdrant is locked out (or just for a quick sanity check).

    Returns triples ``(id_a, id_b, jaccard)`` sorted by score desc.
    """
    mems = list(memories)
    out: list[tuple[str, str, float]] = []
    for i, a in enumerate(mems):
        a_tags = set(a.tags or [])
        if not a_tags:
            continue
        for b in mems[i + 1 :]:
            b_tags = set(b.tags or [])
            if not b_tags:
                continue
            overlap = a_tags & b_tags
            union = a_tags | b_tags
            j = len(overlap) / len(union) if union else 0.0
            if j >= threshold:
                out.append((a.id, b.id, j))
    out.sort(key=lambda t: t[2], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Lint a single memory
# ---------------------------------------------------------------------------

# Heuristic — well-formed long memorias usually carry an "Aprendido el ..."
# line in the body (the convention from CLAUDE.md). We don't enforce it,
# but flag it for short bodies it's expected.
_APRENDIDO_RE = re.compile(r"^##?\s*Aprendido el\s+\d{4}-\d{2}-\d{2}", re.MULTILINE | re.IGNORECASE)


def lint_memory(mem: Memory) -> list[str]:
    """Return a list of issue strings, ``[]`` when the memory is clean."""
    issues: list[str] = []
    tags = mem.tags or []
    body = (mem.body or "").strip()

    if len(tags) < 3:
        issues.append(f"<3 tags ({len(tags)})")
    if not body:
        issues.append("body vacío")
    elif len(body) < 100:
        issues.append(f"body corto ({len(body)} chars)")
    if body and len(body) >= 300 and not _APRENDIDO_RE.search(body):
        issues.append("falta `## Aprendido el YYYY-MM-DD`")
    if not mem.created:
        issues.append("sin `created`")
    if not mem.description:
        issues.append("sin `description`")
    return issues

"""Secret redaction for memory bodies.

Memorias often capture commands, configs, or tracebacks that carry API
keys, bearer tokens, passwords, or other credentials. Once a body is
written to the Obsidian vault, it can sync to iCloud / Syncthing /
Dropbox / git — each one a potential leak path. This module scrubs the
well-known shapes of credentials before they hit disk.

Defensive by design:

- Redaction is a filter, not a classifier. It replaces the matched value
  with a placeholder (``[REDACTED:<kind>]``) and counts how many hits
  were found. It does not try to recover / guess / validate.
- Patterns err on the side of false positives. A redacted fake cert is
  harmless; a leaked real one isn't. Each pattern carries a ``kind``
  label so the caller can render a warning that's specific enough for
  the user to inspect.
- Pure regex, no ML model, no dependency.

Patterns covered (non-exhaustive — you can extend via ``EXTRA_PATTERNS``):

- AWS access/secret keys (``AKIA...``, ``ASIA...``; 40-char secrets).
- GitHub tokens (``ghp_`` / ``gho_`` / ``ghu_`` / ``ghr_`` / ``ghs_``
  prefixes; classic 40-hex + fine-grained 76-base62).
- OpenAI keys (``sk-...``) and Anthropic keys (``sk-ant-...``).
- Slack tokens (``xox[baprs]-...``).
- Google API keys (``AIza...``).
- Generic private-key blocks (``-----BEGIN ... PRIVATE KEY-----`` ... ``END``).
- Password-like ``password = "..."`` / ``passwd: xxx`` / ``secret = …``
  / ``TOKEN="..."`` assignments.
- JWTs (three base64url segments separated by ``.``).
- Bearer tokens in ``Authorization: Bearer <token>``.

The caller receives the redacted text plus a structured summary
``[{kind, count}]`` — the MCP server surfaces it in the
``memory_save`` envelope (``redactions``) so downstream tooling can
warn the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A single pattern entry. ``pattern`` must capture the sensitive value in
# group 1 (or the whole match when no capturing group is present — we
# substitute the whole match in that case).
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # AWS Access Key Id (20 chars, starts with AKIA/ASIA/AGPA/AIDA/AROA/ANPA/ANVA/AIPA)
    (
        "aws_access_key",
        re.compile(r"\b(A(?:KIA|SIA|GPA|IDA|ROA|NPA|NVA|IPA)[A-Z0-9]{16})\b"),
    ),
    # AWS Secret Access Key (40 base64url-ish characters, usually after
    # "aws_secret_access_key" or within a csv of credentials). We look for
    # the explicit assignment shape, not raw 40-char tokens (too many
    # false positives on git hashes).
    (
        "aws_secret_key",
        re.compile(
            r"(?i)\baws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*['\"]?"
            r"([A-Za-z0-9/+=]{40})\b"
        ),
    ),
    # GitHub personal access tokens (classic + fine-grained)
    (
        "github_token",
        re.compile(r"\b(ghp_[A-Za-z0-9]{36,255})\b"),
    ),
    (
        "github_token",
        re.compile(r"\b(gh[ousr]_[A-Za-z0-9]{36,255})\b"),
    ),
    # Anthropic keys — sk-ant-... (listed FIRST because its prefix is a
    # superset of OpenAI's ``sk-``; running them in reverse order would
    # let the generic ``sk-...`` rule swallow every ``sk-ant-...`` key).
    (
        "anthropic_key",
        re.compile(r"\b(sk-ant-[A-Za-z0-9\-_]{20,})\b"),
    ),
    # OpenAI keys — sk-... (legacy and project-scoped)
    (
        "openai_key",
        re.compile(r"\b(sk-[A-Za-z0-9\-_]{20,})\b"),
    ),
    # Slack tokens — xoxb/xoxp/xoxa/xoxr/xoxs-...
    (
        "slack_token",
        re.compile(r"\b(xox[baprs]-[A-Za-z0-9\-]{10,})\b"),
    ),
    # Google API keys
    (
        "google_api_key",
        re.compile(r"\b(AIza[0-9A-Za-z\-_]{35})\b"),
    ),
    # JWT (three base64url-safe segments separated by dots, each at least
    # ~20 chars). Matches are greedy to capture the whole token.
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
    ),
    # Authorization: Bearer <token>
    (
        "bearer_token",
        re.compile(r"(?i)\bAuthorization\s*:\s*Bearer\s+([A-Za-z0-9\-._~+/=]{20,})"),
    ),
    # PEM private-key blocks (single or multi-line). DOTALL so newlines
    # inside the block are captured.
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    # Generic credential assignments — catches shell / TOML / YAML shapes:
    #   PASSWORD="hunter2"
    #   api_token: s3cret
    #   secret = 'xxx'
    # Keys are deliberately narrow (password/passwd/secret/token/api_key)
    # to avoid matching ``name = "foo"``. Value must be >=6 chars to skip
    # boilerplate like ``token=""``.
    (
        "credential_assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|secret|api[_-]?(?:key|token|secret)|"
            r"access[_-]?token|auth[_-]?token|client[_-]?secret|token)"
            r"\s*[:=]\s*['\"]?(?!\[REDACTED)([^\s'\"\n]{6,})['\"]?"
        ),
    ),
]

# Users can extend at runtime (e.g. internal org-specific prefixes).
# Each entry is ``(kind, compiled_regex)`` same as ``_PATTERNS``.
EXTRA_PATTERNS: list[tuple[str, re.Pattern[str]]] = []


@dataclass
class RedactionHit:
    """One redacted match — for the caller's summary."""

    kind: str
    count: int


def _replace(match: re.Match[str], kind: str) -> str:
    """Substitute the matched secret with a placeholder.

    Preserves everything outside the captured group so the surrounding
    context (e.g. ``Authorization: Bearer ``) stays readable.
    """
    if not match.groups():
        return f"[REDACTED:{kind}]"
    # The full match minus the captured token keeps the prefix readable.
    start = match.start(1) - match.start()
    end = match.end(1) - match.start()
    full = match.group(0)
    return full[:start] + f"[REDACTED:{kind}]" + full[end:]


def redact(text: str) -> tuple[str, list[RedactionHit]]:
    """Scan ``text`` for credential-shaped substrings and replace them.

    Returns ``(redacted_text, hits)`` where ``hits`` is a list of
    ``{kind, count}`` entries for each pattern that fired at least
    once. An empty ``hits`` list means the body was clean.

    Idempotent: running ``redact`` on already-redacted text is a no-op
    (the ``[REDACTED:*]`` placeholder doesn't match any pattern).

    The pattern order matters when two patterns overlap (e.g. a
    ``sk-ant-...`` could also match the generic ``sk-...`` rule). We
    iterate the list in order and run each ``re.sub`` on the
    accumulating output, so the first match wins — Anthropic-specific
    patterns are listed before the generic OpenAI one.
    """
    if not text:
        return text, []
    hits: dict[str, int] = {}
    current = text
    for kind, pattern in _PATTERNS + EXTRA_PATTERNS:
        count_ref: list[int] = [0]

        # Bind ``kind`` AND ``count_ref`` via default args to avoid the
        # late-binding closure trap (ruff B023). Without this, every
        # ``_sub`` would see the last iteration's values instead of the
        # current loop's. ``_cr`` is a list so we can mutate the count
        # in-place through the default binding.
        def _sub(
            m: re.Match[str],
            _k: str = kind,
            _cr: list[int] = count_ref,
        ) -> str:
            _cr[0] += 1
            return _replace(m, _k)

        current = pattern.sub(_sub, current)
        if count_ref[0]:
            hits[kind] = hits.get(kind, 0) + count_ref[0]
    summary = [RedactionHit(kind=k, count=c) for k, c in hits.items()]
    return current, summary


def contains_secrets(text: str) -> bool:
    """Fast check — returns True iff any pattern fires. Doesn't produce a summary."""
    if not text:
        return False
    for _kind, pattern in _PATTERNS + EXTRA_PATTERNS:
        if pattern.search(text):
            return True
    return False

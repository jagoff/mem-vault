"""Detect and merge near-duplicate memories.

Strategy:

1. Embed every memory body (we already have these in Qdrant — no work).
2. For each memory, find the K nearest neighbors and emit a candidate pair
   when cosine similarity is above ``threshold`` (default 0.92).
3. For each pair, ask the local Ollama LLM to choose one of:
   - ``MERGE``: produce a single fused body that preserves all unique
     information from both. The result replaces the older memory; the newer
     memory is deleted.
   - ``KEEP_BOTH``: the memories look similar but talk about different
     things. No-op.
   - ``KEEP_FIRST`` / ``KEEP_SECOND``: one of them is fully subsumed by the
     other. Delete the redundant one.
4. Apply the resolution to the vault + reindex Qdrant.

The whole thing runs locally on Ollama. ``--dry-run`` prints the plan
without touching any file. ``--apply`` actually writes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from mem_vault.config import Config
from mem_vault.index import VectorIndex
from mem_vault.storage import Memory, VaultStorage

logger = logging.getLogger(__name__)


_PROMPT_TEMPLATE = """\
You are deduplicating memory entries for an AI agent. Decide what to do with these two memories.

Output STRICT JSON only, no prose, with this exact shape:
{{"action": "MERGE" | "KEEP_BOTH" | "KEEP_FIRST" | "KEEP_SECOND",
  "rationale": "<one short sentence>",
  "merged_body": "<only when action==MERGE; the fused body, markdown OK>",
  "merged_title": "<only when action==MERGE; one short title>"}}

Definitions:
- MERGE: same topic, partially overlapping content. Produce a fused body that
  preserves every unique fact. Use the merged_title to summarize.
- KEEP_BOTH: similar wording but actually different topics. Don't touch them.
- KEEP_FIRST: memory A is a strict superset of B. Delete B.
- KEEP_SECOND: memory B is a strict superset of A. Delete A.

Memory A (id={a_id}, type={a_type}, created={a_created}):
\"\"\"
{a_body}
\"\"\"

Memory B (id={b_id}, type={b_type}, created={b_created}):
\"\"\"
{b_body}
\"\"\"
"""


@dataclass
class Pair:
    a: Memory
    b: Memory
    score: float


@dataclass
class Resolution:
    action: str  # MERGE | KEEP_BOTH | KEEP_FIRST | KEEP_SECOND
    rationale: str
    merged_body: str | None = None
    merged_title: str | None = None


def find_candidate_pairs(
    storage: VaultStorage,
    index: VectorIndex,
    *,
    threshold: float = 0.85,
    user_id: str = "default",
    per_memory_neighbors: int = 4,
) -> list[Pair]:
    """Walk the vault, hit Qdrant for each memory's nearest neighbors, return
    the unique (a, b) pairs that exceed ``threshold``.

    Pairs are deduplicated by ``frozenset({a.id, b.id})`` so each pair is
    surfaced exactly once. Self-hits are skipped.
    """
    seen_pairs: set[frozenset[str]] = set()
    pairs: list[Pair] = []

    # Materialize once because we need the by-id lookup AND the iteration
    # below. Streaming twice (once to build the dict, once to walk) would
    # double the disk I/O. ``list()`` without a limit cap mirrors the
    # historical contract — callers who want bounded memory should pass
    # ``--limit`` to the CLI.
    memories = list(storage.iter_memories())
    by_id = {m.id: m for m in memories}

    for mem in memories:
        if not mem.body:
            continue
        try:
            hits = index.search(
                mem.body,
                user_id=user_id,
                top_k=per_memory_neighbors + 1,  # +1 because the self-hit usually wins
                threshold=threshold,
            )
        except Exception as exc:
            logger.warning("similarity probe for %s failed: %s", mem.id, exc)
            continue

        for hit in hits:
            md = (hit.get("metadata") or {}) if isinstance(hit, dict) else {}
            other_id = md.get("memory_id")
            if not other_id or other_id == mem.id or other_id not in by_id:
                continue
            score = float(hit.get("score") or 0.0)
            if score < threshold:
                continue
            key = frozenset({mem.id, other_id})
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            pairs.append(Pair(a=mem, b=by_id[other_id], score=score))

    pairs.sort(key=lambda p: p.score, reverse=True)
    return pairs


def _ask_llm(config: Config, pair: Pair, *, ollama_client) -> Resolution:
    """Ask Ollama to classify the pair and (optionally) produce the merged body."""
    prompt = _PROMPT_TEMPLATE.format(
        a_id=pair.a.id,
        a_type=pair.a.type,
        a_created=pair.a.created,
        a_body=pair.a.body[:4000],
        b_id=pair.b.id,
        b_type=pair.b.type,
        b_created=pair.b.created,
        b_body=pair.b.body[:4000],
    )

    response = ollama_client.chat(
        model=config.llm_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
        format="json",
    )
    raw = (response.get("message") or {}).get("content") or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Resolution(action="KEEP_BOTH", rationale=f"non-JSON LLM output: {raw[:80]!r}")

    action = str(data.get("action") or "KEEP_BOTH").upper()
    if action not in {"MERGE", "KEEP_BOTH", "KEEP_FIRST", "KEEP_SECOND"}:
        action = "KEEP_BOTH"

    return Resolution(
        action=action,
        rationale=str(data.get("rationale") or ""),
        merged_body=data.get("merged_body"),
        merged_title=data.get("merged_title"),
    )


def apply_resolution(
    storage: VaultStorage,
    index: VectorIndex,
    pair: Pair,
    res: Resolution,
    *,
    user_id: str = "default",
) -> dict[str, Any]:
    """Apply a Resolution to the vault + index. Returns a summary dict.

    Crash-safety contract: every action below is ordered so that an
    exception mid-flight leaves the system in a *recoverable* state. A
    follow-up ``mem-vault reindex`` will reconcile any drift between the
    vault and the embedding index. We never delete a ``.md`` before its
    replacement has been fully embedded — that was the BUG #6 in the
    audit (a mid-merge crash made both memories invisible to search).
    """
    if res.action == "KEEP_BOTH":
        return {"action": "KEEP_BOTH", "kept": [pair.a.id, pair.b.id]}

    if res.action == "KEEP_FIRST":
        # Delete the .md *first*: the file is the source of truth, and a
        # crash between the two calls is recoverable via reindex's orphan
        # sweep. The reverse order would leave the file with no embedding.
        storage.delete(pair.b.id)
        index.delete_by_metadata("memory_id", pair.b.id, user_id)
        return {"action": "KEEP_FIRST", "kept": [pair.a.id], "deleted": [pair.b.id]}

    if res.action == "KEEP_SECOND":
        storage.delete(pair.a.id)
        index.delete_by_metadata("memory_id", pair.a.id, user_id)
        return {"action": "KEEP_SECOND", "kept": [pair.b.id], "deleted": [pair.a.id]}

    # MERGE: rewrite the older memory with the fused body, delete the newer.
    older, newer = sorted([pair.a, pair.b], key=lambda m: m.created or "")
    merged_tags = sorted({*older.tags, *newer.tags, "merged"})
    new_body = res.merged_body or f"{older.body}\n\n---\n\n{newer.body}"
    new_title = res.merged_title or older.name

    # Snapshot what older looked like before the merge so we can roll back
    # if the embed fails. We don't snapshot the index entries themselves —
    # if the rollback also fails, the user falls back to ``mem-vault reindex``.
    previous_older_body = (older.body or "").strip()
    previous_older_name = older.name
    previous_older_tags = list(older.tags)

    storage.update(
        older.id,
        content=new_body,
        title=new_title,
        tags=merged_tags,
    )

    # Re-embed the merged body BEFORE touching ``newer``. If add() raises,
    # we restore the older memory's prior body + try to re-embed it, and
    # leave ``newer`` untouched on disk — the merge is a no-op from the
    # outside, and the user can retry on a healthy Ollama.
    try:
        index.delete_by_metadata("memory_id", older.id, user_id)
        index.add(
            new_body,
            user_id=user_id,
            agent_id=older.agent_id,
            metadata={
                "memory_id": older.id,
                "type": older.type,
                "tags": merged_tags,
            },
            auto_extract=False,
        )
    except Exception as exc:
        logger.warning(
            "MERGE failed mid-embed for %s + %s (%s); rolling back older",
            older.id,
            newer.id,
            exc,
        )
        # Restore older's body on disk so the .md isn't left in the merged
        # state with no matching index entry.
        try:
            storage.update(
                older.id,
                content=previous_older_body,
                title=previous_older_name,
                tags=previous_older_tags,
            )
        except Exception as roll_exc:
            logger.warning("rollback storage.update for %s failed: %s", older.id, roll_exc)
        # Best-effort: re-embed the previous body so search still finds it.
        # If THIS also fails, ``mem-vault reindex`` is the recovery path.
        try:
            index.add(
                previous_older_body,
                user_id=user_id,
                agent_id=older.agent_id,
                metadata={
                    "memory_id": older.id,
                    "type": older.type,
                    "tags": previous_older_tags,
                },
                auto_extract=False,
            )
        except Exception as roll_exc:
            logger.warning("rollback index.add for %s failed: %s", older.id, roll_exc)
        raise

    # Now and only now is it safe to delete ``newer``: the merged body
    # already lives on disk and in the index under ``older.id``.
    storage.delete(newer.id)
    try:
        index.delete_by_metadata("memory_id", newer.id, user_id)
    except Exception as exc:
        # Lower stakes: the .md is gone but a residual index entry would
        # be harmless (search() filters it as an orphan now). Logged for
        # visibility; ``mem-vault reindex`` will sweep it on next run.
        logger.warning("post-merge index cleanup of %s failed: %s", newer.id, exc)

    return {"action": "MERGE", "kept": [older.id], "deleted": [newer.id]}

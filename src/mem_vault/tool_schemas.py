"""Declarative MCP tool schemas for mem-vault.

Why this file exists
--------------------
The schemas — what tools the MCP server advertises, what arguments each
accepts, what's required, and the human-readable descriptions — are
*data*, not logic. Keeping them out of ``server.py`` makes the wiring
file (``_build_handlers``, ``call_tool`` dispatch) stay focused on
mechanism, and the tool list itself stay focused on contract.

The schemas are imported by:

- ``mem_vault.server`` — for ``Server.list_tools()`` and the symmetry
  check in ``_build_handlers``
- ``tests/test_contracts.py`` — to verify every tool has a matching
  service method on both ``MemVaultService`` and ``RemoteMemVaultService``
- ``tests/test_server_dispatch.py`` — same shape, different angle

The variable is exposed as both ``TOOL_SCHEMAS`` (canonical, public-ish)
and ``_TOOLS`` (legacy alias, what the rest of the codebase already
imports). Either works; new code should prefer ``TOOL_SCHEMAS``.
"""

from __future__ import annotations

import mcp.types as types

_TOOLS: list[types.Tool] = [
    types.Tool(
        name="memory_save",
        description=(
            "Persist a memory in the user's Obsidian vault as a markdown file with "
            "YAML frontmatter, then index it locally for semantic search. Use this "
            "to remember decisions, preferences, bug fixes, conventions, or any "
            "piece of context that should survive the current session.\n\n"
            "Set auto_extract=true to let an Ollama LLM extract canonical facts "
            "and dedupe against existing memories (slower, smarter). Default "
            "(false) saves the literal content (faster, deterministic)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The full memory body in markdown.",
                },
                "title": {
                    "type": "string",
                    "description": "Short human-readable title (defaults to first line of content).",
                },
                "description": {
                    "type": "string",
                    "description": "One-line synopsis (defaults to first ~200 chars of content).",
                },
                "type": {
                    "type": "string",
                    "enum": ["feedback", "preference", "decision", "fact", "note", "bug", "todo"],
                    "default": "note",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for filtering later.",
                },
                "auto_extract": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, run the LLM extractor + dedup. If false, save literally.",
                },
                "auto_link": {
                    "type": "boolean",
                    "description": (
                        "If true, after a successful save, run a semantic search "
                        "for similar memorias and stamp their IDs in this memory's "
                        "``related`` frontmatter. Defaults to ``Config.auto_link_default`` "
                        "(true unless globally disabled via MEM_VAULT_AUTO_LINK=0)."
                    ),
                },
                "auto_contradict": {
                    "type": "boolean",
                    "description": (
                        "If true, after a successful save, ask a local LLM whether "
                        "the new body contradicts any of the top-5 semantically "
                        "similar memorias; IDs of contradicting memorias are "
                        "stamped in this memory's ``contradicts`` frontmatter. "
                        "Adds ~3-5 s latency. Defaults to "
                        "``Config.auto_contradict_default`` (false unless "
                        "MEM_VAULT_AUTO_CONTRADICT=1)."
                    ),
                },
                "visible_to": {
                    "description": (
                        "Which agents can see this memory. Defaults to public. Pass "
                        "the string 'private' to scope it to the saving agent only, "
                        "'public' for everyone (default), or a list of agent ids "
                        "(['claude-code', 'cursor']) for explicit allowlist. The "
                        "saving agent is always implicitly included."
                    ),
                    "oneOf": [
                        {"type": "string", "enum": ["public", "private"]},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                },
                "project": {
                    "type": "string",
                    "description": (
                        "Project scope stamped in the index metadata. "
                        "``memory_search`` uses it to filter results to "
                        "memorias from the same project — faster than a "
                        "tag filter because Qdrant indexes the payload. "
                        "When omitted, inferred from the first "
                        "``project:X`` tag or ``Config.project_default``."
                    ),
                },
                "user_id": {"type": "string"},
                "agent_id": {"type": "string"},
            },
            "required": ["content"],
        },
    ),
    types.Tool(
        name="memory_search",
        description=(
            "Semantic search across all memories using local embeddings. Returns "
            "the top-k most relevant memories with their full content. Useful at "
            "the start of a session to recover relevant context, or before "
            "answering questions that depend on past decisions.\n\n"
            "v0.6.0: pass ``expand_hops`` to BFS the local knowledge graph "
            "(``related`` ∪ ``contradicts`` ∪ co-tag) from the top-k and "
            "include neighbors as ``via_graph=true`` hits. Contradictions of "
            "the top-3 are auto-injected as ``edges=['contradicts']`` even "
            "when they didn't make the semantic top-k organically — disable "
            "with ``inject_contradictions=false``."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query."},
                "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                "type": {
                    "type": "string",
                    "description": "Optional type filter.",
                    "enum": ["feedback", "preference", "decision", "fact", "note", "bug", "todo"],
                },
                "user_id": {"type": "string"},
                "threshold": {
                    "type": "number",
                    "default": 0.1,
                    "description": "Minimum similarity score (0-1). Lower = more lenient.",
                },
                "project": {
                    "type": "string",
                    "description": (
                        "Filter to memorias stamped with this project scope. "
                        "Pass ``*`` (or empty string) to bypass "
                        "``Config.project_default`` and search globally."
                    ),
                },
                "expand_hops": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "maximum": 3,
                    "description": (
                        "BFS depth on the local knowledge graph. 0 (default) = "
                        "pure semantic. 1-2 = include neighbors via "
                        "related/contradicts/co-tag for richer context."
                    ),
                },
                "graph_min_shared_tags": {
                    "type": "integer",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Co-tag edge threshold during expansion.",
                },
                "graph_max_nodes": {
                    "type": "integer",
                    "default": 30,
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Soft cap on nodes added via graph expansion.",
                },
                "inject_contradictions": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Auto-add up to 3 ``contradicts`` of the top-3 results "
                        "as ``via_graph`` hits with ``edges=['contradicts']``. "
                        "Off only when you want pure semantic ordering."
                    ),
                },
                "session_id": {
                    "type": "string",
                    "description": (
                        "Optional session id stamped on telemetry rows so the "
                        "Stop hook can correlate citations to this exact turn. "
                        "Defaults to the MEM_VAULT_SESSION_ID env var."
                    ),
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="memory_list",
        description=(
            "List memories sorted by most-recently-modified. Optional filters by "
            "type, tags, or user_id. Use this to browse without a specific query."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["feedback", "preference", "decision", "fact", "note", "bug", "todo"],
                },
                "tags": {"type": "array", "items": {"type": "string"}},
                "user_id": {"type": "string"},
                "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 200},
            },
        },
    ),
    types.Tool(
        name="memory_get",
        description="Read a single memory by id (the file slug, e.g. `feedback_local_free_stack`).",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Memory id (filename without .md)."}
            },
            "required": ["id"],
        },
    ),
    types.Tool(
        name="memory_update",
        description=(
            "Replace fields on an existing memory. Any field omitted is left "
            "unchanged. The `updated` timestamp is bumped automatically."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "content": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "visible_to": {
                    "description": (
                        "Change visibility scope. Pass 'private' to scope "
                        "to the owner agent, 'public' for everyone, or a "
                        "list of agent ids for an explicit allowlist. "
                        "Omit to leave unchanged."
                    ),
                    "oneOf": [
                        {"type": "string", "enum": ["public", "private"]},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                },
            },
            "required": ["id"],
        },
    ),
    types.Tool(
        name="memory_delete",
        description=(
            "Permanently delete a memory (its .md file + every embedding pointing "
            "to it). This is irreversible — confirm with the user before calling "
            "in agent flows."
        ),
        inputSchema={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    ),
    types.Tool(
        name="memory_briefing",
        description=(
            "Compose a project-aware boot briefing: total memorias of the "
            "current project (resolved from ``cwd``) + total global, last 3 "
            "by recency, top 5 co-tags, lint summary. Designed for the skill "
            "to render the 6-line summary on the first ``/mv`` of a session "
            "so the agent enters the conversation knowing what's in the vault."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "cwd": {
                    "type": "string",
                    "description": "Absolute path to the project root. Used to derive the project_tag.",
                },
            },
        },
    ),
    types.Tool(
        name="memory_derive_metadata",
        description=(
            "Run the keyword-priority classifiers on a memory body and "
            "return a suggested ``{title, type, tags, missing_tags}``. "
            "Intended to be called by the skill *before* ``memory_save`` "
            "so the user only types the body and the metadata is filled "
            "in automatically. ``missing_tags > 0`` means the body didn't "
            "match enough patterns to reach 3 tags — the skill should ask "
            "the user for one more before saving."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "cwd": {"type": "string"},
            },
            "required": ["content"],
        },
    ),
    types.Tool(
        name="memory_stats",
        description=(
            "Aggregate counts over the memory corpus: by ``type``, by "
            "``agent_id``, top tags, and age histogram (today / week / "
            "month / older). When ``cwd`` is provided, scopes to memorias "
            "tagged with the resolved ``project_tag``."
        ),
        inputSchema={
            "type": "object",
            "properties": {"cwd": {"type": "string"}},
        },
    ),
    types.Tool(
        name="memory_duplicates",
        description=(
            "Surface pairs of memorias with high tag-overlap Jaccard — cheap "
            "candidate duplicate detection without hitting Qdrant. Use this "
            "when the user asks 'tengo dos memorias parecidas?' for a quick "
            "answer; for deep semantic dedup use ``mem-vault consolidate``."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "default": 0.7,
                    "description": "Jaccard threshold; pairs with score < this are skipped.",
                },
                "cwd": {"type": "string"},
            },
        },
    ),
    types.Tool(
        name="memory_lint",
        description=(
            "List memorias with structural issues: <3 tags, missing "
            "``description``, body shorter than 100 chars, body without "
            "``## Aprendido el YYYY-MM-DD`` line. Useful before "
            "``mem-vault consolidate`` to spot underdeveloped entries."
        ),
        inputSchema={
            "type": "object",
            "properties": {"cwd": {"type": "string"}},
        },
    ),
    types.Tool(
        name="memory_related",
        description=(
            "Walk the local knowledge graph around one memory and return its "
            "neighbors, grouped by relationship type:\n\n"
            "- ``related``: the explicit ``related:`` frontmatter list "
            "(wikilinks stamped by auto-link).\n"
            "- ``contradicts``: the ``contradicts:`` frontmatter list "
            "(populated by auto-contradict).\n"
            "- ``cotag_neighbors``: memorias that share ≥ ``min_shared_tags`` "
            "normalized tags (``project:foo`` splits on colon).\n"
            "- ``semantic_neighbors``: top-``k`` results from a semantic "
            "search over the memory's body. Skipped when ``include_semantic`` "
            "is false (no LLM / Qdrant call).\n\n"
            "Use this to explore what the vault knows around a single node "
            "without reconstructing the graph yourself."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Memory id (filename slug without .md)."},
                "min_shared_tags": {
                    "type": "integer",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Threshold for co-tag neighbor edges.",
                },
                "k": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 50,
                    "description": "How many semantic neighbors to include.",
                },
                "include_semantic": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "If false, skip the semantic search step (cheaper, "
                        "deterministic — relies on tag structure only)."
                    ),
                },
            },
            "required": ["id"],
        },
    ),
    types.Tool(
        name="memory_neighborhood",
        description=(
            "BFS the local knowledge graph from a set of seed memorias and "
            "return every visited node with its hop distance + the edge "
            "kinds that pulled it in. Stronger than ``memory_related`` (one "
            "seed, one hop) — this can take multiple seeds, traverse 2-3 "
            "hops, and lets you narrow the edge kinds. Useful for 'show me "
            "the cluster of decisions around topic X' kinds of questions, "
            "and for the agent to discover context implied by relationships "
            "rather than semantic similarity alone.\n\n"
            "Edge kinds: ``related`` (auto-link wikilinks), ``contradicts`` "
            "(auto-detected tensions), ``cotag`` (≥ ``min_shared_tags`` shared "
            "after ``project:foo``-splitting). Pass ``edge_kinds=['contradicts']`` "
            "to scan only tensions, e.g. before reaffirming a decision."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Seed memory ids. At least one required.",
                },
                "hops": {
                    "type": "integer",
                    "default": 1,
                    "minimum": 0,
                    "maximum": 3,
                    "description": "BFS depth. 0 returns just the seeds.",
                },
                "edge_kinds": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["related", "contradicts", "cotag"],
                    },
                    "description": (
                        "Restrict traversal to these edge kinds. Defaults to "
                        "all three when omitted."
                    ),
                },
                "min_shared_tags": {
                    "type": "integer",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Co-tag edge threshold.",
                },
                "max_nodes": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Soft cap on visited nodes.",
                },
            },
            "required": ["ids"],
        },
    ),
    types.Tool(
        name="memory_history",
        description=(
            "Return the edit history of a memory — every snapshot taken "
            "before an ``update`` (body, title, description, tags, related, "
            "contradicts). Entries are ordered newest-first. Useful when you "
            "want to see what a memory said last week, or to recover a field "
            "that got overwritten. Memorias with no updates return an empty "
            "list."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Memory id (filename slug without .md)."},
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Cap the number of entries returned.",
                },
            },
            "required": ["id"],
        },
    ),
    types.Tool(
        name="memory_feedback",
        description=(
            "Record thumbs up/down on a memory. Bumps ``helpful_count`` or "
            "``unhelpful_count`` in the memory's frontmatter and updates "
            "``last_used``. ``memory_search`` uses these counters to lift "
            "memorias that were positively judged above neutral peers. "
            "Call this right after the agent actually relied on a memory "
            "in its response (manual), or let the Stop hook infer it from "
            "citations (automatic). Set ``helpful`` to null to record a "
            "plain 'I used this' event without a polarity vote."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Memory id (filename slug without .md).",
                },
                "helpful": {
                    "type": ["boolean", "null"],
                    "description": (
                        "true = thumbs up (bumps helpful_count); false = "
                        "thumbs down (bumps unhelpful_count); null / omitted "
                        "= just mark 'used' without polarity."
                    ),
                },
            },
            "required": ["id"],
        },
    ),
    types.Tool(
        name="memory_synthesize",
        description=(
            "Compose an LLM-written summary of what the system knows about "
            "``query``. Internally runs a wide semantic search (default k=10) "
            "and asks the local LLM (Ollama) to weave the matched memorias "
            "into a coherent answer in español rioplatense, citing the source "
            "IDs inline. Use this when the user asks an open-ended question "
            '("resumime todo lo que sé sobre X") — it\'s the difference '
            "between dumping a list of bullets and getting an interlocutor "
            "that actually responds."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic or question to synthesize an answer for.",
                },
                "k": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 30,
                    "description": "How many memorias to feed into the LLM.",
                },
                "threshold": {
                    "type": "number",
                    "default": 0.1,
                    "description": "Minimum similarity for a memory to be included.",
                },
            },
            "required": ["query"],
        },
    ),
]


# Backwards-compat alias. The rest of the codebase still imports
# `_TOOLS`; new readers should reach for `TOOL_SCHEMAS` directly.
TOOL_SCHEMAS: list[types.Tool] = _TOOLS

__all__ = ("TOOL_SCHEMAS", "_TOOLS")

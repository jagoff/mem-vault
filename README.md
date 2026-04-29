# mem-vault

> Local MCP server that gives AI agents **infinite memory backed by an Obsidian
> vault**. 100 % local stack: [Ollama](https://ollama.com) + [Qdrant](https://qdrant.tech)
> embedded + [mem0](https://github.com/mem0ai/mem0). No API keys, no cloud, no
> telemetry. Memories are plain markdown files you can read, edit, sync to
> iCloud, and link from any other note.

```
┌────────────────┐     mcp stdio    ┌─────────────────────────────────────┐
│  AI agent      │ ───────────────▶ │  mem-vault-mcp (this package)       │
│  (Devin /      │                  │   • memory_save / search / list /   │
│   Claude Code  │ ◀─ markdown ───  │     get / update / delete           │
│   / Cursor /…) │                  │                                     │
└────────────────┘                  │   storage: <vault>/<memory_subdir>/ │
                                    │            *.md (one file per       │
                                    │             memory, full content)   │
                                    │   index:   ~/.local/share/mem-      │
                                    │            vault/qdrant (embedded)  │
                                    │   llm:     Ollama (qwen2.5:3b)      │
                                    │   embedder: Ollama (bge-m3, 1024d)  │
                                    └─────────────────────────────────────┘
```

## Why this exists

Most agent memory layers fall into one of three buckets:

1. **Cloud SaaS** ([mem0 Platform](https://docs.mem0.ai/platform/overview),
   [Letta](https://letta.com), …) — pay-per-call API keys, your memories on
   someone else's server.
2. **Opaque local DB** ([engram](https://github.com/engramhq/engram), …) — fast
   and local, but the data is locked in a binary store you can't open in your
   editor.
3. **Plain text in `~/`** (`MEMORY.md` files, [Devin
   skills](https://docs.devin.ai/reference/skills), …) — readable, but no
   semantic search.

**mem-vault tries to be all three at once.** Each memory is a markdown file
inside your Obsidian vault (so iCloud syncs it, Obsidian indexes it, and you
can edit it by hand), *and* it's embedded into a local Qdrant collection so
agents can query it semantically.

## Install

Requires Python 3.11+ and a running [Ollama](https://ollama.com/download).

```bash
git clone https://github.com/jagoff/mem-vault.git
cd mem-vault
uv tool install --editable .

# (optional) hybrid retrieval — adds fastembed for BM25 keyword scoring on top
# of the dense vector search. Adds ~200 MB of deps.
uv tool install --editable '.[hybrid]'

# pull the default models (≈3 GB total)
ollama pull qwen2.5:3b      # LLM extractor (only used when auto_extract=true)
ollama pull bge-m3          # 1024-dim multilingual embedder
```

This installs two binaries on your `PATH`:

- `mem-vault-mcp` — the MCP stdio server (what your agent talks to)
- `mem-vault` — top-level CLI with subcommands (`serve`, `import-engram`,
  `hook-sessionstart`, `hook-stop`, `version`). Bare `mem-vault` (no args)
  boots the MCP server, identical to `mem-vault-mcp`.

## Configure

Resolution order: env vars > `~/.config/mem-vault/config.toml` > defaults.

The only required value is **the path to your Obsidian vault**:

```bash
export MEM_VAULT_PATH="$HOME/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes"
```

Or the TOML version:

```toml
# ~/.config/mem-vault/config.toml
vault_path = "/Users/fer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes"
memory_subdir = "04-Archive/99-obsidian-system/99-AI/memory"  # default
llm_model = "qwen2.5:3b"          # override with qwen2.5:7b for sharper extraction
embedder_model = "bge-m3:latest"
embedder_dims = 1024
qdrant_collection = "mem_vault"
user_id = "default"
agent_id = "devin"                # optional — stamped on every memory
auto_extract_default = false      # opt-in LLM dedup; default off for predictability
```

| Env var | Field | Default |
| --- | --- | --- |
| `MEM_VAULT_PATH` (or `OBSIDIAN_VAULT_PATH`) | `vault_path` | auto-detect iCloud vault |
| `MEM_VAULT_MEMORY_SUBDIR` | `memory_subdir` | `04-Archive/99-obsidian-system/99-AI/memory` |
| `MEM_VAULT_STATE_DIR` | `state_dir` | `~/.local/share/mem-vault` |
| `MEM_VAULT_OLLAMA_HOST` | `ollama_host` | `http://localhost:11434` |
| `MEM_VAULT_LLM_MODEL` | `llm_model` | `qwen2.5:3b` |
| `MEM_VAULT_EMBEDDER_MODEL` | `embedder_model` | `bge-m3:latest` |
| `MEM_VAULT_EMBEDDER_DIMS` | `embedder_dims` | `1024` |
| `MEM_VAULT_COLLECTION` | `qdrant_collection` | `mem_vault_<agent_id>` if `agent_id` is set, else `mem_vault` |
| `MEM_VAULT_USER_ID` | `user_id` | `default` |
| `MEM_VAULT_AGENT_ID` | `agent_id` | `null` |
| `MEM_VAULT_AUTO_EXTRACT` | `auto_extract_default` | `false` |

## Connect your agent

### Devin for Terminal — `~/.config/devin/config.json`

```json
{
  "mcpServers": {
    "mem-vault": {
      "command": "/Users/you/.local/bin/mem-vault-mcp",
      "args": [],
      "transport": "stdio",
      "env": {
        "MEM_VAULT_PATH": "/Users/you/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes",
        "MEM_VAULT_AGENT_ID": "devin"
      }
    }
  }
}
```

### Claude Code / Claude Desktop — `~/.claude.json` or `claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mem-vault": {
      "command": "/Users/you/.local/bin/mem-vault-mcp",
      "env": {
        "MEM_VAULT_PATH": "/Users/you/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes",
        "MEM_VAULT_AGENT_ID": "claude-code"
      }
    }
  }
}
```

### Cursor / Windsurf / VS Code

Same shape — point any MCP-compatible client at `mem-vault-mcp`.

## Lifecycle hooks (Devin / Claude Code)

mem-vault ships two optional [lifecycle hooks](https://docs.anthropic.com/en/docs/claude-code/hooks)
that run inside the same virtualenv as the MCP server, so they always see all
dependencies:

| Hook | Subcommand | What it does |
| --- | --- | --- |
| `SessionStart` | `mem-vault hook-sessionstart` | At every session start, queries the vault for `type=preference` + `type=feedback` memories, emits an `additionalContext` block so the agent sees them right away — no cold start, no re-explaining preferences |
| `Stop` | `mem-vault hook-stop` | Appends a tab-separated audit line to `~/.local/share/mem-vault/sessions.log` whenever the agent finishes its turn |

Wire them up alongside your existing `hooks` config:

```json
{
  "hooks": {
    "SessionStart": [
      { "matcher": "", "hooks": [
        { "type": "command", "command": "/Users/you/.local/bin/mem-vault hook-sessionstart", "timeout": 15 }
      ]}
    ],
    "Stop": [
      { "matcher": "", "hooks": [
        { "type": "command", "command": "/Users/you/.local/bin/mem-vault hook-stop", "timeout": 5 }
      ]}
    ]
  }
}
```

Both hooks are best-effort — they print to stderr and exit 0 if anything goes
wrong (vault missing, Ollama down, etc.). They never block the session. See
[`hooks/README.md`](./hooks/README.md) for failure modes and verification.

## Tools exposed

| Tool | Purpose | Notes |
| --- | --- | --- |
| `memory_save` | Persist a new memory | `auto_extract=false` (default) writes the literal content; `auto_extract=true` runs the LLM to extract canonical facts and dedupe |
| `memory_search` | Semantic search | Returns full memory bodies, not just snippets |
| `memory_list` | Browse with filters | Filter by `type` / `tags` / `user_id` |
| `memory_get` | Read one memory | The id is just the filename slug |
| `memory_update` | Replace fields | Re-indexes if `content` changes |
| `memory_delete` | Remove file + index | Irreversible — confirm with the user |

Memory types are constrained to a small enum:
`feedback`, `preference`, `decision`, `fact`, `note`, `bug`, `todo`. Anything
else falls back to `note`.

## What a memory file looks like

```markdown
---
name: Idioma preferido para agents
description: Fer prefers Spanish rioplatense in agent replies; technical jargon stays English.
type: preference
tags:
  - language
  - rioplatense
created: '2026-04-28T20:21:36-03:00'
updated: '2026-04-28T20:21:36-03:00'
agent_id: devin
user_id: default
---
Fer prefers Spanish rioplatense in agent replies; technical jargon stays English.
```

You can edit this file directly in Obsidian, link to it from other notes
(`[[idioma_preferido_para_agents]]`), or grep it from the shell. Every change
re-syncs to iCloud automatically. The vector index will be rebuilt on the next
`memory_save` for that file (or you can `memory_update` to force a re-embed).

## Save modes: literal vs auto-extract

```python
# Literal (default) — fast, deterministic, agent decides what to remember
memory_save(content="...")           # auto_extract=false implied

# Auto-extract — slower, smarter, lets the LLM decide
memory_save(content="...", auto_extract=true)
# → runs Ollama (qwen2.5:3b)
# → extracts canonical facts, normalizes phrasing
# → diffs against existing memories
# → may emit ADD / UPDATE / NOOP / DELETE events
```

`auto_extract=true` adds latency (≈5–15 s with `qwen2.5:3b`, more with larger
models). Use it when you're ingesting a long conversation transcript and want
mem0's deduplication; stick with the default for one-line preferences and bug
fixes.

## Migrating from engram

If you've been using [engram](https://github.com/engramhq/engram), bulk-import
its memories into mem-vault in one shot:

```bash
engram export /tmp/engram-export.json
mem-vault import-engram /tmp/engram-export.json --agent-id engram --dry-run
mem-vault import-engram /tmp/engram-export.json --agent-id engram
```

Each engram observation becomes one `.md` file in your vault, tagged
`source:engram` plus the original `project:` and `topic_key` slugs so you can
trace it back. Pass `--auto-extract` to additionally run the LLM extractor and
let mem0 dedupe against any memories you already had.

## Smoke test

```bash
uv run python scripts/smoketest.py            # full cycle, including LLM
uv run python scripts/smoketest.py --skip-llm # skip auto_extract path
```

Expected output ends with `ALL CHECKS PASSED`.

## Storage layout

```
<vault>/04-Archive/99-obsidian-system/99-AI/memory/
├── feedback_local_free_stack.md
├── preference_idioma_rioplatense.md
├── decision_100_local_stack.md
└── …

~/.local/share/mem-vault/
├── qdrant/                      # embedded vector store (one collection per agent_id is possible)
│   └── collection/mem_vault/
└── history.db                   # mem0 audit log of ADD/UPDATE/DELETE events
```

The vault stores the **source of truth** in markdown. The local Qdrant
collection is a **derived index** — if it ever gets corrupted, delete
`~/.local/share/mem-vault/qdrant` and re-save the memories.

## Comparison with neighbours

| | mem-vault | [engram](https://github.com/engramhq/engram) | [mem0 Platform](https://docs.mem0.ai/platform/overview) | `MEMORY.md` files |
| --- | --- | --- | --- | --- |
| 100 % local | yes | yes | no (cloud) | yes |
| Editable in Obsidian | yes | no (binary DB) | no | yes |
| Semantic search | yes (Qdrant + bge-m3) | yes (sqlite) | yes | no |
| LLM dedup | optional (Ollama) | no | yes | no |
| Multi-user / multi-agent | yes (`user_id`, `agent_id`) | yes (`project`) | yes | no |
| API key required | no | no | yes (`m0-…`) | no |
| Sync across devices | yes (iCloud / Syncthing on the vault) | no | yes | depends |

Engram is fantastic if you want zero setup and don't care about reading the
data outside of an agent. mem-vault is for the case where the **vault itself
is your knowledge graph** and you want the agent's memory to live inside it.

## Roadmap

- [x] Auto-import existing engram exports (`engram export memories.json` → mem-vault) — `mem-vault import-engram`
- [x] Per-agent collections — auto, derived from `agent_id`
- [x] Optional fastembed BM25 hybrid retrieval — install `'.[hybrid]'`
- [x] Lifecycle hooks for Claude Code / Devin — `hook-sessionstart` + `hook-stop`
- [ ] `UserPromptSubmit` hook that injects per-prompt context from semantic search
- [ ] Per-agent visibility scopes (`agent_id_visible_to: [...]`)
- [ ] Browser UI to triage memories without leaving the terminal
- [ ] Memory consolidation (weekly LLM pass that merges near-duplicates)

## License

[MIT](./LICENSE).

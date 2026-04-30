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
- `mem-vault` — top-level CLI with subcommands (`serve`, `ui`, `doctor`,
  `import-engram`, `reindex`, `consolidate`, `hook-sessionstart`,
  `hook-userprompt`, `hook-stop`, `version`). Bare `mem-vault` (no args)
  boots the MCP server, identical to `mem-vault-mcp`.

Before you wire an agent, run `mem-vault doctor` to verify the setup:

```
$ mem-vault doctor

mem-vault doctor
─────────────────
  ✓  config           vault=… · ollama=http://localhost:11434 · llm=qwen2.5:3b · embedder=bge-m3:latest
  ✓  vault            …/99-AI/memory
  ✓  state            ~/Library/Application Support/mem-vault
  ✓  ollama           http://localhost:11434 responded in 4ms · 5 models
  ✓  model:embedder   bge-m3:latest
  ✓  model:llm        qwen2.5:3b
  ✓  qdrant           collection=mem_vault · 47 entries
  ✓  sync             vault=47 files · index in lockstep
  ✓  rerank           disabled (no fastembed installed; rank stays pure semantic)
  ✓  feedback         tracking on · boost=0.30

✓ all checks passed
```

Red cross = hard failure (fix before going further). Warning = non-blocking hint.
Pass `--skip-ollama` / `--skip-index` to skip the heavier checks in CI.

## Configure

Resolution order: env vars > `~/.config/mem-vault/config.toml` > defaults.

The only required value is **the path to your Obsidian vault**:

```bash
export MEM_VAULT_PATH="$HOME/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes"
```

Or the TOML version:

```toml
# ~/.config/mem-vault/config.toml  (or %APPDATA%\mem-vault\config.toml on Windows)
vault_path = "/path/to/your/Obsidian/Vault"
memory_subdir = "mem-vault"       # subfolder inside the vault — change to taste
llm_model = "qwen2.5:3b"          # bump to qwen2.5:7b for sharper extraction
embedder_model = "bge-m3:latest"
embedder_dims = 1024
qdrant_collection = "mem_vault"   # auto: "mem_vault_<agent_id>" if agent_id set
user_id = "default"
agent_id = "claude-code"          # optional — stamped on every memory
auto_extract_default = false      # opt-in LLM dedup; default off for predictability
```

| Env var | Field | Default |
| --- | --- | --- |
| `MEM_VAULT_PATH` (or `OBSIDIAN_VAULT_PATH`) | `vault_path` | auto-detected from common locations (see below) |
| `MEM_VAULT_MEMORY_SUBDIR` | `memory_subdir` | `mem-vault` |
| `MEM_VAULT_STATE_DIR` | `state_dir` | platform user-data dir (see below) |
| `MEM_VAULT_OLLAMA_HOST` | `ollama_host` | `http://localhost:11434` |
| `MEM_VAULT_LLM_MODEL` | `llm_model` | `qwen2.5:3b` |
| `MEM_VAULT_EMBEDDER_MODEL` | `embedder_model` | `bge-m3:latest` |
| `MEM_VAULT_EMBEDDER_DIMS` | `embedder_dims` | `1024` |
| `MEM_VAULT_COLLECTION` | `qdrant_collection` | `mem_vault_<agent_id>` if `agent_id` is set, else `mem_vault` |
| `MEM_VAULT_USER_ID` | `user_id` | `default` |
| `MEM_VAULT_AGENT_ID` | `agent_id` | `null` |
| `MEM_VAULT_AUTO_EXTRACT` | `auto_extract_default` | `false` |
| `MEM_VAULT_DECAY_HALF_LIFE_DAYS` | `decay_half_life_days` | `0` (disabled) |
| `MEM_VAULT_USERPROMPT_SCRIPTS` | (UserPromptSubmit hook) | `""` (disabled — accept all scripts) |

**Vault auto-detection** (when `MEM_VAULT_PATH` is unset). First match wins:

- `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes` (macOS, iCloud-synced Obsidian)
- `~/Obsidian`
- `~/Documents/Obsidian`
- `~/Notes`
- `~/Documents/Notes`
- `~/OneDrive/Obsidian`, `~/OneDrive/Documents/Obsidian` (Windows)

**State dir** (where the Qdrant index + audit log live):

- macOS: `~/Library/Application Support/mem-vault`
- Linux: `~/.local/share/mem-vault` (respects `$XDG_DATA_HOME`)
- Windows: `%LOCALAPPDATA%\mem-vault\mem-vault`

## Workflow — what using mem-vault feels like

mem-vault grows by **two paths** simultaneously:

1. **Explicit save** — you (or the agent) call `memory_save` for something
   specific you want remembered.
2. **Proactive auto-capture** — the agent observes the turn, decides "this
   is worth remembering for later", and saves on its own. No `/mv save`
   typed by you. This is where the system actually scales: every productive
   turn leaves a trace, and one month in you have 10x the relevant context
   without ever having lifted a finger to curate.

The bundled `SKILL.md` (installed by `mem-vault install-skill`) instructs
the agent to capture proactively when it sees a real bug fix with root
cause, a design decision with trade-offs, a workflow discovery, a codebase
convention, a gotcha, a measured performance finding, or a setup step that
took non-trivial effort. It explicitly does **not** capture cosmetic
changes, pure exploration, or anything already in `CLAUDE.md`/`AGENTS.md`.
You can disable per-session with "no guardes nada en mem-vault esta sesión".

Once configured, you don't think about mem-vault explicitly. Memories
accumulate as a side effect of normal conversation:

```
You: "preferimos rate limits a 60/min, no a 100. Recordá esto."
agent: invokes memory_save(content="...", type="preference",
                          tags=["rate-limit", "convention"])
       → file written to <vault>/<memory_subdir>/preferimos_rate_limits.md
       → embedding stored in Qdrant
```

Days later, a new session:

```
You: "diseñá la API de auth"
[UserPromptSubmit hook fires before the agent reads the prompt]
[hook does memory_search("API de auth diseñá") → finds rate-limit preference at score 0.71]
[the preference is injected into the agent's context]

agent: "Por las preferencias de rate limit (60/min) que ya pediste antes,
        propongo este diseño..."
```

You never had to remind it. The vault is a markdown folder you can open
in Obsidian, edit, link from other notes, and grep. The agent reads it
automatically.

### Three save modes worth knowing

```python
# Literal save — fast, deterministic, no LLM
memory_save(content="X is Y because Z")
# → writes the .md, embeds the body, returns. ~150 ms.

# Auto-extracted — slower, LLM rewrites + dedupes against existing
memory_save(content="long conversation transcript...", auto_extract=True)
# → ollama distills facts, may emit ADD/UPDATE/NOOP per fact. ~5–15 s.

# Private to one agent
memory_save(content="claude-code internal note", visible_to="private")
# → only the saving agent_id sees it via memory_list/memory_search.
```

### Maintenance commands you'll occasionally run

```bash
# Bring the index up to date after editing .md files in Obsidian:
mem-vault reindex

# Detect near-duplicates and ask the LLM to merge them:
mem-vault consolidate                       # dry-run
mem-vault consolidate --apply               # actually merge
mem-vault consolidate --threshold 0.95      # stricter (only obvious dupes)

# Bring memories from another tool:
engram export /tmp/engram-export.json
mem-vault import-engram /tmp/engram-export.json --agent-id engram
```

### Browser UI for triage

When you'd rather click than grep, install the optional UI extra and open
the local-only web app:

```bash
uv tool install --editable '.[ui]'
mem-vault ui                  # serves on http://127.0.0.1:7880
mem-vault ui --port 8088      # custom port
```

The UI runs locally (binds to `127.0.0.1` by default — never to
`0.0.0.0`), no auth, intended for single-machine use. Features:

- Filter by type / tag / search query (semantic via mem-vault, no LLM call)
- Inline edit of body, title, tags, visibility
- Delete memories (with confirmation)
- Live stats: total count, by type, by agent

The UI talks to the same `MemVaultService` as the MCP server — there's no
separate sync to maintain.

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

### Slash commands `/mv` · `/mem_vault` · `/memory` (Devin)

Once mem-vault is installed (`uv tool install mem-vault` or `pip install mem-vault`),
register the slash command skill so you can invoke the MCP from the Devin
prompt without typing the full tool name:

```bash
mem-vault install-skill
```

This drops three alias copies of the bundled `SKILL.md` into Devin's user
skills dir (`~/.config/devin/skills/{mv,mem_vault,memory}/SKILL.md` on
macOS/Linux, `%APPDATA%\devin\skills\...\SKILL.md` on Windows). Open a
new Devin session and try:

```
/mv tests para mem-vault         # semantic search (default)
/mv list                          # last 20 memories
/mv save -e Hoy descubrí que…     # save with LLM extractor
/mv get <id>                      # show one memory
/mv delete <id>                   # delete (asks for confirmation)
```

`/mem_vault` and `/memory` are aliases — pick whichever the dedo te tira
más rápido. Re-running `mem-vault install-skill` is idempotent (skips
existing files); pass `--force` to overwrite, `--dry-run` to preview,
`--no-aliases` to install only `/mv`, `--target <dir>` to install into a
project-local `.devin/skills` instead, or `--uninstall` to clean up.

## Lifecycle hooks (Devin / Claude Code)

mem-vault ships two optional [lifecycle hooks](https://docs.anthropic.com/en/docs/claude-code/hooks)
that run inside the same virtualenv as the MCP server, so they always see all
dependencies:

| Hook | Subcommand | What it does |
| --- | --- | --- |
| `SessionStart` | `mem-vault hook-sessionstart` | At every session start, queries the vault for `type=preference` + `type=feedback` memories, emits an `additionalContext` block so the agent sees them right away — no cold start, no re-explaining preferences |
| `UserPromptSubmit` | `mem-vault hook-userprompt` | Before each user message, runs a semantic search against the index and injects the top-3 most relevant memories. Skips short messages (`<20` chars) and slash-commands automatically. Tunable via `MEM_VAULT_USERPROMPT_K`, `MEM_VAULT_USERPROMPT_THRESHOLD`, `MEM_VAULT_USERPROMPT_MIN_CHARS` |
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
    "UserPromptSubmit": [
      { "matcher": "", "hooks": [
        { "type": "command", "command": "/Users/you/.local/bin/mem-vault hook-userprompt", "timeout": 12 }
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
| `memory_save` | Persist a new memory | `auto_extract=false` (default) writes the literal content; `auto_extract=true` runs the LLM to extract canonical facts and dedupe. `visible_to` controls who can read it (`"public"` / `"private"` / `["agent-a", "agent-b"]`) |
| `memory_search` | Semantic search | Returns full memory bodies, not just snippets. Respects visibility — over-fetches and filters post-hoc |
| `memory_list` | Browse with filters | Filter by `type` / `tags` / `user_id`. Same visibility rules as search |
| `memory_get` | Read one memory | The id is just the filename slug |
| `memory_update` | Replace fields | Re-indexes if `content` changes. Can change `visible_to` |
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

## Browser UI

Sometimes you just want to see what the agent has been remembering, without
opening Obsidian. The `ui` subcommand boots a local web app for that:

```bash
mem-vault ui                       # http://127.0.0.1:7880
mem-vault ui --port 8000           # custom port
mem-vault ui --host 0.0.0.0        # expose on the LAN (no auth! local only)
```

What you get:

- semantic search bar (HTMX-driven, debounced — types over the embedder index)
- filters by `type` and tag chips
- list of memories sorted by most-recently-updated, with type pill, name,
  description preview, tags, agent_id, and a relative timestamp
- click a row → modal with the full body, editable inline
- save → re-indexes the memory and refreshes the row
- delete → removes the `.md` file + every embedding pointing to it

### Graph view

Open `/graph` in the same UI to see the vault as a force-directed graph:

- **Nodes**: every memory, colored by type.
- **Edges**: pairs of memories that share at least N normalized tags
  (default 2). Tags shaped as `project:foo` are split on `:` so
  `project:rag` and `project:rag-obsidian` cluster together.
- Sliders let you tune edge threshold (1-5 shared tags) and node cap
  (20-500 most-recent).
- Click a node → detail card with description, tags, agent, last update.
- Renders [Cytoscape.js](https://js.cytoscape.org/) with the cose
  force-directed layout. Smooth up to ~500 nodes.

Stack: FastAPI server-rendered + HTMX (~14 KB) + Cytoscape.js (~250 KB,
graph view only) + hand-rolled CSS. No build step, no SPA framework, no
JavaScript bundler.

The UI dependencies are shipped as an optional extra:

```bash
uv tool install --editable '.[ui]'
# or, with hybrid retrieval too:
uv tool install --editable '.[ui,hybrid]'
```

## Consolidate (merge near-duplicates)

After a few weeks of `auto_extract=True`, the vault accumulates near-
duplicates: "Fer prefers TS over JS" + "the user uses TypeScript" + "code
should be in TS". The `consolidate` subcommand finds and merges them with
a local LLM:

```bash
mem-vault consolidate                          # dry-run, prints proposed merges
mem-vault consolidate --threshold 0.90         # only very similar pairs (default 0.85)
mem-vault consolidate --max-pairs 10           # limit per run
mem-vault consolidate --apply                  # actually merge them
```

How it works:

1. Walk every memory, hit Qdrant for the K nearest neighbors per memory.
2. Keep pairs above the cosine similarity threshold.
3. For each pair, ask Ollama (`qwen2.5:3b` by default) to choose: `MERGE`,
   `KEEP_BOTH`, `KEEP_FIRST`, or `KEEP_SECOND`. Strict-JSON output.
4. Apply: `MERGE` rewrites the older memory with a fused body and deletes
   the newer one; `KEEP_FIRST/SECOND` deletes the redundant memory; the
   index is kept consistent throughout.

Run it weekly via `cron` / `launchd` to keep the vault tidy.

## Per-agent visibility

Memories support an optional `visible_to: [...]` allowlist:

```yaml
---
name: Internal note for Devin
agent_id: devin
visible_to: []           # private to the owner agent only
# visible_to: [claude-code, codex]  # explicit allowlist
# visible_to: ["*"]                   # public (default — same as omitting the field)
---
```

When a viewer's `agent_id` is passed to `VaultStorage.list(viewer_agent_id=…)`
or to the future `memory_search` filter, restricted memories are filtered
out unless the viewer is the owner or appears in the allowlist. Memories
without the field default to public (backward-compatible — your existing
`.md` files keep working).

This is the building block for "private notes per agent" without renaming
collections or running multiple servers.

## Reindex (after editing memories by hand)

The vector index is updated automatically on every `memory_save` /
`memory_update`. If you edit a memory's `.md` directly in Obsidian, or you
import memories from an external source, the embeddings can drift out of
sync. Rebuild the index any time with:

```bash
mem-vault reindex                  # idempotent: re-embeds every memory
mem-vault reindex --purge          # nukes the Qdrant collection first (clean slate)
mem-vault reindex --auto-extract   # also runs the LLM extractor while reindexing
mem-vault reindex --limit 20       # debugging: stop after N memories
```

A reindex of ~50 memories takes ~10 s on bge-m3 + Apple Silicon.

## Export (backup, migration, or feed-the-LLM)

Dump every memory to a portable file:

```bash
mem-vault export json    -o backup.json         # one JSON object, full data
mem-vault export jsonl   -o backup.jsonl        # one memory per line
mem-vault export csv     -o backup.csv          # spreadsheet-friendly
mem-vault export markdown -o backup.md          # one document, all bodies
```

Filters and trimming:

```bash
mem-vault export json --type preference          # only preferences
mem-vault export jsonl --tag rag-obsidian        # only one tag
mem-vault export csv --no-body -o lite.csv       # drop bodies, keep metadata
```

The `json` and `jsonl` outputs include enough metadata to round-trip back
into a future importer (out of scope for v0.1). The `markdown` format is
useful for pasting the entire vault into one LLM prompt for analysis.

## Time-decay scoring

By default, `memory_search` ranks purely by semantic similarity — a
relevant decision from 2 years ago beats a relevant note from yesterday.
Set `MEM_VAULT_DECAY_HALF_LIFE_DAYS` (or `decay_half_life_days` in the
TOML) to enable time-decay reranking:

```bash
export MEM_VAULT_DECAY_HALF_LIFE_DAYS=90
```

With a half-life of 90 days, the score multiplier is `2 ** (-age_days / 90)`:

| Age of memory | Score multiplier |
| --- | --- |
| Today | 1.00 |
| 30 days ago | 0.79 |
| 90 days ago | 0.50 |
| 180 days ago | 0.25 |
| 365 days ago | 0.06 |

Memories without an `updated` timestamp (legacy / engram imports) get
multiplier 1.0 and aren't punished. The reranker is applied after the
Qdrant kNN — `top_k` is oversampled 3x internally so older items can be
shuffled down past the cut.

Reasonable values: 30 (aggressive — surface-this-week wins), 90
(moderate — last-month context wins), 365 (mild — only graveyard
memories suffer). Set to `0` (default) to disable.

## Locale-aware UserPromptSubmit

By default, the `UserPromptSubmit` hook only skips short messages and
slash-commands. If your vault is mostly written in Latin-script languages
(English, Spanish, French, etc.) and you don't want the embedder to spend
cycles on accidentally-pasted CJK / Arabic / Hebrew / Devanagari, set:

```bash
export MEM_VAULT_USERPROMPT_SCRIPTS="latin"
# or, if you want Latin and Cyrillic:
export MEM_VAULT_USERPROMPT_SCRIPTS="latin,cyrillic"
```

Available script buckets: `latin`, `cyrillic`, `greek`, `arabic`,
`hebrew`, `devanagari`, `cjk`, `thai`, `ethiopic`. The hook detects the
dominant Unicode script of the prompt (using `unicodedata.name()` — no
ML model, no extra dep) and skips the search when it doesn't match the
allowlist.

Prompts that are mostly digits / emoji / punctuation return `unknown`
and pass through (we'd rather over-search than skip a real query).

## Cross-vault sync

mem-vault doesn't ship its own sync engine — it leans on whatever the
user already has in front of the vault: [iCloud Drive](https://support.apple.com/en-us/HT204025),
[Syncthing](https://syncthing.net), [Dropbox](https://dropbox.com),
[OneDrive](https://onedrive.live.com), [Google Drive](https://drive.google.com),
or a `git` repo. Memories are plain markdown — every existing sync
engine handles them natively.

Two helper commands make sync sane:

```bash
mem-vault sync-status        # diff the vault vs the Qdrant index
mem-vault sync-watch         # long-running file watcher → reindex on change
```

`sync-status` reports:

```
  vault files       : 57
  index entries     : 55
  stale (vault > idx): 2     ← .md edited after last embedding
  orphans in index  : 0      ← embedding for a deleted memory
  missing in index  : 2      ← .md exists but never embedded
```

`sync-watch` keeps them in lockstep automatically: every `.md` create /
modify / delete event triggers a re-embed (or a removal). Designed to run
as a long-lived process — point your `cron` / `launchd` / `systemd` at
`mem-vault sync-watch` and forget about it.

> **Important caveat**: the embedded Qdrant DB enforces single-writer.
> `sync-status` and `sync-watch` cannot run while the MCP server is
> active (it holds the lock). Stop your agent first, run the sync
> command, then start the agent again. Both commands give a clear
> error message instead of crashing if the lock is held.

### Recommended sync setups

| Setup | Pros | Cons |
| --- | --- | --- |
| **iCloud Drive** (default if your vault is in Obsidian iCloud) | zero setup, works on iPhone/iPad too | no conflict resolution; can produce `.icloud` placeholders for cold files |
| **Syncthing** between Mac + Linux | LAN-fast, no third party, conflict files visible | one-time setup per device |
| **`git` repo** | full history, atomic commits, conflict resolution | manual `git pull` / `git push`; not real-time |
| **Dropbox / OneDrive / Drive** | mature, reliable | proprietary, requires account |

The only directory you must NOT sync is the `state_dir` (the local
Qdrant index + history.db). It's a derived cache, machine-specific, and
binary — sync conflicts will corrupt it. After every fresh pull on a new
machine, run `mem-vault reindex` to rebuild the local index from the
synced markdown.

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

## Tests

Unit tests covering storage atomicity, slugify normalization, config
resolution, and concurrency safety:

```bash
uv run pytest tests/ -v
# 40 passed in ~1.4s
```

End-to-end smoke test against a live Ollama:

```bash
uv run python scripts/smoketest.py            # full cycle, including LLM
uv run python scripts/smoketest.py --skip-llm # skip auto_extract path
```

Expected output ends with `ALL CHECKS PASSED`.

### Atomic writes

Every memory write goes through `atomic_write_bytes` (`storage.py`): the
file is written to a sibling temp file, `fsync`'d, then `os.replace`'d into
place. On POSIX this is a single `rename(2)` syscall — guaranteed atomic.
On Windows it maps to `MoveFileExW` with `REPLACE_EXISTING`. A reader
that opens the file mid-write sees either the old contents or the new
contents, never a partial mix. Verified by `test_atomic_write_no_partial_on_concurrent_reads`.

## Storage layout

```
<vault>/<memory_subdir>/        # e.g. <vault>/mem-vault/  (default)
├── feedback_local_free_stack.md
├── preference_idioma_rioplatense.md
├── decision_100_local_stack.md
└── …

<state_dir>/                    # platform-appropriate user-data dir
├── qdrant/                      # embedded vector store (one collection per agent_id)
│   └── collection/<qdrant_collection>/
└── history.db                   # mem0 audit log of ADD/UPDATE/DELETE events
```

The vault stores the **source of truth** in markdown. The local Qdrant
collection is a **derived index** — if it ever gets corrupted, run
`mem-vault reindex --purge` and the index rebuilds from the markdown
files in seconds.

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
- [x] Lifecycle hooks for Claude Code / Devin — `hook-sessionstart` + `hook-userprompt` + `hook-stop`
- [x] Reindex command for hand-edited / external-source memories — `mem-vault reindex`
- [x] Per-agent visibility scopes — `visible_to: ["*" | [] | ["agent-a", ...]]`
- [x] Memory consolidation (LLM pass that merges near-duplicates) — `mem-vault consolidate`
- [x] Browser UI for triage (FastAPI + HTMX) — `mem-vault ui`
- [x] Graph visualization (Cytoscape.js, tag co-occurrence edges) — `/graph` route
- [x] Export to JSON / JSONL / CSV / Markdown for backup — `mem-vault export`
- [x] PyPI release pipeline — `.github/workflows/release.yml` (Trusted Publishing on tag)
- [x] Cross-vault sync — `mem-vault sync-status` + `mem-vault sync-watch` (works with iCloud / Syncthing / git / Dropbox / OneDrive)
- [x] Time-decay scoring for `memory_search` — `MEM_VAULT_DECAY_HALF_LIFE_DAYS`
- [x] Locale-aware `UserPromptSubmit` skip — `MEM_VAULT_USERPROMPT_SCRIPTS`

## License

[MIT](./LICENSE).

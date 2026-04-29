# Changelog

All notable changes to mem-vault are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Export**: new `mem-vault export {json,jsonl,csv,markdown}` subcommand
  for backups, migration, or feeding the whole vault to an LLM as one
  prompt. Supports `--type` / `--tag` filters and `--no-body` for compact
  exports.
- **Release pipeline**: `.github/workflows/release.yml` publishes to PyPI
  via Trusted Publishing (OIDC, no API token) on every `v*.*.*` tag, and
  drafts a GitHub Release with notes pulled from this CHANGELOG.

## [0.1.0] - 2026-04-28

Initial public release. Everything below shipped between commits
`f6c844e` and `b345694`.

### Added
- **MCP stdio server** (`mem-vault-mcp`) exposing six tools: `memory_save`,
  `memory_search`, `memory_list`, `memory_get`, `memory_update`,
  `memory_delete`.
- **Markdown source-of-truth**: every memory is a `.md` file with YAML
  frontmatter inside the user's Obsidian vault. Editable by hand,
  greppable, syncable via iCloud / Syncthing / Dropbox.
- **Local vector index**: Qdrant in embedded mode (no Docker) + Ollama
  embeddings (`bge-m3` default, 1024 dims). Zero API keys, no cloud
  calls.
- **Optional LLM extraction**: `auto_extract=True` on `memory_save` runs
  Ollama (`qwen2.5:3b` default) to extract canonical facts and dedupe
  against existing memories.
- **Browser UI** (`mem-vault ui`): FastAPI + HTMX + hand-rolled CSS.
  Search, filter, edit, delete memories without opening Obsidian. No
  build step, no SPA framework.
- **Lifecycle hooks** (`hook-sessionstart`, `hook-userprompt`,
  `hook-stop`): inject relevant memories into agent context
  automatically; compatible with both Devin for Terminal and Claude
  Code hook formats.
- **Engram importer** (`import-engram`): bulk-load `engram export` JSON
  into the vault with `source:engram` provenance tags.
- **Reindex** (`reindex --purge`): rebuild the Qdrant collection from
  the markdown source of truth in seconds.
- **Consolidate** (`consolidate --apply`): detect near-duplicate
  memories with Qdrant kNN + ask the LLM to MERGE / KEEP_BOTH /
  KEEP_FIRST / KEEP_SECOND. Run weekly to keep the vault tidy.
- **Per-agent visibility**: `visible_to: [...]` allowlist in
  frontmatter. Memories without the field default to public.
- **Per-agent collections**: when `agent_id` is set, the default Qdrant
  collection becomes `mem_vault_<agent_id>` for natural isolation.
- **Hybrid retrieval** (extra `[hybrid]`): adds `fastembed` for BM25
  keyword scoring blended with the dense vector search.
- **Atomic writes**: every save goes through `os.replace` with a
  same-directory tempfile. Concurrent readers see either the old or the
  new contents — never a partial mix.
- **Cross-platform paths**: `platformdirs` resolves the state directory
  per OS (macOS / Linux / Windows). Vault auto-detection covers iCloud
  Obsidian, Linux/macOS manual paths, and Windows OneDrive.
- **Tests**: 69 unit tests (storage atomicity, slugify, config, export,
  consolidate, visibility) — all run without Ollama or Qdrant.
- **CI** (GitHub Actions): matrix Ubuntu × macOS × Windows × Python
  3.11 / 3.12; lint job with `ruff check` + `ruff format --check`.

[Unreleased]: https://github.com/jagoff/mem-vault/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jagoff/mem-vault/releases/tag/v0.1.0

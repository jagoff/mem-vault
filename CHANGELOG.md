# Changelog

All notable changes to mem-vault are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-05-01

**UI overhaul: `/memory/` deja de ser un explorador plano y pasa a ser un
dashboard de curaduría.** El panel anterior listaba todas las memorias en
una sola lista con search semántica + filtro por type — útil para encontrar
una memoria específica, pero no daba ninguna pista de qué necesita atención
ni cómo se está comportando el corpus.

### Added — 5 tabs (All / Top / Quality / Duplicates / By project)

Tab bar nuevo en `/memory/`. Cada tab es un fragment HTMX dedicado con
endpoint propio (`/api/quality`, `/api/duplicates`, `/api/top`,
`/api/by-project`); el state del tab activo se persiste en `window.location.hash`
así un reload o un bookmark vuelven al mismo lugar.

- **All** — la lista plana de antes, ahora con search semántica y filtro por type.
- **Top** — 4 sub-rankings side-by-side: 👁 más usadas / 👍 más helpful /
  👎 más unhelpful (revisar) / 🪦 zombies (sin uso ni votos — candidatas a borrar).
- **Quality** — todo lo que `service.lint` flagged, con la lista de issues
  desplegada debajo de cada row para arreglar in-place.
- **Duplicates** — pares con jaccard ≥ 0.7 mostrados side-by-side, con
  instrucción de merge via `mv merge <id1> <id2>` (la fusión real sigue siendo
  CLI, no autoclick — evita merges destructivos no intencionales).
- **By project** — agrupa memorias por su tag de proyecto principal
  (heurística: top-tags con guión y no en una whitelist de tags genéricos
  como `rag`, `frontend`, `launchd`...). Cada grupo es `<details>` collapsable.

### Added — thumbs inline en cada row sin abrir el modal

Antes el thumbs up/down vivía dentro del modal de detalle (`_feedback.html`).
Para puntuar una memoria había que click → modal → buscar el botón → click.
Ahora cada row tiene 👍 / 👎 inline (visible on hover) que postean a
`/api/memories/{id}/feedback?inline=1`. El endpoint devuelve el row entero
re-renderizado y HTMX hace `outerHTML` swap → el counter se actualiza in-place
sin abrir nada.

El endpoint POST acepta `inline=1` opcional; sin el flag mantiene el
comportamiento legacy (devuelve solo el `_feedback.html` chunk para el modal).

### Added — counters condicionales + quality badge en cada row

`_row.html` ahora muestra:

- 👁 `usage_count` / 👍 `helpful_count` / 👎 `unhelpful_count` solo si > 0
  (evita pintar `0 0 0` en cada zombie row, que es ruido visual).
- Badge `⚠ needs work` si la memoria está en el set de lint (`lint_ids` que
  el endpoint inyecta al template). Click en el badge → tab Quality.

### Added — `with_issues` + `duplicates` pills en el header

`/api/stats` ahora devuelve dos pills extra:

- `⚠ N need work` — total de memorias con issues de lint. Click → tab Quality.
- `⧉ N dups` — total de pares con jaccard ≥ 0.7. Click → tab Duplicates.

Los counters se mirrorean a los tabs (vía `data-stat` attributes) así el user
ve "quality 63 / duplicates 17" sin abrir el tab. Los datos se cachean junto
al cache existente de `/api/stats` (TTL 30s) — un solo refresh del header
hace 1 read del corpus, no 3.

### Added — sort param para `/api/memories`

Sorts soportados: `usage_desc`, `helpful_desc`, `unhelpful_desc`, `zombie`
(memorias sin signal alguno, ordenadas por `updated` ascendente para que las
más viejas aparezcan primero), `recent`. Cuando el sort necesita rankear todo
el corpus (no solo las 100 más recientes) el endpoint sube el límite a 500
y después slicea. Esto alimenta la tab Top.

### Changed — `_row.html` truncá tags > 6 con un `+N` chip

Memorias con muchos tags (la del wave-2 fine-tunning tiene 11) hacían wrap
hasta 3 líneas. Ahora se muestran los primeros 6 + un `+N more` chip.

Generated with [Devin](https://cli.devin.ai/docs)

## [0.4.0] - 2026-04-30

**Beta release.** Project bumps from "Alpha" → "Beta" in pyproject classifiers.
658 tests at v0.3.0 grew to 736 tests, +6 commits of features +
1 audit pass + 1 robustness pass + 1 Beta-readiness sprint.

### Added — RemoteMemVaultService remote-mode handler symmetry

When `MEM_VAULT_REMOTE_URL` was set the MCP server crashed at boot
with `AttributeError: Service RemoteMemVaultService is missing
handler 'briefing' for tool 'memory_briefing'.`. Cause: the in-process
`MemVaultService` had grown 9 new tools (briefing, derive_metadata,
stats, duplicates, lint, related, history, feedback, synthesize)
across recent commits while the HTTP-backed `RemoteMemVaultService`
was left at the original 6 (save/search/list/get/update/delete).
The MCP boot-time symmetry check (`_build_handlers`) refused to
start.

Fix: 9 new `/api/v1/*` endpoints in `mem_vault/ui/server.py` (plus
3 Pydantic request models for the POST bodies), and 9 matching
delegating methods in `RemoteMemVaultService`. The contract test
described below would have caught this before it shipped.

### Added — Schema↔implementation contract tests

New `tests/test_contracts.py` (11 tests) parametrically guards 4
contracts that have caused regressions in this repo:

1. Every tool in `_TOOLS` resolves to a callable on `MemVaultService`.
2. Every tool in `_TOOLS` resolves to a callable on `RemoteMemVaultService`.
3. Every entry in `ENV_TO_CONFIG_FIELD` references a real `Config` field.
4. Every Pydantic request model field (`MemoryCreate.visible_to`,
   `MemoryUpdate.tags`, …) is consumed by the matching service method.

Caught a real bug on first run: `MemoryUpdate.visible_to` was declared
in the schema but `MemVaultService.update` never forwarded it to
`storage.update`, silently dropping visibility changes. Fixed in same
commit + 4 new regression tests in `test_visibility.py`.

For introspection (and the test), `ENV_TO_CONFIG_FIELD` is now a
module-level dict in `mem_vault.config` instead of buried inline
inside `load_config`.

### Added — `mem-vault metrics` CLI

The JSONL metrics sink (added in 0.3.0) was write-only — users couldn't
see their own performance data without external tooling. New
`mem-vault metrics` reads `<state_dir>/metrics.jsonl` and reports:

```
mem-vault metrics
───────────────────────────────────────────────────────────────────
  total: 142    errors: 3    error rate:  2.11%

  tool                        count   err     p50     p95     p99
  ─────────────────────────────────────────────────────────────────
  memory_briefing                12    0      216     232     234
  memory_save                    34    0       78   4.72s   5.13s
  memory_search                  91    3!     234     1.07s  1.20s
```

Filters: `--since 24h|7d|30m|2w|<ISO>`, `--tool <name>` (repeatable),
`--errors-only` / `--ok-only`, `--top-slow N` (default 5),
`--json` (machine output). Process-safe (read-while-server-writes is
fine, append-only sink), tolerates malformed lines (warn + skip).

30 tests on the pure helpers (`parse_since`, `percentile`, `aggregate`,
`top_slow_calls`, `filter_lines`) + 4 E2E.

### Added — PT/galego → ES post-filter for `memory_synthesize`

`memory_synthesize` was returning Portuguese-leaked output ~5-15% of
the time when source memorias contained technical vocabulary that
overlapped with PT roots — even though the system prompt asked for
"español rioplatense". Three coordinated fixes:

1. **Filter port**: new `mem_vault/iberian_filter.py` with 100+
   high-confidence regex pairs ported from `obsidian-rag` (commit
   `582406f`). Catches months (`março` → `marzo`), pronouns
   (`você` → `vos`, `essa` → `esa`), demonstratives, articles
   (`à` → `a la`, `os ` + lowercase → `los `), `-ção/-ência/-ância`
   suffixes, and the PT-only verb forms observed in real leaks
   (`fazer`, `feito`, `vou`, `tem` + lowercase, `foram`, …).
   Conservative — never rewrites words shared between PT and rioplatense.

2. **Aggressive REGLA 0 prompt anchor**: the synthesis prompt now
   opens with an explicit deny list of the most-common leak words
   ("Si te encontrás escribiendo 'feito', 'fazer', 'tem'… es BUG"),
   demands voseo argentino, repeats the rule at close.

3. **`Config.synthesis_model`**: new field (env
   `MEM_VAULT_SYNTHESIS_MODEL`) so users can route synthesis through
   a larger model (recommended: `qwen2.5:7b`) without changing the
   3b default for the auto-extract/dedup path. Smaller models drift
   to PT more aggressively under the same prompt.

End-to-end on the same source corpus, with `MEM_VAULT_SYNTHESIS_MODEL=qwen2.5:7b`:
zero PT leaks observed. Tests: 25 new in `test_iberian_filter.py`
including idempotency, pure-Spanish round-trip, and the regression
corpus from the actual 2026-04-30 leak.

### Refactored — `_TOOLS` extracted to `mem_vault/tool_schemas.py`

The 425-line block of declarative MCP tool schemas (every
`types.Tool(name=..., inputSchema=...)` for the 15 verbs) lived
inline in `server.py`. Moved to a dedicated module so the wiring
file (`_build_handlers`, `call_tool` dispatch, `MemVaultService`)
stays focused on mechanism. `server.py` drops from 2148 → ~1755 LOC.

`from mem_vault.server import _TOOLS` continues to work via re-export
for backwards compat with existing tests / external scripts.

### Performance — corpus-list cache

5 service verbs (briefing, stats, duplicates, lint, related) walked
the entire vault on every call (`storage.list(limit=∞)`). At 80
memorias each call took <50 ms; at 1000+ it became the bottleneck
of the boot briefing the `/mv` skill renders on every session start.

Added `MemVaultService._list_corpus(tags=...)` — TTL=30 s memoized
wrapper keyed by `frozenset(tags)`. Mutating verbs (save/update/delete
+ auto-link/auto-contradict secondary writes) call
`_invalidate_corpus_cache()` so a fresh write is visible on the next
discovery call without waiting for the TTL to lapse.

The 5 affected verbs now go through the cache. Discovery batches
(briefing + stats + lint fired back-to-back) hit storage once instead
of three times. 8 new tests in `test_corpus_cache.py` covering
memoization, invalidation, per-tag bucketing, TTL expiration, and
the cross-verb sharing invariant.

### Added — `MemVaultService.update` honors `visible_to`

The HTTP endpoint and the local Python service signature both
accepted `visible_to` (with `"private"` / `"public"` shorthands and
explicit allowlists), but `service.update` was silently dropping the
field. PATCH requests changed nothing. Now forwarded to
`storage.update` exactly like `save` does, with the same shorthand
normalization. The MCP tool schema for `memory_update` also gains
the `visible_to` property so MCP clients can change visibility.

### Added — project-scoped metadata + search filter

``memory_save`` now stamps a ``project`` field into the Qdrant payload
metadata, derived (in order of precedence) from:

1. Explicit ``project`` arg to the tool.
2. The first ``project:X`` tag in the memory's tag list.
3. ``Config.project_default`` (env ``MEM_VAULT_PROJECT=<name>``).

``memory_search`` picks up the same field as a filter when the user
passes ``project`` explicitly, or falls back to the config default.
Passing ``project: "*"`` (or an empty string) bypasses the default
for a truly-global search from a scoped session.

Why a dedicated field instead of the existing ``tags`` filter: Qdrant
indexes scalar payload values (``keyword`` index) much faster than an
array-contains lookup over ``tags``. For a vault with 1k+ memorias
and multi-project tags, the difference is measurable (~3-5x speedup
on the filter step alone) — and the code stays simple because
mem0 already forwards ``filters={"project": "..."}`` straight to
Qdrant.

Also surfaced in ``mem-vault doctor`` under the ``project`` row so
the user can see what scope is active.

Tests: 10 new in `test_project_scope.py` covering save-side derivation
(explicit > tag > default > none), search-side filter composition,
and the wildcard-bypass semantics.

### Added — `mem-vault eval` (retrieval regression harness)

Run a labeled query set through ``memory_search`` and report hit@1/3/5/10
plus MRR. The eval file is a JSON list of ``{query, expected, tag?}``
entries; ``expected`` is the ids that should land in the top-k.

```
$ mem-vault eval --queries tests/fixtures/my_queries.json --threshold 0.6

mem-vault eval
──────────────────────────────────────────────────
  queries     : 20
  hit@1       : 65.00%
  hit@3       : 85.00%
  hit@5       : 90.00%
  hit@10      : 95.00%
  mrr         : 0.7833
```

Non-zero exit when ``hit@5 < --threshold``, so CI can gate on
regressions. ``--json`` dumps the full report (including per-query
rows) for machine processing. Pure helpers in ``mem_vault.eval`` so
the metric math is unit-testable without Ollama (``compute_metrics``,
``reciprocal_rank``, ``hit_at``, ``diff_reports``, ``load_queries``).

Tests: 23 new in `test_eval.py` covering every pure function +
round-trip through JSON.

### Added — secret redaction on save (default on)

Memorias often capture commands, configs, or tracebacks carrying
credentials. Once a body is written it can sync to iCloud / Syncthing
/ Dropbox / git — each one a potential leak path. Every `memory_save`
now scans the body through a regex-based redactor before the `.md`
hits disk or the index. Matches are replaced with `[REDACTED:<kind>]`
and counted in a `redactions` field on the response envelope.

Patterns covered:

- AWS Access Key Id (`AKIA…` / `ASIA…`) and Secret Access Key
  (after `aws_secret_access_key=`)
- GitHub tokens (`ghp_` / `gho_` / `ghu_` / `ghr_` / `ghs_` prefixes)
- Anthropic keys (`sk-ant-…`) — **listed before** the generic
  OpenAI pattern so the superset doesn't swallow them
- OpenAI keys (`sk-…`)
- Slack tokens (`xox[baprs]-…`)
- Google API keys (`AIza…`)
- JWT (three base64url segments)
- `Authorization: Bearer <token>`
- PEM private-key blocks (`-----BEGIN … PRIVATE KEY-----` … `END`)
- Assignment shapes: `password=`, `passwd:`, `secret=`, `api_key:`,
  `api_token:`, `api_secret:`, `access_token:`, `auth_token:`,
  `client_secret:`, bare `token=`

Idempotent — running `redact` on already-redacted text is a no-op
(values starting with `[REDACTED:` are skipped via a negative
lookahead). Extensible via `redaction.EXTRA_PATTERNS` for
org-specific prefixes.

Opt out via `MEM_VAULT_REDACT_SECRETS=0` (not recommended).

Tests: 18 new in `test_redaction.py` covering each pattern, clean
text (no mutation), idempotence, empty input, the fast-check helper,
and integration with `MemVaultService.save` (body on disk + index
both see the redacted version, summary surfaced in the response).

### Added — `memory_related` MCP tool (walk the graph)

Takes a memory id and returns its neighbors grouped by relationship
type — all four sources in one call:

```
{
  "related":            [...],     # frontmatter `related:`
  "contradicts":        [...],     # frontmatter `contradicts:`
  "cotag_neighbors":    [...],     # ≥N shared normalized tags
  "semantic_neighbors": [...],     # top-k semantic search
}
```

Co-tag normalization splits `project:foo` on colon so
`project:rag` / `project:rag-obsidian` cluster naturally. Semantic
step is opt-out via `include_semantic=false` to keep the call cheap
and deterministic.

Tests: 9 new in `test_related.py` covering each neighbor source,
self-exclusion, the include_semantic flag, and the standard
validation / not-found contracts.

### Added — `memory_history` MCP tool + JSONL sidecar snapshots

Every `VaultStorage.update` now snapshots the pre-update state to
`<id>.history.jsonl` next to the `.md`. The new `memory_history` MCP
tool reads that sidecar and returns entries newest-first. Useful for
"what did this memory say last week?" and for recovering a field
that got overwritten.

Snapshot shape: body, name, description, tags, related, contradicts,
plus a timestamp. Usage counters (which churn on every search) are
deliberately NOT snapshotted to keep the sidecar lean. No-op updates
(same value written back) skip the snapshot so the auto-link /
auto-contradict paths don't bloat history with identical entries.

Storage contract:
- `history_path_for(id)` — returns `<memory_dir>/<id>.history.jsonl`.
  Suffix is `.jsonl` so existing `*.md` globs don't pick it up.
- `read_history(id, limit=50)` — newest-first, corrupt lines skipped.
- `delete(id)` also unlinks the sidecar (no orphan history).

Tests: 16 new in `test_history.py` covering snapshot path, first
update creates one entry, multiple updates order correctly, no-op
update skips snapshot, related-only / contradicts-only changes
still snapshot, limit caps output, delete cleans up, corrupt lines
are skipped, sidecar ignored by `list()`, tool envelope validation.

### Added — concurrent `mem-vault reindex` (`--concurrency`)

`reindex` now runs embed calls in parallel via `asyncio.Semaphore`
(default `--concurrency=4`). Cold reindex on a laptop goes from ~2
memorias/sec to ~7-8/sec against `bge-m3`. Counters are protected by
an `asyncio.Lock` so the progress summary stays accurate under
contention. `--concurrency=1` preserves the legacy strictly-
sequential behavior for debug / low-memory environments.

Notes:
- The `--limit N` stop is best-effort with concurrency: when the
  embed call is near-instantaneous (e.g. in tests) all workers enter
  simultaneously and none trip the stop. In production with Ollama
  latency the counter fills quickly and the stop kicks in.
- No change to the content-hash skip logic or the orphan-sweep pass.

Tests: 2 new in `test_hash_reindex.py` (concurrent path indexes all,
limit + concurrency doesn't deadlock). Existing sequential tests
pass via `concurrency=1` default in the test helper.

### Added — `contradicts:` auto-detection on save

The `contradicts` frontmatter field has existed in the `Memory` schema
since v0.1 but was never populated — there was no mechanism to decide
which memorias conflict. `memory_save` now optionally runs a local LLM
pass over the top-5 most similar existing memorias and asks which (if
any) the new body directly contradicts. Matches get stamped in the new
memory's `contradicts:` list.

**Prompt contract** (strict JSON, no prose):

```
{"contradicts": ["<id1>", "<id2>"]}
```

Rules in the prompt stress "direct contradiction" (incompatible claims
about the same subject), NOT "related" or "weaker version of" — that
distinction keeps the field signal-rich. Empty list is the common
correct answer.

**Safety**:

- Prompt sanitizes + truncates each body (same helpers as
  `memory_synthesize`) so injection attempts in memory bodies can't
  redirect the classifier.
- Response parser accepts list or comma-separated string, filters
  hallucinated ids (not in the candidate set), dedupes, preserves order.
- Every failure path (LLM timeout, breaker open, non-JSON output,
  storage write fail) degrades to `contradicts=[]` without breaking
  the save.

**Opt-in** via `auto_contradict=true` per-call or `auto_contradict_default`
(env `MEM_VAULT_AUTO_CONTRADICT=1`) globally. Off by default — adds
3-5 s latency. Composes with auto-link: both fields populate in the
same save (related + contradicts are orthogonal signals).

`storage.update()` now accepts a `contradicts=[...]` kwarg for the
same round-trip contract as `related=[...]`.

**Tests**: 20 new unit tests in `test_contradict.py`: prompt builder
(block wrapping, STRICT JSON instruction, long-body truncation),
response parser (standard list, comma-string variant, hallucination
filter, dedup, missing key, alternate key `contradictions`, empty/non-
JSON/non-object edge cases, non-string items), service integration
(default off, LLM match stamps frontmatter, no match leaves empty,
LLM failure degrades, hallucinated ids filtered, config default
triggers LLM). Suite grows to 459.

### Added — real hybrid retrieval (BM25 + dense + Reciprocal Rank Fusion)

`memory_search` now optionally runs a BM25 sparse retriever in parallel
to the dense vector search and merges the two rankings with
[Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).
Catches exact-keyword matches (error strings, command names, file
paths, identifiers) the dense embedder misses.

**How it composes**:

1. Dense bi-encoder (Ollama `bge-m3`) → top `raw_k` hits.
2. BM25 on the in-memory corpus (`name + body + tags`) → top `raw_k`.
3. RRF: `score(d) = Σ 1 / (k + rank_i(d))` with `k=60`.
4. (optional) Cross-encoder reranker → re-scores the fused union.
5. Usage boost on the rerank/rrf score.
6. Top-k cut.

No new dependency: the BM25 is inline (~120 LOC, standard Okapi with
`k1=1.5`, `b=0.75`). For single-user vaults (≤2k memorias) the rebuild
takes <10 ms; above that consider Qdrant's native sparse support.

**`HybridRetriever`** caches the BM25 index in memory and invalidates
on every `memory_save` / `memory_update` / `memory_delete` (including
the auto-link re-write that happens post-save). Invalidate is O(1);
the rebuild runs lazily on the next search.

**Config** (all off by default, opt-in):

```
MEM_VAULT_HYBRID=1              # enable
MEM_VAULT_HYBRID_RRF_K=60       # smoothing constant
MEM_VAULT_HYBRID_BM25_K1=1.5    # TF saturation
MEM_VAULT_HYBRID_BM25_B=0.75    # length normalization
```

**Tests**: 27 new unit tests in `test_hybrid.py`: tokenize, BM25Index
(empty, single doc, TF saturation, zero-score skip, top-k), RRF
(single-list, intersection, dedup, k-shape, deterministic ties),
fuse_dense_and_bm25 (combined, dense metadata preserved, stub for
BM25-only, empty inputs), HybridRetriever (first-search rebuild,
invalidate picks up new memoria), and integration (service rescues
a BM25-only match via RRF; invalidate fires on save + delete). Suite
grows to 439.

### Added — `mem-vault doctor` (one-shot health diagnostic)

A single command that runs every check new users hit during setup, with a
short colored report and a non-zero exit code when something is broken.
Inspired by `brew doctor` / `nvm doctor`.

```
$ mem-vault doctor

mem-vault doctor
─────────────────
  ✓  config           vault=… · ollama=http://localhost:11434 · llm=qwen2.5:3b
  ✓  vault            …/memory
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

Checks performed:

- **config**: `load_config` succeeds (catches misconfigured `MEM_VAULT_PATH`).
- **vault**: path exists, is a directory, and we can write a canary file
  (catches permission surprises).
- **state**: `state_dir` writable (Qdrant + history.db land).
- **ollama**: `GET /api/tags` on the configured host; reports latency and
  model count.
- **model:embedder** / **model:llm**: each configured model is installed
  (prefix-matched against installed tags — `qwen2.5:3b` matches
  `qwen2.5:3b-instruct-q4_K_M`). Emits a `ollama pull …` hint on miss.
- **qdrant** / **sync**: tries `sync_status`; reports drift counts and
  falls back to a warning (not an error) when the collection is locked
  by a running MCP server.
- **rerank**: cross-signal warning when `fastembed` is installed but
  `reranker_enabled=False` (user likely forgot the flag), and the
  reverse — an error when the flag is on but the extra is missing.
- **fastembed-cache** (macOS): warn if `FASTEMBED_CACHE_PATH` is unset;
  the default resolves to `/var/folders/…` which gets wiped on reboot.
- **feedback**: surface `usage_boost_enabled` / `usage_boost` /
  `usage_tracking_enabled` so the user remembers what's on.

Exit codes: `0` all green · `1` warnings only · `2` at least one hard
failure. Flags `--skip-ollama` / `--skip-index` for CI / offline dev.

**Tests**: 27 new unit tests in `test_doctor.py` covering every check
in isolation (stubbed `ollama.Client` + `sync_status`) plus end-to-end
via `run()`. Suite grows to 412.

### Added — feedback loop + usage-based ranking (the "self-supervised memory")

The vault now learns what's useful from how the agent actually uses it.
Every `memory_search` hit leaves a breadcrumb; every thumbs flip turns the
ranking dial. No training loop, no eval harness, no external signal —
just the agent's own citation pattern feeding back into `memory_search`.

**New `Memory` frontmatter fields** (all optional, back-compat):

- `usage_count` — how many times this memory was returned by search.
- `helpful_count` / `unhelpful_count` — explicit thumbs via
  `memory_feedback` (or inferred via the Stop hook on citation).
- `last_used` — ISO timestamp of the most recent retrieval / feedback
  event, useful for dead-memory detection.

Legacy `.md` files without these fields load with zeros — no migration
needed. Frontmatter stays tight: zero-valued counters are omitted on
write so pristine memorias don't sprout noise.

**New `memory_feedback` MCP tool**:

```python
memory_feedback(id="<id>", helpful=True)   # 👍 helpful_count += 1
memory_feedback(id="<id>", helpful=False)  # 👎 unhelpful_count += 1
memory_feedback(id="<id>")                 # plain "used", no polarity
```

All three bump `last_used`. Storage writes bypass `updated` on purpose
so the hash-based incremental reindex doesn't re-embed a memory just
because its counter moved.

**Usage boost in `memory_search`**:

`score` is now multiplied by `1 + usage_boost * max(0, helpful_ratio)`
where `helpful_ratio = (helpful - unhelpful) / max(1, helpful + unhelpful)`.
Default boost magnitude is `0.3` — a memory with `helpful_ratio=1` gets
its score lifted by 30 %, enough to flip a close neighbor below it.
The clamp at 0 means a single thumbs-down doesn't actively bury a
memory (clamp to a floor of 1.0, not below); it just neutralizes the
boost. Over-fetch + reorder happens before the top-`k` cut so feedback
genuinely changes ordering, not just scores.

Composes with the cross-encoder reranker: when rerank is on, the
rerank score is the base the boost multiplies (rerank × usage_boost),
so both signals stack.

Opt out via `MEM_VAULT_USAGE_BOOST_ENABLED=0` (keeps semantic-only
order) or `MEM_VAULT_USAGE_BOOST=0` (mechanical disable). Per-search
tracking can be disabled via `MEM_VAULT_USAGE_TRACKING=0` for
benchmarking / scripted workloads that shouldn't pollute counters.

**Auto-tracking in `memory_search`**: every returned hit has its
`usage_count += 1` and `last_used` bumped post-hoc. Cheap (one atomic
`.md` rewrite per hit), best-effort (failures swallowed), and produces
the signal that feeds the boost above.

**Auto-feedback via citation in the Stop hook**: `mem-vault hook-stop`
now scans the agent's last response (via `transcript_path` for Claude
Code, `payload["response"]` direct, or a fallback empty) for memory
id citations in three forms:

- `[[id]]` Obsidian wikilinks (strongest signal),
- `` `id` `` inline code spans,
- bare word-bounded mentions of ids ≥8 chars (pre-filtered against
  the known-id set to keep false positives near zero).

Each match calls `record_feedback(helpful=None)` — a neutral "this was
used" bump, no polarity. Controlled via `MEM_VAULT_STOP_AUTO_FEEDBACK`
(default on). The audit line in `sessions.log` now carries
`auto_feedback=N` so you can grep how many memories got bumped per
session.

**Browser UI** picks up the same counters: the detail modal shows
`seen N · 👍 N · 👎 N · last used Xh ago` pills and two action buttons
that POST to `/api/memories/{id}/feedback` with `helpful=true|false`.
HTMX swaps the counter chunk in place without reloading the page.

**Tests**: 42 new unit tests across `test_feedback_loop.py` (storage
counters + round-trip + service boost) and `test_hooks.py` (citation
detection + auto-feedback wiring). Total suite grows from 343 → 385.

### Changed — bundled `SKILL.md` syncs with the in-use richer version

The `SKILL.md` shipped via `mem-vault install-skill` (175 lines) was
materially leaner than the version several users were already running
locally (544 lines). Synced the bundled template with the latter so a
fresh `install-skill` lands the same flow that drove the past few
months of mem-vault usage:

- **Discovery verbs** documented inline: `/mv stats`, `/mv recent [n]`,
  `/mv top <tag>`, `/mv duplicates`, `/mv timeline <project>`,
  `/mv lint`, `/mv merge <id1> <id2>`.
- **Auto-context injection** spec — silent `memory_search` on the first
  tool call of each task when the user message is >20 chars, with
  threshold + skip rules.
- **Auto-save triggers** numbered (10 categories), each with a type
  hint and a real-world example, plus an explicit anti-spam clause
  (no auto-save in the last 10 min, body <200 chars, etc.).
- **Title / type / tags classifier** documented inline as a fallback
  for callers that don't have access to `memory_derive_metadata`. The
  ≥3-tag rule is hard, with a one-line ask-the-user prompt when
  derivation falls short.
- **Project-tag override table** for memories about agent configs
  (`~/.config/devin/skills`, `~/.claude/...`) so they don't pick up
  the cwd's repo tag by mistake.
- **Cross-linking** (`## Memorias relacionadas` + `[[wikilinks]]`)
  spelled out as part of the save flow.
- **Output formatting** examples per verb.

The `test_skill_template_contains_auto_capture_directive` test was
loosened to probe the *contract* (proactive section + main triggers +
opt-out) rather than specific phrasing, so future copy edits don't
trigger false negatives.

### Fixed — loop-closure audit (11 bugs)

Full audit of the eight learning loops (save → index → search → update →
delete → consolidate → reindex → hooks). Suite grew from 323 to 343 tests;
no regressions.

- **`memory_search` skips orphan hits.** A Qdrant entry whose `.md` was
  deleted out-of-band (manual `rm`, post-`consolidate` crash, sync hiccup)
  used to be returned with `memory: null`. Worse, the visibility filter
  silently degraded: a previously `visible_to: []` (private) memory leaked
  to *any* viewer once its file was gone. Now orphans are dropped early —
  the vault is the source of truth, period.
- **`memory_update` surfaces re-index failures.** A failed `index.add` after
  a successful `.md` write used to return `{"ok": True}` with the index
  silently empty. Now the response mirrors `memory_save`'s envelope:
  `{"ok": True, "indexed": False, "indexing_error": "...",
  "indexing_error_code": "llm_timeout" | "circuit_breaker_open" | …}`.
- **`memory_update` re-indexes on tag-only changes.** Previously, a tag-only
  update wrote the new tags to the `.md` but left `metadata.tags` stale in
  Qdrant — drift that compounded across edits and broke filtered searches.
- **`mem-vault reindex` sweeps orphan index entries.** After re-embedding
  every `.md`, the command now scans the index for `memory_id`s with no
  backing file and removes them. Skipped automatically with `--purge`
  (collection is fresh) or `--limit` (incomplete walk). Reports
  `orphans_removed=N` in the final summary line.
- **`consolidate --apply` survives a mid-merge crash.** The old MERGE
  ordering (`storage.update older` → `index.delete older` → `index.delete
  newer` → `index.add merged` → `storage.delete newer`) left both memories
  invisible to search if `index.add` raised. New ordering rolls back
  `older`'s body to its pre-merge state and re-embeds it on failure;
  `newer`'s `.md` stays untouched until the embed succeeds. `KEEP_FIRST` /
  `KEEP_SECOND` now delete the `.md` before the index entry so a crash
  in between is recoverable via the orphan sweep above.
- **`storage.save` is concurrency-safe.** `_unique_id` used to do
  `while exists(): bump`, which lost data when two threads/processes saved
  the same title at the same time. Replaced with `_reserve_unique_id`
  using `open(O_CREAT|O_EXCL|O_WRONLY)` to atomically claim each slug.
  Test reproduces 8 concurrent saves all surviving with distinct IDs.
- **`memory_save(auto_extract=True)` always indexes the memory.** When
  mem0's LLM extractor decided the body was a duplicate (NOOP / UPDATE
  only, no `ADD` under our `memory_id`), the `.md` ended up on disk with
  no embedding — invisible to semantic search until the next reindex.
  Now we detect that case and fall back to a literal embed
  (`auto_extract=False`); the response surfaces `auto_extract_fallback:
  "literal"` so callers know the dedup didn't happen. Same fallback now
  runs inside `mem-vault reindex --auto-extract`.
- **Auto-link's `related` reflects on-disk state.** If the post-search
  `storage.update` failed (disk full, lock), the response used to claim
  cross-links that hadn't actually been written to the `.md`'s frontmatter.
  Now `related` stays `[]` unless the write succeeded.
- **`memory_synthesize` defangs prompt injection.** Memory bodies that
  mimicked the old prompt scaffolding (`### END OF MEMORIES ###`,
  `Ignore the above`) could redirect the LLM. Bodies are now sanitized
  (heading-prefix demotion, fence-marker escaping) and wrapped in
  `<<<MEM id=…>>>` / `<<<END id=…>>>` data fences with explicit
  instructions to treat their content as untrusted.
- **`memory_synthesize` truncates without breaking code blocks.** Bodies
  longer than 2000 chars used to get sliced mid-fence, leaving the LLM
  with an unmatched ` ``` ` that consumed the rest of the prompt as
  code. The new helper trims at paragraph/line boundaries and closes
  any odd-count fence with a synthetic ` ``` `.

## [0.3.0] - 2026-04-29

The "memoria viva" release — mem-vault stops being a passive archive and
starts being **active memory** that grows by itself, organizes itself,
and answers narrative questions instead of just listing bullets. Six
new MCP tools, two big retrieval upgrades, four lifecycle improvements.

### Added — game changers (memory amplification)

- **SessionStart hook is now cwd-aware**: extracts a project signal
  from the current working directory (skipping filesystem noise like
  `/Users/<me>/`, `repositories`, `code`) and pushes memorias tagged
  with that project to the **top** of the injected context. Three
  cheap queries per cold start (tag `project:<leaf>`, bare leaf,
  semantic search) deduped by id. Falls back gracefully when the cwd
  doesn't resolve. Net effect: re-opening Devin in a repo where you
  worked before, the agent comes pre-loaded with that project's
  decisions and gotchas — zero re-explaining.

- **Auto-capture proactive directive in the bundled `SKILL.md`**: the
  shipped skill now instructs the agent to call `memory_save` at the
  end of any non-trivial turn (bug fixes with root cause, design
  decisions with trade-offs, codebase conventions, gotchas, measured
  performance findings, non-trivial setup steps, explicit user
  preferences). Includes 4 anti-trigger categories (cosmetic edits,
  pure exploration, info already in CLAUDE.md/AGENTS.md, explicit
  "no guardes esto"). Without manually typing `/mv save`, every
  productive turn now leaves a trace.

- **Auto-linking on save** (`Config.auto_link_default=True`): after a
  successful `memory_save`, the service runs a second semantic search
  (threshold 0.5, k=5) and stamps the IDs of similar memorias on the
  new memory's `related:` frontmatter, AND inserts a
  `## Memorias relacionadas` section with `[[id]]` Obsidian wikilinks
  in the body itself (cap 3, placed before any `## Aprendido el ...`
  closer). Builds an emergent knowledge graph at zero extra cost —
  Obsidian's graph view becomes navigable without manual curation.
  Per-call opt-out with `auto_link=false`.

- **`memory_synthesize` MCP tool**: takes a topic/query, runs a wide
  semantic search (default k=10, threshold 0.1), then asks the local
  Ollama LLM to compose a narrative summary citing source IDs inline.
  Converts the vault from "list of bullets" into "interlocutor that
  responds". Errors come back as structured envelopes
  (`code: validation_failed | llm_timeout | llm_failed`) so callers
  can fall back to plain `memory_search` if synthesis isn't available.

- **`memory_briefing` MCP tool**: composable boot summary at the first
  `/mv` of a session. Returns `{project_tag, total_global,
  project_total, recent_3, top_tags, lint_summary}` so the skill can
  render the 8-line briefing without a second round-trip. Lets the
  agent enter the conversation knowing what's in the vault.

- **`memory_derive_metadata` MCP tool**: server-side classifier that
  takes content + cwd and returns `{title, type, tags, missing_tags}`
  using regex-priority tables (type: bug → decision → todo →
  preference → feedback → fact → note; 20 domain-tag patterns; 7
  technique-tag patterns; project_tag from cwd with content-based
  overrides for agent-config paths). The skill uses this *before*
  `memory_save` so the user only types the body and the metadata gets
  filled in automatically — `missing_tags > 0` means the body didn't
  hit enough patterns for ≥3 tags and the skill should ask for one
  more before saving.

- **Cross-encoder reranker for `memory_search`** (opt-in via
  `Config.reranker_enabled` + `[hybrid]` extra): wraps `fastembed`'s
  `TextCrossEncoder` (default `jinaai/jina-reranker-v1-tiny-en`,
  ~130 MB, ~30-60 ms for 20 candidates on CPU). Over-fetches a wider
  candidate set from the bi-encoder and re-orders it through a model
  that sees query+candidate together — captures interactions the
  embedder alone misses (negation, specificity, entity matching).
  Falls back silently to bi-encoder order when fastembed isn't
  installed; never raises.

### Added — discovery / hygiene tools

- **`memory_stats(cwd?)`**: counts by `type`, by `agent_id`, top tags,
  age histogram (today / week / month / older). Scopes to the
  resolved `project_tag` when cwd provided.
- **`memory_duplicates(threshold=0.7, cwd?)`**: surfaces candidate
  duplicate pairs by tag-overlap Jaccard. Cheap offline check that
  fits in a chat turn. For deep semantic dedup keep using
  `mem-vault consolidate`.
- **`memory_lint(cwd?)`**: list memorias with structural issues
  (<3 tags, missing description, body shorter than 100 chars, body
  ≥300 chars without `## Aprendido el YYYY-MM-DD`, missing created).

### Added — incremental reindex

- **Hash-based incremental skip in `mem-vault reindex`**: every
  indexed memory now carries a `content_hash` (truncated SHA-256) in
  its Qdrant payload. Subsequent `reindex` runs skip memories whose
  current hash matches the indexed one — re-embedding only the diff.
  For stable vaults (most memories unchanged between runs) this is
  ~50-100× faster. New `--force` flag disables the skip.
- New helpers: `index.compute_content_hash(content)` and
  `index.get_by_metadata(key, value, user_id)`.

### Added — observability + ops

- **Optional JSONL metrics sink** (`Config.metrics_enabled`,
  `MEM_VAULT_METRICS=1`): one structured line per MCP tool call to
  `<state_dir>/metrics.jsonl` with `{ts, tool, duration_ms, ok,
  error?}`. Thread-safe append-mode writer that auto-disables on
  persistent IO error. Designed for `tail -f` + `jq` ad-hoc analysis,
  no Prometheus dependency.
- **mypy** added to the dev tooling. Balanced profile (not strict —
  the FastAPI/MCP decorator soup is too noisy in strict mode) with
  `check_untyped_defs`, `strict_optional`, `warn_unused_ignores`,
  `warn_no_return`. CI now runs `mypy` after `ruff check`.

### Added — distribution

- **`mem-vault install-skill`** subcommand: drops a bundled `SKILL.md`
  template into Devin's user skills directory under three alias names
  (`/mv`, `/mem_vault`, `/memory`), all routing to the same MCP tools.
  Cross-platform (macOS / Linux / Windows). Flags: `--target`
  (custom dir like `.devin/skills` in a repo), `--force` (overwrite),
  `--dry-run` (preview), `--no-aliases` (only `/mv`), `--uninstall`.
  Solves the manual-symlink dance — `pip install mem-vault &&
  mem-vault install-skill` now leaves the slash commands ready.

### Changed

- **`cli.py` (891 lines)** is now a `cli/` package with one module per
  subcommand. `cli/__init__.py` is the dispatcher; `cli/{crud,
  consolidate, export_cmd, hooks_cmd, import_engram, install_skill,
  reindex, sync_cmd, ui}.py` each register their own subparsers and
  expose `run(args)`. Heavy deps are still imported lazily inside the
  `run` functions so `mem-vault --help` stays instant. Zero
  user-facing behavior change (entry point still
  `mem_vault.cli:main`, all flags + help text identical).
- **Bundled `SKILL.md` generalized for public distribution**: removed
  paths hardcoded to the author's vault layout
  (`04-Archive/99-obsidian-system/...`, `01-Projects/...`), bumped
  default search threshold from 0.05 to 0.1 (matches the MCP server
  default, fewer noisy hits). Re-ordered the alias header to put
  `/mv` first (the canonical primary).
- **`memory_save` schema**: new optional `auto_link` boolean to opt
  out per-call.

### Robustness (carry-over from 0.2.0, restated for clarity)

- LLM timeout (`MEM_VAULT_LLM_TIMEOUT_S`, default 60 s) wrapping every
  Ollama call; structured `indexing_error_code: "llm_timeout"`
  envelope when it trips.
- Circuit breaker (3 consecutive failures → 30 s open) preventing
  stacked hangs on a dead Ollama.
- `MEM_VAULT_MAX_CONTENT_SIZE` (default 1 M chars) rejection before
  the vault or index is touched.
- Bearer-token auth on the UI / JSON HTTP server (`MEM_VAULT_HTTP_TOKEN`).
  `serve()` refuses to bind to a non-loopback host without a token.
  Constant-time comparison via `secrets.compare_digest`.

### Tests

Total suite is now **323 tests** in ~13 s, zero Ollama / Qdrant required
(everything stubbed). Highlights of the new coverage:

- `test_breaker.py`, `test_robustness.py`, `test_ui_auth.py` (Phase 1).
- `test_hooks.py`, `test_server_handlers.py`, `test_server_dispatch.py`,
  `test_cli_common.py`, `test_cli_import_engram.py`,
  `test_index_helpers.py` (Phase 3).
- `test_hash_reindex.py` (incremental reindex).
- `test_metrics.py` (JSONL sink).
- `test_install_skill.py` (cross-platform skill installer).
- `test_linking_synthesize.py` (auto-link + memory_synthesize).
- `test_reranker.py` (cross-encoder reranker, fastembed stubbed).
- `test_discovery_tools.py` (briefing + derive_metadata + stats +
  duplicates + lint + wikilinks insertion).

## [0.2.0] - 2026-04-29

Robustness pass — three latent failure modes that could either hang the MCP
server or silently expose memories on a LAN are now closed.

### Added
- **LLM timeout** (`MEM_VAULT_LLM_TIMEOUT_S`, default `60`): every
  Ollama-backed call (embeddings + auto-extractor) is wrapped in
  `asyncio.wait_for`. When Ollama hangs (model loading, OOM, dead host)
  the MCP server now returns a structured `indexing_error_code:
  "llm_timeout"` instead of blocking the call indefinitely. The `.md`
  file is still saved on disk; only the index step degrades. Set to `0`
  to disable the timeout (legacy behavior).
- **Circuit breaker** in `index.py`: after three consecutive Ollama
  failures the breaker opens for 30 s, short-circuiting `add` and
  `search` with `CircuitBreakerOpenError` so a dead Ollama doesn't keep
  stacking 60 s timeouts. Heals on the first success after the cooldown.
- **Content size limit** (`MEM_VAULT_MAX_CONTENT_SIZE`, default
  `1_000_000` chars): `memory_save` and `memory_update` reject oversized
  bodies with `ok: false, code: "content_too_large"` before touching
  the vault or the index. Set to `0` to disable.
- **Optional bearer-token auth** for the UI / JSON HTTP server
  (`MEM_VAULT_HTTP_TOKEN`): when set, every endpoint except `/healthz`
  requires `Authorization: Bearer <token>`. Constant-time comparison
  via `secrets.compare_digest`. The startup helper `serve()` now refuses
  to bind to a non-loopback host without a token, preventing accidental
  LAN exposure of the unauthenticated CRUD API.
- **`RemoteMemVaultService` token support**: the HTTP client picks up
  the bearer token from an explicit `token=...` arg, the `Config`
  field, or `MEM_VAULT_HTTP_TOKEN` (in that order) and attaches it to
  every request. 401 / 403 responses now return a structured
  `code: "unauthorized"` with a hint, instead of looking like a 5xx.

### Tests
- 60+ new unit tests across `test_breaker.py`, `test_robustness.py`,
  and `test_ui_auth.py` covering: breaker state machine, oversized
  content rejection, hung-Ollama timeout (via stub `time.sleep` in a
  fake `VectorIndex`), middleware auth (missing / wrong scheme / wrong
  token / valid / runtime rotation), `_is_loopback_host` parametric
  table, `serve()` startup guards. Total suite is now 132 tests, still
  zero Ollama / Qdrant required for CI.

## [0.1.x preview] - 2026-04-28

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

[Unreleased]: https://github.com/jagoff/mem-vault/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/jagoff/mem-vault/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jagoff/mem-vault/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jagoff/mem-vault/releases/tag/v0.1.0

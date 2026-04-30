---
name: mv
description: "Search/save/list/update/delete/feedback/history/related + verbos de descubrimiento (stats/recent/top/duplicates/timeline/lint/merge/doctor/eval) en mem-vault — el MCP local de memoria infinita backed by Obsidian (markdown plano + Qdrant + Ollama, 100% local). Triggers ALIAS-EQUIVALENTES: `/mem_vault`, `/memory` y `/mv` — los tres invocan EXACTAMENTE el mismo flow. Sin argumentos → boot briefing (1× por sesión: total + últimas 3 + top tags + lint flags) + listar últimas 10. Use cuando el user tipea `/mv <query>` (search semántico hibrido BM25+dense+RRF opt-in, decay opcional, usage-boost auto), `/mv list`, `/mv save <texto>` (auto-deriva title/type/tags/project + cross-link [[wikilinks]] + secret redaction; `-e` LLM extractor; `--contradict` flag opcional), `/mv get <id>`, `/mv update <id> <texto>`, `/mv delete <id>` (PIDE confirmación), `/mv feedback <id> up|down` (👍/👎 lifts ranking), `/mv history <id>` (snapshots pre-update), `/mv related <id>` (walk-the-graph: related/contradicts/cotag/semantic), `/mv stats` (counts), `/mv recent [n]`, `/mv top <tag>`, `/mv duplicates`, `/mv timeline <project>`, `/mv lint`, `/mv merge <id1> <id2>` (PIDE confirmación), `/mv doctor` (health check via CLI), `/mv eval --queries <file>` (hit@k baseline). Spec auto-save triggers (10 casos) + auto-context injection al task start + auto-citation en Stop hook (cita un id `[[X]]` o `` `X` `` en tu respuesta y mem-vault bumpea su last_used). Siempre routeá al MCP tool `mcp__mem-vault__memory_*` correspondiente, NUNCA escribas el .md a mano (el MCP maneja frontmatter + index Qdrant + history sidecar)."
argument-hint: "<query> | list | save [-e] [--contradict] <text> | get <id> | update <id> <text> | delete <id> | feedback <id> [up|down] | history <id> | related <id> | stats | recent [n] | top <tag> | duplicates | timeline <project> | lint | merge <id1> <id2> | doctor | eval --queries <file>"
---

# /mem_vault · /memory · /mv — router para mem-vault MCP

Los tres triggers (`/mem_vault`, `/memory`, `/mv`) son **alias-equivalentes**: invocan el mismo flow, parsean los mismos subcomandos, devuelven el mismo formato. El user puede usar el que más rápido le salga del dedo.

[mem-vault](https://github.com/jagoff/mem-vault) es un MCP local que da memoria persistente backed by Obsidian. Memorias = `.md` planas en el vault (típicamente `04-Archive/99-obsidian-system/99-AI/memory/<id>.md`), index semántico en Qdrant embedded (`~/.local/share/mem-vault/qdrant/`), embeddings via Ollama (bge-m3, 1024d). El user te invocó con `/mem_vault $ARGUMENTS` (o `/memory $ARGUMENTS` o `/mv $ARGUMENTS`) y vos ruteás al MCP tool que corresponde.

## Routing — parseá la PRIMERA palabra de `$ARGUMENTS`

| Primera palabra | Acción | Sección |
|---|---|---|
| (vacío) | boot briefing + listar últimas 10 (briefing 1× por sesión) | "Boot briefing" + Output List |
| `list` | listar últimas 20 | Output List |
| `save` (con `-e` y/o `--contradict` opcional) + texto | auto-derivación + cross-link + redaction + opcional contradict-detect | "Save flow" |
| `get` + id | mostrar memoria completa | Output Get |
| `update` + id + texto | replace content (snapshot al sidecar history antes de pisar) | Output Update |
| `delete` + id | borrar `.md` + sidecar history + index (CON confirmación) | "Delete flow" |
| `feedback <id> [up\|down]` | 👍/👎 explícito (sin polarity = "used"); lifts el score en searches futuras | "Feedback loop" |
| `history <id> [limit]` | snapshots pre-update (newest-first) | "Verbo: history" |
| `related <id>` | walk-the-graph: 4 buckets (related/contradicts/cotag/semantic) | "Verbo: related" |
| `stats` | counts por type, top tags, edades, lint summary | "Verbos de descubrimiento" |
| `recent [n]` | últimas `n` por created (default 10) | "Verbos de descubrimiento" |
| `top <tag>` | memorias con `<tag>`, sorted by recency | "Verbos de descubrimiento" |
| `duplicates` | pairs con similarity ≥0.70 candidatos a merge | "Verbos de descubrimiento" |
| `timeline <project>` | línea temporal de memorias con tag `<project>` | "Verbos de descubrimiento" |
| `lint` | memorias con problemas (<3 tags, sin date, body skinny, etc.) | "Verbos de descubrimiento" |
| `merge <id1> <id2>` | combina 2 memorias en 1 (CON confirmación) | "Verbos de descubrimiento" |
| `doctor` | shell out: `mem-vault doctor` (health check de Ollama/qdrant/vault/models) | "Verbo: doctor" |
| `eval --queries <file> [--threshold N]` | shell out: `mem-vault eval` (hit@k + MRR sobre query set labeled) | "Verbo: eval" |
| cualquier otra cosa | search semántico | `mcp__mem-vault__memory_search({query: $ARGUMENTS, k: 5, threshold: 0.05})` |

## Feedback loop — self-supervised search ranking

El vault aprende qué te sirve a partir de cómo lo usás. Tres señales independientes, todas escritas a frontmatter del `.md`:

| Campo | Cuándo se incrementa | Quién lo escribe |
|---|---|---|
| `usage_count` | cada vez que la memoria sale como hit en `memory_search` | el MCP, post-hoc en search |
| `helpful_count` | tool `memory_feedback({helpful: true})` | el agent (vos) o el user |
| `unhelpful_count` | tool `memory_feedback({helpful: false})` | el agent o el user |
| `last_used` | cualquiera de las anteriores **+** auto-citation en Stop hook | el MCP / Stop hook |

**Auto-citation en Stop hook** (la mecánica que te conviene saber): cuando termina tu turno, el hook escanea tu última respuesta. Si menciona un id de memoria como `[[<id>]]` (wikilink), `` `<id>` `` (inline code), o bare-word ≥8 chars que matchea un memory_id real, ese id recibe un `record_feedback(helpful=None)` automático — bump neutro, sin polarity. **Implicación**: cuando uses una memoria en tu respuesta, citá el id en una de las tres formas. El sistema lo registra solo. Sin esfuerzo manual de tu parte.

### Trigger explícito: `/mv feedback <id> [up|down]`

Cuando el user dice "che esta memoria me sirvió" / "no me sirvió" / "ignorá esa" — o vos identificás que una memoria fue clave para resolver la task — usá el tool:

```python
# 👍 — thumbs up
mcp__mem-vault__memory_feedback({id: "<id>", helpful: true})

# 👎 — thumbs down (no entierra, sólo neutraliza el boost)
mcp__mem-vault__memory_feedback({id: "<id>", helpful: false})

# Sin polarity — "used", bump last_used pero no toca helpful/unhelpful
mcp__mem-vault__memory_feedback({id: "<id>"})
```

Output al user (1 línea):

```
👍 Feedback registrado en `<id>` (helpful=N, unhelpful=M, ratio=+0.XX)
```

### Boost en search

`memory_search` multiplica el score de cada hit por `1 + usage_boost × max(0, helpful_ratio)` donde `helpful_ratio = (helpful − unhelpful) / max(1, helpful + unhelpful)`. **Clamp a 0**: thumbs-down no entierra (floor `1.0×`), sólo neutraliza el boost.

Defaults: `usage_boost=0.3` (memoria con `helpful_ratio=1` levanta su score 30%), `usage_boost_enabled=true`. Disable global via `MEM_VAULT_USAGE_BOOST_ENABLED=0`. Boost compone con cross-encoder rerank: cuando rerank está on, el boost multiplica el `rerank_score`, no el bi-encoder.

El response de `memory_search` ahora incluye `score_raw` (pre-boost) y `usage_boost` (factor) por hit — útiles si querés ver por qué una memoria salió primero.

## Boot briefing — auto-summary al cargar el skill

**Trigger**: la PRIMERA invocación del skill en una sesión nueva (`/mv`, `/memory` o `/mem_vault` sin args, o cualquier subcomando si todavía no se mostró briefing en esta sesión). Una vez por sesión — después no se repite.

**Por qué**: el agent (vos) entra a la task ciego sobre qué memorias existen. Mostrar un briefing al boot expone el corpus relevante al cwd actual y reduce re-descubrimientos.

**Cómo construirlo** (1-2 calls al MCP, ~250ms warm):

1. Derivá el `project_tag` actual del cwd (mismo algoritmo que en "Save flow → Project tag").
2. `mcp__mem-vault__memory_list({tags: [project_tag], limit: 50})` — todas las memorias del proyecto.
3. Localmente agregá:
   - **Total** del proyecto + total global del vault.
   - **Últimas 3** por `created` desc (title + id + type + edad relativa).
   - **Top tags** (top 5 tags por frecuencia, excluyendo el project_tag mismo).
   - **Lint flags**: cuántas tienen <3 tags, cuántas no tienen `## Aprendido el ...` en el body, cuántas con body <100 chars (probablemente skinny).

**Output del briefing** (compacto, ≤8 líneas):

```
📚 mem-vault · <N> memorias en `<project_tag>` (<M> total en el vault)
  Últimas 3:
    - `<id-1>` (<edad>) · <type>
    - `<id-2>` (<edad>) · <type>
    - `<id-3>` (<edad>) · <type>
  Top tags: <tag-1> (<n>) · <tag-2> (<n>) · <tag-3> (<n>) · <tag-4> (<n>) · <tag-5> (<n>)
  ⚠️ <X> con <3 tags · <Y> sin fecha de aprendizaje · <Z> body <100 chars
```

Si `N == 0` (proyecto sin memorias): mostrá una línea sola: `📚 mem-vault · 0 memorias en `<project_tag>`. Cuando guardes la primera, va a aparecer acá.`.

Si todas las cifras de lint son 0, omití la línea ⚠️ entera (no contaminar).

Después del briefing, ejecutá la acción que el user pidió (si pasó args). Si no pasó nada (`/mv` solo), el briefing sólo es suficiente — no listees también las últimas 10.

**Skip flag** opt-in: si el user setea env var `MEMVAULT_SKIP_BRIEFING=1`, no muestres el briefing. Útil para sesiones automáticas / scripts.

## Auto-context injection — search automático al task start

**Trigger**: la PRIMERA tool call de cada task del user. Si en esta task todavía no hubo `memory_search` (manual o automático) Y el último mensaje del user tiene >20 chars, ejecutá un search silencioso ANTES de actuar.

**Cómo**:

```python
mcp__mem-vault__memory_search({
  query:     <último mensaje del user, primeros 500 chars>,
  k:         5,
  threshold: 0.40,
})
```

**Qué hacer con los results**:

- Si hay matches ≥0.50 → **úsalos como contexto interno** (no los muestres al user, pero tomalos en cuenta para responder). Si el match contiene un fix/decision/preference que aplica directo a la task, mencionalo brevemente: `(Memoria relevante: `<id>` — <una línea>)`.
- Si todos los matches están <0.40 → no inyectes nada (señal de que la task es nueva, no hay precedente).

**No hacer ruido**: el auto-search no debe imprimir results crudos al output. Es contexto interno. Sólo el agent (vos) lo ve para mejorar la respuesta.

**Skip cuando**: el user pidió explícitamente algo trivial (1 comando, ej. "qué hora es", "git status"), o ya pasó <2min desde el último auto-search en la sesión.

## Save flow — auto-derivación de title/type/tags

`/mv save <content>` (con o sin `-e`) **NO pide flags al user**. La skill deriva `title`, `type` y `tags` del content + cwd antes de llamar al MCP. El user sólo pasa el body. Esto replica el comportamiento del agent cuando guarda proactivamente al cierre de una tarea (regla de auto-save del CLAUDE.md global).

### Title — auto

1. Si el content arranca con `# <título>\n...` → `title = <título>` (primera línea sin el `#`).
2. Si la primera línea no tiene `#` pero es <100 chars → usar esa línea como title.
3. Si no, dejar que el MCP use su default (primera línea ~80 chars).

### Type — auto (classifier por keywords, prioridad descendente)

Aplicá los matchers en este orden — el primero que pegue gana:

| Prioridad | Type | Regex (case-insensitive, `\b...\b`) |
|---|---|---|
| 1 | `bug` | `(bug\|fix\|broken\|crash\|leak\|regression\|rompi[óo]\|fall[óo]\|gotcha\|foot.?gun\|root cause\|causa raíz)` |
| 2 | `decision` | `(decid[íi]m?os?\|elegim?os?\|vamos con\|opted for\|chose\|decision\|tradeoff\|trade.?off\|arquitectur)` |
| 3 | `todo` | `(TODO\|pending\|pendiente\|por hacer\|hay que\|need to)` **Y** `len(content) < 500` |
| 4 | `preference` | `(prefiero\|me gusta\|always use\|never use\|preferencia)` |
| 5 | `feedback` | `(el user\|the user\|user)\s+(dijo\|said\|prefer\|wants?\|told\|asked\|report[óo]\|pidi[óo]\|quiere\|quiso)` |
| 6 | `fact` | content corto (<300 chars) Y forma "X es Y" / definición / dato puntual |
| 7 | `note` | default si no matchea nada |

Casos límite:
- Content largo (>2000 chars) con secciones `## Contexto / ## Problema / ## Solución` → casi siempre `bug` o `decision`. Re-aplicá el classifier sobre los headings + primer párrafo.
- Si dudás entre `bug` y `decision` (ej. "decidimos arreglar el bug X así") → ganá `bug`.

### Tags — auto (≥3 mandatory, convención CLAUDE.md global)

Construí el set en tres pasos. Cap final a **6 tags max** (más se vuelve ruido).

#### a) Project tag (1, mandatory)

**Primero chequeá el content por paths agent-config** (el cwd puede mentir si la memoria es sobre un skill global o un MCP, no sobre el proyecto donde estás trabajando):

| Content matchea | Tag — override del cwd |
|---|---|
| `~/.config/devin/skills` o `~/.devin/skills` | `devin-config` |
| `~/.claude/skills` o `~/.claude/agents` o `~/.claude/CLAUDE.md` | `claude-config` |
| `~/.config/devin/` o `.devin/config.json` (sin ser skill) | `devin-config` |
| MCP server externo (ej. mem-vault, github, slack mencionado como tool) | nombre del MCP server |

Si NO matchea nada de lo anterior, derivá del cwd (lo conocés del entorno de la sesión; si no estás seguro, ejecutá `pwd`):

| Path del cwd matchea | Tag |
|---|---|
| `obsidian-rag` | `obsidian-rag` |
| `whatsapp-listener` | `whatsapp-listener` |
| `mem-vault` | `mem-vault` |
| `rag-obsidian` | `obsidian-rag` (alias del nombre del repo en GH) |
| cualquier otra cosa | basename del cwd, lowercase, kebab-case |

#### b) Domain tags (1-3, content keyword extraction)

Buscá los keywords en el content (case-insensitive). Cada match agrega su tag al set:

| Keyword regex | Tag(s) |
|---|---|
| `(macos\|darwin\|/var/folders\|Path\.home\|\$HOME)` | `macos` |
| `(launchd\|launchctl\|plist\|LaunchAgents\|LaunchDaemons)` | `launchd` |
| `(fastembed\|Qdrant\|qdrant\|bge-m3)` | `qdrant` |
| `(fastembed)` | `fastembed` |
| `(fastapi\|uvicorn\|@app\.(get\|post)\|Starlette)` | `fastapi` |
| `(sqlite\|sqlite-vec)` | `sqlite` |
| `(pytest\|@pytest\|conftest)` | `tests` |
| `(mem0\|mem-vault)` | `memory-system` |
| `(LLM\|ollama\|Ollama\|bge-m3\|bge-reranker)` | `llm` |
| `(RAG\|retrieve\|rerank\|paraphrase\|hybrid search)` | `rag` |
| `(env.?var\|environment\|HF_HUB\|FASTEMBED_)` | `env-vars` |
| `(commit\|git push\|git rebase\|git pull)` | `git` |
| `(launchctl bootstrap\|launchctl bootout\|launchctl kickstart)` | `launchd` |
| `(SSE\|streaming\|/api/chat)` | `streaming` |
| `(CSS\|HTML\|Chart\.js\|dashboard)` | `frontend` |
| `(WhatsApp\|WA bridge)` | `whatsapp` |
| `(LoRA\|fine-?tune\|reranker)` | `fine-tune` |
| `(skill\|SKILL\.md\|\.devin/skills\|\.claude/skills)` | `devin-skills` |
| `(MCP\b\|MCP server\|mcp_call_tool\|mcp__)` | `mcp` |
| `(Devin\|Claude Code\|agent profile\|run_subagent)` | `agent-tooling` |
| `(frontmatter\|wikilink\|obsidian://\|Obsidian vault)` | `obsidian` |

Cap a **3 domain tags max**. Si matchean más, agarrá los 3 más específicos (preferí los que matchean más veces o son menos genéricos).

#### c) Technique tag (1, opcional pero ayuda al recall)

Aplicá UN matcher (el primero que pegue):

| Regex | Tag |
|---|---|
| `(p50\|p95\|p99\|ms\|latency\|perf\|benchmark)` | `performance` |
| `(gotcha\|foot.?gun)` | `gotcha` |
| `(refactor)` | `refactor` |
| `(root cause\|causa raíz\|fix\|patch\|rompi[óo]\|fall[óo])` | `bugfix` |
| `(setup\|configurar\|configuración\|env var)` | `setup` |
| `(eval gate\|baseline\|hit@5)` | `eval` |
| `(architectural\|arquitectur\|tradeoff)` | `architecture` |

#### Si después de a+b+c el set tiene <3 tags

Pedí al user el tag faltante en una línea, **antes** de llamar al MCP:

```
Derivé tags: [obsidian-rag, fastembed]. Pasame uno más para llegar al mínimo de 3:
```

NO guardes con menos de 3 tags — la convención del repo es estricta y guardar memorias undertagged degrada la recall futura.

### Override manual (escape hatch — uso poco frecuente)

Si el user explícita pasa flags inline, esos override la auto-derivación:

- `--type=bug` (o `type=bug`)
- `--tags=a,b,c` (o `tags=a,b,c`)
- `--title="..."` (o `title="..."`)

Sintaxis tolerante: aceptá con o sin doble dash, separador `=` o ` `. Si el user pasa flags, no corras la auto-derivación para esos campos (los demás sí).

### `-e` (LLM extractor + dedup)

Sigue siendo el flag para `auto_extract=true` del MCP. Cuando se pasa, Ollama canonicaliza el content + dedupea contra memorias existentes. Más lento (~3-8s) pero más smart. La auto-derivación local de title/type/tags **igual corre** y los valores van como hint al MCP.

### Cross-linking — auto-insertar `[[wikilinks]]` a memorias relacionadas

**Antes** de llamar al MCP, ejecutá un search prebuscado por el title derivado:

```python
mcp__mem-vault__memory_search({
  query:     <title derivado>,
  k:         5,
  threshold: 0.50,
})
```

Filtrá los results para que NO incluyan ningún memoria con `id == <slug que va a tener la nueva>` (caso edge: re-save del mismo título). Si quedan ≥1 results con score ≥0.50:

1. **Insertá una sección `## Memorias relacionadas`** al final del body, justo antes de `## Aprendido el ...` si existe (o al final si no):
   ```markdown
   ## Memorias relacionadas
   - [[<id-1>]] (<description recortada a 60 chars>)
   - [[<id-2>]] (<description recortada a 60 chars>)
   ```
2. Cap a **3 wikilinks max** (los top-3 por score).
3. Si el body ya tiene una sección `## Memorias relacionadas` (re-save / update), **mergeá**: agregá los nuevos `[[id]]` que no estén, mantené los previos. Sin duplicados.

**Por qué wikilinks y no IDs sueltos**: Obsidian renderiza `[[id]]` como links navegables y los muestra en graph view. El grafo de tu memoria emerge sin esfuerzo manual.

**Si NO hay related memorias ≥0.50**: no agregues la sección — el body queda como vino.

### `--contradict` (opt-in: detect contradictions vs existing memorias)

Si el user pasa `--contradict` (o `--contradict=true`), agregá `auto_contradict: true` al call. El MCP corre un pass del LLM local sobre las top-5 memorias semánticamente similares y stampea las IDs que contradigan en el campo `contradicts:` del frontmatter. Costo: ~3-5s extra (qwen2.5:3b).

**Cuándo dispararlo automático sin que el user lo pida**:

- El content arranca con `## Decisión revisada` / `## Cambio de criterio` / `## Cambio de mind` / `Updated:` — señales de que estás potencialmente reemplazando una decisión vieja.
- El type=`preference` o `decision` Y existe una memoria muy similar (`memory_search` k=3 threshold 0.65 returns ≥1 match) — high prior de que la nueva pisa la anterior.
- El user dice explícitamente "actualizá X" / "ya no es así" / "ahora prefiero Y en lugar de Z".

Sino, no lo dispares — el costo no compensa.

### `--project <name>` (override del project scope)

Por default, `project` se infiere del primer tag `project:X` o de `Config.project_default`. Si querés stampear scope explícito (ej. la memoria es del proyecto `obsidian-rag` aunque hayas escapado al cwd `mem-vault`), pasá `--project=obsidian-rag` y va al index metadata directo (más rápido para Qdrant filtering que el tag scan).

### Secret redaction — automático, default ON

Antes de que el body toque disco o el index, el MCP escanea por patterns de credentials (AWS/GitHub/OpenAI/Anthropic/Slack/Google/JWT/Bearer/PEM/`password=`/`token=`...) y los reemplaza con `[REDACTED:<kind>]`. La response incluye un campo `redactions: [{kind, count}]`. **Mostrá una warning al user** cuando hay matches:

```
⚠️ Redacted N credenciales del body antes de guardar:
   - openai_key: 1
   - bearer_token: 2
```

NO hace falta que vos hagas nada — es automático. Pero el warning es importante porque significa que el user pasó algo sensible y debería revisarlo.

Disable opt-out via `MEM_VAULT_REDACT_SECRETS=0` (no recomendado — el vault sync a iCloud/Dropbox/git en todo setup real).

### Llamada final al MCP

```python
mcp__mem-vault__memory_save({
  content:         <body literal con sección "Memorias relacionadas" insertada si aplica>,
  title:           <derivado o override>,
  type:            <derivado o override>,
  tags:            <derivados o override, ≥3>,
  project:         <opcional — explicit override; sino se infiere de tags/config>,
  auto_extract:    <true si -e, sino false>,
  auto_contradict: <true si --contradict, sino false>,
})
```

**Response shape** (relevant fields):

```json
{
  "ok": true,
  "memory":      {...},
  "indexed":     true,
  "related":     ["<id-1>", "<id-2>"],
  "contradicts": ["<id-3>"],
  "redactions":  [{"kind": "openai_key", "count": 1}]
}
```

Si `redactions` no es vacío → mostrá la warning. Si `contradicts` no es vacío → mostralos al user con descripción para que pueda revisar:

```
⚠️ Esta memoria contradice 1 existente:
   - [[<id-3>]] — <description>
   Si el cambio es intencional ignorá; si no, considerá `mv merge` o update.
```

## Verbos de descubrimiento

Verbos que extienden el básico CRUD para auditar / explorar el corpus. Todos toman ≤2 calls al MCP y agregan localmente.

### `/mv stats`

Snapshot del corpus completo. Útil para auditoría mensual / "¿cómo viene mi memoria?".

**Cómo construirlo**:

1. `mcp__mem-vault__memory_list({limit: 200})` — top 200 por recency.
2. Localmente agregá:
   - Total memorias
   - Counts por `type` (los 7 valores)
   - Top 10 tags por frecuencia
   - Distribución de edades (buckets: <7d / 7-30d / 30-90d / >90d)
   - Lint summary (cuántas tienen <3 tags / sin date / body skinny)

**Output**:

```
📊 mem-vault stats — <N> memorias totales

Por type:
  bug         <n>  (<%>)
  decision    <n>  (<%>)
  fact        <n>  (<%>)
  note        <n>  (<%>)
  feedback    <n>  (<%>)
  preference  <n>  (<%>)
  todo        <n>  (<%>)

Top 10 tags:
  obsidian-rag    <n>
  rag             <n>
  llm             <n>
  ...

Edades:
  últimos 7d:    <n>
  7-30d:         <n>
  30-90d:        <n>
  >90d:          <n>

Lint:
  <X> con <3 tags
  <Y> sin "## Aprendido el ..."
  <Z> body <100 chars
  → corré `/mv lint` para ver cuáles
```

### `/mv recent [n]`

Últimas `n` memorias por `created` (default `n=10`, max `50`).

```python
mcp__mem-vault__memory_list({limit: <n>})
```

Output: igual a `Output List` pero con edad relativa explícita ("hoy", "ayer", "hace 3d").

```
- `<id>` · <type> · <description (~80 chars)> · <edad>
```

### `/mv top <tag>`

Memorias con `<tag>`, sorted por recency.

```python
mcp__mem-vault__memory_list({tags: [<tag>], limit: 30})
```

Si `count: 0`, decí: `Sin memorias con tag «<tag>». ¿Querés ver todas con `/mv list` o auditar tags con `/mv stats`?`.

### `/mv duplicates`

Pairs con similarity ≥0.70 — candidatos a merge.

**Algoritmo**:

1. `mcp__mem-vault__memory_list({limit: 500})` — todo el corpus.
2. Para cada memoria, `mcp__mem-vault__memory_search({query: <title>, k: 3, threshold: 0.70})`.
3. Filtrá los results donde `result.id != memoria.id`. Reportá pairs únicos (no doble-cuenta `(A,B)` y `(B,A)`).
4. Si el corpus es grande (>200), warnings: "esto puede tardar ~30s".

**Output**:

```
🔁 <N> pairs candidatos a merge:

  similarity 0.83
    `<id-1>` (<description-1>)
    `<id-2>` (<description-2>)
    → `/mv merge <id-1> <id-2>` si querés combinar

  similarity 0.74
    ...
```

Si no hay pairs ≥0.70, mostrá: `🟢 Sin duplicados ≥0.70 — corpus limpio.`.

### `/mv timeline <project>`

Línea temporal de memorias con tag `<project>`, agrupadas por mes.

```python
mcp__mem-vault__memory_list({tags: [<project>], limit: 200})
```

**Output**:

```
📅 Timeline `<project>` — <N> memorias

2026-04 (<n> memorias)
  29 abr · `<id>` · <type> · <description (~70 chars)>
  29 abr · `<id>` · <type> · <description>
  28 abr · `<id>` · <type> · <description>

2026-03 (<n> memorias)
  ...
```

Cap a últimos 6 meses por default. Si el user pasa `since=YYYY-MM-DD` (o `--since=`), respetá ese cutoff.

### `/mv lint`

Reporta memorias con problemas de calidad. Útil después de un período de saves automáticos para detectar drift.

**Algoritmo**:

1. `mcp__mem-vault__memory_list({limit: 500})`.
2. Para cada memoria, evaluá:
   - **Sin tags**: `len(tags) == 0`
   - **Pocos tags**: `len(tags) < 3`
   - **Sin date**: body no contiene `## Aprendido el YYYY-MM-DD` (regex `## Aprendido el \d{4}-\d{2}-\d{2}`)
   - **Body skinny**: `len(body) < 100`
   - **Sin description**: `description == ""` o description es igual a las primeras 200 chars del body sin secciones (caso default-MCP, no description curada)
   - **Project tag missing**: ningún tag matchea un proyecto conocido (`obsidian-rag`, `whatsapp-listener`, `mem-vault`, `devin-config`, `claude-config`, etc.)

**Output** agrupado por problema:

```
🧹 Lint mem-vault — <N> issues en <M> memorias

Pocos tags (<3):
  - `<id-1>` — tags=[<a>]
  - `<id-2>` — tags=[<a>, <b>]

Sin date:
  - `<id-3>` — body no tiene "## Aprendido el ..."
  - ...

Body skinny (<100 chars):
  - `<id-4>` — len=<n>

Sin project tag:
  - `<id-5>` — tags=[<random>, <ad-hoc>]
```

Si todos OK: `🟢 Sin issues — corpus limpio.`.

### `/mv merge <id1> <id2>` (CON confirmación)

Combina 2 memorias en 1. Útil después de `/mv duplicates`.

**Flow**:

1. `mcp__mem-vault__memory_get({id: id1})` y `memory_get({id: id2})` — traé las dos.
2. Mostrá un preview del merge propuesto:
   - **Title**: el del más reciente.
   - **Type**: si los dos coinciden, ese; si difieren, el del más reciente (avisá del conflict).
   - **Tags**: union de los dos sets, sin duplicados, cap 6.
   - **Body**: concatenar — primero el del más reciente, después un separador `\n\n---\n\n## Merged from [[<id-older>]]\n\n` y luego el body del older. Quitá `## Aprendido el` duplicado si ambos lo tienen, dejando sólo el más reciente.
3. Pedí confirmación: `¿Mergeo `<id-newer>` ← `<id-older>`? (sí/no)`.
4. Si sí:
   - `memory_update({id: id-newer, content: <body-merged>, title: ..., tags: ...})`.
   - `memory_delete({id: id-older})`.
   - Avisá: `Mergeada en `<id-newer>`. `<id-older>` eliminada.`.
5. Si no, no hagas nada y avisá: `Cancelado, ambas siguen.`.

**Edge case**: si el body merged supera 5,000 chars, no merges en silencio — mostrá un warning y pedí confirmación adicional ("body resultante <N> chars — ¿confirmás?").

## Verbo: history — snapshots pre-update

`/mv history <id> [limit]` devuelve los snapshots tomados ANTES de cada `update` sobre la memoria. El MCP guarda un sidecar `<id>.history.jsonl` al lado del `.md` y appende una entry cada vez que el body / title / description / tags / related / contradicts cambian. Update no-op (mismo valor) NO genera entry.

```python
mcp__mem-vault__memory_history({id: "<id>", limit: 10})
```

**Output** (newest-first):

```
🕒 Historial de `<id>` — <N> snapshots

#1 · <ts-relativo>  (reason: update)
   tags: [a, b, c]
   description: <prev_desc>
   body (preview ~120 chars): "<prev body...>"

#2 · <ts-relativo>  (reason: update)
   ...
```

Si `count: 0`: `Sin historial — la memoria no fue modificada después del save inicial.`.

**Casos de uso**:

- "qué decía esta memoria hace 3 días" → `/mv history <id>` y leés.
- "necesito recuperar el body anterior" → tomás el snapshot, lo pasás como content al `update`.
- Auditoría de auto-link / auto-contradict: si `related` o `contradicts` cambió, ahí ves cuándo + por qué.

## Verbo: related — walk the graph

`/mv related <id>` devuelve los vecinos de la memoria en 4 buckets, en una sola llamada:

```python
mcp__mem-vault__memory_related({
  id: "<id>",
  min_shared_tags: 2,    # default — co-tag threshold
  k: 5,                   # default — semantic neighbors cap
  include_semantic: true, # default — set false para skip el LLM step
})
```

**Output**:

```
🕸️ Vecinos de `<id>`:

related (frontmatter):
  - `<id-1>` — <description>
  - `<id-2>` — <description>

contradicts (frontmatter):
  - `<id-3>` — <description>

co-tag (≥2 tags compartidos):
  - `<id-4>` — shared=[<tag-a>, <tag-b>]
  - `<id-5>` — shared=[<tag-a>, <tag-c>, <tag-d>]

semantic (top-5 search):
  - `<id-6>` — score 0.78
  - `<id-7>` — score 0.65
  - `<id-8>` — score 0.51
```

Cualquier bucket vacío se muestra como `(ninguno)` o se omite — pero al menos uno suele existir si la memoria está bien linkeada. Si los 4 están vacíos: `🟢 Memoria sin vecinos detectables. Quizás es un hub aislado o el corpus es pobre del topic.`.

**Cuándo usarlo**:

- Pre-merge: chequeás vecindario para entender qué se ve afectado si combinás dos.
- Pre-delete: ves si hay otras que dependen ([[link]]) antes de borrar.
- Exploration: el user pregunta "qué más sabés de X" → `/mv related <id de la principal X>` te da contexto lateral.

## Verbo: doctor — health check

`/mv doctor` shell-out al CLI `mem-vault doctor`. NO es un MCP tool — corré el binario:

```bash
mem-vault doctor [--skip-ollama] [--skip-index]
```

Capturá el output y mostralo al user limpio. El doctor corre 10+ checks: vault path, state dir, Ollama up, modelos pulleados, Qdrant collection, drift sync, fastembed, FASTEMBED_CACHE_PATH gotcha (macOS), feedback loop flags, hybrid retrieval flag, project scope. Exit 0 = todo green, 1 = warnings, 2 = errors críticos.

**Cuándo usarlo**: el user reporta "no me anda mem-vault" / "search devuelve vacío" / "qué tengo activado" → primer paso es siempre `/mv doctor`.

## Verbo: eval — retrieval regression baseline

`/mv eval --queries <file>` shell-out al CLI `mem-vault eval`. Toma un JSON file con queries labeled (`{query, expected, tag?}`) y reporta hit@1/3/5/10 + MRR sobre el corpus actual del user.

```bash
mem-vault eval --queries my_queries.json [--threshold 0.6] [--json]
```

Exit 0 si `hit@5 >= --threshold` (default 0), 1 si below. Útil para regression test cuando el user activa/desactiva flags (rerank, hybrid, decay) y quiere ver qué cambia.

**Cuándo usarlo**: el user dice "quiero ver si activar hybrid mejora el ranking" → corré dos `eval` runs (con y sin la flag) y compará. El output `--json` es para CI / diff automatizado.

## Delete flow (irreversible — pidan confirmación)

1. Primero llamá `mcp__mem-vault__memory_get({id})` y mostrale al user el contenido.
2. Pedí confirmación literal: `¿Borro `<id>`? (sí/no)`.
3. SOLO si responde explícitamente "sí"/"si"/"yes"/"y" → `mcp__mem-vault__memory_delete({id})`.
4. Si dice cualquier otra cosa, no borres y avisá: `Cancelado, `<id>` sigue ahí.`

## Output — español rioplatense, mimetizando convenciones del repo

### Search

Pipeline interno (no expongas al user — sólo entendé qué pasa para razonar):

1. **Dense bi-encoder** (Ollama bge-m3) → top raw_k.
2. (opt) **BM25 sparse** + **RRF fusion** si `hybrid_enabled` → mejora keyword/ID/path queries.
3. (opt) **Cross-encoder rerank** si `reranker_enabled` (`[hybrid]` extra) → reordena con un modelo que ve query+candidate juntos.
4. (opt) **Time-decay** si `decay_half_life_days > 0` → memorias viejas pierden peso.
5. **Usage boost** (default ON) → `score × (1 + 0.3 × max(0, helpful_ratio))`.
6. **Project filter** si `project_default` set → filtra a memorias con ese scope.
7. **Visibility filter** post-search (per-agent allowlist).
8. Top-k cut.

Tuneable via env vars (`MEM_VAULT_HYBRID`, `MEM_VAULT_RERANK`, `MEM_VAULT_DECAY_HALF_LIFE_DAYS`, `MEM_VAULT_USAGE_BOOST`, `MEM_VAULT_PROJECT`). Cuando el user pregunta "por qué salió esto primero", el response trae `score_raw` (pre-boost) y `usage_boost` (factor) por hit — usalos para explicar.

**Output** — top-5 (cantidad real si hay menos), uno por bloque:

```
**<title>** · `<id>` · score `0.XX` · type=<type>
<primer ~120 chars del body, terminando en ` …` si trunca>
```

Score: `≥0.70` = alto, `0.40-0.70` = medio, `<0.40` = bajo. Si todos los hits están bajo 0.40, agregá al final: `(matches débiles — quizás no tenés memoria sobre esto)`.

Si una memoria tiene boost activo (su `score_raw` < `score`), opcional: agregá `(boosted ×<factor>)` al lado del score. Útil cuando el user se sorprende del orden.

Si `count: 0`, decí: `No encontré nada para «<query>». ¿Querés guardarla con `/memory save <texto>`?`.

**Auto-citation hint**: si vas a referenciar un hit en tu respuesta posterior, citalo como `[[<id>]]` o `` `<id>` `` — el Stop hook lo detecta y bumpea el `last_used` automático. Sin esfuerzo manual.

### List

Una línea por memoria, sorted by recency (el MCP ya lo ordena):

```
- `<id>` · <type> · <description (~80 chars)>
```

### Save

Mostrá los valores derivados + memorias relacionadas (si hay):

```
Guardada como `<id>` · type=`<type>` · tags=[a, b, c]
<obsidian:// link al archivo>
```

Si pasaron `-e`, agregá:
```
(LLM extract + dedup activo — verificá que no haya pisado memoria existente)
```

**Post-save: memorias relacionadas**. Después del save, ejecutá `mcp__mem-vault__memory_search({query: <title>, k: 4, threshold: 0.40})` (k=4 para tener margen tras filtrar el self-match). **Filtrá obligatoriamente** los results donde `result.id == saved.id` — la búsqueda con el title del recién-guardado siempre devuelve la memoria misma como top-1 con score ~0.50; si no la filtrás, el output muestra "te dupliqué con vos mismo". Después del filtro, mostrá hasta 3 matches:

```
Memorias relacionadas (revisá si pisaste algo):
  - `<id-1>` · score 0.62 · <description>
  - `<id-2>` · score 0.51 · <description>
```

Si después del filtro no quedan matches ≥0.40, no muestres la sección (no contaminar el output).

### Get / Update

Para `get`: body completo, sin truncar. Para `update`: confirmá con `Actualizada `<id>` (`updated` bumped).`.

### Wikilinks a notas

Si el body de una memoria contiene rutas tipo `01-Projects/...md` o similares, renderizá como `obsidian://` link según la regla del CLAUDE.md global. URL-encode `/` como `%2F`, omití `.md` del param `file`.

## Auto-save triggers — cuándo guardar proactivamente sin que el user pida

El CLAUDE.md global dice "auto-save al cierre de tarea no-trivial". Esta sección hace explícitos los **triggers** que el agent (vos) debe evaluar antes de devolver control al user. Si alguno aplica → ejecutá `mcp__mem-vault__memory_save` con auto-derivación, sin pedir confirmación.

### Triggers que SÍ disparan save automático

| # | Trigger | Type sugerido | Ejemplo |
|---|---|---|---|
| 1 | Commit con `fix(...)` y root cause documentada en commit body (≥3 líneas explicativas) | `bug` | `e21b52e fix(env): pinear FASTEMBED_CACHE_PATH...` |
| 2 | Eval gate run con métricas concretas (hit@5, p50/p95, accuracy, etc.) | `fact` | "Eval gate validado: pre/post hit@5 idénticos 61.11%/84.00%" |
| 3 | Refactor con >50 LOC tocadas en 1 archivo Y mensaje del user incluye `por qué` | `decision` | "Refactoré X de A a B porque..." |
| 4 | Setup operativo nuevo (env var, plist, config persistente) | `note` técnica | `FASTEMBED_CACHE_PATH=$HOME/.cache/fastembed` |
| 5 | User pide explícitamente "guardá esto" / "anotá" / "recordá que" | `note` o `preference` según contenido | "che recordá que prefiero X sobre Y" |
| 6 | User da feedback explícito sobre cómo el agent debería trabajar | `feedback` | "siempre que toques tests corré pytest primero" |
| 7 | Descubrimiento del codebase con root cause no obvio (>10min de investigación) | `bug` o `note` | "el classifier de logs era time-blind por X" |
| 8 | Decisión arquitectónica con tradeoff documentado | `decision` | "Vamos con MMR embedding-based vs Jaccard porque..." |
| 9 | Performance finding con números reales de telemetría | `fact` | "p95 retrieve = 2.4s post-fix vs 3.0s pre" |
| 10 | Workflow operativo nuevo descubierto (gotcha + cómo resolverlo) | `bug` o `note` | "git apply --recount --3way para extraer hunks de peer agent" |

### Triggers que NO disparan save (skip silencioso)

- Cambios cosméticos (rename de variable, format, doc fixes triviales).
- Tareas exploratorias puras donde no se llegó a una conclusión accionable.
- Info ya documentada en `CLAUDE.md` / `AGENTS.md` / docstrings (mejor referenciar).
- Sesiones de <5min sin investigación profunda.
- "Resumen de lo que hicimos hoy" — eso es responsibility del `mem_session_summary` del MCP `engram`, no de `mem-vault` (engram tiene su propio scope para resúmenes de sesión, mem-vault es para insights individuales reusables).

### Cómo el agent debe ejecutar el auto-save

1. **Antes de devolver control al user**: pasar mentalmente la checklist de triggers.
2. Si aplica al menos uno: armá el body con formato CLAUDE.md (`## Contexto`, `## Problema`/`## Causa raíz`, `## Solución`, `## Cómo lo medí`, `## Aprendido el YYYY-MM-DD` con commit SHA si aplica).
3. Llamá al save flow con auto-derivación (sin override flags salvo que algo sea ambiguo).
4. Mostrá al user **una línea concisa** confirmando: `📚 Guardé memoria: \`<id>\` (<type>, <tags>).` — no spammees con el body entero.
5. Si el cross-linking encuentra related memorias, mostralas en otra línea: `Vincula a: [[<id-1>]], [[<id-2>]]`.

### Anti-spam: NO dispares auto-save si

- Ya hubo un auto-save en los últimos 10min de la misma sesión (a menos que sea de un dominio claramente distinto).
- El content total del save iba a ser <200 chars (memoria muy chica → probablemente no aporta).
- El user explícitamente dijo "no me guardes nada" / "skipeá memoria".

## Reglas duras

1. **NUNCA escribas el `.md` directo en el vault** — el MCP es el único path autorizado (mantiene frontmatter, index Qdrant, history).
2. **Si el MCP devuelve `ok: false`**, mostrá el `error` literal del MCP. NO inventes mensajes de error.
3. **Después del comando, frená** — el user invocó `/memory` para algo puntual. NO retomes tareas previas automáticamente.
4. **Si los argumentos son ambiguos** (ej. `/memory get` sin id), pedí el dato faltante en una línea: `Pasame el `<id>` (uno de los slugs que aparecen en `/memory list`).`.
5. **Save con <3 tags después de auto-derivación**: PEDÍ el tercer tag al user antes de llamar al MCP. No guardes undertagged.

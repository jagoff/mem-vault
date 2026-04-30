---
name: mv
description: "Search/save/list/update/delete memorias en mem-vault — el MCP local de memoria infinita backed by Obsidian (markdown plano + Qdrant + Ollama, 100% local). Triggers ALIAS-EQUIVALENTES: `/mem_vault`, `/memory` y `/mv` — los tres invocan EXACTAMENTE el mismo flow. Use cuando el user tipea `/mem_vault <query>` o `/memory <query>` o `/mv <query>` (search semántico — default si no hay verbo), `/mem_vault list` / `/memory list` / `/mv list` (últimas 20 by recency), `/mem_vault save <texto>` (guarda literal; agregar `-e` antes del texto para activar LLM extractor + dedup), `/mem_vault get <id>` (muestra memoria completa), `/mem_vault update <id> <texto>` (replace content), `/mem_vault delete <id>` (PIDE confirmación primero — irreversible). Sin argumentos → lista las últimas 10. Siempre routeá al MCP tool `mcp__mem-vault__memory_*` correspondiente, NUNCA escribas el .md a mano (el MCP maneja frontmatter + index Qdrant)."
argument-hint: "<query> | list | save [-e] <text> | get <id> | update <id> <text> | delete <id>"
---

# /mem_vault · /memory · /mv — router para mem-vault MCP

Los tres triggers (`/mem_vault`, `/memory`, `/mv`) son **alias-equivalentes**: invocan el mismo flow, parsean los mismos subcomandos, devuelven el mismo formato. El user puede usar el que más rápido le salga del dedo.

[mem-vault](https://github.com/jagoff/mem-vault) es un MCP local que da memoria persistente backed by Obsidian. Memorias = `.md` planas en el vault (típicamente `04-Archive/99-obsidian-system/99-AI/memory/<id>.md`), index semántico en Qdrant embedded (`~/.local/share/mem-vault/qdrant/`), embeddings via Ollama (bge-m3, 1024d). El user te invocó con `/mem_vault $ARGUMENTS` (o `/memory $ARGUMENTS` o `/mv $ARGUMENTS`) y vos ruteás al MCP tool que corresponde.

## Routing — parseá la PRIMERA palabra de `$ARGUMENTS`

| Primera palabra | Acción | MCP tool |
|---|---|---|
| (vacío) | listar últimas 10 | `mcp__mem-vault__memory_list({limit: 10})` |
| `list` | listar últimas 20 | `mcp__mem-vault__memory_list({limit: 20})` |
| `save` o `save -e` + texto | guardar (con o sin LLM extract) | `mcp__mem-vault__memory_save({content: <texto>, auto_extract: <true si -e, sino false>})` |
| `get` + id | mostrar memoria completa | `mcp__mem-vault__memory_get({id: <id>})` |
| `update` + id + texto | replace content | `mcp__mem-vault__memory_update({id: <id>, content: <texto>})` |
| `delete` + id | borrar (CON confirmación) | ver "Delete flow" abajo |
| cualquier otra cosa | search semántico | `mcp__mem-vault__memory_search({query: $ARGUMENTS, k: 5, threshold: 0.05})` |

## Delete flow (irreversible — pidan confirmación)

1. Primero llamá `mcp__mem-vault__memory_get({id})` y mostrale al user el contenido.
2. Pedí confirmación literal: `¿Borro `<id>`? (sí/no)`.
3. SOLO si responde explícitamente "sí"/"si"/"yes"/"y" → `mcp__mem-vault__memory_delete({id})`.
4. Si dice cualquier otra cosa, no borres y avisá: `Cancelado, `<id>` sigue ahí.`

## Output — español rioplatense, mimetizando convenciones del repo

### Search

Mostrá top-5 (cantidad real si hay menos), uno por bloque:

```
**<title>** · `<id>` · score `0.XX` · type=<type>
<primer ~120 chars del body, terminando en ` …` si trunca>
```

Score: `≥0.70` = alto, `0.40-0.70` = medio, `<0.40` = bajo. Si todos los hits están bajo 0.40, agregá al final: `(matches débiles — quizás no tenés memoria sobre esto)`.

Si `count: 0`, decí: `No encontré nada para «<query>». ¿Querés guardarla con `/memory save <texto>`?`.

### List

Una línea por memoria, sorted by recency (el MCP ya lo ordena):

```
- `<id>` · <type> · <description (~80 chars)>
```

### Save

`Guardada como `<id>`.` + link `obsidian://` al archivo. Si pasaron `-e`, agregá: `(LLM extract + dedup activo — verificá que no haya pisado memoria existente).`

### Get / Update

Para `get`: body completo, sin truncar. Para `update`: confirmá con `Actualizada `<id>` (`updated` bumped).`.

### Wikilinks a notas

Si el body de una memoria contiene rutas tipo `01-Projects/...md` o similares, renderizá como `obsidian://` link según la regla del CLAUDE.md global. URL-encode `/` como `%2F`, omití `.md` del param `file`.

## Reglas duras

1. **NUNCA escribas el `.md` directo en el vault** — el MCP es el único path autorizado (mantiene frontmatter, index Qdrant, history).
2. **Si el MCP devuelve `ok: false`**, mostrá el `error` literal del MCP. NO inventes mensajes de error.
3. **Después del comando, frená** — el user invocó `/memory` para algo puntual. NO retomes tareas previas automáticamente.
4. **Si los argumentos son ambiguos** (ej. `/memory get` sin id), pedí el dato faltante en una línea: `Pasame el `<id>` (uno de los slugs que aparecen en `/memory list`).`.

---
name: mv
description: "Search/save/list/update/delete memorias en mem-vault — el MCP local de memoria infinita backed by Obsidian (markdown plano + Qdrant + Ollama, 100% local). Triggers ALIAS-EQUIVALENTES: `/mem_vault`, `/memory` y `/mv` — los tres invocan EXACTAMENTE el mismo flow. Use cuando el user tipea `/mem_vault <query>` o `/memory <query>` o `/mv <query>` (search semántico — default si no hay verbo), `/mem_vault list` / `/memory list` / `/mv list` (últimas 20 by recency), `/mem_vault save <texto>` (guarda literal; agregar `-e` antes del texto para activar LLM extractor + dedup), `/mem_vault get <id>` (muestra memoria completa), `/mem_vault update <id> <texto>` (replace content), `/mem_vault delete <id>` (PIDE confirmación primero — irreversible). Sin argumentos → lista las últimas 10. Siempre routeá al MCP tool `mcp__mem-vault__memory_*` correspondiente, NUNCA escribas el .md a mano (el MCP maneja frontmatter + index Qdrant)."
argument-hint: "<query> | list | save [-e] <text> | get <id> | update <id> <text> | delete <id>"
---

# /mv · /mem_vault · /memory — router para mem-vault MCP

Los tres triggers (`/mv`, `/mem_vault`, `/memory`) son **alias-equivalentes**: invocan el mismo flow, parsean los mismos subcomandos, devuelven el mismo formato. El user puede usar el que más rápido le salga del dedo.

[mem-vault](https://github.com/jagoff/mem-vault) es un MCP local que da memoria persistente respaldada por un vault de Obsidian. Las memorias son archivos `.md` planos guardados bajo el subdirectorio configurado en el vault (por default `<vault>/mem-vault/<id>.md`, override con `MEM_VAULT_MEMORY_SUBDIR`); el índice semántico vive en Qdrant embebido (`~/.local/share/mem-vault/qdrant/` en Linux, `~/Library/Application Support/mem-vault/qdrant/` en macOS); los embeddings los genera Ollama localmente (`bge-m3`, 1024 dims). El user te invocó con `/mv $ARGUMENTS` (o `/mem_vault $ARGUMENTS` o `/memory $ARGUMENTS`) y vos ruteás al MCP tool que corresponde.

## Routing — parseá la PRIMERA palabra de `$ARGUMENTS`

| Primera palabra | Acción | MCP tool |
|---|---|---|
| (vacío) | listar últimas 10 + boot briefing si es la primera invocación de la sesión | ver "Boot briefing" abajo |
| `list` | listar últimas 20 | `mcp__mem-vault__memory_list({limit: 20})` |
| `save` o `save -e` + texto | guardar con auto-derivación de title/type/tags | ver "Save flow" abajo |
| `get` + id | mostrar memoria completa | `mcp__mem-vault__memory_get({id: <id>})` |
| `update` + id + texto | replace content | `mcp__mem-vault__memory_update({id: <id>, content: <texto>})` |
| `delete` + id | borrar (CON confirmación) | ver "Delete flow" abajo |
| `stats` | counts por type, top tags, edades | `mcp__mem-vault__memory_stats({cwd: <cwd>})` |
| `duplicates` | pairs ≥0.70 candidatos a merge | `mcp__mem-vault__memory_duplicates({cwd: <cwd>})` |
| `lint` | memorias con problemas (sin tags, sin date, body vacío, etc.) | `mcp__mem-vault__memory_lint({cwd: <cwd>})` |
| `synth` o `synthesize` + query | resumen LLM-compuesto de lo que el sistema sabe sobre el query | `mcp__mem-vault__memory_synthesize({query: <texto>})` |
| cualquier otra cosa | search semántico | `mcp__mem-vault__memory_search({query: $ARGUMENTS, k: 5, threshold: 0.1})` |

## Boot briefing — auto-summary al primer `/mv` de la sesión

**Trigger**: la PRIMERA invocación del skill en una sesión nueva. Una vez por sesión.

**Por qué**: el agente entra ciego al corpus disponible. Mostrar un briefing al boot expone las memorias relevantes al cwd y reduce re-descubrimientos.

**Cómo**: `mcp__mem-vault__memory_briefing({cwd: <cwd actual>})` devuelve `{project_tag, total_global, project_total, recent_3, top_tags, lint_summary}`. Renderizá en ≤8 líneas:

```
📚 mem-vault · <project_total> memorias en `<project_tag>` (<total_global> total)
  Últimas 3:
    - `<id-1>` (<edad>) · <type>
    - `<id-2>` (<edad>) · <type>
    - `<id-3>` (<edad>) · <type>
  Top tags: <tag-1> (<n>) · <tag-2> (<n>) · ...
  ⚠️ <X> con <3 tags · <Y> sin fecha de aprendizaje · <Z> body <100 chars
```

Si `project_total == 0`: una línea sola: `📚 mem-vault · 0 memorias en `<project_tag>`. Cuando guardes la primera, va a aparecer acá.`. Si todos los lint flags son 0, omití la línea ⚠️.

**Skip flag**: `MEMVAULT_SKIP_BRIEFING=1` deshabilita el briefing.

## Save flow — auto-derivación de title/type/tags

`/mv save <content>` (con o sin `-e`) **NO pide flags al user**. La skill llama a `mcp__mem-vault__memory_derive_metadata({content, cwd})` que devuelve `{title, type, tags, missing_tags}`. Si `missing_tags > 0`, **ANTES** de llamar a `memory_save`, pedí el tag faltante al user en una línea:

```
Derivé tags: [<tag-1>, <tag-2>]. Pasame uno más para llegar al mínimo de 3:
```

Si `missing_tags == 0`, llamá directamente a `memory_save` con los valores derivados. Auto-link viene activado por default — el `related:` y la sección `## Memorias relacionadas` con `[[id]]` wikilinks se agregan al body automáticamente.

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

Si `count: 0`, decí: `No encontré nada para «<query>». ¿Querés guardarla con `/mv save <texto>`?`.

### List

Una línea por memoria, sorted by recency (el MCP ya lo ordena):

```
- `<id>` · <type> · <description (~80 chars)>
```

### Save

`Guardada como `<id>`.` + link `obsidian://` al archivo si la ruta del vault está disponible. Si pasaron `-e`, agregá: `(LLM extract + dedup activo — verificá que no haya pisado memoria existente).`

### Get / Update

Para `get`: body completo, sin truncar. Para `update`: confirmá con `Actualizada `<id>` (`updated` bumped).`.

### Wikilinks a otras notas del vault

Si el body de una memoria menciona rutas a archivos `.md` del vault (típicamente paths relativos como `<carpeta>/<nota>.md`), podés renderizarlas como links `obsidian://open?vault=<NOMBRE>&file=<PATH>` para que el user las abra con un click. URL-encode `/` como `%2F`, omití el `.md` final del parámetro `file`. Solo aplicá esto cuando claramente sea una ruta de nota, no para ejemplos de código o comentarios.

## Reglas duras

1. **NUNCA escribas el `.md` directo en el vault** — el MCP es el único path autorizado (mantiene frontmatter, índice Qdrant, history).
2. **Si el MCP devuelve `ok: false`**, mostrá el `error` literal del MCP. NO inventes mensajes de error.
3. **Después del comando, frená** — el user invocó el slash command para algo puntual. NO retomes tareas previas automáticamente.
4. **Si los argumentos son ambiguos** (ej. `/mv get` sin id), pedí el dato faltante en una línea: `Pasame el `<id>` (uno de los slugs que aparecen en `/mv list`).`.

---

## Auto-capture proactivo (el game changer real)

mem-vault no se llena solo con lo que el user te pide guardar con `/mv save`. **El sistema crece cuando vos, como agente, capturás aprendizajes proactivamente al cierre de tareas no-triviales** — sin que el user te lo pida. Cada turno productivo deja huella, y la próxima vez que el user pregunta sobre el mismo tema, vos (u otro agente) tenés contexto inmediato.

**No te limites a lo que el user pidió.** Si en este turno hubo:

- ✅ **Bug fix con root cause no obvio** (más allá de un typo) → guardalo (`type: bug`).
- ✅ **Decisión de diseño con trade-offs** (qué elegiste, qué descartaste, por qué) → guardalo (`type: decision`).
- ✅ **Workflow operativo nuevo** (ej. comando git complejo, secuencia de pasos no triviales que descubriste) → guardalo (`type: fact`).
- ✅ **Convención del codebase** descubierta leyendo (ej. "en este repo `_postprocess_*` siempre devuelve dict con keys `ran`, `ok`, `full`, `ms`") → guardalo (`type: fact`).
- ✅ **Gotcha / foot-gun / invariante fácil de violar** → guardalo (`type: bug` o `fact`).
- ✅ **Performance finding empírico con números medidos** (no estimados) → guardalo (`type: fact`).
- ✅ **Setup operativo no trivial** (env vars, flags de OS, permissions, config previa) → guardalo (`type: fact`).
- ✅ **Preferencia explícita del user** (ej. "siempre escribime en español rioplatense") → guardalo (`type: preference`).
- ✅ **Feedback al sistema** (algo que no funcionó como el user esperaba, o cómo prefiere que lo hagas la próxima) → guardalo (`type: feedback`).

**NO guardes** cuando el turno fue:

- ❌ Tarea exploratoria pura (responder pregunta sin acción, leer logs sin acción).
- ❌ Cambios cosméticos (rename de variable, format, ajustes de copy a pedido literal del user).
- ❌ Información ya documentada en `CLAUDE.md`/`AGENTS.md`/docstrings del repo (mejor referenciar el archivo, no duplicar).
- ❌ El user pidió explícitamente "no guardes esto".

**Cuándo dispararlo**: ANTES de mandar tu respuesta final del turno, evaluá silenciosamente "¿hay algo acá que vale la pena guardar para mí-futuro?". Si la respuesta es sí, **llamá a `mcp__mem-vault__memory_save` ANTES de imprimir tu respuesta al user** y mencioná brevemente al final que guardaste (`Guardé esto como memoria: <id>`). Si la respuesta es no, no hagas nada.

**Formato del `content`** (markdown enriquecido, no wall-of-text):

```markdown
# <título descriptivo>

## Contexto
Qué proyecto, qué archivo, qué problema. 1-2 párrafos.

## Problema (o "## Causa raíz")
Qué fallaba y por qué. Repro si aplica.

## Solución
Bloques de código concretos (```python, ```bash, ```typescript), no solo
descripción. Copiá el snippet clave.

## Cómo lo medí (o "## Tests")
Números, comandos para verificar, test names.

## Cuándo aplicar este patrón
Bajo qué condiciones es relevante (no asumas que el yo-futuro va a recordar
el contexto entero).

## Aprendido el YYYY-MM-DD
Fecha + commit SHA si aplica + 1-2 líneas de qué disparó este aprendizaje.
```

**Args para `memory_save`** (no `auto_extract`, vos ya curaste el contenido):

```
{
  "content": "<markdown estructurado como arriba>",
  "title": "<frase descriptiva, no slug seco>",
  "type": "<decision | bug | fact | feedback | preference | note>",
  "tags": ["<proyecto>", "<dominio>", "<técnica>", "..."],   // mínimo 3
  "auto_extract": false
}
```

**Disable**: si el user dice "no guardes nada en mem-vault esta sesión" / "modo silencioso" / "off the record", respetá esa instrucción hasta que la sesión termine. NO anuncies que vas a parar — simplemente cumplí.

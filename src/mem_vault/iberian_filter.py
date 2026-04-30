"""PT/galego → ES post-filter for LLM responses (rioplatense Spanish target).

Why this exists
---------------
The local LLM models we route through (Ollama qwen2.5:* + command-r) are
multilingual and occasionally "leak" Portuguese (or Galician) words into
responses that are supposed to be Spanish — even when the system prompt
explicitly forbids it. Symptoms observed in mem-vault's
``memory_synthesize`` output during smoke tests on 2026-04-30:

>   "Fix do serviço RemoteMemVaultService está relacionado com a falta
>   de métodos em um dos servidores quando comparado ao outro. O
>   problema ocorreu porque..."

Every word above is unambiguously Portuguese. The system prompt asked
for *español rioplatense, máximo 6 párrafos* — the model partially
honored the format but slid into PT for vocabulary, likely because the
context (the synthesized memories) shared technical roots with PT
Stack Overflow material in the model's training corpus.

This filter is the **last barrier** after the prompt rule. The prompt is
necessary but not sufficient — the model still slips ~2-5% of the time.
We replace word-by-word with the rioplatense equivalent.

Provenance
----------
The replacement table is ported from
[`obsidian-rag`](https://github.com/jagoff/obsidian-rag)'s
``rag/iberian_leak_filter.py`` (commit ``582406f`` and the wave-5 update
on 2026-04-29). Both projects keep their own copy on purpose — the
obsidian-rag filter has additional streaming machinery that mem-vault's
``memory_synthesize`` doesn't need (synthesis returns the full LLM
response in one call, no SSE chunking). Keeping the lists in sync is a
manual operation.

Design
------
Conservative: only high-confidence pairs where the PT/GL word does NOT
exist (or is very rare) in rioplatense Spanish. Words shared between
both languages (``mesa``, ``casa``, ``vida``) are NEVER touched.

Idempotent: applying the filter twice produces the same output as once,
because every replacement targets a string that's already valid Spanish.

Order matters: multi-word phrases come FIRST (``em março`` → ``en
marzo``), otherwise the atomic ``março`` rule would fire and the
leftover ``em`` (Galician) would be missed.

Usage
-----
::

    from mem_vault.iberian_filter import replace_iberian_leaks
    cleaned = replace_iberian_leaks(llm_response)

The ``MemVaultService.synthesize`` path applies it transparently — see
``_call_llm_for_synthesis``.
"""

from __future__ import annotations

import re

# Replacement table — ordered: multi-word phrases FIRST.
#
# Keep in rough sync with ``rag/iberian_leak_filter.py`` in obsidian-rag.
# That file has more entries because it covers WhatsApp-context leaks
# (galician variants, casual conjugations) that mem-vault won't usually
# see. mem-vault's leaks are almost entirely from technical contexts
# (StackOverflow PT, etc.), so the high-frequency entries are the
# meaningful ones here.
_IBERIAN_LEAK_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    # ── Multi-word phrases (must come first) ──────────────────────
    (r"\buma\s+conversa\b", "una conversación"),
    (r"\buma\s+conversação\b", "una conversación"),
    (r"\bem\s+março\b", "en marzo"),
    (r"\bem\s+maio\b", "en mayo"),
    (r"\bem\s+junho\b", "en junio"),
    (r"\bem\s+julho\b", "en julio"),
    (r"\bem\s+setembro\b", "en septiembre"),
    (r"\bem\s+outubro\b", "en octubre"),
    (r"\bem\s+novembro\b", "en noviembre"),
    (r"\bem\s+dezembro\b", "en diciembre"),
    (r"\bem\s+fevereiro\b", "en febrero"),
    (r"\bcontigo\s+em\b", "contigo en"),
    (r"\bnos\s+braços\b", "en los brazos"),
    (r"\bno\s+braço\b", "en el brazo"),
    # PT possessives that don't exist in rioplatense — "tua experiência" etc.
    (r"\btua\s+", "tu "),
    (r"\bteu\s+", "tu "),
    (r"\btuas\s+", "tus "),
    (r"\bteus\s+", "tus "),
    # ── Months (PT spellings — all distinct from ES) ──────────────
    (r"\bmarço\b", "marzo"),
    (r"\bmaio\b", "mayo"),
    (r"\bjunho\b", "junio"),
    (r"\bjulho\b", "julio"),
    (r"\bsetembro\b", "septiembre"),
    (r"\boutubro\b", "octubre"),
    (r"\bnovembro\b", "noviembre"),
    (r"\bdezembro\b", "diciembre"),
    (r"\bfevereiro\b", "febrero"),
    # ── Time words (PT-only spellings) ────────────────────────────
    (r"\bhoje\b", "hoy"),
    (r"\bontem\b", "ayer"),
    (r"\bamanhã\b", "mañana"),
    # ── PT pronouns / negation ────────────────────────────────────
    (r"\bnão\b", "no"),
    (r"\bsim\b", "sí"),
    # ── PT quantity ───────────────────────────────────────────────
    (r"\bmuito\b", "mucho"),
    (r"\bmuita\b", "mucha"),
    (r"\bmuitos\b", "muchos"),
    (r"\bmuitas\b", "muchas"),
    # ── Politeness ────────────────────────────────────────────────
    (r"\bobrigado\b", "gracias"),
    (r"\bobrigada\b", "gracias"),
    # ── Common verbs (PT conjugations not in Spanish) ─────────────
    (r"\besqueças\b", "olvides"),
    (r"\besqueça\b", "olvide"),
    (r"\bdessas\b", "de esas"),
    (r"\bdesses\b", "de esos"),
    # "tem" (PT 3rd-person sing of "ter") — guard against tech acronyms
    # like "TEM-1" by requiring a lowercase letter after the space.
    (r"\btem\s+(?=[a-záéíóúñ])", "tiene "),
    # PT verb leaks observed in rioplatense-target output.
    (r"\bfalam\b", "hablan"),
    (r"\bfalou\b", "habló"),
    (r"\bfala\b(?!\s+un)", "habla"),
    (r"\bvistes\b", "viste"),
    (r"\bprimeira\b", "primera"),
    (r"\bprimeiro\b", "primero"),
    (r"\bprimeiramente\b", "primero"),
    (r"\buma\b", "una"),
    (r"\btambém\b", "también"),
    # ── PT family relations ───────────────────────────────────────
    (r"\bavô\b", "abuelo"),
    (r"\bavó\b", "abuela"),
    (r"\birmão\b", "hermano"),
    (r"\birmã\b", "hermana"),
    (r"\bfilha\b", "hija"),
    (r"\bfilho\b", "hijo"),
    (r"\bcriança\b", "niño"),
    (r"\bmãe\b", "mamá"),
    (r"\bpai\b", "papá"),
    # ── More PT vocabulary observed leaking ──────────────────────
    (r"\balguém\b", "alguien"),
    (r"\bninguém\b", "nadie"),
    (r"\btrouxe\b", "trajo"),
    (r"\btrouxeram\b", "trajeron"),
    (r"\bprimária\b", "primaria"),
    (r"\bsecundária\b", "secundaria"),
    (r"\brevolução\b", "revolución"),
    (r"\brevoluções\b", "revoluciones"),
    (r"\beducação\b", "educación"),
    (r"\binformação\b", "información"),
    (r"\binformações\b", "informaciones"),
    (r"\bemoção\b", "emoción"),
    (r"\bemoções\b", "emociones"),
    (r"\bcanção\b", "canción"),
    (r"\bcanções\b", "canciones"),
    (r"\bnação\b", "nación"),
    (r"\bnações\b", "naciones"),
    (r"\brelação\b", "relación"),
    (r"\brelações\b", "relaciones"),
    # General `-ução` / `-ções` / `-ção` PT suffixes → ES `-ución` / `-ciones` / `-ción`.
    # Catches "construção", "destruição", "atualização", etc.
    (r"\b(\w+?)ução\b", r"\1ución"),
    (r"\b(\w+?)uções\b", r"\1uciones"),
    (r"\b(\w+?)ção\b", r"\1ción"),
    (r"\b(\w+?)ções\b", r"\1ciones"),
    # ── PT contractions a+art (don't exist in Spanish) ────────────
    (r"\bà\b", "a la"),
    (r"\bàs\b", "a las"),
    (r"\bao\b", "al"),
    (r"\baos\b", "a los"),
    # PT possessives composed with "em".
    (r"\bem\s+sua\b", "en su"),
    (r"\bem\s+seu\b", "en su"),
    (r"\bem\s+suas\b", "en sus"),
    (r"\bem\s+seus\b", "en sus"),
    (r"\bem\s+um\b", "en un"),
    (r"\bem\s+uma\b", "en una"),
    # "até" (PT "hasta") — spelling with é-final is PT only.
    (r"\baté\b", "hasta"),
    # "novo" / "nova" / "novos" / "novas" → "nuevo" / etc.
    (r"\bnovo\b", "nuevo"),
    (r"\bnova\b", "nueva"),
    (r"\bnovos\b", "nuevos"),
    (r"\bnovas\b", "nuevas"),
    (r"\bchegada\b", "llegada"),
    (r"\bchegou\b", "llegó"),
    (r"\bchegar\b", "llegar"),
    # PT articles + lookahead to avoid breaking acronyms (OS X, etc.).
    (r"\bda\s+", "de la "),
    (r"\bdo\s+(?=[a-záéíóúñ])", "del "),
    (r"\bdas\s+", "de las "),
    (r"\bna\s+", "en la "),
    (r"\bnas\s+", "en las "),
    (r"\bos\s+(?=[a-záéíóúñ])", "los "),
    (r"\bas\s+(?=[a-záéíóúñ])", "las "),
    (r"\bum\s+(?=[a-záéíóúñ])", "un "),
    (r"\bfamília\b", "familia"),
    (r"\bfamílias\b", "familias"),
    (r"\bescola\b", "escuela"),
    (r"\bescolas\b", "escuelas"),
    # `-ência` / `-ância` PT suffixes — ê/â circumflex doesn't exist in ES.
    (r"\b(\w+?)ência\b", r"\1encia"),
    (r"\b(\w+?)ências\b", r"\1encias"),
    (r"\b(\w+?)ância\b", r"\1ancia"),
    (r"\b(\w+?)âncias\b", r"\1ancias"),
    # PT pronouns — "ela"/"ele" only when followed by a clear PT/ES verb form.
    (
        r"\bela\s+(é|es|era|foi|fue|estava|estaba|tem|tiene|teve|tuvo|disse|dijo)\b",
        r"ella \1",
    ),
    (
        r"\bele\s+(é|es|era|foi|fue|estava|estaba|tem|tiene|teve|tuvo|disse|dijo)\b",
        r"él \1",
    ),
    # "é" (PT verb ser 3rd sing) → "es". The ES verb "es" has no accent.
    (r"\bé\b", "es"),
    # "foi" (PT preterite of ser/ir) → "fue".
    (r"\bfoi\b", "fue"),
    # PT copulative "e" → ES "y". ES uses "e" only before words starting in
    # i/hi ("Pedro e Inés"). Constrain to a closed list of follow-up words
    # so we don't break legit ES "e".
    (
        r"\be\s+(estoy|estás|está|estamos|están|estaba|estaban|son|fue|fueron|"
        r"de|del|el|la|los|las|un|una|en|por|para|con|sin|"
        r"sus|tus|mis|nuestro|nuestra|"
        r"cuando|donde|porque|pero|aunque|"
        r"muy|más|menos|"
        r"yo|vos|él|ella|ellos|ellas)\b",
        r"y \1",
    ),
    # ── PT demonstratives — double-s doesn't exist in ES ──────────
    (r"\besse\b", "ese"),
    (r"\bessa\b", "esa"),
    (r"\besses\b", "esos"),
    (r"\bessas\b", "esas"),
    (r"\bisso\b", "eso"),
    (r"\bisto\b", "esto"),
    (r"\baquilo\b", "aquello"),
    # ── PT adverbs ────────────────────────────────────────────────
    (r"\bAqui\s+estão\b", "Acá están"),
    (r"\baqui\s+está\b", "acá está"),
    (r"\baqui\b", "acá"),
    (r"\bestão\b", "están"),
    (r"\bmelhor\b", "mejor"),
    (r"\bpior\b", "peor"),
    (r"\bajudar\b", "ayudar"),
    (r"\bajuda\b", "ayuda"),
    (r"\bvocê\b", "vos"),
    (r"\bvocês\b", "ustedes"),
    # `com` (PT "con") — guard against URLs containing ".com ".
    (r"(?<![./])\bcom\s+(?=[a-záéíóúñ])", "con "),
    # Common -ção words.
    (r"\bação\b", "acción"),
    (r"\bações\b", "acciones"),
    (r"\bsolução\b", "solución"),
    (r"\bsoluções\b", "soluciones"),
    (r"\bquestão\b", "cuestión"),
    (r"\bquestões\b", "cuestiones"),
    # ── 2026-04-30 wave: synthesize-leaks observed in mem-vault corpus ──
    # PT verb infinitives (-ar/-er/-ir mismatches with ES) and conjugated
    # forms not in any of the prior rules. Each pair requires the PT
    # form to NOT exist in rioplatense Spanish — verified manually.
    (r"\bfazer\b", "hacer"),
    (r"\bfeito\b", "hecho"),
    (r"\bfeita\b", "hecha"),
    (r"\bfeitos\b", "hechos"),
    (r"\bfeitas\b", "hechas"),
    (r"\bvou\s+fazer\b", "voy a hacer"),
    (r"\bvou\b", "voy"),
    (r"\btarefa\b", "tarea"),
    (r"\btarefas\b", "tareas"),
    (r"\bestavam\b", "estaban"),
    (r"\bestiveram\b", "estuvieron"),
    (r"\bestivessem\b", "estuvieran"),
    (r"\bestivesse\b", "estuviera"),
    # Double-ss in PT — never in rioplatense. Lowercase rule + IGNORECASE
    # also rewrites Necessários / NECESSÁRIOS.
    (r"\bnecessário\b", "necesario"),
    (r"\bnecessária\b", "necesaria"),
    (r"\bnecessários\b", "necesarios"),
    (r"\bnecessárias\b", "necesarias"),
    # PT acento agudo on stem vowel where ES uses different stress.
    (r"\busuário\b", "usuario"),
    (r"\busuária\b", "usuaria"),
    (r"\busuários\b", "usuarios"),
    (r"\busuárias\b", "usuarias"),
    # PT specific words / conjugations.
    (r"\bbootar\b", "bootear"),
    (r"\bfluxo\b", "flujo"),
    (r"\bfluxos\b", "flujos"),
    (r"\bmudança\b", "cambio"),
    (r"\bmudanças\b", "cambios"),
    (r"\bgarantindo\b", "garantizando"),
    (r"\bgarantir\b", "garantizar"),
    (r"\bsolicitou\b", "solicitó"),
    (r"\bEntendi\b", "Entendí"),
    (r"\bentendi\b", "entendí"),
    # "Portanto" (PT therefore) → "Por lo tanto"
    (r"\bportanto\b", "por lo tanto"),
    # PT clause starters that are clearly not Spanish.
    (r"\balém\s+de\b", "además de"),
    (r"\btambém\s+foram\b", "también fueron"),
    (r"\bforam\s+adicionados\b", "fueron agregados"),
    (r"\bforam\b", "fueron"),
)


_IBERIAN_LEAK_COMPILED: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), repl) for pat, repl in _IBERIAN_LEAK_REPLACEMENTS
)


def replace_iberian_leaks(text: str | None) -> str:
    """Apply all PT→ES regex pairs in declaration order.

    Safe on ``None`` / empty input. Mixed-case matches like ``Março`` are
    normalized to lowercase replacements (``marzo``) — acceptable because
    the leak is a model bug, not user-intent that needs preserving.

    **Idempotent**: applying twice equals applying once. Every pair
    rewrites toward valid Spanish, and Spanish never matches another PT
    rule, so a second pass is a fixed point.
    """
    if not text:
        return text or ""
    out = text
    for pat, repl in _IBERIAN_LEAK_COMPILED:
        out = pat.sub(repl, out)
    return out


__all__ = (
    "_IBERIAN_LEAK_COMPILED",
    "_IBERIAN_LEAK_REPLACEMENTS",
    "replace_iberian_leaks",
)

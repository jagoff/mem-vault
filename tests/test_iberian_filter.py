"""Tests for the PT/galego → ES post-filter applied to ``memory_synthesize``.

The replacement table itself is mostly correctness-by-construction (each pair
targets a string that doesn't exist or is rare in rioplatense Spanish), but
the test surface here checks the **invariants** that make the filter safe to
apply unconditionally:

1. **Idempotency** — applying twice = applying once. Cheap insurance against
   replacements that accidentally produce strings that re-match another rule.
2. **Pure-Spanish input is untouched** — feeding pristine rioplatense in
   produces the same string out, byte-for-byte. Catches over-eager regexes.
3. **Domain words observed in real leaks are normalized** — every entry on
   our 2026-04-30 leak corpus (the synthesize call from that day's smoke
   test) gets rewritten correctly.
4. **Edge cases** — empty / None input, mixed PT+ES, code blocks (the filter
   is currently text-only, so code stays as a flag for future work).

The point isn't to certify a "complete" PT detector — it's to make sure that
when the synthesizer drops a known leak, the user no longer sees it.
"""

from __future__ import annotations

import pytest

from mem_vault.iberian_filter import replace_iberian_leaks

# ---------------------------------------------------------------------------
# Invariant 1: idempotency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hola, todo bien",
        "Fix do serviço RemoteMemVaultService",  # the actual 2026-04-30 leak
        "uma conversa em março com você sobre essa nota",
        "ele tem un perro",
        "primeiramente, é importante saber",
    ],
)
def test_filter_is_idempotent(text):
    once = replace_iberian_leaks(text)
    twice = replace_iberian_leaks(once)
    assert once == twice, f"not idempotent: {once!r} → {twice!r}"


# ---------------------------------------------------------------------------
# Invariant 2: pure rioplatense Spanish passes through unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Clean rioplatense paragraphs — 0 leaks, must round-trip.
        "Vos podés revisar la memoria con `memory_get` cuando la necesites.",
        "Mirá, el sistema guarda cada decisión en el vault y después la sincroniza.",
        "Acá tenés el resumen: agarrá `mem-vault search`, fijate los resultados.",
        # Technical English mixed with Spanish — common in commit messages
        # and shouldn't trigger anything.
        "El endpoint `/api/v1/search` devuelve JSON con `ok: true`.",
        # Numbers, paths, acronyms — must survive intact.
        "Corré `mem-vault doctor` y revisá el output.",
        "VRAM = 16 GB. Modelos: qwen2.5:14b + bge-m3.",
    ],
)
def test_pure_spanish_passes_through(text):
    assert replace_iberian_leaks(text) == text


def test_filter_is_text_only_does_not_skip_code_blocks():
    """Documented limitation: the filter applies regex over the entire
    string. ``import os\\n...`` will partially rewrite because ``os ``
    looks like the PT article. ``synthesize`` consumers should NOT pass
    code through the filter — the LLM prompt already asks for prose, and
    code blocks in synthesized output would be model hallucination
    (also the wrong shape for a memory summary).

    This test pins the current behavior so anyone considering "let's run
    the filter on rendered markdown" knows the trade-off up front."""
    code = "import os\nfor x in range(10):\n    print(x)"
    out = replace_iberian_leaks(code)
    # ``os `` followed by lowercase ``f`` matches the PT-article rule.
    assert "los " in out, (
        "if this assert breaks, the filter became code-block aware — "
        "great, update this test and document the new behavior"
    )


# ---------------------------------------------------------------------------
# Invariant 3: known leak corpus (regression set)
# ---------------------------------------------------------------------------


def test_2026_04_30_synthesize_corpus():
    """The actual response we got from synthesize on 2026-04-30 must be
    fully rewritten. This is the corpus that motivated porting the filter.

    The filter is conservative — it won't catch *every* PT word ("serviço",
    "ocorreu", "comparado" stay because they're risky against legitimate
    ES use), but it should rewrite the obvious, high-confidence ones
    that show up in technical PT prose.
    """
    leaked = (
        "O problema ocorreu porque o MCP não conseguia iniciar em modo remoto. "
        "A solução foi adicionar e testar dois arquivos. "
        "Ele tem nove métodos faltantes em sua versão remota. "
        "Isso ocorreu porque os métodos foram adicionados em commits sucessivos."
    )
    cleaned = replace_iberian_leaks(leaked)
    # Spot-check the high-confidence pairs from this corpus.
    assert "não" not in cleaned
    assert "Isso" not in cleaned and "isso" not in cleaned
    assert "tem nove" not in cleaned  # "tem " (PT) → "tiene "
    assert "tiene nove" in cleaned
    assert "em sua" not in cleaned  # PT "em sua" → "en su"
    # "Ele tem" → "él tiene" via the ele+verb rule.
    assert "él tiene" in cleaned
    # The PT preterite "foi" → "fue".
    assert "fue" in cleaned and "foi" not in cleaned
    # "solução" → "solución" via the explicit rule.
    assert "solución" in cleaned and "solução" not in cleaned


def test_pt_months_normalize():
    assert replace_iberian_leaks("em março") == "en marzo"
    assert replace_iberian_leaks("em junho") == "en junio"
    assert replace_iberian_leaks("fevereiro de 2026") == "febrero de 2026"


def test_pt_pronouns_and_demonstratives():
    assert replace_iberian_leaks("você dijo") == "vos dijo"
    assert replace_iberian_leaks("vocês saben") == "ustedes saben"
    assert replace_iberian_leaks("essa nota") == "esa nota"
    assert replace_iberian_leaks("esses problemas") == "esos problemas"
    assert replace_iberian_leaks("isso es importante") == "eso es importante"


def test_pt_negation_and_affirmation():
    assert replace_iberian_leaks("não quiero") == "no quiero"
    assert replace_iberian_leaks("sim, está bien") == "sí, está bien"


def test_pt_courtesy_words():
    assert replace_iberian_leaks("obrigado por la ayuda") == "gracias por la ayuda"
    assert replace_iberian_leaks("obrigada") == "gracias"


def test_pt_suffixes_general_pattern():
    """The catch-all `-ção`/`-ução`/`-ência`/`-ância` rules cover words
    not enumerated explicitly.

    Note: the suffix rule is purely mechanical — ``-ução`` becomes
    ``-ución`` so ``construção`` → ``construción`` (single c). The
    legitimate Spanish ``construcción`` (with double-c) comes from
    Latin etymology, not from a regex over the suffix. The single-c
    output is "wrong-but-clearly-Spanish-and-readable" — better than
    leaving the PT word, worse than the manually-curated etymology.
    Words that matter (``revolución``, ``solución``, ``educación``,
    ``información``…) have explicit entries above the generic rule.
    """
    # Generic suffix rules (etymologically lossy).
    assert replace_iberian_leaks("construção") == "construción"
    assert replace_iberian_leaks("destruição") == "destruición"
    # ê/â suffixes are clean (no double-letter etymology in Spanish).
    assert replace_iberian_leaks("experiência") == "experiencia"
    assert replace_iberian_leaks("importância") == "importancia"
    # Explicitly-tabled words get the etymologically-correct form.
    assert replace_iberian_leaks("revolução") == "revolución"
    assert replace_iberian_leaks("solução") == "solución"


def test_pt_articles_lookahead_handles_pt_text():
    """The PT contraction rules use a lowercase lookahead to bound the
    match. With ``re.IGNORECASE`` set on every pattern (so ``Os arquivos``
    at sentence start gets rewritten too), the lookahead ALSO matches
    uppercase letters — meaning ``OS X`` becomes ``los X``. That's a
    known limitation worth keeping the trade-off explicit:

    - Pro: rewrites ``Os arquivos`` (PT at sentence start) →
      ``los arquivos``, which is the common case.
    - Con: rewrites ``OS X`` → ``los X``, breaking the Apple product
      name on the rare chance it shows up in a synthesis.

    For mem-vault's ``synthesize`` output (which is summary prose, not
    technical references), the trade-off favors aggressive PT rewriting.
    A consumer who needs to preserve technical acronyms should not run
    user-facing strings through the filter — same trade-off as
    ``test_filter_is_text_only_does_not_skip_code_blocks``.
    """
    # Common PT case (works as expected).
    assert replace_iberian_leaks("os arquivos") == "los arquivos"
    # PT at sentence start (also works).
    assert replace_iberian_leaks("Os arquivos") == "los arquivos"
    # Documented limitation — ``OS X`` matches because IGNORECASE relaxes
    # the lowercase-lookahead. If this assertion ever breaks (filter
    # gained context awareness), great — update this test.
    assert replace_iberian_leaks("OS X") == "los X"


def test_pt_verb_tem_does_not_break_acronyms():
    """``tem`` (PT 3rd-person ter) → "tiene", but tech acronyms like
    'TEM-1' must not be rewritten."""
    assert replace_iberian_leaks("TEM-1") == "TEM-1"
    assert replace_iberian_leaks("ele tem un perro") == "él tiene un perro"


def test_pt_com_does_not_break_dotcom_urls():
    """``com`` → "con", but ".com" inside a URL must survive."""
    assert "example.com" in replace_iberian_leaks("Visit example.com for more info")
    assert replace_iberian_leaks("com você") == "con vos"


# ---------------------------------------------------------------------------
# Invariant 4: edge cases
# ---------------------------------------------------------------------------


def test_empty_or_none_input():
    assert replace_iberian_leaks("") == ""
    assert replace_iberian_leaks(None) == ""


def test_mixed_pt_es_keeps_es_intact():
    """A response that's 90% Spanish + a few PT words should rewrite ONLY
    the PT, leaving everything else alone."""
    src = "Acá te dejo el resumen: a versión nova trae mejoras importantes."
    out = replace_iberian_leaks(src)
    assert "Acá te dejo el resumen:" in out
    assert "trae mejoras importantes." in out
    # The PT bits ("a versión nova" — "a" as preposition disambiguates;
    # "nova" is PT for "nueva").
    assert "nova" not in out


def test_replacement_table_compiles_at_import_time():
    """If a regex syntax error sneaks in, the import itself crashes —
    this test guards that the module is loadable. Asserting on the
    presence of the public symbol is sufficient."""
    from mem_vault import iberian_filter

    assert callable(iberian_filter.replace_iberian_leaks)
    assert len(iberian_filter._IBERIAN_LEAK_COMPILED) > 50

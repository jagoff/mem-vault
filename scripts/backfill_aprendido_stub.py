"""Backfill stub para memorias sin la sección ``## Aprendido el YYYY-MM-DD``.

La convención del repo (ver CLAUDE.md de obsidian-rag) es que toda memoria
no-trivial cierre con una línea ``## Aprendido el YYYY-MM-DD`` y, cuando
corresponda, el commit SHA. Sin esa marca no se puede bisectar memoria ↔
commit que la "resolvió". El briefing del MCP cuenta cuántas memorias
están en esta condición (``lint_summary.no_aprendido``) — al 2026-05-01
el contador del proyecto obsidian-rag estaba en 29/86 (34%).

Este script lee cada ``.md`` bajo el directorio de memorias, parsea el
frontmatter YAML, mira si falta la sección, y agrega un stub usando la
fecha ``created`` del frontmatter como aproximación. El SHA se deja como
TODO para que el agente / usuario lo complete en una sesión dedicada
(``rg "TODO: commit SHA" memory/`` lista los pendientes).

Uso:

    # Dry-run: lista candidatos, no escribe nada.
    python scripts/backfill_aprendido_stub.py

    # Aplica el stub en disco.
    python scripts/backfill_aprendido_stub.py --apply

    # Filtrar por tag (default: obsidian-rag).
    python scripts/backfill_aprendido_stub.py --tag obsidian-rag --apply

    # Sin filtro de tag (todas las memorias).
    python scripts/backfill_aprendido_stub.py --tag '' --apply
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

DEFAULT_VAULT = Path(
    "/Users/fer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes/"
    "04-Archive/99-obsidian-system/99-AI/memory"
)

APRENDIDO_RE = re.compile(r"^##\s+Aprendido\s+el\s+\d{4}-\d{2}-\d{2}", re.IGNORECASE | re.MULTILINE)
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def parse_memory(path: Path) -> tuple[dict[str, Any], str] | None:
    """Return (frontmatter_dict, body) or None if the file isn't a memoria."""
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    if not match:
        return None
    fm_raw, body = match.group(1), match.group(2)
    try:
        fm = yaml.safe_load(fm_raw) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(fm, dict):
        return None
    return fm, body


def needs_stub(body: str) -> bool:
    return not APRENDIDO_RE.search(body)


def created_to_date(fm: dict[str, Any]) -> str:
    """Best-effort YYYY-MM-DD from the ``created`` frontmatter field."""
    raw = fm.get("created") or fm.get("updated")
    if isinstance(raw, str):
        m = re.match(r"(\d{4}-\d{2}-\d{2})", raw)
        if m:
            return m.group(1)
    if isinstance(raw, date):
        return raw.isoformat()
    return date.today().isoformat()


def build_stub(when: str) -> str:
    return (
        f"\n## Aprendido el {when}\n\n"
        f"<!-- TODO: completar con commit SHA cuando esté disponible -->\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", default=str(DEFAULT_VAULT), help="Memory dir.")
    parser.add_argument(
        "--tag",
        default="obsidian-rag",
        help="Sólo memorias con este tag (vacío = todas).",
    )
    parser.add_argument(
        "--types",
        default="bug,decision",
        help=(
            "Lista CSV de types a backfillear. Default: bug,decision — "
            "que son las que típicamente tienen root cause + commit asociado. "
            "Vacío = todas (incluye preference/reference/note)."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="Escribir cambios al disco.")
    args = parser.parse_args()
    types_filter = {t.strip() for t in args.types.split(",") if t.strip()}

    vault = Path(args.vault)
    if not vault.is_dir():
        print(f"error: vault dir not found: {vault}", file=sys.stderr)
        return 2

    candidates: list[tuple[Path, str]] = []
    skipped_no_fm = 0
    skipped_tag = 0
    skipped_type = 0
    already_ok = 0

    for md in sorted(vault.glob("*.md")):
        parsed = parse_memory(md)
        if parsed is None:
            skipped_no_fm += 1
            continue
        fm, body = parsed
        if args.tag:
            tags = fm.get("tags") or []
            if not isinstance(tags, list) or args.tag not in tags:
                skipped_tag += 1
                continue
        if types_filter:
            mtype = fm.get("type")
            if mtype not in types_filter:
                skipped_type += 1
                continue
        if not needs_stub(body):
            already_ok += 1
            continue
        when = created_to_date(fm)
        candidates.append((md, when))

    print(f"vault:           {vault}")
    print(f"tag filter:      {args.tag or '(none)'}")
    print(f"types filter:    {sorted(types_filter) or '(none)'}")
    print(f"already_ok:      {already_ok}")
    print(f"skipped (no fm): {skipped_no_fm}")
    if args.tag:
        print(f"skipped (tag):   {skipped_tag}")
    if types_filter:
        print(f"skipped (type):  {skipped_type}")
    print(f"candidates:      {len(candidates)}")

    if not candidates:
        print("\nNothing to backfill.")
        return 0

    print("\nMemorias sin ## Aprendido el ...:")
    for md, when in candidates:
        print(f"  - {md.stem} (created={when})")

    if not args.apply:
        print("\n[dry-run] re-run with --apply to write the stubs.")
        return 0

    written = 0
    for md, when in candidates:
        text = md.read_text(encoding="utf-8")
        # Append the stub at the very end. Memorias suelen terminar con
        # un newline; nos aseguramos de no romper formato.
        if not text.endswith("\n"):
            text += "\n"
        text += build_stub(when)
        md.write_text(text, encoding="utf-8")
        written += 1
    print(f"\n[apply] wrote {written} stubs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

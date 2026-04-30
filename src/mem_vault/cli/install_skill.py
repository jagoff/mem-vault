"""``mem-vault install-skill`` — install the ``/mv`` slash command for Devin.

The ``mem-vault`` package ships a ``SKILL.md`` template at
``src/mem_vault/skills/SKILL.md`` that wires three slash-command aliases
(``/mv``, ``/mem_vault``, ``/memory``) to the MCP tools exposed by this
package. By default this subcommand installs all three aliases into
Devin's user-level skills directory:

- macOS / Linux: ``$XDG_CONFIG_HOME/devin/skills/<name>/SKILL.md``
  (defaults to ``~/.config/devin/skills/...``)
- Windows: ``%APPDATA%\\devin\\skills\\<name>\\SKILL.md``

Subsequent runs are idempotent: existing files are left alone unless
``--force`` is passed. ``--dry-run`` prints what would happen without
touching disk. Use ``--target`` to install elsewhere (e.g. into a
project-local ``.devin/skills`` dir).

The MCP server itself isn't configured by this command — that step lives
with the agent's own setup (``devin mcp add mem-vault -- mem-vault-mcp``,
or the equivalent ``claude mcp add ...``).
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import sys
from importlib.resources import files
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The three slash-command aliases we install. Order is intentional: ``mv`` is
# the primary (the source SKILL.md uses ``name: mv``), the other two are
# alias copies with the same body but a rewritten ``name:`` field so
# ``devin skills list`` shows them with the right label.
_DEFAULT_ALIASES: tuple[str, ...] = ("mv", "mem_vault", "memory")

# Frontmatter key to rewrite per alias copy.
_NAME_FRONTMATTER_RE = re.compile(r"^name:\s*\S+\s*$", flags=re.MULTILINE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "install-skill",
        help=(
            "Install the /mv (and /mem_vault, /memory) slash commands into "
            "Devin's skills directory. Idempotent — re-running is safe."
        ),
    )
    p.add_argument(
        "--target",
        type=Path,
        default=None,
        help=(
            "Install into this directory instead of the auto-detected one "
            "(useful for `.devin/skills` in a project repo)."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing skill files instead of skipping them.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the actions without writing anything to disk.",
    )
    p.add_argument(
        "--no-aliases",
        action="store_true",
        help="Install only /mv, skipping the /mem_vault and /memory aliases.",
    )
    p.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove the installed skill directories instead of installing.",
    )


def run(args: argparse.Namespace) -> int:
    target = args.target or _default_skills_dir()
    aliases: tuple[str, ...] = ("mv",) if args.no_aliases else _DEFAULT_ALIASES

    if args.uninstall:
        return _uninstall(target=target, aliases=aliases, dry_run=args.dry_run)

    return _install(
        target=target,
        aliases=aliases,
        force=args.force,
        dry_run=args.dry_run,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _default_skills_dir() -> Path:
    """Return the agent skills directory for the current platform.

    Mirrors what Devin for Terminal documents in ``creating-skills.mdx``:
    ``~/.config/devin/skills/`` on POSIX, ``%APPDATA%\\devin\\skills\\``
    on Windows. We honor ``$XDG_CONFIG_HOME`` when set so the function
    plays nicely with non-standard XDG layouts.
    """
    if platform.system() == "Windows":
        roaming = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(roaming) / "devin" / "skills"
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(xdg) / "devin" / "skills"


def _read_template() -> str:
    """Locate the bundled SKILL.md and return its contents.

    Prefers ``importlib.resources`` (works in wheel installs). Falls back
    to the filesystem when running from an editable install where the
    resource lookup may not surface — e.g. some uv editable layouts.
    """
    try:
        return files("mem_vault").joinpath("skills/SKILL.md").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        # Editable install fallback: resolve relative to this file's parent.
        # ``mem_vault/cli/install_skill.py`` → ``mem_vault/skills/SKILL.md``
        fallback = Path(__file__).resolve().parent.parent / "skills" / "SKILL.md"
        if fallback.exists():
            return fallback.read_text(encoding="utf-8")
        raise


def _rewrite_name(template: str, alias: str) -> str:
    """Return ``template`` with the ``name:`` frontmatter line replaced.

    The bundled template uses ``name: mv`` as the canonical value. When we
    copy it to ``mem_vault/`` or ``memory/``, we want the frontmatter to
    match the directory so ``devin skills list`` shows a coherent label.
    """
    return _NAME_FRONTMATTER_RE.sub(f"name: {alias}", template, count=1)


def _install(
    *,
    target: Path,
    aliases: tuple[str, ...],
    force: bool,
    dry_run: bool,
) -> int:
    template = _read_template()

    if dry_run:
        print(f"would install into: {target}")
    else:
        target.mkdir(parents=True, exist_ok=True)
        print(f"installing into: {target}")

    installed = 0
    skipped = 0
    for alias in aliases:
        dest_dir = target / alias
        dest_file = dest_dir / "SKILL.md"

        if dest_file.exists() and not force:
            print(f"  skip (exists): /{alias} → {dest_file}")
            skipped += 1
            continue

        if dry_run:
            print(f"  would write:   /{alias} → {dest_file}")
            installed += 1
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file.write_text(_rewrite_name(template, alias), encoding="utf-8")
        print(f"  installed:     /{alias} → {dest_file}")
        installed += 1

    if dry_run:
        print(f"\n(dry-run) would install {installed} skill(s), skip {skipped}.")
        return 0

    print(
        f"\n{installed} skill(s) installed, {skipped} skipped. "
        f"Open a new Devin session and try `/mv` to verify."
    )
    if skipped and not force:
        print("(re-run with --force to overwrite the skipped entries.)")

    # Quick reminder about the MCP side. The skill routes to MCP tools, so
    # without the MCP server registered the slash command will fail at the
    # first tool invocation.
    print(
        "\nReminder: this only installs the slash-command skill. The MCP "
        "server itself must be registered with your agent runner. For Devin:\n"
        "  devin mcp add mem-vault -- mem-vault-mcp"
    )
    return 0


def _uninstall(
    *,
    target: Path,
    aliases: tuple[str, ...],
    dry_run: bool,
) -> int:
    """Delete only the SKILL.md files we created plus their (now-empty) dirs.

    We never touch directories we didn't create — if the user dropped extra
    files inside ``<target>/<alias>/`` we leave the dir alone.
    """
    removed = 0
    missing = 0
    for alias in aliases:
        dest_dir = target / alias
        dest_file = dest_dir / "SKILL.md"

        if not dest_file.exists():
            print(f"  not present:   /{alias} ({dest_file})")
            missing += 1
            continue

        if dry_run:
            print(f"  would remove:  /{alias} ({dest_file})")
            removed += 1
            continue

        dest_file.unlink()
        # Try removing the now-empty directory; preserve it if other files exist.
        try:
            dest_dir.rmdir()
            print(f"  removed:       /{alias} ({dest_dir})")
        except OSError:
            print(f"  removed file:  /{alias} (kept dir, other files present)")
        removed += 1

    if dry_run:
        print(f"\n(dry-run) would remove {removed} skill(s), {missing} missing.")
        return 0
    print(f"\n{removed} skill(s) removed, {missing} missing.")
    return 0


# Keep the module callable directly for quick smoke tests:
# ``python -m mem_vault.cli.install_skill --dry-run``
if __name__ == "__main__":  # pragma: no cover — smoke entrypoint only
    parser = argparse.ArgumentParser(prog="install_skill")
    sub = parser.add_subparsers(dest="cmd")
    add_subparser(sub)
    parsed = parser.parse_args(["install-skill", *sys.argv[1:]])
    sys.exit(run(parsed))

"""Tests for ``mem-vault install-skill`` — copies SKILL.md aliases to disk.

We exercise the install / uninstall flow against a temp directory rather
than the user's real ``~/.config/devin/skills/``. The bundled template
is read via ``importlib.resources``; the editable-install fallback path
is also unit-tested.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mem_vault.cli import install_skill


def _make_args(**kwargs) -> argparse.Namespace:
    """Build a Namespace with the fields the CLI parser would produce."""
    defaults = {
        "target": None,
        "force": False,
        "dry_run": False,
        "no_aliases": False,
        "uninstall": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Default skills dir per platform
# ---------------------------------------------------------------------------


def test_default_skills_dir_posix(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/custom/xdg")
    assert install_skill._default_skills_dir() == Path("/custom/xdg/devin/skills")


def test_default_skills_dir_posix_no_xdg(monkeypatch, tmp_path):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert install_skill._default_skills_dir() == tmp_path / ".config" / "devin" / "skills"


def test_default_skills_dir_windows(monkeypatch, tmp_path):
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert install_skill._default_skills_dir() == tmp_path / "Roaming" / "devin" / "skills"


def test_default_skills_dir_windows_no_appdata(monkeypatch, tmp_path):
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    expected = tmp_path / "AppData" / "Roaming" / "devin" / "skills"
    assert install_skill._default_skills_dir() == expected


# ---------------------------------------------------------------------------
# _read_template — both wheel + editable-install paths
# ---------------------------------------------------------------------------


def test_read_template_returns_real_skill_md():
    content = install_skill._read_template()
    assert "name: mv" in content
    assert "memory_save" in content
    assert "Search/save/list/update/delete" in content


def test_skill_template_contains_auto_capture_directive():
    """The bundled SKILL.md must instruct the agent to capture proactively.

    This is the single most important user-facing contract of the skill —
    without it, the system stays manual-save-only and never grows by
    itself. If we ship a SKILL.md that lacks the auto-capture section,
    the whole "memoria que crece sola" promise breaks.
    """
    content = install_skill._read_template()
    assert "Auto-capture proactivo" in content
    # Must mention all the trigger categories so the agent has criteria.
    assert "Bug fix" in content
    assert "Decisión de diseño" in content
    assert "Convención del codebase" in content
    assert "Gotcha" in content
    # Must have an opt-out clause.
    assert "no guardes nada" in content.lower() or "off the record" in content.lower()


def test_read_template_falls_back_to_filesystem(monkeypatch):
    """When ``importlib.resources`` raises, the FS fallback should kick in."""

    def _broken_files(*args, **kwargs):
        raise ModuleNotFoundError("simulated wheel layout breakage")

    monkeypatch.setattr("mem_vault.cli.install_skill.files", _broken_files)
    content = install_skill._read_template()
    # Same content as before — proves the fallback succeeded.
    assert "name: mv" in content


# ---------------------------------------------------------------------------
# _rewrite_name — frontmatter editing
# ---------------------------------------------------------------------------


def test_rewrite_name_replaces_first_match_only():
    template = "---\nname: mv\nfoo: bar\n---\n# body talks about name: ignored\n"
    assert install_skill._rewrite_name(template, "memory") == (
        "---\nname: memory\nfoo: bar\n---\n# body talks about name: ignored\n"
    )


def test_rewrite_name_handles_extra_whitespace():
    template = "---\nname:    weird   \nfoo: bar\n---\n"
    out = install_skill._rewrite_name(template, "mv")
    assert "name: mv" in out
    assert "name:    weird" not in out


# ---------------------------------------------------------------------------
# Install flow
# ---------------------------------------------------------------------------


def test_install_writes_three_aliases(tmp_path, capsys):
    args = _make_args(target=tmp_path)
    rc = install_skill.run(args)
    assert rc == 0

    for alias in ("mv", "mem_vault", "memory"):
        skill_md = tmp_path / alias / "SKILL.md"
        assert skill_md.exists(), f"Missing alias {alias}"
        head = skill_md.read_text(encoding="utf-8").splitlines()[1]
        assert head == f"name: {alias}", f"Wrong frontmatter in {alias}: {head!r}"

    out = capsys.readouterr().out
    assert "3 skill(s) installed" in out


def test_install_no_aliases_writes_only_mv(tmp_path):
    args = _make_args(target=tmp_path, no_aliases=True)
    install_skill.run(args)
    assert (tmp_path / "mv" / "SKILL.md").exists()
    assert not (tmp_path / "mem_vault").exists()
    assert not (tmp_path / "memory").exists()


def test_install_skips_existing_without_force(tmp_path, capsys):
    install_skill.run(_make_args(target=tmp_path))
    capsys.readouterr()  # discard first run

    install_skill.run(_make_args(target=tmp_path))
    out = capsys.readouterr().out
    assert "skip (exists)" in out
    assert "0 skill(s) installed, 3 skipped" in out


def test_install_force_overwrites_existing(tmp_path, capsys):
    args = _make_args(target=tmp_path)
    install_skill.run(args)

    # Tamper with one of the installed files. The next --force run must
    # restore it from the bundled template.
    tampered = tmp_path / "mv" / "SKILL.md"
    tampered.write_text("totally bogus content", encoding="utf-8")

    install_skill.run(_make_args(target=tmp_path, force=True))
    restored = tampered.read_text(encoding="utf-8")
    assert "name: mv" in restored
    assert "memory_save" in restored


def test_install_dry_run_does_not_write_anything(tmp_path, capsys):
    args = _make_args(target=tmp_path, dry_run=True)
    rc = install_skill.run(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "(dry-run)" in out

    # Critically: no files were created.
    assert list(tmp_path.iterdir()) == []


def test_install_creates_target_directory_if_missing(tmp_path):
    nested = tmp_path / "nope" / "yeah" / "skills"
    install_skill.run(_make_args(target=nested))
    assert (nested / "mv" / "SKILL.md").exists()


# ---------------------------------------------------------------------------
# Uninstall flow
# ---------------------------------------------------------------------------


def test_uninstall_removes_the_three_directories(tmp_path):
    install_skill.run(_make_args(target=tmp_path))
    install_skill.run(_make_args(target=tmp_path, uninstall=True))

    for alias in ("mv", "mem_vault", "memory"):
        assert not (tmp_path / alias / "SKILL.md").exists()
        # Empty dir should also be cleaned up.
        assert not (tmp_path / alias).exists()


def test_uninstall_dry_run_keeps_files(tmp_path):
    install_skill.run(_make_args(target=tmp_path))
    install_skill.run(_make_args(target=tmp_path, uninstall=True, dry_run=True))

    for alias in ("mv", "mem_vault", "memory"):
        assert (tmp_path / alias / "SKILL.md").exists()


def test_uninstall_when_nothing_installed_is_noop(tmp_path, capsys):
    rc = install_skill.run(_make_args(target=tmp_path, uninstall=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "3 missing" in out


def test_uninstall_preserves_dir_with_extra_files(tmp_path):
    install_skill.run(_make_args(target=tmp_path))
    # User dropped an extra config file inside the alias dir — we must not nuke it.
    extra = tmp_path / "mv" / "user-extra.md"
    extra.write_text("user content", encoding="utf-8")

    install_skill.run(_make_args(target=tmp_path, uninstall=True))
    # SKILL.md gone, user file kept, dir kept
    assert not (tmp_path / "mv" / "SKILL.md").exists()
    assert extra.exists()
    assert (tmp_path / "mv").is_dir()


# ---------------------------------------------------------------------------
# CLI integration — argparse wiring
# ---------------------------------------------------------------------------


def test_parser_registers_install_skill_subcommand():
    """Build the top-level parser and confirm install-skill is wired in."""
    from mem_vault.cli import _build_parser

    parser = _build_parser()
    # The cleanest way to assert without driving argparse: render the help
    # text and look for our subcommand.
    help_text = parser.format_help()
    assert "install-skill" in help_text

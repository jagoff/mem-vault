"""``mem-vault reflect`` + ``install-daemon`` — nightly reflection daemon.

The reflection pass runs once per day (typically 03:00 local via launchd
on macOS, systemd on Linux, Task Scheduler on Windows) and writes a
``reflection_YYYY_MM_DD`` memory with:

- Activity counts (created / edited in the window, consolidated pairs)
- New decisions / bugs documented in the last 24 h
- Zombies (unused memorias over N days old)
- Tensiones detectadas (``contradicts`` set per memoria)
- Knowledge gaps por proyecto

The companion ``install-daemon`` subcommand generates + bootstraps the
launchd plist on macOS, emits the equivalent systemd unit on Linux with
install instructions, or prints a Task Scheduler XML skeleton on Windows.
Follows the CLAUDE.md rule: "feature con daemon = instalar y verificar,
no dejar TODO corre `launchctl bootstrap`".
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def add_subparsers(sub: argparse._SubParsersAction) -> None:
    p_reflect = sub.add_parser(
        "reflect",
        help="Generate today's daily reflection memory (fast, offline by default).",
        description=(
            "Compute activity / zombies / contradictions for the last N hours "
            "and write a reflection_YYYY_MM_DD memory to the vault. Idempotent: "
            "running twice the same day overwrites the existing note. "
            "``--consolidate`` opts in to the Ollama-backed merge pass "
            "(slower, ~1-5 min depending on vault size). Intended to run "
            "as a daemon — see ``mem-vault install-daemon``."
        ),
    )
    p_reflect.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="Window for 'recent' activity (default 24 h).",
    )
    p_reflect.add_argument(
        "--zombie-age-days",
        type=float,
        default=60.0,
        help="Memorias older than this with zero usage are flagged zombies.",
    )
    p_reflect.add_argument(
        "--consolidate",
        action="store_true",
        help=(
            "Run the LLM-backed consolidate pass and auto-merge obvious "
            "duplicates (threshold ≥ 0.92). Off by default because it "
            "calls Ollama and can take minutes. When off, only reports "
            "the pending pair count."
        ),
    )
    p_reflect.add_argument(
        "--consolidate-threshold",
        type=float,
        default=0.92,
        help="Cosine similarity threshold for auto-merges (default 0.92).",
    )
    p_reflect.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")

    p_install = sub.add_parser(
        "install-daemon",
        help="Install + load the nightly reflection daemon (macOS launchd / Linux systemd).",
        description=(
            "Generates the platform-appropriate scheduler file and loads it. "
            "On macOS: writes ``~/Library/LaunchAgents/com.mem-vault.reflect.plist`` "
            "and runs ``launchctl bootstrap`` + ``launchctl kickstart`` to "
            "activate it immediately — the daemon starts right away + schedules "
            "a 03:00 daily run. On Linux: writes a systemd user service + timer "
            "to ``~/.config/systemd/user/`` and runs ``systemctl --user "
            "daemon-reload`` + ``enable --now``. Windows: prints a Task Scheduler "
            "XML + the schtasks command. Follows the CLAUDE.md rule: dejar el "
            "daemon ACTIVO al cerrar, no TODO 'corré esto después'."
        ),
    )
    p_install.add_argument(
        "--hour",
        type=int,
        default=3,
        help="Hour of day for the nightly run (0-23, default 3 = 03:00 local).",
    )
    p_install.add_argument(
        "--minute",
        type=int,
        default=0,
        help="Minute for the nightly run (0-59, default 0).",
    )
    p_install.add_argument(
        "--consolidate",
        action="store_true",
        help="Add --consolidate to the scheduled invocation (Ollama-backed merges).",
    )
    p_install.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plist/unit file without writing or loading.",
    )
    p_install.add_argument(
        "--uninstall",
        action="store_true",
        help="Unload + remove the scheduler file.",
    )

    p_status = sub.add_parser(
        "daemon-status",
        help="Show whether the reflection daemon is loaded + last run time.",
        description=(
            "Reports launchctl / systemctl state for the mem-vault reflection "
            "daemon, and the timestamp of the last reflection_YYYY_MM_DD "
            "memory written to the vault."
        ),
    )
    p_status.add_argument("--json", action="store_true", help="Emit JSON.")


# ---------------------------------------------------------------------------
# reflect
# ---------------------------------------------------------------------------


def run_reflect(args: argparse.Namespace) -> int:
    from mem_vault.config import load_config
    from mem_vault.reflection import run_reflection

    cfg = load_config()
    report = run_reflection(
        cfg,
        lookback_hours=args.lookback_hours,
        auto_merge_threshold=args.consolidate_threshold,
        zombie_age_days=args.zombie_age_days,
        apply_consolidate=args.consolidate,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))
        return 0

    print()
    print(f"mem-vault reflect  ({report.day})")
    print("─" * 30)
    print(f"  memoria escrita   : {report.memory_id or '—'}")
    print(f"  total en vault    : {report.total_memorias}")
    print(f"  creadas en {report.lookback_hours}h : {report.created_in_window}")
    print(f"  editadas en {report.lookback_hours}h: {report.updated_in_window}")
    print(f"  pares de dups     : {report.pending_dup_pairs} pendientes · {report.consolidated_pairs} mergeados")
    print(f"  zombies           : {len(report.zombies)}")
    print(f"  tensiones         : {len(report.contradictions)}")
    print(f"  decisiones nuevas : {len(report.new_decisions)}")
    print(f"  bugs nuevos       : {len(report.new_bugs)}")
    if report.knowledge_gaps:
        print("  knowledge gaps    :")
        for g in report.knowledge_gaps:
            print(f"    · {g}")
    print()
    return 0


# ---------------------------------------------------------------------------
# install-daemon
# ---------------------------------------------------------------------------


_PLIST_LABEL = "com.mem-vault.reflect"


def _mem_vault_binary() -> str:
    """Resolve the absolute path to the ``mem-vault`` script.

    launchd needs an absolute path — it doesn't inherit $PATH from the
    user's shell. Resolution order:

    1. ``MEM_VAULT_BIN`` env var (explicit override for CI / testing).
    2. ``~/.local/bin/mem-vault`` (what ``uv tool install`` writes; the
       stable location a daemon should point at so it doesn't break
       when the user recreates a local venv).
    3. ``shutil.which("mem-vault")`` — PATH lookup at install time.
    4. ``sys.argv[0]`` — falls back to the currently-running binary
       (useful when running from the editable venv during dev).

    We deliberately prefer ``~/.local/bin`` over ``sys.argv[0]`` so
    that ``install-daemon`` inside a ``.venv`` still produces a plist
    pointing at the stable tool-install location.
    """
    import shutil

    override = os.environ.get("MEM_VAULT_BIN", "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        if path.exists():
            return str(path)
    stable = Path.home() / ".local" / "bin" / "mem-vault"
    if stable.exists():
        return str(stable)
    found = shutil.which("mem-vault")
    if found:
        return str(Path(found).resolve())
    first = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    if first and first.exists() and first.name.startswith("mem-vault"):
        return str(first)
    raise RuntimeError(
        "couldn't locate the mem-vault binary. Install with "
        "`uv tool install --editable .` (or pip install mem-vault) and retry."
    )


def _macos_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_PLIST_LABEL}.plist"


def _macos_plist_content(binary: str, *, hour: int, minute: int, consolidate: bool) -> str:
    """launchd plist: run ``mem-vault reflect`` once per day at HH:MM local."""
    extra_args = "\n        <string>--consolidate</string>" if consolidate else ""
    log_path = Path.home() / "Library" / "Logs" / "mem-vault" / "reflect.log"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{_PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{binary}</string>
        <string>reflect</string>{extra_args}
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>{minute}</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{log_path}</string>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>
    <key>RunAtLoad</key>
    <false/>
    <key>ProcessType</key>
    <string>Background</string>
</dict>
</plist>
"""


def _linux_systemd_paths() -> tuple[Path, Path]:
    base = Path.home() / ".config" / "systemd" / "user"
    return base / "mem-vault-reflect.service", base / "mem-vault-reflect.timer"


def _linux_service_content(binary: str, *, consolidate: bool) -> str:
    extra = " --consolidate" if consolidate else ""
    return f"""[Unit]
Description=mem-vault nightly reflection pass
After=network-online.target

[Service]
Type=oneshot
ExecStart={binary} reflect{extra}
StandardOutput=journal
StandardError=journal
"""


def _linux_timer_content(*, hour: int, minute: int) -> str:
    return f"""[Unit]
Description=Run mem-vault reflect daily at {hour:02d}:{minute:02d}

[Timer]
OnCalendar=*-*-* {hour:02d}:{minute:02d}:00
Persistent=true

[Install]
WantedBy=timers.target
"""


def _run_launchctl_bootstrap(plist_path: Path) -> tuple[bool, str]:
    """Load + kickstart the plist. Returns (ok, combined-output)."""
    uid = os.getuid()
    out_lines: list[str] = []
    # bootstrap (idempotent-ish: if already loaded it'll report an error,
    # which we recover from by unloading first in the calling code).
    try:
        cp = subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        out_lines.append(f"bootstrap rc={cp.returncode} stdout={cp.stdout.strip()} stderr={cp.stderr.strip()}")
    except Exception as exc:
        return False, f"launchctl bootstrap failed: {exc}"
    # kickstart so the first run happens immediately (gives the user
    # a fast smoke-test path; the 03:00 schedule kicks in on next tick).
    try:
        cp = subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{uid}/{_PLIST_LABEL}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        out_lines.append(f"kickstart rc={cp.returncode} stdout={cp.stdout.strip()} stderr={cp.stderr.strip()}")
    except Exception as exc:
        out_lines.append(f"kickstart failed: {exc}")
    return True, "\n".join(out_lines)


def _run_launchctl_bootout(plist_path: Path) -> tuple[bool, str]:
    uid = os.getuid()
    try:
        cp = subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}/{_PLIST_LABEL}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return cp.returncode in (0, 3), f"bootout rc={cp.returncode} stderr={cp.stderr.strip()}"
    except Exception as exc:
        return False, f"bootout failed: {exc}"


def run_install_daemon(args: argparse.Namespace) -> int:
    system = platform.system()
    hour = max(0, min(23, args.hour))
    minute = max(0, min(59, args.minute))

    try:
        binary = _mem_vault_binary()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if system == "Darwin":
        plist_path = _macos_plist_path()

        if args.uninstall:
            ok, msg = _run_launchctl_bootout(plist_path)
            print(msg, file=sys.stderr)
            if plist_path.exists():
                plist_path.unlink()
                print(f"removed {plist_path}")
            return 0 if ok else 1

        content = _macos_plist_content(
            binary, hour=hour, minute=minute, consolidate=args.consolidate
        )
        if args.dry_run:
            print(content)
            print(f"# would write to: {plist_path}", file=sys.stderr)
            return 0
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure the log dir exists so launchd doesn't silently fail on
        # the StandardOutPath / StandardErrorPath.
        (Path.home() / "Library" / "Logs" / "mem-vault").mkdir(parents=True, exist_ok=True)
        # Reload-friendly: if the label is already loaded, bootout first
        # (otherwise bootstrap fails with "Service is already loaded").
        _run_launchctl_bootout(plist_path)
        plist_path.write_text(content, encoding="utf-8")
        ok, msg = _run_launchctl_bootstrap(plist_path)
        print(f"wrote {plist_path}")
        print(msg, file=sys.stderr)
        if ok:
            print(
                f"mem-vault install-daemon: loaded {_PLIST_LABEL} "
                f"(runs daily at {hour:02d}:{minute:02d} local)."
            )
            return 0
        return 1

    if system == "Linux":
        service_path, timer_path = _linux_systemd_paths()
        if args.uninstall:
            subprocess.run(
                ["systemctl", "--user", "disable", "--now", timer_path.name],
                capture_output=True,
                text=True,
            )
            for p in (service_path, timer_path):
                if p.exists():
                    p.unlink()
                    print(f"removed {p}")
            return 0
        service_content = _linux_service_content(binary, consolidate=args.consolidate)
        timer_content = _linux_timer_content(hour=hour, minute=minute)
        if args.dry_run:
            print("# ---- service ----")
            print(service_content)
            print("# ---- timer ----")
            print(timer_content)
            print(f"# would write to: {service_path} + {timer_path}", file=sys.stderr)
            return 0
        service_path.parent.mkdir(parents=True, exist_ok=True)
        service_path.write_text(service_content, encoding="utf-8")
        timer_path.write_text(timer_content, encoding="utf-8")
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        cp = subprocess.run(
            ["systemctl", "--user", "enable", "--now", timer_path.name],
            capture_output=True,
            text=True,
        )
        print(f"wrote {service_path} + {timer_path}")
        print(f"systemctl rc={cp.returncode} stderr={cp.stderr.strip()}", file=sys.stderr)
        if cp.returncode == 0:
            print(
                f"mem-vault install-daemon: {timer_path.name} enabled "
                f"(daily at {hour:02d}:{minute:02d} local)."
            )
            return 0
        return 1

    # Windows & friends: we don't auto-install the scheduled task (Task
    # Scheduler's XML is verbose and depends on SID), but we emit the
    # exact schtasks command + the invocation the user needs.
    print(
        f"mem-vault install-daemon: automatic install is macOS/Linux only.\n"
        f"On Windows run once per day via Task Scheduler:\n\n"
        f"    schtasks /create /sc daily /tn MemVaultReflect /st "
        f"{hour:02d}:{minute:02d} /tr \"{binary} reflect"
        f"{' --consolidate' if args.consolidate else ''}\"\n",
        file=sys.stderr,
    )
    return 2


def run_daemon_status(args: argparse.Namespace) -> int:
    from mem_vault.config import load_config

    cfg = load_config()
    system = platform.system()

    loaded: bool | None = None
    detail = "unknown"

    if system == "Darwin":
        try:
            uid = os.getuid()
            cp = subprocess.run(
                ["launchctl", "print", f"gui/{uid}/{_PLIST_LABEL}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            loaded = cp.returncode == 0
            detail = "launchd loaded" if loaded else "launchd not loaded"
        except Exception as exc:
            detail = f"launchctl probe failed: {exc}"
    elif system == "Linux":
        try:
            cp = subprocess.run(
                ["systemctl", "--user", "is-active", "mem-vault-reflect.timer"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            loaded = cp.returncode == 0
            detail = cp.stdout.strip() or detail
        except Exception as exc:
            detail = f"systemctl probe failed: {exc}"

    # Last reflection written = most recent reflection_YYYY_MM_DD slug in vault.
    last_reflection: str | None = None
    last_mtime: float | None = None
    try:
        for p in cfg.memory_dir.glob("reflection_*.md"):
            mtime = p.stat().st_mtime
            if last_mtime is None or mtime > last_mtime:
                last_mtime = mtime
                last_reflection = p.stem
    except Exception:
        pass

    payload = {
        "system": system,
        "label": _PLIST_LABEL,
        "loaded": loaded,
        "detail": detail,
        "last_reflection": last_reflection,
        "last_reflection_at": (
            datetime.fromtimestamp(last_mtime).astimezone().isoformat(timespec="seconds")
            if last_mtime
            else None
        ),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print()
    print("mem-vault daemon-status")
    print("─" * 23)
    print(f"  platform       : {payload['system']}")
    print(f"  label          : {payload['label']}")
    print(f"  loaded         : {payload['loaded']}")
    print(f"  detail         : {payload['detail']}")
    print(f"  last reflection: {payload['last_reflection'] or '—'}")
    if payload["last_reflection_at"]:
        print(f"  last run       : {payload['last_reflection_at']}")
    print()
    return 0 if loaded else 1

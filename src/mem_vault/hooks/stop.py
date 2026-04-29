"""Stop hook: append a tab-separated audit line to ``~/.local/share/mem-vault/sessions.log``.

Never blocks the agent. Doesn't call the LLM. Doesn't touch the vault. Just a
breadcrumb so you can later answer "when did I last close a session here, and
how many memories were in the vault at that moment?".
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path


def _vault_memory_count() -> int:
    try:
        from mem_vault.config import load_config
    except Exception:
        return -1
    try:
        cfg = load_config()
    except Exception:
        return -1
    try:
        return sum(1 for _ in cfg.memory_dir.glob("*.md"))
    except Exception:
        return -1


def run() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}

    log_dir = Path.home() / ".local" / "share" / "mem-vault"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "sessions.log"

    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    cwd = os.environ.get("DEVIN_PROJECT_DIR") or os.getcwd()
    stop_active = bool(payload.get("stop_hook_active"))
    count = _vault_memory_count()

    line = f"{now}\tstop\tcwd={cwd}\tmemories={count}\tstop_hook_active={stop_active}\n"
    try:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:
        print(f"mem-vault: stop_log write failed: {exc}", file=sys.stderr)

"""Lifecycle hook subcommands.

The actual hook bodies live in :mod:`mem_vault.hooks`; the CLI surface
here is a thin dispatcher so the agent runner (Devin / Claude Code) can
spawn ``mem-vault hook-<event>`` from its hooks config. Imports happen
lazily inside :func:`run` so the no-op cases (every other subcommand)
don't pay for loading Ollama / mem0 just to print ``--help``.
"""

from __future__ import annotations

import argparse


def add_subparsers(sub: argparse._SubParsersAction) -> None:
    sub.add_parser(
        "hook-sessionstart",
        help="SessionStart lifecycle hook: inject preferences into agent context.",
    )
    sub.add_parser(
        "hook-userprompt",
        help="UserPromptSubmit lifecycle hook: inject per-prompt memories via semantic search.",
    )
    sub.add_parser(
        "hook-stop",
        help="Stop lifecycle hook: append a line to the audit log. Never blocks.",
    )


def run(cmd: str) -> None:
    if cmd == "hook-sessionstart":
        from mem_vault.hooks import sessionstart

        sessionstart.run()
        return

    if cmd == "hook-userprompt":
        from mem_vault.hooks import userprompt

        userprompt.run()
        return

    if cmd == "hook-stop":
        from mem_vault.hooks import stop

        stop.run()
        return

    raise ValueError(f"unknown hook subcommand: {cmd}")

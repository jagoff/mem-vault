"""``mem-vault ui`` — start the local browser UI.

The UI dependencies are gated behind the ``[ui]`` extra so plain MCP-only
installs don't pull FastAPI / Jinja2 / uvicorn. We surface a friendly
``ImportError`` if the extra is missing instead of a stack trace.
"""

from __future__ import annotations

import argparse
import sys


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p_ui = sub.add_parser(
        "ui",
        help="Start a local browser UI to browse, search, edit, and delete memories.",
    )
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=7880)
    p_ui.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )


def run(args: argparse.Namespace) -> None:
    try:
        from mem_vault.ui.server import serve as ui_serve
    except ImportError as exc:
        print(
            f"error: ui dependencies not installed ({exc}). "
            "Install with: uv tool install --editable '.[ui]'",
            file=sys.stderr,
        )
        sys.exit(2)
    ui_serve(host=args.host, port=args.port, log_level=args.log_level)

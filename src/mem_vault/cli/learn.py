"""CLI subcommands for the closed-loop ranker (v0.6.0).

Three flat commands, mirroring the ``sync-status`` / ``sync-watch``
naming convention so users don't need to remember a sub-subparser:

- ``mem-vault telemetry-stats`` — print snapshot of the search-events DB
  (counts, citation rate, avg cited rank, DB path & size). Useful for
  "is the closed-loop actually getting signal?" before running ``train``.

- ``mem-vault ranker-train`` — fit a logistic regression on the
  recorded events and write ``state_dir/ranker/ranker_v{N}.pkl``
  (and atomically swap ``active.pkl`` to point at the new file).
  Reports n_train / n_positive / AUC.

- ``mem-vault ranker-eval`` — load the active ranker and reuse the
  ``mem-vault eval`` harness to compute hit@k + MRR with the model
  on. Useful to compare against the heuristic baseline before
  deciding to keep the new version.

- ``mem-vault ranker-rollback`` — atomically replace the active
  pickle with the previous version (kill-switch when a train run
  ends up worse than the last one).

All commands tolerate the ``[learning]`` extra not being installed
and report a friendly install hint instead of a traceback.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime


def _format_ts(ts: float | None) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts, tz=UTC).astimezone().isoformat(timespec="seconds")


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    size = float(n)
    for unit in ("KB", "MB", "GB"):
        size /= 1024.0
        if size < 1024:
            return f"{size:.1f} {unit}"
    return f"{size:.1f} TB"


def add_subparsers(sub: argparse._SubParsersAction) -> None:
    """Register the four flat commands under the top-level CLI parser."""

    p_stats = sub.add_parser(
        "telemetry-stats",
        help="Show search-events DB stats (counts, citation rate, paths).",
        description=(
            "Print a quick snapshot of the search-events SQLite that "
            "feeds ``mem-vault ranker-train``. Use to verify the "
            "closed-loop is actually accumulating signal before "
            "kicking off a training run."
        ),
    )
    p_stats.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")

    p_train = sub.add_parser(
        "ranker-train",
        help="Fit a logistic regression on recorded search events.",
        description=(
            "Reads ``state_dir/search_events.db``, fits a logreg over "
            "[score_dense, score_final, rank, helpful_ratio, usage_count, "
            "recency_days, project_match, agent_id_match], writes a new "
            "versioned pickle, and atomically swaps the active pointer. "
            "Requires the [learning] extra (scikit-learn)."
        ),
    )
    p_train.add_argument(
        "--min-rows",
        type=int,
        default=50,
        help="Refuse to train below this row count (default 50).",
    )
    p_train.add_argument(
        "--min-positives",
        type=int,
        default=5,
        help="Refuse below this many cited rows (default 5).",
    )
    p_train.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")

    p_eval = sub.add_parser(
        "ranker-eval",
        help="Run the eval harness with the learned ranker enabled.",
        description=(
            "Wraps ``mem-vault eval`` with MEM_VAULT_LEARNED_RANKER=1 "
            "for the duration of the run, so you can compare hit@k / "
            "MRR against the heuristic baseline."
        ),
    )
    p_eval.add_argument("--queries", type=str, default=None, help="Path to queries.json (default: discovered).")
    p_eval.add_argument("--k", type=int, default=5, help="Top-k for hit@k (default 5).")
    p_eval.add_argument("--threshold", type=float, default=0.0, help="Score threshold (default 0.0).")
    p_eval.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")

    p_rollback = sub.add_parser(
        "ranker-rollback",
        help="Demote the active ranker pickle and promote the previous version.",
        description=(
            "When a fresh train run ends up worse than the previous one "
            "(check via ``ranker-eval``), use this to atomically restore "
            "the prior pickle. No-op when there's no prior version."
        ),
    )
    p_rollback.add_argument(
        "--json", action="store_true", help="Emit JSON instead of a one-liner."
    )


def run_telemetry_stats(args: argparse.Namespace) -> int:
    from mem_vault import telemetry
    from mem_vault.config import load_config

    cfg = load_config()
    snap = telemetry.stats(cfg.state_dir)

    if getattr(args, "json", False):
        print(json.dumps(snap, indent=2, sort_keys=True))
        return 0

    print()
    print("mem-vault telemetry-stats")
    print("─" * 25)
    print(f"  db path           : {snap['db_path']}")
    print(f"  db size           : {_format_bytes(int(snap['db_size_bytes']))}")
    print(f"  total events      : {snap['total_events']}")
    print(f"  unique queries    : {snap['unique_queries']}")
    print(f"  citations         : {snap['citations']}")
    if snap["total_events"]:
        print(f"  citation rate     : {snap['citation_rate'] * 100:.1f}%")
    if snap["avg_cited_rank"] is not None:
        print(f"  avg cited rank    : {snap['avg_cited_rank']:.2f}")
    print(f"  first event       : {_format_ts(snap['first_event_ts'])}")
    print(f"  last event        : {_format_ts(snap['last_event_ts'])}")
    print()
    if snap["total_events"] == 0:
        print("Tip: run a few searches with the MCP server (or `mem-vault search ...`)")
        print("to populate the DB. The Stop hook will mark citations automatically.")
        print()
    return 0


def run_ranker_train(args: argparse.Namespace) -> int:
    from mem_vault.config import load_config

    cfg = load_config()

    try:
        from mem_vault import ranker
    except ImportError:
        print(
            "error: ranker module unavailable. Install the [learning] extra:\n"
            "    uv tool install --editable '.[learning]'",
            file=sys.stderr,
        )
        return 2

    try:
        result = ranker.train(
            cfg.state_dir,
            min_rows=args.min_rows,
            min_positives=args.min_positives,
        )
    except RuntimeError as exc:
        # Missing scikit-learn — translate to a friendly hint.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        # Not enough data — soft failure, exit 1 (CI-friendly).
        print(f"mem-vault ranker-train: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {**result.__dict__, "feature_columns": list(result.feature_columns)},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print()
    print(f"mem-vault ranker-train  (v{result.version})")
    print("─" * 32)
    print(f"  trained_at  : {_format_ts(result.trained_at)}")
    print(f"  rows        : {result.n_train} ({result.n_positive} positive, {result.n_negative} negative)")
    if result.auc is not None:
        print(f"  AUC (held)  : {result.auc:.3f}")
    else:
        print("  AUC (held)  : — (not enough rows for a stable split)")
    print(f"  pickle      : {result.pickle_path}")
    print()
    print("Activate with: export MEM_VAULT_LEARNED_RANKER=1")
    print("Compare with heuristic baseline: mem-vault ranker-eval")
    print()
    return 0


def run_ranker_eval(args: argparse.Namespace) -> int:
    """Run the existing eval harness with the learned ranker enabled.

    We delegate to ``cli.eval_cmd.run`` for the actual heavy-lifting —
    reusing its hit@k + MRR + threshold scaffolding rather than
    reimplementing it. The only difference is we toggle
    ``MEM_VAULT_LEARNED_RANKER=1`` for the duration of the call.
    """
    import os

    from mem_vault.cli import eval_cmd

    prev = os.environ.get("MEM_VAULT_LEARNED_RANKER")
    os.environ["MEM_VAULT_LEARNED_RANKER"] = "1"
    try:
        return asyncio.run(eval_cmd.run(args))
    finally:
        if prev is None:
            os.environ.pop("MEM_VAULT_LEARNED_RANKER", None)
        else:
            os.environ["MEM_VAULT_LEARNED_RANKER"] = prev


def run_ranker_rollback(args: argparse.Namespace) -> int:
    from mem_vault.config import load_config

    cfg = load_config()

    try:
        from mem_vault import ranker
    except ImportError:
        print(
            "error: ranker module unavailable. Install the [learning] extra.",
            file=sys.stderr,
        )
        return 2

    target = ranker.rollback(cfg.state_dir)
    if target is None:
        msg = "no prior ranker version to roll back to"
        if args.json:
            print(json.dumps({"ok": False, "reason": msg}))
        else:
            print(f"mem-vault ranker-rollback: {msg}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"ok": True, "active_version": target}))
    else:
        print(f"mem-vault ranker-rollback: active = ranker_v{target}.pkl")
    return 0

"""Retrieval eval harness — hit@k and MRR over a labeled query set.

When we tweak the retrieval pipeline (decay, rerank, hybrid, usage
boost) we want a number that says "this change helps" — not a vibe.
This module runs a set of queries against ``MemVaultService.search``
and scores the results against ground-truth relevant memory ids.

Design:

- **Query shape**: a list of ``{query, expected}`` pairs. ``expected``
  is the set of memory ids that "should" rank in the top-k.
- **Metrics**:
  - ``hit@k`` for k in {1, 3, 5, 10}: fraction of queries where at
    least one expected id appears in the top-k results.
  - ``mrr``: mean reciprocal rank — the average of ``1 / rank`` of
    the first relevant hit (0 when none appear in top-k cap).
- **Baseline**: run the suite twice (e.g. with and without reranker)
  and compute deltas.

Pure functions for testability: ``compute_metrics`` takes the runs
(already executed) and produces the summary. The CLI layer owns the
network/LLM calls and passes the runs down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class EvalQuery:
    """One query with its ground-truth relevant memory ids."""

    query: str
    expected: list[str] = field(default_factory=list)
    # Free-form, shown in the report (e.g. "keyword lookup").
    tag: str = ""


@dataclass
class EvalRun:
    """One executed query — the returned ids in rank order."""

    query: EvalQuery
    returned: list[str]

    def reciprocal_rank(self) -> float:
        """Return ``1/rank`` for the first expected id; 0 if none appears."""
        expected_set = set(self.query.expected)
        for pos, mem_id in enumerate(self.returned, start=1):
            if mem_id in expected_set:
                return 1.0 / pos
        return 0.0

    def hit_at(self, k: int) -> bool:
        """True iff any expected id appears in the top-``k`` returned ids."""
        if k <= 0 or not self.query.expected:
            return False
        return any(mid in set(self.query.expected) for mid in self.returned[:k])


@dataclass
class EvalReport:
    """Aggregate metrics across a run set."""

    total_queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    mrr: float
    per_query: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "hit@1": round(self.hit_at_1, 4),
            "hit@3": round(self.hit_at_3, 4),
            "hit@5": round(self.hit_at_5, 4),
            "hit@10": round(self.hit_at_10, 4),
            "mrr": round(self.mrr, 4),
            "per_query": self.per_query,
        }

    def render(self) -> str:
        lines = [
            f"  queries     : {self.total_queries}",
            f"  hit@1       : {self.hit_at_1:.2%}",
            f"  hit@3       : {self.hit_at_3:.2%}",
            f"  hit@5       : {self.hit_at_5:.2%}",
            f"  hit@10      : {self.hit_at_10:.2%}",
            f"  mrr         : {self.mrr:.4f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pure metric computation
# ---------------------------------------------------------------------------


def compute_metrics(runs: list[EvalRun]) -> EvalReport:
    """Aggregate hit@k and MRR over a list of executed queries.

    Empty ``runs`` returns an all-zero report (useful for tests). Per-
    query rows carry enough info to render a detail table in the CLI.
    """
    if not runs:
        return EvalReport(
            total_queries=0,
            hit_at_1=0.0,
            hit_at_3=0.0,
            hit_at_5=0.0,
            hit_at_10=0.0,
            mrr=0.0,
        )
    n = len(runs)
    h1 = sum(1 for r in runs if r.hit_at(1)) / n
    h3 = sum(1 for r in runs if r.hit_at(3)) / n
    h5 = sum(1 for r in runs if r.hit_at(5)) / n
    h10 = sum(1 for r in runs if r.hit_at(10)) / n
    mrr = sum(r.reciprocal_rank() for r in runs) / n
    per_query = [
        {
            "query": r.query.query,
            "tag": r.query.tag,
            "expected": r.query.expected,
            "returned": r.returned[:10],
            "hit@5": r.hit_at(5),
            "rr": round(r.reciprocal_rank(), 4),
        }
        for r in runs
    ]
    return EvalReport(
        total_queries=n,
        hit_at_1=h1,
        hit_at_3=h3,
        hit_at_5=h5,
        hit_at_10=h10,
        mrr=mrr,
        per_query=per_query,
    )


def load_queries(path: Path) -> list[EvalQuery]:
    """Load ``[{query, expected, tag?}, ...]`` from a JSON file.

    Missing fields degrade gracefully: ``expected=[]`` (never hits),
    ``tag=""`` (unrendered). Ids inside ``expected`` are cast to str.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"expected a JSON list, got {type(raw).__name__}")
    queries: list[EvalQuery] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        q = str(item.get("query") or "").strip()
        if not q:
            continue
        expected = [str(x) for x in (item.get("expected") or []) if x]
        tag = str(item.get("tag") or "").strip()
        queries.append(EvalQuery(query=q, expected=expected, tag=tag))
    return queries


def diff_reports(baseline: EvalReport, candidate: EvalReport) -> dict[str, float]:
    """Compute ``candidate - baseline`` deltas on the numeric metrics."""
    return {
        "hit@1": candidate.hit_at_1 - baseline.hit_at_1,
        "hit@3": candidate.hit_at_3 - baseline.hit_at_3,
        "hit@5": candidate.hit_at_5 - baseline.hit_at_5,
        "hit@10": candidate.hit_at_10 - baseline.hit_at_10,
        "mrr": candidate.mrr - baseline.mrr,
    }

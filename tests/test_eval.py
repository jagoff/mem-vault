"""Tests for ``mem_vault.eval`` — pure metric computation + query loading.

The CLI harness (``mem_vault.cli.eval_cmd``) is integration-tested
elsewhere (it needs a real service + Ollama). This file focuses on the
pure functions: ``compute_metrics``, ``reciprocal_rank``, ``hit_at``,
``load_queries``, and ``diff_reports``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mem_vault.eval import (
    EvalQuery,
    EvalRun,
    compute_metrics,
    diff_reports,
    load_queries,
)

# ---------------------------------------------------------------------------
# EvalRun methods
# ---------------------------------------------------------------------------


def test_reciprocal_rank_first_hit_is_one():
    run = EvalRun(
        query=EvalQuery(query="q", expected=["a"]),
        returned=["a", "b", "c"],
    )
    assert run.reciprocal_rank() == 1.0


def test_reciprocal_rank_second_hit_is_half():
    run = EvalRun(
        query=EvalQuery(query="q", expected=["b"]),
        returned=["a", "b", "c"],
    )
    assert run.reciprocal_rank() == 0.5


def test_reciprocal_rank_no_hit_is_zero():
    run = EvalRun(
        query=EvalQuery(query="q", expected=["nope"]),
        returned=["a", "b", "c"],
    )
    assert run.reciprocal_rank() == 0.0


def test_reciprocal_rank_empty_returned_is_zero():
    run = EvalRun(query=EvalQuery(query="q", expected=["a"]), returned=[])
    assert run.reciprocal_rank() == 0.0


def test_reciprocal_rank_multiple_expected_uses_earliest_match():
    """When several expected ids appear, MRR uses the first-ranked one."""
    run = EvalRun(
        query=EvalQuery(query="q", expected=["b", "c"]),
        returned=["a", "b", "c"],
    )
    assert run.reciprocal_rank() == 0.5  # 'b' at rank 2


@pytest.mark.parametrize(
    "returned,k,expected_hit",
    [
        (["a", "b"], 1, True),
        (["b", "a"], 1, False),  # 'a' not in top-1
        (["b", "a"], 2, True),
        (["x", "y"], 5, False),
        ([], 5, False),
    ],
)
def test_hit_at_various_k(returned, k, expected_hit):
    run = EvalRun(
        query=EvalQuery(query="q", expected=["a"]),
        returned=returned,
    )
    assert run.hit_at(k) is expected_hit


def test_hit_at_zero_k_returns_false():
    run = EvalRun(query=EvalQuery(query="q", expected=["a"]), returned=["a"])
    assert run.hit_at(0) is False


def test_hit_at_no_expected_returns_false():
    """Query with empty expected list should not count as a hit."""
    run = EvalRun(query=EvalQuery(query="q", expected=[]), returned=["a"])
    assert run.hit_at(5) is False


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_empty_returns_zeros():
    report = compute_metrics([])
    assert report.total_queries == 0
    assert report.hit_at_1 == 0.0
    assert report.mrr == 0.0
    assert report.per_query == []


def test_compute_metrics_all_hits_at_one():
    runs = [
        EvalRun(query=EvalQuery(query="q1", expected=["a"]), returned=["a", "b"]),
        EvalRun(query=EvalQuery(query="q2", expected=["c"]), returned=["c", "d"]),
    ]
    report = compute_metrics(runs)
    assert report.total_queries == 2
    assert report.hit_at_1 == 1.0
    assert report.hit_at_5 == 1.0
    assert report.mrr == 1.0


def test_compute_metrics_partial_hits():
    runs = [
        # hit@1, rr=1
        EvalRun(query=EvalQuery(query="q1", expected=["a"]), returned=["a", "x"]),
        # no hit, rr=0
        EvalRun(query=EvalQuery(query="q2", expected=["c"]), returned=["x", "y", "z"]),
        # hit@3 (rank 3), rr=1/3
        EvalRun(
            query=EvalQuery(query="q3", expected=["e"]),
            returned=["x", "y", "e", "f"],
        ),
    ]
    report = compute_metrics(runs)
    assert report.total_queries == 3
    assert report.hit_at_1 == pytest.approx(1 / 3)
    assert report.hit_at_3 == pytest.approx(2 / 3)
    assert report.hit_at_5 == pytest.approx(2 / 3)
    # MRR = (1 + 0 + 1/3) / 3
    assert report.mrr == pytest.approx((1 + 0 + 1 / 3) / 3)


def test_compute_metrics_per_query_rows_carry_detail():
    runs = [
        EvalRun(
            query=EvalQuery(query="q", expected=["x"], tag="kw"),
            returned=["a", "b", "x", "d"],
        ),
    ]
    report = compute_metrics(runs)
    assert len(report.per_query) == 1
    row = report.per_query[0]
    assert row["query"] == "q"
    assert row["tag"] == "kw"
    assert row["expected"] == ["x"]
    # Top-10 slice (we returned 4 → all 4 in the row)
    assert row["returned"] == ["a", "b", "x", "d"]
    assert row["hit@5"] is True
    # per_query rows round rr to 4 decimals for display friendliness.
    assert row["rr"] == pytest.approx(1 / 3, abs=1e-3)


def test_to_dict_roundtrips_through_json():
    runs = [
        EvalRun(query=EvalQuery(query="q", expected=["a"]), returned=["a"]),
    ]
    report = compute_metrics(runs)
    s = json.dumps(report.to_dict())  # must be JSON-serializable
    reloaded = json.loads(s)
    assert reloaded["hit@1"] == 1.0
    assert reloaded["total_queries"] == 1


def test_render_produces_table_like_output():
    runs = [
        EvalRun(query=EvalQuery(query="q", expected=["a"]), returned=["a"]),
    ]
    out = compute_metrics(runs).render()
    assert "queries" in out
    assert "hit@1" in out
    assert "hit@5" in out
    assert "mrr" in out


# ---------------------------------------------------------------------------
# load_queries
# ---------------------------------------------------------------------------


def test_load_queries_happy_path(tmp_path: Path):
    queries_file = tmp_path / "q.json"
    queries_file.write_text(
        json.dumps(
            [
                {"query": "where is X", "expected": ["mem1"], "tag": "kw"},
                {"query": "who said Y", "expected": ["mem2", "mem3"]},
            ]
        ),
        encoding="utf-8",
    )
    qs = load_queries(queries_file)
    assert len(qs) == 2
    assert qs[0].query == "where is X"
    assert qs[0].expected == ["mem1"]
    assert qs[0].tag == "kw"
    assert qs[1].expected == ["mem2", "mem3"]


def test_load_queries_skips_invalid_entries(tmp_path: Path):
    queries_file = tmp_path / "q.json"
    queries_file.write_text(
        json.dumps(
            [
                {"query": "valid"},
                {"no_query_field": True},
                "not even a dict",
                {"query": "  "},  # empty after strip
                {"query": "also valid", "expected": [123, "a"]},  # coerce to str
            ]
        ),
        encoding="utf-8",
    )
    qs = load_queries(queries_file)
    assert [q.query for q in qs] == ["valid", "also valid"]
    assert qs[1].expected == ["123", "a"]


def test_load_queries_non_list_raises(tmp_path: Path):
    queries_file = tmp_path / "q.json"
    queries_file.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_queries(queries_file)


# ---------------------------------------------------------------------------
# diff_reports
# ---------------------------------------------------------------------------


def test_diff_reports_positive_delta():
    baseline = compute_metrics(
        [EvalRun(query=EvalQuery(query="q", expected=["a"]), returned=["x", "a"])]
    )
    candidate = compute_metrics(
        [EvalRun(query=EvalQuery(query="q", expected=["a"]), returned=["a"])]
    )
    delta = diff_reports(baseline, candidate)
    assert delta["hit@1"] == pytest.approx(1.0 - 0.0)
    assert delta["mrr"] == pytest.approx(1.0 - 0.5)


def test_diff_reports_same_inputs_zero_delta():
    runs = [EvalRun(query=EvalQuery(query="q", expected=["a"]), returned=["a"])]
    a = compute_metrics(runs)
    b = compute_metrics(runs)
    delta = diff_reports(a, b)
    for v in delta.values():
        assert v == 0.0

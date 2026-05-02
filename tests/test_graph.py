"""Unit tests for ``mem_vault.graph`` — pure BFS over the vault knowledge graph.

These tests pin the contract that:

- ``build_adjacency`` symmetrizes related/contradicts (the on-disk fields
  are asymmetric; the graph traversal must be undirected).
- co-tag edges respect ``min_shared_tags`` and ``include_cotag=False``
  to disable them.
- ``expand_neighborhood`` returns seeds at hop=0, neighbors at hop=1, etc.
- ``edge_kinds`` filter narrows the traversal correctly.
- ``max_nodes`` cap is respected.
- ``contradictions_for`` returns the listed memorias and tolerates dangling
  contradictions ids (memorias deleted out-of-band).
"""

from __future__ import annotations

from mem_vault import graph
from mem_vault.storage import Memory


def _m(
    mid: str,
    *,
    related: list[str] | None = None,
    contradicts: list[str] | None = None,
    tags: list[str] | None = None,
    name: str | None = None,
    type: str = "note",
) -> Memory:
    return Memory(
        id=mid,
        name=name or f"Memory {mid}",
        description=f"desc {mid}",
        type=type,
        tags=tags or [],
        related=related or [],
        contradicts=contradicts or [],
    )


def test_adjacency_symmetrizes_related() -> None:
    corpus = [_m("a", related=["b"]), _m("b"), _m("c")]
    adj = graph.build_adjacency(corpus)
    assert "b" in adj["a"]
    assert "a" in adj["b"]  # symmetrized
    assert "related" in adj["a"]["b"]
    assert "related" in adj["b"]["a"]


def test_adjacency_symmetrizes_contradicts() -> None:
    corpus = [_m("a", contradicts=["b"]), _m("b")]
    adj = graph.build_adjacency(corpus)
    assert adj["a"]["b"] == {"contradicts"}
    assert adj["b"]["a"] == {"contradicts"}


def test_adjacency_includes_cotag_when_enabled() -> None:
    corpus = [
        _m("a", tags=["project:rag", "ollama"]),
        _m("b", tags=["project:rag", "ollama", "qdrant"]),
        _m("c", tags=["unrelated"]),
    ]
    adj = graph.build_adjacency(corpus, min_shared_tags=2, include_cotag=True)
    assert "b" in adj["a"]
    assert "cotag" in adj["a"]["b"]
    # c shares no tags with anyone → no edges from c
    assert adj["c"] == {}


def test_adjacency_drops_cotag_when_disabled() -> None:
    corpus = [
        _m("a", tags=["x", "y"]),
        _m("b", tags=["x", "y"]),
    ]
    adj = graph.build_adjacency(corpus, min_shared_tags=2, include_cotag=False)
    assert adj["a"] == {}
    assert adj["b"] == {}


def test_adjacency_normalizes_project_tags() -> None:
    """``project:rag`` and ``project:rag-experimental`` cluster as ``rag`` / ``rag-experimental``."""
    corpus = [
        _m("a", tags=["project:rag", "lang:py"]),
        _m("b", tags=["project:rag", "lang:py"]),
    ]
    adj = graph.build_adjacency(corpus, min_shared_tags=2)
    # Both share ``rag`` and ``py`` after normalization → cotag.
    assert adj["a"].get("b") == {"cotag"}


def test_expand_seed_returns_at_hop_zero() -> None:
    corpus = [_m("a"), _m("b"), _m("c")]
    nodes = graph.expand_neighborhood(corpus, ["a"], hops=0)
    assert "a" in nodes
    assert nodes["a"].hop == 0
    assert nodes["a"].edges == set()
    # No traversal at hops=0 → only the seed.
    assert len(nodes) == 1


def test_expand_one_hop_via_related() -> None:
    corpus = [
        _m("a", related=["b"]),
        _m("b", related=["c"]),
        _m("c"),
        _m("d"),
    ]
    nodes = graph.expand_neighborhood(corpus, ["a"], hops=1)
    assert set(nodes) == {"a", "b"}
    assert nodes["b"].hop == 1
    assert nodes["b"].edges == {"related"}


def test_expand_two_hops_chains_through_neighbors() -> None:
    corpus = [
        _m("a", related=["b"]),
        _m("b", related=["c"]),
        _m("c"),
    ]
    nodes = graph.expand_neighborhood(corpus, ["a"], hops=2)
    assert set(nodes) == {"a", "b", "c"}
    assert nodes["c"].hop == 2


def test_expand_records_shortest_hop_when_two_paths() -> None:
    """If a node is reachable at hop 1 AND hop 2, store the shortest one."""
    corpus = [
        _m("a", related=["b", "c"]),
        _m("b", related=["c"]),
        _m("c"),
    ]
    nodes = graph.expand_neighborhood(corpus, ["a"], hops=2)
    assert nodes["c"].hop == 1


def test_expand_filters_edge_kinds() -> None:
    """``edge_kinds=("contradicts",)`` only follows contradiction edges."""
    corpus = [
        _m("a", related=["b"], contradicts=["c"]),
        _m("b"),
        _m("c"),
    ]
    nodes = graph.expand_neighborhood(corpus, ["a"], hops=1, edge_kinds=("contradicts",))
    assert set(nodes) == {"a", "c"}
    assert nodes["c"].edges == {"contradicts"}


def test_expand_respects_max_nodes() -> None:
    """A 4-node star with seed=center, max_nodes=3 returns at most 3."""
    corpus = [
        _m("center", related=["a", "b", "c", "d"]),
        _m("a"),
        _m("b"),
        _m("c"),
        _m("d"),
    ]
    nodes = graph.expand_neighborhood(corpus, ["center"], hops=1, max_nodes=3)
    assert len(nodes) <= 3


def test_expand_unions_edges_for_multiply_reached_node() -> None:
    """If A->B via related AND via contradicts, B's edges = {related, contradicts}."""
    corpus = [
        _m("a", related=["b"], contradicts=["b"]),
        _m("b"),
    ]
    nodes = graph.expand_neighborhood(corpus, ["a"], hops=1)
    assert nodes["b"].edges == {"related", "contradicts"}


def test_contradictions_for_returns_listed_memorias() -> None:
    corpus = [
        _m("a", contradicts=["b", "c"]),
        _m("b"),
        _m("c"),
        _m("d"),
    ]
    out = graph.contradictions_for(corpus, "a")
    assert {m.id for m in out} == {"b", "c"}


def test_contradictions_for_skips_dangling_ids() -> None:
    """A memory with contradictions referring to deleted memorias must not crash."""
    corpus = [_m("a", contradicts=["b", "deleted"])]
    out = graph.contradictions_for(corpus, "a")
    assert out == []  # b doesn't exist in corpus either


def test_contradictions_for_unknown_seed_returns_empty() -> None:
    corpus = [_m("a")]
    assert graph.contradictions_for(corpus, "nope") == []


def test_node_to_dict_round_trip_keys() -> None:
    node = graph.GraphNode(
        id="x",
        hop=2,
        edges={"related", "cotag"},
        name="X",
        description="desc",
        type="decision",
    )
    d = node.to_dict()
    assert d["id"] == "x"
    assert d["hop"] == 2
    assert d["edges"] == ["cotag", "related"]  # sorted
    assert d["name"] == "X"
    assert d["type"] == "decision"


def test_seeds_self_loop_does_not_recurse() -> None:
    """A memory whose ``related`` lists itself: the BFS must terminate cleanly."""
    corpus = [_m("a", related=["a"])]
    nodes = graph.expand_neighborhood(corpus, ["a"], hops=2)
    assert set(nodes) == {"a"}

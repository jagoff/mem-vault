"""Knowledge-graph traversal over the local vault — game-changer #2 (v0.6.0).

Until v0.5.x the ``related: [...]`` and ``contradicts: [...]`` fields in
each memory's frontmatter were *write-only* infrastructure: ``memory_save``
populated them via auto-link / auto-contradict, ``/graph`` UI dimly
visualized them, and ``memory_search`` ignored them entirely.

v0.6.0 turns those edges into first-class retrieval signals:

- :func:`expand_neighborhood` — BFS from a set of seed ids out to N hops,
  following ``related`` ∪ ``contradicts`` ∪ co-tag-neighbors. Returns
  every visited node with its hop distance + the edges that got us there.

- :func:`build_adjacency` — pure helper that flattens a corpus into
  ``adj[mid] = {(neighbor_id, edge_kind), ...}``. Memoized by the caller
  (``server._list_corpus`` already TTL-caches the corpus).

- :func:`contradictions_for` — pulls the ``contradicts`` list of one
  memory, used by ``server.search`` to inject "⚠ contradicts top
  result" hits even when they didn't make it into the semantic top-k
  organically.

The module is **pure**: no Ollama / Qdrant / FS calls. Inputs are
``Memory`` objects (already loaded by the caller); outputs are dicts.
This keeps it trivial to test (no fixtures), and lets the same code
back the MCP handler and the FastAPI ``/graph`` endpoint without
duplicating the BFS logic.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from mem_vault.storage import Memory

#: Edge kinds in priority order — earlier = stronger signal. The BFS
#: stamps ``edge_kind`` on each visited node so the caller (UI / MCP
#: tool) can render them differentiated (e.g. red lines for
#: ``contradicts``, green for ``related``, gray for ``cotag``).
EDGE_KINDS = ("contradicts", "related", "cotag")


@dataclass
class GraphNode:
    """One BFS-visited node + the edges that pulled it into the result.

    ``hop`` is the shortest distance (in edges) from any seed; ``edges``
    is the union of edge kinds that connect this node to its parents.
    A node reached via both ``related`` and ``cotag`` keeps both, so
    callers can decide whether co-tag-only is good enough for them.
    """

    id: str
    hop: int
    edges: set[str]
    name: str | None = None
    description: str | None = None
    type: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "hop": self.hop,
            "edges": sorted(self.edges),
            "name": self.name,
            "description": self.description,
            "type": self.type,
        }


def _normalize_tag(t: str) -> str:
    """Split ``project:foo`` style tags on the colon for cluster matching.

    Mirrors the rule used by the FastAPI ``/graph`` UI (cytoscape edges)
    so MCP traversal and visual graph stay coherent.
    """
    return t.split(":", 1)[-1].lower() if t else ""


def build_adjacency(
    corpus: Iterable[Memory],
    *,
    min_shared_tags: int = 2,
    include_cotag: bool = True,
) -> dict[str, dict[str, set[str]]]:
    """Flatten a corpus into ``adj[id] = {neighbor_id: {edge_kinds}}``.

    Edge kinds:

    - ``"related"`` — explicit ``related:`` frontmatter list (auto-link).
      *Asymmetric* on disk (A.related contains B but B.related might
      not contain A); we make the adjacency symmetric so traversal is
      undirected.
    - ``"contradicts"`` — explicit ``contradicts:`` frontmatter list
      (auto-contradict). Same asymmetry treatment as ``related``.
    - ``"cotag"`` — co-tag co-occurrence ≥ ``min_shared_tags``. Only
      computed when ``include_cotag=True``; the most expensive of the
      three, O(N^2) over the corpus.
    """
    by_id = {m.id: m for m in corpus}
    adj: dict[str, dict[str, set[str]]] = {mid: {} for mid in by_id}

    def _add_edge(a: str, b: str, kind: str) -> None:
        if a == b or a not in by_id or b not in by_id:
            return
        adj[a].setdefault(b, set()).add(kind)
        adj[b].setdefault(a, set()).add(kind)

    for m in by_id.values():
        for r in m.related or []:
            _add_edge(m.id, r, "related")
        for c in m.contradicts or []:
            _add_edge(m.id, c, "contradicts")

    if include_cotag and min_shared_tags >= 1:
        # Pre-normalize each memory's tag set — we'd otherwise pay this
        # cost in the inner loop, O(N^2) times.
        tag_sets: dict[str, set[str]] = {}
        for m in by_id.values():
            tag_sets[m.id] = {_normalize_tag(t) for t in (m.tags or []) if t}

        items = list(by_id.values())
        for i, m_a in enumerate(items):
            tags_a = tag_sets[m_a.id]
            if not tags_a:
                continue
            for m_b in items[i + 1 :]:
                tags_b = tag_sets[m_b.id]
                if not tags_b:
                    continue
                if len(tags_a & tags_b) >= min_shared_tags:
                    _add_edge(m_a.id, m_b.id, "cotag")
    return adj


def expand_neighborhood(
    corpus: Iterable[Memory],
    seed_ids: Iterable[str],
    *,
    hops: int = 1,
    min_shared_tags: int = 2,
    include_cotag: bool = True,
    edge_kinds: Iterable[str] | None = None,
    max_nodes: int = 50,
) -> dict[str, GraphNode]:
    """BFS up to ``hops`` from any of ``seed_ids``. Returns id → GraphNode.

    The seeds themselves are present at hop=0. Caller decides whether
    to filter them out (the MCP search handler does, the UI doesn't).

    ``edge_kinds`` lets callers narrow the traversal (e.g. only follow
    ``related`` for hard-line knowledge graph queries; follow ``cotag``
    too for "loosely related" exploration). When None, all kinds in
    :data:`EDGE_KINDS` are followed.

    ``max_nodes`` is a soft cap — once we cross it during BFS we stop
    enqueuing new neighbors. Keeps pathological seeds (a memory tagged
    with a generic ``project:`` slug) from blowing up the response.
    """
    if hops < 0:
        hops = 0
    allowed_kinds = set(edge_kinds or EDGE_KINDS)
    by_id = {m.id: m for m in corpus}
    adj = build_adjacency(
        by_id.values(),
        min_shared_tags=min_shared_tags,
        include_cotag=include_cotag and "cotag" in allowed_kinds,
    )

    visited: dict[str, GraphNode] = {}
    queue: deque[tuple[str, int]] = deque()
    for sid in seed_ids:
        if sid in by_id and sid not in visited:
            m = by_id[sid]
            visited[sid] = GraphNode(
                id=sid,
                hop=0,
                edges=set(),
                name=m.name,
                description=m.description,
                type=m.type,
            )
            queue.append((sid, 0))

    while queue:
        cur, depth = queue.popleft()
        if depth >= hops:
            continue
        if len(visited) >= max_nodes:
            break
        for nb, kinds in adj.get(cur, {}).items():
            kinds_keep = kinds & allowed_kinds
            if not kinds_keep:
                continue
            existing = visited.get(nb)
            if existing is None:
                m = by_id[nb]
                visited[nb] = GraphNode(
                    id=nb,
                    hop=depth + 1,
                    edges=set(kinds_keep),
                    name=m.name,
                    description=m.description,
                    type=m.type,
                )
                queue.append((nb, depth + 1))
                if len(visited) >= max_nodes:
                    break
            else:
                # Already visited at an equal-or-shallower hop — just
                # union the edge kinds so the caller sees every reason
                # this node is reachable.
                existing.edges |= kinds_keep
                if depth + 1 < existing.hop:
                    existing.hop = depth + 1
    return visited


def contradictions_for(corpus: Iterable[Memory], mem_id: str) -> list[Memory]:
    """Return the memories listed under ``contradicts:`` of ``mem_id``.

    Used by ``server.search`` to inject "this contradicts your top
    result" hits even when they don't make it into the semantic top-k
    organically — without this, a stale decision that contradicts the
    top hit silently fails to surface.

    Returns ``[]`` when ``mem_id`` is missing or has no contradictions.
    """
    by_id = {m.id: m for m in corpus}
    target = by_id.get(mem_id)
    if target is None:
        return []
    return [by_id[c] for c in (target.contradicts or []) if c in by_id]


__all__ = [
    "EDGE_KINDS",
    "GraphNode",
    "build_adjacency",
    "contradictions_for",
    "expand_neighborhood",
]

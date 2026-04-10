"""Query API for the Localizer agent to traverse the dependency graph."""

from __future__ import annotations

from collections import deque

from graph.model import DepGraph


def get_neighbors(
    graph: DepGraph,
    node_id: str,
    direction: str = "both",
    edge_kinds: set[str] | None = None,
) -> list[dict]:
    """Return neighbor nodes for a given node.

    Args:
        direction: "out", "in", or "both".
        edge_kinds: Filter to specific edge types (e.g. {"call", "include"}).
    """
    edges = []
    if direction in ("out", "both"):
        edges.extend(graph.adj.get(node_id, []))
    if direction in ("in", "both"):
        edges.extend(graph.rev.get(node_id, []))

    if edge_kinds:
        edges = [e for e in edges if e["kind"] in edge_kinds]

    # Collect neighbor node ids
    neighbor_ids = set()
    for e in edges:
        other = e["target"] if e["source"] == node_id else e["source"]
        neighbor_ids.add(other)

    return [graph.nodes[nid] for nid in neighbor_ids if nid in graph.nodes]


def get_callers(graph: DepGraph, func_id: str) -> list[dict]:
    """Functions that call this function (incoming call edges)."""
    edges = graph.rev.get(func_id, [])
    caller_ids = {e["source"] for e in edges if e["kind"] == "call"}
    return [graph.nodes[nid] for nid in caller_ids if nid in graph.nodes]


def get_callees(graph: DepGraph, func_id: str) -> list[dict]:
    """Functions called by this function (outgoing call edges)."""
    edges = graph.adj.get(func_id, [])
    callee_ids = {e["target"] for e in edges if e["kind"] == "call"}
    return [graph.nodes[nid] for nid in callee_ids if nid in graph.nodes]


def get_file_functions(graph: DepGraph, file_id: str) -> list[dict]:
    """All function nodes in a given file."""
    return [
        n for n in graph.nodes.values()
        if n["kind"] == "function" and n["path"] == file_id
    ]


def expand_hop(
    graph: DepGraph,
    seed_nodes: set[str],
    hops: int = 1,
    edge_kinds: set[str] | None = None,
    min_confidence: float = 0.0,
) -> set[str]:
    """BFS expansion from seed nodes, returning all nodes within N hops.

    Args:
        min_confidence: Only traverse edges with confidence >= this threshold.
    """
    visited = set(seed_nodes)
    frontier = set(seed_nodes)

    for _ in range(hops):
        next_frontier = set()
        for nid in frontier:
            for edge in graph.adj.get(nid, []) + graph.rev.get(nid, []):
                if edge_kinds and edge["kind"] not in edge_kinds:
                    continue
                if edge["confidence"] < min_confidence:
                    continue
                other = edge["target"] if edge["source"] == nid else edge["source"]
                if other not in visited:
                    next_frontier.add(other)
                    visited.add(other)
        frontier = next_frontier

    return visited


def get_test_files(graph: DepGraph, source_file: str) -> list[str]:
    """Test files that exercise a given source file (via test edges)."""
    # Test edges point FROM test file TO source file
    test_ids = set()
    for edge in graph.rev.get(source_file, []):
        if edge["kind"] == "test":
            test_ids.add(edge["source"])
    return sorted(test_ids)

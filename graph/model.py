"""DepGraph data structure for code dependency graphs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


class DepGraph:
    """In-memory code dependency graph with adjacency indexes."""

    def __init__(self, repo: str = "", commit: str = ""):
        self.repo = repo
        self.commit = commit
        self.nodes: dict[str, dict] = {}  # id -> node dict
        self.edges: list[dict] = []
        self.adj: dict[str, list[dict]] = defaultdict(list)  # outgoing
        self.rev: dict[str, list[dict]] = defaultdict(list)  # incoming

    # ── Node helpers ──────────────────────────────────────────────

    def add_file_node(self, path: str) -> str:
        """Add a file node. Returns the node id."""
        nid = path
        if nid not in self.nodes:
            self.nodes[nid] = {"id": nid, "kind": "file", "path": path}
        return nid

    def add_func_node(
        self, path: str, name: str, start_line: int, end_line: int
    ) -> str:
        """Add a function node. Returns the node id."""
        nid = f"{path}::{name}"
        # Handle duplicate function names in same file (rare in C)
        if nid in self.nodes:
            nid = f"{path}::{name}@{start_line}"
        self.nodes[nid] = {
            "id": nid,
            "kind": "function",
            "path": path,
            "name": name,
            "start_line": start_line,
            "end_line": end_line,
        }
        return nid

    # ── Edge helpers ──────────────────────────────────────────────

    def add_edge(
        self,
        source: str,
        target: str,
        kind: str,
        weight: float = 1.0,
        confidence: float = 1.0,
    ) -> None:
        """Add a directed edge and update adjacency indexes."""
        edge = {
            "source": source,
            "target": target,
            "kind": kind,
            "weight": weight,
            "confidence": confidence,
        }
        self.edges.append(edge)
        self.adj[source].append(edge)
        self.rev[target].append(edge)

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "commit": self.commit,
            "nodes": self.nodes,
            "edges": self.edges,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> DepGraph:
        g = cls(repo=data.get("repo", ""), commit=data.get("commit", ""))
        g.nodes = data.get("nodes", {})
        g.edges = data.get("edges", [])
        # Rebuild adjacency indexes
        for edge in g.edges:
            g.adj[edge["source"]].append(edge)
            g.rev[edge["target"]].append(edge)
        return g

    @classmethod
    def from_json(cls, text: str) -> DepGraph:
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: Path | str) -> DepGraph:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # ── Stats ─────────────────────────────────────────────────────

    def summary(self) -> dict:
        file_nodes = sum(1 for n in self.nodes.values() if n["kind"] == "file")
        func_nodes = sum(1 for n in self.nodes.values() if n["kind"] == "function")
        edge_kinds = defaultdict(int)
        for e in self.edges:
            edge_kinds[e["kind"]] += 1
        return {
            "repo": self.repo,
            "commit": self.commit[:10] if self.commit else "",
            "files": file_nodes,
            "functions": func_nodes,
            "edges": dict(edge_kinds),
            "total_edges": len(self.edges),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"DepGraph({s['repo']}, {s['files']} files, "
            f"{s['functions']} funcs, {s['total_edges']} edges)"
        )

#!/usr/bin/env python3
"""
stats_per_repo.py — Sanity-check statistics for all built DepGraph JSON files.

Usage:
    python scripts/stats_per_repo.py                  # reads graphs/ dir
    python scripts/stats_per_repo.py --graphs <path>  # custom graphs dir
    python scripts/stats_per_repo.py --json           # machine-readable output

Checks performed (per repo):
    1. At least 1 graph loaded successfully
    2. Every graph has > 0 file nodes
    3. Every graph has > 0 function nodes
    4. At least one edge type is present
    5. All edge confidence values are in [0, 1]
    6. All edge source/target node IDs exist in nodes dict
    7. No self-loop edges
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graph.model import DepGraph


# ── Helpers ───────────────────────────────────────────────────────


def load_graphs(graphs_dir: Path) -> dict[str, list[tuple[str, DepGraph]]]:
    """Load all JSON graphs and group them by repo slug."""
    repo_graphs: dict[str, list[tuple[str, DepGraph]]] = defaultdict(list)
    for path in sorted(graphs_dir.glob("*.json")):
        try:
            g = DepGraph.load(path)
            repo = g.repo or _infer_repo(path.stem)
            repo_graphs[repo].append((path.stem, g))
        except Exception as e:
            print(f"  WARN  Could not load {path.name}: {e}", file=sys.stderr)
    return dict(repo_graphs)


def _infer_repo(instance_id: str) -> str:
    """Fallback: infer repo slug from instance id like 'facebook__zstd-123'."""
    if "__" in instance_id:
        owner_name = instance_id.split("-")[0]
        return owner_name.replace("__", "/")
    return "unknown"


# ── Per-graph checks ──────────────────────────────────────────────


def check_graph(instance_id: str, g: DepGraph) -> list[str]:
    """Run sanity checks on a single graph. Returns list of failure messages."""
    failures = []

    s = g.summary()
    if s["files"] == 0:
        failures.append(f"{instance_id}: 0 file nodes")
    if s["functions"] == 0:
        failures.append(f"{instance_id}: 0 function nodes")
    if s["total_edges"] == 0:
        failures.append(f"{instance_id}: 0 edges")

    for e in g.edges:
        conf = e.get("confidence", -1)
        if not (0.0 <= conf <= 1.0):
            failures.append(f"{instance_id}: edge confidence out of range: {conf}")
            break  # one report per graph

    node_ids = set(g.nodes.keys())
    for e in g.edges:
        if e["source"] not in node_ids:
            failures.append(f"{instance_id}: dangling edge source '{e['source']}'")
            break
        if e["target"] not in node_ids:
            failures.append(f"{instance_id}: dangling edge target '{e['target']}'")
            break
        if e["source"] == e["target"]:
            failures.append(f"{instance_id}: self-loop on '{e['source']}'")
            break

    return failures


# ── Per-repo aggregation ──────────────────────────────────────────


def aggregate_repo_stats(graphs: list[tuple[str, DepGraph]]) -> dict:
    """Compute aggregate stats across all graphs for one repo."""
    files_list, funcs_list, edges_list = [], [], []
    edge_kind_totals: dict[str, int] = defaultdict(int)
    all_failures: list[str] = []

    for instance_id, g in graphs:
        s = g.summary()
        files_list.append(s["files"])
        funcs_list.append(s["functions"])
        edges_list.append(s["total_edges"])
        for kind, count in s["edges"].items():
            edge_kind_totals[kind] += count
        all_failures.extend(check_graph(instance_id, g))

    n = len(graphs)

    def avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0

    return {
        "instances": n,
        "files": {"min": min(files_list), "max": max(files_list), "avg": avg(files_list)},
        "functions": {"min": min(funcs_list), "max": max(funcs_list), "avg": avg(funcs_list)},
        "edges": {"min": min(edges_list), "max": max(edges_list), "avg": avg(edges_list)},
        "edge_kinds": dict(edge_kind_totals),
        "failures": all_failures,
        "checks_passed": sum(1 for _, g in graphs if not check_graph(_, g)),
        "checks_failed": len(all_failures),
    }


# ── Reporting ─────────────────────────────────────────────────────


def print_repo_report(repo: str, stats: dict) -> None:
    n = stats["instances"]
    ok = "✓" if stats["checks_failed"] == 0 else "✗"
    print(f"\n{'='*60}")
    print(f"  {ok}  {repo}  ({n} instance{'s' if n != 1 else ''})")
    print(f"{'='*60}")

    def fmt_range(d):
        return f"{d['min']}–{d['max']}  (avg {d['avg']})"

    print(f"  Files per graph    : {fmt_range(stats['files'])}")
    print(f"  Functions per graph: {fmt_range(stats['functions'])}")
    print(f"  Edges per graph    : {fmt_range(stats['edges'])}")
    print()
    print("  Edge type totals (across all instances):")
    for kind in ("include", "call", "type_use", "test"):
        count = stats["edge_kinds"].get(kind, 0)
        print(f"    {kind:<10} {count:>8,}")
    other_kinds = {k: v for k, v in stats["edge_kinds"].items()
                   if k not in ("include", "call", "type_use", "test")}
    for kind, count in sorted(other_kinds.items()):
        print(f"    {kind:<10} {count:>8,}  (unexpected)")

    print()
    if stats["failures"]:
        print(f"  FAILURES ({len(stats['failures'])}):")
        for msg in stats["failures"][:10]:
            print(f"    ✗  {msg}")
        if len(stats["failures"]) > 10:
            print(f"    ... and {len(stats['failures']) - 10} more")
    else:
        print("  All sanity checks passed.")


def print_summary(all_stats: dict[str, dict]) -> None:
    total_instances = sum(s["instances"] for s in all_stats.values())
    total_failures = sum(s["checks_failed"] for s in all_stats.values())
    repos_clean = sum(1 for s in all_stats.values() if s["checks_failed"] == 0)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Repos:     {len(all_stats)}")
    print(f"  Instances: {total_instances}")
    print(f"  Repos clean: {repos_clean}/{len(all_stats)}")
    if total_failures:
        print(f"  Total failures: {total_failures}")
    else:
        print("  All checks passed ✓")


# ── Entry point ───────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Sanity-check stats per repo")
    parser.add_argument(
        "--graphs", type=Path, default=PROJECT_ROOT / "graphs",
        help="Path to graphs directory (default: <project_root>/graphs)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output machine-readable JSON instead of human-readable report"
    )
    args = parser.parse_args()

    graphs_dir: Path = args.graphs
    if not graphs_dir.exists():
        print(f"ERROR: graphs directory not found: {graphs_dir}", file=sys.stderr)
        print("Run 'python scripts/build_all_graphs.py' first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading graphs from: {graphs_dir}")
    repo_graphs = load_graphs(graphs_dir)

    if not repo_graphs:
        print("No graphs found. Run build_all_graphs.py first.")
        sys.exit(1)

    all_stats: dict[str, dict] = {}
    for repo, graphs in sorted(repo_graphs.items()):
        all_stats[repo] = aggregate_repo_stats(graphs)

    if args.json:
        print(json.dumps(all_stats, indent=2))
        return

    for repo, stats in all_stats.items():
        print_repo_report(repo, stats)

    print_summary(all_stats)

    any_failures = any(s["checks_failed"] > 0 for s in all_stats.values())
    sys.exit(1 if any_failures else 0)


if __name__ == "__main__":
    main()

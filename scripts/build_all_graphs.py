#!/usr/bin/env python3
"""Build dependency graphs for all SWE-bench-C instances."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graph.builder import build_graph
from graph.model import DepGraph
from graph.repo_checkout import ensure_repo_at_commit

GRAPHS_DIR = PROJECT_ROOT / "graphs"
GRAPHS_DIR.mkdir(exist_ok=True)

# SWE-bench-C dataset: load from local JSON or HuggingFace
SWEBENCH_C_PATH = (
    PROJECT_ROOT.parent / "SWE-bench-c"
    / "gold.validate-gold.json"
)


def load_instances() -> list[dict]:
    """Load SWE-bench-C instances from local dataset file."""
    # Try local JSON files first
    for candidate in [
        SWEBENCH_C_PATH,
        PROJECT_ROOT / "data" / "swe-bench-c.json",
    ]:
        if candidate.exists():
            print(f"Loading instances from {candidate}")
            with open(candidate, encoding="utf-8") as f:
                return json.load(f)

    # Try HuggingFace datasets
    try:
        from datasets import load_dataset
        ds = load_dataset("xingyaoww/swe-bench-c", split="test")
        return list(ds)
    except ImportError:
        pass

    print("ERROR: Cannot find SWE-bench-C dataset.")
    print(f"  Looked for: {SWEBENCH_C_PATH}")
    print("  Install 'datasets' package or place JSON file manually.")
    sys.exit(1)


def build_for_instance(instance: dict, repos_dir: Path) -> tuple[str, DepGraph]:
    """Build graph for a single instance. Returns (instance_id, graph)."""
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]

    # Check cache
    out_path = GRAPHS_DIR / f"{instance_id}.json"
    if out_path.exists():
        return instance_id, DepGraph.load(out_path)

    # Checkout and build
    repo_root = ensure_repo_at_commit(repo, base_commit, repos_dir)
    graph = build_graph(repo_root)
    graph.repo = repo
    graph.commit = base_commit

    # Save
    graph.save(out_path)
    return instance_id, graph


def main():
    repos_dir = PROJECT_ROOT / "repos"
    instances = load_instances()
    print(f"Loaded {len(instances)} instances")

    # Deduplicate by (repo, base_commit) to avoid rebuilding
    seen_commits: dict[tuple[str, str], str] = {}  # (repo, commit) -> instance_id
    graphs_built = 0
    graphs_cached = 0

    for inst in instances:
        instance_id = inst["instance_id"]
        repo = inst["repo"]
        commit = inst["base_commit"]
        key = (repo, commit)

        out_path = GRAPHS_DIR / f"{instance_id}.json"

        if out_path.exists():
            graphs_cached += 1
            continue

        if key in seen_commits:
            # Same repo+commit, reuse the graph
            src = GRAPHS_DIR / f"{seen_commits[key]}.json"
            if src.exists():
                # Copy graph with updated metadata
                g = DepGraph.load(src)
                g.save(out_path)
                graphs_cached += 1
                continue

        t0 = time.time()
        try:
            _, graph = build_for_instance(inst, repos_dir)
            elapsed = time.time() - t0
            s = graph.summary()
            print(
                f"  [{instance_id}] {s['files']} files, "
                f"{s['functions']} funcs, {s['total_edges']} edges "
                f"({elapsed:.1f}s)"
            )
            graphs_built += 1
            seen_commits[key] = instance_id
        except Exception as e:
            print(f"  [{instance_id}] FAILED: {e}")

    print(f"\nDone: {graphs_built} built, {graphs_cached} cached")


if __name__ == "__main__":
    main()

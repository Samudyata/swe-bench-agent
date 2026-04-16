#!/usr/bin/env python3
"""Smoke tests for the graph infrastructure."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graph.builder import build_graph
from graph.model import DepGraph
from graph.query import (
    get_callers, get_callees, get_file_functions,
    expand_hop, get_test_files, get_neighbors,
)
from graph.scoring import keyword_search, evidence_subgraph, extract_keywords

REPO_PATH = PROJECT_ROOT / "repos" / "jqlang__jq"

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        failed += 1


def main():
    global passed, failed

    if not REPO_PATH.exists():
        print(f"ERROR: Clone jq first: git clone https://github.com/jqlang/jq.git {REPO_PATH}")
        sys.exit(1)

    # ── 1. Build graph ────────────────────────────────────────────
    print("\n=== Build ===")
    g = build_graph(REPO_PATH)
    g.repo = "jqlang/jq"
    g.commit = "HEAD"
    s = g.summary()
    print(f"  {s}")

    check("has file nodes", s["files"] > 20, f"got {s['files']}")
    check("has function nodes", s["functions"] > 100, f"got {s['functions']}")
    check("has include edges", s["edges"].get("include", 0) > 10)
    check("has call edges", s["edges"].get("call", 0) > 100)
    check("has test edges", s["edges"].get("test", 0) > 0)

    # ── 2. Node structure ─────────────────────────────────────────
    print("\n=== Node structure ===")
    file_node = g.nodes.get("src/jv.c")
    check("file node exists (src/jv.c)", file_node is not None)
    if file_node:
        check("file node has kind=file", file_node["kind"] == "file")
        check("file node has path", file_node["path"] == "src/jv.c")

    func_node = g.nodes.get("src/jv.c::jv_copy")
    check("func node exists (jv_copy)", func_node is not None)
    if func_node:
        check("func node has kind=function", func_node["kind"] == "function")
        check("func node has name", func_node["name"] == "jv_copy")
        check("func node has start_line", func_node["start_line"] > 0)
        check("func node has end_line", func_node["end_line"] >= func_node["start_line"])

    # ── 3. Edge structure ─────────────────────────────────────────
    print("\n=== Edge structure ===")
    for e in g.edges[:1]:
        check("edge has source", "source" in e)
        check("edge has target", "target" in e)
        check("edge has kind", "kind" in e)
        check("edge has confidence", "confidence" in e and 0 <= e["confidence"] <= 1)

    # Check include edges point between files
    inc_edges = [e for e in g.edges if e["kind"] == "include"]
    if inc_edges:
        e = inc_edges[0]
        check("include edge source is file node", g.nodes[e["source"]]["kind"] == "file")
        check("include edge target is file node", g.nodes[e["target"]]["kind"] == "file")

    # Check call edges point between functions
    call_edges = [e for e in g.edges if e["kind"] == "call"]
    if call_edges:
        e = call_edges[0]
        check("call edge source is function", g.nodes[e["source"]]["kind"] == "function")
        check("call edge target is function", g.nodes[e["target"]]["kind"] == "function")

    # ── 4. Confidence scoring ─────────────────────────────────────
    print("\n=== Confidence scoring ===")
    conf_values = {e["kind"]: set() for e in g.edges}
    for e in g.edges:
        conf_values[e["kind"]].add(e["confidence"])

    check("include edges have conf=1.0", conf_values.get("include") == {1.0})
    check("call edges have conf <= 0.95", all(c <= 0.95 for c in conf_values.get("call", {1})))
    check("test edges have conf=0.8", conf_values.get("test", set()) <= {0.8})

    # Ambiguous calls may have lower confidence (repo-dependent, not required)
    low_conf_calls = [e for e in g.edges if e["kind"] == "call" and e["confidence"] < 0.5]
    print(f"  INFO  ambiguous calls (conf < 0.5): {len(low_conf_calls)}")

    # ── 5. Query API ──────────────────────────────────────────────
    print("\n=== Query API ===")
    callers = get_callers(g, "src/jv.c::jv_copy")
    check("get_callers returns results", len(callers) > 0, f"got {len(callers)}")
    check("callers are function nodes", all(c["kind"] == "function" for c in callers))

    callees = get_callees(g, "src/execute.c::jq_init")
    check("get_callees returns results", len(callees) > 0, f"got {len(callees)}")

    funcs = get_file_functions(g, "src/jv.c")
    check("get_file_functions returns results", len(funcs) > 10, f"got {len(funcs)}")
    check("all funcs in correct file", all(f["path"] == "src/jv.c" for f in funcs))

    tests = get_test_files(g, "src/jv.c")
    check("get_test_files returns results", len(tests) > 0, f"got {len(tests)}")
    check("test files contain 'test'", all("test" in t.lower() or "fuzz" in t.lower() for t in tests))

    # expand_hop
    seeds = {"src/jv.c::jv_copy"}
    expanded = expand_hop(g, seeds, hops=1)
    check("expand_hop grows the set", len(expanded) > len(seeds), f"got {len(expanded)}")
    check("seeds are in expanded set", seeds.issubset(expanded))

    # expand_hop with min_confidence
    expanded_strict = expand_hop(g, seeds, hops=1, min_confidence=0.9)
    expanded_loose = expand_hop(g, seeds, hops=1, min_confidence=0.1)
    check("higher confidence = fewer nodes",
          len(expanded_strict) <= len(expanded_loose),
          f"strict={len(expanded_strict)} loose={len(expanded_loose)}")

    # get_neighbors
    neighbors = get_neighbors(g, "src/jv.c", edge_kinds={"include"})
    check("get_neighbors with include filter", len(neighbors) > 0)

    # ── 6. Scoring / keyword search ───────────────────────────────
    print("\n=== Scoring ===")
    kws = extract_keywords("buffer overflow in jv_parse function")
    check("extract_keywords works", "buffer" in kws and "jv_parse" in kws, f"got {kws}")

    results = keyword_search(g, ["jv_parse"])
    check("keyword_search returns results", len(results) > 0)
    top_id, top_score = results[0]
    check("top result contains keyword", "jv_parse" in top_id.lower(), f"got {top_id}")
    check("exact match scores highest (3.0)", top_score >= 3.0, f"got {top_score}")

    subgraph = evidence_subgraph(g, ["jv_parse", "buffer"], top_k=3, hops=1)
    check("evidence_subgraph returns nodes", len(subgraph) > 3, f"got {len(subgraph)}")

    # ── 7. Serialization round-trip ───────────────────────────────
    print("\n=== Serialization ===")
    import tempfile, os
    tmp = tempfile.mkstemp(suffix=".json")
    g.save(tmp)
    g2 = DepGraph.load(tmp)
    os.remove(tmp)

    check("round-trip preserves nodes", len(g.nodes) == len(g2.nodes))
    check("round-trip preserves edges", len(g.edges) == len(g2.edges))
    check("round-trip preserves repo", g.repo == g2.repo)
    check("round-trip rebuilds adj index", len(g2.adj) > 0)

    # Verify a specific query works on reloaded graph
    callers2 = get_callers(g2, "src/jv.c::jv_copy")
    check("query works on reloaded graph", len(callers2) == len(callers))

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*50}")
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

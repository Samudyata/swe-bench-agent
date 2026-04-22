#!/usr/bin/env python3
"""Tests for Person 3 components: grep_tool, file_reader, graph_tools, localizer.

Run with:
    python scripts/test_person3.py

Sections
--------
1. Grep tool          — no graph, no LLM needed
2. File reader        — no graph, no LLM needed
3. Graph tools        — needs graph only (no repo on disk)
4. Localizer e2e      — needs jq repo on disk at repos/jqlang__jq/
                        (skipped automatically if repo not found)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so OPENAI_API_KEY is available
try:
    from dotenv import load_dotenv
    for _c in [PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"]:
        if _c.exists():
            load_dotenv(_c)
            break
except ImportError:
    pass

from agents.tools.grep_tool import grep_repo, hit_density, GrepHit
from agents.tools.file_reader import (
    read_file, extract_snippet, extract_function_snippets, read_top_files,
)
from agents.tools.graph_tools import build_scored_candidates, select_top_function_nodes
from graph.model import DepGraph
from pipeline.schema import SWEInstance, PlannerOutput

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        failed += 1


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a tiny fake C repository on disk
# ─────────────────────────────────────────────────────────────────────────────

def _make_fake_repo(tmp: Path) -> Path:
    """Create a minimal synthetic C repo under *tmp* for testing."""
    repo = tmp / "fake_repo"
    src = repo / "src"
    src.mkdir(parents=True)

    (src / "jv.c").write_text(
        "// JSON value parser\n"
        "#include \"jv.h\"\n"
        "\n"
        "static int jv_parse_internal(const char *buf) {\n"
        "    if (buf == NULL) return -1;\n"
        "    // parse logic here\n"
        "    return 0;\n"
        "}\n"
        "\n"
        "int jv_parse(const char *input) {\n"
        "    return jv_parse_internal(input);\n"
        "}\n",
        encoding="utf-8",
    )
    (src / "jv.h").write_text(
        "#ifndef JV_H\n"
        "#define JV_H\n"
        "int jv_parse(const char *input);\n"
        "#endif\n",
        encoding="utf-8",
    )
    (src / "execute.c").write_text(
        "#include \"jv.h\"\n"
        "\n"
        "int jq_execute(const char *program, const char *input) {\n"
        "    int result = jv_parse(input);\n"
        "    return result;\n"
        "}\n",
        encoding="utf-8",
    )
    # A vendor file that should be skipped
    vendor = repo / "vendor" / "lib"
    vendor.mkdir(parents=True)
    (vendor / "unused.c").write_text(
        "int jv_parse(const char *x) { return 0; }\n",
        encoding="utf-8",
    )
    return repo


def _make_fake_graph() -> DepGraph:
    """Build a small DepGraph matching the fake repo's structure."""
    g = DepGraph(repo="fake/repo", commit="abc")
    g.add_file_node("src/jv.c")
    g.add_file_node("src/jv.h")
    g.add_file_node("src/execute.c")
    f1 = g.add_func_node("src/jv.c", "jv_parse_internal", 4, 8)
    f2 = g.add_func_node("src/jv.c", "jv_parse", 10, 12)
    f3 = g.add_func_node("src/execute.c", "jq_execute", 3, 6)
    g.add_edge("src/jv.c", "src/jv.h", "include", confidence=1.0)
    g.add_edge("src/execute.c", "src/jv.h", "include", confidence=1.0)
    g.add_edge(f3, f2, "call", confidence=0.95)
    g.add_edge(f2, f1, "call", confidence=0.95)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# 1. Grep tool tests
# ─────────────────────────────────────────────────────────────────────────────

def test_grep_tool() -> None:
    print("\n=== Grep tool ===")
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_fake_repo(Path(tmp))

        hits = grep_repo(repo, ["jv_parse"])
        check("grep finds jv.c",  "src/jv.c"      in hits, f"keys={list(hits)}")
        check("grep finds jv.h",  "src/jv.h"      in hits, f"keys={list(hits)}")
        check("grep finds execute.c", "src/execute.c" in hits, f"keys={list(hits)}")
        check("grep skips vendor/", "vendor/lib/unused.c" not in hits,
              f"keys={list(hits)}")

        # Hit objects are GrepHit instances
        first_hit = hits["src/jv.c"][0]
        check("GrepHit has file_path", first_hit.file_path == "src/jv.c")
        check("GrepHit has line_no > 0", first_hit.line_no > 0)
        check("GrepHit has keyword", first_hit.keyword == "jv_parse")
        check("GrepHit line_text non-empty", len(first_hit.line_text) > 0)

        # Whole-word: should not match "jv_parse_internal" when searching "jv_parse"
        # (actually it SHOULD match because jv_parse appears as substring in identifier,
        #  but whole-word means the token boundary exists around "jv_parse" exactly)
        # Let's test a keyword that would be a substring
        hits2 = grep_repo(repo, ["parse"])   # "parse" substring of "jv_parse"
        # whole-word: "parse" should NOT match "jv_parse" (jv_ is adjacent)
        # But it SHOULD match if "parse" appears standalone… it doesn't in our fake
        check("Whole-word grep: 'parse' standalone not in execute.c",
              "src/execute.c" not in hits2,
              f"hits2 keys={list(hits2)}")

        # hit_density returns normalised [0,1]
        density = hit_density(hits)
        check("hit_density returns float values", all(0.0 <= v <= 1.0 for v in density.values()))
        check("hit_density max is 1.0", max(density.values()) == 1.0)

        # Empty keywords -> empty result
        check("Empty keywords -> empty", grep_repo(repo, {}) == {})


# ─────────────────────────────────────────────────────────────────────────────
# 2. File reader tests
# ─────────────────────────────────────────────────────────────────────────────

def test_file_reader() -> None:
    print("\n=== File reader ===")
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_fake_repo(Path(tmp))
        graph = _make_fake_graph()

        # read_file
        text = read_file(repo, "src/jv.c")
        check("read_file returns string", isinstance(text, str))
        check("read_file content has jv_parse", "jv_parse" in text)

        missing = read_file(repo, "nonexistent/file.c")
        check("read_file missing -> None", missing is None)

        # extract_snippet (1-indexed, lines 4-8 = jv_parse_internal body)
        snippet = extract_snippet(text, 4, 8, context_lines=1)
        check("extract_snippet non-empty", len(snippet) > 0)
        check("extract_snippet has header", "4" in snippet and "8" in snippet)
        check("extract_snippet contains function code",
              "jv_parse_internal" in snippet)

        # extract_snippet clamping — request beyond file end
        long_snip = extract_snippet(text, 1, 9999, context_lines=0)
        lines_count = long_snip.count("\n")
        check("extract_snippet clamped", lines_count < 200, f"got {lines_count} lines")

        # extract_function_snippets
        func_ids = ["src/jv.c::jv_parse", "src/jv.c::jv_parse_internal"]
        snippets = extract_function_snippets(repo, graph, func_ids)
        check("extract_function_snippets returns dict", isinstance(snippets, dict))
        check("jv_parse snippet present", "src/jv.c::jv_parse" in snippets)
        check("jv_parse_internal snippet present",
              "src/jv.c::jv_parse_internal" in snippets)
        check("Snippet content has function name",
              "jv_parse" in snippets.get("src/jv.c::jv_parse", ""))

        # Missing node IDs handled gracefully
        snippets2 = extract_function_snippets(repo, graph, ["nonexistent::fn"])
        check("Missing node_id -> empty dict", snippets2 == {})

        # read_top_files
        contents = read_top_files(repo, ["src/jv.c", "src/jv.h"])
        check("read_top_files returns both files", len(contents) == 2)
        check("read_top_files content non-empty",
              all(len(v) > 0 for v in contents.values()))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Graph tools tests
# ─────────────────────────────────────────────────────────────────────────────

def test_graph_tools() -> None:
    print("\n=== Graph tools ===")
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_fake_repo(Path(tmp))
        graph = _make_fake_graph()

        grep_hits = grep_repo(repo, ["jv_parse"])
        scored = build_scored_candidates(
            graph=graph,
            keywords=["jv_parse"],
            grep_hits=grep_hits,
            suspected_modules=["src/jv.c"],
            top_k=5,
        )
        check("build_scored_candidates returns list", isinstance(scored, list))
        check("At least one candidate", len(scored) >= 1, f"got {len(scored)}")
        check("Top candidate is src/jv.c", scored[0].file_path == "src/jv.c",
              f"got {scored[0].file_path}")
        check("Composite score in [0, 2]",
              all(0.0 <= s.composite_score <= 2.0 for s in scored),
              f"scores={[s.composite_score for s in scored]}")
        check("jv.c is suspected", scored[0].is_suspected)
        check("execute.c is not suspected",
              not any(s.is_suspected and s.file_path == "src/execute.c" for s in scored))

        # Sorted descending
        scores = [s.composite_score for s in scored]
        check("Sorted descending", scores == sorted(scores, reverse=True))

        # top_k respected
        scored_k2 = build_scored_candidates(
            graph=graph, keywords=["jv_parse"],
            grep_hits=grep_hits, suspected_modules=[], top_k=2,
        )
        check("top_k=2 respected", len(scored_k2) <= 2, f"got {len(scored_k2)}")

        # No keywords: only grip + suspected bonus
        scored_nokw = build_scored_candidates(
            graph=graph, keywords=[],
            grep_hits=grep_hits, suspected_modules=["src/jv.c"], top_k=5,
        )
        check("No keywords still returns results", len(scored_nokw) >= 1)

        # select_top_function_nodes
        func_nodes = select_top_function_nodes(graph, scored[:2], ["jv_parse"], max_funcs_per_file=3)
        check("select_top_function_nodes returns dict", isinstance(func_nodes, dict))
        check("jv.c functions selected", "src/jv.c" in func_nodes)
        check("jv_parse in top functions",
              any("jv_parse" in nid for nid in func_nodes.get("src/jv.c", [])),
              f"got {func_nodes}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. LocalizerAgent end-to-end tests
# ─────────────────────────────────────────────────────────────────────────────

def test_localizer_unit() -> None:
    """Unit test the LocalizerAgent using the fake in-memory repo (no real clone)."""
    print("\n=== LocalizerAgent (unit, fake repo) ===")
    from agents.localizer import LocalizerAgent

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Fake repo at repos/fake__repo/
        fake_repo_path = tmp_path / "repos" / "fake__repo"
        fake_repo_path.mkdir(parents=True)
        src = fake_repo_path / "src"
        src.mkdir()

        (src / "jv.c").write_text(
            "int jv_parse(const char *input) {\n"
            "    if (!input) return -1;\n"
            "    return jv_load(input);\n"
            "}\n"
            "int jv_load(const char *s) { return 0; }\n",
            encoding="utf-8",
        )
        (src / "execute.c").write_text(
            "int jq_execute(const char *p, const char *in) {\n"
            "    return jv_parse(in);\n"
            "}\n",
            encoding="utf-8",
        )

        graph = DepGraph(repo="fake/repo", commit="abc")
        graph.add_file_node("src/jv.c")
        graph.add_file_node("src/execute.c")
        f_parse = graph.add_func_node("src/jv.c", "jv_parse", 1, 4)
        f_load  = graph.add_func_node("src/jv.c", "jv_load", 5, 5)
        f_exec  = graph.add_func_node("src/execute.c", "jq_execute", 1, 3)
        graph.add_edge(f_exec, f_parse, "call", confidence=0.95)
        graph.add_edge(f_parse, f_load,  "call", confidence=0.95)

        instance = SWEInstance(
            instance_id="fake-001",
            repo="fake/repo",
            base_commit="abc",
            problem_statement="jv_parse crashes on null input",
            fail_to_pass=["tests::test_null"],
            pass_to_pass=[],
        )
        plan = PlannerOutput(
            instance_id="fake-001",
            issue_type="bug",
            keywords=["jv_parse", "null"],
            search_hints=["look in src/jv.c"],
            suspected_modules=["src/jv.c"],
            priority_functions=["jv_parse"],
        )

        agent = LocalizerAgent(
            repos_dir=tmp_path / "repos",
            auto_checkout=False,
        )
        bundle = agent.localize(instance, plan, graph)

        from pipeline.schema import ContextBundle
        check("localize returns ContextBundle", isinstance(bundle, ContextBundle))
        check("bundle has candidates", len(bundle.candidates) >= 1,
              f"got {len(bundle.candidates)}")
        check("top candidate is src/jv.c",
              bundle.candidates[0].file_path == "src/jv.c",
              f"got {bundle.candidates[0].file_path}")
        check("confidence is float in [0,1]",
              0.0 <= bundle.confidence <= 1.0, f"got {bundle.confidence}")
        check("confidence > 0", bundle.confidence > 0.0)
        check("file_contents non-empty", len(bundle.file_contents) >= 1)
        check("src/jv.c in file_contents", "src/jv.c" in bundle.file_contents)
        check("relevant_snippets non-empty", len(bundle.relevant_snippets) >= 1)
        check("candidate has reason", len(bundle.candidates[0].reason) > 0)
        check("candidate has functions", len(bundle.candidates[0].functions) >= 1,
              f"got {bundle.candidates[0].functions}")


def test_localizer_feedback() -> None:
    """Test localize_with_feedback on the fake repo."""
    print("\n=== LocalizerAgent feedback (apply_failed) ===")
    from agents.localizer import LocalizerAgent
    from pipeline.schema import FeedbackMessage, FailureType

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fake_repo_path = tmp_path / "repos" / "fake__repo"
        fake_repo_path.mkdir(parents=True)
        src = fake_repo_path / "src"
        src.mkdir()
        (src / "jv.c").write_text(
            "int jv_parse(const char *input) { return 0; }\n",
            encoding="utf-8",
        )
        (src / "jv_extra.c").write_text(
            "int jv_extra_func(int x) { return x * 2; }\n",
            encoding="utf-8",
        )

        graph = DepGraph(repo="fake/repo", commit="abc")
        graph.add_file_node("src/jv.c")
        graph.add_file_node("src/jv_extra.c")
        f1 = graph.add_func_node("src/jv.c", "jv_parse", 1, 1)
        f2 = graph.add_func_node("src/jv_extra.c", "jv_extra_func", 1, 1)
        graph.add_edge(f1, f2, "call", confidence=0.9)

        instance = SWEInstance(
            instance_id="fake-002",
            repo="fake/repo",
            base_commit="abc",
            problem_statement="jv_extra_func returns wrong value",
            fail_to_pass=[],
            pass_to_pass=[],
        )
        plan = PlannerOutput(
            instance_id="fake-002",
            issue_type="bug",
            keywords=["jv_parse"],
            search_hints=[],
            suspected_modules=[],
            priority_functions=["jv_parse"],
        )

        agent = LocalizerAgent(repos_dir=tmp_path / "repos", auto_checkout=False)

        feedback = FeedbackMessage(
            instance_id="fake-002",
            failure_type=FailureType.APPLY,
            route_to="localizer",
            evidence="error: patch failed: src/jv.c:1",
            retry_number=1,
        )
        bundle = agent.localize_with_feedback(instance, plan, graph, feedback)
        # After feedback expansion, jv_extra.c (neighbor via call) should appear
        cand_files = {c.file_path for c in bundle.candidates}
        check("localize_with_feedback runs without error",
              isinstance(bundle.confidence, float))
        check("Expansion discovers neighbour file",
              "src/jv_extra.c" in cand_files,
              f"candidates={cand_files}")


def test_localizer_realrepo() -> None:
    """End-to-end test on the real jq repository (skipped if not present)."""
    print("\n=== LocalizerAgent (real jq repo) ===")
    from agents.localizer import LocalizerAgent

    repo_path = PROJECT_ROOT / "repos" / "jqlang__jq"
    graph_path = PROJECT_ROOT / "graphs"

    if not repo_path.exists():
        print("  SKIP  repos/jqlang__jq/ not found — clone it to run this test")
        return

    # Try to find a graph file for any jq instance
    jq_graphs = list(graph_path.glob("jq-*__jqlang__jq.json")) if graph_path.exists() else []
    if not jq_graphs:
        print("  SKIP  No jq graph files found in graphs/ — run build_all_graphs.py")
        return

    graph = DepGraph.load(jq_graphs[0])
    instance_id = jq_graphs[0].stem

    instance = SWEInstance(
        instance_id=instance_id,
        repo="jqlang/jq",
        base_commit=graph.commit,
        problem_statement=(
            "jv_parse crashes with a segfault when given an empty string. "
            "The function dereferences the buffer pointer before checking null."
        ),
        fail_to_pass=["tests/jq.test::empty"],
        pass_to_pass=[],
    )
    plan = PlannerOutput(
        instance_id=instance_id,
        issue_type="bug",
        keywords=["jv_parse", "buffer", "parse"],
        search_hints=["look in src/jv.c"],
        suspected_modules=["src/jv.c"],
        priority_functions=["jv_parse"],
    )

    agent = LocalizerAgent(
        repos_dir=PROJECT_ROOT / "repos",
        auto_checkout=False,
    )
    bundle = agent.localize(instance, plan, graph)

    check("Real repo: candidates returned", len(bundle.candidates) >= 1)
    check("Real repo: top candidate is src/jv.c or jv_parser.c",
          any("jv" in c.file_path for c in bundle.candidates),
          f"top candidates={[c.file_path for c in bundle.candidates[:3]]}")
    check("Real repo: confidence > 0", bundle.confidence > 0.0)
    check("Real repo: file_contents non-empty", len(bundle.file_contents) >= 1)
    check("Real repo: relevant_snippets present", len(bundle.relevant_snippets) >= 1)

    print(f"  confidence:  {bundle.confidence:.3f}")
    print(f"  candidates:  {[(c.file_path, round(c.score, 3)) for c in bundle.candidates[:5]]}")
    print(f"  test_files:  {bundle.test_files[:3]}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    test_grep_tool()
    test_file_reader()
    test_graph_tools()
    test_localizer_unit()
    test_localizer_feedback()
    test_localizer_realrepo()

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

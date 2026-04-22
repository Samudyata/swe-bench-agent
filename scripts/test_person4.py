#!/usr/bin/env python3
"""Tests for Person 4 components: DiagnosticianAgent and PatcherAgent.

Run with:
    python scripts/test_person4.py

Sections
--------
1. DiagnosticianAgent unit test  — fake ContextBundle, live LLM call
2. PatcherAgent unit test        — fake FixPlan + source files, live LLM call
3. Feedback / retry loop test    — simulates Validator routing back
4. Pipeline integration test     — wires both agents through the controller
                                   using stub Localizer + Validator
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    for _c in [PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"]:
        if _c.exists():
            load_dotenv(_c)
            break
except ImportError:
    pass

from pipeline.schema import (
    AffectedRegion,
    ContextBundle,
    FailureType,
    FeedbackMessage,
    FixPlan,
    LocalizerCandidate,
    PatchOutput,
    SWEInstance,
)

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


# ── Shared fixtures ────────────────────────────────────────────────────────────

_FAKE_SOURCE = textwrap.dedent("""\
    /* jv.c - jv string implementation */
    #include <string.h>
    #include <stdlib.h>
    #include "jv.h"

    typedef struct {
      int length;
      char data[];
    } jv_string;

    jv jv_string_append_buf(jv input, const char* buf, int len) {
      jv_string* s = (jv_string*)input.u.ptr;
      int old_len = s->length;
      /* BUG: uses strlen(buf) instead of the len parameter */
      int new_len = old_len + strlen(buf);
      jv_string* ns = malloc(sizeof(jv_string) + new_len + 1);
      memcpy(ns->data, s->data, old_len);
      memcpy(ns->data + old_len, buf, len);
      ns->data[new_len] = '\\0';
      ns->length = new_len;
      free(s);
      input.u.ptr = ns;
      return input;
    }

    int jv_string_length_bytes(jv j) {
      jv_string* s = (jv_string*)j.u.ptr;
      return s->length;
    }
""")

_FAKE_TEST = textwrap.dedent("""\
    void test_string_append_multibyte() {
      jv s = jv_string("hello");
      s = jv_string_append_buf(s, "\\xc3\\xa9", 2);
      assert(jv_string_length_bytes(s) == 7);
      jv_free(s);
    }
""")


def _make_instance() -> SWEInstance:
    return SWEInstance(
        instance_id="jqlang__jq-p4test",
        repo="jqlang/jq",
        base_commit="abc1234",
        problem_statement=(
            "jv_string_append_buf reports wrong length for multibyte strings. "
            "The function uses strlen(buf) instead of the provided len parameter, "
            "causing incorrect length calculation for multibyte UTF-8 input."
        ),
        hints_text="",
        fail_to_pass=["tests::test_string_append_multibyte"],
        pass_to_pass=[],
    )


def _make_bundle() -> ContextBundle:
    return ContextBundle(
        instance_id="jqlang__jq-p4test",
        candidates=[
            LocalizerCandidate(
                file_path="src/jv.c",
                score=0.95,
                reason="graph_score=3.20; grep_hits=4; suspected_by_planner",
                functions=["src/jv.c::jv_string_append_buf"],
            )
        ],
        confidence=0.88,
        file_contents={"src/jv.c": _FAKE_SOURCE},
        relevant_snippets={
            "src/jv.c::jv_string_append_buf": (
                "// --- 12-23 (context ±3) ---\n" + _FAKE_SOURCE
            )
        },
        test_files=["tests/jv_test.c"],
    )


# ── Section 1: DiagnosticianAgent ─────────────────────────────────────────────

def test_diagnostician() -> None:
    print("\n=== DiagnosticianAgent ===")

    if not (os.environ.get("OPENAI_API_KEY")):
        print("  SKIP  no OPENAI_API_KEY — skipping live Diagnostician test")
        return

    from agents.diagnostician import DiagnosticianAgent

    agent = DiagnosticianAgent()
    instance = _make_instance()
    bundle   = _make_bundle()

    plan = agent.diagnose(instance, bundle)

    check("diagnose returns FixPlan", isinstance(plan, FixPlan))
    check("FixPlan has instance_id", plan.instance_id == instance.instance_id)
    check("root_cause non-empty", len(plan.root_cause) > 0,
          f"got: {plan.root_cause!r}")
    check("fix_description non-empty", len(plan.fix_description) > 0)
    check("affected_files contains src/jv.c",
          "src/jv.c" in plan.affected_files,
          f"got: {plan.affected_files}")
    check("at least one affected_region", len(plan.affected_regions) >= 1,
          f"got: {plan.affected_regions}")
    check("region file_path correct",
          plan.affected_regions[0].file_path == "src/jv.c",
          f"got: {plan.affected_regions[0].file_path}")
    check("region start_line > 0", plan.affected_regions[0].start_line > 0)
    check("region end_line >= start_line",
          plan.affected_regions[0].end_line >= plan.affected_regions[0].start_line)

    print(f"  root_cause:      {plan.root_cause[:120]}")
    print(f"  affected_files:  {plan.affected_files}")
    print(f"  regions:         {[(r.file_path, r.start_line, r.end_line) for r in plan.affected_regions]}")


# ── Section 2: PatcherAgent ───────────────────────────────────────────────────

def test_patcher() -> None:
    print("\n=== PatcherAgent ===")

    if not (os.environ.get("OPENAI_API_KEY")):
        print("  SKIP  no OPENAI_API_KEY — skipping live Patcher test")
        return

    from agents.patcher import PatcherAgent

    agent = PatcherAgent()
    instance = _make_instance()
    bundle = _make_bundle()

    # Build a plausible FixPlan (as if Diagnostician produced it)
    fix_plan = FixPlan(
        instance_id=instance.instance_id,
        root_cause=(
            "jv_string_append_buf uses strlen(buf) to compute new_len "
            "instead of the provided len parameter, causing incorrect "
            "byte-length for multibyte strings."
        ),
        fix_description=(
            "Replace strlen(buf) with len in the new_len calculation on line 16. "
            "Also remove the BUG comment on line 15."
        ),
        affected_files=["src/jv.c"],
        affected_regions=[
            AffectedRegion(
                file_path="src/jv.c",
                start_line=15,
                end_line=16,
                description="strlen(buf) should be len",
            )
        ],
        test_constraints=["test_string_append_multibyte must pass"],
    )

    result = agent.patch(instance, fix_plan, bundle)

    check("patch returns PatchOutput", isinstance(result, PatchOutput))
    check("PatchOutput has instance_id", result.instance_id == instance.instance_id)
    check("unified_diff non-empty", len(result.unified_diff) > 0,
          f"got: {result.unified_diff!r}")
    check("diff contains --- header", "---" in result.unified_diff)
    check("diff contains +++ header", "+++" in result.unified_diff)
    check("diff contains @@ hunk",    "@@" in result.unified_diff)
    check("diff mentions src/jv.c",   "jv.c" in result.unified_diff)
    check("affected_files non-empty", len(result.affected_files) >= 1)

    # Verify the line number margin did NOT leak into context lines
    import re
    leaked = bool(re.search(r"^[ +-]\s{0,5}\d+\s+\|", result.unified_diff, re.MULTILINE))
    check("no line-number margin in diff context", not leaked,
          "LLM included '12 | ' style margin in diff lines")

    print(f"  diff length:     {len(result.unified_diff)} chars")
    print(f"  modified files:  {result.affected_files}")
    print(f"  diff preview:\n{result.unified_diff[:600]}")


# ── Section 3: Feedback / retry loops ─────────────────────────────────────────

def test_feedback_loops() -> None:
    print("\n=== Feedback loops ===")

    if not (os.environ.get("OPENAI_API_KEY")):
        print("  SKIP  no OPENAI_API_KEY — skipping live feedback test")
        return

    from agents.diagnostician import DiagnosticianAgent
    from agents.patcher import PatcherAgent

    instance = _make_instance()
    bundle   = _make_bundle()

    # Loop C: test failure → Diagnostician.revise
    diag = DiagnosticianAgent()
    fb_test = FeedbackMessage(
        instance_id=instance.instance_id,
        failure_type=FailureType.TEST,
        route_to="diagnostician",
        evidence=(
            "FAIL tests::test_string_append_multibyte\n"
            "Expected jv_string_length_bytes(s) == 7, got 5\n"
        ),
        retry_number=1,
        context={},
    )
    revised_plan = diag.revise(instance, bundle, fb_test)
    check("revise returns FixPlan", isinstance(revised_plan, FixPlan))
    check("revised plan has affected_files",
          len(revised_plan.affected_files) >= 1)
    print(f"  revised root_cause: {revised_plan.root_cause[:100]}")

    # Loop B: compilation error → Patcher.patch_with_feedback
    fix_plan = FixPlan(
        instance_id=instance.instance_id,
        root_cause="strlen used instead of len",
        fix_description="Replace strlen(buf) with len",
        affected_files=["src/jv.c"],
        affected_regions=[
            AffectedRegion("src/jv.c", 15, 16, "wrong length calc")
        ],
        test_constraints=[],
    )
    patcher = PatcherAgent()
    fb_compile = FeedbackMessage(
        instance_id=instance.instance_id,
        failure_type=FailureType.COMPILE,
        route_to="patcher",
        evidence="src/jv.c:3:10: error: 'string.h' file not found",
        retry_number=1,
        context={},
    )
    retry_result = patcher.patch_with_feedback(instance, fix_plan, bundle, fb_compile)
    check("patch_with_feedback returns PatchOutput",
          isinstance(retry_result, PatchOutput))
    check("retry diff non-empty", len(retry_result.unified_diff) > 0)
    print(f"  retry diff length: {len(retry_result.unified_diff)} chars")


# ── Section 4: Pipeline integration ───────────────────────────────────────────

def test_pipeline_integration() -> None:
    print("\n=== Pipeline integration (controller + stubs) ===")

    from agents.diagnostician import DiagnosticianAgent
    from agents.patcher import PatcherAgent
    from agents.stubs import LocalizerStub, ValidatorStub
    from config import Config
    from graph.model import DepGraph
    from pipeline.controller import PipelineController

    if not (os.environ.get("OPENAI_API_KEY")):
        print("  SKIP  no OPENAI_API_KEY — skipping pipeline integration test")
        return

    # Build a minimal graph
    g = DepGraph(repo="jqlang/jq", commit="abc1234")
    g.add_file_node("src/jv.c")
    g.add_func_node("src/jv.c", "jv_string_append_buf", 12, 23)

    instance = _make_instance()

    # Localizer stub that returns a pre-built bundle with real file content
    class BundleLocalizer:
        def localize(self, inst, plan, graph):
            return _make_bundle()
        def localize_with_feedback(self, inst, plan, graph, feedback):
            return _make_bundle()

    class FakePlanner:
        def plan(self, inst):
            from pipeline.schema import PlannerOutput
            return PlannerOutput(
                instance_id=inst.instance_id,
                issue_type="bug",
                keywords=["jv_string_append_buf", "strlen", "len"],
                search_hints=["look in src/jv.c"],
                suspected_modules=["src/jv.c"],
                priority_functions=["jv_string_append_buf"],
            )
        def replan(self, inst, fb, prev):
            return self.plan(inst)

    with tempfile.TemporaryDirectory() as tmp:
        config = Config(
            gemini_api_key="",
            log_dir=Path(tmp) / "logs",
            results_dir=Path(tmp) / "results",
            confidence_threshold=0.4,
            max_retries=1,
        )

        ctrl = PipelineController(
            config=config,
            planner=FakePlanner(),
            localizer=BundleLocalizer(),
            diagnostician=DiagnosticianAgent(),
            patcher=PatcherAgent(),
            validator=ValidatorStub(stub_mode=True, force_result=FailureType.SUCCESS),
        )

        result = ctrl.run(instance, g)

        check("Controller completes without crash", result is not None)
        check("Result has instance_id",
            result.instance_id == instance.instance_id)
        check("apply_ok True (stub validator)",
            result.apply_ok is True, f"got {result.apply_ok}")
        check("Log dir created",
            any((Path(tmp) / "logs").iterdir()) if (Path(tmp) / "logs").exists() else False)

    print(f"  result.status:   {result.status}")
    print(f"  result.resolved: {result.resolved}")


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> None:
    test_diagnostician()
    test_patcher()
    test_feedback_loops()
    test_pipeline_integration()

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
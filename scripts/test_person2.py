#!/usr/bin/env python3
"""Tests for Person 2 components: schema, logger, planner, and controller.

Run with:
    python scripts/test_person2.py

Tests are split into sections:
  1. Schema round-trip  (no LLM needed)
  2. Logger             (no LLM needed)
  3. Controller + stubs (no LLM needed)
  4. Planner (live LLM) (requires OPENAI_API_KEY)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so OPENAI_API_KEY is available for live tests
try:
    from dotenv import load_dotenv
    for _candidate in [PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"]:
        if _candidate.exists():
            load_dotenv(_candidate)
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
    PlannerOutput,
    SWEInstance,
    ValidationResult,
)
from pipeline.logger import PipelineLogger

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
# 1. Schema round-trip tests
# ─────────────────────────────────────────────────────────────────────────────

def test_schema() -> None:
    print("\n=== Schema round-trip ===")

    # SWEInstance
    inst = SWEInstance(
        instance_id="jq-1__jqlang__jq",
        repo="jqlang/jq",
        base_commit="abc123",
        problem_statement="jv_parse crashes on empty input",
        hints_text="",
        fail_to_pass=["tests/test_jv.sh::test_empty"],
        pass_to_pass=["tests/test_jv.sh::test_basic"],
    )
    d = inst.to_dict()
    inst2 = SWEInstance.from_dict(d)
    check("SWEInstance round-trip instance_id", inst2.instance_id == inst.instance_id)
    check("SWEInstance round-trip repo", inst2.repo == inst.repo)
    check("SWEInstance fail_to_pass preserved", inst2.fail_to_pass == inst.fail_to_pass)

    # PlannerOutput
    plan = PlannerOutput(
        instance_id="jq-1__jqlang__jq",
        issue_type="bug",
        keywords=["jv_parse", "jv_load", "buffer"],
        search_hints=["look in src/jv.c"],
        suspected_modules=["src/jv.c"],
        priority_functions=["jv_parse"],
        reasoning="Issue mentions jv_parse explicitly",
    )
    d = plan.to_dict()
    plan2 = PlannerOutput.from_dict(d)
    check("PlannerOutput round-trip keywords", plan2.keywords == plan.keywords)
    check("PlannerOutput round-trip issue_type", plan2.issue_type == "bug")
    check("PlannerOutput to_dict has instance_id", "instance_id" in d)

    # FailureType enum
    check("FailureType.SUCCESS value", FailureType.SUCCESS.value == "success")
    check("FailureType.APPLY value", FailureType.APPLY.value == "apply_failed")
    ft = FailureType("compile_failed")
    check("FailureType from string", ft == FailureType.COMPILE)

    # FeedbackMessage
    fb = FeedbackMessage(
        instance_id="jq-1__jqlang__jq",
        failure_type=FailureType.COMPILE,
        route_to="patcher",
        evidence="error: undeclared identifier 'jv_null'",
        retry_number=1,
        context={"compile_line": 42},
    )
    d = fb.to_dict()
    fb2 = FeedbackMessage.from_dict(d)
    check("FeedbackMessage round-trip failure_type", fb2.failure_type == FailureType.COMPILE)
    check("FeedbackMessage round-trip route_to", fb2.route_to == "patcher")
    check("FeedbackMessage failure_type serialised as string", isinstance(d["failure_type"], str))

    # ValidationResult
    vr = ValidationResult(
        instance_id="jq-1__jqlang__jq",
        status=FailureType.SUCCESS,
        resolved=True,
        apply_ok=True,
        compile_ok=True,
        tests_passed=["tests/test_jv.sh::test_empty"],
        tests_failed=[],
    )
    d = vr.to_dict()
    vr2 = ValidationResult.from_dict(d)
    check("ValidationResult round-trip resolved", vr2.resolved is True)
    check("ValidationResult status serialised as string", isinstance(d["status"], str))

    # ContextBundle
    bundle = ContextBundle(
        instance_id="jq-1__jqlang__jq",
        candidates=[
            LocalizerCandidate(
                file_path="src/jv.c",
                score=3.5,
                reason="contains jv_parse",
                functions=["src/jv.c::jv_parse"],
            )
        ],
        confidence=0.75,
        file_contents={"src/jv.c": "int jv_parse(...) { ... }"},
        relevant_snippets={"src/jv.c::jv_parse": "int jv_parse(...) { ... }"},
        test_files=["tests/test_jv.sh"],
    )
    d = bundle.to_dict()
    bundle2 = ContextBundle.from_dict(d)
    check("ContextBundle round-trip confidence", bundle2.confidence == 0.75)
    check("ContextBundle candidate count", len(bundle2.candidates) == 1)
    check("ContextBundle candidate file_path", bundle2.candidates[0].file_path == "src/jv.c")

    # FixPlan with AffectedRegion
    fp = FixPlan(
        instance_id="jq-1__jqlang__jq",
        root_cause="jv_parse dereferences null pointer on empty input",
        affected_files=["src/jv.c"],
        affected_regions=[
            AffectedRegion(
                file_path="src/jv.c",
                start_line=100,
                end_line=110,
                description="null check missing before deref",
            )
        ],
        test_constraints=["test_empty must pass"],
        fix_description="Add null check before pointer dereference",
    )
    d = fp.to_dict()
    fp2 = FixPlan.from_dict(d)
    check("FixPlan round-trip root_cause", fp2.root_cause == fp.root_cause)
    check("FixPlan AffectedRegion round-trip", fp2.affected_regions[0].start_line == 100)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Logger tests
# ─────────────────────────────────────────────────────────────────────────────

def test_logger() -> None:
    print("\n=== Logger ===")
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        logger = PipelineLogger("test-instance", log_dir, echo=False)

        check("Logger path created", logger.path.parent.exists())

        logger.log("instance_start", "controller", {"instance_id": "test-instance"})
        logger.log("planner_output", "planner", {"keywords": ["foo", "bar"]})
        logger.log("validation_result", "validator", {"status": "success", "resolved": True})
        logger.log_error("planner", ValueError("test error"))

        entries = logger.read_all()
        check("Logger wrote 4 entries", len(entries) == 4, f"got {len(entries)}")
        check("Each entry has timestamp", all("timestamp" in e for e in entries))
        check("Each entry has event", all("event" in e for e in entries))
        check("Each entry has agent", all("agent" in e for e in entries))

        events = [e["event"] for e in entries]
        check("instance_start logged", "instance_start" in events)
        check("planner_output logged", "planner_output" in events)
        check("error logged", "error" in events)

        # Verify JSONL format (each line is valid JSON)
        raw_lines = logger.path.read_text().strip().splitlines()
        check("JSONL: 4 non-empty lines", len(raw_lines) == 4, f"got {len(raw_lines)}")
        parsed = []
        for line in raw_lines:
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError as e:
                check(f"JSONL valid JSON: {line[:40]}", False, str(e))
        check("All lines are valid JSON", len(parsed) == 4)

        # Test static load
        loaded = PipelineLogger.load(logger.path)
        check("PipelineLogger.load() works", len(loaded) == 4)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Controller + stub agent tests
# ─────────────────────────────────────────────────────────────────────────────

def test_controller_stubs() -> None:
    print("\n=== Controller + Stub agents ===")

    # Build a minimal fake graph (no tree-sitter needed)
    from graph.model import DepGraph
    g = DepGraph(repo="jqlang/jq", commit="abc123")
    g.add_file_node("src/jv.c")
    g.add_file_node("src/execute.c")
    fid = g.add_func_node("src/jv.c", "jv_parse", 1, 50)
    g.add_edge("src/jv.c", "src/execute.c", "include", confidence=1.0)

    instance = SWEInstance(
        instance_id="stub-test-001",
        repo="jqlang/jq",
        base_commit="abc123",
        problem_statement="jv_parse crashes on empty input",
        fail_to_pass=["test_empty"],
        pass_to_pass=["test_basic"],
    )

    from agents.stubs import (
        DiagnosticianStub,
        LocalizerStub,
        PatcherStub,
        ValidatorStub,
    )
    from config import Config
    from pipeline.controller import PipelineController

    with tempfile.TemporaryDirectory() as tmp:
        config = Config(
            gemini_api_key="",
            log_dir=Path(tmp) / "logs",
            results_dir=Path(tmp) / "results",
            confidence_threshold=0.4,
            max_retries=1,
        )

        # Build a fake planner (no LLM)
        class FakePlanner:
            def plan(self, inst):
                return PlannerOutput(
                    instance_id=inst.instance_id,
                    issue_type="bug",
                    keywords=["jv_parse"],
                    search_hints=["look in src/jv.c"],
                    suspected_modules=[],
                    priority_functions=["jv_parse"],
                )
            def replan(self, inst, fb, prev):
                return self.plan(inst)

        # Test 1: successful run (validator returns SUCCESS)
        ctrl = PipelineController(
            config=config,
            planner=FakePlanner(),
            localizer=LocalizerStub(stub_mode=True),
            diagnostician=DiagnosticianStub(stub_mode=True),
            patcher=PatcherStub(stub_mode=True),
            validator=ValidatorStub(stub_mode=True, force_result=FailureType.SUCCESS),
        )
        result = ctrl.run(instance, g)
        check("Controller: SUCCESS run returns resolved=True", result.resolved is True)
        check("Controller: result file written",
              (Path(tmp) / "results" / "stub-test-001.json").exists())
        check("Controller: log file written",
              any(f.suffix == ".jsonl" for f in (Path(tmp) / "logs").iterdir()))

        # Test 2: failure → retry → success
        call_count = [0]
        class FlakyValidator:
            def validate(self, inst, patch):
                call_count[0] += 1
                if call_count[0] == 1:
                    return ValidationResult(
                        instance_id=inst.instance_id,
                        status=FailureType.COMPILE,
                        resolved=False,
                        apply_ok=True,
                        compile_ok=False,
                        tests_passed=[],
                        tests_failed=inst.fail_to_pass,
                        error_output="error: undeclared 'jv_null'",
                    )
                return ValidationResult(
                    instance_id=inst.instance_id,
                    status=FailureType.SUCCESS,
                    resolved=True,
                    apply_ok=True,
                    compile_ok=True,
                    tests_passed=inst.fail_to_pass,
                    tests_failed=[],
                )

        instance2 = SWEInstance(
            instance_id="stub-test-002",
            repo="jqlang/jq",
            base_commit="abc123",
            problem_statement="another bug",
            fail_to_pass=["test_x"],
            pass_to_pass=[],
        )
        ctrl2 = PipelineController(
            config=config,
            planner=FakePlanner(),
            localizer=LocalizerStub(stub_mode=True),
            diagnostician=DiagnosticianStub(stub_mode=True),
            patcher=PatcherStub(stub_mode=True),
            validator=FlakyValidator(),
        )
        result2 = ctrl2.run(instance2, g)
        check("Controller: retry triggers on compile failure", call_count[0] == 2,
              f"validator called {call_count[0]} times (expected 2)")
        check("Controller: resolved after retry", result2.resolved is True)

        # Verify JSONL has retry_routed event
        log_files = list((Path(tmp) / "logs").glob("stub-test-002_*.jsonl"))
        check("Controller: log file for instance-002 exists", len(log_files) == 1)
        if log_files:
            entries = PipelineLogger.load(log_files[0])
            events = [e["event"] for e in entries]
            check("Controller: retry_routed event logged", "retry_routed" in events)
            check("Controller: instance_end logged", "instance_end" in events)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Planner live test (requires OPENAI_API_KEY)
# ─────────────────────────────────────────────────────────────────────────────

def test_planner_live() -> None:
    print("\n=== Planner (live LLM) ===")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("  SKIP  OPENAI_API_KEY not set — skipping live Planner test")
        return

    from agents.planner import PlannerAgent
    planner = PlannerAgent(api_key=api_key, temperature=0.0)

    instance = SWEInstance(
        instance_id="jq-live-test",
        repo="jqlang/jq",
        base_commit="deadbeef",
        problem_statement=(
            "jv_parse crashes with a segfault when given an empty string as input. "
            "The function dereferences the buffer pointer before checking if it is null. "
            "This was introduced in commit 3f2a1b when the parser was refactored."
        ),
        fail_to_pass=["tests/jq.test::empty_string"],
        pass_to_pass=[],
    )

    try:
        plan = planner.plan(instance)
    except Exception as exc:
        exc_str = str(exc) + str(getattr(exc, '__cause__', ''))
        if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str or "quota" in exc_str.lower():
            print("  SKIP  Quota exhausted (429) — API key is valid, call would succeed with quota")
            print("        Wait for the rate-limit window to reset (usually 1 minute or 24 hours)")
            return
        raise

    check("Planner returns PlannerOutput", isinstance(plan, PlannerOutput))
    check("Planner issue_type is 'bug'", plan.issue_type == "bug",
          f"got {plan.issue_type!r}")
    check("Planner keywords non-empty", len(plan.keywords) > 0,
          f"got {plan.keywords}")
    check("Planner keywords ≤ 8", len(plan.keywords) <= 8,
          f"got {len(plan.keywords)}")
    check("Planner 'jv_parse' in keywords or priority_functions",
          any("jv_parse" in k for k in plan.keywords + plan.priority_functions),
          f"keywords={plan.keywords} funcs={plan.priority_functions}")
    check("Planner search_hints non-empty", len(plan.search_hints) > 0)
    check("Planner reasoning non-empty", len(plan.reasoning) > 0)

    print(f"  keywords:           {plan.keywords}")
    print(f"  priority_functions: {plan.priority_functions}")
    print(f"  suspected_modules:  {plan.suspected_modules}")
    print(f"  reasoning:          {plan.reasoning}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global passed, failed
    test_schema()
    test_logger()
    test_controller_stubs()
    test_planner_live()

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

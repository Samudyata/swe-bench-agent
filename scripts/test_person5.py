#!/usr/bin/env python3
"""Tests for Person 5: ValidatorAgent

Run with:
    python scripts/test_person5.py

Sections
--------
1. Full Pipeline Integration — Diagnostician → Patcher → Validator with real jq repo
2. ValidatorAgent unit test    — test with hand-written patch
"""

from __future__ import annotations

import os
import sys
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
    ContextBundle,
    FailureType,
    LocalizerCandidate,
    PatchOutput,
    SWEInstance,
)
from agents.diagnostician import DiagnosticianAgent
from agents.patcher import PatcherAgent
from agents.validator import ValidatorAgent

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


def _make_instance() -> SWEInstance:
    return SWEInstance(
        instance_id="jqlang__jq-p5test",
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
        instance_id="jqlang__jq-p5test",
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


# ── Section 1: Full Pipeline Integration ──────────────────────────────────────

def test_full_pipeline() -> None:
    print("\n=== Full Pipeline: Diagnostician → Patcher → Validator ===")

    if not (os.environ.get("OPENAI_API_KEY")):
        print("  SKIP  no OPENAI_API_KEY — skipping live pipeline test")
        return

    # Check if jq repo exists
    repo_path = PROJECT_ROOT / "repos" / "jqlang__jq"
    if not repo_path.exists():
        print(f"  SKIP  jq repo not found at {repo_path} — clone it first")
        return

    # Initialize agents
    diagnostician = DiagnosticianAgent()
    patcher = PatcherAgent(repo_root=str(repo_path))
    validator = ValidatorAgent(repo_root=str(repo_path))

    # Create test data
    instance = _make_instance()
    bundle = _make_bundle()

    # Step 1: Diagnostician produces FixPlan
    print("\n  Step 1: Diagnostician analyzing issue...")
    try:
        fix_plan = diagnostician.diagnose(instance, bundle)
        check("Diagnostician returns FixPlan", fix_plan is not None)
        check("FixPlan has root_cause", len(fix_plan.root_cause) > 0,
              f"got: {fix_plan.root_cause[:100] if fix_plan.root_cause else 'empty'}")
        print(f"    Root cause: {fix_plan.root_cause[:120]}...")
    except Exception as e:
        check("Diagnostician completes without error", False, str(e))
        return

    # Step 2: Patcher generates unified diff
    print("\n  Step 2: Patcher generating patch...")
    try:
        patch_output = patcher.patch(instance, fix_plan, bundle)
        check("Patcher returns PatchOutput", patch_output is not None)
        check("PatchOutput has unified_diff", len(patch_output.unified_diff) > 0,
              f"got {len(patch_output.unified_diff)} chars")
        check("Diff contains --- header", "---" in patch_output.unified_diff)
        check("Diff contains +++ header", "+++" in patch_output.unified_diff)
        print(f"    Generated {len(patch_output.unified_diff)} char diff")
    except Exception as e:
        check("Patcher completes without error", False, str(e))
        return

    # Step 3: Validator validates the patch
    print("\n  Step 3: Validator checking patch...")
    try:
        validation_result = validator.validate(instance, patch_output)
        check("Validator returns ValidationResult", validation_result is not None)
        check("ValidationResult has status", hasattr(validation_result, 'status'))
        check("ValidationResult has apply_ok", hasattr(validation_result, 'apply_ok'))

        # Check each stage
        print(f"    Apply:   {'✓' if validation_result.apply_ok else '✗'}")
        print(f"    Compile: {'✓' if validation_result.compile_ok else '✗'}")
        print(f"    Status:  {validation_result.status.value}")
        print(f"    Resolved: {validation_result.resolved}")

        # Note: We're using fake source code, so patch may not apply to real jq repo
        # The important thing is validator runs without crashing and returns a result
        print(f"    Note: Using fake source code, patch application may fail (expected)")
        if not validation_result.apply_ok:
            print(f"    Apply error: {validation_result.error_output[:200]}")

    except Exception as e:
        check("Validator completes without error", False, str(e))
        return

    print(f"\n  Pipeline test complete!")


# ── Section 2: ValidatorAgent Unit Test ───────────────────────────────────────

def test_validator_unit() -> None:
    print("\n=== ValidatorAgent Unit Test ===")

    # Check if jq repo exists
    repo_path = PROJECT_ROOT / "repos" / "jqlang__jq"
    if not repo_path.exists():
        print(f"  SKIP  jq repo not found at {repo_path}")
        return

    # Create a simple valid patch that matches actual jq README
    simple_patch = textwrap.dedent("""\
        diff --git a/README.md b/README.md
        index 1234567..abcdefg 100644
        --- a/README.md
        +++ b/README.md
        @@ -1,4 +1,5 @@
         # jq
        +<!-- Test comment -->

         `jq` is a lightweight and flexible command-line JSON processor akin to `sed`,`awk`,`grep`, and friends for JSON data. It's written in portable C and has zero runtime dependencies, allowing you to easily slice, filter, map, and transform structured data.


    """)

    instance = SWEInstance(
        instance_id="test-validator-unit",
        repo="jqlang/jq",
        base_commit="master",
        problem_statement="Test",
        hints_text="",
        fail_to_pass=[],  # No tests for this simple patch
        pass_to_pass=[],
    )

    patch_output = PatchOutput(
        instance_id="test-validator-unit",
        unified_diff=simple_patch,
        affected_files=["README.md"],
    )

    validator = ValidatorAgent(repo_root=str(repo_path))

    try:
        result = validator.validate(instance, patch_output)
        check("Validator returns result", result is not None)
        check("Result has status", hasattr(result, 'status'))

        # For a simple README patch with no tests, we expect:
        # - Apply: should succeed
        # - Compile: should succeed (no code changes)
        # - Tests: no tests to run, should pass
        check("Simple patch applies", result.apply_ok,
              f"Apply failed: {result.error_output[:200] if not result.apply_ok else ''}")

        print(f"    Status: {result.status.value}")
        print(f"    Apply OK: {result.apply_ok}")
        print(f"    Compile OK: {result.compile_ok}")

    except Exception as e:
        check("Validator unit test completes", False, str(e))


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> None:
    test_full_pipeline()
    test_validator_unit()

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

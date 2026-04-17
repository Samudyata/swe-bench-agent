"""
validator.py — Validator Agent (Agent 5)

Applies a patch, compiles the code, and runs tests to verify the fix.
Routes errors back to the appropriate agent for retry.

Inputs  (from Patcher):  PatchResult
Outputs (to orchestrator): validation result dict

The Validator runs four stages in order, stopping at the first failure:
  1. Apply    — `git apply` the patch
  2. Compile  — `make` or `make all`
  3. FAIL_TO_PASS tests — tests that should now pass
  4. PASS_TO_PASS tests — tests that should still pass (regression check)

Retry handling:
  - Loop B: compilation errors → route back to Patcher via revalidate_after_patch()
  - Loop C: test failures → route back to Diagnostician via revalidate_after_diagnosis()
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from agents.schemas import PatchResult

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_RETRIES = 2
COMPILE_TIMEOUT = 120
TEST_TIMEOUT = 60


# ── Git operations ────────────────────────────────────────────────────────────

def _cleanup_repo(repo_root: str) -> None:
    """
    Reset the repository to a clean state.
    Discards all uncommitted changes and untracked files.
    """
    print(f"  [Validator] Cleaning up repository at {repo_root}")

    # Reset all changes
    result = subprocess.run(
        ["git", "reset", "--hard", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        print(f"  [Validator] WARNING: git reset failed: {result.stderr}")

    # Clean untracked files
    result = subprocess.run(
        ["git", "clean", "-fd"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        print(f"  [Validator] WARNING: git clean failed: {result.stderr}")


def _apply_patch(patch_text: str, repo_root: str) -> tuple[bool, str]:
    """
    Apply a unified diff patch using git apply.

    Args:
        patch_text: The unified diff content
        repo_root: Absolute path to repository

    Returns:
        (success: bool, error_output: str)
    """
    print(f"  [Validator] Applying patch ({len(patch_text)} chars)")

    # Write patch to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(patch_text)
        patch_file = f.name

    try:
        # First try: git apply --check (dry run)
        result = subprocess.run(
            ["git", "apply", "--check", patch_file],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            # Try with 3-way merge as fallback
            print(f"  [Validator] Standard apply failed, trying --3way")
            result = subprocess.run(
                ["git", "apply", "--3way", patch_file],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                error_output = result.stdout + result.stderr
                return False, error_output
            # --3way succeeded and applied in-place
            print(f"  [Validator] Patch applied successfully with --3way")
            return True, ""
        else:
            # Dry run succeeded, apply for real
            result = subprocess.run(
                ["git", "apply", patch_file],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                error_output = result.stdout + result.stderr
                return False, error_output
            print(f"  [Validator] Patch applied successfully")
            return True, ""

    finally:
        # Clean up temp file
        if os.path.exists(patch_file):
            os.unlink(patch_file)


# ── Compilation ───────────────────────────────────────────────────────────────

def _compile_code(repo_root: str) -> tuple[bool, str]:
    """
    Run make to compile the code.

    Args:
        repo_root: Absolute path to repository

    Returns:
        (success: bool, error_output: str)
    """
    print(f"  [Validator] Compiling code in {repo_root}")

    # Try make clean first (optional, don't fail if it doesn't work)
    subprocess.run(
        ["make", "clean"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Run make or make all
    for target in ["all", ""]:
        cmd = ["make", target] if target else ["make"]
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT,
        )

        output = result.stdout + result.stderr

        if result.returncode == 0:
            print(f"  [Validator] Compilation succeeded")
            return True, ""

        # If we tried "make all" and it failed, try just "make"
        if target == "all":
            continue

        # Final attempt failed
        print(f"  [Validator] Compilation failed")
        return False, output

    # Should not reach here, but just in case
    return False, "Compilation failed: no make target succeeded"


# ── Test execution ────────────────────────────────────────────────────────────

def _detect_test_command(repo_root: str) -> str:
    """
    Detect the correct test command for the repository.

    Returns:
        The make target to use for running tests
    """
    repo_path = Path(repo_root)
    repo_name = repo_path.name.lower()

    # Heuristic based on repository name
    if "jq" in repo_name:
        return "check"
    elif "zstd" in repo_name:
        return "test"
    elif "redis" in repo_name:
        return "test"
    else:
        # Default fallback
        return "test"


def _run_test(test_name: str, repo_root: str, test_command: str) -> tuple[bool, str]:
    """
    Run a single test.

    Args:
        test_name: Name of the test to run
        repo_root: Absolute path to repository
        test_command: The make target (e.g., "check", "test")

    Returns:
        (success: bool, output: str)
    """
    print(f"  [Validator] Running test: {test_name}")

    # Try running the specific test
    # Format depends on test framework, try common patterns
    for cmd_variant in [
        ["make", test_command, f"TESTS={test_name}"],
        ["make", test_command, f"TEST={test_name}"],
        ["make", test_name],
    ]:
        result = subprocess.run(
            cmd_variant,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT,
        )

        output = result.stdout + result.stderr

        # Check if test actually ran (not just command failed)
        if "no rule" not in output.lower() and "no such file" not in output.lower():
            success = result.returncode == 0
            return success, output

    # Fallback: run all tests and check output
    result = subprocess.run(
        ["make", test_command],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=TEST_TIMEOUT,
    )

    output = result.stdout + result.stderr
    success = result.returncode == 0

    return success, output


def _run_tests(test_names: list[str], repo_root: str, test_command: str) -> tuple[bool, str]:
    """
    Run a list of tests.

    Args:
        test_names: List of test names to run
        repo_root: Absolute path to repository
        test_command: The make target (e.g., "check", "test")

    Returns:
        (all_passed: bool, combined_output: str)
    """
    if not test_names:
        return True, ""

    all_output = []
    all_passed = True

    for test_name in test_names:
        success, output = _run_test(test_name, repo_root, test_command)
        all_output.append(f"=== Test: {test_name} ===")
        all_output.append(output)

        if not success:
            all_passed = False
            print(f"  [Validator] Test FAILED: {test_name}")
        else:
            print(f"  [Validator] Test PASSED: {test_name}")

    combined = "\n\n".join(all_output)
    return all_passed, combined


# ── Helper functions ──────────────────────────────────────────────────────────

def _parse_test_list(value) -> list[str]:
    """
    Parse a test list that might be a JSON string or already a list.

    SWE-bench sometimes stores FAIL_TO_PASS and PASS_TO_PASS as JSON strings.

    Args:
        value: Either a list of test names or a JSON string

    Returns:
        List of test names
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return []


# ── Main validation pipeline ──────────────────────────────────────────────────

def validate(
    patch_result: PatchResult,
    repo_root: str,
    instance: dict,
) -> dict:
    """
    Run the complete validation pipeline on a patch.

    Stages (stop at first failure):
      1. Apply — git apply the patch
      2. Compile — run make
      3. FAIL_TO_PASS — tests that should now pass
      4. PASS_TO_PASS — tests that should still pass

    Args:
        patch_result: The patch from the Patcher agent
        repo_root: Absolute path to repository
        instance: SWE-bench instance dict with FAIL_TO_PASS and PASS_TO_PASS fields

    Returns:
        dict with keys:
          - status: "apply_failed" | "compile_failed" | "test_failed" | "regression" | "success"
          - apply_ok: bool
          - compile_ok: bool (if reached)
          - tests_ok: bool (if reached)
          - regression_ok: bool (if reached)
          - resolved: bool (True only if all stages pass)
          - error_output: str (if any stage failed)
          - patch_result: PatchResult (original input)
    """
    print(f"  [Validator] Starting validation for {patch_result.issue_id} (attempt {patch_result.attempt_number})")

    # Clean up repo before starting
    _cleanup_repo(repo_root)

    result_dict = {
        "issue_id": patch_result.issue_id,
        "attempt_number": patch_result.attempt_number,
        "patch_result": patch_result,
    }

    # Stage 1: Apply patch
    apply_ok, apply_error = _apply_patch(patch_result.patch, repo_root)
    result_dict["apply_ok"] = apply_ok

    if not apply_ok:
        result_dict["status"] = "apply_failed"
        result_dict["error_output"] = apply_error
        result_dict["resolved"] = False
        print(f"  [Validator] FAILED at Apply stage")
        _cleanup_repo(repo_root)
        return result_dict

    # Stage 2: Compile
    compile_ok, compile_error = _compile_code(repo_root)
    result_dict["compile_ok"] = compile_ok

    if not compile_ok:
        result_dict["status"] = "compile_failed"
        result_dict["error_output"] = compile_error
        result_dict["resolved"] = False
        print(f"  [Validator] FAILED at Compile stage")
        _cleanup_repo(repo_root)
        return result_dict

    # Detect test command
    test_command = _detect_test_command(repo_root)

    # Stage 3: FAIL_TO_PASS tests
    fail_to_pass_tests = _parse_test_list(instance.get("FAIL_TO_PASS", []))
    if fail_to_pass_tests:
        tests_ok, test_output = _run_tests(fail_to_pass_tests, repo_root, test_command)
        result_dict["tests_ok"] = tests_ok

        if not tests_ok:
            result_dict["status"] = "test_failed"
            result_dict["error_output"] = test_output
            result_dict["resolved"] = False
            print(f"  [Validator] FAILED at FAIL_TO_PASS stage")
            _cleanup_repo(repo_root)
            return result_dict
    else:
        result_dict["tests_ok"] = True
        print(f"  [Validator] No FAIL_TO_PASS tests to run")

    # Stage 4: PASS_TO_PASS tests (regression check)
    pass_to_pass_tests = _parse_test_list(instance.get("PASS_TO_PASS", []))
    if pass_to_pass_tests:
        regression_ok, regression_output = _run_tests(pass_to_pass_tests, repo_root, test_command)
        result_dict["regression_ok"] = regression_ok

        if not regression_ok:
            result_dict["status"] = "regression"
            result_dict["error_output"] = regression_output
            result_dict["resolved"] = False
            print(f"  [Validator] FAILED at PASS_TO_PASS stage (regression)")
            _cleanup_repo(repo_root)
            return result_dict
    else:
        result_dict["regression_ok"] = True
        print(f"  [Validator] No PASS_TO_PASS tests to run")

    # All stages passed!
    result_dict["status"] = "success"
    result_dict["resolved"] = True
    result_dict["error_output"] = ""
    print(f"  [Validator] SUCCESS — all stages passed for {patch_result.issue_id}")

    _cleanup_repo(repo_root)
    return result_dict


# ── Retry entry points ────────────────────────────────────────────────────────

def revalidate_after_patch(
    patch_result: PatchResult,
    repo_root: str,
    instance: dict,
) -> dict:
    """
    Re-run validation after Patcher generated a new patch (Loop B).

    This is called by the orchestrator when the Patcher has revised
    the patch in response to a compilation error.

    Args:
        patch_result: New PatchResult from Patcher
        repo_root: Absolute path to repository
        instance: SWE-bench instance dict

    Returns:
        Validation result dict
    """
    if patch_result.attempt_number > MAX_RETRIES + 1:
        print(f"  [Validator] Max retries exceeded for {patch_result.issue_id}")
        return {
            "status": "max_retries_exceeded",
            "error_output": f"Exceeded {MAX_RETRIES} retry attempts",
            "resolved": False,
            "patch_result": patch_result,
        }

    print(f"  [Validator] Re-validating after Patcher retry (attempt {patch_result.attempt_number})")
    return validate(patch_result, repo_root, instance)


def revalidate_after_diagnosis(
    patch_result: PatchResult,
    repo_root: str,
    instance: dict,
) -> dict:
    """
    Re-run validation after Diagnostician revised the fix plan (Loop C).

    This is called by the orchestrator when the Diagnostician has
    re-analyzed the problem in response to test failures.

    Args:
        patch_result: New PatchResult from Patcher (using revised FixPlan)
        repo_root: Absolute path to repository
        instance: SWE-bench instance dict

    Returns:
        Validation result dict
    """
    if patch_result.attempt_number > MAX_RETRIES + 1:
        print(f"  [Validator] Max retries exceeded for {patch_result.issue_id}")
        return {
            "status": "max_retries_exceeded",
            "error_output": f"Exceeded {MAX_RETRIES} retry attempts",
            "resolved": False,
            "patch_result": patch_result,
        }

    print(f"  [Validator] Re-validating after Diagnostician retry (attempt {patch_result.attempt_number})")
    return validate(patch_result, repo_root, instance)

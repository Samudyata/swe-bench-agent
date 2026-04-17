"""
validator.py — Validator Agent (Person 5)

Applies a patch, compiles the code, and runs tests to verify the fix.
Routes errors back to the appropriate agent for retry.

Implements the interface defined in agents/stubs.py:
    def validate(instance: SWEInstance, patch: PatchOutput) → ValidationResult

The Validator runs four stages in order, stopping at the first failure:
  1. Apply    — `git apply` the patch
  2. Compile  — `make` or `make all`
  3. FAIL_TO_PASS tests — tests that should now pass
  4. PASS_TO_PASS tests — tests that should still pass (regression check)

Usage
-----
    from agents.validator import ValidatorAgent
    agent = ValidatorAgent(repo_root="/path/to/repo")
    result = agent.validate(instance, patch_output)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from pipeline.schema import (
    FailureType,
    PatchOutput,
    SWEInstance,
    ValidationResult,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

COMPILE_TIMEOUT = 120
TEST_TIMEOUT = 60


# ── Helper functions ──────────────────────────────────────────────────────────

def _parse_test_list(value) -> list[str]:
    """
    Parse a test list that might be a JSON string or already a list.

    SWE-bench sometimes stores fail_to_pass and pass_to_pass as JSON strings.

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


# ── Git operations ────────────────────────────────────────────────────────────

def _cleanup_repo(repo_root: str) -> None:
    """
    Reset the repository to a clean state.
    Discards all uncommitted changes and untracked files.
    """
    logger.info("Cleaning up repository at %s", repo_root)

    # Reset all changes
    result = subprocess.run(
        ["git", "reset", "--hard", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        logger.warning("git reset failed: %s", result.stderr)

    # Clean untracked files
    result = subprocess.run(
        ["git", "clean", "-fd"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        logger.warning("git clean failed: %s", result.stderr)


def _apply_patch(patch_text: str, repo_root: str) -> tuple[bool, str]:
    """
    Apply a unified diff patch using git apply.

    Args:
        patch_text: The unified diff content
        repo_root: Absolute path to repository

    Returns:
        (success: bool, error_output: str)
    """
    logger.info("Applying patch (%d chars)", len(patch_text))

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
            logger.info("Standard apply failed, trying --3way")
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
            logger.info("Patch applied successfully with --3way")
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
            logger.info("Patch applied successfully")
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
    logger.info("Compiling code in %s", repo_root)

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
            logger.info("Compilation succeeded")
            return True, ""

        # If we tried "make all" and it failed, try just "make"
        if target == "all":
            continue

        # Final attempt failed
        logger.error("Compilation failed")
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
    logger.info("Running test: %s", test_name)

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


def _run_tests(test_names: list[str], repo_root: str, test_command: str) -> tuple[list[str], list[str], str]:
    """
    Run a list of tests.

    Args:
        test_names: List of test names to run
        repo_root: Absolute path to repository
        test_command: The make target (e.g., "check", "test")

    Returns:
        (passed: list[str], failed: list[str], combined_output: str)
    """
    if not test_names:
        return [], [], ""

    passed = []
    failed = []
    all_output = []

    for test_name in test_names:
        success, output = _run_test(test_name, repo_root, test_command)
        all_output.append(f"=== Test: {test_name} ===")
        all_output.append(output)

        if success:
            passed.append(test_name)
            logger.info("Test PASSED: %s", test_name)
        else:
            failed.append(test_name)
            logger.error("Test FAILED: %s", test_name)

    combined = "\n\n".join(all_output)
    return passed, failed, combined


# ── Validator Agent ───────────────────────────────────────────────────────────

class ValidatorAgent:
    """
    Validates generated patches through apply, compile, and test stages.

    Implements the interface expected by the pipeline controller:
        validate(instance, patch) → ValidationResult

    Args:
        repo_root: Optional override for repository root path.
                   If not provided, uses instance.repo_root or derives from instance.
    """

    def __init__(self, repo_root: Optional[str] = None) -> None:
        self._repo_root = repo_root

    def validate(
        self,
        instance: SWEInstance,
        patch: PatchOutput,
    ) -> ValidationResult:
        """
        Run the complete validation pipeline on a patch.

        Stages (stop at first failure):
          1. Apply — git apply the patch
          2. Compile — run make
          3. FAIL_TO_PASS — tests that should now pass
          4. PASS_TO_PASS — tests that should still pass

        Args:
            instance: SWE-bench instance with test lists
            patch: Unified diff from the Patcher

        Returns:
            ValidationResult with status and stage completion flags
        """
        logger.info(
            "[%s] Validator starting validation",
            instance.instance_id,
        )

        # Determine repo root
        repo_root = self._repo_root or getattr(instance, "repo_root", "")
        if not repo_root:
            logger.error("[%s] No repo_root available", instance.instance_id)
            return ValidationResult(
                instance_id=instance.instance_id,
                status=FailureType.APPLY,
                resolved=False,
                apply_ok=False,
                compile_ok=False,
                tests_passed=[],
                tests_failed=[],
                error_output="No repo_root provided",
            )

        # Clean up repo before starting
        _cleanup_repo(repo_root)

        # Stage 1: Apply patch
        apply_ok, apply_error = _apply_patch(patch.unified_diff, repo_root)

        if not apply_ok:
            logger.error("[%s] FAILED at Apply stage", instance.instance_id)
            _cleanup_repo(repo_root)
            return ValidationResult(
                instance_id=instance.instance_id,
                status=FailureType.APPLY,
                resolved=False,
                apply_ok=False,
                compile_ok=False,
                tests_passed=[],
                tests_failed=[],
                error_output=apply_error,
            )

        # Stage 2: Compile
        compile_ok, compile_error = _compile_code(repo_root)

        if not compile_ok:
            logger.error("[%s] FAILED at Compile stage", instance.instance_id)
            _cleanup_repo(repo_root)
            return ValidationResult(
                instance_id=instance.instance_id,
                status=FailureType.COMPILE,
                resolved=False,
                apply_ok=True,
                compile_ok=False,
                tests_passed=[],
                tests_failed=[],
                error_output=compile_error,
            )

        # Detect test command
        test_command = _detect_test_command(repo_root)

        # Stage 3: FAIL_TO_PASS tests
        fail_to_pass_tests = _parse_test_list(instance.fail_to_pass)
        tests_passed = []
        tests_failed = []
        test_output = ""

        if fail_to_pass_tests:
            passed, failed, test_output = _run_tests(fail_to_pass_tests, repo_root, test_command)
            tests_passed.extend(passed)
            tests_failed.extend(failed)

            if failed:
                logger.error("[%s] FAILED at FAIL_TO_PASS stage", instance.instance_id)
                _cleanup_repo(repo_root)
                return ValidationResult(
                    instance_id=instance.instance_id,
                    status=FailureType.TEST,
                    resolved=False,
                    apply_ok=True,
                    compile_ok=True,
                    tests_passed=tests_passed,
                    tests_failed=tests_failed,
                    error_output=test_output,
                )
        else:
            logger.info("[%s] No FAIL_TO_PASS tests to run", instance.instance_id)

        # Stage 4: PASS_TO_PASS tests (regression check)
        pass_to_pass_tests = _parse_test_list(instance.pass_to_pass)

        if pass_to_pass_tests:
            passed, failed, regression_output = _run_tests(pass_to_pass_tests, repo_root, test_command)
            tests_passed.extend(passed)
            tests_failed.extend(failed)

            if failed:
                logger.error("[%s] FAILED at PASS_TO_PASS stage (regression)", instance.instance_id)
                _cleanup_repo(repo_root)
                return ValidationResult(
                    instance_id=instance.instance_id,
                    status=FailureType.TEST,
                    resolved=False,
                    apply_ok=True,
                    compile_ok=True,
                    tests_passed=tests_passed,
                    tests_failed=tests_failed,
                    error_output=test_output + "\n\n" + regression_output,
                )
        else:
            logger.info("[%s] No PASS_TO_PASS tests to run", instance.instance_id)

        # All stages passed!
        logger.info("[%s] SUCCESS — all stages passed", instance.instance_id)
        _cleanup_repo(repo_root)

        return ValidationResult(
            instance_id=instance.instance_id,
            status=FailureType.SUCCESS,
            resolved=True,
            apply_ok=True,
            compile_ok=True,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            error_output="",
        )

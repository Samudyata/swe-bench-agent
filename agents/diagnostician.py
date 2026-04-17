"""
diagnostician.py — Diagnostician Agent (Agent 3)

Reads the localized source files + issue description from the Localizer,
reasons about the root cause, and produces a structured FixPlan for the Patcher.

Inputs  (from Person 3 / Localizer):  ContextBundle
Outputs (to   Person 4 / Patcher):    FixPlan

Retry handling:
  - If Person 5 (Validator) detects a test failure, it routes back here
    with retry_reason="test_failure" and retry_error_message=<test output>.
  - The Diagnostician re-reads the test output, revises its hypothesis,
    and emits an updated FixPlan (max MAX_RETRIES attempts).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Optional
from openai import OpenAI

from agents.schemas import (
    AffectedRegion,
    ContextBundle,
    FileContext,
    FixPlan,
)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL = "llama4-scout-17b"
BASE_URL = "https://openai.rc.asu.edu/v1"
MAX_TOKENS = 4096
MAX_RETRIES = 2          # Maximum times Validator can route back here


# ── Prompt builders ───────────────────────────────────────────────────────────

def _format_file_block(fc: FileContext, max_lines: int = 300) -> str:
    """Format a single file for inclusion in the prompt."""
    lines = fc.content.splitlines()
    if len(lines) > max_lines:
        half = max_lines // 2
        truncated = (
            lines[:half]
            + [f"\n... [{len(lines) - max_lines} lines omitted] ...\n"]
            + lines[-half:]
        )
        body = "\n".join(truncated)
    else:
        body = fc.content

    numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(body.splitlines()))
    return f"### FILE: {fc.path}  (relevance={fc.relevance_score:.2f})\n```c\n{numbered}\n```"


def _build_system_prompt() -> str:
    return textwrap.dedent("""\
        You are the Diagnostician agent in a multi-agent software engineering system.
        Your job is to read a bug report and the actual C source files retrieved from
        the repository, then produce a precise, structured fix plan.

        Rules:
        1. Base your analysis ONLY on the actual code shown — never hallucinate content.
        2. Cite exact file paths and line numbers from the numbered source listings.
        3. Be specific: identify the exact variable, condition, or logic that is wrong.
        4. Consider test constraints — your fix must not break passing tests.
        5. Output ONLY the JSON object described. No prose before or after it.

        Output format (strict JSON, no markdown fences):
        {
          "root_cause": "<one paragraph explaining WHY the bug exists>",
          "fix_description": "<what the patch should do, step by step>",
          "affected_files": ["<path>", ...],
          "affected_regions": [
            {"file_path": "<path>", "start_line": <int>, "end_line": <int>, "reason": "<why>"},
            ...
          ],
          "test_constraints": ["<thing the fix must preserve or satisfy>", ...],
          "suggested_approach": "<high-level strategy for generating the diff>",
          "confidence": <0.0–1.0>,
          "requires_new_function": <true|false>,
          "requires_header_change": <true|false>,
          "diagnostician_notes": "<any extra notes for the Patcher>"
        }
    """)


def _build_user_prompt(bundle: ContextBundle, retry_context: Optional[str] = None) -> str:
    parts: list[str] = []

    # Issue
    parts.append("## ISSUE REPORT")
    parts.append(f"**ID:** {bundle.issue_id}")
    parts.append(f"**Repo:** {bundle.repo}  |  **Commit:** {bundle.base_commit}")
    parts.append(f"**Type:** {bundle.planner_issue_type or 'unknown'}")
    parts.append(f"**Keywords:** {', '.join(bundle.planner_keywords)}")
    parts.append(f"\n**Title:** {bundle.issue_title}")
    parts.append(f"\n**Description:**\n{bundle.issue_body}")

    # Retry context (test failure routed back from Validator)
    if retry_context:
        parts.append("\n## PREVIOUS FIX FAILED — TEST OUTPUT")
        parts.append(
            "Your previous fix plan produced a patch that failed tests. "
            "Read the test output carefully and revise your root cause hypothesis.\n"
        )
        parts.append(f"```\n{retry_context}\n```")

    # Source files
    parts.append("\n## LOCALIZED SOURCE FILES")
    parts.append(
        "These files were identified as relevant by the Localizer agent. "
        "Line numbers are shown in the left margin."
    )
    for fc in bundle.source_files:
        parts.append(_format_file_block(fc))

    # Test files
    if bundle.test_files:
        parts.append("\n## RELEVANT TEST FILES")
        parts.append(
            "Use these to understand what the correct behavior should be."
        )
        for fc in bundle.test_files:
            parts.append(_format_file_block(fc, max_lines=150))

    # Task
    parts.append("\n## YOUR TASK")
    parts.append(
        "Analyze the bug described in the issue. "
        "Using only the source code shown above, identify the root cause "
        "and produce the structured JSON fix plan. "
        "Be precise about file paths and line numbers — the Patcher will use "
        "them to generate the actual diff."
    )

    return "\n\n".join(parts)


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Extract the JSON object from the LLM response.
    Handles responses that may have accidental markdown fences.
    """
    # Strip ```json ... ``` fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Find the outermost { ... }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM response:\n{text[:500]}")

    return json.loads(text[start:end])


# ── FixPlan assembly ──────────────────────────────────────────────────────────

def _assemble_fix_plan(issue_id: str, raw: dict) -> FixPlan:
    """Convert the raw JSON dict from the LLM into a typed FixPlan."""
    regions = [
        AffectedRegion(
            file_path=r["file_path"],
            start_line=int(r["start_line"]),
            end_line=int(r["end_line"]),
            reason=r.get("reason", ""),
        )
        for r in raw.get("affected_regions", [])
    ]
    return FixPlan(
        issue_id=issue_id,
        root_cause=raw.get("root_cause", ""),
        fix_description=raw.get("fix_description", ""),
        affected_files=raw.get("affected_files", []),
        affected_regions=regions,
        test_constraints=raw.get("test_constraints", []),
        suggested_approach=raw.get("suggested_approach", ""),
        confidence=float(raw.get("confidence", 0.5)),
        requires_new_function=bool(raw.get("requires_new_function", False)),
        requires_header_change=bool(raw.get("requires_header_change", False)),
        diagnostician_notes=raw.get("diagnostician_notes", ""),
    )


# ── Test runner (optional — runs existing tests to observe failures) ──────────

def run_failing_tests(
    repo_root: str,
    test_files: list[FileContext],
    timeout: int = 60,
) -> str:
    """
    Attempt to run the repository's test suite and capture output.

    This gives the Diagnostician real test failure messages to reason about.
    Returns the combined stdout+stderr string (empty string on error).
    """
    repo_path = Path(repo_root)
    if not repo_path.exists():
        return ""

    # Try `make check` / `make test` — common in C repos
    for target in ("check", "test", "tests"):
        result = subprocess.run(
            ["make", target],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        # Return as soon as we get any meaningful output
        if output.strip():
            # Truncate to avoid giant prompts
            if len(output) > 4000:
                output = output[:2000] + "\n...[truncated]...\n" + output[-2000:]
            return output

    return ""


# ── Main public API ───────────────────────────────────────────────────────────

def diagnose(
    bundle: ContextBundle,
    *,
    run_tests: bool = False,
    retry_error_message: Optional[str] = None,
    attempt: int = 1,
    api_key: Optional[str] = None,
) -> FixPlan:
    """
    Run the Diagnostician agent on a ContextBundle.

    Args:
        bundle:              Output from the Localizer (Person 3).
        run_tests:           If True, run `make check` first and feed output to LLM.
        retry_error_message: Test failure output routed back from Validator.
                             When set, the agent re-diagnoses with this context.
        attempt:             Which retry attempt this is (1 = first run).
        api_key:             Anthropic API key; falls back to ANTHROPIC_API_KEY env var.

    Returns:
        A FixPlan ready for the Patcher agent.
    """
    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
    )
    # Optionally gather live test output
    test_output: Optional[str] = retry_error_message
    if run_tests and not test_output and bundle.repo_root:
        print(f"  [Diagnostician] Running tests in {bundle.repo_root} ...")
        test_output = run_failing_tests(bundle.repo_root, bundle.test_files)
        if test_output:
            print(f"  [Diagnostician] Captured {len(test_output)} chars of test output")

    system = _build_system_prompt()
    user = _build_user_prompt(bundle, retry_context=test_output)

    print(f"  [Diagnostician] Calling LLM (attempt {attempt}) for {bundle.issue_id} ...")
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw_text = response.choices[0].message.content
    raw_dict = _extract_json(raw_text)
    plan = _assemble_fix_plan(bundle.issue_id, raw_dict)

    if retry_error_message:
        plan.retry_reason = "test_failure"
        plan.retry_error_message = retry_error_message

    print(
        f"  [Diagnostician] Done. confidence={plan.confidence:.2f}, "
        f"files={plan.affected_files}, regions={len(plan.affected_regions)}"
    )
    return plan


# ── Retry entry point (called by Validator / orchestrator) ────────────────────

def rediagnose(
    bundle: ContextBundle,
    error_message: str,
    attempt: int = 2,
    api_key: Optional[str] = None,
) -> FixPlan:
    """
    Re-run diagnosis after a test failure (Loop C in the proposal).

    The Validator calls this when `make test` fails after applying the patch.
    The error_message is the test runner's stdout+stderr.
    """
    if attempt > MAX_RETRIES + 1:
        raise RuntimeError(
            f"Diagnostician exceeded MAX_RETRIES ({MAX_RETRIES}) for {bundle.issue_id}"
        )
    return diagnose(
        bundle,
        run_tests=False,         # We already have the error from Validator
        retry_error_message=error_message,
        attempt=attempt,
        api_key=api_key,
    )
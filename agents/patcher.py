"""
patcher.py — Patcher Agent (Agent 4)

Reads the FixPlan from the Diagnostician and the actual source files,
then generates a syntactically correct unified diff ready for `git apply`.

The key guarantee: context lines in the diff are copied from REAL file
content, not hallucinated. This directly solves the 90%+ localization
failure from Phase 1.

Inputs  (from Diagnostician):  FixPlan + real file contents
Outputs (to Person 5):         PatchResult containing a unified diff

Retry handling:
  - If Person 5 (Validator) detects a compilation error, it routes back
    here with retry_reason="compilation_error" and the compiler output.
  - The Patcher re-generates the diff with the error appended (max MAX_RETRIES).
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from typing import Optional

from openai import OpenAI
from agents.schemas import FileContext, FixPlan, PatchResult

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL = "llama4-scout-17b"
BASE_URL = "https://openai.rc.asu.edu/v1"
MAX_TOKENS = 4096
MAX_RETRIES = 2


# ── File reading ──────────────────────────────────────────────────────────────

def _read_file(repo_root: str, rel_path: str) -> Optional[str]:
    """Read a file from disk. Returns None if not found."""
    full = Path(repo_root) / rel_path
    if full.exists():
        try:
            return full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    return None


def _load_affected_files(
    plan: FixPlan,
    repo_root: str,
    preloaded: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """
    Return {rel_path: content} for every file in plan.affected_files.

    Preloaded content (from the ContextBundle) is preferred; disk is the
    fallback to ensure we always have real content.
    """
    result: dict[str, str] = {}
    for rel_path in plan.affected_files:
        if preloaded and rel_path in preloaded:
            result[rel_path] = preloaded[rel_path]
        else:
            content = _read_file(repo_root, rel_path)
            if content is not None:
                result[rel_path] = content
            else:
                print(f"  [Patcher] WARNING: cannot read {rel_path} from {repo_root}")
    return result

def _format_file_with_lines(path: str, content: str) -> str:
    lines = content.splitlines()
    numbered = "\n".join(f"{i+1:5d} | {line}" for i, line in enumerate(lines))
    return (
        f"### {path}\n"
        f"<!-- Line numbers shown for reference only. "
        f"Do NOT include them in the diff context lines. -->\n"
        f"```c\n{numbered}\n```"
    )


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return textwrap.dedent("""\
        You are the Patcher agent in a multi-agent software engineering system.
        Your job is to generate a valid unified diff that implements a fix plan.

        CRITICAL RULES — the patch MUST pass `git apply` on the first try:

        1. COPY context lines EXACTLY from the file listings shown. Do NOT paraphrase,
           reformat, or reconstruct them from memory. Character-for-character accuracy.
        2. Use the correct unified diff format:
             --- a/<path>
             +++ b/<path>
             @@ -<start>,<count> +<start>,<count> @@
           followed by context lines (space prefix), removals (- prefix), additions (+ prefix).
        3. Include 3 lines of unchanged context before and after each changed region.
        4. Line numbers in @@ headers must match the actual file content shown.
        5. Do NOT create new files unless the fix plan explicitly says requires_new_function
           and the function must live in a new file.
        6. If multiple files need changes, concatenate their hunks in one diff output.
        7. Output ONLY the raw unified diff. No prose, no markdown fences, no explanation.
           Start your response with "diff --git" or "---".
        8. Context lines in the diff must NOT include the line number margin
            (e.g. '  12 | ') — copy only the actual code after the pipe character.

        If a compilation error is provided, read it carefully:
        - An "undeclared identifier" means you need to add a declaration or #include.
        - A "type mismatch" means you need to cast or change the type.
        - Fix only what the error describes — do not rewrite unrelated code.
    """)


def _build_user_prompt(
    plan: FixPlan,
    file_contents: dict[str, str],
    compilation_error: Optional[str] = None,
) -> str:
    parts: list[str] = []

    # Fix plan summary
    parts.append("## FIX PLAN FROM DIAGNOSTICIAN")
    parts.append(f"**Issue ID:** {plan.issue_id}")
    parts.append(f"**Root cause:** {plan.root_cause}")
    parts.append(f"**What the patch must do:**\n{plan.fix_description}")
    parts.append(f"**Suggested approach:** {plan.suggested_approach}")

    if plan.diagnostician_notes:
        parts.append(f"**Diagnostician notes:** {plan.diagnostician_notes}")

    if plan.test_constraints:
        parts.append("**Test constraints (must not break):**")
        for c in plan.test_constraints:
            parts.append(f"  - {c}")

    # Affected regions
    if plan.affected_regions:
        parts.append("\n**Affected regions (focus your changes here):**")
        for r in plan.affected_regions:
            parts.append(
                f"  - `{r.file_path}` lines {r.start_line}–{r.end_line}: {r.reason}"
            )

    # Compilation error retry context
    if compilation_error:
        parts.append("\n## COMPILATION ERROR — YOUR PREVIOUS PATCH DID NOT COMPILE")
        parts.append(
            "Read the compiler output below and fix ONLY the compilation issue. "
            "Do not change unrelated code.\n"
        )
        parts.append(f"```\n{compilation_error}\n```")

    # Actual file contents
    parts.append("\n## ACTUAL FILE CONTENTS")
    parts.append(
        "The files below contain the EXACT current source. "
        "You MUST copy context lines verbatim from these listings. "
        "Line numbers are shown in the left margin."
    )
    for rel_path, content in file_contents.items():
        parts.append(_format_file_with_lines(rel_path, content))

    # Task
    parts.append("\n## YOUR TASK")
    parts.append(
        "Generate a unified diff that implements the fix plan above. "
        "Every context line must be copied exactly from the file listings. "
        "Output ONLY the diff — nothing else."
    )

    return "\n\n".join(parts)


# ── Diff validation (lightweight, before handing to Validator) ────────────────

def _basic_diff_checks(diff: str, file_contents: dict[str, str]) -> list[str]:
    """
    Run quick sanity checks on the generated diff.
    Returns a list of warning strings (empty = looks OK).
    """
    warnings: list[str] = []

    if not diff.strip():
        warnings.append("Empty diff generated")
        return warnings

    if "---" not in diff or "+++" not in diff:
        warnings.append("Diff missing --- / +++ headers")

    if "@@ " not in diff:
        warnings.append("Diff has no hunk headers (@@)")

    # Check that files referenced in diff exist in our content dict
    for line in diff.splitlines():
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            path = line.split("/", 1)[1].strip()
            if path not in file_contents and path != "/dev/null":
                warnings.append(f"Diff references unknown file: {path}")

    return warnings


def _extract_modified_files(diff: str) -> list[str]:
    """Parse the diff to find which files are being modified."""
    files: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
            if path not in files:
                files.append(path)
    return files


# ── Main public API ───────────────────────────────────────────────────────────

def patch(
    plan: FixPlan,
    repo_root: str,
    *,
    preloaded_files: Optional[dict[str, str]] = None,
    compilation_error: Optional[str] = None,
    attempt: int = 1,
    api_key: Optional[str] = None,
) -> PatchResult:
    """
    Run the Patcher agent to generate a unified diff.

    Args:
        plan:               FixPlan from the Diagnostician.
        repo_root:          Absolute path to the repository on disk.
        preloaded_files:    {rel_path: content} pre-read by earlier agents.
                            Used so we don't re-read files unnecessarily.
        compilation_error:  Compiler output routed back from Validator (Loop B).
        attempt:            Retry attempt number.
        api_key:            Anthropic API key; falls back to ANTHROPIC_API_KEY env var.

    Returns:
        PatchResult with the unified diff and metadata.
    """
    client = OpenAI(
    base_url=BASE_URL,
    api_key=api_key or os.environ.get("OPENAI_API_KEY"),
    )

    # Load the actual file contents — this is what makes the diff reliable
    file_contents = _load_affected_files(plan, repo_root, preloaded=preloaded_files)
    if not file_contents:
        raise RuntimeError(
            f"[Patcher] Could not read any affected files for {plan.issue_id}. "
            f"Affected: {plan.affected_files}, repo_root: {repo_root}"
        )

    system = _build_system_prompt()
    user = _build_user_prompt(plan, file_contents, compilation_error=compilation_error)

    print(f"  [Patcher] Calling LLM (attempt {attempt}) for {plan.issue_id} ...")
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    diff_text = response.choices[0].message.content.strip()

    # Strip accidental markdown fences if the model added them
    if diff_text.startswith("```"):
        diff_text = re.sub(r"^```[a-z]*\n?", "", diff_text)
        diff_text = re.sub(r"\n?```$", "", diff_text).strip()

    # Sanity checks
    warnings = _basic_diff_checks(diff_text, file_contents)
    if warnings:
        print(f"  [Patcher] Diff warnings: {warnings}")

    modified = _extract_modified_files(diff_text)
    notes = f"attempt={attempt}"
    if warnings:
        notes += f"; warnings={warnings}"
    if compilation_error:
        notes += "; retry_after_compilation_error"

    result = PatchResult(
        issue_id=plan.issue_id,
        patch=diff_text,
        modified_files=modified or plan.affected_files,
        fix_plan=plan,
        patcher_notes=notes,
        attempt_number=attempt,
    )

    print(
        f"  [Patcher] Done. {len(diff_text)} chars, "
        f"files={result.modified_files}, warnings={len(warnings)}"
    )
    return result


# ── Retry entry point (called by Validator on compilation error) ──────────────

def repatch(
    plan: FixPlan,
    repo_root: str,
    compilation_error: str,
    attempt: int = 2,
    preloaded_files: Optional[dict[str, str]] = None,
    api_key: Optional[str] = None,
) -> PatchResult:
    """
    Re-generate the patch after a compilation error (Loop B in the proposal).

    The Validator calls this when `gcc`/`make` rejects the patch.
    The compilation_error is the compiler's stdout+stderr.
    """
    if attempt > MAX_RETRIES + 1:
        raise RuntimeError(
            f"Patcher exceeded MAX_RETRIES ({MAX_RETRIES}) for {plan.issue_id}"
        )
    plan.retry_reason = "compilation_error"
    plan.retry_error_message = compilation_error

    return patch(
        plan,
        repo_root,
        preloaded_files=preloaded_files,
        compilation_error=compilation_error,
        attempt=attempt,
        api_key=api_key,
    )
"""Patcher agent — generates a unified diff from a FixPlan and real source files.

Implements the interface defined in agents/stubs.py:

    patch(instance, fix_plan, bundle)                    → PatchOutput
    patch_with_feedback(instance, fix_plan, bundle, fb)  → PatchOutput

The core guarantee: context lines in every diff hunk are copied from REAL file
content (read from bundle.file_contents, fallback to disk).  This directly
addresses the 90%+ git-apply failure rate from Phase 1.

Usage
-----
    from agents.patcher import PatcherAgent
    agent = PatcherAgent()
    result = agent.patch(instance, fix_plan, bundle)

    # On compilation error from Validator (Loop B):
    result = agent.patch_with_feedback(instance, fix_plan, bundle, feedback)
"""

from __future__ import annotations

import logging
import os
import re
import textwrap
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as genai_types

from pipeline.schema import (
    ContextBundle,
    FeedbackMessage,
    FixPlan,
    PatchOutput,
    SWEInstance,
)

logger = logging.getLogger(__name__)

# ── Model config ──────────────────────────────────────────────────────────────

_DEFAULT_MODEL = "gemini-2.5-flash"
_MAX_TOKENS       = 4096
_MAX_RETRIES      = 2


# ── File loading ──────────────────────────────────────────────────────────────

def _get_file_contents(
    fix_plan: FixPlan,
    bundle: ContextBundle,
    repo_root: str,
) -> dict[str, str]:
    """
    Return {rel_path: content} for every file in fix_plan.affected_files.

    Priority: bundle.file_contents (already in memory) → disk fallback.
    This ensures context lines always come from real source, never hallucination.
    """
    result: dict[str, str] = {}
    for rel in fix_plan.affected_files:
        if rel in bundle.file_contents:
            result[rel] = bundle.file_contents[rel]
        elif repo_root:
            full = Path(repo_root) / rel
            try:
                result[rel] = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                logger.warning("[Patcher] Cannot read %s from disk", full)
        else:
            logger.warning("[Patcher] %s not in bundle and no repo_root", rel)
    return result


def _numbered(content: str, limit: int = 300) -> str:
    """Return file content with 1-indexed line numbers."""
    lines = content.splitlines()
    if len(lines) > limit:
        half = limit // 2
        middle = [f"  ... [{len(lines) - limit} lines omitted] ..."]
        lines = lines[:half] + middle + lines[-half:]
    return "\n".join(f"{i + 1:5d} | {ln}" for i, ln in enumerate(lines))


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = textwrap.dedent("""\
    You are the Patcher agent in a multi-agent software engineering pipeline.
    You generate a valid unified diff that implements a given fix plan.

    CRITICAL — the diff MUST pass `git apply` on the first attempt:

    1. Context lines MUST be copied EXACTLY from the file listings provided.
       Do NOT paraphrase, reformat, or reconstruct them.  Character-for-character.
    2. Do NOT include the line-number margin (e.g. "   12 | ") in the diff.
       Copy only the actual source text that appears after the pipe character.
    3. Unified diff format:
         --- a/<path>
         +++ b/<path>
         @@ -<start>,<count> +<start>,<count> @@
       followed by lines prefixed with ' ' (context), '-' (removed), '+' (added).
    4. Include exactly 3 lines of unchanged context before and after each change.
    5. @@ line numbers must match the actual file content shown.
    6. For multiple files, concatenate hunks — one diff --git header per file.
    7. Output ONLY the raw unified diff.  Start with "diff --git" or "---".
       No prose, no markdown fences, no explanation whatsoever.

    When a compilation error is provided:
    - Read the error message carefully.
    - An "undeclared identifier" → add the missing #include or declaration.
    - A "type mismatch" → add a cast or correct the type.
    - Fix only what the error describes — do not rewrite unrelated code.
""")


# ── User prompt builders ──────────────────────────────────────────────────────

def _build_patch_prompt(
    instance: SWEInstance,
    fix_plan: FixPlan,
    file_contents: dict[str, str],
    feedback: FeedbackMessage | None = None,
) -> str:
    sections: list[str] = []

    sections.append("## Fix plan")
    sections.append(
        f"**Instance:** {instance.instance_id}\n"
        f"**Root cause:** {fix_plan.root_cause}\n\n"
        f"**What the patch must do:**\n{fix_plan.fix_description}"
    )

    if fix_plan.affected_regions:
        sections.append("**Affected regions (focus your changes here):**")
        for r in fix_plan.affected_regions:
            sections.append(
                f"  - `{r.file_path}` lines {r.start_line}–{r.end_line}: {r.description}"
            )

    if fix_plan.test_constraints:
        sections.append("**Must not break:**")
        for c in fix_plan.test_constraints:
            sections.append(f"  - {c}")

    # Retry context from Validator
    if feedback is not None:
        sections.append("## Previous patch rejected — validator evidence")
        sections.append(
            f"Failure type: **{feedback.failure_type.value}**\n\n"
            f"```\n{feedback.evidence}\n```\n\n"
            "Fix only what the evidence describes. Do not rewrite unrelated code."
        )

    sections.append("## Actual file contents")
    sections.append(
        "Copy context lines VERBATIM from these listings — "
        "only the text AFTER the pipe character, not the line number."
    )
    for rel, content in file_contents.items():
        sections.append(f"### {rel}\n```c\n{_numbered(content)}\n```")

    sections.append("## Task")
    sections.append(
        "Generate the unified diff that implements the fix plan above. "
        "Output ONLY the diff — nothing else."
    )
    return "\n\n".join(sections)


# ── Diff cleanup ──────────────────────────────────────────────────────────────

def _clean_diff(text: str) -> str:
    """Strip markdown fences if the model added them despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_modified_files(diff: str) -> list[str]:
    """Parse +++ b/<path> lines to find which files the diff touches."""
    files: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
            if path not in files:
                files.append(path)
    return files


def _basic_diff_sanity(diff: str) -> list[str]:
    """Return a list of warning strings (empty = looks OK)."""
    warnings: list[str] = []
    if not diff.strip():
        warnings.append("empty diff")
        return warnings
    if "---" not in diff or "+++" not in diff:
        warnings.append("missing --- / +++ headers")
    if "@@ " not in diff:
        warnings.append("no hunk headers (@@)")
    # Detect leftover line-number margins like "   12 | "
    if re.search(r"^[ +-]\s{0,5}\d+\s+\|", diff, re.MULTILINE):
        warnings.append("line-number margin leaked into diff context lines")
    return warnings


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(
    client,
    model: str,
    system: str,
    user: str,
) -> str:
    """Call the LLM and return the raw text response, retrying on failure."""
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 2):
        if attempt > 1:
            wait = 2 ** attempt
            logger.warning("Patcher LLM retry %d after %ds", attempt, wait)
            time.sleep(wait)
        try:
            response = client.models.generate_content(
                model=model,
                contents=f"{system}\n\n{user}",
                config=genai_types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=_MAX_TOKENS,
                ),
            )
            return response.text or ""
        except Exception as exc:
            logger.error("Patcher LLM call failed (attempt %d): %s", attempt, exc)
            last_exc = exc
    raise RuntimeError(
        f"Patcher failed after {_MAX_RETRIES + 1} attempts"
    ) from last_exc


# ── Public agent class ────────────────────────────────────────────────────────

class PatcherAgent:
    """LLM-backed Patcher agent.

    Implements the interface required by pipeline/controller.py:
        patch(instance, fix_plan, bundle)                   → PatchOutput
        patch_with_feedback(instance, fix_plan, bundle, fb) → PatchOutput

    Args:
        model_name: Gemini model identifier.
        api_key:    Gemini API key. If None, falls back to GEMINI_API_KEY /
                    GOOGLE_API_KEY env vars (matches the Planner).
        repo_root:  Optional override for where to read files from disk.
                    In normal use the bundle already contains file_contents.
                    The controller may reset this per-instance via
                    `agent.set_repo_root(...)`.

    Environment variables:
        GEMINI_API_KEY:    Gemini API key (preferred)
        GOOGLE_API_KEY:    Fallback key read by the google-genai SDK
        MODEL_NAME:        Model to use (default: gemini-2.5-flash)
        VERTEXAI_PROJECT:  Optional — if set, uses Vertex AI instead of the
                           simple API-key path. VERTEXAI_LOCATION defaults
                           to us-central1.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        repo_root: str = "",
    ) -> None:
        self._model = model_name or os.environ.get("MODEL_NAME", _DEFAULT_MODEL)
        self._repo_root = repo_root

        project = os.environ.get("VERTEXAI_PROJECT")
        if project:
            location = os.environ.get("VERTEXAI_LOCATION", "us-central1")
            self._client = genai.Client(
                vertexai=True, project=project, location=location,
            )
        else:
            self._client = genai.Client(
                api_key=api_key or os.environ.get("GEMINI_API_KEY")
            )

    def set_repo_root(self, repo_root: str) -> None:
        self._repo_root = repo_root

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run(
        self,
        instance: SWEInstance,
        fix_plan: FixPlan,
        bundle: ContextBundle,
        feedback: FeedbackMessage | None,
        attempt: int,
    ) -> PatchOutput:
        repo_root = self._repo_root or getattr(instance, "repo_root", "")
        file_contents = _get_file_contents(fix_plan, bundle, repo_root)

        if not file_contents:
            logger.error(
                "[%s] Patcher: no file contents available for %s",
                instance.instance_id,
                fix_plan.affected_files,
            )
            # Return empty patch rather than crashing — Validator will handle it
            return PatchOutput(
                instance_id=instance.instance_id,
                unified_diff="",
                affected_files=fix_plan.affected_files,
            )

        user = _build_patch_prompt(instance, fix_plan, file_contents, feedback)
        logger.info(
            "[%s] Patcher calling LLM (attempt %d), files=%s",
            instance.instance_id, attempt, list(file_contents.keys()),
        )

        raw  = _call_llm(self._client, self._model, _SYSTEM, user)
        diff = _clean_diff(raw)

        warnings = _basic_diff_sanity(diff)
        if warnings:
            logger.warning("[%s] Patcher diff warnings: %s", instance.instance_id, warnings)

        modified = _extract_modified_files(diff) or fix_plan.affected_files
        logger.info(
            "[%s] Patcher done: %d chars, files=%s, warnings=%s",
            instance.instance_id, len(diff), modified, warnings,
        )
        return PatchOutput(
            instance_id=instance.instance_id,
            unified_diff=diff,
            affected_files=modified,
        )

    # ── Public methods (match stub interface exactly) ─────────────────────────

    def patch(
        self,
        instance: SWEInstance,
        fix_plan: FixPlan,
        bundle: ContextBundle,
    ) -> PatchOutput:
        """Generate a unified diff from the fix plan and localised source code.

        Called by the pipeline controller on the first attempt.
        """
        return self._run(instance, fix_plan, bundle, feedback=None, attempt=1)

    def patch_with_feedback(
        self,
        instance: SWEInstance,
        fix_plan: FixPlan,
        bundle: ContextBundle,
        feedback: FeedbackMessage,
    ) -> PatchOutput:
        """Retry patch generation with compiler/apply error evidence appended.

        Called on Loop B (compilation error) routed back from the Validator.
        """
        logger.info(
            "[%s] Patcher.patch_with_feedback — failure=%s retry=%d",
            instance.instance_id,
            feedback.failure_type.value,
            feedback.retry_number,
        )
        return self._run(
            instance, fix_plan, bundle, feedback=feedback,
            attempt=feedback.retry_number + 1,
        )
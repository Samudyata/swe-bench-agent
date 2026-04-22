"""Diagnostician agent — reads localised context and produces a structured fix plan.

Implements the interface defined in agents/stubs.py:

    diagnose(instance, bundle)         → FixPlan   (first attempt)
    revise(instance, bundle, feedback) → FixPlan   (retry after test/apply failure)

The agent sends the issue description + real source file contents to an LLM and
parses the JSON response into a typed FixPlan.  Real file content comes from the
ContextBundle that Person 3's Localizer already populated, so the LLM reasons
over actual code, not hallucinated reconstructions.

Usage
-----
    from agents.diagnostician import DiagnosticianAgent
    agent = DiagnosticianAgent()
    fix_plan = agent.diagnose(instance, bundle)

    # On test-failure feedback from Validator (Loop C):
    revised  = agent.revise(instance, bundle, feedback)
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from typing import Any

from agents.llm import DEFAULT_MODEL as _DEFAULT_MODEL
from agents.llm import chat, make_client
from pipeline.schema import (
    AffectedRegion,
    ContextBundle,
    FeedbackMessage,
    FixPlan,
    LocalizerCandidate,
    SWEInstance,
)

logger = logging.getLogger(__name__)

# ── Model config ──────────────────────────────────────────────────────────────

_MAX_TOKENS       = 4096
_MAX_RETRIES      = 2          # LLM call retries on parse failure
_FILE_LINE_LIMIT  = 300        # max lines per file shown in prompt


# ── Prompt helpers ────────────────────────────────────────────────────────────

def _numbered(content: str, limit: int = _FILE_LINE_LIMIT) -> str:
    """Return file content with 1-indexed line numbers, optionally truncated."""
    lines = content.splitlines()
    if len(lines) > limit:
        half = limit // 2
        middle = [f"  ... [{len(lines) - limit} lines omitted] ..."]
        lines = lines[:half] + middle + lines[-half:]
    return "\n".join(f"{i + 1:5d} | {ln}" for i, ln in enumerate(lines))


def _format_files(bundle: ContextBundle) -> str:
    """Format all source files from the bundle for the prompt."""
    parts: list[str] = []
    for path, content in bundle.file_contents.items():
        parts.append(f"### {path}\n```c\n{_numbered(content)}\n```")
    # Also include snippet-level context if available
    if bundle.relevant_snippets:
        parts.append("### Relevant function snippets")
        for node_id, snippet in bundle.relevant_snippets.items():
            parts.append(f"#### {node_id}\n```c\n{snippet}\n```")
    return "\n\n".join(parts)


def _format_candidates(candidates: list[LocalizerCandidate]) -> str:
    lines = []
    for c in candidates:
        lines.append(
            f"  - {c.file_path}  (score={c.score:.3f}, {c.reason})"
            + (f"\n    functions: {c.functions}" if c.functions else "")
        )
    return "\n".join(lines)


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = textwrap.dedent("""\
    You are the Diagnostician agent in a multi-agent software engineering pipeline.
    You will be given a GitHub issue and the relevant C source files from the repository.
    Your job is to identify the root cause of the bug and produce a precise fix plan.

    Rules:
    1. Base every claim on the actual code shown — never invent content.
    2. Cite exact file paths and line numbers from the numbered listings.
    3. Be specific: name the exact variable, condition, or logic that is wrong.
    4. Your fix plan will be used by a Patcher agent that generates the actual diff —
       be as concrete as possible about what needs to change and where.
    5. Output ONLY a single JSON object. No prose before or after it.
       No markdown fences. No explanation outside the JSON.

    Required JSON schema (all fields mandatory):
    {
      "root_cause": "<one paragraph: why does this bug exist?>",
      "fix_description": "<step-by-step: what must the patch do?>",
      "affected_files": ["<repo-relative path>", ...],
      "affected_regions": [
        {
          "file_path": "<repo-relative path>",
          "start_line": <int>,
          "end_line": <int>,
          "description": "<why this region needs to change>"
        }
      ],
      "test_constraints": ["<what the fix must not break>", ...],
      "fix_description": "<high-level strategy for the Patcher>",
      "confidence": <float 0.0–1.0>
    }
""")


# ── User prompt builders ──────────────────────────────────────────────────────

def _build_diagnose_prompt(instance: SWEInstance, bundle: ContextBundle) -> str:
    sections: list[str] = []

    sections.append("## Issue report")
    sections.append(
        f"**Repo:** {instance.repo}  |  **Commit:** {instance.base_commit}\n"
        f"**Instance:** {instance.instance_id}\n\n"
        f"{instance.problem_statement.strip()}"
    )

    if instance.hints_text and instance.hints_text.strip():
        sections.append("## Additional hints\n" + instance.hints_text.strip())

    sections.append("## Localizer candidates (ranked by relevance)")
    sections.append(_format_candidates(bundle.candidates))

    sections.append("## Source files")
    sections.append(
        "Line numbers are in the left margin. "
        "Do NOT include them when citing lines — use the integer only."
    )
    sections.append(_format_files(bundle))

    if bundle.test_files:
        sections.append("## Test files")
        sections.append(
            "Use these to understand what correct behaviour looks like."
        )
        for tf in bundle.test_files:
            content = bundle.file_contents.get(tf, "")
            if content:
                sections.append(f"### {tf}\n```c\n{_numbered(content, limit=150)}\n```")

    sections.append("## Task")
    sections.append(
        "Analyse the bug. Using only the source code above, "
        "identify the root cause and output the JSON fix plan."
    )
    return "\n\n".join(sections)


def _build_revise_prompt(
    instance: SWEInstance,
    bundle: ContextBundle,
    feedback: FeedbackMessage,
) -> str:
    base = _build_diagnose_prompt(instance, bundle)
    revision = textwrap.dedent(f"""\
        ## Previous fix failed — validator evidence

        Your previous fix plan produced a patch that was rejected.
        Failure type: **{feedback.failure_type.value}**
        Retry number: {feedback.retry_number}

        Evidence:
        ```
        {feedback.evidence}
        ```

        Re-examine the code with this evidence in mind.
        Revise your root cause hypothesis and produce an updated JSON fix plan.
    """)
    return base + "\n\n" + revision


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Remove accidental ```json ... ``` fences."""
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def _parse_fix_plan(instance_id: str, raw: dict[str, Any]) -> FixPlan:
    regions: list[AffectedRegion] = []
    for r in raw.get("affected_regions", []):
        regions.append(
            AffectedRegion(
                file_path=r["file_path"],
                start_line=int(r["start_line"]),
                end_line=int(r["end_line"]),
                description=r.get("description", r.get("reason", "")),
            )
        )
    return FixPlan(
        instance_id=instance_id,
        root_cause=str(raw.get("root_cause", "")),
        fix_description=str(raw.get("fix_description", "")),
        affected_files=list(raw.get("affected_files", [])),
        affected_regions=regions,
        test_constraints=list(raw.get("test_constraints", [])),
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(
    client,
    model: str,
    system: str,
    user: str,
) -> str:
    return chat(
        client=client,
        model=model,
        system=system,
        user=user,
        temperature=0.2,
        max_tokens=_MAX_TOKENS,
        max_retries=_MAX_RETRIES,
        agent_name="Diagnostician",
    )


# ── Public agent class ────────────────────────────────────────────────────────

class DiagnosticianAgent:
    """LLM-backed Diagnostician agent.

    Implements the interface required by pipeline/controller.py:
        diagnose(instance, bundle)         → FixPlan
        revise(instance, bundle, feedback) → FixPlan

    Args:
        model_name: OpenAI-compatible model identifier
                    (default: qwen3-30b-a3b-instruct-2507).
        api_key:    Voyager API key. If None, reads OPENAI_API_KEY env.
        base_url:   Voyager base URL. If None, reads OPENAI_API_BASE env.

    Environment variables:
        OPENAI_API_KEY:  Voyager API key
        OPENAI_API_BASE: Voyager base URL (default: https://openai.rc.asu.edu/v1)
        MODEL_NAME:      Model to use (default: qwen3-30b-a3b-instruct-2507)
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model_name or os.environ.get("MODEL_NAME", _DEFAULT_MODEL)
        self._client = make_client(api_key=api_key, base_url=base_url)

    # ── Public methods (match stub interface exactly) ─────────────────────────
    def diagnose(self, instance, bundle):
        logger.info(
            "[%s] Diagnostician.diagnose — %d files, confidence=%.2f",
            instance.instance_id,
            len(bundle.file_contents),
            bundle.confidence,
        )
        user = _build_diagnose_prompt(instance, bundle)
        text = _call_llm(self._client, self._model, _SYSTEM, user)
        text = _strip_fences(text)
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object in response: {text[:300]}")
        raw  = json.loads(text[start:end])
        plan = _parse_fix_plan(instance.instance_id, raw)
        logger.info(
            "[%s] FixPlan: files=%s, regions=%d",
            instance.instance_id,
            plan.affected_files,
            len(plan.affected_regions),
        )
        return plan

    def revise(self, instance, bundle, feedback):
        logger.info(
            "[%s] Diagnostician.revise — failure=%s retry=%d",
            instance.instance_id,
            feedback.failure_type.value,
            feedback.retry_number,
        )
        user = _build_revise_prompt(instance, bundle, feedback)
        text = _call_llm(self._client, self._model, _SYSTEM, user)
        text = _strip_fences(text)
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object in response: {text[:300]}")
        raw  = json.loads(text[start:end])
        plan = _parse_fix_plan(instance.instance_id, raw)
        logger.info(
            "[%s] Revised FixPlan: files=%s, regions=%d",
            instance.instance_id,
            plan.affected_files,
            len(plan.affected_regions),
        )
        return plan
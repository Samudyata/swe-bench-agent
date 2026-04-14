"""Planner agent — converts a GitHub issue into a structured search plan.

Migrated to the current Google GenAI SDK (google-genai package).
Install with: pip install google-genai

The Planner is a pure LLM reasoning agent: it receives the raw issue text
and produces a PlannerOutput containing keywords, search hints, and suspected
modules.  It does NOT call any tools or read the repository.

LLM backend: Google Gemini (gemini-2.0-flash by default).
JSON output mode is enabled via response_mime_type so parsing is reliable.

Usage
-----
    from agents.planner import PlannerAgent
    from pipeline.schema import SWEInstance

    planner = PlannerAgent()
    plan = planner.plan(instance)

    # On low-confidence retry from Validator:
    plan = planner.replan(instance, feedback_message, previous_plan)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from google import genai
from google.genai import types as genai_types

from pipeline.schema import FeedbackMessage, PlannerOutput, SWEInstance

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_INSTRUCTION = """\
You are an expert software bug analyst specialising in C codebases.
Given a GitHub issue report for a C repository, you produce a structured
search plan that guides a downstream retrieval agent to find the relevant
source files and functions.

Your output MUST be a single JSON object — no markdown, no prose, no
code fences.  Return only the JSON.
"""

_PLAN_TEMPLATE = """\
Repository: {repo}

=== GitHub Issue ===
{problem_statement}
{hints_section}
===================

Analyse the issue above and return a JSON object with EXACTLY these fields:

{{
  "issue_type": "<one of: bug | feature | refactor | performance>",
  "keywords": [
    "<up to 8 C-identifier-style search terms likely to match function/variable names>"
  ],
  "search_hints": [
    "<up to 4 natural-language hints for the retrieval agent, e.g. 'look in compression functions'>"
  ],
  "suspected_modules": [
    "<file paths relative to the repo root, ONLY if explicitly named in the issue; else []>"
  ],
  "priority_functions": [
    "<specific function names explicitly mentioned or strongly implied by the issue>"
  ],
  "reasoning": "<one sentence explaining your keyword choices>"
}}

Rules:
- keywords must be C identifier-style (e.g. "jv_parse", "compress_block", "ZSTD_compress").
  Avoid generic words like "error", "null", "function".
- suspected_modules: leave as [] unless the issue text mentions a specific file by name.
- priority_functions: leave as [] if no function names appear in the issue.
- Return ONLY valid JSON — no prose, no markdown fences.
"""

_HINTS_SECTION_TEMPLATE = """\

=== Additional Issue Comments ===
{hints_text}
================================
"""

_REPLAN_TEMPLATE = """\
Repository: {repo}

=== GitHub Issue ===
{problem_statement}
{hints_section}
===================

The previous search plan produced low-confidence localisation.
Validator evidence: {evidence}

Your previous plan keywords were: {previous_keywords}
Your previous suspected modules were: {previous_modules}

Generate a REVISED search plan.  Try DIFFERENT keywords and modules.
Think about synonyms, related subsystems, or alternative entry points.

Return the same JSON schema as before:

{{
  "issue_type": "<bug | feature | refactor | performance>",
  "keywords": ["<up to 8 DIFFERENT C-identifier-style terms>"],
  "search_hints": ["<up to 4 hints that differ from the previous attempt>"],
  "suspected_modules": ["<alternative file paths, or [] if unsure>"],
  "priority_functions": ["<alternative function names>"],
  "reasoning": "<one sentence explaining what you changed and why>"
}}

Return ONLY valid JSON.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Planner agent
# ─────────────────────────────────────────────────────────────────────────────

class PlannerAgent:
    """LLM-backed Planner agent.

    Args:
        model_name:    Gemini model identifier.
        api_key:       Gemini API key.  If None, reads from GOOGLE_API_KEY env.
        max_llm_retries: Number of times to retry the LLM call on parse failure.
        temperature:   Sampling temperature (0.0 = deterministic).
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: str | None = None,
        max_llm_retries: int = 2,
        temperature: float = 0.2,
    ) -> None:
        self._model_name = model_name
        self._max_llm_retries = max_llm_retries
        self._temperature = temperature
        # Initialise the GenAI client; api_key=None falls back to GOOGLE_API_KEY env var
        self._client = genai.Client(api_key=api_key)

    # ── Public API ────────────────────────────────────────────────────────────

    def plan(self, instance: SWEInstance) -> PlannerOutput:
        """Produce an initial PlannerOutput from a SWE-bench instance.

        Args:
            instance: The issue to analyse.

        Returns:
            A PlannerOutput with keywords, search hints, and issue type.
        """
        prompt = self._build_plan_prompt(instance)
        raw = self._call_with_retry(prompt)
        return self._parse_output(instance.instance_id, raw)

    def replan(
        self,
        instance: SWEInstance,
        feedback: FeedbackMessage,
        previous_plan: PlannerOutput,
    ) -> PlannerOutput:
        """Produce a revised PlannerOutput after a low-confidence failure.

        Called by the pipeline controller when the Validator routes a
        LOW_CONF FeedbackMessage back to the Planner.

        Args:
            instance:      The original issue.
            feedback:      The FeedbackMessage from the Validator.
            previous_plan: The plan that led to low confidence.

        Returns:
            A revised PlannerOutput with different keywords / modules.
        """
        prompt = self._build_replan_prompt(instance, feedback, previous_plan)
        raw = self._call_with_retry(prompt)
        return self._parse_output(instance.instance_id, raw)

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_plan_prompt(self, instance: SWEInstance) -> str:
        hints_section = ""
        if instance.hints_text and instance.hints_text.strip():
            hints_section = _HINTS_SECTION_TEMPLATE.format(
                hints_text=instance.hints_text.strip()
            )
        return _PLAN_TEMPLATE.format(
            repo=instance.repo,
            problem_statement=instance.problem_statement.strip(),
            hints_section=hints_section,
        )

    def _build_replan_prompt(
        self,
        instance: SWEInstance,
        feedback: FeedbackMessage,
        previous_plan: PlannerOutput,
    ) -> str:
        hints_section = ""
        if instance.hints_text and instance.hints_text.strip():
            hints_section = _HINTS_SECTION_TEMPLATE.format(
                hints_text=instance.hints_text.strip()
            )
        return _REPLAN_TEMPLATE.format(
            repo=instance.repo,
            problem_statement=instance.problem_statement.strip(),
            hints_section=hints_section,
            evidence=feedback.evidence[:500],  # truncate long compiler output
            previous_keywords=previous_plan.keywords,
            previous_modules=previous_plan.suspected_modules,
        )

    # ── LLM call + retry ──────────────────────────────────────────────────────

    def _call_with_retry(self, prompt: str) -> dict[str, Any]:
        """Call the LLM and parse JSON, retrying on failure."""
        last_exc: Exception | None = None

        full_prompt = f"{_SYSTEM_INSTRUCTION}\n\n{prompt}"

        for attempt in range(self._max_llm_retries + 1):
            if attempt > 0:
                backoff = 2 ** attempt
                logger.warning("Planner LLM retry %d/%d after %ds",
                               attempt, self._max_llm_retries, backoff)
                time.sleep(backoff)
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=full_prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=self._temperature,
                        response_mime_type="application/json",
                    ),
                )
                text = response.text.strip()
                # Strip markdown fences if model adds them despite instruction
                if text.startswith("```"):
                    text = "\n".join(
                        line for line in text.splitlines()
                        if not line.startswith("```")
                    ).strip()
                return json.loads(text)
            except json.JSONDecodeError as exc:
                logger.warning("Planner JSON parse failed (attempt %d): %s", attempt, exc)
                last_exc = exc
            except Exception as exc:
                logger.error("Planner LLM call failed (attempt %d): %s", attempt, exc)
                last_exc = exc

        raise RuntimeError(
            f"Planner failed after {self._max_llm_retries + 1} attempts"
        ) from last_exc

    # ── Output parsing ────────────────────────────────────────────────────────

    def _parse_output(self, instance_id: str, raw: dict[str, Any]) -> PlannerOutput:
        """Convert the raw LLM JSON dict into a validated PlannerOutput."""
        # Normalise issue_type
        issue_type = raw.get("issue_type", "bug").lower().strip()
        if issue_type not in {"bug", "feature", "refactor", "performance"}:
            logger.warning("Planner returned unknown issue_type=%r, defaulting to 'bug'", issue_type)
            issue_type = "bug"

        # Ensure lists are actually lists of strings
        def _str_list(key: str, max_items: int) -> list[str]:
            val = raw.get(key, [])
            if not isinstance(val, list):
                return []
            return [str(s).strip() for s in val if str(s).strip()][:max_items]

        return PlannerOutput(
            instance_id=instance_id,
            issue_type=issue_type,
            keywords=_str_list("keywords", 8),
            search_hints=_str_list("search_hints", 4),
            suspected_modules=_str_list("suspected_modules", 10),
            priority_functions=_str_list("priority_functions", 8),
            reasoning=str(raw.get("reasoning", "")).strip(),
        )

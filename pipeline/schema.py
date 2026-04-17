"""Inter-agent message schema for swe-bench-agent.

This module defines ALL typed messages that flow between agents in the
evidence-routed multi-agent pipeline. Every agent consumes and/or produces
one of these dataclasses.

Message flow:
    SWEInstance
        -> [Planner]       -> PlannerOutput
        -> [Localizer]     -> ContextBundle
        -> [Diagnostician] -> FixPlan
        -> [Patcher]       -> PatchOutput
        -> [Validator]     -> ValidationResult
                  |
      (on failure) -> FeedbackMessage -> routed back to specific agent
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_dict(obj) -> dict:
    """Recursively convert a dataclass (or nested structure) to a plain dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Failure taxonomy
# ─────────────────────────────────────────────────────────────────────────────

class FailureType(str, Enum):
    """Classification of pipeline failure types.

    Used by Validator to route evidence back to the agent that can fix it.
    """
    APPLY      = "apply_failed"     # git apply rejected the patch
    COMPILE    = "compile_failed"   # gcc / make compilation error
    TEST       = "test_failed"      # FAIL_TO_PASS tests did not pass
    REGRESSION = "regression"       # PASS_TO_PASS tests broke
    LOW_CONF   = "low_confidence"   # Localizer confidence below threshold
    SUCCESS    = "success"          # all checks passed


# ─────────────────────────────────────────────────────────────────────────────
# Dataset input
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SWEInstance:
    """One SWE-bench-C (or SWE-bench) task instance loaded from the dataset.

    Fields come directly from the dataset JSON; see:
        https://huggingface.co/datasets/SWE-bench/SWE-bench_c
    """
    instance_id: str           # e.g. "jq-493__jqlang__jq"
    repo: str                  # e.g. "jqlang/jq"
    base_commit: str           # git hash before the fix was applied
    problem_statement: str     # the raw GitHub issue text
    hints_text: str = ""       # issue comments / discussion thread (may be empty)
    fail_to_pass: list[str] = field(default_factory=list)  # tests that must start passing
    pass_to_pass: list[str] = field(default_factory=list)  # tests that must keep passing

    def to_dict(self) -> dict:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SWEInstance:
        return cls(
            instance_id=d["instance_id"],
            repo=d["repo"],
            base_commit=d["base_commit"],
            problem_statement=d["problem_statement"],
            hints_text=d.get("hints_text", ""),
            fail_to_pass=d.get("FAIL_TO_PASS", d.get("fail_to_pass", [])),
            pass_to_pass=d.get("PASS_TO_PASS", d.get("pass_to_pass", [])),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Planner output  (Person 2 → Person 3)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlannerOutput:
    """Structured search plan produced by the Planner agent.

    The Planner performs pure LLM reasoning over the issue text and produces
    this structured object to guide the Localizer's graph traversal and grep
    strategy.
    """
    instance_id: str
    issue_type: str                       # "bug" | "feature" | "refactor" | "performance"
    keywords: list[str]                   # C identifier-style search terms, priority order
    search_hints: list[str]               # natural language hints for Localizer strategy
    suspected_modules: list[str]          # relative file paths guessed from issue text
    priority_functions: list[str]         # specific function names from the issue
    reasoning: str = ""                   # short CoT explanation (logged, not forwarded)

    def to_dict(self) -> dict:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PlannerOutput:
        return cls(
            instance_id=d["instance_id"],
            issue_type=d.get("issue_type", "bug"),
            keywords=d.get("keywords", []),
            search_hints=d.get("search_hints", []),
            suspected_modules=d.get("suspected_modules", []),
            priority_functions=d.get("priority_functions", []),
            reasoning=d.get("reasoning", ""),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Localizer output  (Person 3 → Person 4)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LocalizerCandidate:
    """A single file candidate produced by the Localizer, with ranking info."""
    file_path: str           # repo-relative path, e.g. "src/jv.c"
    score: float             # composite relevance score (higher = more relevant)
    reason: str              # human-readable explanation of why this file was selected
    functions: list[str]     # relevant function node IDs within this file

    def to_dict(self) -> dict:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> LocalizerCandidate:
        return cls(
            file_path=d["file_path"],
            score=d["score"],
            reason=d.get("reason", ""),
            functions=d.get("functions", []),
        )


@dataclass
class ContextBundle:
    """Full context package passed from Localizer to Diagnostician.

    Contains ranked candidate files, their full source text, and relevant
    function snippets extracted by the Localizer.
    """
    instance_id: str
    candidates: list[LocalizerCandidate]   # ranked highest→lowest
    confidence: float                       # overall localization confidence [0.0, 1.0]
    file_contents: dict[str, str]           # {file_path: full source text}
    relevant_snippets: dict[str, str]       # {func_node_id: code snippet}
    test_files: list[str]                   # test file paths from graph.get_test_files()

    def to_dict(self) -> dict:
        d = _to_dict(self)
        d["candidates"] = [c.to_dict() for c in self.candidates]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ContextBundle:
        return cls(
            instance_id=d["instance_id"],
            candidates=[LocalizerCandidate.from_dict(c) for c in d.get("candidates", [])],
            confidence=d.get("confidence", 0.0),
            file_contents=d.get("file_contents", {}),
            relevant_snippets=d.get("relevant_snippets", {}),
            test_files=d.get("test_files", []),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostician output  (Person 4 internal)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AffectedRegion:
    """A specific code region identified by the Diagnostician."""
    file_path: str
    start_line: int
    end_line: int
    description: str     # what role this region plays in the bug

    def to_dict(self) -> dict:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AffectedRegion:
        return cls(
            file_path=d["file_path"],
            start_line=d["start_line"],
            end_line=d["end_line"],
            description=d.get("description", ""),
        )


@dataclass
class FixPlan:
    """Structured fix plan from the Diagnostician, consumed by the Patcher.

    The Diagnostician reads the ContextBundle, identifies the root cause,
    and produces this plan specifying exactly what needs to change and why.
    """
    instance_id: str
    root_cause: str                         # natural language: what is broken and why
    affected_files: list[str]               # files that will be modified
    affected_regions: list[AffectedRegion]  # specific line ranges to change
    test_constraints: list[str]             # what the patch must satisfy to pass tests
    fix_description: str                    # how to implement the fix

    def to_dict(self) -> dict:
        d = _to_dict(self)
        d["affected_regions"] = [r.to_dict() for r in self.affected_regions]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FixPlan:
        return cls(
            instance_id=d["instance_id"],
            root_cause=d.get("root_cause", ""),
            affected_files=d.get("affected_files", []),
            affected_regions=[AffectedRegion.from_dict(r) for r in d.get("affected_regions", [])],
            test_constraints=d.get("test_constraints", []),
            fix_description=d.get("fix_description", ""),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Patcher output  (Person 4 → Person 5)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatchOutput:
    """Unified diff patch generated by the Patcher agent.

    The unified_diff must be a valid patch that can be applied with `git apply`.
    Context lines must come from actual source files (not hallucinated).
    """
    instance_id: str
    unified_diff: str        # full .patch content (unified diff format)
    affected_files: list[str]

    def to_dict(self) -> dict:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PatchOutput:
        return cls(
            instance_id=d["instance_id"],
            unified_diff=d.get("unified_diff", ""),
            affected_files=d.get("affected_files", []),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Validator output  (Person 5)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of running the Validator on a generated patch.

    Tracks each stage of validation: apply → compile → test execution.
    The error_output contains raw stdout/stderr for the failing stage.
    """
    instance_id: str
    status: FailureType
    resolved: bool                # True only when all FAIL_TO_PASS tests pass
    apply_ok: bool                # git apply succeeded
    compile_ok: bool              # compilation succeeded
    tests_passed: list[str]       # test IDs that passed
    tests_failed: list[str]       # test IDs that failed
    error_output: str = ""        # raw stdout/stderr from the failing stage

    def to_dict(self) -> dict:
        d = _to_dict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ValidationResult:
        return cls(
            instance_id=d["instance_id"],
            status=FailureType(d["status"]),
            resolved=d.get("resolved", False),
            apply_ok=d.get("apply_ok", False),
            compile_ok=d.get("compile_ok", False),
            tests_passed=d.get("tests_passed", []),
            tests_failed=d.get("tests_failed", []),
            error_output=d.get("error_output", ""),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feedback / retry messages  (Person 5 → Persons 2/3/4)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeedbackMessage:
    """Structured failure evidence routed from Validator back to an upstream agent.

    The controller creates one of these when a ValidationResult is not SUCCESS,
    then passes it to the appropriate agent's retry method.
    """
    instance_id: str
    failure_type: FailureType
    route_to: str           # "planner" | "localizer" | "diagnostician" | "patcher"
    evidence: str           # raw error output for the agent to reason about
    retry_number: int       # 1 or 2 (tracks how many times we've retried)
    context: dict[str, Any] = field(default_factory=dict)  # extra structured info

    def to_dict(self) -> dict:
        d = _to_dict(self)
        d["failure_type"] = self.failure_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FeedbackMessage:
        return cls(
            instance_id=d["instance_id"],
            failure_type=FailureType(d["failure_type"]),
            route_to=d["route_to"],
            evidence=d.get("evidence", ""),
            retry_number=d.get("retry_number", 1),
            context=d.get("context", {}),
        )

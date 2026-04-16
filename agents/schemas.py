"""
schemas.py — Shared data schemas for inter-agent communication.

Defines the structured types that flow between agents:
  Person 3 (Localizer)  →  ContextBundle
  Diagnostician         →  FixPlan
  Patcher               →  PatchResult  →  Person 5 (Validator)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Input: what Person 3 (Localizer) hands us ────────────────────────────────

@dataclass
class FileContext:
    """A single source file and its content, as provided by the Localizer."""
    path: str           # Repo-relative path, e.g. "src/jv.c"
    content: str        # Full file text
    relevance_score: float = 1.0   # Localizer's confidence this file matters
    is_test: bool = False


@dataclass
class ContextBundle:
    """
    Everything the Localizer passes to the Diagnostician.

    Mirrors the output schema Person 3 agreed to produce.
    Fields here must stay in sync with the Localizer's output.
    """
    issue_id: str                           # SWE-bench instance_id
    issue_title: str
    issue_body: str
    repo: str                               # e.g. "jqlang/jq"
    base_commit: str
    repo_root: str                          # Absolute path on disk
    source_files: list[FileContext]         # Ranked: most relevant first
    test_files: list[FileContext]           # Test files that exercise the area
    graph_context: dict = field(default_factory=dict)  # Optional graph metadata
    localization_confidence: float = 1.0
    planner_keywords: list[str] = field(default_factory=list)
    planner_issue_type: str = ""            # "bug_fix" | "feature" | "refactor"

    @classmethod
    def from_dict(cls, d: dict) -> ContextBundle:
        source_files = [FileContext(**f) for f in d.get("source_files", [])]
        test_files = [FileContext(**f) for f in d.get("test_files", [])]
        return cls(
            issue_id=d["issue_id"],
            issue_title=d.get("issue_title", ""),
            issue_body=d.get("issue_body", ""),
            repo=d.get("repo", ""),
            base_commit=d.get("base_commit", ""),
            repo_root=d.get("repo_root", ""),
            source_files=source_files,
            test_files=test_files,
            graph_context=d.get("graph_context", {}),
            localization_confidence=d.get("localization_confidence", 1.0),
            planner_keywords=d.get("planner_keywords", []),
            planner_issue_type=d.get("planner_issue_type", ""),
        )

    @classmethod
    def from_json(cls, text: str) -> ContextBundle:
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ── Middle: Diagnostician output ─────────────────────────────────────────────

@dataclass
class AffectedRegion:
    """A specific range of lines in a file that needs to be changed."""
    file_path: str
    start_line: int
    end_line: int
    reason: str        # Plain-English explanation of why this region is involved


@dataclass
class FixPlan:
    """
    Structured output from the Diagnostician agent.

    The Patcher consumes this alongside real file contents
    to generate a unified diff.
    """
    issue_id: str
    root_cause: str                          # Plain-English root cause
    fix_description: str                     # What the patch should do
    affected_files: list[str]                # File paths that need edits
    affected_regions: list[AffectedRegion]   # Specific line ranges
    test_constraints: list[str]              # Things the fix must not break
    suggested_approach: str                  # High-level strategy for the Patcher
    confidence: float = 1.0                  # Diagnostician's self-confidence
    requires_new_function: bool = False
    requires_header_change: bool = False
    diagnostician_notes: str = ""            # Extra notes for the Patcher

    # Retry context — populated when Validator routes a failure back
    retry_reason: Optional[str] = None       # "compilation_error" | "test_failure"
    retry_error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> FixPlan:
        regions = [AffectedRegion(**r) for r in d.get("affected_regions", [])]
        return cls(
            issue_id=d["issue_id"],
            root_cause=d.get("root_cause", ""),
            fix_description=d.get("fix_description", ""),
            affected_files=d.get("affected_files", []),
            affected_regions=regions,
            test_constraints=d.get("test_constraints", []),
            suggested_approach=d.get("suggested_approach", ""),
            confidence=d.get("confidence", 1.0),
            requires_new_function=d.get("requires_new_function", False),
            requires_header_change=d.get("requires_header_change", False),
            diagnostician_notes=d.get("diagnostician_notes", ""),
            retry_reason=d.get("retry_reason"),
            retry_error_message=d.get("retry_error_message"),
        )

    @classmethod
    def from_json(cls, text: str) -> FixPlan:
        return cls.from_dict(json.loads(text))


# ── Output: what the Patcher hands to Person 5 (Validator) ───────────────────

@dataclass
class PatchResult:
    """
    Final output of the Patcher agent, passed to Person 5's Validator.
    """
    issue_id: str
    patch: str                   # Full unified diff text, ready for `git apply`
    modified_files: list[str]    # Files touched by the patch
    fix_plan: FixPlan            # The plan this patch was generated from
    patcher_notes: str = ""
    attempt_number: int = 1      # Which retry attempt produced this patch

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> PatchResult:
        fix_plan = FixPlan.from_dict(d["fix_plan"])
        return cls(
            issue_id=d["issue_id"],
            patch=d["patch"],
            modified_files=d.get("modified_files", []),
            fix_plan=fix_plan,
            patcher_notes=d.get("patcher_notes", ""),
            attempt_number=d.get("attempt_number", 1),
        )
"""Stub implementations of agent interfaces for Persons 3, 4, and 5.

These stubs let the pipeline controller and logger be fully tested before
teammates have implemented their agents.  Each stub either:

  - Raises NotImplementedError (default) — to catch integration issues early
  - Returns a minimal valid object   — for end-to-end smoke-testing

Switch a stub to minimal mode by instantiating with stub_mode=True:

    localizer  = LocalizerStub(stub_mode=True)
    diagn      = DiagnosticianStub(stub_mode=True)
    patcher    = PatcherStub(stub_mode=True)
    validator  = ValidatorStub(stub_mode=True)

Integration contract
--------------------
Each teammate MUST implement a class that matches the method signatures
below.  Drop the real class where the stub is used in the controller.

    Person 3 → LocalizerAgent   (implements LocalizerStub interface)
    Person 4 → DiagnosticianAgent, PatcherAgent
    Person 5 → ValidatorAgent
"""

from __future__ import annotations

from graph.model import DepGraph
from pipeline.schema import (
    ContextBundle,
    FailureType,
    FeedbackMessage,
    FixPlan,
    LocalizerCandidate,
    PatchOutput,
    PlannerOutput,
    SWEInstance,
    ValidationResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Person 3 stub — Localizer
# ─────────────────────────────────────────────────────────────────────────────

class LocalizerStub:
    """Stub for Person 3's LocalizerAgent.

    Real interface to implement:
        def localize(
            self,
            instance: SWEInstance,
            plan: PlannerOutput,
            graph: DepGraph,
        ) -> ContextBundle: ...
    """

    def __init__(self, stub_mode: bool = False) -> None:
        self._stub_mode = stub_mode

    def localize(
        self,
        instance: SWEInstance,
        plan: PlannerOutput,
        graph: DepGraph,
    ) -> ContextBundle:
        """Find relevant files and functions using the graph and plan keywords."""
        if not self._stub_mode:
            raise NotImplementedError(
                "Person 3 must implement LocalizerAgent.localize()"
            )
        # Minimal stub: return first candidate file from the graph
        file_nodes = [
            nid for nid, n in graph.nodes.items()
            if n["kind"] == "file"
        ]
        candidate_path = file_nodes[0] if file_nodes else "src/main.c"
        return ContextBundle(
            instance_id=instance.instance_id,
            candidates=[
                LocalizerCandidate(
                    file_path=candidate_path,
                    score=0.5,
                    reason="STUB: first file in graph",
                    functions=[],
                )
            ],
            confidence=0.5,
            file_contents={},
            relevant_snippets={},
            test_files=[],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Person 4 stubs — Diagnostician + Patcher
# ─────────────────────────────────────────────────────────────────────────────

class DiagnosticianStub:
    """Stub for Person 4's DiagnosticianAgent.

    Real interface to implement:
        def diagnose(
            self,
            instance: SWEInstance,
            bundle: ContextBundle,
        ) -> FixPlan: ...

        def revise(
            self,
            instance: SWEInstance,
            bundle: ContextBundle,
            feedback: FeedbackMessage,
        ) -> FixPlan: ...
    """

    def __init__(self, stub_mode: bool = False) -> None:
        self._stub_mode = stub_mode

    def diagnose(
        self,
        instance: SWEInstance,
        bundle: ContextBundle,
    ) -> FixPlan:
        """Read localised context and produce a structured fix plan."""
        if not self._stub_mode:
            raise NotImplementedError(
                "Person 4 must implement DiagnosticianAgent.diagnose()"
            )
        return FixPlan(
            instance_id=instance.instance_id,
            root_cause="STUB: root cause not diagnosed",
            affected_files=bundle.candidates[0:1] and [bundle.candidates[0].file_path] or [],
            affected_regions=[],
            test_constraints=[],
            fix_description="STUB: no fix planned",
        )

    def revise(
        self,
        instance: SWEInstance,
        bundle: ContextBundle,
        feedback: FeedbackMessage,
    ) -> FixPlan:
        """Revise an existing fix plan given failure evidence from the Validator."""
        if not self._stub_mode:
            raise NotImplementedError(
                "Person 4 must implement DiagnosticianAgent.revise()"
            )
        return self.diagnose(instance, bundle)


class PatcherStub:
    """Stub for Person 4's PatcherAgent.

    Real interface to implement:
        def patch(
            self,
            instance: SWEInstance,
            fix_plan: FixPlan,
            bundle: ContextBundle,
        ) -> PatchOutput: ...

        def patch_with_feedback(
            self,
            instance: SWEInstance,
            fix_plan: FixPlan,
            bundle: ContextBundle,
            feedback: FeedbackMessage,
        ) -> PatchOutput: ...
    """

    def __init__(self, stub_mode: bool = False) -> None:
        self._stub_mode = stub_mode

    def patch(
        self,
        instance: SWEInstance,
        fix_plan: FixPlan,
        bundle: ContextBundle,
    ) -> PatchOutput:
        """Generate a unified diff from the fix plan and localised source code."""
        if not self._stub_mode:
            raise NotImplementedError(
                "Person 4 must implement PatcherAgent.patch()"
            )
        return PatchOutput(
            instance_id=instance.instance_id,
            unified_diff="--- a/stub\n+++ b/stub\n",   # empty patch (will fail apply)
            affected_files=fix_plan.affected_files,
        )

    def patch_with_feedback(
        self,
        instance: SWEInstance,
        fix_plan: FixPlan,
        bundle: ContextBundle,
        feedback: FeedbackMessage,
    ) -> PatchOutput:
        """Retry patch generation with compiler error evidence appended."""
        if not self._stub_mode:
            raise NotImplementedError(
                "Person 4 must implement PatcherAgent.patch_with_feedback()"
            )
        return self.patch(instance, fix_plan, bundle)


# ─────────────────────────────────────────────────────────────────────────────
# Person 5 stub — Validator
# ─────────────────────────────────────────────────────────────────────────────

class ValidatorStub:
    """Stub for Person 5's ValidatorAgent.

    Real interface to implement:
        def validate(
            self,
            instance: SWEInstance,
            patch: PatchOutput,
        ) -> ValidationResult: ...
    """

    def __init__(self, stub_mode: bool = False, force_result: FailureType = FailureType.SUCCESS) -> None:
        """
        Args:
            stub_mode:    If True, return a canned result instead of raising.
            force_result: The FailureType to return in stub mode.
        """
        self._stub_mode = stub_mode
        self._force_result = force_result

    def validate(
        self,
        instance: SWEInstance,
        patch: PatchOutput,
    ) -> ValidationResult:
        """Apply the patch, compile, and run FAIL_TO_PASS tests."""
        if not self._stub_mode:
            raise NotImplementedError(
                "Person 5 must implement ValidatorAgent.validate()"
            )
        resolved = (self._force_result == FailureType.SUCCESS)
        return ValidationResult(
            instance_id=instance.instance_id,
            status=self._force_result,
            resolved=resolved,
            apply_ok=resolved,
            compile_ok=resolved,
            tests_passed=instance.fail_to_pass if resolved else [],
            tests_failed=[] if resolved else instance.fail_to_pass,
            error_output="STUB validator" if not resolved else "",
        )

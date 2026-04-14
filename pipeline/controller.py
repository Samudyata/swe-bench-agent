"""Pipeline controller — orchestrates the evidence-routed multi-agent pipeline.

One PipelineController instance handles a single SWE-bench issue end-to-end:

    Planner → Localizer → Diagnostician → Patcher → Validator
                 ↑_____________________________|  (feedback routing on failure)

Failure routing
---------------
  apply_failed   → Localizer  (expand graph neighbourhood, re-search)
  compile_failed → Patcher    (append gcc error, retry patch only)
  test_failed    → Diagnostician (revise root cause with test output)
  regression     → Diagnostician (revise to not break PASS_TO_PASS)
  low_confidence → Planner    (replan with different keywords)

Max `max_retries` re-attempts are allowed per agent slot.

Usage
-----
    from pipeline.controller import PipelineController
    from config import Config

    config = Config.from_env()
    controller = PipelineController(config)
    result = controller.run(instance, graph)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from graph.model import DepGraph
from pipeline.logger import PipelineLogger
from pipeline.schema import (
    ContextBundle,
    FailureType,
    FeedbackMessage,
    FixPlan,
    PatchOutput,
    PlannerOutput,
    SWEInstance,
    ValidationResult,
)

logger = logging.getLogger(__name__)

# Routing table: failure type → which agent slot to retry
_ROUTE_TO: dict[FailureType, str] = {
    FailureType.APPLY:      "localizer",
    FailureType.COMPILE:    "patcher",
    FailureType.TEST:       "diagnostician",
    FailureType.REGRESSION: "diagnostician",
    FailureType.LOW_CONF:   "planner",
}


class PipelineController:
    """Orchestrates one SWE-bench instance through the full agent pipeline.

    Agent slots are injected at construction time, making it easy to swap
    in real agents as teammates complete their implementations.

    Args:
        config:        Pipeline configuration (model, thresholds, paths).
        planner:       PlannerAgent instance (Person 2).
        localizer:     LocalizerAgent instance (Person 3).
        diagnostician: DiagnosticianAgent instance (Person 4).
        patcher:       PatcherAgent instance (Person 4).
        validator:     ValidatorAgent instance (Person 5).
    """

    def __init__(
        self,
        config,                 # Config (avoid circular import at type-check time)
        planner=None,
        localizer=None,
        diagnostician=None,
        patcher=None,
        validator=None,
    ) -> None:
        from config import Config
        assert isinstance(config, Config), "config must be a Config instance"

        self.config = config
        self.planner = planner
        self.localizer = localizer
        self.diagnostician = diagnostician
        self.patcher = patcher
        self.validator = validator

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, instance: SWEInstance, graph: DepGraph) -> ValidationResult:
        """Run the full pipeline for one SWE-bench instance.

        Args:
            instance: The SWE-bench task (issue text, test IDs, etc.).
            graph:    Pre-built dependency graph for the instance's repo.

        Returns:
            The final ValidationResult (resolved or not).
        """
        log = PipelineLogger(instance.instance_id, self.config.log_dir)
        log.log("instance_start", "controller", instance.to_dict())

        retry_counts: dict[str, int] = defaultdict(int)

        # ── Step 1: initial plan ──────────────────────────────────────────────
        plan = self._do_plan(instance, log)

        # ── Main retry loop ───────────────────────────────────────────────────
        result: ValidationResult | None = None
        bundle: ContextBundle | None = None
        fix_plan: FixPlan | None = None
        patch: PatchOutput | None = None

        for _attempt in range(self.config.max_retries * 3 + 1):  # generous bound
            # ── Step 2: localise ─────────────────────────────────────────────
            bundle = self._do_localize(instance, plan, graph, log)

            # Low confidence → replan
            if bundle.confidence < self.config.confidence_threshold:
                if retry_counts["planner"] < self.config.max_retries:
                    fb = FeedbackMessage(
                        instance_id=instance.instance_id,
                        failure_type=FailureType.LOW_CONF,
                        route_to="planner",
                        evidence=(
                            f"Localizer confidence {bundle.confidence:.2f} "
                            f"is below threshold {self.config.confidence_threshold:.2f}. "
                            f"Top candidates: {[c.file_path for c in bundle.candidates[:3]]}"
                        ),
                        retry_number=retry_counts["planner"] + 1,
                    )
                    log.log("retry_routed", "controller", fb.to_dict())
                    plan = self._do_replan(instance, fb, plan, log)
                    retry_counts["planner"] += 1
                    continue   # restart from localisation with new plan
                else:
                    logger.warning("[%s] Low confidence but planner retries exhausted",
                                   instance.instance_id)

            # ── Step 3: diagnose ─────────────────────────────────────────────
            fix_plan = self._do_diagnose(instance, bundle, log)

            # ── Step 4: patch ────────────────────────────────────────────────
            patch = self._do_patch(instance, fix_plan, bundle, log)

            # ── Step 5: validate ─────────────────────────────────────────────
            result = self._do_validate(instance, patch, log)

            if result.status == FailureType.SUCCESS:
                break  # done!

            # ── Step 6: route failure evidence ───────────────────────────────
            fb = self._build_feedback(result)
            route_to = fb.route_to
            log.log("retry_routed", "controller", fb.to_dict())

            if retry_counts[route_to] >= self.config.max_retries:
                logger.warning("[%s] Retries exhausted for %s",
                               instance.instance_id, route_to)
                break

            retry_counts[route_to] += 1

            if route_to == "localizer":
                # Inject rejected hunk info into plan search hints and re-localise
                plan = self._inject_apply_feedback(plan, fb)
                continue   # restart from localisation

            elif route_to == "diagnostician":
                fix_plan = self._do_revise(instance, bundle, fb, log)
                patch = self._do_patch(instance, fix_plan, bundle, log)
                result = self._do_validate(instance, patch, log)
                if result.status == FailureType.SUCCESS:
                    break

            elif route_to == "patcher":
                patch = self._do_patch_with_feedback(instance, fix_plan, bundle, fb, log)
                result = self._do_validate(instance, patch, log)
                if result.status == FailureType.SUCCESS:
                    break

            else:
                logger.error("[%s] Unknown route_to=%r", instance.instance_id, route_to)
                break

        # ── Final result ──────────────────────────────────────────────────────
        if result is None:
            result = ValidationResult(
                instance_id=instance.instance_id,
                status=FailureType.APPLY,
                resolved=False,
                apply_ok=False,
                compile_ok=False,
                tests_passed=[],
                tests_failed=instance.fail_to_pass,
                error_output="Pipeline did not reach validation stage",
            )

        log.log("instance_end", "controller", result.to_dict())
        self._save_result(result)
        return result

    # ── Agent call wrappers (with error handling + logging) ───────────────────

    def _do_plan(self, instance: SWEInstance, log: PipelineLogger) -> PlannerOutput:
        plan = self.planner.plan(instance)
        log.log("planner_output", "planner", plan.to_dict())
        return plan

    def _do_replan(
        self,
        instance: SWEInstance,
        feedback: FeedbackMessage,
        previous_plan: PlannerOutput,
        log: PipelineLogger,
    ) -> PlannerOutput:
        plan = self.planner.replan(instance, feedback, previous_plan)
        log.log("planner_output", "planner", {**plan.to_dict(), "is_replan": True})
        return plan

    def _do_localize(
        self,
        instance: SWEInstance,
        plan: PlannerOutput,
        graph: DepGraph,
        log: PipelineLogger,
    ) -> ContextBundle:
        bundle = self.localizer.localize(instance, plan, graph)
        log.log("localizer_output", "localizer", bundle.to_dict())
        return bundle

    def _do_diagnose(
        self,
        instance: SWEInstance,
        bundle: ContextBundle,
        log: PipelineLogger,
    ) -> FixPlan:
        fix_plan = self.diagnostician.diagnose(instance, bundle)
        log.log("diagnostician_output", "diagnostician", fix_plan.to_dict())
        return fix_plan

    def _do_revise(
        self,
        instance: SWEInstance,
        bundle: ContextBundle,
        feedback: FeedbackMessage,
        log: PipelineLogger,
    ) -> FixPlan:
        fix_plan = self.diagnostician.revise(instance, bundle, feedback)
        log.log("diagnostician_output", "diagnostician", {**fix_plan.to_dict(), "is_revision": True})
        return fix_plan

    def _do_patch(
        self,
        instance: SWEInstance,
        fix_plan: FixPlan,
        bundle: ContextBundle,
        log: PipelineLogger,
    ) -> PatchOutput:
        patch = self.patcher.patch(instance, fix_plan, bundle)
        log.log("patch_output", "patcher", patch.to_dict())
        return patch

    def _do_patch_with_feedback(
        self,
        instance: SWEInstance,
        fix_plan: FixPlan,
        bundle: ContextBundle,
        feedback: FeedbackMessage,
        log: PipelineLogger,
    ) -> PatchOutput:
        patch = self.patcher.patch_with_feedback(instance, fix_plan, bundle, feedback)
        log.log("patch_output", "patcher", {**patch.to_dict(), "is_retry": True})
        return patch

    def _do_validate(
        self,
        instance: SWEInstance,
        patch: PatchOutput,
        log: PipelineLogger,
    ) -> ValidationResult:
        result = self.validator.validate(instance, patch)
        log.log("validation_result", "validator", result.to_dict())
        return result

    # ── Feedback construction ─────────────────────────────────────────────────

    def _build_feedback(self, result: ValidationResult) -> FeedbackMessage:
        """Map a ValidationResult failure to a structured FeedbackMessage."""
        route_to = _ROUTE_TO.get(result.status, "patcher")
        return FeedbackMessage(
            instance_id=result.instance_id,
            failure_type=result.status,
            route_to=route_to,
            evidence=result.error_output[:2000],   # truncate very long gcc output
            retry_number=1,
            context={
                "tests_failed": result.tests_failed,
                "apply_ok": result.apply_ok,
                "compile_ok": result.compile_ok,
            },
        )

    def _inject_apply_feedback(
        self,
        plan: PlannerOutput,
        feedback: FeedbackMessage,
    ) -> PlannerOutput:
        """Augment the plan's search hints with patch-rejection context.

        When git apply fails, the rejected hunk indicates which file context
        lines couldn't be matched.  We add a hint so the Localizer tries to
        expand into neighbouring files.
        """
        new_hints = list(plan.search_hints)
        new_hints.append(
            f"Previous patch apply failed — try expanding graph neighborhood: {feedback.evidence[:200]}"
        )
        return PlannerOutput(
            instance_id=plan.instance_id,
            issue_type=plan.issue_type,
            keywords=plan.keywords,
            search_hints=new_hints[:6],    # cap at 6 hints
            suspected_modules=plan.suspected_modules,
            priority_functions=plan.priority_functions,
            reasoning=plan.reasoning,
        )

    # ── Result persistence ────────────────────────────────────────────────────

    def _save_result(self, result: ValidationResult) -> None:
        """Save the final ValidationResult to results/<instance_id>.json."""
        results_dir = self.config.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / f"{result.instance_id}.json"
        out.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

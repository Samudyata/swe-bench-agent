"""
agents/ — Multi-agent system for SWE-bench-C

Public API:
    # Person 2: Planner and Localizer
    from agents.planner import PlannerAgent
    from agents.localizer import LocalizerAgent

    # Person 4 & 5: Diagnostician, Patcher, Validator and schemas
    from agents.schemas import ContextBundle, FixPlan, PatchResult
    from agents.diagnostician import diagnose, rediagnose
    from agents.patcher import patch, repatch
    from agents.validator import validate, revalidate_after_patch, revalidate_after_diagnosis
"""

from agents.planner import PlannerAgent
from agents.localizer import LocalizerAgent

from agents.schemas import (
    AffectedRegion,
    ContextBundle,
    FileContext,
    FixPlan,
    PatchResult,
)
from agents.diagnostician import diagnose, rediagnose
from agents.patcher import patch, repatch
from agents.validator import validate, revalidate_after_patch, revalidate_after_diagnosis

__all__ = [
    # Person 2
    "PlannerAgent",
    "LocalizerAgent",
    # Schemas
    "AffectedRegion",
    "ContextBundle",
    "FileContext",
    "FixPlan",
    "PatchResult",
    # Person 4
    "diagnose",
    "rediagnose",
    "patch",
    "repatch",
    # Person 5
    "validate",
    "revalidate_after_patch",
    "revalidate_after_diagnosis",
]

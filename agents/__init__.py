"""
agents/ — Diagnostician and Patcher agents (Person 4)

Public API:
    from agents.schemas import ContextBundle, FixPlan, PatchResult
    from agents.diagnostician import diagnose, rediagnose
    from agents.patcher import patch, repatch
"""

from agents.schemas import (
    AffectedRegion,
    ContextBundle,
    FileContext,
    FixPlan,
    PatchResult,
)
from agents.diagnostician import diagnose, rediagnose
from agents.patcher import patch, repatch

__all__ = [
    # Schemas
    "AffectedRegion",
    "ContextBundle",
    "FileContext",
    "FixPlan",
    "PatchResult",
    # Agents
    "diagnose",
    "rediagnose",
    "patch",
    "repatch",
]
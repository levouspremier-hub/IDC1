"""M2.1 六个不可变（frozen）Pydantic 契约。"""

from contracts.models import (
    DispatchProposal,
    DispatchResult,
    EvaluationRecord,
    ScenarioBundle,
    SystemSnapshot,
    TaskAllocation,
    TaskState,
)

__all__ = [
    "ScenarioBundle",
    "SystemSnapshot",
    "DispatchProposal",
    "TaskAllocation",
    "DispatchResult",
    "EvaluationRecord",
    "TaskState",
]

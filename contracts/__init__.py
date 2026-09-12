"""Versioned data contracts shared by all methods.

See docs/EXECUTION_PLAN.md §2. These six contracts are the single exchange
vocabulary between the scenario provider, environment, joint feasibility
planner, safe PPO, and unified evaluator.

Versioning rule (docs/AGENTS.md): bump CONTRACT_VERSION_ID on any breaking
change to a contract shape, the action space, or the checkpoint schema. Old
checkpoints/models must fail loudly instead of loading silently.
"""

from contracts.allocation import TaskAllocation
from contracts.base import ContractBase, canonical_json
from contracts.evaluation import EvaluationRecord
from contracts.proposal import DispatchProposal
from contracts.result import DispatchResult, SolveStatus
from contracts.scenario import ScenarioBundle
from contracts.snapshot import SystemSnapshot
from contracts.task import DeadlineStatus, TaskSpec, TaskState, TaskStatus

CONTRACT_VERSION = "0.1.0"
CONTRACT_VERSION_ID = 1

__all__ = [
    "CONTRACT_VERSION",
    "CONTRACT_VERSION_ID",
    "ContractBase",
    "DeadlineStatus",
    "DispatchProposal",
    "DispatchResult",
    "EvaluationRecord",
    "ScenarioBundle",
    "SolveStatus",
    "SystemSnapshot",
    "TaskAllocation",
    "TaskSpec",
    "TaskState",
    "TaskStatus",
    "canonical_json",
]

"""SystemSnapshot: time, tasks, SOC, budgets, corrector context."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from contracts.base import ContractBase
from contracts.task import TaskState


class SystemSnapshot(ContractBase):
    t: int = Field(ge=0)
    task_states: list[TaskState]
    soc: float = Field(ge=0.0, le=1.0, description="battery state of charge")
    budgets: dict[str, float] = Field(default_factory=dict)
    corrector_context: dict[str, Any] = Field(
        default_factory=dict,
        description="group capacities, access limits, and any info the corrector needs",
    )

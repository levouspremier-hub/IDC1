"""Task definitions and runtime states shared across contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from contracts.base import ContractBase

TaskStatus = Literal["not_arrived", "waiting", "running", "paused", "finished", "failed"]

# 逾期三分类（计划书 §2）：未到期积压 / 逾期积压 / 逾期完成；其余为 none。
DeadlineStatus = Literal["none", "on_time_backlog", "overdue_backlog", "overdue_completed"]


class TaskSpec(ContractBase):
    """A task's static definition (arrival, workload, deadline, max rate, criticality)."""

    task_id: int
    workload: float = Field(gt=0, description="total work units")
    arrival_time: int = Field(ge=0, description="hour index of arrival")
    latest_finish_time: int = Field(ge=0, description="deadline hour index (inclusive)")
    max_rate: float = Field(gt=0, description="max execution rate, work per hour")
    priority: float = 1.0
    interruptible: bool = True
    is_critical: bool = False

    @model_validator(mode="after")
    def _deadline_not_before_arrival(self) -> TaskSpec:
        if self.latest_finish_time < self.arrival_time:
            raise ValueError("latest_finish_time must be >= arrival_time")
        return self


class TaskState(ContractBase):
    """A task's runtime state at a decision point (for SystemSnapshot)."""

    task_id: int
    status: TaskStatus = "waiting"
    remaining_work: float = Field(ge=0)
    completed_work: float = Field(default=0.0, ge=0)
    deadline_status: DeadlineStatus = "none"
    start_time: int | None = None
    is_paused: bool = False

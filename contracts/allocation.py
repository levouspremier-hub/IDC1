"""TaskAllocation: actual per-task, per-server-group execution amount."""

from __future__ import annotations

from pydantic import Field, field_validator

from contracts.base import ContractBase


class TaskAllocation(ContractBase):
    # Shape (n_tasks, n_groups): matrix[i][g] == work of task i executed on group g this hour.
    matrix: list[list[float]]
    unit: str = Field(description="unit of executed work per hour")

    @field_validator("unit")
    @classmethod
    def _unit_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("unit must be a non-empty string")
        return v

    @field_validator("matrix")
    @classmethod
    def _non_negative(cls, v: list[list[float]]) -> list[list[float]]:
        if any(x < 0.0 for row in v for x in row):
            raise ValueError("execution amounts must be non-negative")
        return v

    @property
    def n_tasks(self) -> int:
        return len(self.matrix)

    @property
    def n_groups(self) -> int:
        return len(self.matrix[0]) if self.matrix else 0

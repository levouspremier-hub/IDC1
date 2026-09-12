"""DispatchProposal: the raw compute + storage action before any correction."""

from __future__ import annotations

from pydantic import Field, field_validator

from contracts.base import ContractBase


class DispatchProposal(ContractBase):
    # Per-group compute suggestions in [0, 1]; length == n_groups (default 20).
    compute: list[float]
    # Storage suggestion in [-1, 1]; negative == charge, positive == discharge.
    storage: float = Field(ge=-1.0, le=1.0)

    @field_validator("compute")
    @classmethod
    def _compute_in_unit_interval(cls, v: list[float]) -> list[float]:
        if any(x < 0.0 or x > 1.0 for x in v):
            raise ValueError("compute suggestions must lie within [0, 1]")
        return v

    @property
    def n_groups(self) -> int:
        return len(self.compute)

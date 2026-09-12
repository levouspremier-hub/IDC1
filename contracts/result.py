"""DispatchResult: executed action, energy flows, correction reason, solve status."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from contracts.base import ContractBase

SolveStatus = Literal[
    "optimal",
    "feasible",
    "timeout",
    "infeasible",
    "physically_rejected",
    "forecast_out_of_bounds",
]


class DispatchResult(ContractBase):
    raw_compute: list[float]
    raw_storage: float
    exec_compute: list[float]
    exec_storage: float
    energy_flows: dict[str, float] = Field(
        default_factory=dict,
        description="grid_purchase, pv_used, pv_curtail, wind_used, bess_charge, bess_discharge, losses",
    )
    correction_reason: str = "none"
    solve_status: SolveStatus = "feasible"
    business_gap: float = Field(default=0.0, ge=0.0, description="explicit unmet business work")

"""EvaluationRecord: business, economic, carbon, renewable, peak, reliability, time."""

from __future__ import annotations

from pydantic import Field

from contracts.base import ContractBase


class EvaluationRecord(ContractBase):
    scenario_id: str
    business: dict[str, float] = Field(default_factory=dict)
    economic: dict[str, float] = Field(default_factory=dict)
    carbon: dict[str, float] = Field(default_factory=dict)
    renewable: dict[str, float] = Field(default_factory=dict)
    peak: dict[str, float] = Field(default_factory=dict)
    reliability: dict[str, float] = Field(default_factory=dict)
    wall_time_seconds: float = Field(default=0.0, ge=0.0)

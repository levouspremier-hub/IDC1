"""ScenarioBundle: data, forecasts, tasks, capacity, units, sources, hash."""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from contracts.base import ContractBase
from contracts.task import TaskSpec

REQUIRED_UNITS = ("price", "power", "energy", "workload")


class ScenarioBundle(ContractBase):
    scenario_id: str
    units: dict[str, str]
    sources: dict[str, str] = Field(default_factory=dict)
    price: list[float]
    carbon_factor: list[float] | None = None
    temperature: list[float] | None = None
    pv: list[float] | None = None
    wind: list[float] | None = None
    forecast: dict[str, Any] = Field(default_factory=dict)
    tasks: list[TaskSpec]
    capacity: dict[str, Any]
    hash: str = ""

    @field_validator("units")
    @classmethod
    def _required_units(cls, v: dict[str, str]) -> dict[str, str]:
        missing = [k for k in REQUIRED_UNITS if not str(v.get(k, "")).strip()]
        if missing:
            raise ValueError(f"missing required units: {missing}")
        return v

    def freeze(self) -> ScenarioBundle:
        """Return a copy whose ``hash`` field is the deterministic content hash."""
        return self.model_copy(update={"hash": self.content_hash()})

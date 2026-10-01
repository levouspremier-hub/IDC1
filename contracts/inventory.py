"""Versioned planning extension; frozen forecast contracts remain unchanged."""

from typing import ClassVar, Literal

from contracts.models import ContractBase, SystemSnapshot
from contracts.validators import validate_snapshot


class TerminalInventory(ContractBase):
    version: Literal["terminal-inventory-v1"] = "terminal-inventory-v1"
    episode_end_step: int
    remaining_steps: int
    target_kwh: float
    lower_kwh: float
    upper_kwh: float

    UNITS: ClassVar[dict[str, str]] = {
        "episode_end_step": "step index", "remaining_steps": "steps",
        "target_kwh": "kWh", "lower_kwh": "kWh", "upper_kwh": "kWh",
    }


class ArrivedServiceReserve(ContractBase):
    version: Literal["arrived-service-reserve-v1"] = "arrived-service-reserve-v1"
    renewable_reserve_assumption: Literal["zero-renewables-through-real-remainder"] = (
        "zero-renewables-through-real-remainder")
    temperature_margin_c: float = 0.0
    temperature_reserve_assumption: str = "B6 forecast plus registered nonnegative train margin"
    charge_limit_kw: float
    charge_limits_kw: list[float]
    reserved_service_power_kw: float
    group_work_floor: list[float]
    current_allocation: list[list[float]]
    note: str


class InventorySnapshot(SystemSnapshot):
    terminal_inventory_enabled: bool = True
    terminal_inventory: TerminalInventory | None
    service_guard: ArrivedServiceReserve | None = None


def validate_inventory_snapshot(snapshot: InventorySnapshot) -> None:
    validate_snapshot(snapshot)
    terminal = snapshot.terminal_inventory
    if not snapshot.terminal_inventory_enabled or terminal is None:
        raise ValueError("terminal inventory metadata is required")
    if (terminal.episode_end_step - snapshot.step != terminal.remaining_steps
            or terminal.remaining_steps != snapshot.planning_horizon_steps):
        raise ValueError("terminal inventory must refer to the real episode end")
    if not (snapshot.soc_min_kwh <= terminal.lower_kwh <= terminal.target_kwh
            <= terminal.upper_kwh <= snapshot.soc_max_kwh):
        raise ValueError("terminal inventory bounds are invalid")
    guard = snapshot.service_guard
    if guard is not None:
        import numpy as np
        floor = np.asarray(guard.group_work_floor)
        allocation = np.asarray(guard.current_allocation).reshape(
            len(snapshot.tasks), len(snapshot.group_work_capacity))
        charge_limits = np.asarray(guard.charge_limits_kw)
        if (len(floor) != len(snapshot.group_work_capacity)
                or not np.all(np.isfinite(floor)) or np.any(floor < 0)
                or np.any(floor > np.asarray(snapshot.group_work_capacity) + 1e-8)
                or not 0 <= guard.charge_limit_kw <= snapshot.bess_charge_power_max_kw
                or len(charge_limits) != snapshot.planning_horizon_steps
                or not np.all(np.isfinite(charge_limits)) or np.any(charge_limits < 0)
                or np.any(charge_limits > snapshot.bess_charge_power_max_kw)
                or charge_limits[0] != guard.charge_limit_kw
                or not np.isfinite(guard.temperature_margin_c)
                or guard.temperature_margin_c < 0
                or not np.isfinite(guard.reserved_service_power_kw)
                or guard.reserved_service_power_kw < 0
                or not np.all(np.isfinite(allocation)) or np.any(allocation < 0)
                or not np.allclose(allocation.sum(axis=0), floor, atol=1e-8, rtol=0)):
            raise ValueError("invalid arrived service guard")

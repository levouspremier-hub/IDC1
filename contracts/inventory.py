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


class InventorySnapshot(SystemSnapshot):
    terminal_inventory_enabled: bool = True
    terminal_inventory: TerminalInventory | None


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

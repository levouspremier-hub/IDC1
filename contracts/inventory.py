"""Versioned planning extension; frozen forecast contracts remain unchanged."""

from typing import ClassVar, Literal

from pydantic import Field

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
    version: Literal["arrived-service-reserve-v1", "arrived-service-reserve-v2"] = (
        "arrived-service-reserve-v1")
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
    known_service_allocation: list[list[list[float]]] = Field(default_factory=list)
    known_service_order: list[int] = Field(default_factory=list)
    known_service_required_end_work: list[float] = Field(default_factory=list)
    known_service_required_due_work: list[float] = Field(default_factory=list)
    known_service_shortfall_work: float = 0.
    reserve_base_power_kw: list[float] = Field(default_factory=list)
    reserve_power_coefficients_kw_per_work: list[list[float]] = Field(default_factory=list)
    aggregate_arrival_work: list[float] = Field(default_factory=list)
    aggregate_service_work: list[float] = Field(default_factory=list)
    aggregate_backlog_work: list[float] = Field(default_factory=list)


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
        if guard.version == "arrived-service-reserve-v2":
            H, n, G = snapshot.planning_horizon_steps, len(snapshot.tasks), len(floor)
            schedule = np.asarray(guard.known_service_allocation, dtype=float).reshape(H, n, G)
            coefficients = np.asarray(guard.reserve_power_coefficients_kw_per_work, dtype=float)
            base = np.asarray(guard.reserve_base_power_kw, dtype=float)
            arrivals = np.asarray(guard.aggregate_arrival_work, dtype=float)
            served = np.asarray(guard.aggregate_service_work, dtype=float)
            backlog = np.asarray(guard.aggregate_backlog_work, dtype=float)
            end = np.asarray(guard.known_service_required_end_work, dtype=float)
            due = np.asarray(guard.known_service_required_due_work, dtype=float)
            if (coefficients.shape != (H, G) or base.shape != (H,)
                    or arrivals.shape != (H,) or served.shape != (H,)
                    or backlog.shape != (H + 1,) or end.shape != (n,) or due.shape != (n,)
                    or sorted(guard.known_service_order or []) != list(range(n))
                    or any(not np.all(np.isfinite(v)) or np.any(v < 0) for v in
                           (schedule, coefficients, base, arrivals, served, backlog, end, due))
                    or arrivals[0] != 0 or backlog[0] != 0
                    or not np.allclose(backlog[1:], backlog[:-1] + arrivals - served,
                                       atol=1e-8, rtol=0)
                    or np.any(schedule.sum(axis=1) > np.asarray(
                        snapshot.group_work_capacity) + 1e-8)
                    or np.any(schedule.sum(axis=(1, 2)) + served
                              > sum(snapshot.group_work_capacity) + 1e-8)
                    or not np.isfinite(guard.known_service_shortfall_work)
                    or guard.known_service_shortfall_work < 0):
                raise ValueError("invalid schedule-coupled service reserve")


def summarize_inventory_audits(audits: list[dict], actual_gap_kwh: float) -> dict:
    """Explain observed evidence, never infer forecast error from SOC alone."""
    events = [a["transition_review"] for a in audits
              if a.get("transition_review", {}).get("gap_increase_kwh", 0.) > 1e-6]
    unknown = any(a.get("target_reachable") is None for a in audits)
    consistency = any(e["classification"] == "planner_consistency_loss" for e in events)
    if unknown:
        classification, complete = "unproven_reachability", False
    elif consistency:
        classification, complete = "planner_consistency_loss", False
    elif actual_gap_kwh <= 1e-6:
        classification, complete = "target_met", True
    elif not events and audits and audits[0].get("target_reachable") is False:
        initial = audits[0].get("target_gap_kwh")
        complete = initial is not None and actual_gap_kwh <= initial + 1e-6
        classification = "registered_initial_gap" if complete else "unexplained_terminal_gap"
    else:
        classification, complete = "input_or_execution_revision_requires_review", False
    return {"version": "inventory-progress-audit-v1", "classification": classification,
            "explanation_complete": complete, "events": events,
            "initial_planning_gap_kwh": audits[0].get("target_gap_kwh") if audits else None,
            "actual_terminal_gap_kwh": actual_gap_kwh,
            "planner_consistency_loss_count": sum(
                e["classification"] == "planner_consistency_loss" for e in events),
            "prediction_error_proven": False, "physical_unreachability_proven": False}

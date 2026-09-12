"""M4 correctness gates on a tiny instance: completion, mutex, SOC bounds."""

from __future__ import annotations

import numpy as np

from planning.model import PlanningProblem, TaskPlan
from planning.solver import solve


def _tiny() -> PlanningProblem:
    return PlanningProblem(
        horizon=4,
        tasks=[TaskPlan(task_id=0, remaining_work=10.0, max_rate=5.0, start=0, deadline=3)],
        group_cap=np.array([5.0, 5.0]),
        soc0=0.5,
        soc_min=0.1,
        soc_max=0.9,
        cap_kwh=100.0,
        charge_max_kw=20.0,
        discharge_max_kw=20.0,
        charge_eff=0.95,
        discharge_eff=0.95,
        dt=1.0,
        pv_available=np.zeros(4),
        access_limit_kw=1000.0,
        p_base_kw=10.0,
        p_slope_kw_per_work=0.1,
        price=np.full(4, 0.2),
        degradation_per_kwh=0.02,
    )


def _series(x: np.ndarray | None, layout: dict, key: str, size: int) -> np.ndarray:
    assert x is not None
    base = layout[key]
    return np.asarray(x[base : base + size], dtype=float)


def test_tiny_mip_optimal_and_completes() -> None:
    out = solve(_tiny(), mip=True)
    assert out.status == "optimal"
    x = np.asarray(out.x)
    layout = out.layout
    # task execution sums to remaining work over the window
    total = x[layout["X0"] : layout["X0"] + 4].sum()
    assert abs(total - 10.0) < 1e-6


def test_charge_discharge_mutex() -> None:
    out = solve(_tiny(), mip=True)
    c = _series(out.x, out.layout, "C0", 4)
    d = _series(out.x, out.layout, "D0", 4)
    assert not np.any((c > 1e-9) & (d > 1e-9))


def test_soc_within_bounds() -> None:
    out = solve(_tiny(), mip=True)
    soc = _series(out.x, out.layout, "SOC0", 5)  # T+1 values
    assert np.all(soc >= 0.1 - 1e-9) and np.all(soc <= 0.9 + 1e-9)
    assert abs(soc[0] - 0.5) < 1e-9

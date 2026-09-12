"""M4.4 gates: the rolling corrector minimally modifies the raw action."""

from __future__ import annotations

import pytest

from contracts import DispatchProposal, SystemSnapshot
from planning.corrector import RollingCorrector


def _ctx(soc: float, tasks: list[dict], horizon: int = 4, group_cap=None) -> dict:
    return {
        "horizon": horizon,
        "group_cap": group_cap or [10.0, 10.0],
        "tasks": tasks,
        "soc_min": 0.1,
        "soc_max": 0.9,
        "capacity_kwh": 100.0,
        "charge_power_max_kw": 20.0,
        "discharge_power_max_kw": 20.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "delta_t_hours": 1.0,
        "pv_available": [0.0] * horizon,
        "access_limit_kw": 1000.0,
        "p_base_kw": 10.0,
        "p_slope_kw_per_work": 0.1,
        "price": [0.2] * horizon,
        "degradation_per_kwh": 0.02,
    }


def _snap(soc: float, ctx: dict) -> SystemSnapshot:
    return SystemSnapshot(t=0, task_states=[], soc=soc, corrector_context=ctx)


def test_identity_when_raw_is_feasible() -> None:
    ctx = _ctx(
        soc=0.5,
        tasks=[{"remaining_work": 10.0, "max_rate": 10.0, "start": 0, "deadline": 3}],
    )
    r = RollingCorrector().correct(_snap(0.5, ctx), DispatchProposal(compute=[0.5, 0.5], storage=-0.2))
    assert r.solve_status == "optimal"
    assert r.correction_reason == "none"
    assert r.exec_compute == [pytest.approx(x, abs=1e-6) for x in (0.5, 0.5)]
    assert r.exec_storage == pytest.approx(-0.2, abs=1e-6)


def test_storage_clipped_when_soc_near_max() -> None:
    ctx = _ctx(
        soc=0.89,
        tasks=[{"remaining_work": 10.0, "max_rate": 10.0, "start": 0, "deadline": 3}],
    )
    r = RollingCorrector().correct(_snap(0.89, ctx), DispatchProposal(compute=[0.5, 0.5], storage=-1.0))
    assert r.solve_status == "optimal"
    assert r.correction_reason == "corrected"
    # headroom = (0.9 - 0.89) * 100 / 0.95 = 1.0526 kW -> exec storage = -1.0526 / 20
    assert r.exec_storage == pytest.approx(-1.0526 / 20.0, rel=1e-2)


def test_infeasible_task_reports_business_gap() -> None:
    # 100 work, max 5/hour, deadline at step 1 -> at most 10 work; gap = 90.
    ctx = _ctx(
        soc=0.5,
        tasks=[{"remaining_work": 100.0, "max_rate": 5.0, "start": 0, "deadline": 1}],
    )
    r = RollingCorrector().correct(_snap(0.5, ctx), DispatchProposal(compute=[0.5, 0.5], storage=0.0))
    assert r.solve_status == "optimal"
    assert r.correction_reason == "business_gap"
    assert r.business_gap == pytest.approx(90.0, abs=1e-4)

"""M4.3 测试：MILP 二元互斥、LP 松弛、窗口、容量/剩余约束。"""

import pytest

from contracts.models import ScenarioBundle, SystemSnapshot, TaskState
from planning.solver import solve
from planning.window import compute_window


def _forecast() -> ScenarioBundle:
    return ScenarioBundle(
        split="train",
        start="0",
        horizon=24,
        forecast_cutoff=4,
        price_forecast=[0.2] * 24,
        load_forecast=[0.0] * 24,
        pv_forecast=[0.0] * 24,
        wind_forecast=[0.0] * 24,
        temperature_forecast=[28.0] * 24,
        arrival_forecast=[0.0] * 24,
        carbon_forecast=[0.0] * 24,
        source_hashes={"x": "y"},
    )


def _snapshot() -> SystemSnapshot:
    tasks = [
        TaskState(
            task_id="t0",
            remaining_work=5.0,
            deadline=8,
            priority=1.0,
            status="waiting",
            max_rate_work_per_step=1.0,
        ),
        TaskState(
            task_id="t1",
            remaining_work=3.0,
            deadline=6,
            priority=2.0,
            status="waiting",
            max_rate_work_per_step=1.0,
        ),
    ]
    return SystemSnapshot(
        step=0,
        soc_kwh=50.0,
        soc_min_kwh=10.0,
        soc_max_kwh=90.0,
        group_capacity_kw=[4.0, 6.0],
        delta_t_hours=1.0,
        planning_horizon_steps=24,
        soc_capacity_kwh=100.0,
        bess_charge_power_max_kw=20.0,
        bess_discharge_power_max_kw=20.0,
        bess_charge_efficiency=0.95,
        bess_discharge_efficiency=0.95,
        bess_degradation_cost_per_kwh=0.02,
        base_idc_power_forecast_kw=[14.0] * 24,
        group_power_coeff_kw_per_work=[0.01] * 2,
        group_power_upper_kw=[0.7] * 2,
        power_approximation_note="planning approximation; verify with env physics chain",
        access_limit_kw=12.0,
        budget_remaining_sgd=1000.0,
        tasks=tasks,
        forecast=_forecast(),
    )


def test_mip_solves():
    r = solve(_snapshot())
    assert r.success
    assert r.allocation.shape == (2, 2)


def test_mip_mutual_exclusion():
    r = solve(_snapshot())
    assert r.charge * r.discharge == pytest.approx(0.0, abs=1e-6)
    assert abs(r.z - round(r.z)) < 1e-6  # z 为二元


def test_group_capacity_respected():
    snap = _snapshot()
    r = solve(snap)
    for g in range(2):
        col_sum = sum(r.allocation[i][g] for i in range(2))
        assert col_sum <= snap.group_capacity_kw[g] + 1e-6


def test_task_remaining_respected():
    snap = _snapshot()
    r = solve(snap)
    for i, task in enumerate(snap.tasks):
        assert sum(r.allocation[i]) <= task.remaining_work + 1e-6


def test_lp_relaxation_runs():
    r = solve(_snapshot(), allow_lp_relaxation=True)
    assert r.success


def test_window_extends_to_latest_deadline():
    tasks = [
        TaskState(
            task_id="t",
            remaining_work=1.0,
            deadline=30,
            priority=1.0,
            status="waiting",
            max_rate_work_per_step=1.0,
        ),
    ]
    assert compute_window(tasks, default_horizon=24) == 30


def test_window_defaults_to_24():
    tasks = [
        TaskState(
            task_id="t",
            remaining_work=1.0,
            deadline=5,
            priority=1.0,
            status="waiting",
            max_rate_work_per_step=1.0,
        ),
    ]
    assert compute_window(tasks, default_horizon=24) == 24

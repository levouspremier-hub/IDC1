"""M4.4 测试：失败分类与安全回退。"""

from contracts.models import (
    DispatchProposal,
    PlanningExogenousForecast,
    ScenarioBundle,
    SystemSnapshot,
    TaskState,
)
from planning.corrector import FailureClass, correct


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


def _snapshot(access_limit_kw=50.0, n_group=2) -> SystemSnapshot:
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
        group_work_capacity=[4.0] * n_group,
        delta_t_hours=1.0,
        planning_horizon_steps=24,
        soc_capacity_kwh=100.0,
        bess_charge_power_max_kw=20.0,
        bess_discharge_power_max_kw=20.0,
        bess_charge_efficiency=0.95,
        bess_discharge_efficiency=0.95,
        bess_degradation_cost_per_kwh=0.02,
        base_idc_power_forecast_kw=[14.0] * 24,
            planning_forecast=PlanningExogenousForecast(
            horizon_steps=24,
            price=[0.2] * 24, pv=[0.0] * 24, wind=[0.0] * 24,
            temperature=[28.0] * 24, carbon=[0.0] * 24, arrival=[0.0] * 24,
            base_idc_power=[14.0] * 24,
            visible_mask=[False] * 24, assumed_mask=[True] * 24,
            extension_policy="test fixture policy",
        ),

        group_power_coeff_kw_per_work=[0.01] * n_group,
        group_power_upper_kw=[0.7] * n_group,
        power_approximation_note="planning approximation; verify with env physics chain",
        access_limit_kw=access_limit_kw,
        budget_remaining_sgd=1000.0,
        tasks=tasks,
        forecast=_forecast(),
    )


def test_success_is_reviewed():
    snap = _snapshot()
    p = DispatchProposal(compute_actions=[0.5, 0.5], storage_action=0.0)
    c = correct(snap, p)
    assert c.failure == FailureClass.NONE
    assert c.reviewed is True


def test_infeasible_returns_boundary_action(monkeypatch):
    """接入(12) < 基础负载(14) → base-only 不可行 → 零动作回退。"""
    snap = _snapshot(access_limit_kw=12.0)  # 接入 < 基础负载(14) → 物理不可行
    p = DispatchProposal(compute_actions=[0.5, 0.5], storage_action=0.0)
    c = correct(snap, p)
    assert c.failure in (FailureClass.BASE_SHORTAGE, FailureClass.SOLVER_FAILURE)
    assert c.exec_compute_actions == [0.0, 0.0]
    assert c.exec_storage_action == 0.0
    assert c.business_gap == 8.0


def test_proposal_dimension_mismatch_returns_boundary_action():
    snap = _snapshot()
    p = DispatchProposal(compute_actions=[0.5, 0.5, 0.5], storage_action=0.0)  # 维度不符
    c = correct(snap, p)
    assert c.failure == FailureClass.PROPOSAL_INVALID
    assert c.exec_compute_actions == [0.0, 0.0]
    assert c.reviewed is True


def test_proposal_out_of_range_returns_boundary_action():
    snap = _snapshot()
    p = DispatchProposal(compute_actions=[0.5, 1.5], storage_action=0.0)  # 超出 [0,1]
    c = correct(snap, p)
    assert c.failure == FailureClass.PROPOSAL_INVALID
    assert c.exec_compute_actions == [0.0, 0.0]

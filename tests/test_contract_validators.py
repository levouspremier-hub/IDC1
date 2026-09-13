"""M2.2 跨契约校验器测试：每种错误明确失败并指出具体字段。"""

import pytest

from contracts import (
    DispatchProposal,
    DispatchResult,
    ScenarioBundle,
    SystemSnapshot,
    TaskAllocation,
)
from contracts.models import PlanningExogenousForecast
from contracts.validators import (
    validate_dispatch_proposal,
    validate_dispatch_result,
    validate_scenario,
    validate_snapshot,
    validate_task_allocation,
)


def _scenario(**overrides) -> ScenarioBundle:
    kwargs = dict(
        split="train",
        start="2026-01-01T00:00:00",
        horizon=24,
        forecast_cutoff=6,
        price_forecast=[0.2] * 6,
        load_forecast=[6000.0] * 6,
        pv_forecast=[0.0] * 6,
        wind_forecast=[0.0] * 6,
        temperature_forecast=[28.0] * 6,
        arrival_forecast=[0.0] * 6,
        carbon_forecast=[0.0] * 6,
        source_hashes={"a": "x"},
    )
    kwargs.update(overrides)
    return ScenarioBundle(**kwargs)  # type: ignore[arg-type]


def _snapshot(**overrides) -> SystemSnapshot:
    kwargs = dict(
        step=0,
        delta_t_hours=1.0,
        planning_horizon_steps=24,
        soc_kwh=50.0,
        soc_min_kwh=10.0,
        soc_max_kwh=90.0,
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

        power_approximation_note="planning approximation; verify with env physics chain",
        group_work_capacity=[1.0] * 20,
        access_limit_kw=18.0,
        budget_remaining_sgd=100.0,
        tasks=[],
        forecast=_scenario(),
    )
    kwargs.update(overrides)
    n_group = len(kwargs["group_work_capacity"])  # type: ignore[arg-type]
    kwargs.setdefault("group_power_coeff_kw_per_work", [0.01] * n_group)
    kwargs.setdefault("group_power_upper_kw", [0.7] * n_group)
    return SystemSnapshot(**kwargs)  # type: ignore[arg-type]


def test_scenario_length_mismatch_fails():
    s = _scenario(load_forecast=[1.0] * 5)
    with pytest.raises(ValueError, match="时间轴长度不一致"):
        validate_scenario(s)


def test_scenario_missing_source_fails():
    s = _scenario(source_hashes={})
    with pytest.raises(ValueError, match="来源"):
        validate_scenario(s)


def test_scenario_valid_passes():
    validate_scenario(_scenario())


def test_snapshot_soc_out_of_range_fails():
    with pytest.raises(ValueError, match="soc_kwh"):
        validate_snapshot(_snapshot(soc_kwh=200.0))


def test_snapshot_negative_capacity_fails():
    with pytest.raises(ValueError, match="group_work_capacity"):
        validate_snapshot(_snapshot(group_work_capacity=[1.0] * 19 + [-1.0]))


def test_snapshot_valid_passes():
    validate_snapshot(_snapshot())


def test_proposal_dimension_mismatch_fails():
    snap = _snapshot()
    p = DispatchProposal(compute_actions=[0.5] * 21, storage_action=0.0)
    with pytest.raises(ValueError, match="compute_actions 长度"):
        validate_dispatch_proposal(p, snap)


def test_proposal_storage_out_of_range_fails():
    p = DispatchProposal(compute_actions=[0.5] * 20, storage_action=1.5)
    with pytest.raises(ValueError, match="storage_action"):
        validate_dispatch_proposal(p, _snapshot())


def test_allocation_exceeds_max_rate_fails():
    snap = _snapshot(group_work_capacity=[10.0] * 2)
    alloc = TaskAllocation(
        task_ids=["t1"], group_ids=[0, 1], matrix=[[6.0, 5.0]]
    )
    with pytest.raises(ValueError, match="超 max_rate"):
        validate_task_allocation(alloc, snap, max_rate={"t1": 8.0})


def test_allocation_exceeds_group_capacity_fails():
    snap = _snapshot(group_work_capacity=[10.0, 1.0])
    alloc = TaskAllocation(task_ids=["t1", "t2"], group_ids=[0, 1], matrix=[[0.0, 0.0], [0.0, 5.0]])
    with pytest.raises(ValueError, match="超容量"):
        validate_task_allocation(alloc, snap, max_rate={"t1": 100.0, "t2": 100.0})


def test_allocation_valid_passes():
    snap = _snapshot(group_work_capacity=[10.0, 10.0])
    alloc = TaskAllocation(task_ids=["t1"], group_ids=[0, 1], matrix=[[3.0, 2.0]])
    validate_task_allocation(alloc, snap, max_rate={"t1": 8.0})


def test_dispatch_result_dimension_mismatch_fails():
    r = DispatchResult(
        raw_compute_actions=[0.5] * 20,
        raw_storage_action=0.0,
        exec_compute_actions=[0.4] * 19,
        exec_storage_action=0.0,
        correction_reason="x",
        business_gap=0.0,
        solve_time_s=0.0,
        p_grid_kw=1.0,
        soc_next_kwh=50.0,
        cost_sgd=1.0,
        carbon_kg=0.5,
    )
    with pytest.raises(ValueError, match="raw/exec"):
        validate_dispatch_result(r)

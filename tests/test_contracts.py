"""M2.1 契约测试：JSON 往返、矩形/非负、单位、hash 键序无关、frozen。"""

import pytest
from pydantic import ValidationError

from contracts import (
    DispatchProposal,
    DispatchResult,
    EvaluationRecord,
    ScenarioBundle,
    SystemSnapshot,
    TaskAllocation,
    TaskState,
)


def _scenario(source_hashes: dict[str, str] | None = None) -> ScenarioBundle:
    return ScenarioBundle(
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
        source_hashes=source_hashes or {"a": "x", "b": "y"},
    )


def test_scenario_json_roundtrip():
    s = _scenario()
    s2 = ScenarioBundle.model_validate_json(s.model_dump_json())
    assert s == s2
    assert s.model_dump() == s2.model_dump()


def test_scenario_hash_key_order_independent():
    a = _scenario({"a": "x", "b": "y"})
    b = _scenario({"b": "y", "a": "x"})
    assert a.content_hash() == b.content_hash()


def test_contracts_frozen():
    s = _scenario()
    with pytest.raises(ValidationError):
        s.horizon = 12  # type: ignore[misc]


def test_task_allocation_rectangular_nonnegative():
    a = TaskAllocation(
        task_ids=["t1", "t2"], group_ids=[0, 1, 2], matrix=[[1.0, 0.0, 2.0], [0.5, 1.5, 0.0]]
    )
    assert a.matrix[0][2] == 2.0
    assert a.matrix[1][1] == 1.5


def test_task_allocation_rejects_ragged():
    with pytest.raises(ValidationError):
        TaskAllocation(task_ids=["t1", "t2"], group_ids=[0, 1], matrix=[[1.0], [0.0, 1.0]])


def test_task_allocation_rejects_negative():
    with pytest.raises(ValidationError):
        TaskAllocation(task_ids=["t1"], group_ids=[0, 1], matrix=[[1.0, -0.5]])


def test_units_declared_for_physical_fields():
    assert DispatchResult.UNITS["cost_sgd"] == "SGD"
    assert DispatchResult.UNITS["carbon_kg"] == "kgCO2"
    assert DispatchResult.UNITS["soc_next_kwh"] == "kWh"
    assert SystemSnapshot.UNITS["access_limit_kw"] == "kW"
    assert EvaluationRecord.UNITS["total_cost_sgd"] == "SGD"


def test_nested_roundtrip():
    snap = SystemSnapshot(
        step=0,
        soc_kwh=50.0,
        soc_min_kwh=10.0,
        soc_max_kwh=90.0,
        group_capacity_kw=[1.0] * 20,
        delta_t_hours=1.0,
        planning_horizon_steps=24,
        soc_capacity_kwh=100.0,
        bess_charge_power_max_kw=20.0,
        bess_discharge_power_max_kw=20.0,
        bess_charge_efficiency=0.95,
        bess_discharge_efficiency=0.95,
        bess_degradation_cost_per_kwh=0.02,
        base_idc_power_forecast_kw=[14.0] * 24,
        group_power_coeff_kw_per_work=[0.01] * 1,
        group_power_upper_kw=[0.7] * 1,
        power_approximation_note="planning approximation; verify with env physics chain",
        access_limit_kw=18.0,
        budget_remaining_sgd=100.0,
        tasks=[
            TaskState(
                task_id="t0", remaining_work=10.0, deadline=5, priority=1.0,
                status="backlog", max_rate_work_per_step=2.0,
            )
        ],
        forecast=_scenario(),
    )
    snap2 = SystemSnapshot.model_validate_json(snap.model_dump_json())
    assert snap == snap2


def test_proposal_and_result_roundtrip():
    p = DispatchProposal(compute_actions=[0.5] * 20, storage_action=-0.3)
    r = DispatchResult(
        raw_compute_actions=[0.5] * 20,
        raw_storage_action=-0.3,
        exec_compute_actions=[0.4] * 20,
        exec_storage_action=-0.2,
        correction_reason="soc_clip",
        business_gap=0.0,
        solve_time_s=0.01,
        p_grid_kw=12.0,
        soc_next_kwh=48.0,
        cost_sgd=3.0,
        carbon_kg=1.0,
    )
    assert p.model_validate_json(p.model_dump_json()) == p
    assert r.model_validate_json(r.model_dump_json()) == r

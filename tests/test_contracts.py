"""M2.1 契约测试：JSON 往返、矩形/非负、单位、hash 键序无关、frozen。

**M1.3e 迁移 / M1.3g-0 版本升级**：`ScenarioBundle` 现为 `contract-v9`——自由 dict `source_hashes`
退役为结构化 `forecast_provenance`（七个字段一一对应），`synthetic: bool`
由显式 `mode` 取代。以下 fixture 因此显式提供 v8 provenance，
**原有断言一条都未弱化**。
"""

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
from contracts.models import BUNDLE_FORECAST_FIELDS, PlanningExogenousForecast


def _series_provenance(field: str) -> dict:
    return dict(
        series_name=field,
        source_kind="synthetic",
        method="test_fixture",
        generated_at="2026-01-01T00:00:00+08:00",
        information_cutoff_exclusive="2026-01-01T00:00:00+08:00",
        target_start="2026-01-01T00:00:00+08:00",
        target_end_exclusive="2026-01-01T06:00:00+08:00",
        lookback_start=None,
        lookback_end_exclusive=None,
        model_name="test_fixture",
        model_version="v1",
        code_revision="a" * 40,
        seed=None,
        sources=[
            {"role": "fixture", "logical_path": "tests/fixtures/none",
             "sha256": "b" * 64}
        ],
    )


def _provenance(*, reverse: bool = False) -> dict:
    fields = list(BUNDLE_FORECAST_FIELDS)
    if reverse:
        fields = fields[::-1]
    return {field: _series_provenance(field) for field in fields}


def _scenario(*, reverse_provenance: bool = False) -> ScenarioBundle:
    return ScenarioBundle.model_validate({
        "split": "train",
        "start": "2026-01-01T00:00:00",
        "horizon": 24,
        "forecast_cutoff": 6,
        "price_forecast": [0.2] * 6,
        "load_forecast": [6000.0] * 6,
        "pv_forecast": [0.0] * 6,
        "wind_forecast": [0.0] * 6,
        "temperature_forecast": [28.0] * 6,
        "arrival_forecast": [0.0] * 6,
        "carbon_forecast": [0.0] * 6,
        "mode": "synthetic",
        "generated_at": "2026-01-01T00:00:00+08:00",
        "forecast_provenance": _provenance(reverse=reverse_provenance),
    })


def test_scenario_json_roundtrip():
    s = _scenario()
    s2 = ScenarioBundle.model_validate_json(s.model_dump_json())
    assert s == s2
    assert s.model_dump() == s2.model_dump()


def test_scenario_hash_key_order_independent():
    """hash 与 dict 键顺序无关（provenance 的键序同样不得影响）。"""
    a = _scenario()
    b = _scenario(reverse_provenance=True)
    assert a.model_dump() == b.model_dump()
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
        group_work_capacity=[1.0] * 20,
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


# --- M1.3g-0：contract-v9 与 carbon 的 human-approved source_kind -----------

CARBON_KIND = "human_approved_external_low_resolution"
NON_CARBON_FIELDS = tuple(
    f for f in BUNDLE_FORECAST_FIELDS if f != "carbon_forecast"
)


def _formal_provenance_kwargs(*, carbon_kind: str = CARBON_KIND) -> dict:
    provenance = {
        field: _series_provenance(field) for field in BUNDLE_FORECAST_FIELDS
    }
    for field in BUNDLE_FORECAST_FIELDS:
        provenance[field]["source_kind"] = "persistence"
    provenance["carbon_forecast"]["source_kind"] = carbon_kind
    return dict(
        split="train", start="2026-01-01T00:00:00+08:00", horizon=24,
        forecast_cutoff=4,
        **{f: [0.0] * 4 for f in BUNDLE_FORECAST_FIELDS},
        mode="formal",
        generated_at="2026-01-01T00:00:00+08:00",
        forecast_provenance=provenance,
    )


def test_contract_v9_accepts_the_human_approved_kind_for_formal_carbon():
    """D1：只有 formal 的 `carbon_forecast` 可以使用该 source_kind。"""
    bundle = ScenarioBundle(**_formal_provenance_kwargs())
    assert bundle.mode == "formal"
    assert bundle.forecast_provenance.carbon_forecast.source_kind == CARBON_KIND


@pytest.mark.parametrize("field", NON_CARBON_FIELDS)
def test_contract_v9_rejects_the_human_approved_kind_elsewhere(field):
    """D1：该 kind **只**绑定 `carbon_forecast`，六个非 carbon 序列一律拒绝。"""
    kwargs = _formal_provenance_kwargs()
    kwargs["forecast_provenance"][field]["source_kind"] = CARBON_KIND
    with pytest.raises((ValidationError, ValueError)):
        ScenarioBundle(**kwargs)


def test_contract_v9_still_rejects_unavailable_in_a_formal_bundle():
    """D1 不放松既有规则：`unavailable` 仍不得进入任何完整 bundle。"""
    kwargs = _formal_provenance_kwargs()
    kwargs["forecast_provenance"]["pv_forecast"]["source_kind"] = "unavailable"
    with pytest.raises((ValidationError, ValueError)):
        ScenarioBundle(**kwargs)

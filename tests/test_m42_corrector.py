"""M4.2 测试：单步修正器修正计算范围、SOC、接入上限；确定性；记录 raw/exec/reason。"""

from contracts.models import DispatchProposal, ScenarioBundle, SystemSnapshot
from safe_rl.corrector import correct


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


def _snapshot(soc_kwh=50.0, access_limit_kw=10.0, capacity=None) -> SystemSnapshot:
    capacity = capacity if capacity is not None else [10.0] * 3
    return SystemSnapshot(
        step=0,
        soc_kwh=soc_kwh,
        soc_min_kwh=10.0,
        soc_max_kwh=90.0,
        group_work_capacity=capacity,
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
        access_limit_kw=access_limit_kw,
        budget_remaining_sgd=1000.0,
        tasks=[],
        forecast=_forecast(),
    )


def test_compute_clip():
    snap = _snapshot(access_limit_kw=1000.0)
    p = DispatchProposal(compute_actions=[1.5, -0.5, 0.5], storage_action=0.0)
    r = correct(snap, p)
    assert r.exec_compute_actions == [1.0, 0.0, 0.5]
    assert "compute_clip" in r.correction_reason


def test_soc_full_blocks_charge():
    snap = _snapshot(soc_kwh=90.0)
    p = DispatchProposal(compute_actions=[0.5, 0.5, 0.5], storage_action=-1.0)  # 负值=充电
    r = correct(snap, p)
    assert r.exec_storage_action == 0.0
    assert "soc_full" in r.correction_reason


def test_soc_empty_blocks_discharge():
    snap = _snapshot(soc_kwh=10.0)
    p = DispatchProposal(compute_actions=[0.5, 0.5, 0.5], storage_action=1.0)  # 正值=放电
    r = correct(snap, p)
    assert r.exec_storage_action == 0.0
    assert "soc_empty" in r.correction_reason


def test_access_limit_reduces_compute_and_records_gap():
    snap = _snapshot(access_limit_kw=10.0, capacity=[10.0, 10.0, 10.0])
    p = DispatchProposal(compute_actions=[1.0, 1.0, 1.0], storage_action=0.0)
    r = correct(snap, p)
    assert sum(r.exec_compute_actions) <= 1.0 + 1e-9  # 缩减后 idc_power <= access_limit
    assert r.business_gap > 0.0
    assert "access_limit" in r.correction_reason


def test_deterministic():
    snap = _snapshot()
    p = DispatchProposal(compute_actions=[0.3, 0.7, 0.9], storage_action=0.2)
    r1 = correct(snap, p)
    r2 = correct(snap, p)
    assert r1.model_dump() == r2.model_dump()


def test_records_raw_exec():
    snap = _snapshot(access_limit_kw=1000.0)
    p = DispatchProposal(compute_actions=[0.1, 0.2, 0.3], storage_action=0.0)
    r = correct(snap, p)
    assert r.raw_compute_actions == [0.1, 0.2, 0.3]
    assert r.raw_storage_action == 0.0
    assert r.exec_compute_actions == [0.1, 0.2, 0.3]
    assert r.correction_reason == "none"

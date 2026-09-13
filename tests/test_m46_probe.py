"""M4.6 测试：probe 只报告现行 H 步规划路径（M4.4b 起旧单步路径已退役）。"""

import importlib
import json
import subprocess

import pytest

from contracts.models import (
    PlanningExogenousForecast,
    ScenarioBundle,
    SystemSnapshot,
    TaskState,
)
from planning.model import solve_time_indexed_lp, solve_time_indexed_mip


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


def _snapshot(horizon: int = 4) -> SystemSnapshot:
    return SystemSnapshot(
        step=0,
        soc_kwh=50.0,
        soc_min_kwh=10.0,
        soc_max_kwh=90.0,
        group_work_capacity=[4.0, 6.0],
        delta_t_hours=1.0,
        planning_horizon_steps=horizon,
        soc_capacity_kwh=100.0,
        bess_charge_power_max_kw=20.0,
        bess_discharge_power_max_kw=20.0,
        bess_charge_efficiency=0.95,
        bess_discharge_efficiency=0.95,
        bess_degradation_cost_per_kwh=0.02,
        base_idc_power_forecast_kw=[14.0] * horizon,
        planning_forecast=PlanningExogenousForecast(
            horizon_steps=horizon,
            price=[0.2] * horizon, pv=[0.0] * horizon, wind=[0.0] * horizon,
            temperature=[28.0] * horizon, carbon=[0.0] * horizon, arrival=[0.0] * horizon,
            base_idc_power=[14.0] * horizon,
            visible_mask=[False] * horizon, assumed_mask=[True] * horizon,
            extension_policy="test fixture policy",
        ),
        group_power_coeff_kw_per_work=[0.01] * 2,
        group_power_upper_kw=[0.7] * 2,
        power_approximation_note="planning approximation; verify with env physics chain",
        access_limit_kw=1000.0,
        budget_remaining_sgd=1000.0,
        tasks=[
            TaskState(
                task_id="t0", remaining_work=5.0, deadline=8, priority=1.0,
                status="waiting", max_rate_work_per_step=1.0,
            ),
        ],
        forecast=_forecast(),
    )


# --- H 步规划器规模验收（取代旧单步计数断言） ---

def test_h_step_lp_and_mip_var_and_integer_counts():
    snap = _snapshot(horizon=4)
    lp = solve_time_indexed_lp(snap)
    mip = solve_time_indexed_mip(snap, time_limit_s=5.0)

    assert lp.backend == "lp"
    assert lp.n_integer_variables == 0          # LP 为松弛，无整数变量
    assert mip.backend == "mip"
    assert mip.n_integer_variables == mip.horizon_steps  # 每步一个二元 z
    assert mip.n_variables == lp.n_variables + mip.horizon_steps


def test_mip_storage_mutual_exclusion_holds():
    mip = solve_time_indexed_mip(_snapshot(horizon=4), time_limit_s=5.0)
    for c, d in zip(mip.charge_kw, mip.discharge_kw, strict=True):
        assert c * d == pytest.approx(0.0, abs=1e-6)


# --- 防回归：旧单步路径已退役 ---

def test_legacy_solver_module_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("planning.solver")


def test_build_milp_no_longer_exported():
    import planning.model as mod

    assert not hasattr(mod, "build_milp")
    assert not hasattr(mod, "MilpModel")


def test_probe_output_has_no_legacy_fields():
    out = subprocess.run(
        ["uv", "run", "python", "-m", "planning.probe"],
        capture_output=True, text=True, check=True,
    ).stdout
    report = json.loads(out)
    assert not any(k.startswith("legacy_single_step_milp") for k in report), report.keys()
    for key in ("lp_backend", "mip_backend", "mip_n_integer", "corrector_total_mean_s"):
        assert key in report, key


def test_probe_timing_and_memory_are_separated():
    """M4.6a：计时区间不含 tracemalloc；内存采样独立，且字段分区标注。"""
    out = subprocess.run(
        ["uv", "run", "python", "-m", "planning.probe"],
        capture_output=True, text=True, check=True,
    ).stdout
    report = json.loads(out)

    timing = report["timing_without_tracemalloc"]
    memory = report["memory_with_tracemalloc"]
    assert timing["tracemalloc_active"] is False   # 计时区间绝不启用 tracemalloc
    assert memory["tracemalloc_active"] is True
    assert "peak_memory_bytes" in memory
    for key in ("corrector_solve_mean_s", "stage_a_solve_mean_s",
                "stage_b_solve_mean_s", "corrector_total_mean_s",
                "rollout_steps_per_s", "corrector_timeout_steps"):
        assert key in timing, key
    # 不得在顶层以未标注的名字输出计时/吞吐（避免被误读为真实训练吞吐）
    for key in ("corrector_solve_mean_s", "rollout_steps_per_s",
                "env_step_mean_s", "tracemalloc_instrumented"):
        assert key not in report, key

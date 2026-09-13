"""M4.3a1 测试：基础负载专用可行性诊断（与主 LP 同物理约束）。"""

import numpy as np
import pytest

from envs.idc_price_env import IDCPriceEnv20D
from planning.model import (
    FAILURE_BASE_SHORTAGE,
    FAILURE_NONE,
    diagnose_base_feasibility,
    solve_time_indexed_lp,
)
from planning.snapshot_adapter import build_snapshot

NG = 20
TOL = 1e-6


def _env(
    access_limit_kw: float = 10.0,
    soc_init: float = 0.5,
    horizon: int = 24,
    pv: float = 0.0,
    wind: float = 0.0,
) -> IDCPriceEnv20D:
    h = horizon
    env = IDCPriceEnv20D(
        horizon=h, forecast_cutoff=4, access_limit_kw=access_limit_kw,
        bess_soc_init=soc_init,
        price_t=0.12 + 0.01 * np.arange(h),
        pv_t=np.full(h, pv),
        wt_t=np.full(h, wind),
        carbon_factor_t=np.full(h, 0.5),
        T_amb=np.full(h, 25.0),
    )
    env.reset(seed=0)
    return env


# --- 1. 短时 SOC 放电足够 → 可行，不得误报 ---

def test_short_horizon_soc_discharge_sufficient_is_feasible():
    """接入(10) < 基础负载(≈14.7)，但 2 步内电池足以补足 → base-only 可行。"""
    snap = build_snapshot(_env(horizon=2, access_limit_kw=10.0, soc_init=0.5))
    assert snap.planning_horizon_steps == 2
    assert snap.planning_forecast.base_idc_power[0] > snap.access_limit_kw  # 静态启发式会误报
    diag = diagnose_base_feasibility(snap)
    assert diag.feasible is True
    assert diag.shortfall_kwh == 0.0


# --- 2. 跨时域 SOC 耗尽 → 报 base_shortage ---

def test_long_horizon_soc_exhaustion_is_infeasible():
    """同样接入，但 24 步内电池耗尽 → base-only 不可行。"""
    snap = build_snapshot(_env(horizon=24, access_limit_kw=10.0, soc_init=0.5))
    assert snap.planning_horizon_steps == 24
    diag = diagnose_base_feasibility(snap)
    assert diag.feasible is False
    assert diag.shortfall_kwh > 0.0


def test_main_lp_classified_base_shortage_matches_diagnostic():
    snap = build_snapshot(_env(horizon=24, access_limit_kw=10.0, soc_init=0.5))
    diag = diagnose_base_feasibility(snap)
    res = solve_time_indexed_lp(snap)
    assert diag.feasible is False
    assert res.failure_class == FAILURE_BASE_SHORTAGE
    assert res.base_diagnostic_status
    assert res.base_diagnostic_solve_time_s >= 0.0
    assert res.base_shortfall_kwh > 0.0


# --- 3. 可再生足够 → 不报 base_shortage ---

def test_renewables_cover_base_load_is_feasible():
    base_kw = build_snapshot(_env()).planning_forecast.base_idc_power[0]
    snap = build_snapshot(_env(access_limit_kw=1.0, soc_init=0.1, pv=base_kw + 5.0))
    diag = diagnose_base_feasibility(snap)
    assert diag.feasible is True
    res = solve_time_indexed_lp(snap)
    assert res.failure_class != FAILURE_BASE_SHORTAGE


# --- 4. 无储能、无可再生、接入不足 → 报 base_shortage ---

def test_no_storage_no_renewable_insufficient_access_is_shortage():
    snap = build_snapshot(_env(access_limit_kw=1.0, soc_init=0.1))
    diag = diagnose_base_feasibility(snap)
    assert diag.feasible is False
    res = solve_time_indexed_lp(snap)
    assert res.failure_class == FAILURE_BASE_SHORTAGE
    # 失败语义不变：零分配、业务缺口 = 全部剩余工作
    np.testing.assert_allclose(res.allocation, 0.0)
    assert res.business_shortfall_work == pytest.approx(
        sum(t.remaining_work for t in snap.tasks), abs=1e-9
    )


# --- 5. 诊断模型自身的物理残差 ---

def test_diagnostic_respects_soc_and_power_bounds():
    """诊断可行时，其隐含解必须满足 SOC/功率/能量平衡（用主 LP 结果交叉校验）。"""
    snap = build_snapshot(_env(horizon=4, access_limit_kw=10.0, soc_init=0.5))
    res = solve_time_indexed_lp(snap)
    assert res.failure_class == FAILURE_NONE
    assert res.max_constraint_residual < 1e-6
    for k in range(res.horizon_steps):
        assert -TOL <= res.p_grid_kw[k] <= snap.access_limit_kw + TOL
        assert res.charge_kw[k] <= snap.bess_charge_power_max_kw + TOL
        assert res.discharge_kw[k] <= snap.bess_discharge_power_max_kw + TOL
        lhs = (res.p_grid_kw[k] + res.pv_used_kw[k] + res.wind_used_kw[k]
               + res.discharge_kw[k])
        rhs = res.p_idc_kw[k] + res.charge_kw[k] + res.curtail_kw[k]
        assert lhs == pytest.approx(rhs, abs=1e-6)
    for k in range(res.horizon_steps + 1):
        assert snap.soc_min_kwh - TOL <= res.soc_kwh[k] <= snap.soc_max_kwh + TOL


def test_diagnostic_excludes_task_variables():
    """诊断结果不得依赖任务：改变任务集不改变 base-only 诊断。"""
    env = _env(horizon=6, access_limit_kw=10.0, soc_init=0.5)
    before = diagnose_base_feasibility(build_snapshot(env))
    env.tasks = []
    after = diagnose_base_feasibility(build_snapshot(env))
    assert before.feasible == after.feasible
    assert before.shortfall_kwh == pytest.approx(after.shortfall_kwh, abs=1e-9)


# --- 6. 回归 ---

def test_lp_failure_result_still_zero_allocation():
    snap = build_snapshot(_env(horizon=24, access_limit_kw=10.0, soc_init=0.5))
    res = solve_time_indexed_lp(snap)
    np.testing.assert_allclose(res.allocation, 0.0)
    assert res.electricity_cost_sgd == 0.0
    assert res.power_approximation_used is True

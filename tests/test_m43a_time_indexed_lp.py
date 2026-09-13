"""M4.3a 测试：H 步时间索引 LP 规划核心（work/kW/kWh/SGD 严格分离）。"""

import numpy as np
import pytest

from contracts.models import SystemSnapshot
from contracts.validators import validate_snapshot
from envs.idc_price_env import IDCPriceEnv20D
from idc_model.task import Task
from planning.model import (
    BUSINESS_SHORTFALL_PENALTY_SGD_PER_WORK,
    DEADLINE_SHORTFALL_PENALTY_SGD_PER_WORK,
    DEGRADATION_COST_UNIT,
    ELECTRICITY_COST_UNIT,
    FAILURE_BASE_SHORTAGE,
    FAILURE_DEADLINE_SHORTFALL,
    FAILURE_NONE,
    solve_time_indexed_lp,
)
from planning.snapshot_adapter import build_snapshot

HORIZON = 24
CUTOFF = 4
N_GROUP = 20
DT = 1.0
TOL = 1e-6


def _env(access_limit_kw: float = 1000.0, soc_init: float = 0.5, t: int = 0) -> IDCPriceEnv20D:
    h = HORIZON
    env = IDCPriceEnv20D(
        horizon=h, forecast_cutoff=CUTOFF, access_limit_kw=access_limit_kw,
        bess_soc_init=soc_init,
        price_t=0.12 + 0.01 * np.arange(h),
        pv_t=0.5 + 0.01 * np.arange(h),
        wt_t=0.4 + 0.01 * np.arange(h),
        carbon_factor_t=0.5 + 0.01 * np.arange(h),
        T_amb=25.0 + 0.1 * np.arange(h),
    )
    env.reset(seed=0)
    action = np.concatenate(
        [np.full(N_GROUP, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)]
    )
    for _ in range(t):
        env.step(action)
    return env


def _with_task(env, workload: float, duration: int, deadline: int, tid: int = 999):
    task = Task(
        task_id=tid, profile_key="k", name="t", arrival_time=0, duration=duration,
        load_profile=np.array([0.5]), workload=workload, deadline=deadline,
        priority=1.0, interruptible=True, parallelizable=False,
    )
    task.status = "waiting"
    env.tasks = [task]
    env.Q_t = workload
    return env


# --- 1. 单位分离 ---

def test_cost_units_declared_and_distinct():
    assert ELECTRICITY_COST_UNIT == "SGD"
    assert DEGRADATION_COST_UNIT == "SGD"
    assert "work-unit" in BUSINESS_SHORTFALL_PENALTY_SGD_PER_WORK[1]
    assert "work-unit" in DEADLINE_SHORTFALL_PENALTY_SGD_PER_WORK[1]
    assert DEADLINE_SHORTFALL_PENALTY_SGD_PER_WORK[0] > 0.0
    res = solve_time_indexed_lp(build_snapshot(_env()))
    for name in ("electricity_cost_sgd", "degradation_cost_sgd",
                 "business_shortfall_cost_sgd", "deadline_shortfall_cost_sgd"):
        assert getattr(res, name) >= 0.0
    assert res.total_objective_sgd == pytest.approx(
        res.electricity_cost_sgd + res.degradation_cost_sgd
        + res.business_shortfall_cost_sgd + res.deadline_shortfall_cost_sgd,
        abs=1e-6,
    )


def test_electricity_cost_recomputed_from_kw_and_price():
    snap = build_snapshot(_env())
    res = solve_time_indexed_lp(snap)
    expected = sum(
        p * DT * price
        for p, price in zip(res.p_grid_kw, snap.planning_forecast.price, strict=True)
    )
    assert res.electricity_cost_sgd == pytest.approx(expected, abs=1e-6)


def test_degradation_cost_recomputed_from_throughput():
    snap = build_snapshot(_env())
    res = solve_time_indexed_lp(snap)
    throughput = sum(c + d for c, d in zip(res.charge_kw, res.discharge_kw, strict=True))
    expected = throughput * DT * snap.bess_degradation_cost_per_kwh
    assert res.degradation_cost_sgd == pytest.approx(expected, abs=1e-6)


def test_power_equation_uses_only_planning_approximation():
    snap = build_snapshot(_env())
    res = solve_time_indexed_lp(snap)
    assert res.power_approximation_used is True
    pf = snap.planning_forecast
    for k in range(res.horizon_steps):
        work_k = sum(
            res.allocation[i][g][k]
            for i in range(len(snap.tasks))
            for g in range(N_GROUP)
        )
        expected = pf.base_idc_power[k] + sum(
            snap.group_power_coeff_kw_per_work[g]
            * sum(res.allocation[i][g][k] for i in range(len(snap.tasks)))
            for g in range(N_GROUP)
        )
        assert res.p_idc_kw[k] == pytest.approx(expected, abs=1e-6)
        assert work_k >= 0.0


# --- 2. 约束不超限 ---

def test_per_task_rate_and_group_capacity_and_remaining():
    env = _with_task(_env(), workload=500.0, duration=100, deadline=100)  # max_rate=5
    snap = build_snapshot(env)
    res = solve_time_indexed_lp(snap)
    n_task, n_group, H = len(snap.tasks), N_GROUP, res.horizon_steps
    for i, task in enumerate(snap.tasks):
        total = 0.0
        for k in range(H):
            per_step = sum(res.allocation[i][g][k] for g in range(n_group))
            assert per_step <= task.max_rate_work_per_step + TOL
            total += per_step
        assert total <= task.remaining_work + TOL
    for g in range(n_group):
        for k in range(H):
            col = sum(res.allocation[i][g][k] for i in range(n_task))
            assert col <= snap.group_work_capacity[g] + TOL


def test_energy_and_storage_constraints_hold():
    snap = build_snapshot(_env())
    res = solve_time_indexed_lp(snap)
    pf = snap.planning_forecast
    for k in range(res.horizon_steps):
        assert -TOL <= res.p_grid_kw[k] <= snap.access_limit_kw + TOL
        assert -TOL <= res.pv_used_kw[k] <= pf.pv[k] + TOL
        assert -TOL <= res.wind_used_kw[k] <= pf.wind[k] + TOL
        assert res.charge_kw[k] <= snap.bess_charge_power_max_kw + TOL
        assert res.discharge_kw[k] <= snap.bess_discharge_power_max_kw + TOL
        # 修正后形式（与 env M3.5 一致）：弃电是未被消费的可再生，不入需求侧
        lhs = (res.p_grid_kw[k] + res.pv_used_kw[k] + res.wind_used_kw[k]
               + res.discharge_kw[k])
        rhs = res.p_idc_kw[k] + res.charge_kw[k]
        assert lhs == pytest.approx(rhs, abs=1e-6)
        assert res.curtail_kw[k] == pytest.approx(
            pf.pv[k] - res.pv_used_kw[k] + pf.wind[k] - res.wind_used_kw[k], abs=1e-6
        )
    for k in range(res.horizon_steps + 1):
        assert snap.soc_min_kwh - TOL <= res.soc_kwh[k] <= snap.soc_max_kwh + TOL
    for k in range(res.horizon_steps):
        expected = (res.soc_kwh[k]
                    + snap.bess_charge_efficiency * res.charge_kw[k] * DT
                    - res.discharge_kw[k] / snap.bess_discharge_efficiency * DT)
        assert res.soc_kwh[k + 1] == pytest.approx(expected, abs=1e-6)


def test_max_constraint_residuals_reported():
    res = solve_time_indexed_lp(build_snapshot(_env()))
    assert res.max_constraint_residual >= 0.0
    assert res.max_constraint_residual < 1e-6
    assert isinstance(res.residuals_by_constraint, dict)
    assert any(k.startswith("energy_balance") for k in res.residuals_by_constraint)
    assert any(k.startswith("soc_dynamics") for k in res.residuals_by_constraint)
    assert all(v >= 0.0 for v in res.residuals_by_constraint.values())


# --- 3. 失败分类 ---

def test_base_shortage_reported_not_fabricated():
    # access=1kW、SOC 触底、无可再生 → 基础负载无法满足 → 不可行
    env = _env(access_limit_kw=1.0, soc_init=0.1)
    env.pv_t[:] = 0.0
    env.wt_t[:] = 0.0
    snap = build_snapshot(env)
    res = solve_time_indexed_lp(snap)
    assert res.failure_class == FAILURE_BASE_SHORTAGE
    assert res.solver_status != "optimal"
    np.testing.assert_allclose(res.allocation, 0.0)


def test_deadline_shortfall_reported():
    # 期限极紧且速率受限 → 必然留下 deadline slack
    env = _with_task(_env(), workload=100.0, duration=100, deadline=1)  # max_rate=1
    snap = build_snapshot(env)
    res = solve_time_indexed_lp(snap)
    assert res.deadline_shortfall_work > 0.0
    assert res.failure_class == FAILURE_DEADLINE_SHORTFALL


def test_feasible_case_classified_none():
    env = _with_task(_env(), workload=10.0, duration=10, deadline=HORIZON)  # max_rate=1
    res = solve_time_indexed_lp(build_snapshot(env))
    assert res.failure_class == FAILURE_NONE
    assert res.solver_status == "optimal"


# --- 4. 充放电松弛标记 ---

def test_storage_relaxation_flag_present():
    res = solve_time_indexed_lp(build_snapshot(_env()))
    assert isinstance(res.storage_relaxation_active, bool)
    if res.storage_relaxation_active:
        # LP 允许同时充放；标记为真时结果不得声称互斥已满足
        assert any(
            c > TOL and d > TOL
            for c, d in zip(res.charge_kw, res.discharge_kw, strict=True)
        )


# --- 5. 窗口外真值不影响求解 ---

@pytest.mark.leakage
def test_out_of_window_truth_does_not_change_solution():
    env = _env(t=1)
    before = solve_time_indexed_lp(build_snapshot(env))
    for idx in range(1 + CUTOFF, HORIZON):
        env.price_t[idx] = 9999.0
        env.pv_t[idx] = 9999.0
        env.wt_t[idx] = 9999.0
        env.T_amb[idx] = 9999.0
        env.carbon_factor_t[idx] = 9999.0
        env.true_task_arrival_profile[idx] = 9999.0
    after = solve_time_indexed_lp(build_snapshot(env))
    np.testing.assert_allclose(before.allocation, after.allocation, atol=1e-9)
    assert before.electricity_cost_sgd == pytest.approx(after.electricity_cost_sgd, abs=1e-9)


# --- 6. 规模与耗时可审计 ---

def test_result_reports_scale_and_time():
    res = solve_time_indexed_lp(build_snapshot(_env()))
    assert res.n_variables > 0
    assert res.n_constraints > 0
    assert res.solve_time_s >= 0.0
    assert res.horizon_steps == build_snapshot(_env()).planning_horizon_steps


def test_snapshot_validator_still_passes():
    validate_snapshot(build_snapshot(_env()))
    assert isinstance(build_snapshot(_env()), SystemSnapshot)

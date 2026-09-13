"""M4.3b 测试：H 步 MIP 后端（逐步强制储能充/放电互斥）。"""

import numpy as np
import pytest

from envs.idc_price_env import IDCPriceEnv20D
from idc_model.task import Task
from planning.model import (
    FAILURE_BASE_SHORTAGE,
    FAILURE_NONE,
    FAILURE_SOLVER_FAILURE,
    FAILURE_TIMEOUT,
    solve_time_indexed_lp,
    solve_time_indexed_mip,
)
from planning.snapshot_adapter import build_snapshot

HORIZON = 24
CUTOFF = 4
N_GROUP = 20
TOL = 1e-6
MUTUAL_EXCLUSION_TOL = 1e-6


def _env(
    access_limit_kw: float = 1000.0,
    soc_init: float = 0.5,
    horizon: int = HORIZON,
    pv: float = 0.0,
    wind: float = 0.0,
    price_high_at: tuple[int, ...] = (),
) -> IDCPriceEnv20D:
    h = horizon
    price = 0.10 + 0.005 * np.arange(h)
    for idx in price_high_at:
        price[idx] = 0.60
    env = IDCPriceEnv20D(
        horizon=h, forecast_cutoff=CUTOFF, access_limit_kw=access_limit_kw,
        bess_soc_init=soc_init,
        price_t=price,
        pv_t=np.full(h, pv),
        wt_t=np.full(h, wind),
        carbon_factor_t=np.full(h, 0.5),
        T_amb=np.full(h, 25.0),
    )
    env.reset(seed=0)
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


# --- 1. 互斥本身 ---

def test_mip_enforces_mutual_exclusion_every_step():
    # 储能压力：SOC 中位、价格前段低后段高 → 诱导充放
    snap = build_snapshot(_env(access_limit_kw=25.0, soc_init=0.5, price_high_at=(12, 13, 14)))
    res = solve_time_indexed_mip(snap)
    assert res.backend == "mip"
    assert res.solver_status == "optimal"
    for k in range(res.horizon_steps):
        assert res.charge_kw[k] * res.discharge_kw[k] == pytest.approx(
            0.0, abs=MUTUAL_EXCLUSION_TOL
        )


def test_mip_mutual_exclusion_under_storage_pressure():
    """储能压力场景：MIP 逐步满足 charge×discharge≈0（无条件断言，不做条件式覆盖）。"""
    snap = build_snapshot(
        _env(access_limit_kw=25.0, soc_init=0.5, price_high_at=(10, 11, 12, 13))
    )
    mip = solve_time_indexed_mip(snap)
    assert mip.solver_status == "optimal"
    assert mip.horizon_steps > 1
    for k in range(mip.horizon_steps):
        assert mip.charge_kw[k] * mip.discharge_kw[k] == pytest.approx(0.0, abs=TOL)
    # 该场景确有储能活动（否则断言空洞）
    assert max(max(mip.charge_kw), max(mip.discharge_kw)) > TOL


def test_lp_is_non_executable_relaxation_by_structure():
    """LP 的松弛是**结构性**的：不含任何互斥约束行；MIP 含 2H 条。

    说明：严格正退化成本 + 有损往返下，LP 最优解不会真的同步充放
    （同时充放严格劣于同时置零），故此处**不伪造 coflow**，而以约束结构取证。
    """
    snap = build_snapshot(_env(access_limit_kw=25.0, soc_init=0.5))
    lp = solve_time_indexed_lp(snap)
    mip = solve_time_indexed_mip(snap)
    assert lp.backend == "lp"
    assert lp.n_integer_variables == 0
    assert mip.backend == "mip"
    assert mip.n_integer_variables == mip.horizon_steps

    lp_excl = [k for k in lp.residuals_by_constraint if k.startswith("storage_excl")]
    mip_excl = [k for k in mip.residuals_by_constraint if k.startswith("storage_excl")]
    assert lp_excl == []                     # LP 无互斥约束 → 松弛且不可执行
    assert len(mip_excl) == 2 * mip.horizon_steps  # MIP 每步 2 条互斥约束


# --- 2. 物理与结构约束 ---

def test_mip_energy_balance_and_curtailment_definition():
    snap = build_snapshot(_env(access_limit_kw=25.0, soc_init=0.5))
    pf = snap.planning_forecast
    res = solve_time_indexed_mip(snap)
    for k in range(res.horizon_steps):
        lhs = (res.p_grid_kw[k] + res.pv_used_kw[k] + res.wind_used_kw[k]
               + res.discharge_kw[k])
        rhs = res.p_idc_kw[k] + res.charge_kw[k]
        assert lhs == pytest.approx(rhs, abs=1e-6)
        assert res.curtail_kw[k] == pytest.approx(
            pf.pv[k] - res.pv_used_kw[k] + pf.wind[k] - res.wind_used_kw[k], abs=1e-6
        )
        assert -TOL <= res.p_grid_kw[k] <= snap.access_limit_kw + TOL
        assert -TOL <= res.pv_used_kw[k] <= pf.pv[k] + TOL
        assert -TOL <= res.wind_used_kw[k] <= pf.wind[k] + TOL
        assert res.charge_kw[k] <= snap.bess_charge_power_max_kw + TOL
        assert res.discharge_kw[k] <= snap.bess_discharge_power_max_kw + TOL


def test_mip_soc_dynamics_and_bounds():
    snap = build_snapshot(_env(access_limit_kw=25.0, soc_init=0.5))
    res = solve_time_indexed_mip(snap)
    dt = snap.delta_t_hours
    assert res.soc_kwh[0] == pytest.approx(snap.soc_kwh, abs=1e-6)
    for k in range(res.horizon_steps):
        expected = (res.soc_kwh[k]
                    + snap.bess_charge_efficiency * res.charge_kw[k] * dt
                    - res.discharge_kw[k] / snap.bess_discharge_efficiency * dt)
        assert res.soc_kwh[k + 1] == pytest.approx(expected, abs=1e-6)
    for k in range(res.horizon_steps + 1):
        assert snap.soc_min_kwh - TOL <= res.soc_kwh[k] <= snap.soc_max_kwh + TOL


def test_mip_task_group_and_rate_constraints():
    env = _with_task(_env(access_limit_kw=1000.0), workload=500.0, duration=100, deadline=100)
    snap = build_snapshot(env)
    res = solve_time_indexed_mip(snap)
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
    assert res.max_constraint_residual < 1e-6


def test_mip_deadline_slack_reported():
    env = _with_task(_env(), workload=100.0, duration=100, deadline=1)  # max_rate=1
    snap = build_snapshot(env)
    res = solve_time_indexed_mip(snap)
    assert res.deadline_shortfall_work > 0.0


# --- 3. LP 下界性质 ---

def test_mip_objective_not_better_than_lp_relaxation():
    env = _with_task(_env(access_limit_kw=25.0, soc_init=0.5), 200.0, 100, 100)
    snap = build_snapshot(env)
    lp = solve_time_indexed_lp(snap)
    mip = solve_time_indexed_mip(snap)
    assert lp.solver_status == "optimal" and mip.solver_status == "optimal"
    # MIP 是 LP 的受限版本 → 其最小目标值不得低于 LP 松弛下界
    assert mip.total_objective_sgd >= lp.total_objective_sgd - 1e-6


def test_integer_variable_counts():
    snap = build_snapshot(_env(access_limit_kw=25.0))
    lp = solve_time_indexed_lp(snap)
    mip = solve_time_indexed_mip(snap)
    assert lp.n_integer_variables == 0
    assert mip.n_integer_variables == mip.horizon_steps
    assert lp.backend == "lp" and mip.backend == "mip"
    assert mip.n_variables == lp.n_variables + mip.horizon_steps


# --- 4. 边界情形 ---

def test_no_storage_pressure_pure_grid():
    snap = build_snapshot(_env(access_limit_kw=1000.0, soc_init=0.5))
    res = solve_time_indexed_mip(snap)
    assert res.solver_status == "optimal"
    for k in range(res.horizon_steps):
        assert res.charge_kw[k] * res.discharge_kw[k] == pytest.approx(0.0, abs=TOL)


def test_renewables_sufficient_case():
    base_kw = build_snapshot(_env()).planning_forecast.base_idc_power[0]
    snap = build_snapshot(_env(horizon=4, access_limit_kw=1.0, soc_init=0.1, pv=base_kw + 5.0))
    res = solve_time_indexed_mip(snap)
    assert res.solver_status == "optimal"
    assert res.failure_class != FAILURE_BASE_SHORTAGE
    # 可再生充足时基础负载可服务；任务可以（也应当）被执行，此处不断言零分配


def test_infeasible_returns_structured_failure_no_fabricated_allocation():
    snap = build_snapshot(_env(access_limit_kw=1.0, soc_init=0.1))
    res = solve_time_indexed_mip(snap)
    assert res.solver_status == "infeasible"
    assert res.failure_class == FAILURE_BASE_SHORTAGE
    np.testing.assert_allclose(res.allocation, 0.0)
    assert res.electricity_cost_sgd == 0.0
    assert res.total_objective_sgd == 0.0


# --- 5. 窗口外真值不影响 MIP ---

@pytest.mark.leakage
def test_out_of_window_truth_does_not_change_mip():
    env = _env(access_limit_kw=25.0, soc_init=0.5)
    action = np.concatenate(
        [np.full(N_GROUP, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)]
    )
    env.step(action)
    before = solve_time_indexed_mip(build_snapshot(env))
    for idx in range(env.current_step + CUTOFF, HORIZON):
        env.price_t[idx] = 9999.0
        env.pv_t[idx] = 9999.0
        env.wt_t[idx] = 9999.0
        env.T_amb[idx] = 9999.0
        env.carbon_factor_t[idx] = 9999.0
        env.true_task_arrival_profile[idx] = 9999.0
    after = solve_time_indexed_mip(build_snapshot(env))
    np.testing.assert_allclose(before.allocation, after.allocation, atol=1e-9)
    assert before.total_objective_sgd == pytest.approx(after.total_objective_sgd, abs=1e-9)


# --- 6. 回归 ---

def test_mip_feasible_case_classified_none():
    env = _with_task(_env(access_limit_kw=1000.0), 10.0, 10, HORIZON)
    res = solve_time_indexed_mip(build_snapshot(env))
    assert res.failure_class == FAILURE_NONE
    assert res.solver_status == "optimal"


# --- 7. 失败路径语义（求解器边界单元测试，不依赖真实超时） ---

def _fake_milp(status: int, x=None):
    """构造一个假的 scipy.optimize.milp 返回值（仅用于边界测试）。"""
    class _Res:
        pass

    r = _Res()
    r.status = status
    r.success = status == 0
    r.message = f"fake status {status}"
    r.fun = 0.0
    r.x = x
    return r


def test_mip_time_limit_classified_as_timeout(monkeypatch):
    snap = build_snapshot(_env(access_limit_kw=25.0, soc_init=0.5))
    monkeypatch.setattr(
        "scipy.optimize.milp", lambda **kw: _fake_milp(1)
    )
    res = solve_time_indexed_mip(snap)
    assert res.solver_status == "time_limit"
    assert res.failure_class == FAILURE_TIMEOUT          # 不得混入 solver_failure
    assert res.backend == "mip"
    assert res.n_integer_variables == res.horizon_steps
    # 立即零执行返回，不伪造任何执行结果
    np.testing.assert_allclose(res.allocation, 0.0)
    assert res.electricity_cost_sgd == 0.0
    assert res.degradation_cost_sgd == 0.0
    assert res.total_objective_sgd == 0.0
    assert all(v == 0.0 for v in res.p_grid_kw)
    assert all(v == 0.0 for v in res.charge_kw)
    assert all(v == 0.0 for v in res.discharge_kw)
    assert res.business_shortfall_work == pytest.approx(
        sum(t.remaining_work for t in snap.tasks), abs=1e-9
    )


def test_mip_timeout_does_not_run_second_solve(monkeypatch):
    """timeout 后不得调用 diagnose_base_feasibility（不得再启动第二次求解）。"""
    import planning.model as mod

    called = {"n": 0}
    real = mod.diagnose_base_feasibility

    def _spy(snapshot):
        called["n"] += 1
        return real(snapshot)

    monkeypatch.setattr(mod, "diagnose_base_feasibility", _spy)
    monkeypatch.setattr("scipy.optimize.milp", lambda **kw: _fake_milp(1))
    res = solve_time_indexed_mip(build_snapshot(_env(access_limit_kw=25.0)))
    assert res.failure_class == FAILURE_TIMEOUT
    assert called["n"] == 0


def test_mip_unknown_status_classified_as_solver_failure(monkeypatch):
    snap = build_snapshot(_env(access_limit_kw=25.0, soc_init=0.5))
    for status in (3, 4):
        monkeypatch.setattr("scipy.optimize.milp", lambda *a, s=status, **kw: _fake_milp(s))
        res = solve_time_indexed_mip(snap)
        assert res.solver_status == "solver_failure"
        assert res.failure_class == FAILURE_SOLVER_FAILURE
        np.testing.assert_allclose(res.allocation, 0.0)
        assert res.electricity_cost_sgd == 0.0

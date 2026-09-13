"""M3.7c 测试：独立 brownout 分解（base/task 断供独立计算）。"""

import numpy as np
import pytest

from envs.idc_price_env import IDCPriceEnv20D, decompose_supply
from idc_model.task import Task


# --- 纯分解测试 ---

def test_decompose_base_insufficient():
    d = decompose_supply(base_demand_kW=15.0, total_demand_kW=15.0, budget_kW=9.0)
    assert d["base_served_kW"] == pytest.approx(9.0)
    assert d["task_incremental_demand_kW"] == pytest.approx(0.0)
    assert d["task_served_kW"] == pytest.approx(0.0)
    assert d["unserved_base_load_kW"] == pytest.approx(6.0)
    assert d["unserved_task_power_kW"] == pytest.approx(0.0)


def test_decompose_task_insufficient():
    d = decompose_supply(base_demand_kW=10.0, total_demand_kW=20.0, budget_kW=15.0)
    assert d["base_served_kW"] == pytest.approx(10.0)
    assert d["task_incremental_demand_kW"] == pytest.approx(10.0)
    assert d["task_served_kW"] == pytest.approx(5.0)
    assert d["unserved_base_load_kW"] == pytest.approx(0.0)
    assert d["unserved_task_power_kW"] == pytest.approx(5.0)


def test_decompose_full_supply():
    d = decompose_supply(base_demand_kW=10.0, total_demand_kW=15.0, budget_kW=20.0)
    assert d["base_served_kW"] == pytest.approx(10.0)
    assert d["task_served_kW"] == pytest.approx(5.0)
    assert d["unserved_base_load_kW"] == pytest.approx(0.0)
    assert d["unserved_task_power_kW"] == pytest.approx(0.0)


def test_decompose_invariant():
    for base, total, budget in [(15, 15, 9), (10, 20, 15), (10, 15, 20), (5, 30, 12)]:
        d = decompose_supply(base, total, budget)
        demand = total
        served = d["idc_served_kW"]
        unserved = d["unserved_base_load_kW"] + d["unserved_task_power_kW"]
        assert demand - served == pytest.approx(unserved, abs=1e-9)


# --- 环境集成测试 ---

def _env(access_limit_kw=18.0, soc_init=0.5) -> IDCPriceEnv20D:
    return IDCPriceEnv20D(
        access_limit_kw=access_limit_kw,
        wt_t=np.zeros(24), pv_t=np.zeros(24), bess_soc_init=soc_init,
    )


def _full_compute(env) -> np.ndarray:
    return np.concatenate([np.ones(20, dtype=np.float32), np.array([0.0], dtype=np.float32)])


def test_env_normal_low_access_task_served():
    env = _env(access_limit_kw=15.0, soc_init=0.1)
    env.reset(seed=0)
    env.tasks = [Task(task_id=1, profile_key="k", name="t", arrival_time=0, duration=1,
                      load_profile=np.array([0.5]), workload=1000.0, deadline=10,
                      priority=1.0, interruptible=True, parallelizable=False)]
    env.Q_t = 1000.0
    _, _, _, _, info = env.step(_full_compute(env))
    assert info["unserved_task_power_kW"] == pytest.approx(0.0, abs=1e-3)
    assert info["P_grid_kW"] <= 15.0 + 1e-6


def test_env_base_infeasible():
    env = _env(access_limit_kw=9.0, soc_init=0.1)
    env.reset(seed=0)
    env.tasks = [Task(task_id=1, profile_key="k", name="t", arrival_time=0, duration=1,
                      load_profile=np.array([0.5]), workload=100.0, deadline=10,
                      priority=1.0, interruptible=True, parallelizable=False)]
    env.Q_t = 100.0
    _, _, _, _, info = env.step(_full_compute(env))
    assert info["completed_work"] == pytest.approx(0.0, abs=1e-6)
    assert info["unserved_base_load_kW"] > 0
    assert info["unserved_task_power_kW"] == pytest.approx(0.0, abs=1e-6)
    assert info["P_base_served_kW"] == pytest.approx(info["P_IDC_served_kW"], abs=1e-3)


def test_env_energy_balance_and_grid_limit():
    env = _env(access_limit_kw=16.0, soc_init=0.5)
    env.reset(seed=0)
    a = np.concatenate([np.full(20, 0.7, dtype=np.float32), np.array([0.3], dtype=np.float32)])
    for _ in range(5):
        _, _, _, _, info = env.step(a)
        lhs = info["P_grid_kW"] + info["pv_available_kW"] + info["wind_available_kW"] + info["bess_discharge_power_kW"]
        rhs = info["P_IDC_served_kW"] + info["bess_charge_power_kW"] + info["pv_curtail_kW"] + info["wind_curtail_kW"]
        assert lhs == pytest.approx(rhs, rel=1e-5, abs=1e-5)
        assert info["P_grid_kW"] <= 16.0 + 1e-6


def test_env_m32_regression_and_max_rate_zero_curtailment():
    env = _env(access_limit_kw=1000.0, soc_init=0.5)
    env.reset(seed=0)
    env.tasks = [Task(task_id=1, profile_key="k", name="t", arrival_time=0, duration=2,
                      load_profile=np.array([0.5]), workload=10.0, deadline=10,
                      priority=1.0, interruptible=True, parallelizable=False)]
    env.Q_t = 10.0
    _, _, _, _, info = env.step(_full_compute(env))
    assert info["completed_work"] <= 5.0 + 1e-6  # max_rate=5
    assert info["access_curtailment_work"] == pytest.approx(0.0, abs=1e-6)
    assert info["completed_work"] == pytest.approx(
        float(np.asarray(info["completed_work_by_group"]).sum()), abs=1e-6
    )

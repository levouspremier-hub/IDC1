"""M3.7b 测试：brownout 与能量削减账务（demand/served/curtailment 分离）。"""

import numpy as np
import pytest

from envs.idc_price_env import IDCPriceEnv20D
from idc_model.task import Task


def _env(access_limit_kw=18.0, wt=None, pv=None, soc_init=0.5) -> IDCPriceEnv20D:
    return IDCPriceEnv20D(
        access_limit_kw=access_limit_kw,
        wt_t=wt if wt is not None else np.zeros(24),
        pv_t=pv if pv is not None else np.zeros(24),
        bess_soc_init=soc_init,
    )


def _task(tid, workload, duration, deadline=10) -> Task:
    return Task(
        task_id=tid, profile_key="k", name="t", arrival_time=0, duration=duration,
        load_profile=np.array([0.5]), workload=workload, deadline=deadline,
        priority=1.0, interruptible=True, parallelizable=False,
    )


def _full_compute(env) -> np.ndarray:
    return np.concatenate([np.ones(20, dtype=np.float32), np.array([0.0], dtype=np.float32)])


def _energy_balance(info) -> None:
    lhs = (
        info["P_grid_kW"] + info["pv_available_kW"] + info["wind_available_kW"]
        + info["bess_discharge_power_kW"]
    )
    rhs = (
        info["P_IDC_served_kW"] + info["bess_charge_power_kW"]
        + info["pv_curtail_kW"] + info["wind_curtail_kW"]
    )
    assert lhs == pytest.approx(rhs, rel=1e-5, abs=1e-5)


def test_access_curtailment_zero_when_max_rate_limited():
    # 接入上限极大，任务受 max_rate 限制 → 削减不是接入导致
    env = _env(access_limit_kw=1000.0, soc_init=0.5)
    env.reset(seed=0)
    env.tasks = [_task(1, workload=10.0, duration=10)]  # max_rate=1
    env.Q_t = 10.0
    _, _, _, _, info = env.step(_full_compute(env))
    assert info["access_curtailment_work"] == pytest.approx(0.0, abs=1e-6)


def test_access_curtailment_positive_when_low_access():
    env = _env(access_limit_kw=15.0, soc_init=0.1)
    env.reset(seed=0)
    env.tasks = [_task(1, workload=1000.0, duration=1)]  # 无速率限制
    env.Q_t = 1000.0
    _, _, _, _, info = env.step(_full_compute(env))
    assert info["access_curtailment_work"] > 0


def test_access_curtailment_zero_when_empty_queue():
    env = _env(access_limit_kw=15.0, soc_init=0.1)
    env.reset(seed=0)
    env.tasks = []
    env.Q_t = 0.0
    _, _, _, _, info = env.step(_full_compute(env))
    assert info["access_curtailment_work"] == pytest.approx(0.0, abs=1e-6)


def test_base_load_infeasible_fields():
    env = _env(access_limit_kw=9.0, soc_init=0.1)
    env.reset(seed=0)
    env.tasks = [_task(1, workload=100.0, duration=1)]
    env.Q_t = 100.0
    _, _, _, _, info = env.step(_full_compute(env))
    assert info["completed_work"] == pytest.approx(0.0, abs=1e-6)
    assert info["P_IDC_demand_kW"] > info["P_IDC_served_kW"]
    assert info["unserved_base_load_kW"] == pytest.approx(
        info["P_IDC_demand_kW"] - info["P_IDC_served_kW"], abs=1e-3
    )
    assert info["unserved_task_power_kW"] == pytest.approx(0.0, abs=1e-6)


def test_energy_balance_renewable_charge_discharge():
    # 可再生充足
    env = _env(access_limit_kw=18.0, wt=np.full(24, 40.0), pv=np.full(24, 40.0), soc_init=0.5)
    env.reset(seed=0)
    _, _, _, _, info = env.step(_full_compute(env))
    _energy_balance(info)
    assert info["P_grid_kW"] <= 18.0 + 1e-6
    # 充电
    env2 = _env(access_limit_kw=18.0, soc_init=0.3)
    env2.reset(seed=0)
    a_charge = np.concatenate([np.ones(20, dtype=np.float32), np.array([-0.5], dtype=np.float32)])
    _, _, _, _, info2 = env2.step(a_charge)
    _energy_balance(info2)
    assert info2["P_grid_kW"] <= 18.0 + 1e-6
    # 放电
    env3 = _env(access_limit_kw=18.0, soc_init=0.7)
    env3.reset(seed=0)
    a_discharge = np.concatenate([np.ones(20, dtype=np.float32), np.array([0.5], dtype=np.float32)])
    _, _, _, _, info3 = env3.step(a_discharge)
    _energy_balance(info3)
    assert info3["P_grid_kW"] <= 18.0 + 1e-6


def test_m32_regression_rate_and_a_matrix():
    env = _env(access_limit_kw=18.0, soc_init=0.5)
    env.reset(seed=0)
    env.tasks = [_task(1, workload=10.0, duration=2)]  # max_rate=5
    env.Q_t = 10.0
    _, _, _, _, info = env.step(_full_compute(env))
    assert info["completed_work"] <= 5.0 + 1e-6  # 速率约束
    assert info["completed_work"] == pytest.approx(
        float(np.asarray(info["completed_work_by_group"]).sum()), abs=1e-6
    )

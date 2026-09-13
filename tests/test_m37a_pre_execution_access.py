"""M3.7a 测试：接入容量约束前置到任务执行前（物理可行性投影）。

验证任务不再「已完成但电力被记为 unserved」；业务缺口/基础负载缺口进入执行前计算。
"""

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


def _full_compute(env) -> np.ndarray:
    return np.concatenate([np.ones(20, dtype=np.float32), np.array([0.0], dtype=np.float32)])


def _base_load_power_kw(env) -> float:
    lb = np.clip(env.base_load, 0.0, 1.0).reshape(1, -1)
    p, *_ = env.model.calc_pue_and_total_power(lb, np.array([env.T_amb[0]]))
    return float(p[0]) / 1000.0


def _energy_balance(env, info) -> None:
    lhs = (
        info["P_grid_kW"]
        + info["pv_available_kW"]
        + info["wind_available_kW"]
        + info["bess_discharge_power_kW"]
    )
    rhs = (
        info["P_IDC_kW"]
        + info["bess_charge_power_kW"]
        + info["pv_curtail_kW"]
        + info["wind_curtail_kW"]
    )
    assert lhs == pytest.approx(rhs, rel=1e-5, abs=1e-5)


def test_low_access_no_renewable_no_storage_limits_tasks():
    env = _env(access_limit_kw=15.0, soc_init=0.1)  # 无可再生、SOC 底、接入仅略高于基础负载
    env.reset(seed=0)
    # 注入确定性任务，避免依赖随机生成的默认任务集（reset 不控制任务生成 RNG）
    env.tasks = [
        Task(
            task_id=1, profile_key="k", name="t", arrival_time=0, duration=1,
            load_profile=np.array([0.5]), workload=1000.0, deadline=10,
            priority=1.0, interruptible=True, parallelizable=False,
        )
    ]
    env.tasks[0].status = "waiting"
    env.Q_t = 1000.0
    _, _, _, _, info = env.step(_full_compute(env))
    assert info["P_grid_kW"] <= 15.0 + 1e-6
    assert info["access_curtailment_work"] > 0  # 接入投影导致的任务削减非零
    assert info["unserved_base_load_kW"] == pytest.approx(0.0, abs=1e-6)


def test_base_load_infeasible_zero_task():
    env = _env(access_limit_kw=9.0, soc_init=0.1)  # 接入低于基础负载
    env.reset(seed=0)
    base_power = _base_load_power_kw(env)
    _, _, _, _, info = env.step(_full_compute(env))
    assert info["completed_work"] == pytest.approx(0.0, abs=1e-6)
    assert info["unserved_base_load_kW"] > 0
    # 缺口 ≈ 基础负载功率 - 实际缩放后的接入预算
    assert info["unserved_base_load_kW"] == pytest.approx(
        base_power - env.access_limit_kw, abs=0.5
    )


def test_high_renewables_no_overcompress():
    env = _env(access_limit_kw=18.0, wt=np.full(24, 50.0), pv=np.full(24, 50.0), soc_init=0.5)
    env.reset(seed=0)
    _, _, _, _, info = env.step(_full_compute(env))
    assert info["P_grid_kW"] >= 0.0  # 不为负（无反送电）
    assert info["completed_work"] > 0  # 不错误压缩任务


def test_charge_squeezes_access_no_fake_completion():
    env = _env(access_limit_kw=18.0, soc_init=0.5)
    env.reset(seed=0)
    a = np.concatenate(
        [np.ones(20, dtype=np.float32), np.array([-1.0], dtype=np.float32)]
    )  # 满充电
    _, _, _, _, info = env.step(a)
    assert info["P_grid_kW"] <= 18.0 + 1e-6
    _energy_balance(env, info)


def test_limited_soc_discharge_support():
    env = _env(access_limit_kw=15.0, soc_init=0.6)
    env.reset(seed=0)
    a = np.concatenate([np.ones(20, dtype=np.float32), np.array([0.5], dtype=np.float32)])  # 放电
    _, _, _, _, info = env.step(a)
    assert info["bess_discharge_power_kW"] > 0
    assert info["P_grid_kW"] <= 15.0 + 1e-6
    _energy_balance(env, info)


def test_energy_balance_every_step():
    env = _env(access_limit_kw=16.0, soc_init=0.5)
    env.reset(seed=0)
    a = np.concatenate([np.full(20, 0.7, dtype=np.float32), np.array([0.3], dtype=np.float32)])
    for _ in range(5):
        _, _, _, _, info = env.step(a)
        assert info["P_grid_kW"] <= 16.0 + 1e-6
        _energy_balance(env, info)


def test_m32_regression_holds():
    # 每任务速率 + 逐组容量 + A 总量仍成立（回归）
    env = _env(access_limit_kw=18.0, soc_init=0.5)
    env.reset(seed=0)
    _, _, _, _, info = env.step(_full_compute(env))
    assert info["completed_work"] == pytest.approx(
        float(np.asarray(info["completed_work_by_group"]).sum()), abs=1e-6
    )

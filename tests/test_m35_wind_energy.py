"""M3.5 测试：风电接入无反送电能量平衡，四类守恒。"""

import numpy as np
import pytest

from envs.idc_price_env import IDCPriceEnv20D


def _env(wt=None, pv=None) -> IDCPriceEnv20D:
    e = IDCPriceEnv20D(wt_t=wt, pv_t=pv)
    e.reset(seed=0)
    return e


def _action(env) -> np.ndarray:
    return np.full(env.action_dim, 0.5, dtype=np.float32)


def _check_balance(env):
    info = env.step(_action(env))[4]
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
    assert lhs == pytest.approx(rhs, rel=1e-6, abs=1e-6)


def test_wind_fields_in_info():
    env = _env(wt=np.full(24, 5.0), pv=np.zeros(24))
    info = env.step(_action(env))[4]
    assert "wind_available_kW" in info
    assert "wind_used_kW" in info
    assert "wind_curtail_kW" in info


def test_energy_balance_zero_wind():
    _check_balance(_env(wt=np.zeros(24), pv=np.zeros(24)))


def test_energy_balance_high_wind():
    _check_balance(_env(wt=np.full(24, 20.0), pv=np.zeros(24)))


def test_energy_balance_charge_discharge():
    env = _env(wt=np.full(24, 5.0), pv=np.full(24, 5.0))
    for _ in range(3):
        _check_balance(env)


def test_energy_balance_insufficient_load_curtailment():
    # 可再生充足（超过 IDC 负荷）→ 出现弃电，仍守恒
    env = _env(wt=np.full(24, 50.0), pv=np.full(24, 50.0))
    info = env.step(_action(env))[4]
    assert info["pv_curtail_kW"] + info["wind_curtail_kW"] > 0
    _check_balance(env)

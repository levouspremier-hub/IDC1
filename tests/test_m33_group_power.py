"""M3.3 测试：逐组功耗由完成工作/组能力导出，α 不影响正式负载，能量平衡成立。"""

import numpy as np
import pytest

from envs.idc_price_env import IDCPriceEnv20D


@pytest.fixture()
def env() -> IDCPriceEnv20D:
    return IDCPriceEnv20D()


def test_alpha_does_not_affect_actual_load(env):
    """废除 α：不同 planned_load_reserve_alpha 下实际负载应相同（改造前失败）。"""
    a = np.full(env.action_dim, 0.5, dtype=np.float32)
    env.reset(seed=0)
    env.planned_load_reserve_alpha = 0.0
    _, _, _, _, i1 = env.step(a)
    env.reset(seed=0)
    env.planned_load_reserve_alpha = 0.9
    _, _, _, _, i2 = env.step(a)
    assert i1["actual_total_load_mean"] == pytest.approx(i2["actual_total_load_mean"])


def test_loads_from_group_completion_uses_capacity(env):
    """每组负载 = 完成工作 / 组能力（不同分布同总量产生不同负载）。"""
    a = np.full(env.action_dim, 0.5, dtype=np.float32)
    env.reset(seed=0)
    _, _, _, _, info = env.step(a)
    cg = np.asarray(info["completed_work_by_group"], dtype=np.float64)
    c_server = np.asarray(env.model.C_server, dtype=np.float64)
    expected = cg / np.maximum(c_server, 1e-6)
    actual = env._loads_from_group_completion(cg)
    assert actual == pytest.approx(expected)


def test_energy_balance_holds(env):
    """每步能量平衡：grid + unserved + pv + wind + discharge = IDC + charge + curtail。"""
    a = np.full(env.action_dim, 0.5, dtype=np.float32)
    env.reset(seed=0)
    for _ in range(5):
        _, _, _, _, info = env.step(a)
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

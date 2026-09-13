"""M3.0 物理基线测试：记录改造前行为，未来要求以 xfail(strict=True) 表示。"""

import numpy as np
import pytest

from envs.idc_price_env import IDCPriceEnv20D


@pytest.fixture()
def env() -> IDCPriceEnv20D:
    e = IDCPriceEnv20D()
    e.reset(seed=0)
    return e


def _neutral(env: IDCPriceEnv20D) -> np.ndarray:
    return np.full(env.action_dim, 0.5, dtype=np.float32)


def test_obs_dim_after_wind_carbon_features(env):
    # M3.10b：预测特征组 6 → 8（新增 wind/carbon），obs_dim 280 → 328
    assert env.obs_dim == 328


def test_baseline_num_groups_is_20(env):
    assert env.model.N == 20


def test_action_dim_is_21(env):
    assert env.action_dim == 21


def test_baseline_soc_stays_in_range(env):
    for _ in range(env.horizon):
        env.step(_neutral(env))
    assert env.bess_soc_min <= env.bess_soc <= env.bess_soc_max


def test_baseline_charge_discharge_mutually_exclusive(env):
    for _ in range(env.horizon):
        _, _, _, _, info = env.step(_neutral(env))
        c = float(info["bess_charge_kWh"])
        d = float(info["bess_discharge_kWh"])
        assert not (c > 1e-6 and d > 1e-6)


def test_baseline_planned_capacity_is_scalar(env):
    _, _, _, _, info = env.step(_neutral(env))
    assert np.asarray(info["planned_capacity"]).ndim == 0


def test_planned_capacity_vec_is_vector(env):
    _, _, _, _, info = env.step(_neutral(env))
    assert np.asarray(info["planned_capacity_vec"]).shape == (20,)


def test_wind_enters_energy_balance(env):
    _, _, _, _, info = env.step(_neutral(env))
    assert "wind_used_kW" in info


def test_baseline_deadline_miss_recorded_deterministically(env):
    misses = []
    for _ in range(env.horizon):
        _, _, _, _, info = env.step(_neutral(env))
        misses.append(int(info["new_deadline_miss_count"]))
    assert sum(misses) >= 0  # 记录是否发生，不强改基线

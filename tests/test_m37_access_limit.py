"""M3.7 测试：access_limit_kw 硬上限、缺口记录、与计费阈值分离。"""

import numpy as np
import pytest

from envs.idc_price_env import IDCPriceEnv20D


def _env(**kw) -> IDCPriceEnv20D:
    e = IDCPriceEnv20D(**kw)
    e.reset(seed=0)
    return e


def test_access_limit_enforced_hard_clamp():
    env = _env(access_limit_kw=10.0)
    a = np.full(env.action_dim, 0.9, dtype=np.float32)
    saw_over = False
    for _ in range(5):
        _, _, _, _, info = env.step(a)
        assert info["P_grid_kW"] <= 10.0 + 1e-6
        if info["unserved_load_kW"] > 0:
            saw_over = True
            assert info["P_grid_kW"] == pytest.approx(10.0, abs=1e-6)
    assert saw_over  # 高负载下出现缺口


def test_access_limit_separate_from_billing():
    env = _env(access_limit_kw=10.0, peak_power_threshold_kW=6.0)
    assert env.access_limit_kw != env.peak_power_threshold_kW


def test_access_limit_gap_recorded():
    env = _env(access_limit_kw=5.0)
    a = np.full(env.action_dim, 1.0, dtype=np.float32)
    _, _, _, _, info = env.step(a)
    assert "unserved_load_kW" in info
    assert info["unserved_load_kW"] >= 0.0

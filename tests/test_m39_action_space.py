"""M3.9 测试：21 维动作空间（20 compute + 1 signed storage）。"""

import numpy as np
import pytest

from envs.idc_price_env import IDCPriceEnv20D


def test_action_dim_is_21():
    env = IDCPriceEnv20D()
    assert env.action_dim == 21


def test_storage_action_is_signed():
    env = IDCPriceEnv20D()
    assert env.action_space.low[20] == pytest.approx(-1.0)
    assert env.action_space.high[20] == pytest.approx(1.0)
    env.reset(seed=0)
    a = np.concatenate([np.full(20, 0.5, dtype=np.float32), np.array([-0.5], dtype=np.float32)])
    _, _, _, _, info = env.step(a)
    assert info["bess_raw_action"] == pytest.approx(-0.5)


def test_23_dim_rejected():
    env = IDCPriceEnv20D()
    env.reset(seed=0)
    a23 = np.full(23, 0.5, dtype=np.float32)
    with pytest.raises(ValueError):
        env.step(a23)

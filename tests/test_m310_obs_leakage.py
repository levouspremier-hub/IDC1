"""M3.10 测试：未来信息泄漏回归（篡改未来真值不改变当前 observation）。"""

import numpy as np
import pytest

from envs.idc_price_env import IDCPriceEnv20D


@pytest.mark.leakage
def test_future_truth_mutation_does_not_change_obs():
    env = IDCPriceEnv20D(forecast_cutoff=2)
    env.reset(seed=0)
    obs_before = np.asarray(env._get_obs(), dtype=np.float64)

    # 篡改未来真值（第 10 步及以后，超出 forecast_cutoff=2）
    env.price_t[10] = 9999.0
    env.pv_t[10] = 9999.0
    env.T_amb[10] = 9999.0
    env.true_task_arrival_profile[10] = 9999.0

    obs_after = np.asarray(env._get_obs(), dtype=np.float64)
    assert np.allclose(obs_before, obs_after)


@pytest.mark.leakage
def test_current_truth_still_visible():
    env = IDCPriceEnv20D(forecast_cutoff=2)
    env.reset(seed=0)
    obs_before = np.asarray(env._get_obs(), dtype=np.float64)

    # 篡改当前步真值（应改变 observation，因当前真值可见）
    env.price_t[0] = 9999.0
    obs_after = np.asarray(env._get_obs(), dtype=np.float64)
    assert not np.allclose(obs_before, obs_after)

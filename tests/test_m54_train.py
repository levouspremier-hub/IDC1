"""M5.4 测试：dry rollout 更新一次记录三 value/乘子；策略动作与未来真值无关。"""

import numpy as np
import torch

from envs.idc_price_env import IDCPriceEnv20D
from safe_rl_v2.lagrangian import Lagrangian
from safe_rl_v2.policy import SafePPOPolicy
from safe_rl_v2.train import dry_run_update


def test_dry_run_updates_once():
    env = IDCPriceEnv20D()
    policy = SafePPOPolicy(obs_dim=env.obs_dim)
    lagrangian = Lagrangian({"business": 5.0, "carbon": 3.0})
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    metrics = dry_run_update(env, policy, lagrangian, optimizer, corrector_on=False, steps=6)

    assert "reward_value" in metrics
    assert "business_value" in metrics
    assert "carbon_value" in metrics
    assert "multipliers" in metrics
    assert metrics["corrector_on"] is False


def test_policy_action_independent_of_future_truth():
    env = IDCPriceEnv20D(forecast_cutoff=2)
    env.reset(seed=0)
    policy = SafePPOPolicy(obs_dim=env.obs_dim)
    policy.eval()

    obs_before = np.asarray(env._get_obs(), dtype=np.float32)
    action_before = policy.act_mean(torch.from_numpy(obs_before)).detach().numpy()

    env.price_t[10] = 9999.0
    env.pv_t[10] = 9999.0
    obs_after = np.asarray(env._get_obs(), dtype=np.float32)
    action_after = policy.act_mean(torch.from_numpy(obs_after)).detach().numpy()

    assert np.allclose(action_before, action_after)

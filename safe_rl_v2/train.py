"""M5.4 训练闭环：短 dry rollout 更新一次，记录三套 value/约束/乘子。

不因短训练奖励高而宣称有效（仅验证闭环可更新）。
"""

from __future__ import annotations

import numpy as np
import torch

from safe_rl_v2.lagrangian import Lagrangian
from safe_rl_v2.models import compute_three_value_targets
from safe_rl_v2.policy import SafePPOPolicy


def _clip_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    action[:20] = np.clip(action[:20], 0.0, 1.0)
    action[20] = np.clip(action[20], -1.0, 1.0)
    return action


def dry_run_update(
    env,
    policy: SafePPOPolicy,
    lagrangian: Lagrangian,
    optimizer: torch.optim.Optimizer,
    *,
    corrector_on: bool = False,
    steps: int = 8,
) -> dict:
    """短 dry rollout + 一次更新。返回三套 value、乘子与损耗。"""
    obs, _ = env.reset(seed=0)
    obs = np.asarray(obs, dtype=np.float32)

    obs_list: list[np.ndarray] = []
    logp_list: list[torch.Tensor] = []
    rew_list: list[float] = []
    bc_list: list[float] = []
    cc_list: list[float] = []

    for _ in range(steps):
        obs_t = torch.from_numpy(obs)
        action, log_prob, _ = policy.act(obs_t)
        action_np = _clip_action(action.detach().numpy())
        next_obs, reward, terminated, truncated, info = env.step(action_np)

        business_cost = float(info.get("business_gap", 0.0)) + float(info.get("cost", 0.0))
        carbon_cost = float(info.get("carbon_emission", 0.0))

        obs_list.append(obs)
        logp_list.append(log_prob)
        rew_list.append(float(reward))
        bc_list.append(business_cost)
        cc_list.append(carbon_cost)

        obs = np.asarray(next_obs, dtype=np.float32)
        if terminated or truncated:
            break

    t_len = len(rew_list)
    with torch.no_grad():
        _, final_values = policy.forward(torch.from_numpy(obs))
        v_reward = np.zeros(t_len + 1)
        v_business = np.zeros(t_len + 1)
        v_carbon = np.zeros(t_len + 1)
        v_reward[-1] = float(final_values[0])
        v_business[-1] = float(final_values[1])
        v_carbon[-1] = float(final_values[2])
        # 用 critic 对已收集 obs 重新估值
        obs_t = torch.from_numpy(np.stack(obs_list))
        _, values = policy.forward(obs_t)
        v_reward[:t_len] = values[:, 0].numpy()
        v_business[:t_len] = values[:, 1].numpy()
        v_carbon[:t_len] = values[:, 2].numpy()

    targets = compute_three_value_targets(
        np.array(rew_list), np.array(bc_list), np.array(cc_list),
        {"reward": v_reward, "business": v_business, "carbon": v_carbon},
    )

    adv_reward = torch.tensor(targets["reward"][0], dtype=torch.float32)
    tgt_reward = torch.tensor(targets["reward"][1], dtype=torch.float32)
    tgt_business = torch.tensor(targets["business"][1], dtype=torch.float32)
    tgt_carbon = torch.tensor(targets["carbon"][1], dtype=torch.float32)

    log_probs = torch.stack(logp_list)
    obs_t = torch.from_numpy(np.stack(obs_list))
    _, values = policy.forward(obs_t)

    actor_loss = -(adv_reward * log_probs).mean()
    critic_loss = (
        (values[:, 0] - tgt_reward).pow(2).mean()
        + (values[:, 1] - tgt_business).pow(2).mean()
        + (values[:, 2] - tgt_carbon).pow(2).mean()
    )
    loss = actor_loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lagrangian.update({"business": float(np.mean(bc_list)), "carbon": float(np.mean(cc_list))})

    return {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "reward_value": float(values[0, 0].item()),
        "business_value": float(values[0, 1].item()),
        "carbon_value": float(values[0, 2].item()),
        "multipliers": lagrangian.multipliers(),
        "corrector_on": corrector_on,
    }

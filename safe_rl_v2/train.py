"""M5.1c 训练入口：dry-run 更新，**唯一数据来源**是 contract-v6 `RolloutBuffer`。

定位不变：只验证「真实采集 → 一次更新」闭环可跑通，**不因短训练奖励高而宣称有效**。

**本模块不是 PPO**：没有 ratio、没有 clip、没有熵项。
GAE 与拉格朗日乘子仍是 M5.4 的占位实现，按任务卡 M5.1c §4.1 与 §5.4
列为 **M5.2 / M5.3 待重建**内容。**不得**据此宣称训练有效、收敛或任何性能结论。

红线：
- 采样只经 `collect_rollout`；raw 动作**原样**执行，**不存在** `_clip_action`；
- 可导 likelihood 只对 buffer 的 `raw_action` 计算
  （`evaluate_raw_actions`），**绝不对 `exec_action` 求概率**；
- `old_raw_log_prob` 是**旧策略审计值**，不进入任何损失项；
- 业务违规量 / 碳排放量 / 电费**分别**取自 buffer 具名字段，不混单位；
- 不使用 `info.get(..., 0)` 之类默认值；环境字段缺失在采集期直接报错；
- 不重新播种全局 Torch RNG：采样随机性由调用方传入的 `generator` 决定。
"""

from __future__ import annotations

import numpy as np
import torch

from safe_rl_v2.buffer import RolloutBuffer
from safe_rl_v2.lagrangian import Lagrangian
from safe_rl_v2.models import compute_three_value_targets
from safe_rl_v2.policy import SafePPOPolicy
from safe_rl_v2.rollout import collect_rollout

# 可导 likelihood 的来源（供审计断言，不参与计算）
LOG_PROB_SOURCE = "evaluate_raw_actions(raw_action)"


def dry_run_update(
    env,
    policy: SafePPOPolicy,
    lagrangian: Lagrangian,
    optimizer: torch.optim.Optimizer,
    *,
    corrector_on: bool = False,
    corrector_time_limit_s: float | None = None,
    steps: int = 8,
    seed: int = 0,
    generator: torch.Generator | None = None,
) -> dict:
    """短 dry rollout + 一次更新。返回三套 value、乘子、损耗与**真实采集**的 buffer。

    observation / raw_action / reward / 业务违规量 / 碳排放量 / 终止状态全部取自
    `collect_rollout` 写入的 buffer，不再自行采集。
    """
    buffer = RolloutBuffer()
    stats = collect_rollout(
        env,
        policy,
        buffer,
        steps=steps,
        seed=seed,
        corrector_on=corrector_on,
        corrector_time_limit_s=corrector_time_limit_s,
        generator=generator,
    )

    n_steps = len(buffer)
    if n_steps == 0:
        raise RuntimeError("collect_rollout 未采集到任何 transition，无法执行更新")

    obs = torch.as_tensor(
        np.stack([t.observation for t in buffer.transitions]), dtype=torch.float32
    )
    raw_actions = torch.as_tensor(
        np.stack([t.raw_action for t in buffer.transitions]), dtype=torch.float32
    )
    rewards = np.array([t.reward for t in buffer.transitions], dtype=np.float64)
    business_violations = np.array(
        [t.business_cost for t in buffer.transitions], dtype=np.float64
    )
    carbon_emissions = np.array([t.carbon_cost for t in buffer.transitions], dtype=np.float64)
    electricity_costs = np.array(
        [t.electricity_cost_sgd for t in buffer.transitions], dtype=np.float64
    )
    terminals = np.array([t.terminated for t in buffer.transitions], dtype=bool)

    # bootstrap 用最后一步的 next_observation；终止步不 bootstrap
    bootstrap_is_terminal = bool(terminals[-1])
    next_observation = torch.as_tensor(
        buffer.transitions[-1].next_observation, dtype=torch.float32
    )

    with torch.no_grad():
        _, final_values = policy.forward(next_observation)
        v_reward = np.zeros(n_steps + 1)
        v_business = np.zeros(n_steps + 1)
        v_carbon = np.zeros(n_steps + 1)
        if not bootstrap_is_terminal:
            v_reward[-1] = float(final_values[0])
            v_business[-1] = float(final_values[1])
            v_carbon[-1] = float(final_values[2])
        # 用 critic 对已收集 obs 重新估值
        _, values = policy.forward(obs)
        v_reward[:n_steps] = values[:, 0].numpy()
        v_business[:n_steps] = values[:, 1].numpy()
        v_carbon[:n_steps] = values[:, 2].numpy()

    targets = compute_three_value_targets(
        rewards,
        business_violations,
        carbon_emissions,
        {"reward": v_reward, "business": v_business, "carbon": v_carbon},
    )

    adv_reward = torch.tensor(targets["reward"][0], dtype=torch.float32)
    tgt_reward = torch.tensor(targets["reward"][1], dtype=torch.float32)
    tgt_business = torch.tensor(targets["business"][1], dtype=torch.float32)
    tgt_carbon = torch.tensor(targets["carbon"][1], dtype=torch.float32)

    # 可导 likelihood：只对 buffer 的 raw_action 重算（绝不对 exec_action）
    log_probs = policy.evaluate_raw_actions(obs, raw_actions)
    _, values = policy.forward(obs)

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
    lagrangian.update(
        {
            "business": float(np.mean(business_violations)),
            "carbon": float(np.mean(carbon_emissions)),
        }
    )

    return {
        # --- 既有键（M5.4 语义保持不变）---
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "reward_value": float(values[0, 0].item()),
        "business_value": float(values[0, 1].item()),
        "carbon_value": float(values[0, 2].item()),
        "multipliers": lagrangian.multipliers(),
        "corrector_on": corrector_on,
        # --- M5.1c 采集链证据（**不是**训练结论）---
        "buffer": buffer,
        "stats": stats,
        "steps_collected": n_steps,
        "logprob_source": LOG_PROB_SOURCE,
        "old_raw_log_prob_mean": float(
            np.mean([t.old_raw_log_prob for t in buffer.transitions])
        ),
        "business_violation_sum": float(np.sum(business_violations)),
        "carbon_emission_sum_kg": float(np.sum(carbon_emissions)),
        "electricity_cost_sum_sgd": float(np.sum(electricity_costs)),
        "env_seed": stats["env_seed"],
        "policy_rng_source": stats["policy_rng_source"],
        "bootstrap_is_terminal": bootstrap_is_terminal,
    }

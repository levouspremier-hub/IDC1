"""M5.1c–M5.3f 训练入口：dry-run 更新，**唯一数据来源**是 contract-v7 `RolloutBuffer`。

定位不变：只验证「真实采集 → 一次更新」闭环可跑通，**不因短训练奖励高而宣称有效**。

**本模块不是 PPO**：没有 ratio、没有 clip、没有熵项。
**M5.3 已完成**：乘子目标接线 —— actor 有效优势为
`A_reward − lambda_business·A_business − lambda_carbon·A_carbon`，
乘子取自 `Lagrangian`（更新前的值），约束信号的非负物理域与状态自洽性均已强制。
**M5.4 尚未完成**：训练入口（本模块仍无 `__main__`）、未来信息泄漏门禁、
corrector 开/关下的可复现性。
**不得**据此宣称训练有效、收敛或任何性能改善。

红线：
- 采样只经 `collect_rollout`；raw 动作**原样**执行，**不存在** `_clip_action`；
- 三套 critic target 的**每一个输入**都来自 buffer 的具名字段
  （`observation` / `next_observation` / `terminated` / `truncated` /
  `reward` / `business_cost` / `carbon_cost`），不回读 env 内部数组或未来真值；
- bootstrap **逐 transition** 取各自 `next_observation` 的 critic value
  （M5.2c 勘误：不得用下标 `t+1`，也不得只读最后一条）；
  是否使用由 `terminated` 掩码决定，调用方不自行判断；
- critic loss 逐头对应各自 target，三个独立 MSE，不混列、不共享、不以电费顶替；
- actor likelihood 只对 buffer 的 `raw_action` 计算（`evaluate_raw_actions`），
  **绝不对 `exec_action` 求概率**；`old_raw_log_prob` 是**旧策略审计值**，不进入损失；
- 不使用 `info.get(..., 0)` 之类默认值；环境字段缺失在采集期直接报错；
- 不重新播种全局 Torch RNG：采样随机性由调用方传入的 `generator` 决定。
"""

from __future__ import annotations

import numpy as np
import torch

from safe_rl_v2.buffer import RolloutBuffer
from safe_rl_v2.lagrangian import Lagrangian, validate_constraint_signals
from safe_rl_v2.models import compute_three_value_targets
from safe_rl_v2.policy import SafePPOPolicy
from safe_rl_v2.rollout import collect_rollout

# 可导 likelihood 的来源（供审计断言，不参与计算）
LOG_PROB_SOURCE = "evaluate_raw_actions(raw_action)"
# 三套 target 的唯一来源（M5.2a 的终端感知 API）
TARGETS_SOURCE = "compute_three_value_targets(terminated=, truncated=)"
# actor 有效优势的唯一形式（M5.3b；乘子为**更新前**值）
ACTOR_OBJECTIVE_SOURCE = "A_reward - lambda_business * A_business - lambda_carbon * A_carbon"

GAMMA = 0.99
LAM = 0.95

# 单位口径（M5.2a §5）：业务违规量是**每步活跃逾期 SLA 违规计数**，
# rollout 内累计为「违规任务·步」；碳排放为 kgCO2e；电费为 SGD。三者不得混算。
METRIC_UNITS = {
    "reward": "dimensionless",
    "business_violations": "violation_task_steps",
    "carbon_emissions": "kgCO2e",
    "electricity_cost_sgd": "SGD",
}

_CLAIMS = {
    "trained": False,
    "performance_evaluated": False,
    "convergence_claimed": False,
}


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
    """短 dry rollout + 一次更新。返回三套 value、乘子、损耗与**真实采集**的 buffer。"""
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
    next_obs = torch.as_tensor(
        np.stack([t.next_observation for t in buffer.transitions]), dtype=torch.float32
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
    truncations = np.array([t.truncated for t in buffer.transitions], dtype=bool)

    # --- M5.3f 预检：必须在**任何** forward / backward / optimizer.step 之前 ---
    # 复用 lagrangian 的共享校验规则（不在此处另写一套）。失败时本函数尚未产生
    # 任何副作用：policy 参数、optimizer state、Lagrangian state 三者都不变。
    validate_constraint_signals(
        {"business": business_violations, "carbon": carbon_emissions},
        lagrangian.constraints,
    )

    # bootstrap：**逐 transition** 取各自 next_observation 的 critic 估计
    # （M5.2c：不得用下标 t+1，也不得只读最后一条）
    with torch.no_grad():
        _, current_v = policy.forward(obs)
        _, next_v = policy.forward(next_obs)

    current_values = {
        "reward": current_v[:, 0].numpy(),
        "business": current_v[:, 1].numpy(),
        "carbon": current_v[:, 2].numpy(),
    }
    next_values = {
        "reward": next_v[:, 0].numpy(),
        "business": next_v[:, 1].numpy(),
        "carbon": next_v[:, 2].numpy(),
    }

    targets = compute_three_value_targets(
        rewards,
        business_violations,
        carbon_emissions,
        current_values,
        next_values,
        terminated=terminals,
        truncated=truncations,
        gamma=GAMMA,
        lam=LAM,
    )

    adv_reward = torch.tensor(targets["reward"][0], dtype=torch.float32)
    adv_business = torch.tensor(targets["business"][0], dtype=torch.float32)
    adv_carbon = torch.tensor(targets["carbon"][0], dtype=torch.float32)
    tgt_reward = torch.tensor(targets["reward"][1], dtype=torch.float32)
    tgt_business = torch.tensor(targets["business"][1], dtype=torch.float32)
    tgt_carbon = torch.tensor(targets["carbon"][1], dtype=torch.float32)

    # 可导 likelihood：只对 buffer 的 raw_action 重算（绝不对 exec_action）
    log_probs = policy.evaluate_raw_actions(obs, raw_actions)
    _, values = policy.forward(obs)

    # 逐头对应的 critic loss：三个独立 MSE，分别上报，不混列
    critic_loss_by_head = {
        "reward": (values[:, 0] - tgt_reward).pow(2).mean(),
        "business": (values[:, 1] - tgt_business).pow(2).mean(),
        "carbon": (values[:, 2] - tgt_carbon).pow(2).mean(),
    }
    critic_loss = (
        critic_loss_by_head["reward"]
        + critic_loss_by_head["business"]
        + critic_loss_by_head["carbon"]
    )

    # M5.3b：本轮 actor 目标使用**更新前**的乘子（两约束各自独立）
    multipliers_pre = lagrangian.multipliers()
    lambda_business = float(multipliers_pre["business"])
    lambda_carbon = float(multipliers_pre["carbon"])

    reward_term = adv_reward
    business_term = -lambda_business * adv_business
    carbon_term = -lambda_carbon * adv_carbon
    effective_advantage = reward_term + business_term + carbon_term

    actor_loss = -(effective_advantage * log_probs).mean()
    loss = actor_loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 乘子更新发生在 optimizer.step() **之后**，且只消费逐 transition 序列
    # （聚合口径 per-transition mean 由 lagrangian 模块内部固定）。
    lagrangian.update(
        {
            "business": business_violations,
            "carbon": carbon_emissions,
        }
    )
    multipliers_post = lagrangian.multipliers()

    return {
        # --- 既有键（M5.1c/M5.4 语义保持不变）---
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "reward_value": float(values[0, 0].item()),
        "business_value": float(values[0, 1].item()),
        "carbon_value": float(values[0, 2].item()),
        "multipliers": multipliers_post,
        "corrector_on": corrector_on,
        # --- 采集链证据（**不是**训练结论）---
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
        "bootstrap_is_terminal": bool(terminals[-1]),
        # --- M5.2b/M5.2d：三套 target 的来源与口径 ---
        "targets_source": TARGETS_SOURCE,
        "gae": {"gamma": GAMMA, "lam": LAM},
        "bootstrap": {
            "source": "buffer_transition_next_observations",
            "terminated_count": int(terminals.sum()),
            "truncated_count": int(truncations.sum()),
            "bootstrapped_count": int((~terminals).sum()),
        },
        "critic_loss_by_head": {head: float(v.item()) for head, v in critic_loss_by_head.items()},
        "critic_targets": {head: pair[1] for head, pair in targets.items()},
        "units": dict(METRIC_UNITS),
        "claims": dict(_CLAIMS),
        # --- M5.3b：actor objective 的乘子口径与诊断（**不是**性能结论）---
        "actor_objective_source": ACTOR_OBJECTIVE_SOURCE,
        "multipliers_pre_update": dict(multipliers_pre),
        "multipliers_post_update": dict(multipliers_post),
        "constraint_means": {
            "business": float(np.mean(business_violations)),
            "carbon": float(np.mean(carbon_emissions)),
        },
        "constraint_budgets": {
            name: float(state.budget) for name, state in lagrangian.constraints.items()
        },
        "constraint_units": {
            name: state.unit for name, state in lagrangian.constraints.items()
        },
        "effective_advantage": {
            "formula": ACTOR_OBJECTIVE_SOURCE,
            "multipliers_used": "pre_update",
            "mean": float(effective_advantage.mean().item()),
            "std": float(effective_advantage.std(unbiased=False).item()),
            "min": float(effective_advantage.min().item()),
            "max": float(effective_advantage.max().item()),
        },
        "effective_advantage_terms": {
            "reward_mean": float(reward_term.mean().item()),
            "business_mean": float(business_term.mean().item()),
            "carbon_mean": float(carbon_term.mean().item()),
        },
    }

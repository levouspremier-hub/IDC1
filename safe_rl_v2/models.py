"""M5.2 三套独立价值：收益 / 业务约束 / 碳约束各有独立 value 与 GAE。

不把期望约束学习表述为逐步硬安全保证（这是约束 RL 的估计，非硬保证）。
"""

from __future__ import annotations

import numpy as np


def compute_gae(
    costs: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """广义优势估计。values 长度 = len(costs)+1（含 bootstrap）。"""
    costs = np.asarray(costs, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    t_len = len(costs)
    advantages = np.zeros(t_len, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(t_len)):
        delta = costs[t] + gamma * values[t + 1] - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    targets = advantages + values[:t_len]
    return advantages, targets


def compute_three_value_targets(
    rewards: np.ndarray,
    business_costs: np.ndarray,
    carbon_costs: np.ndarray,
    values: dict[str, np.ndarray],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """三套独立 target：收益、业务、碳各自独立 GAE，改变一种 cost 不影响另两种。"""
    adv_r, tgt_r = compute_gae(rewards, values["reward"], gamma, lam)
    adv_b, tgt_b = compute_gae(business_costs, values["business"], gamma, lam)
    adv_c, tgt_c = compute_gae(carbon_costs, values["carbon"], gamma, lam)
    return {
        "reward": (adv_r, tgt_r),
        "business": (adv_b, tgt_b),
        "carbon": (adv_c, tgt_c),
    }

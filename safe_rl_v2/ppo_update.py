"""M1.3g-f-c-c：基于 formal buffer 的**单次** PPO 更新。

把既有件接成一次更新：

```text
三头 target（compute_three_value_targets）
  → 有效优势 A_reward − λ_business·A_business − λ_carbon·A_carbon（**更新前**乘子）
  → clipped actor objective（safe_rl_v2.ppo_objective）
  → 三个**各自独立**的 critic MSE
  → optimizer.zero_grad(); loss.backward(); optimizer.step()   ← 恰好一次
  → lagrangian.update(...)                                      ← 在 step **之后**
```

**边界（本卡不做的事）**：不接训练入口、不做多轮训练、不写 checkpoint、
不修改 `train.py` / env / 冻结资产 / 发布产物 v1；
**不**以短更新结果声称训练有效（返回的 `claims` 三项恒为 `False`）。

红线：

- `old_raw_log_prob` 与三头优势在更新中**固定**，**不**参与反传（进入损失前
  一律 `detach()`）；
- 新 log-prob **只**由 `raw_action` 重算（`evaluate_raw_actions`），
  **绝不**改用 `exec_action`——即使 corrector 开启使 `raw ≠ exec`；
- 三个 critic **各自**对自己的 target，**不**混列、**不**共享；
- 乘子取**本次更新前**的值参与 actor 目标；
- 损失非有限或 buffer 为空时**明确失败**，且**不得**已经 `step()`。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from safe_rl_v2.models import compute_three_value_targets
from safe_rl_v2.ppo_objective import (
    effective_advantage,
    ppo_clipped_actor_objective,
)

__all__ = [
    "compute_targets_from_buffer",
    "new_raw_log_prob_from_buffer",
    "single_ppo_update",
    "single_ppo_update_from_arrays",
]

CLAIMS = {"trained": False, "performance_evaluated": False, "convergence_claimed": False}

HEADS = ("reward", "business", "carbon")


def _stack_buffer(buffer, field: str) -> torch.Tensor:
    transitions = list(buffer.transitions)
    return torch.as_tensor(
        np.stack([np.asarray(getattr(t, field), dtype=np.float32) for t in transitions]),
        dtype=torch.float32)


def _floats(buffer, field: str) -> torch.Tensor:
    return torch.as_tensor(
        [float(getattr(t, field)) for t in buffer.transitions], dtype=torch.float32)


def _require_non_empty(buffer) -> list:
    transitions = list(getattr(buffer, "transitions", []))
    if not transitions:
        raise ValueError("buffer 为空：单次 PPO 更新无从计算，明确失败")
    return transitions


def new_raw_log_prob_from_buffer(policy: Any, buffer: Any) -> torch.Tensor:
    """新 log-prob：**只**由 `buffer` 的 `observation` + **`raw_action`** 重算。"""
    _require_non_empty(buffer)
    return policy.evaluate_raw_actions(
        _stack_buffer(buffer, "observation"), _stack_buffer(buffer, "raw_action"))


def compute_targets_from_buffer(policy: Any, buffer: Any, *, gamma: float,
                                lam: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """用**当前** policy 的 critic 估计三头 target（各头独立 GAE）。"""
    _require_non_empty(buffer)
    obs = _stack_buffer(buffer, "observation")
    next_obs = _stack_buffer(buffer, "next_observation")
    with torch.no_grad():
        _, current_v = policy.forward(obs)
        _, next_v = policy.forward(next_obs)
    current_values = {head: current_v[:, i].numpy() for i, head in enumerate(HEADS)}
    next_values = {head: next_v[:, i].numpy() for i, head in enumerate(HEADS)}

    transitions = list(buffer.transitions)
    return compute_three_value_targets(
        np.array([float(t.reward) for t in transitions], dtype=np.float64),
        np.array([float(t.business_cost) for t in transitions], dtype=np.float64),
        np.array([float(t.carbon_cost) for t in transitions], dtype=np.float64),
        current_values,
        next_values,
        terminated=np.array([bool(t.terminated) for t in transitions], dtype=bool),
        truncated=np.array([bool(t.truncated) for t in transitions], dtype=bool),
        gamma=gamma,
        lam=lam,
    )


def _single_update(
    policy: Any,
    optimizer: Any,
    lagrangian: Any,
    *,
    observation: torch.Tensor,
    next_observation: torch.Tensor,
    raw_action: torch.Tensor,
    old_raw_log_prob: torch.Tensor,
    rewards: torch.Tensor,
    business_violations: torch.Tensor,
    carbon_emissions: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    adv_reward: torch.Tensor | None,
    adv_business: torch.Tensor | None,
    adv_carbon: torch.Tensor | None,
    clip_epsilon: float,
    gamma: float,
    lam: float,
) -> dict[str, Any]:
    n = int(observation.shape[0])
    if n == 0:
        raise ValueError("单次 PPO 更新需要至少一个 transition")

    # --- 三头 target：各头独立 ------------------------------------------------
    if adv_reward is None or adv_business is None or adv_carbon is None:
        raise ValueError("必须给出三头优势（adv_reward/adv_business/adv_carbon）")

    # --- 固定性：old log-prob 与优势**一律 detach**，不可经 actor loss 反传 ----
    old_detached = old_raw_log_prob.detach()
    advantage = effective_advantage(
        adv_reward.detach(), adv_business.detach(), adv_carbon.detach(),
        lambda_business=float(lagrangian.multipliers()["business"]),
        lambda_carbon=float(lagrangian.multipliers()["carbon"]),
    )

    multiplier_pre = {k: float(v) for k, v in lagrangian.multipliers().items()}
    updates_before = int(getattr(lagrangian, "_updates", 0))

    # --- actor：clipped objective（新 log-prob 只由 raw_action 现算）----------
    actor = ppo_clipped_actor_objective(
        policy, observation=observation, raw_action=raw_action,
        old_raw_log_prob=old_detached, adv_reward=advantage,
        adv_business=torch.zeros_like(advantage),
        adv_carbon=torch.zeros_like(advantage),
        lambda_business=0.0, lambda_carbon=0.0, clip_epsilon=clip_epsilon)
    actor_loss = actor["loss"]

    # --- critic：三个**各自独立**的 MSE，分别对自己的 target ------------------
    _, values = policy.forward(observation)
    targets = _critic_targets(
        policy, observation, next_observation, rewards, business_violations,
        carbon_emissions, terminated, truncated, gamma=gamma, lam=lam)
    critic_by_head = {
        head: (values[:, i] - targets[head]).pow(2).mean()
        for i, head in enumerate(HEADS)
    }
    critic_loss = sum(critic_by_head.values())
    loss = actor_loss + critic_loss

    if not bool(torch.isfinite(loss)):
        raise ValueError(
            f"损失非有限（{float(loss)!r}）：明确失败，**不执行** optimizer.step()")

    # --- 恰好一次 step -------------------------------------------------------
    params_before = [p.detach().clone() for p in policy.parameters()]
    optimizer.zero_grad()
    loss.backward()
    grad_norm_actor = float(
        torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in policy.parameters()
                       if p.grad is not None)).item())
    optimizer.step()

    delta = sum(float(((p.detach() - b) ** 2).sum().item())
                for p, b in zip(policy.parameters(), params_before, strict=True))

    # --- 乘子更新在 step **之后**（与既有入口一致）----------------------------
    lagrangian.update({
        "business": business_violations.detach().numpy(),
        "carbon": carbon_emissions.detach().numpy(),
    })

    return {
        "loss_total": loss.detach(),
        "actor_loss": actor_loss.detach(),
        "critic_loss_total": critic_loss.detach(),
        "critic_loss_by_head": {h: v.detach() for h, v in critic_by_head.items()},
        "ratio": actor["ratio"],
        "clip_fraction": actor["clip_fraction"],
        "clip_epsilon": float(clip_epsilon),
        "logprob_source": actor["logprob_source"],
        "multipliers_pre_update": multiplier_pre,
        "multipliers_post_update": {k: float(v)
                                    for k, v in lagrangian.multipliers().items()},
        "lagrangian_updates_before": updates_before,
        "lagrangian_updates_after": int(getattr(lagrangian, "_updates", 0)),
        "optimizer_steps": 1,
        "grad_norm_actor": grad_norm_actor,
        "param_delta_norm": float(np.sqrt(delta)),
        "num_transitions": n,
        "claims": dict(CLAIMS),
    }


def _critic_targets(policy, observation, next_observation, rewards,
                    business_violations, carbon_emissions, terminated, truncated, *,
                    gamma, lam):
    """用**当前** critic 对 `observation` / `next_observation` 的估计算三头 target。

    每头各跑一次 GAE，**互不影响**；返回 tensor 以便反传（target 本身已 detach）。
    """
    with torch.no_grad():
        _, current_v = policy.forward(observation)
        _, next_v = policy.forward(next_observation)
    current_values = {head: current_v[:, i].numpy() for i, head in enumerate(HEADS)}
    next_values = {head: next_v[:, i].numpy() for i, head in enumerate(HEADS)}
    targets = compute_three_value_targets(
        rewards.detach().numpy(), business_violations.detach().numpy(),
        carbon_emissions.detach().numpy(), current_values, next_values,
        terminated=terminated.detach().numpy(), truncated=truncated.detach().numpy(),
        gamma=gamma, lam=lam)
    return {head: torch.as_tensor(targets[head][1], dtype=torch.float32)
            for head in HEADS}


def single_ppo_update_from_arrays(
    policy: Any,
    optimizer: Any,
    lagrangian: Any,
    *,
    observation: torch.Tensor,
    next_observation: torch.Tensor,
    raw_action: torch.Tensor,
    old_raw_log_prob: torch.Tensor,
    rewards: torch.Tensor,
    business_violations: torch.Tensor,
    carbon_emissions: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    adv_reward: torch.Tensor,
    adv_business: torch.Tensor,
    adv_carbon: torch.Tensor,
    clip_epsilon: float,
    gamma: float,
    lam: float,
) -> dict[str, Any]:
    """数组入口：调用方直接给出三头优势（用于固定性与失败语义的精确控制）。"""
    return _single_update(
        policy, optimizer, lagrangian,
        observation=observation, next_observation=next_observation,
        raw_action=raw_action,
        old_raw_log_prob=old_raw_log_prob, rewards=rewards,
        business_violations=business_violations, carbon_emissions=carbon_emissions,
        terminated=terminated, truncated=truncated,
        adv_reward=adv_reward, adv_business=adv_business, adv_carbon=adv_carbon,
        clip_epsilon=clip_epsilon, gamma=gamma, lam=lam)


def single_ppo_update(
    policy: Any,
    optimizer: Any,
    lagrangian: Any,
    buffer: Any,
    *,
    clip_epsilon: float,
    gamma: float,
    lam: float,
) -> dict[str, Any]:
    """buffer 入口：三头优势由**本次更新前**的 critic + 真实 `next_observation` 现算。"""
    transitions = _require_non_empty(buffer)

    obs = _stack_buffer(buffer, "observation")
    next_obs = _stack_buffer(buffer, "next_observation")
    rewards = _floats(buffer, "reward")
    business = _floats(buffer, "business_cost")
    carbon = _floats(buffer, "carbon_cost")
    terminated = torch.as_tensor([bool(t.terminated) for t in transitions])
    truncated = torch.as_tensor([bool(t.truncated) for t in transitions])

    # 三头 target（各头独立；真实 next_observation 参与 bootstrap）
    targets = compute_targets_from_buffer(policy, buffer, gamma=gamma, lam=lam)
    adv_reward = torch.as_tensor(targets["reward"][0], dtype=torch.float32)
    adv_business = torch.as_tensor(targets["business"][0], dtype=torch.float32)
    adv_carbon = torch.as_tensor(targets["carbon"][0], dtype=torch.float32)

    return _single_update(
        policy, optimizer, lagrangian,
        observation=obs, next_observation=next_obs,
        raw_action=_stack_buffer(buffer, "raw_action"),
        old_raw_log_prob=_floats(buffer, "old_raw_log_prob"),
        rewards=rewards, business_violations=business, carbon_emissions=carbon,
        terminated=terminated, truncated=truncated,
        adv_reward=adv_reward, adv_business=adv_business, adv_carbon=adv_carbon,
        clip_epsilon=clip_epsilon, gamma=gamma, lam=lam)


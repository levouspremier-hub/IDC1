"""M1.3g-f-c-b：PPO clipped actor objective 核心（**纯数值计算**）。

本模块**只**从「已采集的 buffer 数值」计算 PPO 的 clipped actor objective：

```text
ratio            = exp(new_raw_log_prob − old_raw_log_prob)
advantage        = A_reward − λ_business·A_business − λ_carbon·A_carbon
clipped surrogate = min(ratio × A, clip(ratio, 1−ε, 1+ε) × A)
actor loss        = −mean(clipped surrogate)
```

**边界（本卡不做的事）**：不接训练入口、不运行正式训练、不写 checkpoint、
不修改 `train.py` / env / 冻结资产 / 发布产物 v1、不擅自冻结训练超参数。

红线：

- **新 log-prob 只能由 `raw_action` 求得**（`policy.evaluate_raw_actions`），
  **绝不**以 `exec_action` 替代；本模块**不提供**任何传入新 log-prob 或
  `exec_action` 的入口，从而在接口层面排除该误用。
- **`clip_epsilon` 必须由调用方显式传入**（keyword-only、**无默认值**），
  本卡不替调用方冻结该超参数。
- 三头优势的**乘子取自调用方**（通常是更新前的 Lagrangian 值），不在本模块内部
  选择或更新乘子。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

__all__ = [
    "clipped_surrogate",
    "compute_ratio",
    "effective_advantage",
    "ppo_actor_objective_from_buffer",
    "ppo_clipped_actor_objective",
]


def _require_same_shape(name_a: str, a: torch.Tensor, name_b: str,
                        b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{name_a} 与 {name_b} 形状必须一致：{tuple(a.shape)} vs {tuple(b.shape)}")


def _require_clip_epsilon(clip_epsilon: Any) -> float:
    """`clip_epsilon` 必须是显式给出的有限实数，且 `0 < ε < 1`。"""
    if isinstance(clip_epsilon, bool) or not isinstance(clip_epsilon, (int, float)):
        raise TypeError(f"clip_epsilon 必须是实数，实际 {type(clip_epsilon).__name__}")
    value = float(clip_epsilon)
    if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError(f"clip_epsilon 必须满足 0 < ε < 1 且有限，实际 {clip_epsilon!r}")
    return value


def compute_ratio(new_raw_log_prob: torch.Tensor,
                  old_raw_log_prob: torch.Tensor) -> torch.Tensor:
    """`ratio = exp(new − old)`；两者相同时**精确**为 1。"""
    _require_same_shape("new_raw_log_prob", new_raw_log_prob,
                        "old_raw_log_prob", old_raw_log_prob)
    return torch.exp(new_raw_log_prob - old_raw_log_prob)


def effective_advantage(adv_reward: torch.Tensor, adv_business: torch.Tensor,
                        adv_carbon: torch.Tensor, *,
                        lambda_business: float,
                        lambda_carbon: float) -> torch.Tensor:
    """三头优势的**唯一**形式：`A_reward − λ_business·A_business − λ_carbon·A_carbon`。

    乘子由调用方给出（通常为**更新前**的 Lagrangian 值）。
    """
    _require_same_shape("adv_reward", adv_reward, "adv_business", adv_business)
    _require_same_shape("adv_reward", adv_reward, "adv_carbon", adv_carbon)
    return adv_reward - float(lambda_business) * adv_business \
        - float(lambda_carbon) * adv_carbon


def clipped_surrogate(ratio: torch.Tensor, advantage: torch.Tensor, *,
                      clip_epsilon: float) -> torch.Tensor:
    """逐样本 `min(ratio×A, clip(ratio, 1−ε, 1+ε)×A)`。

    对**正、负**优势都取 min —— 故越界且优势为负时**不**会把损失「救回」。
    """
    eps = _require_clip_epsilon(clip_epsilon)
    _require_same_shape("ratio", ratio, "advantage", advantage)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantage
    return torch.minimum(unclipped, clipped)


def ppo_clipped_actor_objective(
    policy: Any,
    *,
    observation: torch.Tensor,
    raw_action: torch.Tensor,
    old_raw_log_prob: torch.Tensor,
    adv_reward: torch.Tensor,
    adv_business: torch.Tensor,
    adv_carbon: torch.Tensor,
    lambda_business: float,
    lambda_carbon: float,
    clip_epsilon: float,
) -> dict[str, Any]:
    """计算 clipped actor objective。

    **新 log-prob 由 `raw_action` 现算**（`policy.evaluate_raw_actions`）；
    本函数**没有**任何传入新 log-prob 或 `exec_action` 的参数。
    """
    new_raw_log_prob = policy.evaluate_raw_actions(observation, raw_action)
    # Diagnostic ratios do not need an exp gradient (which is inf on overflow).
    ratio = compute_ratio(new_raw_log_prob.detach(), old_raw_log_prob.detach())
    advantage = effective_advantage(
        adv_reward, adv_business, adv_carbon,
        lambda_business=lambda_business, lambda_carbon=lambda_carbon)
    eps = _require_clip_epsilon(clip_epsilon)
    log_ratio = new_raw_log_prob - old_raw_log_prob
    # For nonnegative A and overflowing positive ratio, the prescribed minimum
    # is exactly (1+eps)*A, with zero derivative w.r.t. the actor log probability.
    # Avoid evaluating exp(inf) in that backward path: 0*inf would yield NaN.
    # Negative-A overflow remains unbounded and must fail the finite-loss guard.
    overflow_clipped = torch.isinf(ratio) & (log_ratio > 0) & (advantage >= 0)
    safe_log_ratio = torch.where(overflow_clipped, torch.zeros_like(log_ratio), log_ratio)
    differentiable_ratio = torch.exp(safe_log_ratio)
    surrogate = clipped_surrogate(differentiable_ratio, advantage, clip_epsilon=eps)
    surrogate = torch.where(overflow_clipped, (1.0 + eps) * advantage, surrogate)
    loss = -surrogate.mean()

    ratios_clipped = torch.clamp(ratio, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon))
    return {
        "loss": loss,
        "surrogate": surrogate,
        "ratio": ratio.detach(),
        "new_raw_log_prob": new_raw_log_prob.detach(),
        "effective_advantage": advantage.detach(),
        "clip_epsilon": float(clip_epsilon),
        "clip_fraction": float((ratio != ratios_clipped).to(torch.float64).mean().item()),
        "num_samples": int(ratio.numel()),
        "logprob_source": "evaluate_raw_actions(raw_action)",
    }


def _as_tensor(values: Sequence[float] | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(values):
        return values.detach().to(torch.float32)
    return torch.as_tensor(np.asarray(list(values), dtype=np.float32))


def ppo_actor_objective_from_buffer(
    policy: Any,
    buffer: Any,
    *,
    adv_reward: Sequence[float] | torch.Tensor,
    adv_business: Sequence[float] | torch.Tensor,
    adv_carbon: Sequence[float] | torch.Tensor,
    lambda_business: float,
    lambda_carbon: float,
    clip_epsilon: float,
) -> dict[str, Any]:
    """从 `RolloutBuffer` 契约字段（`observation` / `raw_action` /
    `old_raw_log_prob`）计算同一 objective。

    优势由调用方按既有三头语义给出（本模块不重算 GAE）。
    """
    transitions = list(buffer.transitions)
    if not transitions:
        raise ValueError("buffer 为空，无法计算 actor objective")
    obs = torch.as_tensor(
        np.stack([np.asarray(t.observation, dtype=np.float32) for t in transitions]),
        dtype=torch.float32)
    raw = torch.as_tensor(
        np.stack([np.asarray(t.raw_action, dtype=np.float32) for t in transitions]),
        dtype=torch.float32)
    old = _as_tensor([float(t.old_raw_log_prob) for t in transitions])

    out = ppo_clipped_actor_objective(
        policy, observation=obs, raw_action=raw, old_raw_log_prob=old,
        adv_reward=_as_tensor(adv_reward), adv_business=_as_tensor(adv_business),
        adv_carbon=_as_tensor(adv_carbon),
        lambda_business=lambda_business, lambda_carbon=lambda_carbon,
        clip_epsilon=clip_epsilon)
    out["num_transitions"] = len(transitions)
    return out

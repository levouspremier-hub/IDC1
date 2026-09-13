"""M5.4 / M5.1b 策略网络：actor（obs → 21 维**有界** raw 动作）+ critic（obs → 3 套 value）。

raw 动作空间与 `envs/idc_price_env.py` 完全一致：前 `action_dim-1` 维计算强度 ∈ [0,1]，
末维有符号储能 ∈ [-1,1]（负充电、正放电）。

采样用 tanh-squash：`u ~ Normal(mean, std)`，`raw = squash(u)`。
`old_raw_log_prob` 一律由 `evaluate_raw_actions()` 从**最终 raw 动作**重算（含 Jacobian 修正），
因此概率记录永远对应真正落库的 `raw_action`——**禁止**先 clip 再沿用 clip 前的概率。

本卡（M5.1b）不实现 PPO ratio/clip、GAE、actor loss、熵项或乘子更新。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

# atanh / log-Jacobian 的数值护栏。`act()` 与 `evaluate_raw_actions()` 走**同一条**
# 逆变换路径，故护栏不会造成两者之间的概率不一致。
RAW_ACTION_EPS = 1e-6


def raw_action_bounds(action_dim: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """返回 (low, high)：前 action_dim-1 维 [0,1]，末维 [-1,1]。"""
    low = np.concatenate(
        [np.zeros(action_dim - 1, dtype=np.float32), np.array([-1.0], dtype=np.float32)]
    )
    high = np.ones(action_dim, dtype=np.float32)
    return low, high


def _sample_pre_squash(
    mean: torch.Tensor, std: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """从 N(mean, std) 采样。给定 `generator` 时**只**消耗该 generator。

    不使用 `torch.manual_seed`：全局 RNG 状态属于调用方，本模块不得改动。
    """
    expanded = std.expand_as(mean)
    if generator is None:
        return torch.normal(mean, expanded)
    return torch.normal(mean, expanded, generator=generator)


def _squash(u: torch.Tensor, action_dim: int) -> torch.Tensor:
    """无界样本 → 有界 raw 动作：计算维 0.5*(tanh+1) ∈ (0,1)，储能维 tanh ∈ (-1,1)。"""
    compute = 0.5 * (torch.tanh(u[..., : action_dim - 1]) + 1.0)
    storage = torch.tanh(u[..., action_dim - 1 :])
    return torch.cat([compute, storage], dim=-1)


def _pre_squash(raw: torch.Tensor, action_dim: int) -> torch.Tensor:
    """有界 raw 动作 → 无界样本（`_squash` 的逆，带数值护栏）。"""
    compute = torch.atanh(
        torch.clamp(
            2.0 * raw[..., : action_dim - 1] - 1.0,
            -1.0 + RAW_ACTION_EPS,
            1.0 - RAW_ACTION_EPS,
        )
    )
    storage = torch.atanh(
        torch.clamp(raw[..., action_dim - 1 :], -1.0 + RAW_ACTION_EPS, 1.0 - RAW_ACTION_EPS)
    )
    return torch.cat([compute, storage], dim=-1)


def _squash_log_det(u: torch.Tensor, action_dim: int) -> torch.Tensor:
    """log |det ∂raw/∂u|，逐元素求和到最后一维。"""
    t = torch.tanh(u)
    jac = torch.cat(
        [
            0.5 * (1.0 - t[..., : action_dim - 1] ** 2),
            1.0 - t[..., action_dim - 1 :] ** 2,
        ],
        dim=-1,
    )
    return torch.log(torch.clamp(jac, min=RAW_ACTION_EPS)).sum(-1)


class SafePPOPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 21, hidden: int = 64) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, action_dim)
        )
        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, 3))
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs)

    def act(
        self, obs: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样**有界** raw 动作，并返回与**该最终动作**严格对应的 log-prob。

        `generator` 给定时采样只消耗该 generator，全局 Torch RNG 不受影响。
        """
        mean, values = self.forward(obs)
        u = _sample_pre_squash(mean, self.log_std.exp(), generator)
        raw_action = _squash(u, self.action_dim)
        # 概率从最终 raw 动作重算：与 evaluate_raw_actions 同路径，故二者恒等
        log_prob = self.evaluate_raw_actions(obs, raw_action)
        return raw_action, log_prob, values

    def evaluate_raw_actions(self, obs: torch.Tensor, raw_action) -> torch.Tensor:
        """给定 raw 动作重算 log-prob；`act()` 对同一样本返回同一数值。"""
        mean, _ = self.forward(obs)
        if torch.is_tensor(raw_action):
            raw = raw_action.to(dtype=mean.dtype)
        else:
            raw = torch.as_tensor(np.asarray(raw_action), dtype=mean.dtype)
        u = _pre_squash(raw, self.action_dim)
        log_prob = Normal(mean, self.log_std.exp()).log_prob(u).sum(-1)
        return log_prob - _squash_log_det(u, self.action_dim)

    def act_mean(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return mean

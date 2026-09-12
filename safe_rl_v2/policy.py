"""M5.4 策略网络：actor（obs → 21 维动作）+ critic（obs → 3 套 value）。"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


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

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, values = self.forward(obs)
        dist = Normal(mean, self.log_std.exp())
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, values

    def act_mean(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return mean

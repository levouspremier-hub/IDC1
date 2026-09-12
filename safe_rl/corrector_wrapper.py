"""M4.5 训练/评估同语义 wrapper。

同一版修正器同时用于训练与评估。transition/info 同时存 raw_action、exec_action、
correction_reason、求解时间与 business_gap；reward/next state 由 exec 动作产生；
raw_action 单独记录（PPO buffer 仅对 raw 计算 log-prob）。
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np

from contracts.models import DispatchProposal
from planning.corrector import correct
from planning.snapshot_adapter import build_snapshot


class CorrectorWrapper(gym.Wrapper):
    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        n_group = self.env.model.N
        proposal = DispatchProposal(
            compute_actions=[float(x) for x in action[:n_group]],
            storage_action=float(action[n_group]),
        )

        snapshot = build_snapshot(self.env)
        t0 = time.perf_counter()
        correction = correct(snapshot, proposal)
        solve_time = time.perf_counter() - t0

        exec_action = np.concatenate(
            [
                np.asarray(correction.exec_compute_actions, dtype=np.float32),
                np.array([np.clip(correction.exec_storage_action, -1.0, 1.0)], dtype=np.float32),
            ]
        )

        obs, reward, terminated, truncated, info = self.env.step(exec_action)
        info["raw_action"] = action
        info["exec_action"] = exec_action
        info["correction_reason"] = str(correction.failure)
        info["correction_solve_time_s"] = float(solve_time)
        info["business_gap"] = float(correction.business_gap)
        return obs, reward, terminated, truncated, info

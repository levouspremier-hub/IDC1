"""M4.5 / M4.4a 训练/评估同语义 wrapper。

同一版正确器同时用于训练与评估。raw action **逐元素原样**记录；exec action 是
正确器（H 步 MIP 原始动作投影）的**第 0 步**输出；reward、next observation 与
环境 info 均由 exec action 产生。

红线：不修改 raw log-prob 或任何 PPO 记录；exec action 绝不覆盖 raw action。
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from contracts.models import DispatchProposal
from planning.corrector import correct
from planning.snapshot_adapter import build_snapshot


class CorrectorWrapper(gym.Wrapper):
    def step(self, action):
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        n_group = self.env.model.N
        proposal = DispatchProposal(
            compute_actions=[float(x) for x in raw_action[:n_group]],
            storage_action=float(raw_action[n_group]),
        )

        snapshot = build_snapshot(self.env)
        correction = correct(snapshot, proposal)

        exec_action = np.concatenate(
            [
                np.asarray(correction.exec_compute_actions, dtype=np.float32),
                np.array(
                    [np.clip(correction.exec_storage_action, -1.0, 1.0)], dtype=np.float32
                ),
            ]
        )

        obs, reward, terminated, truncated, info = self.env.step(exec_action)

        # raw 原样保留（逐元素）；exec 为正确器第 0 步投影输出
        info["raw_action"] = raw_action
        info["exec_action"] = exec_action
        info["correction_reason"] = str(correction.failure)
        info["correction_detail"] = correction.reason
        info["correction_solve_time_s"] = float(correction.solve_time_s)
        info["business_gap"] = float(correction.business_gap)
        info["deadline_shortfall_work"] = float(correction.deadline_shortfall_work)
        # 审计字段（两阶段与后端）
        info["planner_backend"] = correction.planner_backend
        info["stage_a_status"] = correction.stage_a_status
        info["stage_b_status"] = correction.stage_b_status
        info["stage_a_solve_time_s"] = float(correction.stage_a_solve_time_s)
        info["stage_b_solve_time_s"] = float(correction.stage_b_solve_time_s)
        info["stage_a_objective"] = float(correction.stage_a_objective)
        info["stage_b_objective"] = float(correction.stage_b_objective)
        info["projection_offset"] = float(correction.projection_offset)
        return obs, reward, terminated, truncated, info

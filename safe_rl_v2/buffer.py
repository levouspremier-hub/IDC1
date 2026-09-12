"""M5.1 独立新训练 buffer。

同存 observation、raw action、raw log-prob、reward、业务 cost、碳 cost、exec action、
修正信息与 contract version。exec action 绝不覆盖 raw action（log-prob 只关联 raw）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONTRACT_VERSION = "contract-v1"
ACTION_DIM = 21  # 20 compute + 1 signed storage


@dataclass
class Transition:
    observation: np.ndarray
    raw_action: np.ndarray
    raw_log_prob: float
    reward: float
    business_cost: float
    carbon_cost: float
    exec_action: np.ndarray
    correction_info: dict
    contract_version: str = CONTRACT_VERSION


class RolloutBuffer:
    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    def add(self, transition: Transition) -> None:
        if transition.raw_action.shape != (ACTION_DIM,):
            raise ValueError(f"raw_action 必须 {ACTION_DIM} 维，got {transition.raw_action.shape}")
        if transition.exec_action.shape != (ACTION_DIM,):
            raise ValueError(
                f"exec_action 必须 {ACTION_DIM} 维，got {transition.exec_action.shape}"
            )
        self.transitions.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)

    def clear(self) -> None:
        self.transitions.clear()

    def to_dict(self) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "transitions": [
                {
                    "observation": t.observation.tolist(),
                    "raw_action": t.raw_action.tolist(),
                    "raw_log_prob": t.raw_log_prob,
                    "reward": t.reward,
                    "business_cost": t.business_cost,
                    "carbon_cost": t.carbon_cost,
                    "exec_action": t.exec_action.tolist(),
                    "correction_info": t.correction_info,
                    "contract_version": t.contract_version,
                }
                for t in self.transitions
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> RolloutBuffer:
        buffer = cls()
        for d in data["transitions"]:
            buffer.add(
                Transition(
                    observation=np.asarray(d["observation"]),
                    raw_action=np.asarray(d["raw_action"]),
                    raw_log_prob=d["raw_log_prob"],
                    reward=d["reward"],
                    business_cost=d["business_cost"],
                    carbon_cost=d["carbon_cost"],
                    exec_action=np.asarray(d["exec_action"]),
                    correction_info=d["correction_info"],
                    contract_version=d["contract_version"],
                )
            )
        return buffer

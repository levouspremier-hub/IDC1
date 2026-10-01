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

INVENTORY_OBSERVATION_VERSION = "terminal-state-observation-v1"
INVENTORY_OBSERVATION_FIELDS = ("current_soc", "target_soc", "episode_remaining_fraction")
ORIGINAL_REWARD_VERSION = "original-env-reward-v1"
COMMON_SGD_REWARD_VERSION = "common-sgd-degradation-v1"


class CorrectorWrapper(gym.Wrapper):
    def __init__(self, env, *, corrector_time_limit_s: float):
        """`corrector_time_limit_s` 必须显式给出（单次投影调用的全局预算，秒）。

        不提供隐式默认值：运行时预算决策必须由调用方显式作出。
        """
        super().__init__(env)
        if not (isinstance(corrector_time_limit_s, (int, float)) and corrector_time_limit_s > 0.0):
            raise ValueError(
                f"corrector_time_limit_s 必须为显式正数，实际 {corrector_time_limit_s!r}"
            )
        self.corrector_time_limit_s = float(corrector_time_limit_s)
        self.reward_version = getattr(
            env.unwrapped, "terminal_inventory_reward_version", ORIGINAL_REWARD_VERSION)
        if self.reward_version not in (ORIGINAL_REWARD_VERSION, COMMON_SGD_REWARD_VERSION):
            raise ValueError("unknown inventory reward semantics")
        if self.reward_version == COMMON_SGD_REWARD_VERSION:
            if not bool(getattr(env.unwrapped, "terminal_inventory_enabled", False)):
                raise ValueError("inventory reward semantics require terminal inventory contract")
            if env.unwrapped.cost_ref <= 0 or env.unwrapped.bess_degradation_cost_ref <= 0:
                raise ValueError("inventory reward semantics require positive frozen scales")
        self.inventory_observation_version = getattr(
            env.unwrapped, "terminal_inventory_observation_version", None)
        if self.inventory_observation_version is not None:
            if self.inventory_observation_version != INVENTORY_OBSERVATION_VERSION:
                raise ValueError("unknown terminal inventory observation version")
            if not bool(getattr(env.unwrapped, "terminal_inventory_enabled", False)):
                raise ValueError(
                    "inventory observation version requires terminal inventory contract")
            self.observation_space = gym.spaces.Box(
                low=np.concatenate((env.observation_space.low, np.zeros(3, dtype=np.float32))),
                high=np.concatenate((env.observation_space.high, np.ones(3, dtype=np.float32))),
                dtype=np.float32)

    def _inventory_observation(self, observation):
        if self.inventory_observation_version is None:
            return observation
        base = self.env.unwrapped
        remaining = (int(base.horizon) - int(base.current_step)) / int(base.horizon)
        state = np.asarray([base.bess_soc, base.bess_soc_target, remaining], dtype=np.float32)
        if not np.all(np.isfinite(state)) or np.any(state < 0) or np.any(state > 1):
            raise ValueError("terminal policy observation state is invalid")
        return np.concatenate((np.asarray(observation, dtype=np.float32), state))

    def policy_observation(self):
        """Read current state with the same transformation used by reset/step."""
        base = self.env.unwrapped
        observation = (np.zeros(self.env.observation_space.shape, dtype=np.float32)
                       if int(base.current_step) >= int(base.horizon)
                       else base._get_obs())
        return self._inventory_observation(observation)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        if self.inventory_observation_version is not None:
            info["policy_observation_version"] = self.inventory_observation_version
            info["inventory_observation_fields"] = INVENTORY_OBSERVATION_FIELDS
        return self._inventory_observation(observation), info

    def step(self, action):
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        n_group = self.env.model.N
        proposal = DispatchProposal(
            compute_actions=[float(x) for x in raw_action[:n_group]],
            storage_action=float(raw_action[n_group]),
        )

        snapshot = build_snapshot(self.env)
        correction = correct(
            snapshot, proposal, time_limit_s=self.corrector_time_limit_s
        )

        exec_action = np.concatenate(
            [
                np.asarray(correction.exec_compute_actions, dtype=np.float32),
                np.array(
                    [np.clip(correction.exec_storage_action, -1.0, 1.0)], dtype=np.float32
                ),
            ]
        )

        obs, reward, terminated, truncated, info = self.env.step(exec_action)
        original_reward = float(reward)
        original_degradation_reward = float(info["r_bess_degradation"])
        purchase_slope = float(self.env.reward_cost_weight / self.env.cost_ref)
        degradation_slope = float(
            self.env.reward_bess_degradation_weight / self.env.bess_degradation_cost_ref)
        if self.reward_version == COMMON_SGD_REWARD_VERSION:
            degradation_slope = purchase_slope
            info["r_bess_degradation"] = -degradation_slope * float(info["bess_degradation_cost"])
            reward = original_reward - original_degradation_reward + info["r_bess_degradation"]
        info["reward_semantics_audit"] = {
            "version": self.reward_version, "original_env_reward": original_reward,
            "original_degradation_reward": original_degradation_reward,
            "purchase_reward_per_sgd": purchase_slope,
            "degradation_reward_per_sgd": degradation_slope,
            "equivalent_degradation_weight": degradation_slope
            * float(self.env.bess_degradation_cost_ref),
        }

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
        if bool(getattr(snapshot, "terminal_inventory_enabled", False)):
            info["inventory_audit"] = {
                **correction.inventory_audit,
                "actual_energy_kwh": float(self.env.bess_energy_kWh),
                "actual_soc": float(self.env.bess_soc),
                "episode_complete": bool(terminated),
                "actual_terminal_band_gap_kwh": (max(
                    (float(self.env.bess_soc_target)
                     - float(self.env.bess_soc_final_tolerance))
                    * float(self.env.bess_capacity_kWh) - float(self.env.bess_energy_kWh),
                    float(self.env.bess_energy_kWh) - (float(self.env.bess_soc_target)
                        + float(self.env.bess_soc_final_tolerance))
                    * float(self.env.bess_capacity_kWh), 0.0) if terminated else None),
                "actual_terminal_target_gap_kwh": (
                    abs(float(self.env.bess_energy_kWh) - float(self.env.bess_soc_target
                        * self.env.bess_capacity_kWh)) if terminated else None),
                "terminal_execution_residual_kwh": (
                    float(self.env.bess_energy_kWh)
                    - correction.inventory_audit["predicted_terminal_kwh"]
                    if terminated and correction.inventory_audit.get("predicted_terminal_kwh")
                    is not None
                    else None),
            }
            info["inventory_training_diagnostics"] = {
                key: float(value) for key, value in info.items() if key.startswith("r_")
            }
            info["inventory_training_diagnostics"].update(
                charge_kw=float(info["bess_charge_power_kW"]),
                discharge_kw=float(info["bess_discharge_power_kW"]),
                actual_storage_action=(
                    float(info["bess_discharge_power_kW"])
                    / float(self.env.bess_discharge_power_max_kW)
                    - float(info["bess_charge_power_kW"])
                    / float(self.env.bess_charge_power_max_kW)),
            )
        if self.inventory_observation_version is not None:
            info["policy_observation_version"] = self.inventory_observation_version
        return self._inventory_observation(obs), reward, terminated, truncated, info

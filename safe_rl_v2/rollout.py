"""M5.1b 真实 raw/exec rollout 采集接线。

职责：把「策略采样有界 raw 动作 → （可选）修正器 → 环境」这条链真实接通，
并把每一步写入 M5.1a 冻结的 contract-v6 `RolloutBuffer`。

红线：
- raw 动作**原样**送入 `CorrectorWrapper`（corrector 开）或基础 env（corrector 关），
  **不做任何 clip / 缩放 / 覆盖**；
- `old_raw_log_prob` 由 `policy.evaluate_raw_actions()` 从**最终 raw 动作**重算，
  并在采集时与 `act()` 的返回值交叉核对（偏差超容差即报错）；
- 环境 info **缺失字段直接报错**，禁止 `info.get(key, 0.0)` 之类伪造零约束；
- `corrector_on=True` 时 `corrector_time_limit_s` 必须显式给出，禁止隐式默认值；
- `exec_action` 只审计：**不存在**任何 exec 侧 log-prob。

本模块**不实现** PPO ratio/clip、GAE、advantage、actor loss、熵项或乘子更新，
也不做任何参数更新（见 `docs/task_cards/M5.1b.md` §4）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from safe_rl.corrector_wrapper import CorrectorWrapper
from safe_rl_v2.buffer import ACTION_DIM, CONTRACT_VERSION, RolloutBuffer, Transition
from safe_rl_v2.policy import SafePPOPolicy

# 具名环境量：业务违规量（计数）/ 碳排放量（kgCO2e）/ 电费（SGD）
# 单位元数据见 `safe_rl_v2.buffer.UNIT_METADATA`；三者量纲互不相同，禁止互相顶替。
BUSINESS_VIOLATION_INFO_KEY = "sla_violation_count"
CARBON_EMISSION_INFO_KEY = "carbon_emission"
ELECTRICITY_COST_INFO_KEY = "electricity_cost"

# corrector 开启时必须完整落库的审计字段（全部来自 CorrectorWrapper.step 写入的 info）
CORRECTION_AUDIT_INFO_KEYS = (
    "raw_action",
    "exec_action",
    "correction_reason",
    "correction_detail",
    "correction_solve_time_s",
    "business_gap",
    "deadline_shortfall_work",
    "planner_backend",
    "stage_a_status",
    "stage_b_status",
    "stage_a_solve_time_s",
    "stage_b_solve_time_s",
    "stage_a_objective",
    "stage_b_objective",
    "projection_offset",
)

# `act()` 与 `evaluate_raw_actions()` 的 log-prob 允许偏差（同路径实现，实际为 0）
_LOG_PROB_TOLERANCE = 1e-4


class MissingEnvInfoError(KeyError):
    """环境 info 缺少必需字段：必须报错，不得用默认值伪造。"""


def _require_info(info: dict, key: str):
    if key not in info:
        raise MissingEnvInfoError(
            f"环境 info 缺少必需字段 {key!r}；禁止用默认值（如 0）伪造缺失约束"
        )
    return info[key]


def _json_safe(value):
    """把环境 info 里的 numpy 标量/数组转换成 JSON 安全对象；不可转换即报错。"""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    raise TypeError(
        f"环境 info 字段不可 JSON 序列化：{type(value).__name__}（correction_info 必须 JSON 安全）"
    )


def _action_bounds(action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    low = np.concatenate(
        [np.zeros(action_dim - 1, dtype=np.float64), np.array([-1.0], dtype=np.float64)]
    )
    high = np.ones(action_dim, dtype=np.float64)
    return low, high


def collect_rollout(
    env,
    policy: SafePPOPolicy,
    buffer: RolloutBuffer,
    *,
    steps: int,
    seed: int = 0,
    corrector_on: bool = False,
    corrector_time_limit_s: float | None = None,
    generator: torch.Generator | None = None,
) -> dict:
    """采集 `steps` 步真实 rollout 并写入 `buffer`；遇 terminated/truncated 提前停止。

    `generator` 为采样 RNG：给定时策略采样只消耗该 generator，**不得**由本函数
    重新播种全局 Torch RNG（调用方的随机性归调用方所有）。

    返回本次采集的统计（供 probe 与验收记录使用），**不包含任何训练结果**。
    """
    corrector_time_limit: float | None
    if corrector_on:
        if corrector_time_limit_s is None:
            raise ValueError(
                "corrector_on=True 时必须显式给出 corrector_time_limit_s（禁止隐式默认值）"
            )
        if not (isinstance(corrector_time_limit_s, (int, float)) and corrector_time_limit_s > 0.0):
            raise ValueError(
                f"corrector_time_limit_s 必须为显式正数，实际 {corrector_time_limit_s!r}"
            )
        corrector_time_limit = float(corrector_time_limit_s)
        target_env = CorrectorWrapper(env, corrector_time_limit_s=corrector_time_limit)
    else:
        if corrector_time_limit_s is not None:
            raise ValueError(
                "corrector_on=False 时不得给出 corrector_time_limit_s："
                "传入修正预算会让调用方误以为修正器已生效"
            )
        corrector_time_limit = None
        target_env = env

    if not isinstance(steps, int) or steps <= 0:
        raise ValueError(f"steps 必须为正整数，实际 {steps!r}")

    action_dim = int(policy.action_dim)
    if action_dim != ACTION_DIM:
        raise ValueError(f"策略动作维度必须为 {ACTION_DIM}，实际 {action_dim}")
    low, high = _action_bounds(action_dim)

    obs, _ = target_env.reset(seed=seed)
    stats: dict[str, Any] = {
        "corrector_on": bool(corrector_on),
        "corrector_time_limit_s": corrector_time_limit,
        "steps_requested": int(steps),
        "transitions": 0,
        "raw_exec_difference_count": 0,
        "terminated_count": 0,
        "truncated_count": 0,
        "contract_version": CONTRACT_VERSION,
        "action_dim": action_dim,
        # RNG 溯源：环境种子 + 策略采样 RNG 来源（供训练入口与 probe 记录）
        "env_seed": int(seed),
        "policy_rng_source": (
            "explicit_generator" if generator is not None else "global_torch_rng"
        ),
    }
    inventory_enabled = bool(getattr(env.unwrapped, "terminal_inventory_enabled", False))
    inventory_physical_rows: list[dict] = []
    inventory_base_unserved_steps = 0

    for _ in range(steps):
        obs_arr = np.asarray(obs, dtype=np.float32)
        obs_t = torch.as_tensor(obs_arr)

        raw_t, act_log_prob, _ = policy.act(obs_t, generator=generator)
        raw_action = raw_t.detach().numpy().astype(np.float32).copy()  # 原样，不 clip

        if raw_action.shape != (action_dim,):
            raise ValueError(f"策略返回的 raw_action 形状错误：{raw_action.shape}")
        if not np.all(np.isfinite(raw_action)):
            raise ValueError("策略返回的 raw_action 含非有限数值")
        if np.any(raw_action < low) or np.any(raw_action > high):
            raise ValueError(
                f"策略返回的 raw_action 越界（不得靠 clip 掩盖）：{raw_action}"
            )

        # 概率以**最终 raw 动作**为准；与 act() 交叉核对，任何偏差都是契约违例
        log_prob = float(policy.evaluate_raw_actions(obs_t, torch.as_tensor(raw_action)).detach())
        act_log_prob_f = float(act_log_prob.detach())
        if abs(log_prob - act_log_prob_f) > _LOG_PROB_TOLERANCE:
            raise RuntimeError(
                "old_raw_log_prob 与 act() 返回值不一致："
                f"{log_prob} vs {act_log_prob_f}（概率必须对应最终 raw 动作）"
            )

        next_obs, reward, terminated, truncated, info = target_env.step(raw_action)

        business_violation = _require_info(info, BUSINESS_VIOLATION_INFO_KEY)
        carbon_emission = _require_info(info, CARBON_EMISSION_INFO_KEY)
        electricity_cost = _require_info(info, ELECTRICITY_COST_INFO_KEY)

        if corrector_on:
            correction_info = {"corrector_on": True}
            for key in CORRECTION_AUDIT_INFO_KEYS:
                correction_info[key] = _json_safe(_require_info(info, key))
            if inventory_enabled:
                for key in ("inventory_audit", "inventory_training_diagnostics"):
                    correction_info[key] = _json_safe(_require_info(info, key))
            exec_action = np.asarray(
                _require_info(info, "exec_action"), dtype=np.float32
            ).reshape(-1).copy()
            if exec_action.shape != (action_dim,):
                raise ValueError(f"exec_action 形状错误：{exec_action.shape}")
        else:
            correction_info = {"corrector_on": False}
            exec_action = raw_action.copy()  # 无修正器：exec 即 raw，逐元素相同

        buffer.add(
            Transition(
                observation=obs_arr,
                next_observation=np.asarray(next_obs, dtype=np.float32),
                raw_action=raw_action,
                old_raw_log_prob=log_prob,
                exec_action=exec_action,
                reward=float(reward),
                business_cost=float(_json_safe(business_violation)),
                carbon_cost=float(_json_safe(carbon_emission)),
                electricity_cost_sgd=float(_json_safe(electricity_cost)),
                terminated=terminated,
                truncated=truncated,
                correction_info=correction_info,
            )
        )

        stats["transitions"] += 1
        if inventory_enabled:
            from evaluation.metrics import check_physical_step, step_power_from_info
            unwrapped = env.unwrapped
            inventory_base_unserved_steps += int(info["unserved_base_load_kW"] > 1e-6)
            inventory_physical_rows.append(check_physical_step(
                step_power_from_info(info), access_limit_kw=unwrapped.access_limit_kw,
                soc_kwh=info["bess_energy_kWh"],
                soc_min_kwh=unwrapped.bess_soc_min * unwrapped.bess_capacity_kWh,
                soc_max_kwh=unwrapped.bess_soc_max * unwrapped.bess_capacity_kWh))
        if not np.array_equal(raw_action, exec_action):
            stats["raw_exec_difference_count"] += 1
        if terminated:
            stats["terminated_count"] += 1
        if truncated:
            stats["truncated_count"] += 1

        obs = next_obs
        if terminated or truncated:
            break

    if inventory_enabled:
        from evaluation.adapter import qualify_service, violations_in
        from evaluation.metrics import aggregate_physical, classify_tasks
        from evaluation.service_standard import FROZEN_PROJECT_SERVICE_STANDARD
        unwrapped = env.unwrapped
        service, _ = classify_tasks(
            unwrapped.tasks, horizon=unwrapped.horizon,
            non_interruptible_interruptions=unwrapped.total_non_interruptible_interruption_count)
        physical = aggregate_physical(
            inventory_physical_rows, base_load_unserved_steps=inventory_base_unserved_steps)
        qualified, note = qualify_service(service, FROZEN_PROJECT_SERVICE_STANDARD, physical)
        gap = abs(unwrapped.bess_energy_kWh - unwrapped.bess_soc_target
                  * unwrapped.bess_capacity_kWh)
        stats["inventory_episode"] = {
            "service_qualified": qualified, "service_note": note,
            "physical_violation_count": len(violations_in(physical)),
            "episode_complete": bool(stats["terminated_count"] == 1),
            "final_soc": unwrapped.bess_soc, "target_gap_kwh": gap,
            "target_qualified": gap <= 1e-6,
            "inventory_qualified": abs(unwrapped.bess_soc - unwrapped.bess_soc_target)
                <= unwrapped.bess_soc_final_tolerance,
            "charge_kwh": unwrapped.total_bess_charge_kWh,
            "discharge_kwh": unwrapped.total_bess_discharge_kWh,
        }
    return stats

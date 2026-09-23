"""M1.3g-f-c-f：**三批** formal PPO 更新连续性 probe。

在**三个**互不相同的合法 train origins（本地 48 / 96 / 144）上对照：

```text
A 连续：      batch1(48) → batch2(96) → batch3(144)
B 两批+恢复： batch1(48) → batch2(96) → 保存批次边界 checkpoint
              → 调用方构造的**全新** policy/optimizer/Lagrangian/generator
              → 恢复 → batch3(144)
```

两条路径的**前两批是同一次执行**（故批 2 边界状态天然同源）；差别只在第三批的
起始状态 —— A 用**活**对象，B 用**从 checkpoint 恢复**的新对象。

**边界（本卡不做的事）**：不接正式训练入口、不启动正式训练或 M6、不读未来真值、
不把 `exec_action` 写入 `raw_action` 的概率记录、不声明训练有效
（`claims` 三项恒 `False`）。

**参数所有权**：两组对象（路径 A 的、以及路径 B 恢复用的**全新**对象）与全部
超参数 / 种子均由**调用方**提供；本模块**不构造**任何训练对象、
**不选择**任何超参数、**不播种**任何 RNG。

单批语义复用 `ppo_two_batch.run_single_batch`（同一实现 ⇒ 两条路径同源）。
"""

from __future__ import annotations

from typing import Any

import torch

from safe_rl_v2.ppo_two_batch import (
    TRAIN_CANDIDATE_STARTS as _TWO_BATCH_STARTS,
)
from safe_rl_v2.ppo_two_batch import (
    deep_snapshot,
    run_single_batch,
    save_two_batch_checkpoint,
)
from safe_rl_v2.ppo_two_batch import resume_two_batch_checkpoint as _resume_checkpoint

__all__ = [
    "THIRD_CANDIDATE_START",
    "TRANSITION_FIELDS",
    "run_three_batch_probe",
    "snapshot_objects",
    "three_batch_starts",
]

# 第三个 origin（前两个沿用两批 probe 的已验证取值）。
THIRD_CANDIDATE_START = "2024-01-04T00:00:00+08:00"

CLAIMS = {"trained": False, "performance_evaluated": False, "convergence_claimed": False}

# 逐字段对照的 Transition 契约字段（顺序即证据顺序）。
TRANSITION_FIELDS: tuple[str, ...] = (
    "observation",
    "next_observation",
    "raw_action",
    "old_raw_log_prob",
    "exec_action",
    "reward",
    "business_cost",
    "carbon_cost",
    "electricity_cost_sgd",
    "terminated",
    "truncated",
    "correction_info",
    "contract_version",
)


def three_batch_starts() -> tuple[str, str, str]:
    """三个**互不相同**的合法 train origin。"""
    starts = (_TWO_BATCH_STARTS[0], _TWO_BATCH_STARTS[1], THIRD_CANDIDATE_START)
    if len(set(starts)) != 3:
        raise RuntimeError(f"三个 origin 必须互不相同，实际 {starts}")
    return starts


def snapshot_objects(policy: Any, optimizer: Any, lagrangian: Any,
                     generator: torch.Generator) -> dict:
    """四个对象的**完整**状态快照（**独立**于活对象，可直接用于逐项精确比较）。"""
    return deep_snapshot({
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lagrangian": lagrangian.state_dict(),
        "generator": generator.get_state(),
    })


def run_three_batch_probe(
    policy: Any,
    optimizer: Any,
    lagrangian: Any,
    generator: torch.Generator,
    resume_objects: tuple[Any, Any, Any, Any],
    *,
    checkpoint_path: Any,
    clip_epsilon: float,
    gamma: float,
    lam: float,
    steps: int,
    horizon: int,
    forecast_cutoff: int,
    delta_t_hours: float,
    env_seed: int,
    env_seed_kwargs: dict[str, int],
    obs_dim: int,
    action_dim: int,
    starts: tuple[str, str, str] | None = None,
) -> dict[str, Any]:
    """跑路径 A 与路径 B，返回两者证据。

    `resume_objects` 是**调用方构造**的全新 `(policy, optimizer, lagrangian,
    generator)`，仅供路径 B 恢复使用；本模块不构造它们。
    `checkpoint_path` 亦由调用方给出（测试一律写 `tmp_path`）。

    返回的 `policy` / `optimizer` / `lagrangian` / `generator` 是**路径 A** 的对象。
    """
    starts = three_batch_starts() if starts is None else starts
    if len(starts) != 3 or len(set(starts)) != 3:
        raise ValueError(f"必须给出三个互不相同的 origin，实际 {starts}")
    r_policy, r_optimizer, r_lagrangian, r_generator = resume_objects

    def _one(objects, start: str, index: int) -> dict[str, Any]:
        """单批（显式 kwargs；两条路径共用同一实现）。"""
        batch_policy, batch_optimizer, batch_lagrangian, batch_generator = objects
        return run_single_batch(
            batch_policy, batch_optimizer, batch_lagrangian, batch_generator,
            start=start, index=index, clip_epsilon=clip_epsilon, gamma=gamma,
            lam=lam, steps=steps, horizon=horizon, forecast_cutoff=forecast_cutoff,
            delta_t_hours=delta_t_hours, env_seed=env_seed,
            env_seed_kwargs=env_seed_kwargs)

    path_a = (policy, optimizer, lagrangian, generator)

    # --- 路径 A：连续三批（同一组调用方对象）-------------------------------
    batch_a1 = _one(path_a, starts[0], 0)
    batch_a2 = _one(path_a, starts[1], 1)
    # **批 3 之前**取边界快照（独立深快照；随后的批 3 不得影响它）
    boundary_a = snapshot_objects(policy, optimizer, lagrangian, generator)
    # 同一时点保存批次边界 checkpoint（供路径 B 恢复）
    save_two_batch_checkpoint(
        checkpoint_path, policy=policy, optimizer=optimizer, lagrangian=lagrangian,
        generator=generator, next_start=starts[2], obs_dim=obs_dim,
        action_dim=action_dim)
    batch_a3 = _one(path_a, starts[2], 2)
    final_a = snapshot_objects(policy, optimizer, lagrangian, generator)

    # --- 路径 B：全新对象 → 恢复 → 批 3 ------------------------------------
    resumed = _resume_checkpoint(
        checkpoint_path, policy=r_policy, optimizer=r_optimizer,
        lagrangian=r_lagrangian, generator=r_generator,
        expected_obs_dim=obs_dim, expected_action_dim=action_dim)
    boundary_b = snapshot_objects(r_policy, r_optimizer, r_lagrangian, r_generator)
    batch_b3 = _one(resume_objects, resumed["next_start"], 2)
    final_b = snapshot_objects(r_policy, r_optimizer, r_lagrangian, r_generator)

    return {
        "entry": "safe_rl_v2.ppo_three_batch.run_three_batch_probe",
        "probe_only": True,
        "starts": tuple(starts),
        "obs_dim": obs_dim,
        # 调用方对象原样回传（供**同一性**断言，而非仅相等）
        "policy": policy,
        "optimizer": optimizer,
        "lagrangian": lagrangian,
        "generator": generator,
        "resume_objects": resume_objects,
        "batches": (batch_a1, batch_a2, batch_a3),
        "third_batch_resumed": batch_b3,
        "boundary_a": boundary_a,
        "boundary_b": boundary_b,
        "final_a": final_a,
        "final_b": final_b,
        "claims": dict(CLAIMS),
    }

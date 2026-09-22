"""M1.3g-f-c-d：双批次 formal PPO 更新**连通性探针**。

用**两个不同**的 formal episode（从 verified train v5 candidate origins
**48** / **96** 推导），验证「采集 → 更新」可以**连续**跑通两批：

```text
同一 policy / optimizer / Lagrangian / **连续**显式采样 RNG
  batch 1: collect_rollout(env@origin 48) → single_ppo_update
  batch 2: collect_rollout(env@origin 96) → single_ppo_update
```

**边界（本卡不做的事）**：不写 checkpoint、不接正式训练入口、
**不做性能或收敛评价**、不以 probe 声称训练有效（`claims` 三项恒 `False`）；
不修改 `train.py` / env / 冻结资产 / 发布产物 v1。

**不得**为「制造一致」而每批重建 policy / optimizer / Lagrangian，
或重播种采样 RNG —— 本模块按构造即共用一份，且把 RNG 的**前进**与策略状态的
**变化**记入证据，供测试独立核对。
"""

from __future__ import annotations

import hashlib
import importlib
from typing import Any

import numpy as np
import torch

from safe_rl_v2.ppo_update import single_ppo_update

__all__ = [
    "TRAIN_CANDIDATE_STARTS",
    "build_formal_env",
    "run_two_batch_probe",
]

# verified train v5 candidate origins（本地行 48 / 96）。
TRAIN_CANDIDATE_STARTS: tuple[str, ...] = (
    "2024-01-02T00:00:00+08:00",
    "2024-01-03T00:00:00+08:00",
)

CLAIMS = {"trained": False, "performance_evaluated": False, "convergence_claimed": False}


def _digest_arrays(arrays: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.float32)).tobytes())
    return h.hexdigest()


def _policy_state_digest(policy: Any) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _generator_state_digest(generator: torch.Generator) -> str:
    return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()


def build_formal_env(start: str, *, horizon: int, forecast_cutoff: int,
                     delta_t_hours: float, env_seed_kwargs: dict[str, int]):
    """由 **verified** train v5 candidate origin 构造一个 formal env。"""
    from scenario.arrival_mapper import load_verified_mapper_chain

    injection_module = importlib.import_module("scenario.env_injection")
    env_cls = importlib.import_module("envs.idc_price_env").IDCPriceEnv20D

    # 先走唯一 verified 链（含 mapper 参数 manifest）再构造注入
    load_verified_mapper_chain("train")
    injection = injection_module.build_verified_formal_env_injection(
        "train", start=start, horizon=horizon, forecast_cutoff=forecast_cutoff)
    env = env_cls(horizon=horizon, delta_t_hours=delta_t_hours,
                  formal_injection=injection, **env_seed_kwargs)
    return env, injection


def run_two_batch_probe(
    *,
    clip_epsilon: float,
    gamma: float,
    lam: float,
    steps: int,
    horizon: int,
    forecast_cutoff: int,
    delta_t_hours: float,
    seed: int,
    policy_seed: int,
    env_seed_kwargs: dict[str, int],
    starts: tuple[str, ...] = TRAIN_CANDIDATE_STARTS,
) -> dict[str, Any]:
    """跑两个 formal 批次（每批 `collect_rollout → single_ppo_update`）并返回证据。

    所有超参数由调用方**显式给出**；本函数不选择、不冻结任何超参数。
    """
    from safe_rl_v2.buffer import RolloutBuffer
    from safe_rl_v2.lagrangian import (
        UNIT_KG_CO2E,
        UNIT_VIOLATION_TASK_STEPS,
        ConstraintSpec,
        Lagrangian,
    )
    from safe_rl_v2.policy import SafePPOPolicy
    from safe_rl_v2.rollout import collect_rollout

    if len(starts) != 2:
        raise ValueError(f"本 probe 恰好需要两个 origin，实际 {len(starts)}")

    # --- 共用一份：policy / optimizer / Lagrangian / **连续** generator --------
    first_env, _ = build_formal_env(
        starts[0], horizon=horizon, forecast_cutoff=forecast_cutoff,
        delta_t_hours=delta_t_hours, env_seed_kwargs=env_seed_kwargs)
    obs_dim = int(first_env.obs_dim)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(policy_seed)
        policy = SafePPOPolicy(obs_dim=obs_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    lagrangian = Lagrangian((
        ConstraintSpec(name="business", budget=5.0, unit=UNIT_VIOLATION_TASK_STEPS,
                       learning_rate=0.01, max_multiplier=100.0),
        ConstraintSpec(name="carbon", budget=3.0, unit=UNIT_KG_CO2E,
                       learning_rate=0.01, max_multiplier=100.0),
    ))
    # **同一** generator 贯穿两批：不重播种（重播种会掩盖 RNG 未前进的问题）
    generator = torch.Generator()
    generator.manual_seed(seed)

    batches: list[dict[str, Any]] = []
    batch_digests: list[str] = []

    for index, start in enumerate(starts):
        env, injection = build_formal_env(
            start, horizon=horizon, forecast_cutoff=forecast_cutoff,
            delta_t_hours=delta_t_hours, env_seed_kwargs=env_seed_kwargs)

        # 采集时刻的策略 / RNG 状态快照（供测试**独立**重算）
        state_at_collection = {k: v.detach().clone()
                               for k, v in policy.state_dict().items()}
        generator_state_at_collection = _generator_state_digest(generator)

        buffer = RolloutBuffer()
        stats = collect_rollout(env, policy, buffer, steps=steps, seed=seed,
                                generator=generator)
        if stats["transitions"] == 0:
            raise RuntimeError(f"批 {index} 未采集到任何 transition")

        transitions = list(buffer.transitions)
        observations = np.stack([np.asarray(t.observation, dtype=np.float32)
                                 for t in transitions])
        raw_actions = np.stack([np.asarray(t.raw_action, dtype=np.float32)
                                for t in transitions])
        old_lp = np.asarray([float(t.old_raw_log_prob) for t in transitions],
                            dtype=np.float32)
        rewards = [float(t.reward) for t in transitions]

        batch_digest = _digest_arrays([observations, raw_actions, old_lp])
        batch_digests.append(batch_digest)

        multipliers_pre = {k: float(v) for k, v in lagrangian.multipliers().items()}
        update = single_ppo_update(
            policy, optimizer, lagrangian, buffer,
            clip_epsilon=clip_epsilon, gamma=gamma, lam=lam)

        batches.append({
            "index": index,
            "start": start,
            "origin": int(injection.local_origin) if hasattr(injection, "local_origin")
            else None,
            "formal": bool(env.formal),
            "transitions": int(stats["transitions"]),
            "observations": observations,
            "raw_actions": raw_actions,
            "old_raw_log_prob": old_lp,
            "rewards": rewards,
            "batch_digest": batch_digest,
            "policy_state_digest": _digest_arrays(
                [v.numpy() for _, v in sorted(state_at_collection.items())]),
            "policy_state_dict": state_at_collection,
            "generator_state_digest": generator_state_at_collection,
            "loss_total": float(update["loss_total"]),
            "actor_loss": float(update["actor_loss"]),
            "param_delta_norm": float(update["param_delta_norm"]),
            "multipliers_pre_update": multipliers_pre,
            "multipliers_post_update": dict(update["multipliers_post_update"]),
            "optimizer_steps_cumulative": index + 1,
            "lagrangian_updates_cumulative": int(update["lagrangian_updates_after"]),
        })

    # origin 需要从 injection 的全局映射取，统一回填一次
    for batch, start in zip(batches, starts, strict=True):
        from scenario.b6_split_manifests import local_origin_from_start

        batch["origin"] = local_origin_from_start("train", start)

    final_digest = _policy_state_digest(policy)
    rerun_payload = "|".join(batch_digests) + "|" + final_digest + "|" + "|".join(
        f"{k}={v}" for b in batches for k, v in sorted(b["multipliers_post_update"].items()))

    return {
        "entry": "safe_rl_v2.ppo_two_batch.run_two_batch_probe",
        "probe_only": True,
        "obs_dim": obs_dim,
        "batches": batches,
        "batch_digests": batch_digests,
        "total_transitions": int(sum(b["transitions"] for b in batches)),
        "optimizer_steps_total": int(len(batches)),
        "lagrangian_updates_total": int(batches[-1]["lagrangian_updates_cumulative"]),
        "final_policy_state_digest": final_digest,
        "rerun_digest": hashlib.sha256(rerun_payload.encode()).hexdigest(),
        # 结构性事实：本函数**按构造**只建一份 policy / optimizer / Lagrangian，
        # 且从不重播种 generator；实质核对由测试用 digest 完成（RNG 前进、
        # 策略状态在两批之间变化）。
        "shared_policy": True,
        "shared_optimizer": True,
        "shared_lagrangian": True,
        "generator_reseeded_between_batches": False,
        "claims": dict(CLAIMS),
    }

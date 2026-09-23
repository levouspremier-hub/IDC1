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

**参数所有权（M1.3g-f-c-d-R1）**：`policy` / `optimizer` / `Lagrangian` /
**显式采样 generator** 全部由**调用方**构造并传入；本模块**只负责在两批之间
持续使用它们**，内部**不得**选择 Adam 学习率、乘子预算 / 学习率 / 上限、
策略隐藏层尺寸，也**不得**播种 RNG。故调用方的训练配置不会在探测过程中被
默默冻结或被误读为「正式训练参数」；RNG 的**前进**与策略状态的**变化**被记入
证据，供测试独立核对。
"""

from __future__ import annotations

import copy
import hashlib
import importlib
from typing import Any

import numpy as np
import torch

from safe_rl_v2.ppo_update import single_ppo_update

__all__ = [
    "TRAIN_CANDIDATE_STARTS",
    "build_formal_env",
    "deep_snapshot",
    "resume_two_batch_checkpoint",
    "run_single_batch",
    "run_two_batch_probe",
    "save_two_batch_checkpoint",
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


def run_single_batch(
    policy: Any,
    optimizer: Any,
    lagrangian: Any,
    generator: torch.Generator,
    *,
    start: str,
    clip_epsilon: float,
    gamma: float,
    lam: float,
    steps: int,
    horizon: int,
    forecast_cutoff: int,
    delta_t_hours: float,
    env_seed: int,
    env_seed_kwargs: dict[str, int],
    index: int = 0,
) -> dict[str, Any]:
    """**单批**编排：`collect_rollout → single_ppo_update`，返回该批证据。

    `policy` / `optimizer` / `lagrangian` / `generator` 由**调用方**提供；
    本函数只使用它们，**不构造**、**不播种**。两批入口与 checkpoint 适配
    （M1.3g-f-c-e）都调用本函数，故「连续」与「恢复后」的单批语义**同源**。
    """
    from safe_rl_v2.buffer import RolloutBuffer
    from safe_rl_v2.rollout import collect_rollout

    env, injection = build_formal_env(
        start, horizon=horizon, forecast_cutoff=forecast_cutoff,
        delta_t_hours=delta_t_hours, env_seed_kwargs=env_seed_kwargs)

    # 采集时刻的策略 / RNG 状态快照（供测试**独立**重算）
    state_at_collection = {k: v.detach().clone()
                           for k, v in policy.state_dict().items()}
    generator_state_at_collection = _generator_state_digest(generator)

    buffer = RolloutBuffer()
    stats = collect_rollout(env, policy, buffer, steps=steps, seed=env_seed,
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

    multipliers_pre = {k: float(v) for k, v in lagrangian.multipliers().items()}
    update = single_ppo_update(
        policy, optimizer, lagrangian, buffer,
        clip_epsilon=clip_epsilon, gamma=gamma, lam=lam)

    # origin：由 verified 映射取全局行号
    from scenario.b6_split_manifests import local_origin_from_start

    return {
        "index": index,
        "start": start,
        "origin": local_origin_from_start("train", start),
        # **M1.3g-f-c-f 最小扩展**：暴露**完整** transition 记录，供逐字段精确对照
        # （observation / next_observation / raw_action / old_raw_log_prob /
        # exec_action / reward / 三类 cost / terminated / truncated /
        # correction_info / contract_version）。**不改变**既有键
        # `transitions`（它仍是**计数**）。
        "transition_records": transitions,
        "formal": bool(env.formal),
        "injection_local_origin": getattr(injection, "local_origin", None),
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
        "generator_state_digest_after": _generator_state_digest(generator),
        "loss_total": float(update["loss_total"]),
        "actor_loss": float(update["actor_loss"]),
        "param_delta_norm": float(update["param_delta_norm"]),
        "multipliers_pre_update": multipliers_pre,
        "multipliers_post_update": dict(update["multipliers_post_update"]),
        "optimizer_steps_cumulative": _optimizer_steps(optimizer),
        "lagrangian_updates_cumulative": int(update["lagrangian_updates_after"]),
    }


def deep_snapshot(value: Any) -> Any:
    """**递归**深快照：张量 `detach().clone()`，容器重建，其余 `deepcopy`。

    ⚠️ **必须的语义**（M1.3g-f-c-e-R2 的教训）：`torch.optim.Optimizer.state_dict()`
    返回**新的外层 dict**，但其 `state` 里的 `exp_avg` / `exp_avg_sq` / `step` 是
    **活张量的同一对象**，会被后续 `step()` **原位改写**。若快照直接持有它们，
    「某批次**之前**」的对照会退化成「之后 vs 之后」——**空洞**。
    """
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: deep_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(deep_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _optimizer_steps(optimizer: Any) -> int:
    """optimizer **真实**已执行的步数（取自 Adam 状态，而非调用次数）。"""
    steps = [float(v["step"]) for v in optimizer.state.values()
             if isinstance(v, dict) and "step" in v]
    return int(max(steps)) if steps else 0


def run_two_batch_probe(
    policy: Any,
    optimizer: Any,
    lagrangian: Any,
    generator: torch.Generator,
    *,
    clip_epsilon: float,
    gamma: float,
    lam: float,
    steps: int,
    horizon: int,
    forecast_cutoff: int,
    delta_t_hours: float,
    env_seed: int,
    env_seed_kwargs: dict[str, int],
    starts: tuple[str, ...] = TRAIN_CANDIDATE_STARTS,
) -> dict[str, Any]:
    """跑两个 formal 批次（每批 `collect_rollout → single_ppo_update`）并返回证据。

    `policy` / `optimizer` / `lagrangian` / `generator` 由**调用方**提供并贯穿两批；
    本函数只做编排与证据记录，**不构造**任何上述对象，也**不**播种 `generator`。
    其余超参数（`clip_epsilon` / `gamma` / `lam` / 步数 / horizon 等）同样由调用方
    显式给出，本函数不选择、不冻结。
    """
    if len(starts) != 2:
        raise ValueError(f"本 probe 恰好需要两个 origin，实际 {len(starts)}")

    # **调用方**提供对象；本函数不构造、不播种。obs_dim 由环境推出。
    first_env, _ = build_formal_env(
        starts[0], horizon=horizon, forecast_cutoff=forecast_cutoff,
        delta_t_hours=delta_t_hours, env_seed_kwargs=env_seed_kwargs)
    obs_dim = int(first_env.obs_dim)

    batches = [
        run_single_batch(policy, optimizer, lagrangian, generator, start=start,
                         clip_epsilon=clip_epsilon, gamma=gamma, lam=lam,
                         steps=steps, horizon=horizon,
                         forecast_cutoff=forecast_cutoff,
                         delta_t_hours=delta_t_hours, env_seed=env_seed,
                         env_seed_kwargs=env_seed_kwargs, index=index)
        for index, start in enumerate(starts)
    ]
    batch_digests = [b["batch_digest"] for b in batches]

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
        # 调用方对象原样回传（供测试断言**同一性**，而非仅相等）
        "policy": policy,
        "optimizer": optimizer,
        "lagrangian": lagrangian,
        "generator": generator,
        "batches": batches,
        "batch_digests": batch_digests,
        "total_transitions": int(sum(b["transitions"] for b in batches)),
        "optimizer_steps_total": int(batches[-1]["optimizer_steps_cumulative"]),
        "lagrangian_updates_total": int(batches[-1]["lagrangian_updates_cumulative"]),
        "final_policy_state_digest": final_digest,
        "rerun_digest": hashlib.sha256(rerun_payload.encode()).hexdigest(),
        # 结构性事实：调用方只传一份对象，本函数在两批之间**持续使用**同一份，
        # 且从不重播种 generator；实质核对由测试用 digest 完成（RNG 前进、
        # 策略状态在两批之间变化）与对象**同一性**断言完成。
        "shared_policy": True,
        "shared_optimizer": True,
        "shared_lagrangian": True,
        "generator_reseeded_between_batches": False,
        "claims": dict(CLAIMS),
    }


# --- M1.3g-f-c-e：批次边界的 checkpoint / resume -----------------------------
#
# 复用现有 `checkpointing.VersionedCheckpoint`：
# `contract_version_id` / `action_dim` / `obs_dim` / `schema_hash` 由它校验。
# 本模块只补「把**四个对象的状态** + 下一 origin 放进 state」与「写回新对象」。
#
# **只**证明**批次边界**恢复；不宣称中途恢复。

CHECKPOINT_STATE_KEYS: tuple[str, ...] = (
    "policy", "optimizer", "lagrangian", "generator", "next_start",
)
# 本 checkpoint **格式**的标识（确定性常量，不含时间戳）。
# 张量形状不在此处编码：obs / action 维度由 `VersionedCheckpoint` 的
# `obs_dim` / `action_dim` 单独校验，两者合起来即覆盖格式与形状。
CHECKPOINT_SCHEMA = "m1.3g-f-c-e-two-batch-resume-v1"


def save_two_batch_checkpoint(
    path: Any,
    *,
    policy: Any,
    optimizer: Any,
    lagrangian: Any,
    generator: torch.Generator,
    next_start: str,
    obs_dim: int,
    action_dim: int,
    contract_version_id: str | None = None,
    schema_hash: str | None = None,
    code_revision: str = "",
) -> dict[str, Any]:
    """把四个对象的状态 + **下一 origin** 存成版本化 checkpoint。

    仅供**测试用适配**；调用方负责提供路径（测试一律写 `tmp_path`）。
    """
    from checkpointing import CURRENT_CONTRACT_VERSION, VersionedCheckpoint

    state = {
        "policy": {k: v.detach().clone() for k, v in policy.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "lagrangian": lagrangian.state_dict(),
        "generator": generator.get_state(),
        "next_start": str(next_start),
    }
    checkpoint = VersionedCheckpoint(
        contract_version_id=contract_version_id or CURRENT_CONTRACT_VERSION,
        action_dim=int(action_dim),
        obs_dim=int(obs_dim),
        schema_hash=schema_hash or CHECKPOINT_SCHEMA,
        code_revision=str(code_revision),
        state=state,
    )
    checkpoint.save(path)
    return {"path": str(path), "schema_hash": checkpoint.schema_hash,
            "next_start": state["next_start"]}


def resume_two_batch_checkpoint(
    path: Any,
    *,
    policy: Any,
    optimizer: Any,
    lagrangian: Any,
    generator: torch.Generator,
    expected_obs_dim: int,
    expected_action_dim: int,
    expected_schema_hash: str | None = None,
) -> dict[str, Any]:
    """把 checkpoint 的状态**写回调用方提供的新对象**，返回 `next_start`。

    版本 / 动作维度 / obs 维度 / schema 由 `VersionedCheckpoint.load` 校验；
    状态字段缺失由本函数**明确拒绝**（不接受无版本或不完整状态）。
    """
    from checkpointing import VersionedCheckpoint

    expected_schema = (expected_schema_hash
                       if expected_schema_hash is not None
                       else CHECKPOINT_SCHEMA)
    checkpoint = VersionedCheckpoint.load(
        path, expected_action_dim=int(expected_action_dim),
        expected_obs_dim=int(expected_obs_dim), expected_schema_hash=expected_schema)

    state = checkpoint.state
    if not isinstance(state, dict) or set(state) != set(CHECKPOINT_STATE_KEYS):
        missing = sorted(set(CHECKPOINT_STATE_KEYS) - set(state or {}))
        raise ValueError(
            f"checkpoint 状态字段必须精确等于 {list(CHECKPOINT_STATE_KEYS)}；"
            f"缺少={missing}")

    policy.load_state_dict(state["policy"])
    optimizer.load_state_dict(state["optimizer"])
    lagrangian.load_state_dict(state["lagrangian"])
    generator.set_state(state["generator"])
    return {"next_start": str(state["next_start"]),
            "schema_hash": checkpoint.schema_hash}

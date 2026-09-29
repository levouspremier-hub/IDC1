"""M1.3g-f-c-j：**冻结配置驱动**的训练闭环（共享实现 + 批次边界训练恢复）。

本模块只做三件事：

1. **读取并核对** `configs/training/idc_training_config_v1.json`（唯一数值来源）；
2. 用它构造 policy / Adam / 两个 Lagrangian 约束乘子，并跑**一批**真实训练；
3. 在**批次边界**保存 / 恢复版本化**训练恢复** checkpoint。

一批的语义（冻结配置 `sampling` / `scale` 段）：

```text
采集：episodes_per_batch(=4) 个 train episode × horizon(=48) 步 = 192 transitions
更新：epochs_per_batch(=4) × minibatches_per_epoch(=4) × minibatch_size(=48) = 16 次 Adam step
乘子：两个约束各自用**整批 192 条**信号更新**一次**（在 16 次 step 之后）
```

红线：

- 超参数**只**来自冻结配置；本模块**不**提供任何覆盖训练超参数的参数；
- `old_raw_log_prob` / 三头优势 / critic target 在**该批首次更新前**算好，
  4 个 epoch 内**固定**；新 log-prob **始终**由 `raw_action` 现算；
  `exec_action` 只用于环境执行与审计；
- 乘子**不得**按 minibatch 更新；
- 训练恢复 checkpoint 与 `checkpointing/eval_input.py` 的**评估输入**角色明确区分：
  本模块的 schema 与 `artifact_role` 都**不在** `EVAL_INPUT_ARTIFACT_ROLES` 中；
- 本模块**不**声称任何训练结果（`claims` 三项恒为 `False`），
  `training_scope` 恒为 `controlled_short_run`。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from safe_rl_v2.buffer import ACTION_DIM, CONTRACT_VERSION, RolloutBuffer
from safe_rl_v2.lagrangian import ConstraintSpec, Lagrangian
from safe_rl_v2.policy import SafePPOPolicy
from safe_rl_v2.ppo_update import (
    compute_advantage_oriented_arrays,
    minibatch_ppo_step,
)
from safe_rl_v2.rollout import collect_rollout

__all__ = [
    "BATCH_KEYS",
    "CONFIG_SCHEMA",
    "CONFIG_STATUS_FROZEN",
    "CONTROLLED_BATCHES",
    "CONTROLLED_ORIGINS_COUNT",
    "FORMAL_RESUME_ARTIFACT_ROLE",
    "FORMAL_RESUME_SCHEMA",
    "FORMAL_TRAINING_SCOPE",
    "FROZEN_CONFIG_LOGICAL_PATH",
    "RESUME_ARTIFACT_ROLE",
    "RESUME_SCHEMA",
    "TRAINING_SCOPE",
    "build_lagrangian",
    "build_optimizer",
    "build_policy",
    "build_seeded_policy",
    "config_provenance_summary",
    "build_train_env",
    "controlled_origins",
    "live_asset_hash_check",
    "load_frozen_training_config",
    "load_resume_checkpoint",
    "optimizer_state_digest",
    "run_training_batch",
    "save_resume_checkpoint",
    "train_env_seeds",
    "training_source_ledger",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

FROZEN_CONFIG_LOGICAL_PATH = "configs/training/idc_training_config_v1.json"
CONFIG_SCHEMA = "idc-training-config-v1"
CONFIG_STATUS_FROZEN = "frozen"

TRAINING_SCOPE = "controlled_short_run"
CLAIMS = {"trained": False, "performance_evaluated": False, "convergence_claimed": False}

#: 训练恢复格式的 schema（**不得**与 `checkpointing.eval_input.EVAL_INPUT_SCHEMA` 混用）。
RESUME_SCHEMA = "m1.3g-f-c-j-controlled-resume-v1"
#: 训练恢复产物的角色（**不**属于 `EVAL_INPUT_ARTIFACT_ROLES`）。
RESUME_ARTIFACT_ROLE = "controlled_training_resume"

#: **正式训练**（M1.3g-f-c-k）的 scope / role / schema —— 与受控短跑**明确可区分**，
#: 两者不得互相冒充（受控短跑永远不能标成正式训练）。
FORMAL_TRAINING_SCOPE = "formal_training"
FORMAL_RESUME_ARTIFACT_ROLE = "formal_training_resume"
FORMAL_RESUME_SCHEMA = "m13gfck-formal-train-resume-v1"
TRAINING_SCOPES: tuple[str, ...] = (TRAINING_SCOPE, FORMAL_TRAINING_SCOPE)

#: 受控短跑用**前 12 个** train-only origin（= 3 批 × 4 episode）。
CONTROLLED_BATCHES = 3
CONTROLLED_ORIGINS_COUNT = 12

#: checkpoint 状态字段（精确集合；缺一即拒绝，不补默认值）。
BATCH_KEYS = (
    "policy", "optimizer", "lagrangian", "sampling_generator", "shuffle_generator",
    "next_batch_index", "origins", "episodes_per_batch", "steps_per_episode",
    "frozen_config", "config_summary", "training_scope", "artifact_role",
    # M1.3g-f-c-j-R1：来源账本与 code revision 必须随 checkpoint 一起可核对
    "source_ledger", "origin_provenance", "code_revision",
)


class FormalTrainLoopError(ValueError):
    """冻结配置 / 闭环 / 恢复约定被违反。"""


def load_frozen_training_config(path: str | Path | None = None) -> dict:
    """读取**冻结**训练配置并核对 schema / status；失败即明确报错。"""
    resolved = Path(path) if path is not None else REPO_ROOT / FROZEN_CONFIG_LOGICAL_PATH
    if not resolved.is_file():
        raise FormalTrainLoopError(f"冻结训练配置不存在：{resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FormalTrainLoopError(f"冻结训练配置不是合法 JSON：{error}") from error
    if not isinstance(payload, dict):
        raise FormalTrainLoopError("冻结训练配置必须是 JSON object")
    if payload.get("schema") != CONFIG_SCHEMA:
        raise FormalTrainLoopError(
            f"schema 必须等于 {CONFIG_SCHEMA!r}，实际 {payload.get('schema')!r}")
    if payload.get("status") != CONFIG_STATUS_FROZEN:
        raise FormalTrainLoopError(
            f"status 必须等于 {CONFIG_STATUS_FROZEN!r}（候选配置不得驱动训练），"
            f"实际 {payload.get('status')!r}")
    if "training" not in payload or not isinstance(payload["training"], dict):
        raise FormalTrainLoopError("冻结训练配置缺少 `training` 段")
    return payload


def config_provenance_summary(config: dict) -> dict:
    """来源摘要：版本、所依据的标定 run、资产 hash、服务标准 ID。"""
    calibration = config.get("calibration")
    decision = config.get("frozen_decision")
    if not isinstance(calibration, dict) or not isinstance(decision, dict):
        raise FormalTrainLoopError("冻结配置缺少 `calibration` / `frozen_decision` 段")
    return {
        "config_schema": config["schema"],
        "config_version": config.get("version"),
        "config_status": config["status"],
        "calibration_run_id": calibration.get("run_id"),
        "calibration_run_manifest_sha256": calibration.get("run_manifest_sha256"),
        "calibration_run_report_sha256": calibration.get("run_report_sha256"),
        "asset_hashes": dict(calibration.get("asset_hashes") or {}),
        "service_standard_id": decision.get("service_standard_id"),
        "service_standard_frozen": decision.get("service_standard_frozen"),
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }


def build_policy(config: dict, *, obs_dim: int) -> SafePPOPolicy:
    """按冻结配置构造策略网络（hidden / hidden_layers / activation / action_dim）。"""
    policy_cfg = config["training"]["policy"]
    hidden = int(policy_cfg["hidden"])
    layers = int(policy_cfg["hidden_layers"])
    if layers != 1 or str(policy_cfg["activation"]) != "Tanh":
        raise FormalTrainLoopError(
            "当前 policy 实现只支持 hidden_layers=1 且 activation=Tanh；"
            f"冻结配置给出 {layers} / {policy_cfg['activation']!r}")
    action_dim = int(policy_cfg["action_dim"])
    if action_dim != ACTION_DIM:
        raise FormalTrainLoopError(
            f"冻结配置 action_dim={action_dim} 与契约 {ACTION_DIM} 不一致")
    return SafePPOPolicy(obs_dim=int(obs_dim), action_dim=action_dim, hidden=hidden)


def build_seeded_policy(config: dict, *, obs_dim: int, seed: int) -> SafePPOPolicy:
    """**确定性**初始化策略：在 `fork_rng` 区间内以显式 `seed` 播种后构造。

    为什么需要它：`nn.Linear` 的默认初始化消耗**全局** Torch RNG，若不播种，
    两个进程会得到不同的初始权重，整条训练闭环随之不可复现（M1.3g-f-c-j 实测：
    corrector off 时两次同命令的 raw 动作摘要也不同）。

    本函数**不**重播种全局 RNG：只在受控区间内播种，退出时由 `fork_rng` 还原，
    故调用方的全局随机性归调用方所有（与 `rollout.collect_rollout` 的 RNG 纪律一致）。
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return build_policy(config, obs_dim=obs_dim)


def train_env_seeds(config: dict, *, master_seed: int) -> dict[str, int]:
    """由**本次 master seed** 加冻结配置的 `seed_offsets` 派生三种环境种子。"""
    offsets = config["training"]["scale"]["seed_offsets"]
    return {
        "task_seed": int(master_seed) + int(offsets["task"]),
        "server_seed": int(master_seed) + int(offsets["server"]),
        "forecast_seed": int(master_seed) + int(offsets["forecast"]),
    }


def build_train_env(origin: int, *, master_seed: int, config: dict):
    """按**本次 master seed** + 冻结配置构造 train formal env（M1.3g-f-c-j-R1）。

    此前训练环境经 `scripts.calibrate_training_config.build_env_for_origin` 构造，其
    `MASTER_SEED` 固定为 0 ⇒ 换 seed **不会**改变环境构造种子。本函数改为由调用方的
    master seed 派生（seed 0 与原语义**逐值相同**：0/1/300000）。

    复用既有 verified 链（mapper chain + `build_verified_formal_env_injection` + 环境类），
    不另造资产验签体系。
    """
    import importlib

    from scenario.arrival_mapper import load_verified_mapper_chain
    from scripts.calibrate_training_config import DELTA_T_HOURS, start_for_origin

    injection_module = importlib.import_module("scenario.env_injection")
    env_cls = importlib.import_module("envs.idc_price_env").IDCPriceEnv20D

    sampling = config["training"]["sampling"]
    horizon = int(sampling["horizon"])
    forecast_cutoff = int(sampling["forecast_cutoff"])
    seeds = train_env_seeds(config, master_seed=master_seed)

    load_verified_mapper_chain("train")
    injection = injection_module.build_verified_formal_env_injection(
        "train", start=start_for_origin(int(origin)), horizon=horizon,
        forecast_cutoff=forecast_cutoff)
    env = env_cls(horizon=horizon, forecast_cutoff=forecast_cutoff,
                  delta_t_hours=float(DELTA_T_HOURS), formal_injection=injection,
                  **seeds)
    return env, injection


def training_source_ledger(origin_provenance: dict[int, str]) -> dict[str, str]:
    """本次短跑**可重算**的来源账本（M1.3g-f-c-j-R1）。

    - `dependency_lock_hash`：`uv.lock` 的 sha256（重算：对该文件做 sha256）；
    - `data_hash`：复用 `evaluation.sources.canonical_source_digests()` 的现有口径
      （12 个已验签来源的 `(role, logical_path, sha256)` 规范化 JSON 之 sha256）；
    - `scenario_hash`：本次**实际使用**的 train origin 与其 formal injection
      `provenance_hash` 的规范化 JSON 之 sha256。
    """
    from evaluation.sources import canonical_source_digests

    lock = REPO_ROOT / "uv.lock"
    if not lock.is_file():
        raise FormalTrainLoopError(f"依赖锁不存在：{lock}")
    digests = canonical_source_digests()
    data_hash = hashlib.sha256(json.dumps(
        [[d.role, d.logical_path, d.sha256] for d in digests],
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    if not origin_provenance:
        raise FormalTrainLoopError("场景 hash 需要本次实际使用的 origin provenance")
    rows: list[list[Any]] = []
    for origin in sorted(origin_provenance):
        prov = str(origin_provenance[origin])
        if len(prov) != 64 or any(c not in "0123456789abcdef" for c in prov):
            raise FormalTrainLoopError(f"injection provenance_hash 非法：{prov!r}")
        rows.append([int(origin), prov])
    scenario_hash = hashlib.sha256(json.dumps(
        rows, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {
        "dependency_lock_hash": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "data_hash": data_hash,
        "scenario_hash": scenario_hash,
    }


def live_asset_hash_check(config: dict) -> dict[str, dict[str, Any]]:
    """把冻结配置记录的六项资产 hash 与 **live 文件**逐项实测比对。"""
    recorded = dict(config.get("calibration", {}).get("asset_hashes") or {})
    if not recorded:
        raise FormalTrainLoopError("冻结配置缺少 calibration.asset_hashes")
    paths = {
        "refs_v4": "configs/frozen_refs/refs_v4.json",
        "formal_split_v5_train": "data/manifest/formal_splits_v5/train.json",
        "m13g_arrival_mapper_v1": "data/manifest/m13g_arrival_mapper_v1.json",
        "env_release_v1": "configs/release/idc_formal_env_release_v1.json",
        "canonical_parquet": "data/processed/singapore_2024/half_hour.parquet",
        "exogenous_v3_parquet":
            "data/processed/singapore_2024/exogenous_drivers_v3.parquet",
    }
    out: dict[str, dict[str, Any]] = {}
    for role, logical in paths.items():
        if role not in recorded:
            raise FormalTrainLoopError(f"冻结配置缺少资产 {role!r}")
        live = hashlib.sha256((REPO_ROOT / logical).read_bytes()).hexdigest()
        out[role] = {"logical_path": logical, "recorded": str(recorded[role]),
                     "live": live, "match": bool(live == recorded[role])}
    return out


def build_optimizer(config: dict, policy: SafePPOPolicy) -> torch.optim.Adam:
    """按冻结配置构造 Adam（lr / betas / eps / weight_decay）。"""
    opt = config["training"]["optimizer"]
    if str(opt["name"]) != "Adam":
        raise FormalTrainLoopError(f"冻结配置优化器必须是 Adam，实际 {opt['name']!r}")
    if float(opt["weight_decay"]) != 0.0 or str(opt["schedule"]) != "none":
        raise FormalTrainLoopError("冻结配置要求 weight_decay=0.0 且 schedule=none")
    return torch.optim.Adam(
        policy.parameters(), lr=float(opt["lr"]),
        betas=(float(opt["betas"][0]), float(opt["betas"][1])),
        eps=float(opt["eps"]), weight_decay=float(opt["weight_decay"]))


def build_lagrangian(config: dict) -> Lagrangian:
    """按冻结配置构造两个约束乘子（初值 / lr / 上限全部取自 `multipliers` 段）。"""
    multipliers = config["training"]["multipliers"]
    specs = []
    for name, budget_key, unit in (
        ("business", "business_budget", "violation_task_steps"),
        ("carbon", "carbon_budget", "kgCO2e"),
    ):
        block = multipliers[name]
        # 既有 `Lagrangian` 的乘子初值恒为 0（`ConstraintState.multiplier` 默认值），
        # **没有** initial_multiplier 入参。冻结配置的该字段也是 0.0；若将来配置要求
        # 非零初值，必须明确失败而不是静默用 0 代替。
        if float(block["initial_multiplier"]) != 0.0:
            raise FormalTrainLoopError(
                f"{name} 的 initial_multiplier 必须为 0.0（既有 Lagrangian 只支持 0 初值），"
                f"冻结配置给出 {block['initial_multiplier']!r}")
        specs.append(ConstraintSpec(
            name=name,
            budget=float(config["training"]["budgets"][budget_key]),
            unit=unit,
            learning_rate=float(block["learning_rate"]),
            max_multiplier=float(block["max_multiplier"]),
        ))
    return Lagrangian(specs)


def controlled_origins(count: int = CONTROLLED_ORIGINS_COUNT) -> list[int]:
    """受控短跑用的 origin 列表：现有 train-only `select_origins()` 的前 `count` 个。

    **长期** origin 抽样规则**不**在本模块内定；此处只是把既有已选定的 train origin
    取前若干个供受控短跑使用，长期安排留给 M9.2 冻结。
    """
    from scripts.calibrate_training_config import select_origins

    origins = list(select_origins())
    if count > len(origins):
        raise FormalTrainLoopError(
            f"受控短跑需要 {count} 个 origin，但 select_origins() 只给出 {len(origins)} 个")
    return origins[:count]


def peak_rss_bytes() -> int:
    """本进程峰值常驻内存（`ru_maxrss`；macOS 单位是字节、Linux 是 KB，此处归一化）。

    **不使用** `tracemalloc`（M9.1 卡明确禁止在计时区间启用）。
    """
    import resource
    import sys

    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def corrector_solve_stats(buffer: RolloutBuffer) -> dict[str, Any]:
    """本批 `correction_solve_time_s` 的中位数 / P95 / 最大值（实测，非差值）。"""
    values = [
        float(t.correction_info["correction_solve_time_s"])
        for t in buffer.transitions
        if t.correction_info.get("correction_solve_time_s") is not None
    ]
    if not values:
        return {"measured": 0, "median_s": None, "p95_s": None, "max_s": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "measured": len(values),
        "median_s": float(np.median(ordered)),
        "p95_s": float(ordered[p95_index]),
        "max_s": float(ordered[-1]),
    }


def _optimizer_steps(optimizer: Any) -> int:
    """optimizer **真实**已执行的步数（取自 Adam 状态，而非调用次数）。"""
    steps = [float(v["step"]) for v in optimizer.state.values()
             if isinstance(v, dict) and "step" in v]
    return int(max(steps)) if steps else 0


def _generator_state_digest(generator: torch.Generator) -> str:
    return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()


def _policy_state_digest(policy: Any) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run_training_batch(
    policy: SafePPOPolicy,
    optimizer: torch.optim.Adam,
    lagrangian: Lagrangian,
    sampling_generator: torch.Generator,
    shuffle_generator: torch.Generator,
    *,
    config: dict,
    origins: list[int],
    batch_index: int,
    env_seed: int,
    corrector_time_limit_s: float,
    master_seed: int,
) -> dict[str, Any]:
    """跑**一批**真实训练：采集 → 固定 advantage/target → 16 次 Adam step → 1 次乘子更新。"""
    training = config["training"]
    sampling = training["sampling"]
    episodes_per_batch = int(sampling["episodes_per_batch"])
    horizon = int(sampling["horizon"])
    if len(origins) != episodes_per_batch:
        raise FormalTrainLoopError(
            f"一批需要 {episodes_per_batch} 个 origin，实际给出 {len(origins)}")
    epochs = int(sampling["epochs_per_batch"])
    minibatch_size = int(sampling["minibatch_size"])
    ppo = training["ppo"]

    # --- 1. 采集（corrector on，预算取自冻结配置）--------------------------------
    _batch_t0 = time.perf_counter()
    buffer, origin_provenance, collect_timing = _collect_with_config(
        policy, sampling_generator, origins=origins, horizon=horizon,
        env_seed=env_seed, corrector_time_limit_s=corrector_time_limit_s,
        master_seed=master_seed, config=config)
    rollout_collect_s = float(collect_timing["collect_s"])
    n = len(buffer)
    expected = episodes_per_batch * horizon
    if n != expected:
        raise FormalTrainLoopError(f"批 {batch_index} 采到 {n} 条，期望 {expected}")
    if n % minibatch_size != 0:
        raise FormalTrainLoopError(
            f"批大小 {n} 不是 minibatch_size={minibatch_size} 的整数倍")

    # --- 2. 该批首次更新**之前**算出并固定 advantage / target -------------------
    frozen = compute_advantage_oriented_arrays(
        policy, buffer, gamma=float(ppo["gamma_per_step"]), lam=float(ppo["gae_lambda"]))
    observation = _stack(buffer, "observation")
    raw_action = _stack(buffer, "raw_action")
    multipliers_pre = {k: float(v) for k, v in lagrangian.multipliers().items()}
    steps_before = _optimizer_steps(optimizer)
    updates_before = int(getattr(lagrangian, "_updates", 0))

    # --- 3. epochs × minibatches：每 minibatch **恰好一次** Adam step ------------
    minibatches_per_epoch = n // minibatch_size
    if minibatches_per_epoch != int(sampling["minibatches_per_epoch"]):
        raise FormalTrainLoopError(
            f"每 epoch 的 minibatch 数 {minibatches_per_epoch} 与冻结配置 "
            f"{sampling['minibatches_per_epoch']} 不一致")
    _ppo_t0 = time.perf_counter()
    step_records: list[dict[str, Any]] = []
    for epoch in range(epochs):
        order = torch.randperm(n, generator=shuffle_generator).tolist()
        for mb in range(minibatches_per_epoch):
            idx = torch.as_tensor(
                order[mb * minibatch_size:(mb + 1) * minibatch_size], dtype=torch.long)
            out = minibatch_ppo_step(
                policy, optimizer,
                observation=observation[idx], raw_action=raw_action[idx],
                old_raw_log_prob=frozen["old_raw_log_prob"][idx],
                adv_reward=frozen["adv_reward"][idx],
                adv_business=frozen["adv_business"][idx],
                adv_carbon=frozen["adv_carbon"][idx],
                critic_targets={
                    "reward": frozen["target_reward"][idx],
                    "business": frozen["target_business"][idx],
                    "carbon": frozen["target_carbon"][idx],
                },
                lambda_business=multipliers_pre["business"],
                lambda_carbon=multipliers_pre["carbon"],
                clip_epsilon=float(ppo["clip_epsilon"]))
            step_records.append({
                "epoch": epoch, "minibatch": mb,
                "optimizer_steps_cumulative": _optimizer_steps(optimizer),
                "loss_total": float(out["loss_total"]),
                "clip_fraction": float(out["clip_fraction"]),
                "param_delta_norm": float(out["param_delta_norm"]),
            })

    ppo_update_s = time.perf_counter() - _ppo_t0

    # --- 4. 乘子：**整批 192 条**信号各更新一次（不在 minibatch 上更新）----------
    lagrangian.update({
        "business": np.asarray([float(t.business_cost) for t in buffer.transitions]),
        "carbon": np.asarray([float(t.carbon_cost) for t in buffer.transitions]),
    })
    multipliers_post = {k: float(v) for k, v in lagrangian.multipliers().items()}

    return {
        "batch_index": int(batch_index),
        "origins": [int(o) for o in origins],
        "transitions": int(n),
        "epochs": int(epochs),
        "minibatches_per_epoch": int(minibatches_per_epoch),
        "minibatch_size": int(minibatch_size),
        "adam_steps_this_batch": _optimizer_steps(optimizer) - steps_before,
        "optimizer_steps_cumulative": _optimizer_steps(optimizer),
        "lagrangian_updates_this_batch": int(
            getattr(lagrangian, "_updates", 0)) - updates_before,
        "lagrangian_updates_cumulative": int(getattr(lagrangian, "_updates", 0)),
        "multipliers_pre_update": multipliers_pre,
        "multipliers_post_update": multipliers_post,
        "constraint_signal_mean": {
            "business": float(np.mean([float(t.business_cost)
                                       for t in buffer.transitions])),
            "carbon": float(np.mean([float(t.carbon_cost)
                                     for t in buffer.transitions])),
        },
        "raw_exec_difference_count": sum(
            1 for t in buffer.transitions
            if not np.array_equal(t.raw_action, t.exec_action)),
        "deadline_shortfall_steps": sum(
            1 for t in buffer.transitions
            if str(t.correction_info.get("correction_reason")) == "deadline_shortfall"),
        "zero_action_fallback_steps": sum(
            1 for t in buffer.transitions
            if str(t.correction_info.get("correction_reason")) in
            ("timeout", "base_shortage", "solver_failure", "proposal_invalid")),
        "policy_state_digest": _policy_state_digest(policy),
        "optimizer_state_digest": optimizer_state_digest(optimizer),
        "lagrangian_state": lagrangian.state_dict(),
        "sampling_generator_state_digest": _generator_state_digest(sampling_generator),
        "shuffle_generator_state_digest": _generator_state_digest(shuffle_generator),
        # 该批**关键 transition**的摘要：观测 / raw 动作 / old_raw_log_prob 的逐元素字节
        # （用于「连续第 3 批」与「恢复后第 3 批」的精确对照）
        "env_seeds": train_env_seeds(config, master_seed=master_seed),
        "timing_s": {
            "env_build_s": float(collect_timing["env_build_s"]),
            "rollout_collect_s": float(rollout_collect_s),
            "ppo_update_s": float(ppo_update_s),
            "total_batch_s": float(time.perf_counter() - _batch_t0),
            "residual_unattributed_s": float(
                (time.perf_counter() - _batch_t0)
                - collect_timing["env_build_s"] - rollout_collect_s - ppo_update_s),
        },
        "timing_note": (
            "分项为独立计时器读数；`residual_unattributed_s` 是**差值**（整批减去三项），"
            "**不是**独立测量值。checkpoint 写入耗时由入口单独测量。"),
        "corrector_solve_stats": corrector_solve_stats(buffer),
        "peak_rss_bytes": peak_rss_bytes(),
        "origin_provenance": {str(o): p for o, p in sorted(origin_provenance.items())},
        # 该批**关键 transition**摘要的覆盖范围（如实说明，不夸大为「全部字段」）：
        # 仅 observation / raw_action / old_raw_log_prob 三个数组的逐元素字节
        "batch_transition_digest_scope":
            "observation + raw_action + old_raw_log_prob（不含 exec/成本/correction_info）",
        "batch_transition_digest": _digest_arrays(
            [_stack(buffer, "observation").numpy(),
             _stack(buffer, "raw_action").numpy(),
             frozen["old_raw_log_prob"].numpy()]),
        "step_records": step_records,
        "claims": dict(CLAIMS),
    }


def optimizer_state_digest(optimizer: Any) -> str:
    """Adam **完整**状态的摘要（`exp_avg` / `exp_avg_sq` / `step` 逐张量）。"""
    h = hashlib.sha256()
    for key in sorted(optimizer.state, key=lambda k: str(k)):
        entry = optimizer.state[key]
        if not isinstance(entry, dict):
            continue
        for name in sorted(entry):
            value = entry[name]
            h.update(f"{name}".encode())
            if torch.is_tensor(value):
                h.update(value.detach().cpu().numpy().tobytes())
            else:
                h.update(str(value).encode())
    return h.hexdigest()


def _digest_arrays(arrays: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.float32)).tobytes())
    return h.hexdigest()


def _collect_with_config(policy, sampling_generator, *, origins, horizon, env_seed,
                         corrector_time_limit_s, master_seed, config):
    """采集一批；环境由**本次 master seed** + 冻结配置 `seed_offsets` 构造。

    返回 `(buffer, origin_provenance)`：后者是 origin → injection `provenance_hash`，
    供 `training_source_ledger` 计算场景 hash。
    """
    buffer = RolloutBuffer()
    provenance: dict[int, str] = {}
    env_build_s = 0.0
    collect_s = 0.0
    for origin in origins:
        _t0 = time.perf_counter()
        env, injection = build_train_env(int(origin), master_seed=master_seed,
                                         config=config)
        env_build_s += time.perf_counter() - _t0
        _c0 = time.perf_counter()
        if int(env.horizon) != int(horizon):
            raise FormalTrainLoopError(
                f"环境 horizon={env.horizon} 与冻结配置 {horizon} 不一致")
        provenance[int(origin)] = str(injection.provenance_hash)
        stats = collect_rollout(
            env, policy, buffer, steps=int(horizon), seed=int(env_seed),
            corrector_on=True, corrector_time_limit_s=float(corrector_time_limit_s),
            generator=sampling_generator)
        if int(stats["transitions"]) != int(horizon):
            raise FormalTrainLoopError(
                f"origin {origin} 只采到 {stats['transitions']} 条 transition，"
                f"期望 {horizon}")
        collect_s += time.perf_counter() - _c0
    return buffer, provenance, {"env_build_s": env_build_s, "collect_s": collect_s}


def _stack(buffer: RolloutBuffer, field: str) -> torch.Tensor:
    return torch.as_tensor(
        np.stack([np.asarray(getattr(t, field), dtype=np.float32)
                  for t in buffer.transitions]), dtype=torch.float32)


def save_resume_checkpoint(
    path: str | Path,
    *,
    policy: SafePPOPolicy,
    optimizer: torch.optim.Adam,
    lagrangian: Lagrangian,
    sampling_generator: torch.Generator,
    shuffle_generator: torch.Generator,
    config: dict,
    origins: list[int],
    next_batch_index: int,
    obs_dim: int,
    origin_provenance: dict[int, str],
    master_seed: int,
    code_revision: str = "",
    training_scope: str = TRAINING_SCOPE,
    artifact_role: str = RESUME_ARTIFACT_ROLE,
    schema: str = RESUME_SCHEMA,
) -> dict[str, Any]:
    """**批次边界**训练恢复 checkpoint：下一批游标 + 两个 RNG + 全部训练状态。

    M1.3g-f-c-j-R1：同时写入**可核对**的来源账本（三个 hash）、截至本边界的
    origin→injection provenance，以及构造实际运行代码的 `code_revision`。
    """
    from checkpointing import CURRENT_CONTRACT_VERSION, VersionedCheckpoint

    training = config["training"]
    state = {
        "policy": {k: v.detach().clone() for k, v in policy.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "lagrangian": lagrangian.state_dict(),
        "sampling_generator": sampling_generator.get_state(),
        "shuffle_generator": shuffle_generator.get_state(),
        "next_batch_index": int(next_batch_index),
        "origins": [int(o) for o in origins],
        "episodes_per_batch": int(training["sampling"]["episodes_per_batch"]),
        "steps_per_episode": int(training["sampling"]["horizon"]),
        "frozen_config": config,
        "config_summary": config_provenance_summary(config),
        "training_scope": str(training_scope),
        "artifact_role": str(artifact_role),
        "source_ledger": {
            **training_source_ledger(origin_provenance),
            "env_seeds": train_env_seeds(config, master_seed=master_seed),
            "master_seed": int(master_seed),
        },
        "origin_provenance": {str(o): str(p)
                              for o, p in sorted(origin_provenance.items())},
        "code_revision": str(code_revision),
    }
    checkpoint = VersionedCheckpoint(
        contract_version_id=CURRENT_CONTRACT_VERSION,
        action_dim=int(training["policy"]["action_dim"]),
        obs_dim=int(obs_dim),
        schema_hash=str(schema),
        code_revision=str(code_revision),
        state=state,
    )
    checkpoint.save(path)
    return {
        "path": str(path),
        "schema_hash": str(schema),
        "artifact_role": str(artifact_role),
        "training_scope": str(training_scope),
        "next_batch_index": state["next_batch_index"],
        "contract_version": CONTRACT_VERSION,
    }


def load_resume_checkpoint(
    path: str | Path,
    *,
    policy: SafePPOPolicy,
    optimizer: torch.optim.Adam,
    lagrangian: Lagrangian,
    sampling_generator: torch.Generator,
    shuffle_generator: torch.Generator,
    config: dict,
    expected_obs_dim: int,
    expected_schema: str = RESUME_SCHEMA,
    expected_scope: str = TRAINING_SCOPE,
    expected_role: str = RESUME_ARTIFACT_ROLE,
) -> dict[str, Any]:
    """把 checkpoint 状态写回**调用方提供的新对象**，返回下一批游标与摘要。

    schema / 版本 / 维度不符即拒绝；状态字段缺一即拒绝（不补默认值）。
    **评估输入**（`checkpointing.eval_input`）的 schema 会被本函数拒绝——两者角色不同。
    """
    from checkpointing import VersionedCheckpoint
    from checkpointing.eval_input import EVAL_INPUT_SCHEMA

    if expected_schema == EVAL_INPUT_SCHEMA:
        raise FormalTrainLoopError("训练恢复 schema 不得与评估输入 schema 相同")
    checkpoint = VersionedCheckpoint.load(
        path, expected_action_dim=int(config["training"]["policy"]["action_dim"]),
        expected_obs_dim=int(expected_obs_dim), expected_schema_hash=str(expected_schema))
    state = checkpoint.state
    if not isinstance(state, dict) or set(state) != set(BATCH_KEYS):
        missing = sorted(set(BATCH_KEYS) - set(state or {}))
        extra = sorted(set(state or {}) - set(BATCH_KEYS))
        raise FormalTrainLoopError(
            f"训练恢复状态字段必须精确等于 {list(BATCH_KEYS)}；缺少={missing} 多出={extra}")
    if state["artifact_role"] != str(expected_role):
        raise FormalTrainLoopError(
            f"artifact_role 必须是 {str(expected_role)!r}，实际 {state['artifact_role']!r}"
            "（受控短跑与正式训练不得互相冒充）")
    if state["training_scope"] != str(expected_scope):
        raise FormalTrainLoopError(
            f"training_scope 必须是 {str(expected_scope)!r}，实际 {state['training_scope']!r}"
            "（受控短跑与正式训练不得互相冒充）")
    if state["frozen_config"] != config:
        raise FormalTrainLoopError(
            "checkpoint 的冻结配置与当前冻结配置不一致：拒绝跨配置恢复")

    policy.load_state_dict(state["policy"])
    optimizer.load_state_dict(state["optimizer"])
    lagrangian.load_state_dict(state["lagrangian"])
    sampling_generator.set_state(state["sampling_generator"])
    shuffle_generator.set_state(state["shuffle_generator"])
    return {
        "next_batch_index": int(state["next_batch_index"]),
        "origins": [int(o) for o in state["origins"]],
        "episodes_per_batch": int(state["episodes_per_batch"]),
        "steps_per_episode": int(state["steps_per_episode"]),
        "config_summary": dict(state["config_summary"]),
        "training_scope": str(state["training_scope"]),
        "artifact_role": str(state["artifact_role"]),
        "schema_hash": checkpoint.schema_hash,
        # M1.3g-f-c-j-R1：随 checkpoint 携带的来源账本 / provenance / code revision
        "source_ledger": dict(state["source_ledger"]),
        "origin_provenance": {int(o): str(p)
                              for o, p in dict(state["origin_provenance"]).items()},
        "code_revision": str(state["code_revision"]),
        "checkpoint_code_revision": str(checkpoint.code_revision),
    }

"""M5.1c–M5.3f 训练入口：dry-run 更新，**唯一数据来源**是 contract-v7 `RolloutBuffer`。

定位不变：只验证「真实采集 → 一次更新」闭环可跑通，**不因短训练奖励高而宣称有效**。

**本模块不是 PPO**：没有 ratio、没有 clip、没有熵项。
**M5.3 已完成**：乘子目标接线 —— actor 有效优势为
`A_reward − lambda_business·A_business − lambda_carbon·A_carbon`，
乘子取自 `Lagrangian`（更新前的值），约束信号的非负物理域与状态自洽性均已强制。
**M5.4 尚未完成**：未来信息泄漏门禁、corrector 开/关下的跨进程可复现性。
（训练入口见 §「命令行」；本模块**不写 checkpoint**。）

**预检的位置**：`validate_constraint_signals` 在 **rollout 收集之后**、
**训练更新阶段的任何 forward / backward / optimizer.step 之前**执行。
注意它**不**在采样前向之前 —— `collect_rollout` 内部会调用 `policy.act(...)`，
而 `act` 会执行 `forward`。预检保证的是「**更新阶段零副作用**」，
使得非法约束信号既不更新参数、也不推进 optimizer 状态。

**不得**据此宣称训练有效、收敛或任何性能改善。

红线：
- 采样只经 `collect_rollout`；raw 动作**原样**执行，**不存在** `_clip_action`；
- 三套 critic target 的**每一个输入**都来自 buffer 的具名字段
  （`observation` / `next_observation` / `terminated` / `truncated` /
  `reward` / `business_cost` / `carbon_cost`），不回读 env 内部数组或未来真值；
- bootstrap **逐 transition** 取各自 `next_observation` 的 critic value
  （M5.2c 勘误：不得用下标 `t+1`，也不得只读最后一条）；
  是否使用由 `terminated` 掩码决定，调用方不自行判断；
- critic loss 逐头对应各自 target，三个独立 MSE，不混列、不共享、不以电费顶替；
- actor likelihood 只对 buffer 的 `raw_action` 计算（`evaluate_raw_actions`），
  **绝不对 `exec_action` 求概率**；`old_raw_log_prob` 是**旧策略审计值**，不进入损失；
- 不使用 `info.get(..., 0)` 之类默认值；环境字段缺失在采集期直接报错；
- 不重新播种全局 Torch RNG：采样随机性由调用方传入的 `generator` 决定。
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from contracts import CONTRACT_VERSION_ID
from envs.idc_price_env import IDCPriceEnv20D
from planning.corrector import PRODUCTION_CORRECTOR_TIME_LIMIT_S, resolve_corrector_budget
from runs.writer import write_run
from safe_rl_v2.buffer import ACTION_DIM, RolloutBuffer
from safe_rl_v2.lagrangian import Lagrangian, validate_constraint_signals
from safe_rl_v2.models import compute_three_value_targets
from safe_rl_v2.policy import SafePPOPolicy
from safe_rl_v2.rollout import collect_rollout

# 可导 likelihood 的来源（供审计断言，不参与计算）
LOG_PROB_SOURCE = "evaluate_raw_actions(raw_action)"
# 三套 target 的唯一来源（M5.2a 的终端感知 API）
TARGETS_SOURCE = "compute_three_value_targets(terminated=, truncated=)"
# actor 有效优势的唯一形式（M5.3b；乘子为**更新前**值）
ACTOR_OBJECTIVE_SOURCE = "A_reward - lambda_business * A_business - lambda_carbon * A_carbon"

GAMMA = 0.99
LAM = 0.95

# 单位口径（M5.2a §5）：业务违规量是**每步活跃逾期 SLA 违规计数**，
# rollout 内累计为「违规任务·步」；碳排放为 kgCO2e；电费为 SGD。三者不得混算。
METRIC_UNITS = {
    "reward": "dimensionless",
    "business_violations": "violation_task_steps",
    "carbon_emissions": "kgCO2e",
    "electricity_cost_sgd": "SGD",
}

_CLAIMS = {
    "trained": False,
    "performance_evaluated": False,
    "convergence_claimed": False,
}


def dry_run_update(
    env,
    policy: SafePPOPolicy,
    lagrangian: Lagrangian,
    optimizer: torch.optim.Optimizer,
    *,
    corrector_on: bool = False,
    corrector_time_limit_s: float | None = None,
    steps: int = 8,
    seed: int = 0,
    generator: torch.Generator | None = None,
) -> dict:
    """短 dry rollout + 一次更新。返回三套 value、乘子、损耗与**真实采集**的 buffer。"""
    buffer = RolloutBuffer()
    stats = collect_rollout(
        env,
        policy,
        buffer,
        steps=steps,
        seed=seed,
        corrector_on=corrector_on,
        corrector_time_limit_s=corrector_time_limit_s,
        generator=generator,
    )

    n_steps = len(buffer)
    if n_steps == 0:
        raise RuntimeError("collect_rollout 未采集到任何 transition，无法执行更新")

    obs = torch.as_tensor(
        np.stack([t.observation for t in buffer.transitions]), dtype=torch.float32
    )
    next_obs = torch.as_tensor(
        np.stack([t.next_observation for t in buffer.transitions]), dtype=torch.float32
    )
    raw_actions = torch.as_tensor(
        np.stack([t.raw_action for t in buffer.transitions]), dtype=torch.float32
    )
    rewards = np.array([t.reward for t in buffer.transitions], dtype=np.float64)
    business_violations = np.array(
        [t.business_cost for t in buffer.transitions], dtype=np.float64
    )
    carbon_emissions = np.array([t.carbon_cost for t in buffer.transitions], dtype=np.float64)
    electricity_costs = np.array(
        [t.electricity_cost_sgd for t in buffer.transitions], dtype=np.float64
    )
    terminals = np.array([t.terminated for t in buffer.transitions], dtype=bool)
    truncations = np.array([t.truncated for t in buffer.transitions], dtype=bool)

    # --- M5.3f 预检：在**更新阶段**的 forward / backward / optimizer.step 之前 ---
    # 注意位置在 rollout 收集**之后**（采样前向已由 collect_rollout 完成），
    # 保证的是「更新阶段零副作用」：失败时 policy 参数、optimizer state、
    # Lagrangian state 三者都不变。复用 lagrangian 的共享校验规则。
    validate_constraint_signals(
        {"business": business_violations, "carbon": carbon_emissions},
        lagrangian.constraints,
    )

    # bootstrap：**逐 transition** 取各自 next_observation 的 critic 估计
    # （M5.2c：不得用下标 t+1，也不得只读最后一条）
    with torch.no_grad():
        _, current_v = policy.forward(obs)
        _, next_v = policy.forward(next_obs)

    current_values = {
        "reward": current_v[:, 0].numpy(),
        "business": current_v[:, 1].numpy(),
        "carbon": current_v[:, 2].numpy(),
    }
    next_values = {
        "reward": next_v[:, 0].numpy(),
        "business": next_v[:, 1].numpy(),
        "carbon": next_v[:, 2].numpy(),
    }

    targets = compute_three_value_targets(
        rewards,
        business_violations,
        carbon_emissions,
        current_values,
        next_values,
        terminated=terminals,
        truncated=truncations,
        gamma=GAMMA,
        lam=LAM,
    )

    adv_reward = torch.tensor(targets["reward"][0], dtype=torch.float32)
    adv_business = torch.tensor(targets["business"][0], dtype=torch.float32)
    adv_carbon = torch.tensor(targets["carbon"][0], dtype=torch.float32)
    tgt_reward = torch.tensor(targets["reward"][1], dtype=torch.float32)
    tgt_business = torch.tensor(targets["business"][1], dtype=torch.float32)
    tgt_carbon = torch.tensor(targets["carbon"][1], dtype=torch.float32)

    # 可导 likelihood：只对 buffer 的 raw_action 重算（绝不对 exec_action）
    log_probs = policy.evaluate_raw_actions(obs, raw_actions)
    _, values = policy.forward(obs)

    # 逐头对应的 critic loss：三个独立 MSE，分别上报，不混列
    critic_loss_by_head = {
        "reward": (values[:, 0] - tgt_reward).pow(2).mean(),
        "business": (values[:, 1] - tgt_business).pow(2).mean(),
        "carbon": (values[:, 2] - tgt_carbon).pow(2).mean(),
    }
    critic_loss = (
        critic_loss_by_head["reward"]
        + critic_loss_by_head["business"]
        + critic_loss_by_head["carbon"]
    )

    # M5.3b：本轮 actor 目标使用**更新前**的乘子（两约束各自独立）
    multipliers_pre = lagrangian.multipliers()
    lambda_business = float(multipliers_pre["business"])
    lambda_carbon = float(multipliers_pre["carbon"])

    reward_term = adv_reward
    business_term = -lambda_business * adv_business
    carbon_term = -lambda_carbon * adv_carbon
    effective_advantage = reward_term + business_term + carbon_term

    actor_loss = -(effective_advantage * log_probs).mean()
    loss = actor_loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 乘子更新发生在 optimizer.step() **之后**，且只消费逐 transition 序列
    # （聚合口径 per-transition mean 由 lagrangian 模块内部固定）。
    lagrangian.update(
        {
            "business": business_violations,
            "carbon": carbon_emissions,
        }
    )
    multipliers_post = lagrangian.multipliers()

    return {
        # --- 既有键（M5.1c/M5.4 语义保持不变）---
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "reward_value": float(values[0, 0].item()),
        "business_value": float(values[0, 1].item()),
        "carbon_value": float(values[0, 2].item()),
        "multipliers": multipliers_post,
        "corrector_on": corrector_on,
        # --- 采集链证据（**不是**训练结论）---
        "buffer": buffer,
        "stats": stats,
        "steps_collected": n_steps,
        "logprob_source": LOG_PROB_SOURCE,
        "old_raw_log_prob_mean": float(
            np.mean([t.old_raw_log_prob for t in buffer.transitions])
        ),
        "business_violation_sum": float(np.sum(business_violations)),
        "carbon_emission_sum_kg": float(np.sum(carbon_emissions)),
        "electricity_cost_sum_sgd": float(np.sum(electricity_costs)),
        "env_seed": stats["env_seed"],
        "policy_rng_source": stats["policy_rng_source"],
        "bootstrap_is_terminal": bool(terminals[-1]),
        # --- M5.2b/M5.2d：三套 target 的来源与口径 ---
        "targets_source": TARGETS_SOURCE,
        "gae": {"gamma": GAMMA, "lam": LAM},
        "bootstrap": {
            "source": "buffer_transition_next_observations",
            "terminated_count": int(terminals.sum()),
            "truncated_count": int(truncations.sum()),
            "bootstrapped_count": int((~terminals).sum()),
        },
        "critic_loss_by_head": {head: float(v.item()) for head, v in critic_loss_by_head.items()},
        "critic_targets": {head: pair[1] for head, pair in targets.items()},
        "units": dict(METRIC_UNITS),
        "claims": dict(_CLAIMS),
        # --- M5.3b：actor objective 的乘子口径与诊断（**不是**性能结论）---
        "actor_objective_source": ACTOR_OBJECTIVE_SOURCE,
        "multipliers_pre_update": dict(multipliers_pre),
        "multipliers_post_update": dict(multipliers_post),
        "constraint_means": {
            "business": float(np.mean(business_violations)),
            "carbon": float(np.mean(carbon_emissions)),
        },
        "constraint_budgets": {
            name: float(state.budget) for name, state in lagrangian.constraints.items()
        },
        "constraint_units": {
            name: state.unit for name, state in lagrangian.constraints.items()
        },
        "effective_advantage": {
            "formula": ACTOR_OBJECTIVE_SOURCE,
            "multipliers_used": "pre_update",
            "mean": float(effective_advantage.mean().item()),
            "std": float(effective_advantage.std(unbiased=False).item()),
            "min": float(effective_advantage.min().item()),
            "max": float(effective_advantage.max().item()),
        },
        "effective_advantage_terms": {
            "reward_mean": float(reward_term.mean().item()),
            "business_mean": float(business_term.mean().item()),
            "carbon_mean": float(carbon_term.mean().item()),
        },
    }


# ===========================================================================
# M5.4a 命令行入口
# ===========================================================================
#
# 用法：`python -m safe_rl_v2.train [--synthetic-smoke] ...`（Makefile 用 `make train`）。
#
# 定位：**可执行但不冒充正式实验**。
# - 默认路径要求 M1.2 的真实冻结数据；缺失时**明确失败**，绝不回退到合成数据；
# - 只有显式 `--synthetic-smoke` 才跑一次短 dry rollout，产物全程自我标注
#   `synthetic=true` / `dry_run_only=true`，claims 三项全为 false。
# 本入口**不写 checkpoint**，也**不实现** PPO ratio/clip/熵项。

DEFAULT_STEPS = 8
DEFAULT_SEED = 0
# corrector 生产默认预算**不在本文件定义**：唯一来源是
# `planning.corrector.PRODUCTION_CORRECTOR_TIME_LIMIT_S`（M5.4i）。
# 环境三类种子必须同时显式给定：任一为 None 时 env 使用 default_rng(None) 熵源，
# 跨进程不可复现。**不允许**留空。
DEFAULT_ENV_SEED_KWARGS = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}

SCENARIO_TYPE_SYNTHETIC = "synthetic_internal_env"
SCENARIO_TYPE_FROZEN_REAL = "frozen_real_scenario"

REPORT_STATEMENT = (
    "合成 dry-run：仅验证「采集 → 一次更新」闭环可执行，并产出运行产物。"
    "本 run **不是**正式训练、**不是**性能评估、**不构成**论文结果；"
    "不得据此宣称训练有效或收敛。"
)

_DRY_RUN_CLAIMS = {
    "trained": False,
    "performance_evaluated": False,
    "convergence_claimed": False,
}


class TrainEntryError(RuntimeError):
    """入口级失败（参数、数据阻塞、运行期错误）；用于写失败 manifest。"""


class RunIdConflictError(RuntimeError):
    """run_id 撞上已有成功 run：保留成功结果，不写失败 manifest。"""


# 命令账本：`(flag, dest, kind)`。kind ∈ {"flag", "value"}。
# 必须覆盖 `build_parser()` 的**全部** action —— 由
# `tests/test_m54c_run_artifact_integrity.py::test_command_ledger_covers_every_cli_action`
# 结构性地断言，新增 CLI 参数而忘记登记时该测试会失败。
COMMAND_ARGV_SPEC: tuple[tuple[str, str, str], ...] = (
    ("--synthetic-smoke", "synthetic_smoke", "flag"),
    ("--steps", "steps", "value"),
    ("--seed", "seed", "value"),
    ("--corrector", "corrector", "value"),
    ("--corrector-time-limit-s", "corrector_time_limit_s", "value"),
    ("--base-dir", "base_dir", "value"),
    ("--run-id", "run_id", "value"),
    ("--task-seed", "task_seed", "value"),
    ("--server-seed", "server_seed", "value"),
    ("--forecast-seed", "forecast_seed", "value"),
    ("--horizon", "horizon", "value"),
)


def _effective_argv(args: argparse.Namespace, run_id: str) -> list[str]:
    """由**解析后的 namespace** 重建完整调用 argv（结构上不可能遗漏参数）。

    `run_id` 单独传入：默认 run_id 是运行时生成的，未必等于 `args.run_id`，
    但账本必须记录**真正使用**的那个。
    """
    argv = ["python", "-m", "safe_rl_v2.train"]
    for flag, dest, kind in COMMAND_ARGV_SPEC:
        if dest == "run_id":
            argv += [flag, run_id]
            continue
        value = getattr(args, dest)
        if kind == "flag":
            if value:
                argv.append(flag)
        elif value is None:
            continue  # 未给出的可选参数不进账本
        else:
            argv += [flag, str(value)]
    return argv


def _dependency_lock_hash() -> str | None:
    import hashlib

    lock = Path(__file__).resolve().parent.parent / "uv.lock"
    return hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None


def _unique_run_id(base_dir: Path, seed: int, *, kind: str) -> str:
    """默认 run_id 每次调用唯一，使重复运行不覆盖既有成功结果。

    `kind` 决定前缀：`"synthetic"` / `"real"`。默认真实路径**不得**写成 synthetic
    —— 它因 M1.2 阻塞而失败，从未使用合成数据（M5.4c 修复）。
    """
    if kind not in ("synthetic", "real"):
        raise ValueError(f"kind 必须为 'synthetic' 或 'real'，got {kind!r}")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = f"train_{kind}_s{seed}_{stamp}"
    candidate, suffix = base, 1
    while (base_dir / candidate / "manifest.json").exists():
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m safe_rl_v2.train",
        description="安全 PPO v2 训练入口（默认要求 M1.2 真实数据；合成 dry-run 需显式开关）",
    )
    parser.add_argument(
        "--synthetic-smoke", action="store_true",
        help="显式启用合成短 dry rollout（产物标注 synthetic=true，不代表正式训练）",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="policy 采样 RNG 种子")
    parser.add_argument("--corrector", choices=("on", "off"), default="off")
    parser.add_argument("--corrector-time-limit-s", type=float, default=None)
    parser.add_argument("--base-dir", default="runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--task-seed", type=int, default=DEFAULT_ENV_SEED_KWARGS["task_seed"])
    parser.add_argument("--server-seed", type=int, default=DEFAULT_ENV_SEED_KWARGS["server_seed"])
    parser.add_argument(
        "--forecast-seed", type=int, default=DEFAULT_ENV_SEED_KWARGS["forecast_seed"]
    )
    parser.add_argument("--horizon", type=int, default=24)
    return parser


def _require_frozen_real_scenario(args) -> None:
    """正式路径：必须要有 M1.2 的真实冻结数据；缺失即失败，**不回退合成**。"""
    from scenario.scenario import build_scenario

    try:
        build_scenario("train", start="2023-01-01", horizon=args.horizon, forecast_cutoff=4)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise TrainEntryError(
            "训练默认路径需要 M1.2 的冻结真实数据，当前不可用，故明确失败"
            "（**不**回退到合成数据）。原始错误："
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _budget_resolution(args) -> tuple[float | None, str]:
    """解析本次调用的 `(有效预算, 来源)`；来源由**是否显式给出**决定。

    四个入口共用 `planning.corrector.resolve_corrector_budget`，避免默认值再次分叉。
    """
    return resolve_corrector_budget(
        args.corrector_time_limit_s, enabled=args.corrector == "on"
    )


def _run_synthetic_dry_run(args, run_id: str, env_seed_kwargs: dict[str, int]) -> dict:
    """跑一次合成短 dry rollout，返回 report。只调用既有 dry_run_update。"""
    from safe_rl_v2.lagrangian import UNIT_KG_CO2E, UNIT_VIOLATION_TASK_STEPS, ConstraintSpec

    corrector_on = args.corrector == "on"
    # M5.4i：未显式给出预算时解析为**生产默认**（不再拒绝）；
    # 显式给出则原样使用，绝不静默替换。
    effective_budget, _ = _budget_resolution(args)

    env = IDCPriceEnv20D(horizon=args.horizon, **env_seed_kwargs)
    # 权重初始化借用全局 RNG，但用 fork_rng 还原：调用方的全局 RNG 状态不受影响。
    # **采样**另有显式 generator（红线：不得用全局 RNG 采样、不得靠重播种伪造可复现）。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        policy = SafePPOPolicy(obs_dim=env.obs_dim)
    lagrangian = Lagrangian((
        ConstraintSpec(
            name="business", budget=5.0, unit=UNIT_VIOLATION_TASK_STEPS,
            learning_rate=0.01, max_multiplier=100.0,
        ),
        ConstraintSpec(
            name="carbon", budget=3.0, unit=UNIT_KG_CO2E,
            learning_rate=0.01, max_multiplier=100.0,
        ),
    ))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    return dry_run_update(
        env, policy, lagrangian, optimizer,
        steps=args.steps, seed=args.seed,
        corrector_on=corrector_on,
        corrector_time_limit_s=effective_budget,
        generator=generator,
    )


def _config_for(args, run_id: str, env_seed_kwargs: dict[str, int], scenario_type: str) -> dict:
    from runs.writer import git_revision

    return {
        "entry": "python -m safe_rl_v2.train",
        "run_id": run_id,
        "synthetic": scenario_type == SCENARIO_TYPE_SYNTHETIC,
        "dry_run_only": True,
        "scenario_type": scenario_type,
        "action_dim": ACTION_DIM,
        "contract_version": CONTRACT_VERSION_ID,
        "env_seed_kwargs": dict(env_seed_kwargs),
        "policy_seed": int(args.seed),
        "policy_rng": "explicit torch.Generator（不使用 torch.manual_seed 采样）",
        "corrector_on": args.corrector == "on",
        "production_corrector_time_limit_s": PRODUCTION_CORRECTOR_TIME_LIMIT_S,
        "effective_corrector_time_limit_s": _budget_resolution(args)[0],
        "corrector_time_limit_source": _budget_resolution(args)[1],
        "corrector_time_limit_s": _budget_resolution(args)[0],
        "steps": int(args.steps),
        "horizon": int(args.horizon),
        "gamma": GAMMA,
        "lam": LAM,
        "code_revision": git_revision(),
        "units": dict(METRIC_UNITS),
    }


def _report_for(result: dict, scenario_type: str, args) -> dict:
    effective, source = _budget_resolution(args)
    return {
        "entry": "python -m safe_rl_v2.train",
        "synthetic": scenario_type == SCENARIO_TYPE_SYNTHETIC,
        "dry_run_only": True,
        "statement": REPORT_STATEMENT,
        "claims": dict(_DRY_RUN_CLAIMS),
        "scenario_type": scenario_type,
        "steps_collected": result["steps_collected"],
        "corrector_on": result["corrector_on"],
        "production_corrector_time_limit_s": PRODUCTION_CORRECTOR_TIME_LIMIT_S,
        "effective_corrector_time_limit_s": effective,
        "corrector_time_limit_source": source,
        "multipliers_pre_update": result["multipliers_pre_update"],
        "multipliers_post_update": result["multipliers_post_update"],
        "constraint_means": result["constraint_means"],
        "constraint_units": result["constraint_units"],
        "actor_loss": result["actor_loss"],
        "critic_loss": result["critic_loss"],
        "actor_objective_source": result["actor_objective_source"],
        "units": result["units"],
    }


def _metrics_for(result: dict) -> pd.DataFrame:
    buffer = result["buffer"]
    return pd.DataFrame(
        [
            {
                "step": index,
                "reward": t.reward,
                "business_violations": t.business_cost,
                "carbon_emissions_kg": t.carbon_cost,
                "electricity_cost_sgd": t.electricity_cost_sgd,
                "old_raw_log_prob": t.old_raw_log_prob,
                "raw_exec_differs": bool(not np.array_equal(t.raw_action, t.exec_action)),
                "terminated": bool(t.terminated),
                "truncated": bool(t.truncated),
            }
            for index, t in enumerate(buffer.transitions)
        ]
    )


def _write_failed_run(
    run_id: str, base_dir: Path, command: str, seed: int, error: Exception
) -> None:
    """已获得 run_id 的失败也必须落一个失败 manifest。

    若该 run_id 已有**成功**结果，则**保留它**并抛 `RunIdConflictError` ——
    绝不覆盖，也绝不把它说成「失败 manifest 已写入」（M5.4c）。
    """
    try:
        write_run(
            run_id,
            config={
                "entry": "python -m safe_rl_v2.train",
                "run_id": run_id,
                "dry_run_only": True,
                "status": "failed",
            },
            metrics=pd.DataFrame(),
            report={
                "entry": "python -m safe_rl_v2.train",
                "dry_run_only": True,
                "synthetic": False,
                "statement": "本 run 失败，不代表任何训练或评估结果。",
                "claims": dict(_DRY_RUN_CLAIMS),
                "failure": f"{type(error).__name__}: {error}",
            },
            base_dir=str(base_dir),
            seed=seed,
            command=command,
            status="failed",
            failure_classification=type(error).__name__,
        )
    except FileExistsError as exc:
        raise RunIdConflictError(
            f"run_id {run_id!r} 已存在成功结果；**保留该成功结果**，未写入失败 manifest。"
            f"原始错误：{exc}"
        ) from exc
    except Exception as exc:  # pragma: no cover - 写失败 manifest 本身失败时不再掩盖
        print(f"警告：写失败 manifest 时又出错：{exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.steps <= 0:
        parser.error(f"--steps 必须为正整数，got {args.steps}")
    if args.corrector_time_limit_s is not None and args.corrector_time_limit_s <= 0.0:
        parser.error(
            f"--corrector-time-limit-s 必须为正数，got {args.corrector_time_limit_s}"
        )
    if args.corrector == "off" and args.corrector_time_limit_s is not None:
        parser.error(
            "--corrector off 时不得给出 --corrector-time-limit-s"
            "（传入预算会让调用方误以为修正器已生效）"
        )

    env_seed_kwargs = {
        "task_seed": args.task_seed,
        "server_seed": args.server_seed,
        "forecast_seed": args.forecast_seed,
    }
    for key, value in env_seed_kwargs.items():
        if value is None:
            parser.error(f"--{key.replace('_', '-')} 不得为空（env 会用熵源，导致不可复现）")

    base_dir = Path(args.base_dir)
    run_kind = "synthetic" if args.synthetic_smoke else "real"
    run_id = args.run_id or _unique_run_id(base_dir, args.seed, kind=run_kind)
    # 命令账本：由 argv 列表 + shell 安全转义构造，结构上不可能遗漏参数
    command = shlex.join(_effective_argv(args, run_id))

    try:
        if not args.synthetic_smoke:
            _require_frozen_real_scenario(args)
            raise TrainEntryError(  # pragma: no cover - 数据到位前不可达
                "真实数据路径的正式训练尚未实现（需 M5.4/M5.5 的训练循环）"
            )
        scenario_type = SCENARIO_TYPE_SYNTHETIC
        result = _run_synthetic_dry_run(args, run_id, env_seed_kwargs)
        config = _config_for(args, run_id, env_seed_kwargs, scenario_type)
        report = _report_for(result, scenario_type, args)
        metrics = _metrics_for(result)
    except Exception as exc:
        print(f"训练入口失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        try:
            _write_failed_run(run_id, base_dir, command, args.seed, exc)
        except RunIdConflictError as conflict:
            # 保留已有成功结果；**不得**声称写了失败 manifest
            print(f"训练入口失败且未写失败 manifest：{conflict}", file=sys.stderr)
            return 3
        return 1

    try:
        run_dir = write_run(
            run_id,
            config=config,
            metrics=metrics,
            report=report,
            base_dir=str(base_dir),
            seed=args.seed,
            command=command,
            scenario_hash=None,
            data_hash=None,
            dependency_lock_hash=_dependency_lock_hash(),
            status="success",
        )
    except FileExistsError as exc:
        print(f"拒绝覆盖既有成功 run：{exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nrun 产物：{run_dir}")
    print(f"scenario_type={scenario_type}  synthetic=true  dry_run_only=true  trained=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

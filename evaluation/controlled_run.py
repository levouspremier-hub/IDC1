"""M6-P1：受控短跑 → 评估输入 checkpoint → 评估 → 标准 runs 产物。

**本模块证明的是「契约与计量可用」，不是算法优劣**：

1. 在 **train split** 的固定 origin 上构造正式环境（走既有已验证注入链）；
2. 用**固定种子**初始化策略并取一小段受控 rollout（`--steps`，**不做任何参数更新**）；
3. 按评估输入契约保存 checkpoint（角色 = 受控短跑评估输入）；
4. 重新**读取**该 checkpoint（并校验来源对当前资产的绑定），跑一个评估 episode，
   产出 `EvaluationRecord`；
5. 通过 `runs.writer.write_run` 写出 `config.yaml / metrics.parquet / report.json /
   manifest.json`，并在 report 中如实标注**未判定**（本链路**不显式传入**服务标准，
   故不判定资格；项目标准 `m6-service-standard-v1` 已冻结但不会被隐式采用）与
   **未评估**（五类正式方法一个都还没跑）。

用法：

```bash
uv run python -m evaluation.controlled_run --run-id <id> [--steps 8] [--origin 48]
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from checkpointing.eval_input import (
    CONTROLLED_ROLE,
    load_evaluation_checkpoint,
    save_evaluation_checkpoint,
)
from evaluation.adapter import PLANNED_METHODS, evaluate, planned_method_rows
from evaluation.sources import canonical_source_digests, verify_evaluation_input_sources

REPO_ROOT = Path(__file__).resolve().parent.parent

SPLIT = "train"
DEFAULT_ORIGIN = 48
DEFAULT_START = "2024-01-02T00:00:00+08:00"
HORIZON = 48
FORECAST_CUTOFF = 48
DELTA_T_HOURS = 0.5
ACTION_DIM = 21
POLICY_HIDDEN = 64
POLICY_SEED = 0
ENV_SEED_OFFSETS = {"task": 0, "server": 1, "forecast": 300000}
DEFAULT_SHORT_RUN_STEPS = 8
CHECKPOINT_FILENAME = "eval_input_checkpoint.pt"

#: 受控短跑**不是**协议 §1 的五类正式方法之一。
CONTROLLED_METHOD = "controlled_short_run_policy"

CLAIMS = {"trained": False, "performance_evaluated": False, "convergence_claimed": False}
STATEMENT = (
    "受控短跑评估链路验证：train-only、固定种子、**无参数更新**。"
    "只证明评估输入契约与指标计算可用，**不宣称任何性能结论**；"
    "本链路不显式传入服务标准 ⇒ 服务资格为「未判定」（冻结标准不被隐式采用）。"
)
NOT_EVALUATED_STATEMENT = (
    "协议 §1 的五类正式方法在本卡**一个都没有评估**（缺方法 ⇒ 「未评估」）；"
    "不得据此伪造比较结果。"
)


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def source_ledger_hashes(env) -> dict[str, str]:
    """受控 run 的**实测、可重算**来源账本（写入 runs manifest）。

    - `dependency_lock_hash`：`uv.lock` 的 SHA-256（重算：对该文件做 sha256）。
    - `data_hash`：12 个已验签来源 `(role, 规范逻辑路径, sha256)` 的规范化 JSON
      之 SHA-256（重算：`evaluation.sources.canonical_source_digests()` 同样序列化后 sha256）。
    - `scenario_hash`：本 episode 正式注入载荷的 `provenance_hash`
      （`scenario.env_injection.build_verified_formal_env_injection` 的内容 hash，
      覆盖 split/origin/horizon/整数账本/realized/因果预测/refs/任务规格）。
    """
    import json

    lock = REPO_ROOT / "uv.lock"
    if not lock.exists():
        raise FileNotFoundError(f"依赖锁不存在：{lock}")
    digests = canonical_source_digests()
    data_hash = _sha256_bytes(json.dumps(
        [[d.role, d.logical_path, d.sha256] for d in digests],
        sort_keys=True, ensure_ascii=False).encode("utf-8"))
    injection = getattr(env, "formal_injection", None)
    if injection is None:
        raise ValueError("受控 run 必须使用正式注入环境（缺少 provenance_hash）")
    scenario_hash = str(injection.provenance_hash)
    if len(scenario_hash) != 64 or any(c not in "0123456789abcdef" for c in scenario_hash):
        raise ValueError(f"scenario provenance_hash 非法：{scenario_hash!r}")
    return {
        "dependency_lock_hash": _sha256_bytes(lock.read_bytes()),
        "data_hash": data_hash,
        "scenario_hash": scenario_hash,
    }


def build_train_env(origin: int = DEFAULT_ORIGIN):
    """走既有已验证注入链构造 **train** 环境（不读 validation/test）。"""
    import importlib

    injection_module = importlib.import_module("scenario.env_injection")
    env_cls = importlib.import_module("envs.idc_price_env").IDCPriceEnv20D
    injection = injection_module.build_verified_formal_env_injection(
        SPLIT, start=DEFAULT_START, horizon=HORIZON, forecast_cutoff=FORECAST_CUTOFF)
    if int(injection.local_origin) != int(origin):
        raise ValueError(
            f"注入链给出的本地 origin {injection.local_origin} 与请求的 {origin} 不符")
    return env_cls(
        horizon=HORIZON, forecast_cutoff=FORECAST_CUTOFF, delta_t_hours=DELTA_T_HOURS,
        formal_injection=injection,
        task_seed=ENV_SEED_OFFSETS["task"], server_seed=ENV_SEED_OFFSETS["server"],
        forecast_seed=ENV_SEED_OFFSETS["forecast"],
    )


def initialize_policy(*, obs_dim: int, seed: int = POLICY_SEED):
    """固定种子初始化策略（**不做任何参数更新**；全局 RNG 由 fork_rng 还原）。"""
    from safe_rl_v2.policy import SafePPOPolicy

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        policy = SafePPOPolicy(obs_dim=obs_dim, action_dim=ACTION_DIM, hidden=POLICY_HIDDEN)
    policy.eval()
    return policy


def deterministic_action(policy, obs: np.ndarray) -> np.ndarray:
    """确定性均值动作（**有界** raw 动作）。

    与 `safe_rl_v2.policy._squash` 的**动作语义逐字相同**：
    计算维 `0.5*(tanh(u)+1) ∈ (0,1)`，储能维 `tanh(u) ∈ (-1,1)`；
    由 `tests/test_m6p1_controlled_run.py` 对齐断言（不重复实现第二套语义）。
    """
    with torch.no_grad():
        mean = policy.act_mean(torch.as_tensor(np.asarray(obs), dtype=torch.float32))
        compute = 0.5 * (torch.tanh(mean[: ACTION_DIM - 1]) + 1.0)
        storage = torch.tanh(mean[ACTION_DIM - 1:])
        raw = torch.cat([compute, storage]).numpy().astype(np.float32)
    return raw


def run_short_rollout(env, policy, *, steps: int, corrector_time_limit_s: float | None) -> dict:
    """受控 rollout：固定策略走 `steps` 步，**只采集，不更新参数**。"""
    if steps <= 0:
        raise ValueError(f"steps 必须为正，实际 {steps}")
    from planning.corrector import PRODUCTION_CORRECTOR_TIME_LIMIT_S
    from safe_rl.corrector_wrapper import CorrectorWrapper

    # 未显式给出预算时用**生产默认**（`planning.corrector` 的唯一来源）。
    budget = (PRODUCTION_CORRECTOR_TIME_LIMIT_S if corrector_time_limit_s is None
              else float(corrector_time_limit_s))
    wrapped = CorrectorWrapper(env, corrector_time_limit_s=budget)
    wrapped.reset(seed=0)
    completed = 0.0
    reasons: list[str] = []
    for _ in range(steps):
        obs = np.asarray(env._get_obs(), dtype=np.float32)
        _o, _r, terminated, truncated, info = wrapped.step(deterministic_action(policy, obs))
        completed += float(info["completed_work"])
        reasons.append(str(info.get("correction_reason", "none")))
        if terminated or truncated:
            break
    return {
        "steps": len(reasons),
        "completed_work": completed,
        "correction_reasons": sorted(set(reasons)),
        "corrector_time_limit_s": float(budget),
        "parameter_updates": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.controlled_run",
        description="受控短跑评估链路（M6-P1；不训练、不宣称性能）")
    parser.add_argument("--run-id", default="m6p1_controlled_short_run")
    parser.add_argument("--base-dir", default="runs")
    parser.add_argument("--origin", type=int, default=DEFAULT_ORIGIN)
    parser.add_argument("--steps", type=int, default=DEFAULT_SHORT_RUN_STEPS)
    args = parser.parse_args(argv)

    from runs.writer import git_revision, write_run

    base = REPO_ROOT / args.base_dir
    run_dir = base / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / CHECKPOINT_FILENAME
    command = (f"python -m evaluation.controlled_run --run-id {args.run_id} "
               f"--origin {args.origin} --steps {args.steps}")

    try:
        env = build_train_env(args.origin)
        policy = initialize_policy(obs_dim=int(env.obs_dim))
        rollout = run_short_rollout(env, policy, steps=int(args.steps),
                                    corrector_time_limit_s=None)
        digests = canonical_source_digests()
        ledger_hashes = source_ledger_hashes(env)
        checkpoint_info = save_evaluation_checkpoint(
            checkpoint_path,
            policy=policy,
            obs_dim=int(env.obs_dim),
            code_revision=git_revision(),
            artifact_role=CONTROLLED_ROLE,
            action_mode="deterministic_mean",
            policy_config={
                "hidden": POLICY_HIDDEN, "hidden_layers": 1, "activation": "Tanh",
                "obs_dim": int(env.obs_dim), "action_dim": ACTION_DIM,
            },
            train_split=SPLIT,
            train_origin=int(args.origin),
            train_start=DEFAULT_START,
            seeds=dict(ENV_SEED_OFFSETS),
            sources=[digest.model_dump() for digest in digests],
        )

        # 评估：**重新读取** checkpoint（并校验来源对当前资产的绑定）后再跑 episode
        loaded = load_evaluation_checkpoint(checkpoint_path)
        verify_evaluation_input_sources(loaded)
        # 评估与被评估的配置必须一致：受控短跑是「固定策略 + 单步修正器」，
        # 故评估 episode 同样经 `CorrectorWrapper`（生产默认预算），
        # 否则 `EvaluationRecord.correction` 会是不适用（None），与受控输入不符。
        from planning.corrector import PRODUCTION_CORRECTOR_TIME_LIMIT_S
        from safe_rl.corrector_wrapper import CorrectorWrapper

        eval_env = CorrectorWrapper(
            build_train_env(args.origin),
            corrector_time_limit_s=PRODUCTION_CORRECTOR_TIME_LIMIT_S)
        record = evaluate(
            eval_env,
            CONTROLLED_METHOD,
            lambda obs: deterministic_action(loaded.policy, obs),
            run_id=args.run_id,
            service_standard=None,          # 不显式传入 ⇒ 未判定（冻结标准不被隐式采用）
            seed=0,
            action_mode=loaded.action_mode,
            checkpoint_id=str(checkpoint_path),
            checkpoint_role=loaded.artifact_role,
        )
        report = {
            "entry": "python -m evaluation.controlled_run",
            "statement": STATEMENT,
            "claims": dict(CLAIMS),
            "split": SPLIT,
            "origin": int(args.origin),
            "start": DEFAULT_START,
            "horizon": HORIZON,
            "forecast_cutoff": FORECAST_CUTOFF,
            "action_dim": ACTION_DIM,
            "obs_dim": int(env.obs_dim),
            "policy_seed": POLICY_SEED,
            "env_seed_offsets": dict(ENV_SEED_OFFSETS),
            "action_mode": loaded.action_mode,
            "short_rollout": rollout,
            "checkpoint": checkpoint_info,
            "source_ledger_hashes": dict(ledger_hashes),
            "evaluation": record.model_dump(mode="json"),
            "planned_methods": list(PLANNED_METHODS),
            "planned_method_matrix": planned_method_rows([]),
            "planned_method_note": NOT_EVALUATED_STATEMENT,
            "code_revision": git_revision(),
        }
        config = {
            "entry": "python -m evaluation.controlled_run",
            "run_id": args.run_id,
            "status": "success",
            "split": SPLIT,
            "origin": int(args.origin),
            "horizon": HORIZON,
            "forecast_cutoff": FORECAST_CUTOFF,
            "policy_config": {
                "hidden": POLICY_HIDDEN, "hidden_layers": 1, "activation": "Tanh",
                "obs_dim": int(env.obs_dim), "action_dim": ACTION_DIM,
            },
            "policy_seed": POLICY_SEED,
            "env_seed_offsets": dict(ENV_SEED_OFFSETS),
            "action_mode": loaded.action_mode,
            "corrector_time_limit_s": rollout["corrector_time_limit_s"],
            "corrector_time_limit_source": "production_default",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_info["sha256"],
            "sources": [digest.model_dump() for digest in digests],
            "source_ledger_hashes": dict(ledger_hashes),
            "claims": dict(CLAIMS),
            "code_revision": git_revision(),
        }
    except Exception as exc:  # noqa: BLE001 - 失败也如实记录
        write_run(
            args.run_id,
            config={"entry": "python -m evaluation.controlled_run", "run_id": args.run_id,
                    "status": "failed"},
            metrics=pd.DataFrame(), base_dir=str(base), seed=POLICY_SEED, command=command,
            report={"entry": "python -m evaluation.controlled_run",
                    "statement": "本 run 失败，不代表任何评估或性能结果。",
                    "claims": dict(CLAIMS),
                    "failure": f"{type(exc).__name__}: {exc}"},
            status="failed", failure_classification=type(exc).__name__)
        print(f"受控短跑失败：{type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 1

    metrics = pd.DataFrame([{
        "method": record.method,
        "run_id": record.run_id,
        "action_mode": record.action_mode,
        "seed": record.seed,
        "steps": record.steps,
        "service_qualified": record.service_qualified,
        "purchase_cost_sgd": record.purchase_cost_sgd,
        "bess_degradation_cost_sgd": record.bess_degradation_cost_sgd,
        "mixed_objective_cost": record.mixed_objective_cost,
        "grid_energy_kwh": record.grid_energy_kwh,
        "carbon_kg_co2e": record.carbon_kg_co2e,
        "carbon_per_completed_work": record.carbon_per_completed_work,
        "completed_work": record.completed_work,
        "pv_available_kwh": record.pv.available_kwh,
        "pv_used_kwh": record.pv.used_kwh,
        "pv_curtail_kwh": record.pv.curtail_kwh,
        "pv_utilization": record.pv.utilization,
        "wind_available_kwh": record.wind.available_kwh,
        "wind_used_kwh": record.wind.used_kwh,
        "wind_curtail_kwh": record.wind.curtail_kwh,
        "wind_utilization": record.wind.utilization,
        "renewable_share": record.renewable_share,
        "grid_peak_kw": record.grid_peak_kw,
        "on_time_task_rate": record.service.on_time_task_rate,
        "on_time_work_rate": record.service.on_time_work_rate,
        "end_leftover_work": record.service.end_leftover_work,
        "end_leftover_work_fraction": record.service.end_leftover_work_fraction,
        "failed_tasks": record.service.failed_tasks,
        "non_interruptible_interruption_count":
            record.service.non_interruptible_interruption_count,
        "energy_conservation_violations": record.physical.energy_conservation_violations,
        "access_limit_violation_steps": record.physical.access_limit_violation_steps,
        "soc_violation_steps": record.physical.soc_violation_steps,
        "charge_discharge_exclusion_violations":
            record.physical.charge_discharge_exclusion_violations,
        "failure_classification": record.failure_classification,
        "not_computable": ",".join(record.not_computable),
    }])

    run_path = write_run(
        args.run_id, config=config, metrics=metrics, report=report,
        base_dir=str(base), seed=POLICY_SEED, command=command, status="success",
        dependency_lock_hash=ledger_hashes["dependency_lock_hash"],
        data_hash=ledger_hashes["data_hash"],
        scenario_hash=ledger_hashes["scenario_hash"],
        manifest_metadata={"checkpoint_sha256": checkpoint_info["sha256"],
                           "checkpoint_role": loaded.artifact_role})
    print(f"run 产物：{run_path}")
    print(f"checkpoint：{checkpoint_path}（sha256={checkpoint_info['sha256']}）")
    print(f"  service_qualified = {record.service_qualified}"
          f"（未判定：本链路未显式传入服务标准）")
    print(f"  购电费 {record.purchase_cost_sgd:.6f} SGD / 退化费 "
          f"{record.bess_degradation_cost_sgd:.6f} SGD / 碳排 "
          f"{record.carbon_kg_co2e:.6f} kgCO2e / 购电 "
          f"{record.grid_energy_kwh:.6f} kWh")
    print(f"  not_computable = {list(record.not_computable)}")
    print(f"  dependency_lock_hash={ledger_hashes['dependency_lock_hash']}")
    print(f"  data_hash={ledger_hashes['data_hash']}")
    print(f"  scenario_hash={ledger_hashes['scenario_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

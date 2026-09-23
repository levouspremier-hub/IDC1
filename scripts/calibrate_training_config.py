"""M1.3g-f-c-h：正式训练配置候选与 **train-only 预算标定**。

本脚本做三件事，**全部可重算**：

1. 按**预先写定**的等间隔规则，从 **verified train split** 取 24 个合法、不重叠的
   完整日 origin（`dh.4`）；**不读 validation/test**；
2. 在**同一** formal env、`corrector=on` / `0.25 s`、`horizon=48` 下，跑**三种固定
   raw 提案**（20 计算维全 1.0 / 0.5 / 0.25，储能维 0）——**无学习参考策略**，
   **不是**训练结果；
3. 按 `dh.6` 公式标定 `business` / `carbon` 的 **budget** 与**乘子**缩放。

产物：`runs/<run_id>/{config.yaml,metrics.parquet,report.json,figures/,manifest.json}`
（按仓库规范，**失败也如实记录**）。

用法：

```bash
uv run python -m scripts.calibrate_training_config --run-id <id>
uv run python -m scripts.calibrate_training_config --help
```
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

SPLIT = "train"
SLOTS_PER_DAY = 48          # 24 h / 0.5 h
HORIZON = 48                # 24 小时 episode
FORECAST_CUTOFF = 48
DELTA_T_HOURS = 0.5
N_ORIGINS = 24

PROPOSALS: tuple[float, ...] = (1.0, 0.5, 0.25)

# 标定公式常量（dh.6）
MULTIPLIER_LR_BASE = 0.01
MULTIPLIER_CAP_BASE = 10.0
MULTIPLIER_INIT = 0.0

CORRECTOR_TIME_LIMIT_S = 0.25   # 唯一生产预算（M5.4i）

# 分类依据：`planning/corrector.py:1-11` 的**权威失败语义**：
#   - `none` / `deadline_shortfall`：**MIP 最优、可执行**第 0 步候选
#     （deadline_shortfall 仅**业务风险**标记）；
#   - `timeout` / `base_shortage` / `solver_failure`：**零动作回退**，候选解**不执行**；
#   - `proposal_invalid`：raw proposal 非法，同样**零动作回退**。
BENIGN_CORRECTION_REASONS = frozenset({"none", "deadline_shortfall"})
ZERO_ACTION_FALLBACK_REASONS = frozenset({"timeout", "base_shortage",
                                          "solver_failure", "proposal_invalid"})

# 环境种子固定偏移（由主种子派生；本卡只用主种子 0）
SEED_OFFSETS = {"task": 0, "server": 1, "forecast": 300000}
MASTER_SEED = 0


class CalibrationError(RuntimeError):
    """标定过程中的明确失败。"""


# --- 24 个 train origin 的**预先写定**规则（dh.4） ---------------------------

def day_aligned_origins() -> tuple[int, ...]:
    """日对齐 origin 集合 `{48*(1+k)}`，上界满足 `origin + 48 <= 10224`。"""
    from scenario.b6_split_manifests import load_verified_split_manifest_v5

    payload = load_verified_split_manifest_v5(expected_split=SPLIT)
    end = int(payload["split_rows"]["end_exclusive"])
    start = int(payload["candidate_origins"]["start"])
    last = end - HORIZON
    if start % SLOTS_PER_DAY or last % SLOTS_PER_DAY:
        raise CalibrationError(
            f"日对齐假设不成立：start={start} last={last} 非 {SLOTS_PER_DAY} 的整数倍")
    return tuple(range(start, last + 1, SLOTS_PER_DAY))


def select_origins(n: int = N_ORIGINS) -> tuple[int, ...]:
    """等间隔取 `n` 个：`index_k = floor(k * (N-1) / (n-1))`。"""
    pool = day_aligned_origins()
    total = len(pool)
    if n > total:
        raise CalibrationError(f"可用日 origin 仅 {total} 个，不足 {n} 个")
    idx = [(k * (total - 1)) // (n - 1) for k in range(n)]
    if len(set(idx)) != n:
        raise CalibrationError(f"等间隔索引出现重复：{idx}")
    return tuple(pool[i] for i in idx)


def start_for_origin(origin: int) -> str:
    """`local_origin` -> 规范带时区 ISO start（经 canonical 网格）。"""
    from scenario.formal_scenario_b6 import _canonical_parquet_path  # noqa: PLC2701

    stamps = pd.DatetimeIndex(pd.read_parquet(_canonical_parquet_path())["timestamp"])
    if not (0 <= origin < len(stamps)):
        raise CalibrationError(f"origin {origin} 越出 canonical 网格")
    start = stamps[origin].isoformat()

    from scenario.b6_split_manifests import local_origin_from_start

    if local_origin_from_start(SPLIT, start) != origin:
        raise CalibrationError(f"origin {origin} 往返映射失败：start={start}")
    return start


# --- 固定 raw 提案 -----------------------------------------------------------

def reference_action(compute_value: float, action_dim: int, n_groups: int) -> np.ndarray:
    """20 个计算维全取 `compute_value`，储能维 0。"""
    action = np.zeros(action_dim, dtype=np.float64)
    action[:n_groups] = float(compute_value)
    return action


def build_env_for_origin(origin: int):
    """在建formal env（`corrector` 由调用方包装）。"""
    import importlib

    injection_module = importlib.import_module("scenario.env_injection")
    env_cls = importlib.import_module("envs.idc_price_env").IDCPriceEnv20D
    from scenario.arrival_mapper import load_verified_mapper_chain

    load_verified_mapper_chain(SPLIT)
    injection = injection_module.build_verified_formal_env_injection(
        SPLIT, start=start_for_origin(origin), horizon=HORIZON,
        forecast_cutoff=FORECAST_CUTOFF)
    env = env_cls(horizon=HORIZON, forecast_cutoff=FORECAST_CUTOFF,
                  delta_t_hours=DELTA_T_HOURS, formal_injection=injection,
                  task_seed=MASTER_SEED + SEED_OFFSETS["task"],
                  server_seed=MASTER_SEED + SEED_OFFSETS["server"],
                  forecast_seed=MASTER_SEED + SEED_OFFSETS["forecast"])
    return env, injection


def run_reference_proposal(compute_value: float, origins: tuple[int, ...]) -> dict[str, Any]:
    """在 24 个 origin 上跑一个固定 raw 提案（**无学习参考策略**）。

    返回每 origin 的 `transitions / business_sum / carbon_sum / failures / timeouts`
    与资产 hash。**这不是训练结果。**
    """
    from safe_rl.corrector_wrapper import CorrectorWrapper
    from safe_rl_v2.rollout import (
        BUSINESS_VIOLATION_INFO_KEY,
        CARBON_EMISSION_INFO_KEY,
    )

    per_origin: list[dict[str, Any]] = []
    total_transitions = 0
    business_sum = 0.0
    carbon_sum = 0.0
    failures = 0
    timeouts = 0

    for origin in origins:
        env, injection = build_env_for_origin(origin)
        wrapped = CorrectorWrapper(env, corrector_time_limit_s=CORRECTOR_TIME_LIMIT_S)
        obs, _info = wrapped.reset(seed=MASTER_SEED)
        action = reference_action(compute_value, env.action_dim, env.model.N)

        o_transitions = 0
        o_business = 0.0
        o_carbon = 0.0
        o_fallbacks = 0
        o_deadline_shortfall = 0
        o_reasons: dict[str, int] = {}

        for _step in range(HORIZON):
            obs, _reward, terminated, truncated, info = wrapped.step(action)
            reason = str(info.get("correction_reason", "none"))
            o_reasons[reason] = o_reasons.get(reason, 0) + 1
            if reason in ZERO_ACTION_FALLBACK_REASONS:
                o_fallbacks += 1
            elif reason == "deadline_shortfall":
                o_deadline_shortfall += 1
            elif reason not in BENIGN_CORRECTION_REASONS:
                raise CalibrationError(f"未知 correction_reason：{reason!r}")
            o_business += float(info[BUSINESS_VIOLATION_INFO_KEY])
            o_carbon += float(info[CARBON_EMISSION_INFO_KEY])
            o_transitions += 1
            if terminated or truncated:
                break

        per_origin.append({
            "origin": int(origin),
            "start": start_for_origin(origin),
            "transitions": o_transitions,
            "business_sum": o_business,
            "carbon_sum": o_carbon,
            "zero_action_fallbacks": o_fallbacks,
            "deadline_shortfall": o_deadline_shortfall,
            "reasons": dict(sorted(o_reasons.items())),
            "ledger_micro_sum": int(sum(injection.ledger_micro)),
        })
        total_transitions += o_transitions
        business_sum += o_business
        carbon_sum += o_carbon
        failures += o_fallbacks
        timeouts += o_reasons.get("timeout", 0)

    if total_transitions == 0:
        raise CalibrationError(f"提案 compute={compute_value} 未采到任何 transition")

    return {
        "compute_value": float(compute_value),
        "storage_value": 0.0,
        "origins": [int(o) for o in origins],
        "per_origin": per_origin,
        "transitions": total_transitions,
        "business_sum": business_sum,
        "carbon_sum": carbon_sum,
        "business_mean": business_sum / total_transitions,
        "carbon_mean": carbon_sum / total_transitions,
        "zero_action_fallbacks": failures,
        "timeouts": timeouts,
        "deadline_shortfall": sum(p["deadline_shortfall"] for p in per_origin),
        "reasons": dict(sorted(
            collections.Counter(
                k for p in per_origin for k, v in p["reasons"].items()
                for _ in range(v)).items())),
        "reasons_note": ("`none` / `deadline_shortfall` 均为 **MIP 最优、可执行**"
                         "（后者仅业务风险标记）；`timeout` / `base_shortage` / `solver_failure` / "
                         "`proposal_invalid` 为**零动作回退**（候选解不执行）。"),
    }


def calibrate_budgets(results: list[dict[str, Any]]) -> dict[str, Any]:
    """按 `dh.6` 标定 budget：business 取三提案最小；carbon 取达标提案中最低。"""
    business_budget = min(r["business_mean"] for r in results)
    qualified = [r for r in results if r["business_mean"] == business_budget]
    carbon_budget = min(r["carbon_mean"] for r in qualified)

    business_ref = next(r for r in results if r["business_mean"] == business_budget)
    carbon_ref = next(r for r in qualified if r["carbon_mean"] == carbon_budget)
    return {
        "business_budget": business_budget,
        "business_budget_unit": "violation_task_steps (per transition mean)",
        "business_budget_from_proposal": business_ref["compute_value"],
        "carbon_budget": carbon_budget,
        "carbon_budget_unit": "kgCO2e (per transition mean)",
        "carbon_budget_from_proposal": carbon_ref["compute_value"],
        "qualified_proposals": [r["compute_value"] for r in qualified],
    }


def multiplier_scaling(results: list[dict[str, Any]], budgets: dict[str, Any]) -> dict[str, Any]:
    """`scale = max(1, 参考均值)`；`lr = 0.01/scale^2`；`cap = 10/scale`。"""
    ref_by_proposal = {r["compute_value"]: r for r in results}

    def one(name: str, mean_key: str, from_key: str) -> dict[str, Any]:
        ref = ref_by_proposal[budgets[from_key]]
        mean = float(ref[mean_key])
        scale = max(1.0, mean)
        return {
            "constraint": name,
            "reference_proposal": budgets[from_key],
            "reference_transition_mean": mean,
            "scale": scale,
            "scale_formula": "max(1, 对应参考策略的该约束逐 transition 均值)",
            "learning_rate": MULTIPLIER_LR_BASE / (scale ** 2),
            "learning_rate_formula": "0.01 / scale^2",
            "max_multiplier": MULTIPLIER_CAP_BASE / scale,
            "max_multiplier_formula": "10 / scale",
            "initial_multiplier": MULTIPLIER_INIT,
        }

    return {
        "business": one("business", "business_mean", "business_budget_from_proposal"),
        "carbon": one("carbon", "carbon_mean", "carbon_budget_from_proposal"),
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def asset_hashes() -> dict[str, str]:
    """本次标定实际消费的冻结资产 hash。"""
    files = {
        "refs_v4": REPO_ROOT / "configs/frozen_refs/refs_v4.json",
        "formal_split_v5_train": REPO_ROOT / "data/manifest/formal_splits_v5/train.json",
        "m13g_arrival_mapper_v1": REPO_ROOT / "data/manifest/m13g_arrival_mapper_v1.json",
        "env_release_v1": REPO_ROOT / "configs/release/idc_formal_env_release_v1.json",
        "canonical_parquet": REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet",
        "exogenous_v3_parquet":
            REPO_ROOT / "data/processed/singapore_2024/exogenous_drivers_v3.parquet",
    }
    return {name: _sha256(path) for name, path in files.items()}


def training_config_candidate(
    budgets: dict[str, Any], multipliers: dict[str, Any], obs_dim: int,
) -> dict[str, Any]:
    """机器可读的正式训练配置候选（**候选**，非冻结）。"""
    return {
        "schema": "idc-training-config-candidate-v1",
        "status": "candidate_not_frozen",
        "note": ("本文件是**候选**；冻结须另开卡。"
                 "docs/training_config_candidates.json 为历史审计，分级不予改写。"),
        "corrector": {
            "mode": "on",
            "time_limit_s": CORRECTOR_TIME_LIMIT_S,
            "source": "production_default (M5.4i)",
            "note": "训练/评估同一修正语义；off 留作日后独立重训的机制对照。",
        },
        "backend": {"device": "cpu", "torch_num_threads": 1,
                    "note": ("求解器既有确定性选项不改"
                             "（DETERMINISTIC_RANDOM_SEED=0, DETERMINISTIC_PARALLEL=False）。")},
        "policy": {"hidden": 64, "hidden_layers": 1, "activation": "Tanh",
                   "log_std_init": 0.0, "action_dim": 21, "critic_heads": 3,
                   "obs_dim": int(obs_dim),
                   "obs_dim_source": "formal env（horizon=48 时实测；不硬编码）"},
        "optimizer": {"name": "Adam", "lr": 3e-4, "betas": [0.9, 0.999],
                      "eps": 1e-8, "weight_decay": 0.0, "schedule": "none"},
        "ppo": {"clip_epsilon": 0.2, "gae_lambda": 0.95,
                "gamma_per_hour": 0.99,
                "gamma_per_step": float(np.sqrt(0.99)),
                "gamma_formula": "sqrt(0.99)（半小时一步，按每小时 0.99 定义）"},
        "sampling": {"horizon": HORIZON, "episodes_per_batch": 4,
                     "transitions_per_batch": HORIZON * 4,
                     "epochs_per_batch": 4, "minibatch_size": 48,
                     "minibatches_per_epoch": (HORIZON * 4) // 48,
                     "frozen_within_batch": ["old_raw_log_prob", "advantage", "critic_target"],
                     "lagrangian_updates_per_batch": 1},
        "scale": {"batches_per_seed": 512,
                  "transitions_per_seed": 512 * HORIZON * 4,
                  "train_seeds": [0, 1, 2],
                  "seed_offsets": dict(SEED_OFFSETS),
                  "seed_note": "各 RNG 由主种子按固定偏移显式派生；不重播种全局 RNG。"},
        "budgets": budgets,
        "multipliers": multipliers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.calibrate_training_config",
        description="正式训练配置候选与 train-only 预算标定（M1.3g-f-c-h）")
    parser.add_argument("--run-id", default="m13gch_calibration")
    parser.add_argument("--base-dir", default="runs")
    args = parser.parse_args(argv)

    from runs.writer import write_run

    origins = select_origins()
    gaps = [(b - a) // SLOTS_PER_DAY for a, b in zip(origins, origins[1:], strict=False)]
    base = REPO_ROOT / args.base_dir

    try:
        results = [run_reference_proposal(v, origins) for v in PROPOSALS]
        budgets = calibrate_budgets(results)
        multipliers = multiplier_scaling(results, budgets)
        _env, _inj = build_env_for_origin(origins[0])
        obs_dim = int(_env.obs_dim)
        candidate = training_config_candidate(budgets, multipliers, obs_dim)
        hashes = asset_hashes()
    except Exception as exc:  # noqa: BLE001 - 失败也如实记录
        write_run(
            args.run_id,
            config={"entry": "python -m scripts.calibrate_training_config",
                    "run_id": args.run_id, "status": "failed"},
            metrics=pd.DataFrame(), base_dir=str(base), seed=MASTER_SEED,
            command=f"python -m scripts.calibrate_training_config --run-id {args.run_id}",
            report={"entry": "python -m scripts.calibrate_training_config",
                    "statement": "本 run 失败，不代表任何训练或标定结果。",
                    "claims": {"trained": False, "performance_evaluated": False,
                               "convergence_claimed": False},
                    "failure": f"{type(exc).__name__}: {exc}"},
            status="failed", failure_classification=type(exc).__name__)
        print(f"标定失败：{type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 1

    rows = [{"proposal_compute": r["compute_value"],
             "transitions": r["transitions"],
             "business_mean": r["business_mean"],
             "carbon_mean": r["carbon_mean"],
             "zero_action_fallbacks": r["zero_action_fallbacks"],
             "timeouts": r["timeouts"],
             "deadline_shortfall": r["deadline_shortfall"]}
            for r in results]
    report = {
        "entry": "python -m scripts.calibrate_training_config",
        "statement": ("train-only 预算标定：三种**无学习参考策略**在 24 个 verified "
                      "train origin 上的结果。**不是训练结果**，不宣称收敛或性能。"),
        "claims": {"trained": False, "performance_evaluated": False,
                   "convergence_claimed": False},
        "origins": list(origins),
        "origin_rule": "index_k = floor(k*(212-1)/23); origin = 48 + 48*index_k",
        "min_origin_gap_days": min(gaps),
        "proposals": [{"compute_value": r["compute_value"],
                       "storage_value": r["storage_value"],
                       "transitions": r["transitions"],
                       "business_mean": r["business_mean"],
                       "carbon_mean": r["carbon_mean"],
                       "zero_action_fallbacks": r["zero_action_fallbacks"],
                       "timeouts": r["timeouts"],
                       "deadline_shortfall": r["deadline_shortfall"],
                       "reasons": r["reasons"],
                       "per_origin": r["per_origin"]}
                      for r in results],
        "budgets": budgets,
        "multipliers": multipliers,
        "asset_hashes": hashes,
        "obs_dim": obs_dim,
    }

    run_dir = write_run(
        args.run_id,
        config=candidate,
        metrics=pd.DataFrame(rows),
        report=report,
        base_dir=str(base), seed=MASTER_SEED,
        command=f"python -m scripts.calibrate_training_config --run-id {args.run_id}",
        status="success", manifest_metadata={"asset_hashes": hashes})

    # 机器可读候选（入库路径）
    out = REPO_ROOT / "configs/training/idc_training_config_candidate_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(candidate, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")

    print(f"run 产物：{run_dir}")
    print(f"配置候选：{out}")
    for r in results:
        print(f"  提案 compute={r['compute_value']}: transitions={r['transitions']} "
              f"business_mean={r['business_mean']:.10f} carbon_mean={r['carbon_mean']:.10f} "
              f"zero_action_fallbacks={r['zero_action_fallbacks']} "
              f"timeouts={r['timeouts']} deadline_shortfall={r['deadline_shortfall']}")
    print(f"business_budget={budgets['business_budget']:.10f} "
          f"(来自 compute={budgets['business_budget_from_proposal']})")
    print(f"carbon_budget  ={budgets['carbon_budget']:.10f} "
          f"(来自 compute={budgets['carbon_budget_from_proposal']})")
    for name, m in multipliers.items():
        print(f"  乘子 {name}: scale={m['scale']:.6f} lr={m['learning_rate']:.10f} "
              f"cap={m['max_multiplier']:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""M1.3g-f-c-h1：业务约束**敏感性诊断**。

目标：查清「三种计算强度下业务结果相同」究竟由**哪一层**造成 ——
环境执行 / 任务可用量 / 物理容量 / 修正器投影。

做法：

1. 仅用 verified train 的 **origin 48 / 4848 / 10176**，**固定同一环境种子**；
2. 比较 `compute = 0 / 0.25 / 1.0`（储能 = 0）× `corrector on / off`；
3. **逐步**记录八字段，找出**最早分叉步骤**；
4. 在**相同 origin、相同提案**下比较**预测窗口 4 / 48**。

**`corrector=off` 只用于定位动作作用**，**不得**称为安全或服务合格的基线。

**不**修改 env / scenario / planning 求解语义 / PPO 核心 / 冻结资产 / release；
**不**冻结任何 budget。

用法：

```bash
uv run python -m scripts.diagnose_business_sensitivity --help
uv run python -m scripts.diagnose_business_sensitivity --run-id <id>
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

SPLIT = "train"
DIAGNOSTIC_ORIGINS: tuple[int, ...] = (48, 4848, 10176)
COMPUTE_LEVELS: tuple[float, ...] = (0.0, 0.25, 1.0)
FORECAST_CUTOFFS: tuple[int, ...] = (4, 48)
HORIZON = 48
DELTA_T_HOURS = 0.5
STORAGE_ACTION = 0.0

# 固定同一环境种子（诊断变量只有 compute / corrector / forecast_cutoff）
FIXED_SEED = 0
SEED_KWARGS = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}

BENIGN_REASONS = frozenset({"none", "deadline_shortfall"})

# 逐 step 记录的字段（`business_gap` 仅 corrector on 时存在）
STEP_FIELDS: tuple[str, ...] = (
    "raw_action", "exec_action", "planned_capacity", "completed_work",
    "business_gap", "sla_violation_count", "deadline_shortfall", "carbon",
)

CLAIMS = {"trained": False, "performance_evaluated": False, "convergence_claimed": False}

OFF_BASELINE_DISCLAIMER = ("corrector=off 只用于定位动作作用，"
                           "**不得**称为安全或服务合格的基线。")

BUSINESS_GAP_NOTE = ("`business_gap` 由 safe_rl/corrector_wrapper.py:63 提供，"
                     "**仅 corrector on 时存在**；off 时记为**不适用**（None），"
                     "**不是** 0。")


class DiagnosticError(RuntimeError):
    """诊断过程中的明确失败。"""


def start_for_origin(origin: int) -> str:
    from scenario.formal_scenario_b6 import _canonical_parquet_path  # noqa: PLC2701

    stamps = pd.DatetimeIndex(pd.read_parquet(_canonical_parquet_path())["timestamp"])
    start = stamps[origin].isoformat()

    from scenario.b6_split_manifests import local_origin_from_start

    if local_origin_from_start(SPLIT, start) != origin:
        raise DiagnosticError(f"origin {origin} 往返映射失败")
    return start


def build_env(origin: int, *, forecast_cutoff: int):
    """同一 formal env；诊断**不**改任何 env 语义。"""
    import importlib

    from scenario.arrival_mapper import load_verified_mapper_chain

    injection_module = importlib.import_module("scenario.env_injection")
    env_cls = importlib.import_module("envs.idc_price_env").IDCPriceEnv20D

    load_verified_mapper_chain(SPLIT)
    injection = injection_module.build_verified_formal_env_injection(
        SPLIT, start=start_for_origin(origin), horizon=HORIZON,
        forecast_cutoff=forecast_cutoff)
    env = env_cls(horizon=HORIZON, forecast_cutoff=forecast_cutoff,
                  delta_t_hours=DELTA_T_HOURS, formal_injection=injection,
                  **SEED_KWARGS)
    return env, injection


def fixed_action(compute: float, action_dim: int, n_groups: int) -> np.ndarray:
    action = np.zeros(action_dim, dtype=np.float64)
    action[:n_groups] = float(compute)
    action[n_groups] = STORAGE_ACTION
    return action


def _exec_env(env, *, corrector_on: bool, budget: float | None):
    if not corrector_on:
        return env
    if budget is None:
        raise DiagnosticError("corrector=on 时预算必须由调用方给出")
    from safe_rl.corrector_wrapper import CorrectorWrapper

    return CorrectorWrapper(env, corrector_time_limit_s=float(budget))


def run_episode(origin: int, *, compute: float, corrector_on: bool,
                forecast_cutoff: int, budget: float | None) -> dict[str, Any]:
    """跑一个 episode，**逐 step** 记录八字段。"""
    env, injection = build_env(origin, forecast_cutoff=forecast_cutoff)
    wrapped = _exec_env(env, corrector_on=corrector_on, budget=budget)
    wrapped.reset(seed=FIXED_SEED)
    action = fixed_action(compute, env.action_dim, env.model.N)

    steps: list[dict[str, Any]] = []
    for step in range(HORIZON):
        _obs, _reward, terminated, truncated, info = wrapped.step(action)
        raw = np.asarray(info.get("raw_action", action), dtype=np.float64)
        exec_ = np.asarray(info.get("exec_action", action), dtype=np.float64)
        steps.append({
            "step": step,
            "raw_action_head": [float(v) for v in raw[:3]],
            "exec_action_head": [float(v) for v in exec_[:3]],
            "exec_equals_raw": bool(np.array_equal(raw, exec_)),
            "planned_capacity": float(info["planned_capacity"]),
            "completed_work": float(info["completed_work"]),
            # on 时由 corrector 提供；off 时**不适用**（记 None，不是 0）
            "business_gap": (None if "business_gap" not in info
                             else float(info["business_gap"])),
            "sla_violation_count": int(info["sla_violation_count"]),
            "deadline_shortfall": (None if "deadline_shortfall_work" not in info
                                   else float(info["deadline_shortfall_work"])),
            "carbon": float(info["carbon_emission"]),
            "correction_reason": str(info.get("correction_reason", "n/a")),
        })
        if terminated or truncated:
            break

    return {
        "origin": int(origin),
        "forecast_cutoff": int(forecast_cutoff),
        "compute": float(compute),
        "corrector_on": bool(corrector_on),
        "steps": steps,
        "ledger_micro_sum": int(sum(injection.ledger_micro)),
    }


def first_divergence(a: dict[str, Any], b: dict[str, Any], field: str) -> int | None:
    """最早出现 `field` 差异的 step（无差异返回 None）。"""
    for sa, sb in zip(a["steps"], b["steps"], strict=False):
        if sa[field] != sb[field]:
            return int(sa["step"])
    return None


def _summarise(episodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """配对比较：compute 0 vs 0.25 vs 1.0（同 corrector / 同 cutoff）。"""
    out: dict[str, Any] = {}
    for suffix in ("on", "off"):
        keys = [f"{suffix}_compute{c}" for c in COMPUTE_LEVELS]
        base = episodes[keys[0]]
        cmp025, cmp100 = episodes[keys[1]], episodes[keys[2]]
        entry: dict[str, Any] = {}
        for field in ("exec_action_head", "planned_capacity", "completed_work",
                      "sla_violation_count", "carbon"):
            entry[f"first_divergence_0_vs_0.25_{field}"] = first_divergence(
                base, cmp025, field)
            entry[f"first_divergence_0_vs_1.0_{field}"] = first_divergence(
                base, cmp100, field)
        entry["sla_violation_count_total"] = {
            "compute0": sum(s["sla_violation_count"] for s in base["steps"]),
            "compute025": sum(s["sla_violation_count"] for s in cmp025["steps"]),
            "compute100": sum(s["sla_violation_count"] for s in cmp100["steps"]),
        }
        entry["completed_work_total"] = {
            "compute0": sum(s["completed_work"] for s in base["steps"]),
            "compute025": sum(s["completed_work"] for s in cmp025["steps"]),
            "compute100": sum(s["completed_work"] for s in cmp100["steps"]),
        }
        out[suffix] = entry
    return out


def _cutoff_summary(episodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for compute in COMPUTE_LEVELS:
        a, b = episodes[f"on_compute{compute}_c4"], episodes[f"on_compute{compute}_c48"]
        out[f"compute{compute}"] = {
            "first_divergence_completed_work": first_divergence(
                a, b, "completed_work"),
            "first_divergence_sla_violation_count": first_divergence(
                a, b, "sla_violation_count"),
            "sla_total_c4": sum(s["sla_violation_count"] for s in a["steps"]),
            "sla_total_c48": sum(s["sla_violation_count"] for s in b["steps"]),
            "completed_work_total_c4": sum(s["completed_work"] for s in a["steps"]),
            "completed_work_total_c48": sum(s["completed_work"] for s in b["steps"]),
        }
    return out


def diagnose(budget: float, run_id: str) -> dict[str, Any]:
    """在三个 origin 上跑完整诊断矩阵。"""
    per_origin: list[dict[str, Any]] = []
    for origin in DIAGNOSTIC_ORIGINS:
        episodes: dict[str, dict[str, Any]] = {}
        for corrector_on, suffix in ((True, "on"), (False, "off")):
            for compute in COMPUTE_LEVELS:
                key = f"{suffix}_compute{compute}"
                episodes[key] = run_episode(
                    origin, compute=compute, corrector_on=corrector_on,
                    forecast_cutoff=FORECAST_CUTOFFS[-1],
                    budget=budget if corrector_on else None)
        # 预测窗口对照（corrector on，两个 cutoff）
        for compute in COMPUTE_LEVELS:
            for cutoff in FORECAST_CUTOFFS:
                episodes[f"on_compute{compute}_c{cutoff}"] = run_episode(
                    origin, compute=compute, corrector_on=True,
                    forecast_cutoff=cutoff, budget=budget)
        per_origin.append({
            "origin": int(origin),
            "episodes": episodes,
            "pairwise": _summarise(episodes),
            "cutoff": _cutoff_summary(episodes),
        })
    return {
        "entry": "python -m scripts.diagnose_business_sensitivity",
        "probe_only": True,
        "run_id": run_id,
        "origins": list(DIAGNOSTIC_ORIGINS),
        "compute_levels": list(COMPUTE_LEVELS),
        "forecast_cutoffs": list(FORECAST_CUTOFFS),
        "fixed_seed": FIXED_SEED,
        "seed_kwargs": dict(SEED_KWARGS),
        "corrector_time_limit_s": float(budget),
        "per_origin": per_origin,
        "claims": dict(CLAIMS),
        "off_is_not_a_baseline": OFF_BASELINE_DISCLAIMER,
        "business_gap_note": BUSINESS_GAP_NOTE,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.diagnose_business_sensitivity",
        description="业务约束敏感性诊断（M1.3g-f-c-h1）")
    parser.add_argument("--run-id", default="m13gch1_diagnosis")
    parser.add_argument("--base-dir", default="runs")
    args = parser.parse_args(argv)

    from planning.corrector import resolve_corrector_budget
    from runs.writer import write_run

    budget, source = resolve_corrector_budget(None, enabled=True)
    base = REPO_ROOT / args.base_dir

    try:
        report = diagnose(float(budget), args.run_id)
        report["corrector_time_limit_source"] = source
    except Exception as exc:  # noqa: BLE001 - 失败也如实记录
        write_run(
            args.run_id,
            config={"entry": "python -m scripts.diagnose_business_sensitivity",
                    "run_id": args.run_id, "status": "failed"},
            metrics=pd.DataFrame(), base_dir=str(base), seed=FIXED_SEED,
            command=("python -m scripts.diagnose_business_sensitivity "
                     f"--run-id {args.run_id}"),
            report={"entry": "python -m scripts.diagnose_business_sensitivity",
                    "statement": "本 run 失败，不代表任何诊断结论。",
                    "claims": dict(CLAIMS),
                    "failure": f"{type(exc).__name__}: {exc}"},
            status="failed", failure_classification=type(exc).__name__)
        print(f"诊断失败：{type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 1

    rows = [
        {"origin": po["origin"], "corrector_on": ep["corrector_on"],
         "compute": ep["compute"], "forecast_cutoff": ep["forecast_cutoff"],
         "steps": len(ep["steps"]),
         "sla_total": sum(s["sla_violation_count"] for s in ep["steps"]),
         "completed_work_total": sum(s["completed_work"] for s in ep["steps"]),
         "carbon_total": sum(s["carbon"] for s in ep["steps"])}
        for po in report["per_origin"] for ep in po["episodes"].values()
    ]
    run_dir = write_run(
        args.run_id, config=report, metrics=pd.DataFrame(rows), report=report,
        base_dir=str(base), seed=FIXED_SEED,
        command=(f"python -m scripts.diagnose_business_sensitivity "
                 f"--run-id {args.run_id}"),
        status="success")

    print(f"run 产物：{run_dir}")
    print(f"corrector 预算 = {budget}（来源 {source}）")
    for po in report["per_origin"]:
        pw = po["pairwise"]["on"]
        print(f"  origin {po['origin']}: corrector=on "
              f"sla_total={pw['sla_violation_count_total']}")
        print(f"    first_divergence 0vs1.0: "
              f"completed_work={pw['first_divergence_0_vs_1.0_completed_work']} "
              f"exec_action={pw['first_divergence_0_vs_1.0_exec_action_head']} "
              f"planned_capacity={pw['first_divergence_0_vs_1.0_planned_capacity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""M4.6a corrector 时间预算矩阵 benchmark（**无 tracing 干扰**）。

计时只用 `time.perf_counter`，**绝不启用 `tracemalloc`**（其开销实测可膨胀约 6.7×）。
本 benchmark 的数值仅供后续 **M5/M9 人工选择预算**参考，
**不是**正式实验结论，也**不是** RL 训练吞吐结论。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许以 `python scripts/benchmark_corrector.py` 直接运行（补仓库根到 sys.path）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from envs.idc_price_env import IDCPriceEnv20D
from safe_rl.corrector_wrapper import CorrectorWrapper

# 固定预算矩阵（秒）
BUDGET_MATRIX_S = [0.01, 0.02, 0.05, 0.10, 0.25]

# 场景：正常接入 / 紧接入 + 储能压力
SCENARIOS = {
    "normal": {"access_limit_kw": 1000.0, "soc_init": 0.5},
    "tight": {"access_limit_kw": 18.0, "soc_init": 0.25},
}

N_GROUP = 20
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_EPISODES = 1
DEFAULT_HORIZON = 24
DEFAULT_WARMUP_STEPS = 2


def _stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean_s": 0.0, "median_s": 0.0, "p95_s": 0.0, "max_s": 0.0, "n": 0}
    return {
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
        "max_s": float(arr.max()),
        "n": int(arr.size),
    }


def _action() -> np.ndarray:
    return np.concatenate(
        [np.full(N_GROUP, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)]
    )


def _measure_raw_env(seed: int, horizon: int, warmup: int) -> list[float]:
    env = IDCPriceEnv20D(horizon=horizon)
    env.reset(seed=seed)
    a = _action()
    for _ in range(warmup):
        env.step(a)
    times: list[float] = []
    for _ in range(horizon):
        t0 = time.perf_counter()
        _, _, term, trunc, _ = env.step(a)
        times.append(time.perf_counter() - t0)
        if term or trunc:
            break
    return times


def _measure_budget(scenario: str, budget_s: float, seeds: list[int],
                    episodes: int, horizon: int, warmup: int) -> dict:
    cfg = SCENARIOS[scenario]
    wrapper_times: list[float] = []
    corrector_times: list[float] = []
    stage_a: list[float] = []
    stage_b: list[float] = []
    timeout_count = 0
    optimal_count = 0
    zero_action_count = 0
    business_gap_sum = 0.0
    n_steps = 0

    for seed in seeds:
        for ep in range(episodes):
            env = CorrectorWrapper(
                IDCPriceEnv20D(
                    horizon=horizon,
                    access_limit_kw=cfg["access_limit_kw"],
                    bess_soc_init=cfg["soc_init"],
                ),
                corrector_time_limit_s=float(budget_s),
            )
            env.reset(seed=seed + ep)
            a = _action()
            for _ in range(warmup):          # 固定 warm-up，不计入
                env.step(a)
            for _ in range(horizon):
                t0 = time.perf_counter()
                _, _, term, trunc, info = env.step(a)
                wrapper_times.append(time.perf_counter() - t0)
                n_steps += 1

                reason = str(info.get("correction_reason", ""))
                if reason == "timeout":
                    timeout_count += 1
                    zero_action_count += 1   # timeout 即安全零动作回退
                else:
                    optimal_count += 1
                if np.allclose(np.asarray(info.get("exec_action", [])), 0.0):
                    zero_action_count += 1 if reason != "timeout" else 0
                business_gap_sum += float(info.get("business_gap", 0.0))

                corrector_times.append(float(info.get("correction_solve_time_s", 0.0)))
                stage_a.append(float(info.get("stage_a_solve_time_s", 0.0)))
                stage_b.append(float(info.get("stage_b_solve_time_s", 0.0)))
                if term or trunc:
                    break

    total = max(sum(wrapper_times), 1e-9)
    return {
        "budget_s": float(budget_s),
        "n_steps": n_steps,
        "wrapper_step": _stats(wrapper_times),
        "corrector_total": _stats(corrector_times),
        "stage_a": _stats(stage_a),
        "stage_b": _stats(stage_b),
        "steps_per_s": float(len(wrapper_times) / total),
        "timeout_count": timeout_count,
        "timeout_rate": float(timeout_count / max(n_steps, 1)),
        "optimal_count": optimal_count,
        "zero_action_fallback_count": zero_action_count,
        "business_gap_sum": float(business_gap_sum),
    }


def run_benchmark(
    *, seeds=None, budgets=None, scenarios=None,
    episodes: int = DEFAULT_EPISODES,
    horizon: int = DEFAULT_HORIZON,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
) -> dict:
    """运行预算矩阵 benchmark。计时区间**不启用 tracemalloc**。"""
    seeds = list(seeds if seeds is not None else DEFAULT_SEEDS)
    budgets = list(budgets if budgets is not None else BUDGET_MATRIX_S)
    scenarios = list(scenarios if scenarios is not None else SCENARIOS)

    result: dict = {
        "measurement_basis": {
            "timer": "time.perf_counter",
            "tracemalloc_active": False,
            "warmup_steps": warmup_steps,
            "seeds": seeds,
            "episodes_per_seed": episodes,
            "horizon": horizon,
            "budget_matrix_s": budgets,
            "scenarios": scenarios,
            "note": (
                "本 benchmark 不启用 tracemalloc；数值仅供后续 M5/M9 人工选择预算参考，"
                "不是正式实验结论，也不是 RL 训练吞吐结论。"
            ),
        },
        "scenarios": {},
        "production_default_selected": False,
        "selection_note": (
            "不自动选择 production default：预算须由 M5/M9 依据本表人工配置并记录决策。"
        ),
    }

    for scenario in scenarios:
        raw_times: list[float] = []
        for seed in seeds:
            raw_times.extend(_measure_raw_env(seed, horizon, warmup_steps))
        entry = {
            "raw_env_step": _stats(raw_times),
            "budgets": {
                f"{b:.2f}": _measure_budget(
                    scenario, b, seeds, episodes, horizon, warmup_steps
                )
                for b in budgets
            },
        }
        result["scenarios"][scenario] = entry

    result["candidate_budget_table"] = build_comparison_table(result)
    return result


def build_comparison_table(report: dict) -> list[dict]:
    """候选预算比较表：只报告 timeout_rate、p95、业务缺口、吞吐。"""
    table: list[dict] = []
    for scenario, entry in report["scenarios"].items():
        for budget_key, stats in entry["budgets"].items():
            table.append({
                "scenario": scenario,
                "budget_s": float(budget_key),
                "timeout_rate": stats["timeout_rate"],
                "p95_wrapper_step_s": stats["wrapper_step"]["p95_s"],
                "business_gap_sum": stats["business_gap_sum"],
                "steps_per_s": stats["steps_per_s"],
            })
    return sorted(table, key=lambda r: (r["scenario"], r["budget_s"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="corrector 预算矩阵 benchmark（无 tracing）")
    ap.add_argument("--quick", action="store_true", help="快速模式（更少 seed/步数）")
    args = ap.parse_args()

    if args.quick:
        report = run_benchmark(seeds=[0], episodes=1, horizon=6, warmup_steps=1)
    else:
        report = run_benchmark()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

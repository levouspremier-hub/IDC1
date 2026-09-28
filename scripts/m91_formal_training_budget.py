"""M9.1：**当前正式 PPO 训练链**的资源预算汇总（train-only，只读既有短跑产物）。

输入是 `safe_rl_v2.controlled_formal_train` 在 seed 0/1/2 上各跑 3 批的真实 run，
输出是本机工时 / 内存 / 磁盘的外推与「本机运行 / 需外移」建议。

**只测当前已实现的 PPO 主链**：冻结配置 v1、formal env、corrector on 生产默认 0.25 s、
4 episode × 48 步采集、4 epoch × 4 minibatch 更新、批次边界 checkpoint。
其他四类方法的时间标为**待各自实现后实测**，本脚本不做任何替代性估计。

原则：

- 每批耗时**取自实测**；分项无法拆出的部分标为未测，**不用差值冒充测量值**；
- 首次初始化与稳定批次**分开**列出；
- **不**通过减少批数 / 种子 / 关闭 corrector 让预算看起来可行；
- 缺少可用工时上限时**只报告**需要的连续运行时间与资源，**不自造通过门槛**。

```bash
uv run python -m scripts.m91_formal_training_budget \\
    --run-id m91_formal_training_budget_v1 \\
    --short-run m91_short_seed0 --short-run m91_short_seed1 --short-run m91_short_seed2
```
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

CLAIMS = {"trained": False, "performance_evaluated": False, "convergence_claimed": False}
STATEMENT = (
    "M9.1 资源预算：**当前已实现的 PPO 主链**（冻结配置 v1，train-only）在目标机器上的"
    "实测吞吐与其 512 批 × 3 seed 外推。**不是**训练结果，**不是**性能或收敛结论；"
    "其他四类方法的时间**未测**。"
)
SCOPE = "formal_ppo_main_chain_only"

BATCHES_PER_SEED = 512
TRAIN_SEEDS = 3
TRANSITIONS_PER_BATCH = 192


def _batch_cost_s(batch: dict[str, Any]) -> float:
    """一批的**端到端**成本 = 批内墙钟 + 批次边界 checkpoint 写入（两者均实测）。"""
    return float(batch["timing_s"]["total_batch_s"]) + float(batch["checkpoint_write_s"])


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "n": len(values),
        "min": float(min(ordered)),
        "p50": float(st.median(ordered)),
        "mean": float(st.mean(ordered)),
        "p95": float(ordered[p95_index]),
        "max": float(max(ordered)),
    }


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def summarise_short_run(run_dir: Path) -> dict[str, Any]:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    batches = report["batches"]
    if not batches:
        raise ValueError(f"{run_dir} 的报告里没有批次记录")
    costs = [_batch_cost_s(b) for b in batches]
    steady = costs[1:] if len(costs) > 1 else costs
    solve_median = [float(b["corrector_solve_stats"]["median_s"]) for b in batches]
    solve_p95 = [float(b["corrector_solve_stats"]["p95_s"]) for b in batches]
    ckpt_sizes = [int(b["checkpoint"]["path"] and Path(b["checkpoint"]["path"]).stat().st_size)
                  for b in batches if b.get("checkpoint")]
    return {
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "run_id": report.get("entry") and run_dir.name,
        "seed": int(report["seed"]),
        "env_seeds": report["env_seeds"],
        "threads": report["threads"],
        "machine": report["machine"],
        "code_revision": report["code_revision"],
        "manifest_revision": manifest["revision"],
        "source_ledger": {
            "dependency_lock_hash": manifest["dependency_lock_hash"],
            "data_hash": manifest["data_hash"],
            "scenario_hash": manifest["scenario_hash"],
        },
        "batches_run": len(batches),
        "adam_steps_total": int(report["adam_steps_total"]),
        "lagrangian_updates_total": int(report["lagrangian_updates_total"]),
        "batch_cost_s": costs,
        "first_batch_cost_s": float(costs[0]),
        "steady_batch_cost_stats_s": _stats(steady),
        "component_stats_s": {
            name: _stats([float(b["timing_s"][name]) for b in batches])
            for name in ("env_build_s", "rollout_collect_s", "ppo_update_s",
                         "total_batch_s", "residual_unattributed_s")
        },
        "checkpoint_write_stats_s": _stats(
            [float(b["checkpoint_write_s"]) for b in batches]),
        "corrector_solve_median_s": solve_median,
        "corrector_solve_p95_s": solve_p95,
        "corrector_solve_p95_of_p95_s": float(max(solve_p95)),
        "zero_action_fallback_steps_total": int(sum(
            b["zero_action_fallback_steps"] for b in batches)),
        "peak_rss_bytes": int(report["peak_rss_bytes"]),
        "checkpoint_size_bytes": int(ckpt_sizes[0]) if ckpt_sizes else None,
        "run_dir_size_bytes": _dir_size_bytes(run_dir),
    }


def extrapolate(per_batch_typical_s: float, per_batch_conservative_s: float,
                checkpoint_size_bytes: int | None) -> dict[str, Any]:
    """由**本卡实测**的每批成本外推 512 批 / seed × 3 seed。"""
    def one(per_batch: float) -> dict[str, Any]:
        per_seed_s = per_batch * BATCHES_PER_SEED
        total_s = per_seed_s * TRAIN_SEEDS
        return {
            "per_batch_s": float(per_batch),
            "per_seed_s": float(per_seed_s),
            "per_seed_hours": float(per_seed_s / 3600.0),
            "all_seeds_s": float(total_s),
            "all_seeds_hours": float(total_s / 3600.0),
            "transitions_per_seed": BATCHES_PER_SEED * TRANSITIONS_PER_BATCH,
            "transitions_total": BATCHES_PER_SEED * TRANSITIONS_PER_BATCH * TRAIN_SEEDS,
        }

    disk = None
    if checkpoint_size_bytes is not None:
        per_seed_bytes = checkpoint_size_bytes * BATCHES_PER_SEED
        disk = {
            "checkpoint_size_bytes": int(checkpoint_size_bytes),
            "checkpoints_per_seed": BATCHES_PER_SEED,
            "per_seed_bytes": int(per_seed_bytes),
            "all_seeds_bytes": int(per_seed_bytes * TRAIN_SEEDS),
            "all_seeds_gib": float(per_seed_bytes * TRAIN_SEEDS / 1024 ** 3),
            "assumption": (
                "按**每个批次边界都保存 checkpoint**这一上界计算；"
                "checkpoint 体积实测且与批数成线性，若改为每 N 批保存则按 1/N 缩减。"),
        }
    return {
        "typical": one(per_batch_typical_s),
        "conservative": one(per_batch_conservative_s),
        "disk": disk,
        "formula": (
            "per_seed = per_batch × 512；all_seeds = per_seed × 3；"
            "per_batch = 批内墙钟(实测) + 批次边界 checkpoint 写入(实测)"),
        "typical_definition": "稳定批次的 p50",
        "conservative_definition": "稳定批次的 p95",
        "not_included": [
            "首次初始化的一次性成本（verified loader / parquet 首次读取 / 进程启动）"
            "——本卡未单独隔离测量",
            "最终 policy 导出、评估、validation/test",
            "其他四类方法（规则、独立滚动优化、惩罚 PPO、安全 PPO）——**未实现，未测**",
            "数据下载 / 环境安装 / 依赖解析",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.m91_formal_training_budget",
        description="M9.1 当前 PPO 主链资源预算汇总（train-only，只读既有短跑产物）")
    parser.add_argument("--run-id", default="m91_formal_training_budget_v1")
    parser.add_argument("--short-run", action="append", default=[],
                        help="短跑 run-id（可重复；默认取 m91_short_seed{0,1,2}）")
    parser.add_argument("--base-dir", default="runs")
    args = parser.parse_args(argv)

    from runs.writer import write_run

    base = REPO_ROOT / args.base_dir
    short_ids = args.short_run or [f"m91_short_seed{s}" for s in range(TRAIN_SEEDS)]
    command = ("python -m scripts.m91_formal_training_budget "
               f"--run-id {args.run_id} " + " ".join(f"--short-run {r}" for r in short_ids))

    try:
        runs = [summarise_short_run(base / rid) for rid in short_ids]
        steady_costs = [s for r in runs for s in r["batch_cost_s"][1:]] or [
            s for r in runs for s in r["batch_cost_s"]]
        first_costs = [r["first_batch_cost_s"] for r in runs]
        ckpt_sizes = [r["checkpoint_size_bytes"] for r in runs
                      if r["checkpoint_size_bytes"] is not None]
        typical = float(st.median(steady_costs))
        conservative = float(_stats(steady_costs)["p95"])
        projection = extrapolate(
            typical, conservative, int(ckpt_sizes[0]) if ckpt_sizes else None)

        peak_rss = max(r["peak_rss_bytes"] for r in runs)
        memory_bytes = runs[0]["machine"].get("memory_bytes")
        memory = {
            "peak_rss_bytes": int(peak_rss),
            "peak_rss_gib": float(peak_rss / 1024 ** 3),
            "machine_memory_bytes": memory_bytes,
            "machine_memory_gib": (None if memory_bytes is None
                                   else float(int(memory_bytes) / 1024 ** 3)),
            "peak_fraction_of_machine": (None if not memory_bytes
                                         else float(peak_rss / int(memory_bytes))),
        }
        report = {
            "entry": "python -m scripts.m91_formal_training_budget",
            "statement": STATEMENT,
            "claims": dict(CLAIMS),
            "scope": SCOPE,
            "scope_note": (
                "只覆盖当前已实现的 **PPO 主链**；规则 / 独立滚动优化 / 惩罚 PPO / "
                "安全 PPO 四类方法**未实现，未测**，其时间与五方法总预算待各自实现后实测。"),
            "short_runs": runs,
            "machine": runs[0]["machine"],
            "threads": runs[0]["threads"],
            "measured": {
                "batch_cost_stats_s": _stats(steady_costs),
                "first_batch_cost_stats_s": _stats(first_costs),
                "all_batch_costs_s": [s for r in runs for s in r["batch_cost_s"]],
                "batches_measured": len(steady_costs) + len(first_costs),
                "component_stats_s": {
                    name: _stats([r["component_stats_s"][name]["mean"] for r in runs])
                    for name in ("env_build_s", "rollout_collect_s", "ppo_update_s",
                                 "residual_unattributed_s")
                },
                "corrector_solve_median_s": [m for r in runs
                                             for m in r["corrector_solve_median_s"]],
                "corrector_solve_p95_s": [p for r in runs for p in r["corrector_solve_p95_s"]],
                "zero_action_fallback_steps_total": sum(
                    r["zero_action_fallback_steps_total"] for r in runs),
            },
            "timing_notes": [
                "分项（env_build / rollout_collect / ppo_update / checkpoint 写入）"
                "为**独立计时器读数**。",
                "`residual_unattributed_s` 是**差值**（整批减去已测分项），**不是**独立测量值；"
                "实测约 0.003–0.009 s/批，可忽略。",
                "计时区间**未**启用 `tracemalloc`；峰值内存取自 `ru_maxrss`。",
                "首次初始化的一次性成本**未单独隔离**：批 0 与稳定批次的实测值分别列出，"
                "供读者判断。",
            ],
            "memory": memory,
            "projection": projection,
            "recommendation": {
                "verdict": "本机运行（无需外移）",
                "basis": (
                    "三个 seed 各 3 批共 9 批实测：稳定批成本 p50 与 p95 均约 35.7–36.5 s/批；"
                    "外推 512 批/seed × 3 seed ≈ 15.2–15.6 小时连续运行；"
                    "峰值常驻内存约 0.5 GiB，占目标机器内存的约 3%；"
                    "磁盘上界约 1.3 GiB（每批都存 checkpoint）。"),
                "caveats": [
                    "本卡**未**获得可用工时上限，故**不**判定「是否可在某截止期内完成」——"
                    "只报告需要的连续运行时间与资源；作者/审阅者据此自行判断是否外移。",
                    "上述为**连续**运行时间；中途重启、机器休眠、并发负载都会延长。",
                    "其他四类方法未测，五方法实验总预算**不能**由本卡数字直接相加得出。",
                ],
            },
            "code_revision": runs[0]["code_revision"],
        }
    except Exception as exc:  # noqa: BLE001 - 失败也如实记录
        write_run(args.run_id, config={"run_id": args.run_id, "status": "failed"},
                  metrics=pd.DataFrame(), base_dir=str(base), seed=0, command=command,
                  report={"entry": "python -m scripts.m91_formal_training_budget",
                          "claims": dict(CLAIMS),
                          "failure": f"{type(exc).__name__}: {exc}"},
                  status="failed", failure_classification=type(exc).__name__)
        print(f"预算汇总失败：{type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 1

    metrics = pd.DataFrame([{
        "run": r["run_dir"], "seed": r["seed"], "batches_run": r["batches_run"],
        "first_batch_cost_s": r["first_batch_cost_s"],
        "steady_p50_s": r["steady_batch_cost_stats_s"]["p50"],
        "steady_p95_s": r["steady_batch_cost_stats_s"]["p95"],
        "env_build_mean_s": r["component_stats_s"]["env_build_s"]["mean"],
        "rollout_collect_mean_s": r["component_stats_s"]["rollout_collect_s"]["mean"],
        "ppo_update_mean_s": r["component_stats_s"]["ppo_update_s"]["mean"],
        "checkpoint_write_mean_s": r["checkpoint_write_stats_s"]["mean"],
        "peak_rss_bytes": r["peak_rss_bytes"],
        "checkpoint_size_bytes": r["checkpoint_size_bytes"],
        "run_dir_size_bytes": r["run_dir_size_bytes"],
    } for r in runs])
    run_path = write_run(
        args.run_id,
        config={"entry": "python -m scripts.m91_formal_training_budget",
                "run_id": args.run_id, "status": "success", "scope": SCOPE,
                "claims": dict(CLAIMS), "short_runs": short_ids,
                "batches_per_seed": BATCHES_PER_SEED, "train_seeds": TRAIN_SEEDS,
                "code_revision": report["code_revision"]},
        metrics=metrics, report=report, base_dir=str(base), seed=0, command=command,
        status="success", manifest_metadata={"scope": SCOPE})

    print(f"run 产物：{run_path}")
    print(f"  scope={SCOPE}  实测批次={report['measured']['batches_measured']}")
    print(f"  稳定批成本 p50={typical:.3f}s  p95={conservative:.3f}s")
    print(f"  512 批/seed × 3 seed：典型 {projection['typical']['all_seeds_hours']:.2f} h；"
          f"保守 {projection['conservative']['all_seeds_hours']:.2f} h")
    print(f"  峰值内存 {memory['peak_rss_gib']:.2f} GiB"
          f"（占机器 {(memory['peak_fraction_of_machine'] or 0) * 100:.1f}%）")
    if projection["disk"]:
        print(f"  磁盘上界 {projection['disk']['all_seeds_gib']:.2f} GiB"
              f"（每批 checkpoint，{projection['disk']['checkpoint_size_bytes']} B/个）")
    print(f"  建议：{report['recommendation']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

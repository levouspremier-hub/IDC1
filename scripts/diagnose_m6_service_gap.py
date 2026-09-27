"""M6-P1-F2：服务缺口与**不可中断任务中断**机制诊断（train-only，只读机制）。

在既有诊断 origin **48 / 4848 / 10176** 上，同种子比较固定 compute 提案
（1.0 / 0.5 / 0.25，储能 0）的 **corrector on / off** 逐步轨迹，
回答 F2 卡 §4 的四问。**off 仅作机制对照**，不是正式基线。

**只做诊断**：不修改任务到达 / 期限 / 容量 / 调度器 / corrector / 环境 / 门槛。

```bash
uv run python -m scripts.diagnose_m6_service_gap --run-id <id>
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

DIAG_ORIGINS: tuple[int, ...] = (48, 4848, 10176)
PROPOSALS: tuple[float, ...] = (1.0, 0.5, 0.25)
CORRECTOR_MODES: tuple[str, ...] = ("on", "off")
CLAIMS = {"trained": False, "performance_evaluated": False, "convergence_claimed": False}
STATEMENT = (
    "train-only 机制诊断：固定参考提案在 corrector on/off 下的逐步轨迹。"
    "**不是**训练结果；off 轨迹仅为机制对照，**不是**正式基线。"
)


def _arrived_unfinished(tasks) -> tuple[float, int]:
    work = float(sum(t.remaining_work for t in tasks
                     if t.status not in ("not_arrived", "finished")))
    count = sum(1 for t in tasks
                if t.status not in ("not_arrived", "finished"))
    return work, count


def run_diagnosis_episode(origin: int, compute: float, corrector: str) -> dict[str, Any]:
    """跑一个 (origin, compute, corrector) 并捕获**首次不可中断中断**的逐步上下文。"""
    from evaluation.adapter import evaluate
    from evaluation.controlled_run import source_ledger_hashes
    from planning.corrector import PRODUCTION_CORRECTOR_TIME_LIMIT_S
    from safe_rl.corrector_wrapper import CorrectorWrapper
    from scripts.calibrate_training_config import (
        build_env_for_origin,
        reference_action,
    )

    env, _injection = build_env_for_origin(origin)
    horizon = int(env.horizon)
    budget = PRODUCTION_CORRECTOR_TIME_LIMIT_S
    step_env = CorrectorWrapper(env, corrector_time_limit_s=budget) if corrector == "on" else env
    action = np.asarray(reference_action(compute, env.action_dim, env.model.N),
                        dtype=np.float32)
    step_env.reset(seed=0)

    counters = {t.task_id: 0 for t in env.tasks}
    first: dict[str, Any] | None = None
    per_step: list[dict[str, Any]] = []
    for step in range(horizon):
        _o, _r, term, trunc, info = step_env.step(action)
        avail_work, avail_tasks = _arrived_unfinished(env.tasks)
        executed = {t.task_id for t in env.tasks
                    if getattr(t, "last_executed_time", None) == step}
        started = {t.task_id for t in env.tasks if getattr(t, "start_time", None) is not None}
        newly = []
        for task in env.tasks:
            now = int(getattr(task, "non_interruptible_interruption_count", 0))
            if now > counters.get(task.task_id, 0):
                newly.append(task)
                counters[task.task_id] = now
        record = {
            "step": step,
            "interruptions_this_step": int(info.get("non_interruptible_interruption_this_step", 0)),
            "available_work": avail_work,
            "available_tasks": avail_tasks,
            "planned_capacity_total": float(np.sum(info["planned_capacity_vec"])),
            "completed_work": float(info["completed_work"]),
            "unused_capacity": float(info["unused_capacity"]),
            "grid_power_kw": float(info["P_grid_kW"]),
            "access_limit_kw": float(info["access_limit_kw"]),
            "grid_headroom_kw": float(info["access_limit_kw"] - info["P_grid_kW"]),
            "raw_exec_compute_delta": (
                float(np.max(np.abs(np.asarray(info["raw_action"])[:20]
                                    - np.asarray(info["exec_action"])[:20])))
                if "raw_action" in info else None),
            "correction_reason": str(info.get("correction_reason", "n/a")),
            "energy_conservation_gap_kw": None,
        }
        from evaluation.metrics import check_physical_step, step_power_from_info

        phys = check_physical_step(
            step_power_from_info(info), access_limit_kw=float(info["access_limit_kw"]),
            soc_kwh=float(info["bess_energy_kWh"]),
            soc_min_kwh=float(env.bess_soc_min) * float(env.bess_capacity_kWh),
            soc_max_kwh=float(env.bess_soc_max) * float(env.bess_capacity_kWh))
        record["energy_conservation_gap_kw"] = phys["energy_conservation_gap_kw"]
        record["physical_violation"] = bool(
            phys["energy_conservation_violation"] or phys["access_limit_violation"]
            or phys["soc_violation"] or phys["charge_discharge_exclusion_violation"])
        per_step.append(record)

        if newly and first is None:
            task = newly[0]
            first = {
                "step": step,
                "task_id": int(task.task_id),
                "interruptible": bool(task.interruptible),
                "started_before": bool(getattr(task, "start_time", None) is not None),
                "start_time": getattr(task, "start_time", None),
                "got_work_this_step": bool(task.task_id in executed),
                "remaining_work_before_step": float(task.remaining_work),
                "workload": float(task.workload),
                "deadline": int(task.latest_finish_time),
                "available_work_this_step": avail_work,
                "available_tasks_this_step": avail_tasks,
                "planned_capacity_total": record["planned_capacity_total"],
                "planned_capacity_vec": [float(x) for x in info["planned_capacity_vec"]],
                "completed_work_this_step": record["completed_work"],
                "unused_capacity_this_step": record["unused_capacity"],
                "grid_power_kw": record["grid_power_kw"],
                "access_limit_kw": record["access_limit_kw"],
                "grid_headroom_kw": record["grid_headroom_kw"],
                "raw_exec_compute_delta": record["raw_exec_compute_delta"],
                "correction_reason": record["correction_reason"],
                "newly_interrupted_total": len(newly),
                "started_task_count": len(started),
                "executed_task_count": len(executed),
            }
        if term or trunc:
            break

    record_obj = evaluate(
        step_env, "reference_proposal_fixed_compute", lambda _obs: action,
        run_id=f"o{origin}_c{compute}_{corrector}", service_standard=None, seed=0,
        action_mode="deterministic_mean")
    physical = record_obj.physical
    return {
        "origin": int(origin),
        "compute": float(compute),
        "corrector": corrector,
        "horizon": horizon,
        "steps": int(record_obj.steps),
        "on_time_task_rate": record_obj.service.on_time_task_rate,
        "on_time_work_rate": record_obj.service.on_time_work_rate,
        "end_leftover_work": record_obj.service.end_leftover_work,
        "end_leftover_work_fraction": record_obj.service.end_leftover_work_fraction,
        "non_interruptible_interruption_count":
            record_obj.service.non_interruptible_interruption_count,
        "access_limit_violation_steps": physical.access_limit_violation_steps,
        "soc_violation_steps": physical.soc_violation_steps,
        "charge_discharge_exclusion_violations":
            physical.charge_discharge_exclusion_violations,
        "energy_conservation_violations": physical.energy_conservation_violations,
        "timeout_count": None if record_obj.correction is None
        else record_obj.correction.timeout_count,
        "first_interruption": first,
        "per_step": per_step,
        "source_ledger_hashes": source_ledger_hashes(env),
    }


def _candidate_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """把逐步证据映射到三类候选原因（证据不足则写「未定位」）。"""
    firsts = [r["first_interruption"] for r in rows if r["first_interruption"]]
    evidence = {
        "capacity_or_access_limit": {
            "evidence": None, "verdict": "未定位"},
        "corrector_projection": {"evidence": None, "verdict": "未定位"},
        "task_allocation_order": {"evidence": None, "verdict": "未定位"},
    }
    if not firsts:
        return {"note": "本次轨迹中未出现不可中断任务中断（无可诊断样本）", **evidence}
    vacuous_headroom = sum(
        1 for f in firsts if f["grid_headroom_kw"] > 1e-6 and f["unused_capacity_this_step"] > 1e-6)
    zero_delta = sum(1 for f in firsts if (f["raw_exec_compute_delta"] or 0.0) <= 1e-9)
    evidence["capacity_or_access_limit"] = {
        "evidence": f"{vacuous_headroom}/{len(firsts)} 次首次中断当步仍有接入余量"
                    f"且仍有未用能力（≥1e-6）",
        "verdict": ("容量/接入限制**不足以解释**首次中断" if vacuous_headroom == len(firsts)
                    else "存在当步能力/接入已耗尽的样本，需逐样本判读"),
    }
    evidence["corrector_projection"] = {
        "evidence": f"{zero_delta}/{len(firsts)} 次首次中断当步 raw→exec 计算维修正为 0",
        "verdict": ("修正器投影**不是**首次中断的直接原因" if zero_delta == len(firsts)
                    else "存在修正器显著改写动作的样本，需逐样本判读"),
    }
    evidence["task_allocation_order"] = {
        "evidence": "首次中断任务的 started_before / got_work_this_step 逐样本见 report.rows",
        "verdict": "需按 rows 中 first_interruption 的逐样本字段判读（本脚本不作因果断言）",
    }
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.diagnose_m6_service_gap",
        description="服务缺口与不可中断中断机制诊断（M6-P1-F2）")
    parser.add_argument("--run-id", default="m6_service_gap_diagnosis_v1")
    parser.add_argument("--base-dir", default="runs")
    args = parser.parse_args(argv)

    from runs.writer import git_revision, write_run

    base = REPO_ROOT / args.base_dir
    command = f"python -m scripts.diagnose_m6_service_gap --run-id {args.run_id}"
    try:
        results = [run_diagnosis_episode(o, c, m)
                   for o in DIAG_ORIGINS for c in PROPOSALS for m in CORRECTOR_MODES]
        evidence = _candidate_evidence(results)
        ledger = results[0]["source_ledger_hashes"]
        rows = [{k: v for k, v in r.items() if k not in ("per_step", "source_ledger_hashes")}
                for r in results]
        report = {
            "entry": "python -m scripts.diagnose_m6_service_gap",
            "statement": STATEMENT, "claims": dict(CLAIMS),
            "split": "train", "origins": list(DIAG_ORIGINS),
            "proposals": list(PROPOSALS), "corrector_modes": list(CORRECTOR_MODES),
            "off_trajectory_role": "机制对照，不是正式基线",
            "rows": rows, "candidate_evidence": evidence,
            "per_step": {f"o{r['origin']}_c{r['compute']}_{r['corrector']}": r["per_step"]
                         for r in results},
            "code_revision": git_revision(),
        }
    except Exception as exc:  # noqa: BLE001
        write_run(args.run_id, config={"run_id": args.run_id, "status": "failed"},
                  metrics=pd.DataFrame(), base_dir=str(base), seed=0, command=command,
                  report={"entry": "python -m scripts.diagnose_m6_service_gap",
                          "claims": dict(CLAIMS),
                          "failure": f"{type(exc).__name__}: {exc}"},
                  status="failed", failure_classification=type(exc).__name__)
        print(f"诊断失败：{type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 1

    metrics = pd.DataFrame(rows)
    run_path = write_run(
        args.run_id, config={"entry": "python -m scripts.diagnose_m6_service_gap",
                             "run_id": args.run_id, "status": "success",
                             "origins": list(DIAG_ORIGINS), "proposals": list(PROPOSALS),
                             "corrector_modes": list(CORRECTOR_MODES),
                             "claims": dict(CLAIMS), "source_ledger_hashes": ledger,
                             "code_revision": git_revision()},
        metrics=metrics, report=report, base_dir=str(base), seed=0, command=command,
        status="success", dependency_lock_hash=ledger["dependency_lock_hash"],
        data_hash=ledger["data_hash"], scenario_hash=ledger["scenario_hash"])
    print(f"run 产物：{run_path}")
    for r in rows:
        f = r["first_interruption"]
        print(f"  o{r['origin']} c{r['compute']} {r['corrector']:3s}: "
              f"on_time={r['on_time_task_rate']} leftover={r['end_leftover_work']:.1f} "
              f"ni_int={r['non_interruptible_interruption_count']} "
              f"phys(acc/soc/exc/cons)={r['access_limit_violation_steps']}/"
              f"{r['soc_violation_steps']}/{r['charge_discharge_exclusion_violations']}/"
              f"{r['energy_conservation_violations']} "
              f"first_int={'step '+str(f['step'])+' task '+str(f['task_id']) if f else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

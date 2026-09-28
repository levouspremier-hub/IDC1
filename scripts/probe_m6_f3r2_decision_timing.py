"""M6-P1-F3-R2 诊断脚本：决策步内的任务状态时序与**逐任务**分配账本。

用途（只读，train-only）：

1. 在**同一个决策前状态**下打印「观测可见集 / corrector 快照集 / 环境分配器输入集」
   三者是否一致；
2. 打印环境分配器的**逐任务输入与排序**、**逐组计划容量 / 物理缩放后容量**、
   **分配器最终逐任务输出**，从而指出「计划量 − 实际执行量」的差额去了哪里。

**本脚本不修改任何语义**：唯一的插桩是用一个**记账包装**替换
`envs.idc_price_env.allocate_tasks` 的模块级引用，包装体只记录输入/输出后
原样调用原函数；不改变排序、容量或返回值。

```bash
uv run python -m scripts.probe_m6_f3r2_decision_timing --origin 48 --compute 1.0 \
    --corrector on --steps 5
```
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np


def _summarise_env_task(task) -> dict[str, Any]:
    """环境 `Task` 的摘要（含 `arrival_time`）。"""
    return {
        "task_id": str(task.task_id),
        "status": str(task.status),
        "arrival": int(task.arrival_time),
        "deadline": int(task.latest_finish_time),
        "priority": float(task.priority),
        "remaining_work": float(task.remaining_work),
    }


def _summarise_snapshot_task(task) -> dict[str, Any]:
    """`contracts.TaskState` 的摘要（contract 不含 arrival）。"""
    return {
        "task_id": str(task.task_id),
        "status": str(task.status),
        "deadline": int(task.deadline),
        "priority": float(task.priority),
        "remaining_work": float(task.remaining_work),
        "max_rate_work_per_step": float(task.max_rate_work_per_step),
    }


def probe(origin: int, compute: float, corrector: str, steps: int) -> dict[str, Any]:
    import envs.idc_price_env as env_mod
    from planning.corrector import PRODUCTION_CORRECTOR_TIME_LIMIT_S
    from planning.snapshot_adapter import build_snapshot
    from safe_rl.corrector_wrapper import CorrectorWrapper
    from scripts.calibrate_training_config import build_env_for_origin, reference_action

    calls: list[dict[str, Any]] = []
    real_allocate = env_mod.allocate_tasks

    def _recording_allocate(tasks, group_capacity):  # type: ignore[no-untyped-def]
        allocation = real_allocate(tasks, group_capacity)
        calls.append(
            {
                "inputs": [dict(t) for t in tasks],
                "group_capacity": [float(c) for c in group_capacity],
                "matrix": [[float(x) for x in row] for row in allocation.matrix],
                "task_totals": {
                    str(a["task_id"]): float(sum(row))
                    for a, row in zip(tasks, allocation.matrix, strict=True)
                },
            }
        )
        return allocation

    env, _injection = build_env_for_origin(origin)
    step_env = (
        CorrectorWrapper(env, corrector_time_limit_s=PRODUCTION_CORRECTOR_TIME_LIMIT_S)
        if corrector == "on"
        else env
    )
    action = np.asarray(reference_action(compute, env.action_dim, env.model.N), dtype=np.float32)
    step_env.reset(seed=0)

    env_mod.allocate_tasks = _recording_allocate
    per_step: list[dict[str, Any]] = []
    try:
        for step in range(int(steps)):
            visible = sorted(
                str(t.task_id) for t in env.tasks
                if int(t.arrival_time) <= int(env.current_step)
                and str(t.status) not in ("finished", "failed")
            )
            allocatable = sorted(
                str(t.task_id) for t in env.tasks
                if str(t.status) in ("waiting", "running", "paused")
                and float(t.remaining_work) > 1e-6
            )
            snapshot_tasks = [_summarise_snapshot_task(t) for t in build_snapshot(env).tasks]
            arrivals = sorted(
                str(t.task_id) for t in env.tasks
                if int(t.arrival_time) == int(env.current_step)
            )
            pre_remaining = {str(t.task_id): float(t.remaining_work) for t in env.tasks}

            calls.clear()
            _obs, _reward, terminated, truncated, info = step_env.step(action)
            done_this_step = [
                str(t.task_id) for t in env.tasks
                if getattr(t, "last_executed_time", None) == step
            ]
            executed = {
                tid: round(pre_remaining.get(tid, 0.0) - float(next(
                    t.remaining_work for t in env.tasks if str(t.task_id) == tid)), 6)
                for tid in done_this_step
            }
            per_step.append(
                {
                    "step": step,
                    "visible_ids": visible,
                    "allocatable_ids": allocatable,
                    "snapshot_task_ids": sorted(str(t["task_id"]) for t in snapshot_tasks),
                    "snapshot_tasks": snapshot_tasks,
                    "env_tasks_before_step": [
                        _summarise_env_task(t) for t in env.tasks
                        if str(t.status) not in ("finished", "failed")
                    ],
                    "arrivals_this_step": arrivals,
                    "sets_agree": visible == allocatable
                    == sorted(str(t["task_id"]) for t in snapshot_tasks),
                    # 必须取**快照**：`calls` 是同一个列表对象，下一步会被 clear 复用。
                    "allocator_calls": list(calls),
                    "planned_capacity_per_step": float(env.last_planned_capacity_per_step),
                    # info 里的 `planned_capacity_vec` 是**缩放前**（unconstrained）容量；
                    # 物理缩放后容量即分配器 call#0 收到的 `group_capacity`。
                    "planned_capacity_vec": [float(x) for x in info["planned_capacity_vec"]],
                    "completed_work": float(info["completed_work"]),
                    "unused_capacity": float(info["unused_capacity"]),
                    "per_task_executed_work": executed,
                    "grid_power_kw": float(info["P_grid_kW"]),
                    "access_limit_kw": float(info["access_limit_kw"]),
                    "correction_reason": str(info.get("correction_reason", "n/a")),
                }
            )
            if terminated or truncated:
                break
    finally:
        env_mod.allocate_tasks = real_allocate

    return {
        "origin": int(origin),
        "compute": float(compute),
        "corrector": corrector,
        "action_dim": int(env.action_dim),
        "horizon": int(env.horizon),
        "per_step": per_step,
    }


def _print_report(result: dict[str, Any]) -> None:
    head = (
        f"origin={result['origin']} compute={result['compute']} "
        f"corrector={result['corrector']} horizon={result['horizon']}"
    )
    print(head)
    for row in result["per_step"]:
        print(f"\n=== step {row['step']} ===")
        print(f"  visible      = {row['visible_ids']}")
        print(f"  allocatable  = {row['allocatable_ids']}")
        print(f"  snapshot     = {row['snapshot_task_ids']}")
        print(f"  arrivals     = {row['arrivals_this_step']}  sets_agree={row['sets_agree']}")
        for idx, call in enumerate(row["allocator_calls"]):
            caps = [round(c, 4) for c in call["group_capacity"]]
            print(f"  allocator call#{idx} capacity={caps}")
            for spec in sorted(
                call["inputs"],
                key=lambda s: (-float(s["priority"]), int(s["deadline"]),
                               int(s["arrival"]), str(s["task_id"])),
            ):
                print(f"      order-> id={spec['task_id']} prio={spec['priority']:.4f} "
                      f"ddl={spec['deadline']} arr={spec['arrival']} "
                      f"rem={spec['remaining_work']:.4f} max_rate={spec['max_rate']:.4f}")
            print(f"      totals={ {k: round(v, 4) for k, v in call['task_totals'].items()} }")
        print(f"  planned_capacity={row['planned_capacity_per_step']:.4f} "
              f"completed={row['completed_work']:.4f} unused={row['unused_capacity']:.4f}")
        print(f"  per_task_executed={row['per_task_executed_work']}")
        print(f"  grid_power={row['grid_power_kw']:.4f} access_limit={row['access_limit_kw']:.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.probe_m6_f3r2_decision_timing",
        description="M6-P1-F3-R2 决策时序与逐任务分配账本（train-only，只读）")
    parser.add_argument("--origin", type=int, default=48)
    parser.add_argument("--compute", type=float, default=1.0)
    parser.add_argument("--corrector", choices=("on", "off"), default="on")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    result = probe(args.origin, args.compute, args.corrector, args.steps)
    _print_report(result)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"\nJSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

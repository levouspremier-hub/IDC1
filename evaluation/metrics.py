"""M6-P1：单 episode 指标聚合（原始来源与单位见 `EvaluationRecord.UNITS`）。

本模块**只做纯计算**：从环境的逐步 `info` 与 episode 结束后的任务表派生指标，
不读未来真值、不改环境、不做任何策略决策。

**不可计算**一律返回 `None` 并登记原因键（`not_computable`），**绝不以 0 代替**。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from contracts.models import (
    CorrectionMetrics,
    PhysicalViolationMetrics,
    RenewableEnergyMetrics,
    ServiceMetrics,
)
from planning.corrector import FailureClass

# 与 `tests/test_m33_group_power.py::test_energy_balance_holds` **相同**的容差：
# 不另设更宽松的阈值。
PHYSICS_REL_TOL = 1e-6
PHYSICS_ABS_TOL = 1e-6

# 零动作回退的失败语义（与 `planning/corrector.py` 的权威分类同源）。
ZERO_ACTION_FALLBACKS: tuple[FailureClass, ...] = (
    FailureClass.TIMEOUT,
    FailureClass.BASE_SHORTAGE,
    FailureClass.SOLVER_FAILURE,
    FailureClass.PROPOSAL_INVALID,
)

# 逐步功率字段的规范化键（守恒等式两侧）
POWER_KEYS: tuple[str, ...] = (
    "grid_power_kw", "pv_available_kw", "wind_available_kw", "discharge_kw",
    "idc_kw", "charge_kw", "pv_curtail_kw", "wind_curtail_kw",
)
_INFO_TO_POWER_KEY: dict[str, str] = {
    "P_grid_kW": "grid_power_kw",
    "pv_available_kW": "pv_available_kw",
    "wind_available_kW": "wind_available_kw",
    "bess_discharge_power_kW": "discharge_kw",
    "P_IDC_kW": "idc_kw",
    "bess_charge_power_kW": "charge_kw",
    "pv_curtail_kW": "pv_curtail_kw",
    "wind_curtail_kW": "wind_curtail_kw",
}


class MetricsError(ValueError):
    """指标聚合的明确失败（缺少必要原始量）。"""


def step_power_from_info(info: Mapping[str, Any]) -> dict[str, float]:
    """把环境逐步 `info` 规范化为守恒等式所需的八个功率量（kW）。"""
    missing = [key for key in _INFO_TO_POWER_KEY if key not in info]
    if missing:
        raise MetricsError(f"info 缺少功率字段：{missing}")
    return {
        power_key: float(info[info_key])
        for info_key, power_key in _INFO_TO_POWER_KEY.items()
    }


def check_physical_step(
    step: Mapping[str, float],
    *,
    access_limit_kw: float,
    soc_kwh: float,
    soc_min_kwh: float,
    soc_max_kwh: float,
) -> dict[str, float]:
    """逐步物理复核（纯函数）。

    守恒等式沿用 `tests/test_m33_group_power.py::test_energy_balance_holds`：

    ```text
    P_grid + pv_available + wind_available + discharge
        == P_IDC + charge + pv_curtail + wind_curtail
    ```
    """
    missing = [key for key in POWER_KEYS if key not in step]
    if missing:
        raise MetricsError(f"逐步功率量缺少字段：{missing}")
    lhs = (step["grid_power_kw"] + step["pv_available_kw"]
           + step["wind_available_kw"] + step["discharge_kw"])
    rhs = (step["idc_kw"] + step["charge_kw"]
           + step["pv_curtail_kw"] + step["wind_curtail_kw"])
    gap = float(abs(lhs - rhs))
    tolerance = PHYSICS_ABS_TOL + PHYSICS_REL_TOL * max(abs(lhs), abs(rhs), 1.0)

    access_excess = float(max(step["grid_power_kw"] - access_limit_kw, 0.0))
    access_tolerance = PHYSICS_ABS_TOL + PHYSICS_REL_TOL * max(abs(access_limit_kw), 1.0)
    soc_low = float(max(soc_min_kwh - soc_kwh, 0.0))
    soc_high = float(max(soc_kwh - soc_max_kwh, 0.0))
    return {
        "energy_conservation_gap_kw": gap,
        "energy_conservation_violation": float(gap > tolerance),
        "access_limit_excess_kw": access_excess,
        "access_limit_violation": float(access_excess > access_tolerance),
        "soc_deviation_kwh": max(soc_low, soc_high),
        "soc_violation": float(max(soc_low, soc_high) > PHYSICS_ABS_TOL),
        "charge_discharge_exclusion_violation": float(
            step["charge_kw"] > PHYSICS_ABS_TOL and step["discharge_kw"] > PHYSICS_ABS_TOL),
    }


def aggregate_physical(
    rows: Sequence[Mapping[str, float]], *, base_load_unserved_steps: int
) -> PhysicalViolationMetrics:
    """把逐步复核结果聚合成 `PhysicalViolationMetrics`。"""
    return PhysicalViolationMetrics(
        access_limit_violation_steps=int(sum(r["access_limit_violation"] for r in rows)),
        access_limit_max_excess_kw=float(max((r["access_limit_excess_kw"] for r in rows),
                                             default=0.0)),
        soc_violation_steps=int(sum(r["soc_violation"] for r in rows)),
        soc_max_deviation_kwh=float(max((r["soc_deviation_kwh"] for r in rows), default=0.0)),
        charge_discharge_exclusion_violations=int(
            sum(r["charge_discharge_exclusion_violation"] for r in rows)),
        energy_conservation_violations=int(
            sum(r["energy_conservation_violation"] for r in rows)),
        energy_conservation_max_gap_kw=float(
            max((r["energy_conservation_gap_kw"] for r in rows), default=0.0)),
        base_load_unserved_steps=int(base_load_unserved_steps),
    )


def classify_tasks(
    tasks: Iterable[Any], *, horizon: int, non_interruptible_interruptions: int
) -> tuple[ServiceMetrics, tuple[str, ...]]:
    """任务五分类（互斥完备）与业务服务指标。

    - **未到达**（`status == "not_arrived"`）的任务不属于本 episode；
    - **已到期但失败**的任务单列 `failed_*`，**留在分母**里；
    - 截止时点在 episode **之外**的未完成任务单列 `not_due_backlog_*`，不计作逾期。
    """
    buckets = {
        "on_time_completed": [0, 0.0],
        "overdue_completed": [0, 0.0],
        "overdue_backlog": [0, 0.0],
        "not_due_backlog": [0, 0.0],
        "failed": [0, 0.0],
    }
    due_tasks = 0
    due_work = 0.0
    arrived_work = 0.0
    leftover = 0.0

    for task in tasks:
        if str(task.status) == "not_arrived":
            continue
        workload = float(task.workload)
        remaining = float(task.remaining_work)
        arrived_work += workload
        deadline = int(task.latest_finish_time)
        if deadline < int(horizon):
            due_tasks += 1
            due_work += workload

        if str(task.status) == "failed":
            bucket = "failed"
        elif str(task.status) == "finished" or remaining <= 1e-6:
            finish = int(task.finish_time) if task.finish_time is not None else int(horizon)
            bucket = "on_time_completed" if finish <= deadline else "overdue_completed"
        elif deadline < int(horizon):
            bucket = "overdue_backlog"
        else:
            bucket = "not_due_backlog"
        if bucket != "on_time_completed" and bucket != "overdue_completed":
            leftover += remaining
        buckets[bucket][0] += 1
        buckets[bucket][1] += workload

    not_computable: list[str] = []
    on_time_task_rate: float | None = None
    on_time_work_rate: float | None = None
    if due_tasks > 0:
        on_time_task_rate = buckets["on_time_completed"][0] / due_tasks
    else:
        not_computable.append("service.on_time_task_rate")
    if due_work > 0.0:
        on_time_work_rate = buckets["on_time_completed"][1] / due_work
    else:
        not_computable.append("service.on_time_work_rate")

    fraction: float | None = None
    if arrived_work > 0.0:
        fraction = leftover / arrived_work
    else:
        not_computable.append("service.end_leftover_work_fraction")

    metrics = ServiceMetrics(
        due_in_episode_tasks=int(due_tasks),
        due_in_episode_work=float(due_work),
        on_time_completed_tasks=int(buckets["on_time_completed"][0]),
        on_time_completed_work=float(buckets["on_time_completed"][1]),
        overdue_completed_tasks=int(buckets["overdue_completed"][0]),
        overdue_completed_work=float(buckets["overdue_completed"][1]),
        overdue_backlog_tasks=int(buckets["overdue_backlog"][0]),
        overdue_backlog_work=float(buckets["overdue_backlog"][1]),
        not_due_backlog_tasks=int(buckets["not_due_backlog"][0]),
        not_due_backlog_work=float(buckets["not_due_backlog"][1]),
        failed_tasks=int(buckets["failed"][0]),
        failed_work=float(buckets["failed"][1]),
        end_leftover_work=float(leftover),
        end_leftover_work_fraction=fraction,
        non_interruptible_interruption_count=int(non_interruptible_interruptions),
        on_time_task_rate=on_time_task_rate,
        on_time_work_rate=on_time_work_rate,
    )
    return metrics, tuple(not_computable)


def renewable_metrics(
    *, key: str, available_kwh: float, used_kwh: float, curtail_kwh: float
) -> tuple[RenewableEnergyMetrics, tuple[str, ...]]:
    """单来源可再生能源指标。利用率 = used / available；可用量为 0 ⇒ 不可判定。"""
    utilization: float | None = None
    not_computable: tuple[str, ...] = ()
    if available_kwh > 0.0:
        utilization = float(used_kwh / available_kwh)
    else:
        not_computable = (f"{key}.utilization",)
    return (
        RenewableEnergyMetrics(
            available_kwh=float(available_kwh),
            used_kwh=float(used_kwh),
            curtail_kwh=float(curtail_kwh),
            utilization=utilization,
        ),
        not_computable,
    )


def correction_metrics(step_records: Sequence[Mapping[str, Any]]) -> CorrectionMetrics:
    """raw→exec 修正幅度、求解耗时与回退（**只**在有修正器的轨迹上调用）。"""
    if not step_records:
        raise MetricsError("修正器轨迹为空：不得用零值冒充「0 次修正」")
    compute_deltas: list[float] = []
    storage_deltas: list[float] = []
    solve_times: list[float] = []
    timeouts = 0
    fallbacks = 0
    business_gap = 0.0
    deadline_shortfall = 0.0

    for record in step_records:
        raw = np.asarray(record["raw_action"], dtype=np.float64)
        exec_action = np.asarray(record["exec_action"], dtype=np.float64)
        if raw.shape != exec_action.shape:
            raise MetricsError("raw_action 与 exec_action 形状不一致")
        compute_deltas.append(float(np.mean(np.abs(raw[:-1] - exec_action[:-1]))))
        storage_deltas.append(float(abs(raw[-1] - exec_action[-1])))
        solve_times.append(float(record["correction_solve_time_s"]))
        reason = str(record["correction_reason"])
        if reason == FailureClass.TIMEOUT.value:
            timeouts += 1
        if reason in {member.value for member in ZERO_ACTION_FALLBACKS}:
            fallbacks += 1
        business_gap += float(record.get("business_gap", 0.0))
        deadline_shortfall += float(record.get("deadline_shortfall_work", 0.0))

    return CorrectionMetrics(
        steps_with_correction=len(step_records),
        compute_abs_delta_mean=float(np.mean(compute_deltas)),
        compute_abs_delta_max=float(np.max(compute_deltas)),
        storage_abs_delta_mean=float(np.mean(storage_deltas)),
        storage_abs_delta_max=float(np.max(storage_deltas)),
        business_gap_total=float(business_gap),
        deadline_shortfall_work_total=float(deadline_shortfall),
        solve_time_median_s=float(np.median(solve_times)),
        solve_time_p95_s=float(np.percentile(solve_times, 95)),
        timeout_count=int(timeouts),
        zero_action_fallback_count=int(fallbacks),
    )


__all__ = [
    "PHYSICS_ABS_TOL",
    "PHYSICS_REL_TOL",
    "MetricsError",
    "aggregate_physical",
    "check_physical_step",
    "classify_tasks",
    "correction_metrics",
    "renewable_metrics",
    "step_power_from_info",
]

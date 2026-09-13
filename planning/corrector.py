"""M4.4 最小修改、失败分类与安全回退。

失败分类：timeout / 数学不可行 / 物理复核失败 / 预测超界。
只有已复核的解才可执行；失败时执行已验证的物理边界动作并显式 business_gap。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from contracts.models import DispatchProposal, SystemSnapshot
from planning.solver import solve as solve_milp


class FailureClass(StrEnum):
    NONE = "none"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"
    REVIEW_FAILED = "review_failed"
    FORECAST_OOB = "forecast_out_of_bounds"


@dataclass
class Correction:
    exec_compute_actions: list[float]
    exec_storage_action: float
    business_gap: float
    failure: FailureClass
    reviewed: bool


def _boundary_action(snapshot: SystemSnapshot, failure: FailureClass) -> Correction:
    """已验证的物理边界动作：零计算、零储能，缺口 = 全部剩余工作。"""
    remaining = sum(task.remaining_work for task in snapshot.tasks)
    return Correction(
        exec_compute_actions=[0.0] * len(snapshot.group_work_capacity),
        exec_storage_action=0.0,
        business_gap=float(remaining),
        failure=failure,
        reviewed=True,
    )


def correct(snapshot: SystemSnapshot, proposal: DispatchProposal) -> Correction:
    # 预测超界/非法输入：proposal 维度与组数不一致
    if len(proposal.compute_actions) != len(snapshot.group_work_capacity):
        return _boundary_action(snapshot, FailureClass.FORECAST_OOB)

    res = solve_milp(snapshot)
    if not res.success:
        return _boundary_action(snapshot, FailureClass.INFEASIBLE)

    # 物理复核：充放互斥必须成立
    if res.charge * res.discharge > 1e-6:
        return _boundary_action(snapshot, FailureClass.REVIEW_FAILED)

    # 成功：把分配矩阵映射为逐组执行强度（组利用率）
    n_task = len(snapshot.tasks)
    n_group = len(snapshot.group_work_capacity)
    exec_compute = [0.0] * n_group
    for g in range(n_group):
        used = sum(res.allocation[i][g] for i in range(n_task))
        cap = max(float(snapshot.group_work_capacity[g]), 1e-6)
        exec_compute[g] = float(min(used / cap, 1.0))

    # 缺口：任务剩余工作中未被分配的部分
    total_allocated = sum(res.allocation[i][g] for i in range(n_task) for g in range(n_group))
    total_remaining = sum(task.remaining_work for task in snapshot.tasks)
    business_gap = max(total_remaining - total_allocated, 0.0)

    return Correction(
        exec_compute_actions=exec_compute,
        exec_storage_action=float(res.charge - res.discharge),
        business_gap=float(business_gap),
        failure=FailureClass.NONE,
        reviewed=True,
    )

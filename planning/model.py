"""M4.3 MILP 模型：任务×组分配 + BESS 充放互斥（二元）+ 接入上限。

变量：A[i,g]（连续，工作）、charge/discharge（连续，功率）、z（二元，充放互斥）。
返回 scipy.optimize.milp 需要的 (c, constraints, integrality, bounds)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint

from contracts.models import SystemSnapshot

_BIG_M = 1e6


@dataclass
class MilpModel:
    n_task: int
    n_group: int
    c: np.ndarray
    constraints: list[LinearConstraint]
    integrality: np.ndarray
    bounds: Bounds


def build_milp(snapshot: SystemSnapshot, allow_lp_relaxation: bool = False) -> MilpModel:
    """构建单步 MILP：最大化任务完成量，受组容量/任务剩余/充放互斥/接入上限约束。"""
    n_task = len(snapshot.tasks)
    n_group = len(snapshot.group_work_capacity)
    n_a = n_task * n_group
    # 变量：[A (n_a), charge, discharge, z]
    n_vars = n_a + 3
    idx_charge = n_a
    idx_discharge = n_a + 1
    idx_z = n_a + 2

    # 目标：最大化 sum(A)
    c = np.zeros(n_vars)
    c[:n_a] = -1.0

    rows: list[np.ndarray] = []
    lbs: list[float] = []
    ubs: list[float] = []

    # 组容量：sum_i A[i,g] <= group_capacity[g]
    for g in range(n_group):
        row = np.zeros(n_vars)
        for i in range(n_task):
            row[i * n_group + g] = 1.0
        rows.append(row)
        lbs.append(-np.inf)
        ubs.append(max(float(snapshot.group_work_capacity[g]), 0.0))

    # 任务剩余：sum_g A[i,g] <= remaining_work[i]
    for i, task in enumerate(snapshot.tasks):
        row = np.zeros(n_vars)
        for g in range(n_group):
            row[i * n_group + g] = 1.0
        rows.append(row)
        lbs.append(-np.inf)
        ubs.append(max(float(task.remaining_work), 0.0))

    # 充放互斥：charge - M*z <= 0；discharge + M*z <= M
    row_charge = np.zeros(n_vars)
    row_charge[idx_charge] = 1.0
    row_charge[idx_z] = -_BIG_M
    rows.append(row_charge)
    lbs.append(-np.inf)
    ubs.append(0.0)

    row_discharge = np.zeros(n_vars)
    row_discharge[idx_discharge] = 1.0
    row_discharge[idx_z] = _BIG_M
    rows.append(row_discharge)
    lbs.append(-np.inf)
    ubs.append(_BIG_M)

    # 接入上限：sum(A) + charge - discharge <= access_limit（简化功率映射）
    row_access = np.zeros(n_vars)
    row_access[:n_a] = 1.0
    row_access[idx_charge] = 1.0
    row_access[idx_discharge] = -1.0
    rows.append(row_access)
    lbs.append(-np.inf)
    ubs.append(max(float(snapshot.access_limit_kw), 0.0))

    constraints = [
        LinearConstraint(np.vstack(rows), np.array(lbs), np.array(ubs))
    ] if rows else []

    integrality = np.zeros(n_vars)
    if not allow_lp_relaxation:
        integrality[idx_z] = 1  # z 二元

    bounds = Bounds(
        lb=np.concatenate([np.zeros(n_a), [0.0, 0.0, 0.0]]),
        ub=np.concatenate([np.full(n_a, np.inf), [_BIG_M, _BIG_M, 1.0]]),
    )

    return MilpModel(
        n_task=n_task,
        n_group=n_group,
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
    )

"""M4.3a 时间索引 LP 规划核心（H 步）。

单位严格分离：
- **work**（work-units）：任务分配 `A[i,g,k]`、剩余工作、slack；
- **kW**：`P_idc` / `P_grid` / `pv_used` / `wind_used` / `charge` / `discharge` / `curtail`；
- **kWh**：SOC、能量吞吐（= kW × dt）；
- **SGD**：电费、储能退化成本、slack 罚项。

红线：
- 功率映射仅使用 snapshot 的**规划近似**字段（`base_idc_power` + `group_power_coeff_kw_per_work`），
  结果标记 `power_approximation_used=True`，执行前后须由环境物理链复核；
- 本 LP **不保证充放电互斥**（连续松弛）；若同一步同时充放，标记 `storage_relaxation_active=True`，
  该结果**不得进入执行路径**，互斥留待 M4.3b/M4.5 的 MIP；
- 不做窗口外真值读取（规划输入由 `planning_forecast` 提供，窗口外为声明假设）。

注：文件末尾保留**未接线遗留块**（旧单步 `build_milp`），供 `planning/solver.py` /
`planning/corrector.py` 继续使用；本卡不重新接线，遗留块将在 M4.3b/M4.5 移除。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog
from scipy.sparse import csr_matrix, lil_matrix

from contracts.models import SystemSnapshot

# --- 单位声明（供调用方与测试断言） ---
WORK_UNIT = "work-unit"
POWER_UNIT = "kW"
ENERGY_UNIT = "kWh"
ELECTRICITY_COST_UNIT = "SGD"
DEGRADATION_COST_UNIT = "SGD"
# (默认值, 单位) —— 研究用罚项系数，显式声明，不是从数据反推的货币值
BUSINESS_SHORTFALL_PENALTY_SGD_PER_WORK = (0.05, "SGD/work-unit")
DEADLINE_SHORTFALL_PENALTY_SGD_PER_WORK = (0.10, "SGD/work-unit")

FAILURE_NONE = "none"
FAILURE_BASE_SHORTAGE = "base_shortage"
FAILURE_DEADLINE_SHORTFALL = "deadline_shortfall"
FAILURE_SOLVER_FAILURE = "solver_failure"

_TOL = 1e-6


@dataclass
class LPPlanResult:
    """LP 规划结果（全部单位显式）。"""

    solver_status: str
    solve_time_s: float
    horizon_steps: int
    n_variables: int
    n_constraints: int

    # work-units
    allocation: np.ndarray  # (n_task, n_group, H)
    business_shortfall_work: float
    deadline_shortfall_work: float

    # kW
    p_idc_kw: list[float]
    p_grid_kw: list[float]
    pv_used_kw: list[float]
    wind_used_kw: list[float]
    charge_kw: list[float]
    discharge_kw: list[float]
    curtail_kw: list[float]

    # kWh
    soc_kwh: list[float]

    # SGD
    electricity_cost_sgd: float
    degradation_cost_sgd: float
    business_shortfall_cost_sgd: float
    deadline_shortfall_cost_sgd: float
    total_objective_sgd: float

    max_constraint_residual: float
    residuals_by_constraint: dict[str, float]

    power_approximation_used: bool
    storage_relaxation_active: bool
    failure_class: str


def _empty_result(
    snapshot: SystemSnapshot, H: int, n_vars: int, n_cons: int, status: str,
    solve_time_s: float, failure_class: str,
) -> LPPlanResult:
    n_task, n_group = len(snapshot.tasks), len(snapshot.group_work_capacity)
    zeros = [0.0] * H
    return LPPlanResult(
        solver_status=status,
        solve_time_s=solve_time_s,
        horizon_steps=H,
        n_variables=n_vars,
        n_constraints=n_cons,
        allocation=np.zeros((n_task, n_group, H)),
        business_shortfall_work=float(
            sum(t.remaining_work for t in snapshot.tasks)
        ),
        deadline_shortfall_work=0.0,
        p_idc_kw=list(zeros),
        p_grid_kw=list(zeros),
        pv_used_kw=list(zeros),
        wind_used_kw=list(zeros),
        charge_kw=list(zeros),
        discharge_kw=list(zeros),
        curtail_kw=list(zeros),
        soc_kwh=[float(snapshot.soc_kwh)] * (H + 1),
        electricity_cost_sgd=0.0,
        degradation_cost_sgd=0.0,
        business_shortfall_cost_sgd=0.0,
        deadline_shortfall_cost_sgd=0.0,
        total_objective_sgd=0.0,
        max_constraint_residual=0.0,
        residuals_by_constraint={},
        power_approximation_used=True,
        storage_relaxation_active=False,
        failure_class=failure_class,
    )


def _base_shortage_possible(snapshot: SystemSnapshot, H: int) -> bool:
    """基础负载是否在接入 + 可再生 + 最大放电之下仍无法满足。"""
    pf = snapshot.planning_forecast
    for k in range(H):
        available = snapshot.access_limit_kw + pf.pv[k] + pf.wind[k]
        if pf.base_idc_power[k] > available + _TOL:
            return True
    return False


def solve_time_indexed_lp(
    snapshot: SystemSnapshot,
    *,
    business_penalty_sgd_per_work: float = BUSINESS_SHORTFALL_PENALTY_SGD_PER_WORK[0],
    deadline_penalty_sgd_per_work: float = DEADLINE_SHORTFALL_PENALTY_SGD_PER_WORK[0],
) -> LPPlanResult:
    """求解 H 步时间索引 LP。不可行时显式分类并返回零分配（不伪造完成）。"""
    import time

    t0 = time.perf_counter()
    H = int(snapshot.planning_horizon_steps)
    n_task = len(snapshot.tasks)
    n_group = len(snapshot.group_work_capacity)
    dt = float(snapshot.delta_t_hours)
    pf = snapshot.planning_forecast

    n_a = n_task * n_group * H

    def a(i: int, g: int, k: int) -> int:
        return (i * n_group + g) * H + k

    off_pidc = n_a
    off_pgrid = off_pidc + H
    off_pv = off_pgrid + H
    off_wind = off_pv + H
    off_curtail = off_wind + H
    off_charge = off_curtail + H
    off_discharge = off_charge + H
    off_soc = off_discharge + H
    off_bus = off_soc + (H + 1)
    off_dls = off_bus + n_task
    n_vars = off_dls + n_task

    c = np.zeros(n_vars)
    for k in range(H):
        c[off_pgrid + k] = dt * pf.price[k]
        c[off_charge + k] = dt * snapshot.bess_degradation_cost_per_kwh
        c[off_discharge + k] = dt * snapshot.bess_degradation_cost_per_kwh
    for i in range(n_task):
        c[off_bus + i] = business_penalty_sgd_per_work
        c[off_dls + i] = deadline_penalty_sgd_per_work

    lb = np.full(n_vars, 0.0)
    ub = np.full(n_vars, np.inf)
    lb[off_pidc:off_pidc + H] = -np.inf  # 由功率等式决定
    ub[off_pgrid:off_pgrid + H] = max(float(snapshot.access_limit_kw), 0.0)
    for k in range(H):
        ub[off_pv + k] = max(float(pf.pv[k]), 0.0)
        ub[off_wind + k] = max(float(pf.wind[k]), 0.0)
        ub[off_charge + k] = float(snapshot.bess_charge_power_max_kw)
        ub[off_discharge + k] = float(snapshot.bess_discharge_power_max_kw)
    lb[off_soc:off_soc + H + 1] = float(snapshot.soc_min_kwh)
    ub[off_soc:off_soc + H + 1] = float(snapshot.soc_max_kwh)

    rows: list[dict[int, float]] = []
    lbs: list[float] = []
    ubs: list[float] = []
    names: list[str] = []

    def add(row: dict[int, float], lo: float, hi: float, name: str) -> None:
        rows.append(row)
        lbs.append(lo)
        ubs.append(hi)
        names.append(name)

    # 功率等式：P_idc[k] - sum_g coeff[g] * sum_i A = base_idc_power[k]
    for k in range(H):
        row = {off_pidc + k: 1.0}
        for g in range(n_group):
            coeff = float(snapshot.group_power_coeff_kw_per_work[g])
            if coeff:
                for i in range(n_task):
                    row[a(i, g, k)] = -coeff
        base_kw = float(pf.base_idc_power[k])
        add(row, base_kw, base_kw, f"power_definition[{k}]")

    # 能量平衡：P_grid + pv_used + wind_used + discharge - P_idc - charge - curtail = 0
    for k in range(H):
        row = {
            off_pgrid + k: 1.0, off_pv + k: 1.0, off_wind + k: 1.0,
            off_discharge + k: 1.0, off_pidc + k: -1.0,
            off_charge + k: -1.0, off_curtail + k: -1.0,
        }
        add(row, 0.0, 0.0, f"energy_balance[{k}]")

    # 弃电定义：curtail[k] + pv_used[k] + wind_used[k] = pv[k] + wind[k]
    for k in range(H):
        row = {off_curtail + k: 1.0, off_pv + k: 1.0, off_wind + k: 1.0}
        avail = float(pf.pv[k]) + float(pf.wind[k])
        add(row, avail, avail, f"curtailment_definition[{k}]")

    # 储能动态：SOC[k+1] - SOC[k] - eta_c*dt*charge + (dt/eta_d)*discharge = 0
    for k in range(H):
        row = {
            off_soc + k + 1: 1.0, off_soc + k: -1.0,
            off_charge + k: -snapshot.bess_charge_efficiency * dt,
            off_discharge + k: dt / max(snapshot.bess_discharge_efficiency, 1e-9),
        }
        add(row, 0.0, 0.0, f"soc_dynamics[{k}]")

    # 初始 SOC
    add({off_soc: 1.0}, float(snapshot.soc_kwh), float(snapshot.soc_kwh), "soc_initial")

    # 每任务每步速率
    for i, task in enumerate(snapshot.tasks):
        rate = float(task.max_rate_work_per_step)
        for k in range(H):
            row = {a(i, g, k): 1.0 for g in range(n_group)}
            add(row, -np.inf, rate, f"task_rate[{i},{k}]")

    # 每组每步容量
    for g in range(n_group):
        cap = float(snapshot.group_work_capacity[g])
        for k in range(H):
            row = {a(i, g, k): 1.0 for i in range(n_task)}
            add(row, -np.inf, cap, f"group_capacity[{g},{k}]")

    # 每任务全时域剩余工作 + business slack
    for i, task in enumerate(snapshot.tasks):
        row = {off_bus + i: 1.0}
        for g in range(n_group):
            for k in range(H):
                row[a(i, g, k)] = 1.0
        remaining = float(task.remaining_work)
        add(row, remaining, remaining, f"business_balance[{i}]")

    # 期限：deadline 落在时域内的任务建 deadline slack
    for i, task in enumerate(snapshot.tasks):
        kd = int(task.deadline) - int(snapshot.step)
        if kd < 0:
            kd = 0
        if kd > H:
            kd = H
        if kd >= H:
            continue  # 期限在时域之外
        row = {off_dls + i: 1.0}
        for g in range(n_group):
            for k in range(kd):
                row[a(i, g, k)] = 1.0
        remaining = float(task.remaining_work)
        add(row, remaining, np.inf, f"deadline_balance[{i}]")

    A_mat = lil_matrix((len(rows), n_vars))
    for r, row in enumerate(rows):
        for col, val in row.items():
            A_mat[r, col] = val
    A_csr = csr_matrix(A_mat)

    # 拆分等式与不等式（linprog 接口）
    eq_idx, ub_idx, lb_idx = [], [], []
    for r in range(len(rows)):
        if np.isneginf(lbs[r]):
            ub_idx.append(r)
        elif np.isposinf(ubs[r]):
            lb_idx.append(r)
        else:
            eq_idx.append(r)
    a_eq = A_csr[eq_idx] if eq_idx else None
    b_eq = np.array([ubs[r] for r in eq_idx]) if eq_idx else None
    if ub_idx or lb_idx:
        blocks = []
        b_ub_list = []
        for r in ub_idx:
            blocks.append(A_csr[r])
            b_ub_list.append(ubs[r])
        for r in lb_idx:
            blocks.append(-A_csr[r])
            b_ub_list.append(-lbs[r])
        a_ub = csr_matrix(np.vstack([b.toarray() for b in blocks]))
        b_ub = np.array(b_ub_list)
    else:
        a_ub, b_ub = None, None

    res = linprog(
        c=c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
        bounds=list(zip(lb, ub, strict=True)), method="highs",
    )
    solve_time = time.perf_counter() - t0

    if not res.success:
        base_short = _base_shortage_possible(snapshot, H)
        failure = FAILURE_BASE_SHORTAGE if base_short else FAILURE_SOLVER_FAILURE
        return _empty_result(snapshot, H, n_vars, len(rows), str(res.message),
                             solve_time, failure)

    x = res.x
    allocation = np.zeros((n_task, n_group, H))
    for i in range(n_task):
        for g in range(n_group):
            for k in range(H):
                allocation[i, g, k] = max(x[a(i, g, k)], 0.0)

    p_idc = [float(x[off_pidc + k]) for k in range(H)]
    p_grid = [float(x[off_pgrid + k]) for k in range(H)]
    pv_used = [float(x[off_pv + k]) for k in range(H)]
    wind_used = [float(x[off_wind + k]) for k in range(H)]
    charge = [float(x[off_charge + k]) for k in range(H)]
    discharge = [float(x[off_discharge + k]) for k in range(H)]
    curtail = [float(x[off_curtail + k]) for k in range(H)]
    soc = [float(x[off_soc + k]) for k in range(H + 1)]
    business = [float(x[off_bus + i]) for i in range(n_task)]
    deadline = [float(x[off_dls + i]) for i in range(n_task)]

    electricity_cost = sum(
        p_grid[k] * dt * pf.price[k] for k in range(H)
    )
    degradation_cost = sum(
        (charge[k] + discharge[k]) * dt * snapshot.bess_degradation_cost_per_kwh
        for k in range(H)
    )
    business_cost = sum(business) * business_penalty_sgd_per_work
    deadline_cost = sum(deadline) * deadline_penalty_sgd_per_work

    ax = np.asarray(A_csr.dot(x)).ravel()
    residuals = {}
    for r, name in enumerate(names):
        value = float(ax[r])
        if np.isneginf(lbs[r]):
            residuals[name] = max(value - ubs[r], 0.0)
        elif np.isposinf(ubs[r]):
            residuals[name] = max(lbs[r] - value, 0.0)
        else:
            residuals[name] = abs(value - lbs[r])
    max_residual = max(residuals.values()) if residuals else 0.0

    storage_relaxation = any(
        c_k > _TOL and d_k > _TOL for c_k, d_k in zip(charge, discharge, strict=True)
    )
    total_business = float(sum(business))
    total_deadline = float(sum(deadline))
    if total_deadline > _TOL:
        failure = FAILURE_DEADLINE_SHORTFALL
    else:
        failure = FAILURE_NONE

    return LPPlanResult(
        solver_status="optimal",
        solve_time_s=solve_time,
        horizon_steps=H,
        n_variables=n_vars,
        n_constraints=len(rows),
        allocation=allocation,
        business_shortfall_work=total_business,
        deadline_shortfall_work=total_deadline,
        p_idc_kw=p_idc,
        p_grid_kw=p_grid,
        pv_used_kw=pv_used,
        wind_used_kw=wind_used,
        charge_kw=charge,
        discharge_kw=discharge,
        curtail_kw=curtail,
        soc_kwh=soc,
        electricity_cost_sgd=electricity_cost,
        degradation_cost_sgd=degradation_cost,
        business_shortfall_cost_sgd=business_cost,
        deadline_shortfall_cost_sgd=deadline_cost,
        total_objective_sgd=electricity_cost + degradation_cost + business_cost + deadline_cost,
        max_constraint_residual=max_residual,
        residuals_by_constraint=residuals,
        power_approximation_used=True,
        storage_relaxation_active=storage_relaxation,
        failure_class=failure,
    )


# ---------------------------------------------------------------------------
# 未接线遗留块（M4.3a 保留）：旧单步 MILP，仍被 planning/solver.py 与
# planning/corrector.py 使用。本卡不重新接线；M4.3b/M4.5 重新接线后移除。
# ---------------------------------------------------------------------------
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
    """[遗留] 单步 MILP：最大化任务完成量，受组容量/任务剩余/充放互斥/接入上限约束。"""
    n_task = len(snapshot.tasks)
    n_group = len(snapshot.group_work_capacity)
    n_a = n_task * n_group
    n_vars = n_a + 3
    idx_charge = n_a
    idx_discharge = n_a + 1
    idx_z = n_a + 2

    c = np.zeros(n_vars)
    c[:n_a] = -1.0

    rows: list[np.ndarray] = []
    lbs: list[float] = []
    ubs: list[float] = []

    for g in range(n_group):
        row = np.zeros(n_vars)
        for i in range(n_task):
            row[i * n_group + g] = 1.0
        rows.append(row)
        lbs.append(-np.inf)
        ubs.append(max(float(snapshot.group_work_capacity[g]), 0.0))

    for i, task in enumerate(snapshot.tasks):
        row = np.zeros(n_vars)
        for g in range(n_group):
            row[i * n_group + g] = 1.0
        rows.append(row)
        lbs.append(-np.inf)
        ubs.append(max(float(task.remaining_work), 0.0))

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
        integrality[idx_z] = 1

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

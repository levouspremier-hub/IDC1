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

现行规划路径（唯一）：**H 步 LP**（诊断 / 松弛下界）、**H 步 MIP**（执行候选）、
**H 步两阶段 MIP raw projection**（corrector / wrapper 活链路）。
旧单步规划实现及其专用求解模块已于 **M4.4b** 整体退役，本文件不再导出它们。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog
from scipy.sparse import csr_matrix, lil_matrix, vstack

from contracts.inventory import InventorySnapshot, validate_inventory_snapshot
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

# --- M5.4f：确定性 MIP 配置 -------------------------------------------------
# HiGHS 在 time_limit 下可能走到**不同最优顶点**（同代价、均通过物理校验），
# 也可能因「预算够不够」的时序竞争而有时超时、有时收敛 —— 两者都破坏跨进程可复现。
# 这里固定可用的确定性杠杆：
#   random_seed=0  —— 固定内部随机化起点；
#   parallel=False —— 关闭并行（HiGHS 内部不可复现的主要来源之一）。
# 注意：HiGHS 的 `threads` 选项经 scipy 的 _highs_wrapper 会直接崩溃（实测 TypeError），
# 因此**不可用**；`parallel=False` 是当前可用的最接近手段。
# 这些选项**不改变**可行域、约束或目标函数，只影响求解路径。
DETERMINISTIC_RANDOM_SEED = 0
DETERMINISTIC_PARALLEL = False
INVENTORY_SOLVER_FEASIBILITY_TOLERANCE = 1e-9


def deterministic_mip_options(*, time_limit_s: float | None) -> dict:
    """返回传给 `scipy.optimize.milp` 的确定性选项（纯函数，与调用次数无关）。"""
    options: dict = {
        "random_seed": DETERMINISTIC_RANDOM_SEED,
        "parallel": DETERMINISTIC_PARALLEL,
    }
    if time_limit_s is not None:
        options["time_limit"] = float(time_limit_s)
    return options


SOLVER_OPTIMAL = "optimal"
SOLVER_TIME_LIMIT = "time_limit"
SOLVER_INFEASIBLE = "infeasible"
SOLVER_FAILURE = "solver_failure"

FAILURE_NONE = "none"
FAILURE_BASE_SHORTAGE = "base_shortage"
FAILURE_DEADLINE_SHORTFALL = "deadline_shortfall"
FAILURE_SOLVER_FAILURE = "solver_failure"
FAILURE_TIMEOUT = "timeout"
FAILURE_PROPOSAL_INVALID = "proposal_invalid"

_TOL = 1e-6


def _monotonic() -> float:
    """单调时钟（可被测试替换，用于确定性 deadline 测试）。"""
    import time

    return time.monotonic()


def _sparse_rows(rows: list[dict[int, float]], n_variables: int) -> csr_matrix:
    """Build the identical constraint matrix without Python sparse assignment."""
    # NumPy buffers avoid one GC-tracked tuple per coefficient. Keep the
    # canonical column ordering produced by the previous COO -> CSR path.
    counts = np.fromiter((sum(v != 0.0 for v in row.values()) for row in rows),
                         dtype=np.int64, count=len(rows))
    indptr = np.empty(len(rows) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1])
    indices = np.fromiter((c for row in rows for c, v in row.items() if v != 0.0),
                          dtype=np.int64, count=nnz)
    values = np.fromiter((v for row in rows for v in row.values() if v != 0.0),
                         dtype=np.float64, count=nnz)
    matrix = csr_matrix((values, indices, indptr), shape=(len(rows), n_variables))
    matrix.sort_indices()
    return matrix


def _append_sparse_row(matrix: csr_matrix, row: dict[int, float]) -> csr_matrix:
    """Stage B adds only its offset bound; all Stage A coefficients are reused."""
    return vstack((matrix, _sparse_rows([row], matrix.shape[1])), format="csr")


def _base_only_terminal_certificate(snapshot: InventorySnapshot) -> dict | None:
    """Construct a zero-gap witness in a subset of the existing feasible domain.

    Task allocation is zero, or the registered current arrived service reserve.
    Existing business/deadline slacks retain all unallocated work.
    Signed charge/discharge controls have continuous interval images. Propagate
    those images with SOC bounds, then backtrack a target witness. Failure here
    proves nothing about the full model (tasks can support additional discharge).
    The caller must validate the witness against its complete actual matrix.
    """
    terminal = snapshot.terminal_inventory
    if terminal is None:
        return None
    dt, eta_c, eta_d = (snapshot.delta_t_hours, snapshot.bess_charge_efficiency,
                       snapshot.bess_discharge_efficiency)
    if min(dt, eta_c, eta_d) <= 0:
        return None
    intervals = [(snapshot.soc_kwh, snapshot.soc_kwh)]
    increments = []
    pf = snapshot.planning_forecast
    guard = snapshot.service_guard
    coupled = guard is not None and guard.version in (
        "arrived-service-reserve-v2", "arrived-service-reserve-v3")
    for k in range(snapshot.planning_horizon_steps):
        base = pf.base_idc_power[k]
        if coupled and guard is not None:
            group_work = np.asarray(guard.known_service_allocation[k]).reshape(
                len(snapshot.tasks), len(snapshot.group_work_capacity)).sum(axis=0)
            base += float(np.dot(snapshot.group_power_coeff_kw_per_work, group_work))
        elif k == 0 and guard is not None:
            base += sum(c * w for c, w in zip(snapshot.group_power_coeff_kw_per_work,
                                            guard.group_work_floor, strict=True))
        supply = snapshot.access_limit_kw + pf.pv[k] + pf.wind[k]
        forced_discharge = max(base - supply, 0.)
        max_discharge = min(snapshot.bess_discharge_power_max_kw, base)
        if forced_discharge > max_discharge:
            return None
        lower_increment = -max_discharge * dt / eta_d
        upper_increment = (-forced_discharge * dt / eta_d if forced_discharge > 0
                           else min(snapshot.bess_charge_power_max_kw,
                                    guard.charge_limits_kw[k] if guard is not None
                                    else snapshot.bess_charge_power_max_kw,
                                    max(supply - base, 0.)) * dt * eta_c)
        lo = max(snapshot.soc_min_kwh, intervals[-1][0] + lower_increment)
        hi = min(snapshot.soc_max_kwh, intervals[-1][1] + upper_increment)
        if lo > hi:
            return None
        increments.append((lower_increment, upper_increment))
        intervals.append((lo, hi))
    if not intervals[-1][0] <= terminal.target_kwh <= intervals[-1][1]:
        return None
    energy = [terminal.target_kwh]
    for k in reversed(range(snapshot.planning_horizon_steps)):
        lo = max(intervals[k][0], energy[-1] - increments[k][1])
        hi = min(intervals[k][1], energy[-1] - increments[k][0])
        if lo > hi + 1e-10:
            return None
        energy.append(min(max(energy[-1], lo), hi))
    energy.reverse()
    charge, discharge = [], []
    for before, after in zip(energy[:-1], energy[1:], strict=True):
        change = after - before
        charge.append(max(change, 0.) / (dt * eta_c))
        discharge.append(max(-change, 0.) * eta_d / dt)
    return {"energy_kwh": energy, "charge_kw": charge, "discharge_kw": discharge}


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

    # 后端与整数规模（M4.3b）
    backend: str
    n_integer_variables: int

    # 基础负载诊断（M4.3a1）
    base_diagnostic_status: str
    base_diagnostic_solve_time_s: float
    base_shortfall_kwh: float


def _empty_result(
    snapshot: SystemSnapshot, H: int, n_vars: int, n_cons: int, status: str,
    solve_time_s: float, failure_class: str,
    base_diag: BaseFeasibilityDiagnostic | None = None,
    backend: str = "lp", n_integer_variables: int = 0,
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
        backend=backend,
        n_integer_variables=n_integer_variables,
        base_diagnostic_status=(base_diag.status if base_diag else "not_run"),
        base_diagnostic_solve_time_s=(base_diag.solve_time_s if base_diag else 0.0),
        base_shortfall_kwh=(base_diag.shortfall_kwh if base_diag else 0.0),
    )


@dataclass
class BaseFeasibilityDiagnostic:
    """基础负载专用可行性诊断结果（M4.3a1）。"""

    feasible: bool
    status: str
    solve_time_s: float
    shortfall_kwh: float


@dataclass
class _BaseOnlyModel:
    c: np.ndarray
    a_eq: csr_matrix
    b_eq: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    n_vars: int


def _base_only_lp(
    snapshot: SystemSnapshot, H: int, with_shortfall_slack: bool
) -> _BaseOnlyModel:
    """构建 base-only 模型：P_idc 固定为 base_idc_power，不含任何任务变量。

    `with_shortfall_slack=True` 时加入非负「未服务基础功率」松弛量并最小化其总能量（kWh），
    用于量化缺口；否则做纯可行性判定。
    """
    pf = snapshot.planning_forecast
    dt = float(snapshot.delta_t_hours)

    off_pgrid, off_pv, off_wind = 0, H, 2 * H
    off_curtail, off_charge, off_discharge = 3 * H, 4 * H, 5 * H
    off_soc = 6 * H
    off_unserved = off_soc + (H + 1)
    n_vars = off_unserved + (H if with_shortfall_slack else 0)

    c = np.zeros(n_vars)
    if with_shortfall_slack:
        for k in range(H):
            c[off_unserved + k] = dt  # 最小化未服务基础能量（kWh）

    lb = np.full(n_vars, 0.0)
    ub = np.full(n_vars, np.inf)
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

    def add(row: dict[int, float], lo: float, hi: float) -> None:
        rows.append(row)
        lbs.append(lo)
        ubs.append(hi)

    for k in range(H):
        # 能量平衡（与主 LP / env 一致）：
        # P_grid + pv_used + wind_used + discharge (+ unserved) = P_idc + charge
        row = {
            off_pgrid + k: 1.0, off_pv + k: 1.0, off_wind + k: 1.0,
            off_discharge + k: 1.0, off_charge + k: -1.0,
        }
        const = float(pf.base_idc_power[k])
        if with_shortfall_slack:
            row[off_unserved + k] = 1.0
        add(row, const, const)
        # 弃电定义：curtail + pv_used + wind_used = pv + wind
        add(
            {off_curtail + k: 1.0, off_pv + k: 1.0, off_wind + k: 1.0},
            float(pf.pv[k]) + float(pf.wind[k]),
            float(pf.pv[k]) + float(pf.wind[k]),
        )
        # SOC 动态
        add(
            {
                off_soc + k + 1: 1.0, off_soc + k: -1.0,
                off_charge + k: -snapshot.bess_charge_efficiency * dt,
                off_discharge + k: dt / max(snapshot.bess_discharge_efficiency, 1e-9),
            },
            0.0, 0.0,
        )
    add({off_soc: 1.0}, float(snapshot.soc_kwh), float(snapshot.soc_kwh))

    A_mat = lil_matrix((len(rows), n_vars))
    for r, row in enumerate(rows):
        for col, val in row.items():
            A_mat[r, col] = val
    return _BaseOnlyModel(
        c=c, a_eq=csr_matrix(A_mat), b_eq=np.array(ubs), lb=lb, ub=ub, n_vars=n_vars
    )


def diagnose_base_feasibility(
    snapshot: SystemSnapshot, *, time_limit_s: float | None = None
) -> BaseFeasibilityDiagnostic:
    """基础负载专用可行性诊断（M4.3a1 / M4.4a2）。

    用与主 LP **相同的物理约束**（接入、PV/wind 使用与弃电、充放电、SOC、效率、无反送电、
    能量平衡）判定「仅基础负载」是否可服务；不含任何任务变量、任务 slack 或未来真实任务。

    `time_limit_s` 为**诊断可用的剩余预算**：内部两次 LP **共享同一 deadline**，
    不会各自重新获得完整预算；任一次超时 → `outcome="time_limit"`（不得据此报 base_shortage）。
    """
    import time

    H = int(snapshot.planning_horizon_steps)
    t0 = time.perf_counter()
    deadline = None if time_limit_s is None else _monotonic() + float(time_limit_s)

    def _remaining() -> float | None:
        if deadline is None:
            return None
        return max(deadline - _monotonic(), 0.0)

    def _lp_options() -> dict:
        rem = _remaining()
        return {} if rem is None else {"time_limit": float(rem)}

    if H <= 0:
        return BaseFeasibilityDiagnostic(True, "trivial_empty_horizon", 0.0, 0.0)

    rem0 = _remaining()
    if rem0 is not None and rem0 <= 0.0:
        return BaseFeasibilityDiagnostic(False, "diagnostic_not_started", 0.0, float("nan"))

    model = _base_only_lp(snapshot, H, with_shortfall_slack=False)
    res = linprog(
        c=model.c, A_eq=model.a_eq, b_eq=model.b_eq,
        bounds=list(zip(model.lb, model.ub, strict=True)),
        method="highs", options=_lp_options(),
    )
    if res.success:
        return BaseFeasibilityDiagnostic(
            True, str(res.message), time.perf_counter() - t0, 0.0
        )
    if int(getattr(res, "status", 4)) == 1:
        return BaseFeasibilityDiagnostic(
            False, "diagnostic_time_limit", time.perf_counter() - t0, float("nan")
        )

    # 不可行：用松弛量量化最小未服务基础能量（kWh）——仍受同一 deadline 约束
    rem1 = _remaining()
    if rem1 is not None and rem1 <= 0.0:
        return BaseFeasibilityDiagnostic(
            False, "diagnostic_time_limit", time.perf_counter() - t0, float("nan")
        )
    model2 = _base_only_lp(snapshot, H, with_shortfall_slack=True)
    res2 = linprog(
        c=model2.c, A_eq=model2.a_eq, b_eq=model2.b_eq,
        bounds=list(zip(model2.lb, model2.ub, strict=True)),
        method="highs", options=_lp_options(),
    )
    if int(getattr(res2, "status", 4)) == 1:
        return BaseFeasibilityDiagnostic(
            False, "diagnostic_time_limit", time.perf_counter() - t0, float("nan")
        )
    shortfall = float(res2.fun) if res2.success else float("nan")
    status = f"infeasible ({res.message})" if res2.success else str(res2.message)
    return BaseFeasibilityDiagnostic(False, status, time.perf_counter() - t0, shortfall)


def _add_core_constraints(
    add, snapshot, pf, *, H, n_task, n_group, dt, a, off, mutual_exclusion
) -> None:
    """H 步时间索引模型的核心约束（LP/MIP/投影共用，避免模型分叉）。

    含：功率定义、能量平衡、弃电定义、SOC 动力学与初值、储能互斥（可选）、
    逐任务速率、逐组容量、任务剩余工作 + business slack、期限 slack。
    """
    # 功率等式：P_idc[k] - sum_g coeff[g] * sum_i A = base_idc_power[k]
    for k in range(H):
        row = {off["pidc"] + k: 1.0}
        for g in range(n_group):
            coeff = float(snapshot.group_power_coeff_kw_per_work[g])
            if coeff:
                for i in range(n_task):
                    row[a(i, g, k)] = -coeff
        base_kw = float(pf.base_idc_power[k])
        add(row, base_kw, base_kw, f"power_definition[{k}]")

    # 能量平衡（与 env M3.5 一致）：P_grid + pv_used + wind_used + discharge = P_idc + charge
    # 注：弃电是**未被消费**的可再生，不得出现在需求侧（否则等于双重计数）。
    for k in range(H):
        row = {
            off["pgrid"] + k: 1.0, off["pv"] + k: 1.0, off["wind"] + k: 1.0,
            off["discharge"] + k: 1.0, off["pidc"] + k: -1.0, off["charge"] + k: -1.0,
        }
        add(row, 0.0, 0.0, f"energy_balance[{k}]")

    # 弃电定义：curtail[k] + pv_used[k] + wind_used[k] = pv[k] + wind[k]
    for k in range(H):
        row = {off["curtail"] + k: 1.0, off["pv"] + k: 1.0, off["wind"] + k: 1.0}
        avail = float(pf.pv[k]) + float(pf.wind[k])
        add(row, avail, avail, f"curtailment_definition[{k}]")

    # 储能动态：SOC[k+1] - SOC[k] - eta_c*dt*charge + (dt/eta_d)*discharge = 0
    for k in range(H):
        row = {
            off["soc"] + k + 1: 1.0, off["soc"] + k: -1.0,
            off["charge"] + k: -snapshot.bess_charge_efficiency * dt,
            off["discharge"] + k: dt / max(snapshot.bess_discharge_efficiency, 1e-9),
        }
        add(row, 0.0, 0.0, f"soc_dynamics[{k}]")

    # 初始 SOC
    add({off["soc"]: 1.0}, float(snapshot.soc_kwh), float(snapshot.soc_kwh), "soc_initial")

    # 储能互斥（仅 MIP）：charge <= charge_max*z；discharge <= discharge_max*(1-z)
    if mutual_exclusion:
        charge_max = float(snapshot.bess_charge_power_max_kw)
        discharge_max = float(snapshot.bess_discharge_power_max_kw)
        for k in range(H):
            add(
                {off["charge"] + k: 1.0, off["z"] + k: -charge_max},
                -np.inf, 0.0, f"storage_excl_charge[{k}]",
            )
            add(
                {off["discharge"] + k: 1.0, off["z"] + k: discharge_max},
                -np.inf, discharge_max, f"storage_excl_discharge[{k}]",
            )

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
        row = {off["bus"] + i: 1.0}
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
        row = {off["dls"] + i: 1.0}
        for g in range(n_group):
            for k in range(kd):
                row[a(i, g, k)] = 1.0
        remaining = float(task.remaining_work)
        add(row, remaining, np.inf, f"deadline_balance[{i}]")


def solve_time_indexed_lp(
    snapshot: SystemSnapshot,
    *,
    business_penalty_sgd_per_work: float = BUSINESS_SHORTFALL_PENALTY_SGD_PER_WORK[0],
    deadline_penalty_sgd_per_work: float = DEADLINE_SHORTFALL_PENALTY_SGD_PER_WORK[0],
) -> LPPlanResult:
    """求解 H 步时间索引 **LP**（连续松弛）。

    仅作为**诊断与松弛下界**：允许同一时刻同时充放电，不得作为执行候选；
    可执行候选须用 `solve_time_indexed_mip`。
    """
    return _solve_time_indexed(
        snapshot, backend="lp", mutual_exclusion=False, time_limit_s=None,
        business_penalty_sgd_per_work=business_penalty_sgd_per_work,
        deadline_penalty_sgd_per_work=deadline_penalty_sgd_per_work,
    )


def solve_time_indexed_mip(
    snapshot: SystemSnapshot,
    *,
    time_limit_s: float | None = None,
    business_penalty_sgd_per_work: float = BUSINESS_SHORTFALL_PENALTY_SGD_PER_WORK[0],
    deadline_penalty_sgd_per_work: float = DEADLINE_SHORTFALL_PENALTY_SGD_PER_WORK[0],
) -> LPPlanResult:
    """求解 H 步时间索引 **MIP**：逐步强制储能充/放电互斥。

    每步 k 引入二元变量 `z[k]`：`z[k]=1` 允许充电（放电=0），`z[k]=0` 允许放电（充电=0）：
        charge[k]    <= charge_max    * z[k]
        discharge[k] <= discharge_max * (1 - z[k])
    因此 `charge[k] * discharge[k] == 0` 对每个 k 严格成立。

    这是**唯一允许在后续接线时作为执行候选**的规划后端。
    """
    return _solve_time_indexed(
        snapshot, backend="mip", mutual_exclusion=True, time_limit_s=time_limit_s,
        business_penalty_sgd_per_work=business_penalty_sgd_per_work,
        deadline_penalty_sgd_per_work=deadline_penalty_sgd_per_work,
    )


def _solve_time_indexed(
    snapshot: SystemSnapshot,
    *,
    backend: str,
    mutual_exclusion: bool,
    time_limit_s: float | None,
    business_penalty_sgd_per_work: float,
    deadline_penalty_sgd_per_work: float,
) -> LPPlanResult:
    """LP/MIP 共用的 H 步时间索引求解器。不可行/超时显式分类并返回零分配（不伪造完成）。"""
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
    off_z = off_dls + n_task  # 储能互斥二元变量（仅 MIP 使用）
    n_z = H if mutual_exclusion else 0
    n_vars = off_z + n_z

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
    if mutual_exclusion:
        lb[off_z:off_z + H] = 0.0
        ub[off_z:off_z + H] = 1.0

    rows: list[dict[int, float]] = []
    lbs: list[float] = []
    ubs: list[float] = []
    names: list[str] = []

    def add(row: dict[int, float], lo: float, hi: float, name: str) -> None:
        rows.append(row)
        lbs.append(lo)
        ubs.append(hi)
        names.append(name)

    _add_core_constraints(
        add, snapshot, pf, H=H, n_task=n_task, n_group=n_group, dt=dt, a=a,
        off={
            "pidc": off_pidc, "pgrid": off_pgrid, "pv": off_pv, "wind": off_wind,
            "curtail": off_curtail, "charge": off_charge, "discharge": off_discharge,
            "soc": off_soc, "bus": off_bus, "dls": off_dls, "z": off_z,
        },
        mutual_exclusion=mutual_exclusion,
    )

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

    deadline = None if time_limit_s is None else _monotonic() + float(time_limit_s)

    def _remaining() -> float | None:
        if deadline is None:
            return None
        return max(deadline - _monotonic(), 0.0)

    if backend == "mip":
        from scipy.optimize import milp as _milp

        integrality = np.zeros(n_vars)
        integrality[off_z:off_z + H] = 1
        rem = _remaining()
        # M5.4f：固定确定性选项（不改可行域与目标，只固定求解路径）
        options = deterministic_mip_options(time_limit_s=rem)
        res = _milp(
            c=c,
            constraints=[LinearConstraint(A_csr, np.array(lbs), np.array(ubs))],
            integrality=integrality,
            bounds=Bounds(lb=lb, ub=ub),
            options=options,
        )
    else:
        res = linprog(
            c=c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
            bounds=list(zip(lb, ub, strict=True)), method="highs",
        )
    solve_time = time.perf_counter() - t0

    # 结构化状态：optimal / time_limit / infeasible / solver_failure（不得包装成成功）
    status_code = int(getattr(res, "status", 4))
    if status_code == 0:
        solver_status = SOLVER_OPTIMAL
    elif status_code == 1:
        solver_status = SOLVER_TIME_LIMIT
    elif status_code == 2:
        solver_status = SOLVER_INFEASIBLE
    else:
        solver_status = SOLVER_FAILURE

    n_int = n_z
    if solver_status != SOLVER_OPTIMAL:
        if solver_status == SOLVER_TIME_LIMIT:
            # 超时：立即返回零执行量 + 全部剩余业务缺口；
            # **不调用 diagnose_base_feasibility**（不得为失败分类再启动第二次求解），
            # 也不得使用超时 incumbent 作为结果。
            return _empty_result(
                snapshot, H, n_vars, len(rows), solver_status, solve_time,
                FAILURE_TIMEOUT, None,
                backend=backend, n_integer_variables=n_int,
            )
        base_diag = None
        failure = FAILURE_SOLVER_FAILURE
        if solver_status == SOLVER_INFEASIBLE:
            rem = _remaining()
            if rem is not None and rem <= 0.0:
                # 预算耗尽 → 不启动诊断，整体 timeout（不得误报 base_shortage）
                return _empty_result(
                    snapshot, H, n_vars, len(rows), SOLVER_TIME_LIMIT, solve_time,
                    FAILURE_TIMEOUT, None,
                    backend=backend, n_integer_variables=n_int,
                )
            base_diag = diagnose_base_feasibility(snapshot, time_limit_s=rem)
            if "time_limit" in base_diag.status or "not_started" in base_diag.status:
                return _empty_result(
                    snapshot, H, n_vars, len(rows), SOLVER_TIME_LIMIT, solve_time,
                    FAILURE_TIMEOUT, base_diag,
                    backend=backend, n_integer_variables=n_int,
                )
            failure = (
                FAILURE_BASE_SHORTAGE if not base_diag.feasible else FAILURE_SOLVER_FAILURE
            )
        return _empty_result(
            snapshot, H, n_vars, len(rows), solver_status, solve_time, failure,
            base_diag, backend=backend, n_integer_variables=n_int,
        )

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
    business_slack = [float(x[off_bus + i]) for i in range(n_task)]
    deadline_slack = [float(x[off_dls + i]) for i in range(n_task)]

    electricity_cost = sum(
        p_grid[k] * dt * pf.price[k] for k in range(H)
    )
    degradation_cost = sum(
        (charge[k] + discharge[k]) * dt * snapshot.bess_degradation_cost_per_kwh
        for k in range(H)
    )
    business_cost = sum(business_slack) * business_penalty_sgd_per_work
    deadline_cost = sum(deadline_slack) * deadline_penalty_sgd_per_work

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
    total_business = float(sum(business_slack))
    total_deadline = float(sum(deadline_slack))
    if total_deadline > _TOL:
        failure = FAILURE_DEADLINE_SHORTFALL
    else:
        failure = FAILURE_NONE

    return LPPlanResult(
        solver_status=SOLVER_OPTIMAL,
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
        backend=backend,
        n_integer_variables=n_int,
        base_diagnostic_status="not_required_optimal",
        base_diagnostic_solve_time_s=0.0,
        base_shortfall_kwh=0.0,
    )


# --- M4.4a 原始动作最小偏移投影（两阶段，仅 MIP） ---

RAW_STAGE_B_TOL = 1e-6  # 阶段 B 的偏移容差（无量纲）


def check_projection_candidate(x, lb, ub, integrality, matrix, row_lb, row_ub):
    """Independent primal check, including ranged rows and integer variables."""
    if x is None or np.shape(x) != np.shape(lb) or not np.all(np.isfinite(x)):
        return {"passed": False, "reason": "missing_shape_or_nonfinite"}
    bound = float(max(np.max(np.maximum(lb - x, 0.), initial=0.),
                      np.max(np.maximum(x - ub, 0.), initial=0.)))
    integers = np.asarray(x)[np.asarray(integrality) != 0]
    integer = float(np.max(np.abs(integers - np.rint(integers)), initial=0.))
    ax = np.asarray(matrix @ x).ravel()
    row = float(max(np.max(np.maximum(row_lb - ax, 0.), initial=0.),
                    np.max(np.maximum(ax - row_ub, 0.), initial=0.)))
    return {"passed": bool(max(bound, integer, row) <= _TOL),
            "bound_residual": bound, "integer_residual": integer,
            "row_residual": row, "tolerance": _TOL}


@dataclass
class RawProjectionResult:
    """第 0 步原始动作投影结果（全部单位显式、可审计）。"""

    backend: str
    solver_status: str
    failure_class: str
    horizon_steps: int
    n_variables: int
    n_integer_variables: int
    n_constraints: int

    # 两阶段审计
    stage_a_status: str
    stage_b_status: str
    stage_a_solve_time_s: float
    stage_b_solve_time_s: float
    stage_a_objective: float
    stage_b_objective: float
    projection_offset: float

    # 诊断审计（M4.4a2）：not_run / not_started / diagnostic_time_limit / infeasible(...)
    diagnostic_status: str
    diagnostic_solve_time_s: float

    # 第 0 步 exec action
    exec_compute_actions: list[float]
    exec_storage_action: float

    # 业务缺口（work-units）
    business_gap_work: float
    deadline_shortfall_work: float

    # 逐时域解（供审计；功率为规划近似）
    allocation: np.ndarray
    charge_kw: list[float]
    discharge_kw: list[float]
    soc_kwh: list[float]
    residuals_by_constraint: dict[str, float]
    max_constraint_residual: float
    power_approximation_used: bool

    # M6-P1-F3-R1：**只读**审计导出——阶段 A 自身的第 0 步逐组 exec 动作。
    # 用于判定稀疏分配究竟产生于阶段 A 还是阶段 B；**不参与**任何求解语义。
    stage_a_exec_compute_actions: list[float] = field(default_factory=list)
    # 只读：阶段 A 自身解在**第 0 步**给每个任务的分配量（work-units）与其 deadline slack
    stage_a_task_step0_work: list[float] = field(default_factory=list)
    stage_a_deadline_slack: list[float] = field(default_factory=list)
    inventory_audit: dict = field(default_factory=dict)
    execution_source: str = "none"
    candidate_check: dict = field(default_factory=dict)


def _projection_empty(
    snapshot: SystemSnapshot, status: str, failure: str, audit: dict | None = None
) -> RawProjectionResult:
    """失败/timeout 结果：仍保留已知结构审计信息（不得用全 0 掩盖求解规模）。"""
    audit = audit or {}
    n_group = len(snapshot.group_work_capacity)
    H = int(snapshot.planning_horizon_steps)
    return RawProjectionResult(
        backend="mip", solver_status=status, failure_class=failure,
        horizon_steps=H,
        n_variables=int(audit.get("n_variables", 0)),
        n_integer_variables=int(audit.get("n_integer_variables", H)),
        n_constraints=int(audit.get("n_constraints", 0)),
        stage_a_status=str(audit.get("stage_a_status", status)),
        stage_b_status=str(audit.get("stage_b_status", "not_run")),
        stage_a_solve_time_s=float(audit.get("stage_a_solve_time_s", 0.0)),
        stage_b_solve_time_s=float(audit.get("stage_b_solve_time_s", 0.0)),
        stage_a_objective=0.0, stage_b_objective=0.0, projection_offset=0.0,
        diagnostic_status=str(audit.get("diagnostic_status", "not_run")),
        diagnostic_solve_time_s=float(audit.get("diagnostic_solve_time_s", 0.0)),
        exec_compute_actions=[0.0] * n_group, exec_storage_action=0.0,
        business_gap_work=float(sum(t.remaining_work for t in snapshot.tasks)),
        deadline_shortfall_work=0.0,
        allocation=np.zeros((len(snapshot.tasks), n_group, H)),
        charge_kw=[0.0] * H, discharge_kw=[0.0] * H,
        soc_kwh=[float(snapshot.soc_kwh)] * (H + 1),
        residuals_by_constraint={}, max_constraint_residual=0.0,
        power_approximation_used=True,
        inventory_audit=({"target_reachable": None, "band_reachable": None,
                         "target_gap_kwh": None, "band_gap_kwh": None,
                         "predicted_terminal_kwh": None, "status": "unproven",
                         "reachability_solve_time_s": 0.0,
                         **audit.get("inventory_audit", {})}
                        if isinstance(snapshot, InventorySnapshot) else {}),
    )


def _validate_raw(proposal, n_group: int) -> str | None:
    """raw proposal 维度/范围验证；返回失败原因或 None（合法）。合法时不得 clip。"""
    if len(proposal.compute_actions) != n_group:
        return f"compute_actions 维度 {len(proposal.compute_actions)} != 组数 {n_group}"
    if any((x < 0.0 or x > 1.0) for x in proposal.compute_actions):
        return "compute_actions 存在超出 [0,1] 的取值"
    if not (-1.0 <= float(proposal.storage_action) <= 1.0):
        return f"storage_action {proposal.storage_action} 超出 [-1,1]"
    return None


def solve_time_indexed_mip_raw_projection(
    snapshot: SystemSnapshot,
    proposal,
    *,
    time_limit_s: float | None = None,
    business_penalty_sgd_per_work: float = BUSINESS_SHORTFALL_PENALTY_SGD_PER_WORK[0],
    deadline_penalty_sgd_per_work: float = DEADLINE_SHORTFALL_PENALTY_SGD_PER_WORK[0],
    stage_b_tolerance: float = RAW_STAGE_B_TOL,
) -> RawProjectionResult:
    """第 0 步原始动作最小偏移投影（两阶段确定性 MIP）。

    第 0 步映射（N = 组数）：
        u_g          = sum_i A[i,g,0] / group_work_capacity[g]      （capacity<=0 时 u_g=0）
        exec_storage = discharge[0]/discharge_max − charge[0]/charge_max   （>0 放电、<0 充电）

    投影目标（无量纲 L1）：
        offset = (1/N) * Σ_g |u_g − raw_compute_g| + |exec_storage − raw_storage|

    阶段 A：最小化 offset（不混入经济目标）；
    阶段 B：约束 offset <= 阶段A最优 + tolerance，再以规划经济/服务目标（SGD）确定性 tie-break。
    """
    import time

    inventory_enabled = isinstance(snapshot, InventorySnapshot)
    guard = snapshot.service_guard if isinstance(snapshot, InventorySnapshot) else None
    coupled = guard is not None and guard.version in (
        "arrived-service-reserve-v2", "arrived-service-reserve-v3")
    executor_consistent = guard is not None and guard.version == "arrived-service-reserve-v3"
    terminal = snapshot.terminal_inventory if isinstance(snapshot, InventorySnapshot) else None
    if isinstance(snapshot, InventorySnapshot):
        validate_inventory_snapshot(snapshot)
    call_deadline = (_monotonic() + float(time_limit_s)
                     if inventory_enabled and time_limit_s is not None
                     else None)

    H = int(snapshot.planning_horizon_steps)
    n_task = len(snapshot.tasks)
    n_group = len(snapshot.group_work_capacity)

    reason = _validate_raw(proposal, n_group)
    if reason is not None:
        return _projection_empty(snapshot, "invalid_proposal", FAILURE_PROPOSAL_INVALID)
    if H <= 0:
        return _projection_empty(snapshot, "optimal", FAILURE_NONE)

    dt = float(snapshot.delta_t_hours)
    pf = snapshot.planning_forecast
    raw_compute = [float(x) for x in proposal.compute_actions]
    raw_storage = float(proposal.storage_action)
    cap = [float(c) for c in snapshot.group_work_capacity]
    dmax = max(float(snapshot.bess_discharge_power_max_kw), 1e-9)
    cmax = max(float(snapshot.bess_charge_power_max_kw), 1e-9)

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
    off_z = off_dls + n_task
    off_d = off_z + H          # 逐组 |u_g - raw_compute_g|
    off_e = off_d + n_group    # |exec_storage - raw_storage|
    off_inventory_gap = off_e + 1
    off_aggregate_work = off_inventory_gap + (1 if inventory_enabled else 0)
    off_aggregate_backlog = off_aggregate_work + (H if coupled else 0)
    off_prefix = off_aggregate_backlog + (H + 1 if coupled else 0)
    off_aggregate_busy = off_prefix + (n_task if coupled and not executor_consistent else 0)
    off_active = off_aggregate_busy + (H if coupled else 0)
    active_tasks = [i for i, t in enumerate(snapshot.tasks)
                    if executor_consistent and t.remaining_work > 1e-8
                    and t.max_rate_work_per_step > 1e-8]
    active_index = {i: j for j, i in enumerate(active_tasks)}
    off_finished = off_active + len(active_tasks) * H
    off_begun = off_finished + len(active_tasks) * H
    begun_tasks = [i for i in active_tasks if guard is not None
                   and not guard.known_task_interruptible[i] and not guard.known_task_started[i]]
    begun_index = {i: j for j, i in enumerate(begun_tasks)}
    n_vars = off_begun + len(begun_tasks) * H

    def active(i, k):
        return off_active + active_index[i] * H + k

    def finished(i, k):
        return off_finished + active_index[i] * H + k

    def begun(i, k):
        return off_begun + begun_index[i] * H + k

    # 经济/服务目标（阶段 B 的 tie-break，单位 SGD）
    c_econ = np.zeros(n_vars)
    for k in range(H):
        c_econ[off_pgrid + k] = dt * pf.price[k]
        c_econ[off_charge + k] = dt * snapshot.bess_degradation_cost_per_kwh
        c_econ[off_discharge + k] = dt * snapshot.bess_degradation_cost_per_kwh
    for i in range(n_task):
        c_econ[off_bus + i] = business_penalty_sgd_per_work
        c_econ[off_dls + i] = deadline_penalty_sgd_per_work

    # 阶段 A 目标：无量纲 L1 偏移
    c_off = np.zeros(n_vars)
    for g in range(n_group):
        c_off[off_d + g] = 1.0 / max(n_group, 1)
    c_off[off_e] = 1.0

    lb = np.full(n_vars, 0.0)
    ub = np.full(n_vars, np.inf)
    lb[off_pidc:off_pidc + H] = -np.inf
    ub[off_pgrid:off_pgrid + H] = max(float(snapshot.access_limit_kw), 0.0)
    for k in range(H):
        ub[off_pv + k] = max(float(pf.pv[k]), 0.0)
        ub[off_wind + k] = max(float(pf.wind[k]), 0.0)
        ub[off_charge + k] = float(snapshot.bess_charge_power_max_kw)
        ub[off_discharge + k] = float(snapshot.bess_discharge_power_max_kw)
    lb[off_soc:off_soc + H + 1] = float(snapshot.soc_min_kwh)
    ub[off_soc:off_soc + H + 1] = float(snapshot.soc_max_kwh)
    lb[off_z:off_z + H] = 0.0
    ub[off_z:off_z + H] = 1.0
    if coupled and guard is not None:
        ub[off_aggregate_work:off_aggregate_work + H] = sum(cap)
        ub[off_aggregate_work] = 0.
        ub[off_aggregate_backlog:off_aggregate_backlog + H + 1] = sum(
            guard.aggregate_arrival_work)
        ub[off_aggregate_backlog] = 0.
        ub[off_aggregate_backlog + H] = guard.aggregate_backlog_work[-1]
        known_rate_upper = sum(min(task.remaining_work, task.max_rate_work_per_step)
                               for task in snapshot.tasks)
        backlog_upper = 0.
        for k in range(H):
            available_work_upper = backlog_upper + guard.aggregate_arrival_work[k]
            ub[off_aggregate_work + k] = min(sum(cap), available_work_upper)
            backlog_upper = max(available_work_upper - max(sum(cap) - known_rate_upper, 0.), 0.)
            ub[off_aggregate_backlog + k + 1] = min(
                ub[off_aggregate_backlog + k + 1], backlog_upper)
            ub[off_aggregate_busy + k] = int(backlog_upper > 1e-8)
        for i, task in enumerate(snapshot.tasks):
            if not executor_consistent:
                ub[off_prefix + i] = int(
                    min(task.remaining_work, task.max_rate_work_per_step) > 1e-8)
            ub[off_bus + i] = max(
                task.remaining_work - guard.known_service_required_end_work[i], 0.)
            ub[off_dls + i] = max(
                task.remaining_work - guard.known_service_required_due_work[i], 0.)

    if executor_consistent and guard is not None:
        ub[off_active:n_vars] = 1.
        for i in active_tasks:
            task = snapshot.tasks[i]
            due = max(0, min(H, task.deadline - snapshot.step))
            for k in range(H):
                if (k + 1) * task.max_rate_work_per_step < task.remaining_work - 1e-8:
                    ub[finished(i, k)] = 0.
                if guard.known_service_required_due_work[i] >= task.remaining_work - 1e-8:
                    if k >= due - 1:
                        lb[finished(i, k)] = 1.
                    if k >= due:
                        ub[active(i, k)] = 0.
                if (k == H - 1 and guard.known_service_required_end_work[i]
                        >= task.remaining_work - 1e-8):
                    lb[finished(i, k)] = 1.

    if executor_consistent and guard is not None:
        for i, task in enumerate(snapshot.tasks):
            due = max(0, min(H, task.deadline - snapshot.step))
            fully_due = guard.known_service_required_due_work[i] >= task.remaining_work - 1e-8
            for k in range(H):
                for g in range(n_group):
                    ub[a(i, g, k)] = min(cap[g], task.remaining_work,
                                         task.max_rate_work_per_step)
                    if fully_due and k >= due:
                        ub[a(i, g, k)] = 0.
                if i in begun_index and fully_due:
                    duration = int(np.ceil(task.remaining_work / task.max_rate_work_per_step))
                    if k >= due - duration:
                        lb[begun(i, k)] = 1.

    rows: list[dict[int, float]] = []
    lbs: list[float] = []
    ubs: list[float] = []
    names: list[str] = []

    def add(row: dict[int, float], lo: float, hi: float, name: str) -> None:
        rows.append(row)
        lbs.append(lo)
        ubs.append(hi)
        names.append(name)

    _add_core_constraints(
        add, snapshot, pf, H=H, n_task=n_task, n_group=n_group, dt=dt, a=a,
        off={
            "pidc": off_pidc, "pgrid": off_pgrid, "pv": off_pv, "wind": off_wind,
            "curtail": off_curtail, "charge": off_charge, "discharge": off_discharge,
            "soc": off_soc, "bus": off_bus, "dls": off_dls, "z": off_z,
        },
        mutual_exclusion=True,
    )

    if inventory_enabled and terminal is not None:
        if guard is not None:
            if not coupled:
                for k, limit in enumerate(guard.charge_limits_kw):
                    ub[off_charge + k] = min(ub[off_charge + k], limit)
            else:
                for k in range(H):
                    coefficient = guard.reserve_power_coefficients_kw_per_work[k]
                    aggregate_coefficient = max(coefficient)
                    # Forecast work is a reservation only: conserve its queue,
                    # share physical compute capacity, retain unavoidable backlog.
                    add({off_aggregate_backlog + k + 1: 1.,
                         off_aggregate_backlog + k: -1., off_aggregate_work + k: 1.},
                        guard.aggregate_arrival_work[k], guard.aggregate_arrival_work[k],
                        f"aggregate_backlog_balance[{k}]")
                    capacity_row = {off_aggregate_work + k: 1.}
                    reserve_row = {off_charge + k: 1.,
                                   off_aggregate_work + k: aggregate_coefficient}
                    for g in range(n_group):
                        for i in range(n_task):
                            capacity_row[a(i, g, k)] = 1.
                            reserve_row[a(i, g, k)] = coefficient[g]
                    add(capacity_row, -np.inf, sum(cap), f"aggregate_shared_capacity[{k}]")
                    add({off_aggregate_backlog + k + 1: 1.,
                         off_aggregate_busy + k: -sum(guard.aggregate_arrival_work)},
                        -np.inf, 0., f"aggregate_backlog_nonempty[{k}]")
                    greedy_row = {**capacity_row, off_aggregate_busy + k: -sum(cap)}
                    add(greedy_row, 0., np.inf, f"aggregate_earliest_processing[{k}]")
                    headroom = max(snapshot.access_limit_kw - guard.reserve_base_power_kw[k], 0.)
                    big_m = cmax + sum(c * w for c, w in zip(coefficient, cap, strict=True))
                    big_m += aggregate_coefficient * sum(cap)
                    # Only charging invokes the zero-renewable reserve. Storage
                    # exclusion and the original physical constraints stay intact.
                    reserve_row[off_z + k] = big_m
                    add(reserve_row, -np.inf, headroom + big_m,
                        f"schedule_coupled_charging_reserve[{k}]")
                if not executor_consistent:
                    previous = None
                    for i in guard.known_service_order:
                        demand = min(snapshot.tasks[i].remaining_work,
                                     snapshot.tasks[i].max_rate_work_per_step)
                        if demand <= 1e-8:
                            continue
                        row = {a(i, g, 0): 1. for g in range(n_group)}
                        row[off_prefix + i] = -demand
                        add(row, -np.inf, 0., f"executor_prefix_active[{i}]")
                        if previous is not None:
                            j, preceding = previous
                            row = {a(j, g, 0): 1. for g in range(n_group)}
                            row[off_prefix + i] = -preceding
                            add(row, 0., np.inf, f"executor_prefix_preceding[{i}]")
                        previous = (i, demand)
            if coupled and guard is not None:
                add({a(i, g, 0): 1. for i in range(n_task) for g in range(n_group)},
                    sum(guard.group_work_floor), np.inf, "arrived_service_prefix_floor")
            else:
                for g, work in enumerate(guard.group_work_floor):
                    add({a(i, g, 0): 1.0 for i in range(n_task)}, work, np.inf,
                        f"arrived_service_floor[{g}]")
        add({off_inventory_gap: 1.0, off_soc + H: -1.0},
            -terminal.target_kwh, np.inf, "terminal_gap_below")
        add({off_inventory_gap: 1.0, off_soc + H: 1.0},
            terminal.target_kwh, np.inf, "terminal_gap_above")

    if executor_consistent and guard is not None:
        for k in range(H):
            preceding_tasks: list[int] = []
            for i in guard.known_service_order:
                if i not in active_index:
                    continue
                task = snapshot.tasks[i]
                rate, work = task.max_rate_work_per_step, task.remaining_work
                row = {a(i, g, k): 1. for g in range(n_group)}
                add({**row, active(i, k): -rate}, -np.inf, 0.,
                    f"known_executor_active[{i},{k}]")
                cumulative = {a(i, g, t): 1. for g in range(n_group) for t in range(k + 1)}
                add({**cumulative, finished(i, k): -work}, 0., np.inf,
                    f"known_executor_finished[{i},{k}]")
                due = max(0, min(H, task.deadline - snapshot.step))
                minimum_progress = max(guard.known_service_required_end_work[i]
                                       - rate * (H - k - 1), 0.)
                if k < due:
                    minimum_progress = max(minimum_progress,
                                           guard.known_service_required_due_work[i]
                                           - rate * (due - k - 1))
                add(cumulative, minimum_progress, min(work, (k + 1) * rate),
                    f"known_executor_derived_progress_bounds[{i},{k}]")
                if k:
                    add({finished(i, k): 1., finished(i, k - 1): -1.}, 0., np.inf,
                        f"known_executor_finished_monotone[{i},{k}]")
                for j in preceding_tasks:
                    preceding_rate = snapshot.tasks[j].max_rate_work_per_step
                    add({**{a(j, g, k): 1. for g in range(n_group)},
                         active(i, k): -preceding_rate, finished(j, k): preceding_rate},
                        0., np.inf, f"known_executor_priority[{j},{i},{k}]")
                preceding_tasks.append(i)
                if not guard.known_task_interruptible[i]:
                    if guard.known_task_started[i]:
                        demand = min(rate, max(work - k * rate, 0.))
                        add(row, demand, demand, f"known_executor_running[{i},{k}]")
                        lb[active(i, k)] = ub[active(i, k)] = int(demand > 1e-8)
                        lb[finished(i, k)] = ub[finished(i, k)] = int(
                            work <= (k + 1) * rate + 1e-8)
                    else:
                        prior = {begun(i, k - 1): -1.} if k else {}
                        add({begun(i, k): 1., **prior}, 0., np.inf,
                            f"known_executor_begun_monotone[{i},{k}]")
                        add({begun(i, k): 1., active(i, k): -1.}, 0., np.inf,
                            f"known_executor_begun_active[{i},{k}]")
                        add({begun(i, k): 1., active(i, k): -1., **prior}, -np.inf, 0.,
                            f"known_executor_begun_only_if_active[{i},{k}]")
                        continuity = {**row, finished(i, k): rate}
                        if k:
                            continuity[begun(i, k - 1)] = -rate
                        add(continuity, 0., np.inf, f"known_executor_continuity[{i},{k}]")

    # 投影绝对值线性化：d_g >= |u_g - raw_g|（cap<=0 时 u_g=0）
    for g in range(n_group):
        row_ge = {off_d + g: 1.0}
        row_le = {off_d + g: 1.0}
        if cap[g] > 0.0:
            inv = 1.0 / cap[g]
            for i in range(n_task):
                row_ge[a(i, g, 0)] = -inv
                row_le[a(i, g, 0)] = inv
        add(row_ge, -raw_compute[g], np.inf, f"proj_ge[{g}]")
        add(row_le, raw_compute[g], np.inf, f"proj_le[{g}]")

    # e >= |exec_storage - raw_storage|
    add(
        {off_e: 1.0, off_discharge: -1.0 / dmax, off_charge: 1.0 / cmax},
        -raw_storage, np.inf, "proj_ge_storage",
    )
    add(
        {off_e: 1.0, off_discharge: 1.0 / dmax, off_charge: -1.0 / cmax},
        raw_storage, np.inf, "proj_le_storage",
    )

    if inventory_enabled:
        A_csr = _sparse_rows(rows, n_vars)
    else:
        row_mat = lil_matrix((len(rows), n_vars))
        for r, row in enumerate(rows):
            for col, val in row.items():
                row_mat[r, col] = val
        A_csr = csr_matrix(row_mat)
    bounds_vec = Bounds(lb=lb, ub=ub)
    integrality = np.zeros(n_vars)
    integrality[off_z:off_z + H] = 1
    if coupled and guard is not None:
        if not executor_consistent:
            integrality[off_prefix:off_prefix + n_task] = 1
        integrality[off_active:n_vars] = 1
        integrality[off_aggregate_busy:off_aggregate_busy + H] = 1
    # 全局 deadline：阶段 A 与 B 共享同一次调用的总预算
    deadline = (call_deadline if call_deadline is not None else
                None if time_limit_s is None else _monotonic() + float(time_limit_s))

    def _remaining() -> float | None:
        if deadline is None:
            return None
        return max(deadline - _monotonic(), 0.0)

    def _options() -> dict:
        # M5.4g：阶段 A/B 也必须走同一份确定性选项构造。
        # **只替换 options 构造**：time_limit 仍来自上面这个共享 deadline 的剩余预算，
        # remaining-budget 算法未变；不触碰目标、约束、边界。
        options = deterministic_mip_options(time_limit_s=_remaining())
        if inventory_enabled:
            # Recorded seed1 input: A at 1e-8 returned a near-integer witness
            # whose offset bound made B report infeasible. Tightening only B
            # did not help; both stages at 1e-9 finish with integer residual 0.
            # Keep all physical/offset bounds and the shared deadline unchanged.
            tolerance = INVENTORY_SOLVER_FEASIBILITY_TOLERANCE
            options.update(mip_feasibility_tolerance=tolerance,
                           primal_feasibility_tolerance=tolerance,
                           dual_feasibility_tolerance=tolerance)
        return options

    offset_row = {**{off_d + g: 1.0 / max(n_group, 1) for g in range(n_group)}, off_e: 1.0}
    _audit: dict = {
        "n_variables": n_vars, "n_integer_variables": int(np.count_nonzero(integrality)),
        "n_constraints": len(rows) + 1,
    }

    def _status(res) -> str:
        code = int(getattr(res, "status", 4))
        return (
            SOLVER_OPTIMAL if code == 0
            else SOLVER_TIME_LIMIT if code == 1
            else SOLVER_INFEASIBLE if code == 2
            else SOLVER_FAILURE
        )

    # --- 阶段 A：最小化原始动作偏移 ---
    from scipy.optimize import milp as _milp

    inventory_audit: dict = {}
    if inventory_enabled and terminal is not None:
        rem = _remaining()
        if rem is not None and rem <= 0:
            return _projection_empty(snapshot, SOLVER_TIME_LIMIT, FAILURE_TIMEOUT, {
                **_audit, "stage_a_status": "not_run",
                "inventory_audit": {"timeout_phase": "model_assembly"},
            })
        c_inventory = np.zeros(n_vars)
        c_inventory[off_inventory_gap] = 1.0
        t_inventory = time.perf_counter()
        certificate = (_base_only_terminal_certificate(snapshot)
                       if isinstance(snapshot, InventorySnapshot) else None)
        certified_x = None
        if certificate is not None:
            witness = np.zeros(n_vars)
            if guard is not None:
                for i in range(n_task):
                    for g in range(n_group):
                        if coupled and guard is not None:
                            for k in range(H):
                                witness[a(i, g, k)] = guard.known_service_allocation[k][i][g]
                        else:
                            witness[a(i, g, 0)] = guard.current_allocation[i][g]
            if coupled and guard is not None:
                witness[off_aggregate_work:off_aggregate_work + H] = guard.aggregate_service_work
                witness[off_aggregate_backlog:off_aggregate_backlog + H + 1] = (
                    guard.aggregate_backlog_work)
                for k in range(H):
                    witness[off_aggregate_busy + k] = int(
                        guard.aggregate_backlog_work[k + 1] > 1e-8)
                for i in ([] if executor_consistent else range(n_task)):
                    witness[off_prefix + i] = int(sum(
                        guard.known_service_allocation[0][i]) > 1e-8)
            if executor_consistent and guard is not None:
                for i in active_tasks:
                    served_total = 0.
                    started = False
                    for k in range(H):
                        amount = sum(guard.known_service_allocation[k][i])
                        served_total += amount
                        started = started or amount > 1e-8
                        witness[active(i, k)] = int(amount > 1e-8)
                        witness[finished(i, k)] = int(
                            served_total >= snapshot.tasks[i].remaining_work - 1e-8)
                        if i in begun_index:
                            witness[begun(i, k)] = int(started)
            witness[off_soc:off_soc + H + 1] = certificate["energy_kwh"]
            witness[off_charge:off_charge + H] = certificate["charge_kw"]
            witness[off_discharge:off_discharge + H] = certificate["discharge_kw"]
            for k in range(H):
                task_power = sum(snapshot.group_power_coeff_kw_per_work[g]
                                 * sum(witness[a(i, g, k)] for i in range(n_task))
                                 for g in range(n_group))
                idc_power = pf.base_idc_power[k] + task_power
                demand = (idc_power + certificate["charge_kw"][k]
                          - certificate["discharge_kw"][k])
                pv_used = min(pf.pv[k], max(demand, 0.))
                wind_used = min(pf.wind[k], max(demand - pv_used, 0.))
                witness[off_pidc + k] = idc_power
                witness[off_pgrid + k] = demand - pv_used - wind_used
                witness[off_pv + k], witness[off_wind + k] = pv_used, wind_used
                witness[off_curtail + k] = pf.pv[k] + pf.wind[k] - pv_used - wind_used
                witness[off_z + k] = int(certificate["charge_kw"][k] > 0)
            for i, task in enumerate(snapshot.tasks):
                witness[off_bus + i] = task.remaining_work - sum(
                    witness[a(i, g, k)] for g in range(n_group) for k in range(H))
                due = max(0, min(H, task.deadline - snapshot.step))
                witness[off_dls + i] = max(task.remaining_work - sum(
                    witness[a(i, g, k)] for g in range(n_group) for k in range(due)), 0.)
            witness[off_d:off_d + n_group] = [
                abs(sum(witness[a(i, g, 0)] for i in range(n_task)) / cap[g]
                    - raw_compute[g]) if cap[g] > 0 else abs(raw_compute[g])
                for g in range(n_group)]
            witness[off_e] = abs(witness[off_discharge] / dmax
                                     - witness[off_charge] / cmax - raw_storage)
            lhs = A_csr @ witness
            # Check ALL actual constraints/bounds, rather than trusting a relaxed
            # interval model. 1e-8 covers floating arithmetic, below physics 1e-6.
            if (np.all(lhs >= np.array(lbs) - 1e-8)
                    and np.all(lhs <= np.array(ubs) + 1e-8)
                    and np.all(witness >= lb - 1e-8)
                    and np.all(witness <= ub + 1e-8)):
                certified_x = witness
        reachability_method = "mip"
        if certified_x is not None:
            from types import SimpleNamespace
            # Feasible objective 0 + global nonnegative gap bound proves optimality.
            res_inventory = SimpleNamespace(status=0, x=certified_x)
            reachability_method = ("complete_service_zero_gap_certificate" if coupled else
                                   "arrived_service_zero_gap_certificate" if guard is not None
                                   else "base_only_zero_gap_certificate")
        else:
            rem = _remaining()
            if rem is not None and rem <= 0:
                return _projection_empty(snapshot, SOLVER_TIME_LIMIT, FAILURE_TIMEOUT, {
                    **_audit, "stage_a_status": "not_run",
                    "inventory_audit": {"timeout_phase": "inventory_certificate"},
                })
            res_inventory = _milp(
                c=c_inventory,
                constraints=[LinearConstraint(A_csr, np.array(lbs), np.array(ubs))],
                integrality=integrality, bounds=bounds_vec, options=_options(),
            )
        elapsed_inventory = time.perf_counter() - t_inventory
        inventory_status = _status(res_inventory)
        if inventory_status != SOLVER_OPTIMAL:
            failure = (FAILURE_TIMEOUT if inventory_status == SOLVER_TIME_LIMIT
                       else FAILURE_SOLVER_FAILURE)
            result = _projection_empty(snapshot, inventory_status, failure, {
                **_audit, "stage_a_status": "not_run",
            })
            result.inventory_audit.update(
                status="unproven", reachability_solve_time_s=elapsed_inventory,
                **({"timeout_phase": "inventory_reachability"}
                   if inventory_status == SOLVER_TIME_LIMIT else {}))
            return result
        minimum_gap = max(float(res_inventory.x[off_inventory_gap]), 0.0)
        # Preserve the measured gap, including sub-micro-kWh positive deficits.
        # Rounding it to zero can make A infeasible at an attained capacity bound.
        # Bind the terminal energy directly; constraining a tiny auxiliary gap
        # upper bound causes an observed HiGHS presolve infeasibility at t=43.
        bounds_vec.lb[off_soc + H] = max(
            float(snapshot.soc_min_kwh), terminal.target_kwh - minimum_gap
            - (1e-7 if minimum_gap > 0 else 0))
        bounds_vec.ub[off_soc + H] = min(
            float(snapshot.soc_max_kwh), terminal.target_kwh + minimum_gap
            + (1e-7 if minimum_gap > 0 else 0))
        attainable = float(res_inventory.x[off_soc + H])
        band_gap = max(terminal.lower_kwh - attainable, attainable - terminal.upper_kwh, 0.0)
        inventory_audit = {
            "version": terminal.version, "episode_end_step": terminal.episode_end_step,
            "remaining_steps": terminal.remaining_steps, "target_kwh": terminal.target_kwh,
            "lower_kwh": terminal.lower_kwh, "upper_kwh": terminal.upper_kwh,
            "target_reachable": bool(minimum_gap <= 1e-6),
            "band_reachable": bool(band_gap <= 1e-6),
            "target_gap_kwh": minimum_gap, "band_gap_kwh": float(band_gap),
            "reachability_solve_time_s": elapsed_inventory,
            "reachability_method": reachability_method,
            "reachability_scope": "registered planning assumptions; not realized physical proof",
            "physical_unreachability_proven": False,
            "service_guard": (guard.model_dump()
                              if guard is not None else None),
            "known_service_compatible": (guard.known_service_shortfall_work <= 1e-6
                                         if coupled and guard is not None else None),
            "inventory_numerical_allowance_kwh": 1e-7,
            "status": "target_reachable" if minimum_gap <= 1e-6 else "target_unreachable",
        }
        _audit["inventory_audit"] = inventory_audit
        rem = _remaining()
        if rem is not None and rem <= 0:
            result = _projection_empty(snapshot, SOLVER_TIME_LIMIT, FAILURE_TIMEOUT, {
                **_audit, "stage_a_status": "not_run",
                "inventory_audit": {**inventory_audit, "timeout_phase": "before_stage_a"},
            })
            return result

    tA0 = time.perf_counter()
    res_a = _milp(
        c=c_off, constraints=[LinearConstraint(A_csr, np.array(lbs), np.array(ubs))],
        integrality=integrality, bounds=bounds_vec, options=_options(),
    )
    tA = time.perf_counter() - tA0
    status_a = _status(res_a)
    if status_a != SOLVER_OPTIMAL:
        diag_audit = {"diagnostic_status": "not_run", "diagnostic_solve_time_s": 0.0}
        if status_a == SOLVER_TIME_LIMIT:
            failure = FAILURE_TIMEOUT  # timeout 路径不运行 base-only 诊断
        elif status_a == SOLVER_INFEASIBLE:
            rem = _remaining()
            if rem is not None and rem <= 0.0:
                # 预算耗尽 → 不启动诊断，直接 timeout
                diag_audit["diagnostic_status"] = "not_started"
                return _projection_empty(
                    snapshot, SOLVER_TIME_LIMIT, FAILURE_TIMEOUT,
                    {**_audit, "stage_a_status": status_a, "stage_a_solve_time_s": tA,
                     "stage_b_status": "not_run", **diag_audit},
                )
            base_diag = diagnose_base_feasibility(snapshot, time_limit_s=rem)
            diag_audit = {
                "diagnostic_status": base_diag.status,
                "diagnostic_solve_time_s": base_diag.solve_time_s,
            }
            if "time_limit" in base_diag.status or "not_started" in base_diag.status:
                # 诊断自身 timeout → 整体 timeout，不得误报 base_shortage
                return _projection_empty(
                    snapshot, SOLVER_TIME_LIMIT, FAILURE_TIMEOUT,
                    {**_audit, "stage_a_status": status_a, "stage_a_solve_time_s": tA,
                     "stage_b_status": "not_run", **diag_audit},
                )
            failure = (
                FAILURE_BASE_SHORTAGE if not base_diag.feasible else FAILURE_SOLVER_FAILURE
            )
        else:
            failure = FAILURE_SOLVER_FAILURE
        return _projection_empty(
            snapshot, status_a, failure,
            {**_audit, "stage_a_status": status_a, "stage_a_solve_time_s": tA, **diag_audit},
        )

    initial_check = check_projection_candidate(
        res_a.x, lb, ub, integrality, A_csr, np.array(lbs), np.array(ubs))
    if not initial_check["passed"]:
        rejected = _projection_empty(
            snapshot, SOLVER_FAILURE, FAILURE_SOLVER_FAILURE,
            {**_audit, "stage_a_status": status_a, "stage_a_solve_time_s": tA})
        rejected.candidate_check = initial_check
        return rejected

    offset_a = float(c_off @ res_a.x)

    # 只读审计：阶段 A 自身解的逐组 u（与最终 exec 公式逐字相同）
    def _exec_compute_from(vec) -> list[float]:
        out: list[float] = []
        for g in range(n_group):
            used = sum(max(float(vec[a(i, g, 0)]), 0.0) for i in range(n_task))
            out.append(float(used / cap[g]) if cap[g] > 0.0 else 0.0)
        return out

    stage_a_exec_compute = _exec_compute_from(res_a.x)
    stage_a_task_step0_work = [
        float(sum(max(float(res_a.x[a(i, g, 0)]), 0.0) for g in range(n_group)))
        for i in range(n_task)
    ]
    stage_a_deadline_slack = [float(res_a.x[off_dls + i]) for i in range(n_task)]
    if inventory_audit:
        inventory_audit["stage_a_terminal_kwh"] = float(res_a.x[off_soc + H])

    # B only adds the primary-offset bound; A itself satisfies that bound.
    rows_b = list(rows) + [offset_row]
    lbs_b = list(lbs) + [-np.inf]
    ubs_b = list(ubs) + [offset_a + stage_b_tolerance]
    A_b = _append_sparse_row(A_csr, offset_row)
    check_a = check_projection_candidate(
        res_a.x, lb, ub, integrality, A_b, np.array(lbs_b), np.array(ubs_b))
    if not check_a["passed"]:
        rejected = _projection_empty(snapshot, SOLVER_FAILURE, FAILURE_SOLVER_FAILURE,
                                     {**_audit, "stage_a_status": status_a})
        rejected.candidate_check = check_a
        return rejected

    rem_b = _remaining()
    tB = 0.0
    if rem_b is not None and rem_b <= 0.0:
        status_b = "not_run"
        x = res_a.x
        execution_source = "stage_a"
    else:
        tB0 = time.perf_counter()
        res_b = _milp(
            c=c_econ, constraints=[LinearConstraint(A_b, np.array(lbs_b), np.array(ubs_b))],
            integrality=integrality, bounds=bounds_vec, options=_options())
        tB = time.perf_counter() - tB0
        status_b = _status(res_b)
        if status_b == SOLVER_OPTIMAL:
            x = res_b.x
            execution_source = "stage_b"
        elif status_b == SOLVER_TIME_LIMIT:
            x = res_a.x
            execution_source = "stage_a"
        else:
            return _projection_empty(
                snapshot, status_b, FAILURE_SOLVER_FAILURE,
                {**_audit, "stage_a_status": status_a, "stage_b_status": status_b,
                 "stage_a_solve_time_s": tA, "stage_b_solve_time_s": tB})
    candidate_check = check_projection_candidate(
        x, lb, ub, integrality, A_b, np.array(lbs_b), np.array(ubs_b))
    if not candidate_check["passed"]:
        rejected = _projection_empty(snapshot, SOLVER_FAILURE, FAILURE_SOLVER_FAILURE,
                                     {**_audit, "stage_a_status": status_a,
                                      "stage_b_status": status_b,
                                      "stage_a_solve_time_s": tA, "stage_b_solve_time_s": tB})
        rejected.candidate_check = candidate_check
        return rejected

    allocation = np.zeros((n_task, n_group, H))
    for i in range(n_task):
        for g in range(n_group):
            for k in range(H):
                allocation[i, g, k] = max(x[a(i, g, k)], 0.0)

    exec_compute = [0.0] * n_group
    for g in range(n_group):
        used = sum(allocation[i, g, 0] for i in range(n_task))
        exec_compute[g] = float(used / cap[g]) if cap[g] > 0.0 else 0.0
    exec_storage = float(x[off_discharge] / dmax - x[off_charge] / cmax)
    exec_storage = float(np.clip(exec_storage, -1.0, 1.0))

    offset_val = float(
        sum(abs(exec_compute[g] - raw_compute[g]) for g in range(n_group)) / max(n_group, 1)
        + abs(exec_storage - raw_storage)
    )

    charge = [float(x[off_charge + k]) for k in range(H)]
    discharge = [float(x[off_discharge + k]) for k in range(H)]
    business_slack = [float(x[off_bus + i]) for i in range(n_task)]
    deadline_slack = [float(x[off_dls + i]) for i in range(n_task)]

    ax = np.asarray(A_b.dot(x)).ravel()
    residuals = {name: float(max(lbs_b[r] - ax[r], ax[r] - ubs_b[r], 0.))
                 for r, name in enumerate([*names, "projection_offset_bound"])}
    max_residual = max(residuals.values(), default=0.)

    total_business = float(sum(business_slack))
    total_deadline = float(sum(deadline_slack))
    failure = FAILURE_DEADLINE_SHORTFALL if total_deadline > _TOL else FAILURE_NONE

    if coupled and guard is not None:
        planned_reserve_power, planned_limits = [], []
        for k in range(H):
            power_coefficients = np.asarray(guard.reserve_power_coefficients_kw_per_work[k])
            power = (guard.reserve_base_power_kw[k]
                     + float(power_coefficients @ allocation[:, :, k].sum(axis=0))
                     + float(power_coefficients.max()) * float(x[off_aggregate_work + k]))
            planned_reserve_power.append(power)
            planned_limits.append(min(cmax, max(snapshot.access_limit_kw - power, 0.)))
        inventory_audit.update(
            planned_reserved_power_kw=planned_reserve_power,
            planned_charge_limits_kw=planned_limits,
            aggregate_service_work=x[off_aggregate_work:off_aggregate_work + H].tolist(),
            aggregate_backlog_work=x[off_aggregate_backlog:off_aggregate_backlog + H + 1].tolist())
    if inventory_audit:
        inventory_audit.update(
            planned_next_energy_kwh=float(x[off_soc + 1]),
            planned_step0_task_work={str(task.task_id): float(allocation[i, :, 0].sum())
                                     for i, task in enumerate(snapshot.tasks)})

    if execution_source == "stage_a":
        failure = "stage_b_timeout_feasible"
    return RawProjectionResult(
        backend="mip", solver_status=(SOLVER_OPTIMAL if execution_source == "stage_b"
                                      else SOLVER_TIME_LIMIT), failure_class=failure,
        execution_source=execution_source, candidate_check=candidate_check,
        horizon_steps=H, n_variables=n_vars,
        n_integer_variables=int(np.count_nonzero(integrality)), n_constraints=len(rows_b),
        stage_a_status=status_a, stage_b_status=status_b,
        stage_a_solve_time_s=tA, stage_b_solve_time_s=tB,
        stage_a_objective=offset_a, stage_b_objective=float(c_econ @ x),
        projection_offset=offset_val,
        diagnostic_status="not_required_optimal", diagnostic_solve_time_s=0.0,
        exec_compute_actions=exec_compute, exec_storage_action=exec_storage,
        business_gap_work=total_business, deadline_shortfall_work=total_deadline,
        allocation=allocation, charge_kw=charge, discharge_kw=discharge,
        soc_kwh=[float(x[off_soc + k]) for k in range(H + 1)],
        residuals_by_constraint=residuals, max_constraint_residual=max_residual,
        power_approximation_used=True,
        stage_a_exec_compute_actions=stage_a_exec_compute,
        stage_a_task_step0_work=stage_a_task_step0_work,
        stage_a_deadline_slack=stage_a_deadline_slack,
        inventory_audit={**inventory_audit, "predicted_terminal_kwh": float(x[off_soc + H])}
        if inventory_audit else {},
    )

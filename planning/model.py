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
    n_vars = off_e + 1

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

    A_csr = csr_matrix(lil_matrix((len(rows), n_vars)))
    row_mat = lil_matrix((len(rows), n_vars))
    for r, row in enumerate(rows):
        for col, val in row.items():
            row_mat[r, col] = val
    A_csr = csr_matrix(row_mat)
    bounds_vec = Bounds(lb=lb, ub=ub)
    integrality = np.zeros(n_vars)
    integrality[off_z:off_z + H] = 1
    # 全局 deadline：阶段 A 与 B 共享同一次调用的总预算
    deadline = None if time_limit_s is None else _monotonic() + float(time_limit_s)

    def _remaining() -> float | None:
        if deadline is None:
            return None
        return max(deadline - _monotonic(), 0.0)

    def _options() -> dict:
        # M5.4g：阶段 A/B 也必须走同一份确定性选项构造。
        # **只替换 options 构造**：time_limit 仍来自上面这个共享 deadline 的剩余预算，
        # remaining-budget 算法未变；不触碰目标、约束、边界。
        return deterministic_mip_options(time_limit_s=_remaining())

    offset_row = {**{off_d + g: 1.0 / max(n_group, 1) for g in range(n_group)}, off_e: 1.0}
    _audit = {
        "n_variables": n_vars, "n_integer_variables": H, "n_constraints": len(rows) + 1,
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

    # 阶段 A 后预算已耗尽 → 不启动阶段 B，直接 timeout
    rem_b = _remaining()
    if rem_b is not None and rem_b <= 0.0:
        return _projection_empty(
            snapshot, SOLVER_TIME_LIMIT, FAILURE_TIMEOUT,
            {**_audit, "stage_a_status": status_a, "stage_a_solve_time_s": tA,
             "stage_b_status": "not_run"},
        )

    # --- 阶段 B：固定偏移上界，再以经济/服务目标 tie-break ---
    rows_b = list(rows) + [offset_row]
    lbs_b = list(lbs) + [-np.inf]
    ubs_b = list(ubs) + [offset_a + stage_b_tolerance]
    row_mat_b = lil_matrix((len(rows_b), n_vars))
    for r, row in enumerate(rows_b):
        for col, val in row.items():
            row_mat_b[r, col] = val
    A_b = csr_matrix(row_mat_b)

    tB0 = time.perf_counter()
    res_b = _milp(
        c=c_econ, constraints=[LinearConstraint(A_b, np.array(lbs_b), np.array(ubs_b))],
        integrality=integrality, bounds=bounds_vec, options=_options(),
    )
    tB = time.perf_counter() - tB0
    status_b = _status(res_b)
    if status_b != SOLVER_OPTIMAL:
        # 保真分类：stage-B time_limit 不得被改写为 solver_failure
        failure = (
            FAILURE_TIMEOUT if status_b == SOLVER_TIME_LIMIT else FAILURE_SOLVER_FAILURE
        )
        return _projection_empty(
            snapshot, status_b, failure,
            {**_audit, "stage_a_status": status_a, "stage_b_status": status_b,
             "stage_a_solve_time_s": tA, "stage_b_solve_time_s": tB},
        )

    x = res_b.x
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

    total_business = float(sum(business_slack))
    total_deadline = float(sum(deadline_slack))
    failure = FAILURE_DEADLINE_SHORTFALL if total_deadline > _TOL else FAILURE_NONE

    return RawProjectionResult(
        backend="mip", solver_status=SOLVER_OPTIMAL, failure_class=failure,
        horizon_steps=H, n_variables=n_vars, n_integer_variables=H, n_constraints=len(rows_b),
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
    )

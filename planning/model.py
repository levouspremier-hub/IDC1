"""M4.4 — rolling-window dispatch MILP model.

Decision variables (window of ``horizon`` steps):
  * x[t, i] : task i execution rate (work/hour), rate-limited + deadline completion.
  * g[t, j] : group j executed work (work/hour), capacity-limited.
  * c[t], d[t] : battery charge / discharge power (kW), mutually exclusive.
  * z[t] : binary mutex selector (charge=1 / discharge=0).
  * soc[t] : battery state of charge.
  * grid[t], pv[t] : grid purchase and PV usage (no sell-back).

When ``raw_compute`` is provided, auxiliary deviation variables u_j / u_c / u_d
encode the minimal-modification objective: |g[0,j] - raw_work[j]| and
|c[0] - raw_charge| + |d[0] - raw_discharge|, weighted by ``w_dev`` so that
deviation dominates economic cost (the corrector does not re-optimize economics).

The LP relaxation (M4.5) is obtained by dropping integrality on z; it is NOT
produced by a single net-power variable, so mutex is only guaranteed in MIP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint


@dataclass
class TaskPlan:
    task_id: int
    remaining_work: float
    max_rate: float
    start: int  # first schedulable step (window-relative, inclusive)
    deadline: int  # last schedulable step (window-relative, inclusive)


@dataclass
class PlanningProblem:
    horizon: int
    tasks: list[TaskPlan]
    group_cap: np.ndarray  # (G,) work per hour
    soc0: float
    soc_min: float
    soc_max: float
    cap_kwh: float
    charge_max_kw: float
    discharge_max_kw: float
    charge_eff: float
    discharge_eff: float
    dt: float
    pv_available: np.ndarray  # (T,) kW
    access_limit_kw: float
    p_base_kw: float  # idle facility power
    p_slope_kw_per_work: float  # linear power slope per unit executed work
    price: np.ndarray  # (T,) per kWh
    degradation_per_kwh: float
    raw_compute: np.ndarray | None = None  # (G,) raw per-group compute fraction [0,1]
    raw_storage: float = 0.0
    w_dev: float = 1e6  # minimal-modification weight (normalized, dimensionless)
    w_slack: float = 1e9  # business-gap penalty per unit of unmet work


def build_milp(problem: PlanningProblem, mip: bool = True) -> dict:
    """Build scipy.optimize.milp inputs for the dispatch problem."""
    T = int(problem.horizon)
    n = len(problem.tasks)
    G = len(problem.group_cap)

    X0 = 0
    G0 = n * T
    C0 = G0 + G * T
    D0 = C0 + T
    Z0 = D0 + T
    SOC0 = Z0 + T
    GRID0 = SOC0 + (T + 1)
    PV0 = GRID0 + T
    U0 = PV0 + T
    U_SIZE = (G + 2) if problem.raw_compute is not None else 0
    S0 = U0 + U_SIZE
    NVAR = S0 + n

    lb = np.zeros(NVAR)
    ub = np.zeros(NVAR)
    integrality = np.zeros(NVAR, dtype=int)

    # x[t, i]
    for i, task in enumerate(problem.tasks):
        upper = min(float(task.max_rate), float(task.remaining_work))
        for t in range(T):
            idx = X0 + t * n + i
            ub[idx] = upper if task.start <= t <= task.deadline else 0.0
    # g[t, j]
    for t in range(T):
        for j in range(G):
            ub[G0 + t * G + j] = float(problem.group_cap[j])
    # c / d
    ub[C0:C0 + T] = problem.charge_max_kw
    ub[D0:D0 + T] = problem.discharge_max_kw
    # z (binary)
    ub[Z0:Z0 + T] = 1.0
    integrality[Z0:Z0 + T] = 1
    # soc
    lb[SOC0:SOC0 + T + 1] = problem.soc_min
    ub[SOC0:SOC0 + T + 1] = problem.soc_max
    # grid
    ub[GRID0:GRID0 + T] = problem.access_limit_kw
    # pv
    ub[PV0:PV0 + T] = np.maximum(problem.pv_available, 0.0)
    # deviation auxiliaries (>= 0)
    if U_SIZE:
        ub[U0:U0 + U_SIZE] = np.inf
    # business-gap slack (>= 0)
    ub[S0:S0 + n] = np.inf

    # Objective: minimize grid energy cost + battery degradation + deviation.
    c = np.zeros(NVAR)
    c[GRID0:GRID0 + T] = problem.price
    c[C0:C0 + T] += problem.degradation_per_kwh
    c[D0:D0 + T] += problem.degradation_per_kwh
    if U_SIZE:
        c[U0:U0 + G] = problem.w_dev / np.maximum(problem.group_cap, 1e-9)
        c[U0 + G] = problem.w_dev / max(problem.charge_max_kw, 1e-9)
        c[U0 + G + 1] = problem.w_dev / max(problem.discharge_max_kw, 1e-9)
    c[S0:S0 + n] = problem.w_slack

    eq_rows: list[np.ndarray] = []
    eq_rhs: list[float] = []
    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []
    ge_rows: list[np.ndarray] = []
    ge_lb: list[float] = []

    # 1. Task completion with business-gap slack: sum_t x[t,i] + s_i == remaining_work.
    for i, task in enumerate(problem.tasks):
        row = np.zeros(NVAR)
        row[X0 + np.arange(T) * n + i] = 1.0
        row[S0 + i] = 1.0
        eq_rows.append(row)
        eq_rhs.append(float(task.remaining_work))

    # 2. Work conservation per step (equality): sum_i x[t,i] == sum_j g[t,j].
    for t in range(T):
        row = np.zeros(NVAR)
        row[X0 + t * n : X0 + t * n + n] = 1.0
        row[G0 + t * G : G0 + t * G + G] = -1.0
        eq_rows.append(row)
        eq_rhs.append(0.0)

    # 3. Charge/discharge mutual exclusion (inequality).
    for t in range(T):
        r_c = np.zeros(NVAR)
        r_c[C0 + t] = 1.0
        r_c[Z0 + t] = -problem.charge_max_kw
        ub_rows.append(r_c)
        ub_rhs.append(0.0)

        r_d = np.zeros(NVAR)
        r_d[D0 + t] = 1.0
        r_d[Z0 + t] = problem.discharge_max_kw
        ub_rows.append(r_d)
        ub_rhs.append(problem.discharge_max_kw)

    # 4. SOC dynamics (equality): soc[t+1] = soc[t] + (c*eta_c - d/eta_d)*dt/cap.
    for t in range(T):
        row = np.zeros(NVAR)
        row[SOC0 + t + 1] = 1.0
        row[SOC0 + t] = -1.0
        row[C0 + t] = -problem.charge_eff * problem.dt / problem.cap_kwh
        row[D0 + t] = (1.0 / problem.discharge_eff) * problem.dt / problem.cap_kwh
        eq_rows.append(row)
        eq_rhs.append(0.0)

    # 5. SOC initial (equality).
    row = np.zeros(NVAR)
    row[SOC0] = 1.0
    eq_rows.append(row)
    eq_rhs.append(float(problem.soc0))

    # 6. Energy balance (inequality): p_slope*sum g + c - d - pv - grid <= -p_base.
    for t in range(T):
        row = np.zeros(NVAR)
        row[G0 + t * G : G0 + t * G + G] = problem.p_slope_kw_per_work
        row[C0 + t] = 1.0
        row[D0 + t] = -1.0
        row[PV0 + t] = -1.0
        row[GRID0 + t] = -1.0
        ub_rows.append(row)
        ub_rhs.append(-problem.p_base_kw)

    # 7. Minimal-modification deviation (>=): u >= |g - raw|, u_c/d >= |c/d - raw|.
    if U_SIZE:
        raw = np.asarray(problem.raw_compute, dtype=float)
        raw_work = raw * problem.group_cap
        raw_charge = max(-problem.raw_storage, 0.0) * problem.charge_max_kw
        raw_discharge = max(problem.raw_storage, 0.0) * problem.discharge_max_kw
        for j in range(G):
            r1 = np.zeros(NVAR)
            r1[U0 + j] = 1.0
            r1[G0 + j] = -1.0
            ge_rows.append(r1)
            ge_lb.append(-raw_work[j])
            r2 = np.zeros(NVAR)
            r2[U0 + j] = 1.0
            r2[G0 + j] = 1.0
            ge_rows.append(r2)
            ge_lb.append(raw_work[j])
        r3 = np.zeros(NVAR)
        r3[U0 + G] = 1.0
        r3[C0] = -1.0
        ge_rows.append(r3)
        ge_lb.append(-raw_charge)
        r4 = np.zeros(NVAR)
        r4[U0 + G] = 1.0
        r4[C0] = 1.0
        ge_rows.append(r4)
        ge_lb.append(raw_charge)
        r5 = np.zeros(NVAR)
        r5[U0 + G + 1] = 1.0
        r5[D0] = -1.0
        ge_rows.append(r5)
        ge_lb.append(-raw_discharge)
        r6 = np.zeros(NVAR)
        r6[U0 + G + 1] = 1.0
        r6[D0] = 1.0
        ge_rows.append(r6)
        ge_lb.append(raw_discharge)

    constraints: list[LinearConstraint] = []
    if eq_rows:
        A_eq = np.vstack(eq_rows)
        rhs = np.asarray(eq_rhs, dtype=float)
        constraints.append(LinearConstraint(A_eq, rhs, rhs))
    if ub_rows:
        A_ub = np.vstack(ub_rows)
        constraints.append(LinearConstraint(A_ub, -np.inf, np.asarray(ub_rhs, dtype=float)))
    if ge_rows:
        A_ge = np.vstack(ge_rows)
        constraints.append(LinearConstraint(A_ge, np.asarray(ge_lb, dtype=float), np.full(len(ge_lb), np.inf)))

    n_int = int(np.sum(integrality)) if mip else 0
    return {
        "c": c,
        "integrality": integrality if mip else np.zeros(NVAR, dtype=int),
        "bounds": Bounds(lb, ub),
        "constraints": constraints,
        "nvar": NVAR,
        "n_integer": n_int,
        "n_continuous": NVAR - n_int,
        "layout": {
            "X0": X0, "G0": G0, "C0": C0, "D0": D0, "Z0": Z0,
            "SOC0": SOC0, "GRID0": GRID0, "PV0": PV0, "U0": U0, "S0": S0,
        },
    }

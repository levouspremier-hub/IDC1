"""M4.4 — rolling-window corrector: a_exec = S(s, a_raw) via the dispatch MILP.

Reads only (SystemSnapshot, DispatchProposal); never touches policy parameters
or learned value functions. The corrector_context carries the physical +
planning data the corrector needs (the env populates it in M3).
"""

from __future__ import annotations

import numpy as np

from contracts import DispatchProposal, DispatchResult, SystemSnapshot
from planning.model import PlanningProblem, TaskPlan
from planning.solver import solve

_REQUIRED_CTX = (
    "horizon",
    "group_cap",
    "tasks",
    "soc_min",
    "soc_max",
    "capacity_kwh",
    "charge_power_max_kw",
    "discharge_power_max_kw",
    "charge_efficiency",
    "discharge_efficiency",
    "delta_t_hours",
    "pv_available",
    "access_limit_kw",
    "p_base_kw",
    "p_slope_kw_per_work",
    "price",
    "degradation_per_kwh",
)


class RollingCorrector:
    def correct(self, snapshot: SystemSnapshot, proposal: DispatchProposal) -> DispatchResult:
        problem = self._to_problem(snapshot, proposal)
        out = solve(problem, mip=True)
        if out.x is None:
            return DispatchResult(
                raw_compute=list(proposal.compute),
                raw_storage=proposal.storage,
                exec_compute=list(proposal.compute),
                exec_storage=proposal.storage,
                energy_flows={},
                correction_reason="infeasible",
                solve_status=out.status,
                business_gap=0.0,
            )
        return self._to_result(out, proposal, problem)

    def _to_problem(self, snapshot: SystemSnapshot, proposal: DispatchProposal) -> PlanningProblem:
        ctx = snapshot.corrector_context
        missing = [k for k in _REQUIRED_CTX if k not in ctx]
        if missing:
            raise ValueError(f"corrector_context missing required keys: {missing}")
        tasks = [TaskPlan(task_id=i, **t) for i, t in enumerate(ctx["tasks"])]
        return PlanningProblem(
            horizon=int(ctx["horizon"]),
            tasks=tasks,
            group_cap=np.asarray(ctx["group_cap"], dtype=float),
            soc0=float(snapshot.soc),
            soc_min=float(ctx["soc_min"]),
            soc_max=float(ctx["soc_max"]),
            cap_kwh=float(ctx["capacity_kwh"]),
            charge_max_kw=float(ctx["charge_power_max_kw"]),
            discharge_max_kw=float(ctx["discharge_power_max_kw"]),
            charge_eff=float(ctx["charge_efficiency"]),
            discharge_eff=float(ctx["discharge_efficiency"]),
            dt=float(ctx["delta_t_hours"]),
            pv_available=np.asarray(ctx["pv_available"], dtype=float),
            access_limit_kw=float(ctx["access_limit_kw"]),
            p_base_kw=float(ctx["p_base_kw"]),
            p_slope_kw_per_work=float(ctx["p_slope_kw_per_work"]),
            price=np.asarray(ctx["price"], dtype=float),
            degradation_per_kwh=float(ctx["degradation_per_kwh"]),
            raw_compute=np.asarray(proposal.compute, dtype=float),
            raw_storage=float(proposal.storage),
        )

    def _to_result(self, out, proposal: DispatchProposal, problem: PlanningProblem) -> DispatchResult:
        layout = out.layout
        x = np.asarray(out.x)
        G = len(problem.group_cap)
        g0 = x[layout["G0"] : layout["G0"] + G]
        exec_compute = list(g0 / np.maximum(problem.group_cap, 1e-9))
        c0 = float(x[layout["C0"]])
        d0 = float(x[layout["D0"]])
        if c0 > d0:
            exec_storage = -c0 / max(problem.charge_max_kw, 1e-9)
        else:
            exec_storage = d0 / max(problem.discharge_max_kw, 1e-9)

        raw_work = np.asarray(proposal.compute) * problem.group_cap
        dev = float(np.max(np.abs(g0 - raw_work) / np.maximum(problem.group_cap, 1e-9)))
        storage_dev = abs(exec_storage - proposal.storage)
        business_gap = float(np.sum(x[layout["S0"] : layout["S0"] + len(problem.tasks)]))

        if business_gap > 1e-6:
            reason = "business_gap"
        elif dev > 1e-6 or storage_dev > 1e-6:
            reason = "corrected"
        else:
            reason = "none"

        return DispatchResult(
            raw_compute=list(proposal.compute),
            raw_storage=proposal.storage,
            exec_compute=exec_compute,
            exec_storage=exec_storage,
            energy_flows={
                "bess_charge_kWh": c0 * problem.dt,
                "bess_discharge_kWh": d0 * problem.dt,
            },
            correction_reason=reason,
            solve_status=out.status,
            business_gap=business_gap,
        )

"""M4.5/M4.6 — HiGHS solver wrapper and failure classification."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import milp

from planning.model import PlanningProblem, build_milp


@dataclass
class SolveOutcome:
    status: str  # optimal | feasible | infeasible | timeout | unbounded | other
    elapsed: float
    x: np.ndarray | None
    nvar: int
    n_integer: int
    layout: dict


def _classify(res, elapsed: float, time_limit: float) -> str:
    if res.x is not None:
        return "optimal" if res.status == 0 else "feasible"
    msg = (res.message or "").lower()
    if "infeasible" in msg:
        return "infeasible"
    if "unbounded" in msg:
        return "unbounded"
    if "time" in msg or "limit" in msg or elapsed >= time_limit - 1e-3:
        return "timeout"
    return "other"


def solve(problem: PlanningProblem, mip: bool = True, time_limit: float = 10.0) -> SolveOutcome:
    spec = build_milp(problem, mip=mip)
    t0 = time.perf_counter()
    res = milp(
        c=spec["c"],
        integrality=spec["integrality"],
        bounds=spec["bounds"],
        constraints=spec["constraints"],
        options={"time_limit": time_limit},
    )
    elapsed = time.perf_counter() - t0
    return SolveOutcome(
        status=_classify(res, elapsed, time_limit),
        elapsed=elapsed,
        x=res.x,
        nvar=spec["nvar"],
        n_integer=spec["n_integer"],
        layout=spec["layout"],
    )

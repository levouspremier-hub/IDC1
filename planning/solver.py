"""M4.3 求解器：默认 MIP（二元互斥），`--allow-lp-relaxation` 显式开启 LP 松弛。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult, milp

from contracts.models import SystemSnapshot
from planning.model import build_milp


@dataclass
class SolveResult:
    success: bool
    status: str
    allocation: np.ndarray  # n_task × n_group
    charge: float
    discharge: float
    z: float
    objective: float


def solve(snapshot: SystemSnapshot, *, allow_lp_relaxation: bool = False) -> SolveResult:
    model = build_milp(snapshot, allow_lp_relaxation=allow_lp_relaxation)
    res: OptimizeResult = milp(
        c=model.c,
        constraints=model.constraints,
        integrality=model.integrality,
        bounds=model.bounds,
    )
    if not res.success:
        return SolveResult(
            success=False,
            status=str(res.message),
            allocation=np.zeros((model.n_task, model.n_group)),
            charge=0.0,
            discharge=0.0,
            z=0.0,
            objective=0.0,
        )
    n_a = model.n_task * model.n_group
    allocation = res.x[:n_a].reshape(model.n_task, model.n_group)
    return SolveResult(
        success=True,
        status=str(res.message),
        allocation=allocation,
        charge=float(res.x[n_a]),
        discharge=float(res.x[n_a + 1]),
        z=float(res.x[n_a + 2]),
        objective=float(-res.fun),
    )

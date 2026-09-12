"""M4.3 — minimal solver probe: variable scale, integer count, timing, memory.

This is the Week-1 hard gate: the measured numbers decide whether multi-seed
experiments stay local or move off-machine. Run:

    uv run python planning/probe.py
"""

from __future__ import annotations

import resource
import sys
from pathlib import Path

import numpy as np


def _ensure_project_root_on_path() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "planning").exists() and (candidate / "contracts").exists():
            root = candidate
            break
    else:
        raise RuntimeError("Cannot locate project root")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


_ensure_project_root_on_path()

from planning.model import PlanningProblem, TaskPlan
from planning.solver import solve


def _make_problem(n_tasks: int, n_groups: int, horizon: int, seed: int = 0) -> PlanningProblem:
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n_tasks):
        remaining = float(rng.uniform(2.0, 10.0))
        max_rate = float(rng.uniform(2.0, 6.0))
        min_steps = max(1, int(np.ceil(remaining / max_rate)))
        start = int(rng.integers(0, max(1, horizon - min_steps - 2)))
        deadline = min(horizon - 1, start + min_steps + int(rng.integers(0, 4)))
        tasks.append(TaskPlan(task_id=i, remaining_work=remaining, max_rate=max_rate, start=start, deadline=deadline))
    group_cap = rng.uniform(3.0, 6.0, size=n_groups)
    return PlanningProblem(
        horizon=horizon,
        tasks=tasks,
        group_cap=group_cap,
        soc0=0.5,
        soc_min=0.1,
        soc_max=0.9,
        cap_kwh=10000.0,
        charge_max_kw=2000.0,
        discharge_max_kw=2000.0,
        charge_eff=0.95,
        discharge_eff=0.95,
        dt=1.0,
        pv_available=np.maximum(rng.normal(400.0, 150.0, size=horizon), 0.0),
        access_limit_kw=25000.0,
        p_base_kw=11000.0,
        p_slope_kw_per_work=0.02,
        price=np.full(horizon, 0.2),
        degradation_per_kwh=0.02,
    )


def _median_p95(times: list[float]) -> tuple[float, float]:
    arr = np.asarray(times)
    return float(np.median(arr)), float(np.percentile(arr, 95))


def main() -> None:
    scales = [
        (5, 20, 24),
        (10, 20, 24),
        (20, 20, 24),
        (30, 20, 24),
        (30, 20, 48),
    ]
    print(f"{'n_task':>6} {'n_grp':>5} {'T':>3} {'nvar':>6} {'n_int':>5} {'med_ms':>8} {'p95_ms':>8} {'infeas%':>8} {'status'}")
    for n_tasks, n_groups, horizon in scales:
        times = []
        statuses = []
        nvar = nin = 0
        for seed in range(20):
            problem = _make_problem(n_tasks, n_groups, horizon, seed=seed)
            out = solve(problem, mip=True, time_limit=10.0)
            times.append(out.elapsed * 1000.0)
            statuses.append(out.status)
            nvar, nin = out.nvar, out.n_integer
        med, p95 = _median_p95(times)
        infeas = statuses.count("infeasible") / len(statuses) * 100.0
        print(
            f"{n_tasks:>6} {n_groups:>5} {horizon:>3} {nvar:>6} {nin:>5} "
            f"{med:>8.2f} {p95:>8.2f} {infeas:>8.1f} {statuses[0]}"
        )

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = rss / (1024.0 * 1024.0) if sys.platform == "darwin" else rss / 1024.0
    print(f"\npeak RSS (this process): {peak_mb:.1f} MB")


if __name__ == "__main__":
    main()

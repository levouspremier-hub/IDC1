"""Minimal vertical slice: contracts -> corrector -> result -> eval -> artifacts.

Wires the six contracts, the rolling corrector, and a minimal evaluation record
into one runnable loop at the target scale (20 groups, 24 h). It proves the
architecture end-to-end WITHOUT the M3 env rewrite, and writes JSON artifacts to
``runs/smoke/`` for review.

Run:
    uv run python scripts/vertical_slice.py      # or: make smoke
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _ensure_project_root_on_path() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "envs").exists() and (candidate / "contracts").exists():
            root = candidate
            break
    else:
        raise RuntimeError("Cannot locate project root")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


PROJECT_ROOT = _ensure_project_root_on_path()

import numpy as np

from contracts import (
    DispatchProposal,
    DispatchResult,
    EvaluationRecord,
    ScenarioBundle,
    SystemSnapshot,
    TaskAllocation,
    TaskSpec,
)
from planning.corrector import RollingCorrector


def _build_scenario() -> ScenarioBundle:
    tasks = [
        TaskSpec(task_id=0, workload=20.0, arrival_time=0, latest_finish_time=8, max_rate=6.0, is_critical=True),
        TaskSpec(task_id=1, workload=12.0, arrival_time=2, latest_finish_time=14, max_rate=4.0),
        TaskSpec(task_id=2, workload=30.0, arrival_time=0, latest_finish_time=23, max_rate=5.0),
    ]
    return ScenarioBundle(
        scenario_id="smoke",
        units={"price": "SGD/kWh", "power": "kW", "energy": "kWh", "workload": "work"},
        sources={"price": "synthetic", "pv": "synthetic"},
        price=[0.2 + 0.01 * (t % 12) for t in range(24)],
        pv=[500.0 * max(0.0, np.sin(np.pi * (t - 6) / 12)) for t in range(24)],
        tasks=tasks,
        capacity={"n_groups": 20, "group_capacity_work_per_hour": [10.0] * 20},
    ).freeze()


def _snapshot(scenario: ScenarioBundle) -> SystemSnapshot:
    horizon = 24
    ctx = {
        "horizon": horizon,
        "group_cap": [10.0] * 20,
        "tasks": [
            {"remaining_work": 20.0, "max_rate": 6.0, "start": 0, "deadline": 8},
            {"remaining_work": 12.0, "max_rate": 4.0, "start": 0, "deadline": 14},
            {"remaining_work": 30.0, "max_rate": 5.0, "start": 0, "deadline": 23},
        ],
        "soc_min": 0.1,
        "soc_max": 0.9,
        "capacity_kwh": 10000.0,
        "charge_power_max_kw": 2000.0,
        "discharge_power_max_kw": 2000.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "delta_t_hours": 1.0,
        "pv_available": scenario.pv,
        "access_limit_kw": 25000.0,
        "p_base_kw": 11000.0,
        "p_slope_kw_per_work": 0.02,
        "price": scenario.price,
        "degradation_per_kwh": 0.02,
    }
    return SystemSnapshot(t=0, task_states=[], soc=0.5, budgets={"carbon": 500.0}, corrector_context=ctx)


def main() -> int:
    scenario = _build_scenario()
    snapshot = _snapshot(scenario)
    proposal = DispatchProposal(compute=[0.5] * 20, storage=-0.3)

    result: DispatchResult = RollingCorrector().correct(snapshot, proposal)

    # Minimal allocation/evaluation record derived from the corrected result.
    allocation = TaskAllocation(matrix=[[0.0] * 20] * 3, unit="work/hour")
    evaluation = EvaluationRecord(
        scenario_id=scenario.scenario_id,
        business={"completed_work": 0.0, "deadline_misses": 0, "business_gap": result.business_gap},
        economic={"total_cost": 0.0, "bess_degradation_cost": 0.0},
        carbon={"emissions_kg": 0.0},
        renewable={"pv_curtail_kWh": 0.0},
    )

    out_dir = PROJECT_ROOT / "runs" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scenario.json").write_text(scenario.to_json(), encoding="utf-8")
    (out_dir / "snapshot.json").write_text(snapshot.to_json(), encoding="utf-8")
    (out_dir / "proposal.json").write_text(proposal.to_json(), encoding="utf-8")
    (out_dir / "result.json").write_text(result.to_json(), encoding="utf-8")
    (out_dir / "evaluation.json").write_text(evaluation.to_json(), encoding="utf-8")
    (out_dir / "allocation.json").write_text(allocation.to_json(), encoding="utf-8")

    # Green = loop works: contracts serialize, solver optimal, tasks fully completable.
    # (A "corrected" reason is legitimate here: the raw proposal deliberately over-allocates.)
    ok = (
        result.solve_status == "optimal"
        and result.business_gap < 1e-6
        and scenario.hash == scenario.content_hash()
    )
    print(json.dumps({
        "scenario_hash": scenario.hash,
        "solve_status": result.solve_status,
        "correction_reason": result.correction_reason,
        "business_gap": result.business_gap,
        "exec_storage": round(result.exec_storage, 4),
        "artifacts": str(out_dir),
        "slice_green": ok,
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

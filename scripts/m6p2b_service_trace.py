"""Train-only arrived-task service traces; instrumentation never changes planning inputs."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from unittest.mock import patch

import pandas as pd

import planning.corrector as planner
import safe_rl.corrector_wrapper as wrapper
from runs.writer import write_run
from safe_rl_v2.formal_train_loop import build_train_env, training_source_ledger
from safe_rl_v2.inventory_diagnostics import evaluate_origin
from safe_rl_v2.rollout import _json_safe
from scenario.inventory_release import ROOT, diagnostic_candidate_config, semantics_binding, sha
from scripts.calibrate_training_config import select_origins
from scripts.m6p2b_reward_counterfactual import proposal


def calibrate_temperature(run_id, config):
    import math

    import numpy as np
    rows, provenance = [], {}
    for origin in select_origins():
        env, injection = build_train_env(origin, master_seed=0, config=config)
        for step in range(48):
            forecast = float(env.temperature_forecast_t[step])
            realized = float(env.T_amb[step])
            rows.append(dict(origin=origin, step=step, forecast_c=forecast,
                             realized_c=realized, underestimation_c=max(realized-forecast, 0.)))
        provenance[origin] = injection.provenance_hash
        print(f"temperature origin={origin}", flush=True)
    maximum = max(row["underestimation_c"] for row in rows)
    report = dict(scope="train_only_temperature_reserve_calibration",
                  origins=list(select_origins()), samples=len(rows),
                  maximum_underestimation_c=maximum,
                  margin_c=math.ceil(maximum*10)/10,
                  formula="ceil(max(train_realized_c - signed_B6_forecast_c, 0) * 10) / 10",
                  parameter_updates=0, validation_run=False, test_run=False,
                  physical_error_bound_proven=False,
                  forecast_arrays_modified=False,
                  instrumentation_sha256=sha(__file__),
                  maximum_event=[r for r in rows if np.isclose(r["underestimation_c"],maximum)],
                  candidate_semantics_binding=semantics_binding())
    folder = write_run(run_id, config=config, metrics=pd.DataFrame(rows), report=report,
                       base_dir=str(ROOT/"runs"), seed=0,
                       command="python -m scripts.m6p2b_service_trace --temperature-calibration "
                       + "--run-id " + run_id, **training_source_ledger(provenance))
    if json.loads((folder/"report.json").read_text()) != report:
        raise ValueError("temperature calibration read-back differs")
    if len(pd.read_parquet(folder/"metrics.parquet")) != 1152:
        raise ValueError("temperature calibration row count differs")
    print(json.dumps(report), flush=True)



def tasks_now(env):
    return [{key: getattr(t, key) for key in (
        "task_id", "status", "remaining_work", "workload", "duration", "arrival_time",
        "latest_finish_time", "interruptible", "start_time", "finish_time",
        "pause_count", "non_interruptible_interruption_count")}
        for t in env.tasks if t.status != "not_arrived"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--temperature-calibration", action="store_true")
    args = parser.parse_args()
    if (ROOT / "runs" / args.run_id).exists():
        raise FileExistsError(args.run_id)
    config = diagnostic_candidate_config()
    if args.temperature_calibration:
        calibrate_temperature(args.run_id, config)
        return
    rows, episodes, provenance = [], [], {}
    active = {}
    original_snapshot = wrapper.build_snapshot
    original_solve = planner.solve_time_indexed_mip_raw_projection
    original_step = wrapper.CorrectorWrapper.step

    def snapshot(env):
        snap = original_snapshot(env)
        active["snapshot"] = snap.model_dump()
        return snap

    def solve(*a, **kw):
        result = original_solve(*a, **kw)
        active["allocation"] = result.allocation[:, :, 0].tolist()
        return result

    def step(self, action):
        env = self.env.unwrapped
        t = int(env.current_step)
        before = tasks_now(env)
        # Diagnostic evidence only: no realized array is passed to the planner.
        actual = {"temperature": float(env.T_amb[t]),
                  "pv": float(env.pv_t[t]), "wind": float(env.wt_t[t])}
        active.clear()
        result = original_step(self, action)
        rows.append(_json_safe({"origin": active_origin, "step": t,
                                "before_tasks": before, "after_tasks": tasks_now(env),
                                "realized_current_diagnostic_only": actual,
                                "snapshot": active["snapshot"],
                                "planned_current_allocation": active.get("allocation"),
                                "info": result[-1]}))
        return result

    error = None
    with patch.object(wrapper, "build_snapshot", snapshot), patch.object(
            planner, "solve_time_indexed_mip_raw_projection", solve), patch.object(
            wrapper.CorrectorWrapper, "step", step), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            for active_origin in (4416, 6624):
                result, _, service, _ = evaluate_origin(
                    config, active_origin, 0, proposal(.1, 523), run_id=args.run_id)
                episodes.append({**result, "service": service.service.model_dump()})
                provenance[active_origin] = result["injection_provenance"]
                print(json.dumps({"origin": active_origin,
                                  "service": service.service.model_dump()}), flush=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    report = {"scope": "train_only_service_trace", "parameter_updates": 0,
              "validation_run": False, "test_run": False, "failure": error,
              "episodes": episodes, "steps": rows,
              "candidate_semantics_binding": semantics_binding(),
              "instrumentation_sha256": sha(__file__),
              "realized_current_fields_used_for_planning": False}
    metrics = pd.DataFrame([{ "origin": row["origin"], "step": row["step"],
                              **row["info"]} for row in rows])
    folder = write_run(args.run_id, config=config, metrics=metrics, report=report,
                       base_dir=str(ROOT / "runs"), seed=0,
                       command="python -m scripts.m6p2b_service_trace " + " ".join(sys.argv[1:]),
                       status="failed" if error else "success",
                       failure_classification="trace_failure" if error else None,
                       **training_source_ledger(provenance))
    if json.loads((folder / "report.json").read_text()) != report:
        raise ValueError("trace read-back differs")
    if len(pd.read_parquet(folder / "metrics.parquet")) != len(rows):
        raise ValueError("trace row count differs")
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()

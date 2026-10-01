"""Train failure snapshot comparison under the unchanged .25 s global budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import types
import warnings
from unittest.mock import patch

import pandas as pd
import scipy.optimize

import planning.model as candidate
from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from runs.writer import write_run
from scenario.inventory_release import ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-run", required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if (ROOT / "runs" / args.run_id).exists():
        raise FileExistsError("probe run already exists")
    trace = ROOT / "runs" / args.trace_run
    report = json.loads((trace / "report.json").read_text())
    source = subprocess.check_output([
        "git", "show", f"{args.baseline_revision}:planning/model.py"], cwd=ROOT)
    if hashlib.sha256(source).hexdigest() != report["inventory_binding"][
            "semantics_binding"]["planning/model.py"]:
        raise ValueError("baseline source does not match the recorded training binding")
    baseline = types.ModuleType("m6p2b_recorded_planner")
    sys.modules[baseline.__name__] = baseline
    exec(compile(source, "recorded_planning_model.py", "exec"), baseline.__dict__)
    snapshots = [f for f in report["failure_snapshots"]
                 if f["info"]["correction_reason"] == "timeout"][:8]
    if not snapshots:
        raise ValueError("trace contains no timed-out snapshot")
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for repetition in range(2):
            for f in snapshots:
                snapshot = InventorySnapshot.model_validate(f["snapshot"])
                raw = f["info"]["raw_action"]
                proposal = DispatchProposal(compute_actions=raw[:20], storage_action=raw[-1])
                arms = (("baseline", baseline), ("candidate", candidate))
                for arm, module in arms if repetition == 0 else reversed(arms):
                    solver_calls = []
                    original_milp = scipy.optimize.milp

                    def measured_milp(_original=original_milp, _calls=solver_calls, **kwargs):
                        output = _original(**kwargs)
                        _calls.append({"status": int(output.status),
                                       "message": str(output.message)})
                        return output

                    start = time.perf_counter()
                    with patch.object(scipy.optimize, "milp", measured_milp):
                        result = module.solve_time_indexed_mip_raw_projection(
                            snapshot, proposal, time_limit_s=.25)
                    rows.append({
                        "repetition": repetition, "origin": f["origin"], "step": f["step"],
                        "arm": arm, "status": result.solver_status,
                        "failure": result.failure_class, "wall_s": time.perf_counter() - start,
                        "stage_r_s": result.inventory_audit.get("reachability_solve_time_s"),
                        "stage_a_s": result.stage_a_solve_time_s,
                        "stage_b_s": result.stage_b_solve_time_s,
                        "n_variables": result.n_variables,
                        "n_constraints": result.n_constraints,
                        "offset": result.projection_offset,
                        "target_gap_kwh": result.inventory_audit.get("target_gap_kwh"),
                        "solver_calls": solver_calls,
                    })
    frame = pd.DataFrame(rows)
    result = {
        "scope": "train_only_unreleased_candidate_budget_probe", "global_budget_s": .25,
        "baseline_revision": args.baseline_revision,
        "baseline_planning_sha256": hashlib.sha256(source).hexdigest(),
        "candidate_planning_sha256": sha(ROOT / "planning/model.py"),
        "instrumentation_sha256": sha(__file__),
        "trace_report_sha256": sha(trace / "report.json"), "rows": rows,
        "summary": {arm: {"optimal": int((group.status == "optimal").sum()),
                           "timeouts": int((group.failure == "timeout").sum()),
                           "solver_failures": int((group.failure == "solver_failure").sum()),
                           "mean_wall_s": float(group.wall_s.mean()), "calls": len(group)}
                    for arm, group in frame.groupby("arm")},
        "parameters_updated": 0, "validation_run": False, "test_run": False,
        "formal_acceptance_claimed": False,
    }
    source_manifest = json.loads((trace / "manifest.json").read_text())
    folder = write_run(
        args.run_id, config={"global_budget_s": .25, "recorded_snapshots_only": True},
        metrics=frame, report=result, base_dir=str(ROOT / "runs"), seed=report["seed"],
        command="python -m scripts.m6p2b_budget_probe " + " ".join(sys.argv[1:]),
        **{k: source_manifest[k] for k in (
            "data_hash", "scenario_hash", "dependency_lock_hash")})
    if json.loads((folder / "report.json").read_text()) != result:
        raise ValueError("budget report read-back differs")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

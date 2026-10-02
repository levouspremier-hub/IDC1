"""Fixed train intraday same-state probes; no policy updates or future planner truth."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from planning.snapshot_adapter import build_snapshot
from runs.writer import write_run
from safe_rl.corrector_wrapper import CorrectorWrapper
from safe_rl_v2 import inventory_diagnostics as diagnostics
from safe_rl_v2.formal_train_loop import build_train_env, training_source_ledger
from scenario.inventory_release import ROOT, checkpoint_binding, load_config, sha
from scripts.calibrate_training_config import select_origins
from scripts.m6p2b_reward_counterfactual import proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if (ROOT / "runs" / args.run_id).exists():
        raise FileExistsError("state probe run already exists")
    config, binding = load_config(), checkpoint_binding()
    origins, times = list(select_origins()), (0, 12, 24, 36)
    rows, episodes, provenance = [], [], {}
    failure = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for origin in origins:
                active = {}
                policy = proposal(.1, config["training"]["policy"]["obs_dim"])

                def builder(*args, active_state=active, **kwargs):
                    env, injection = build_train_env(*args, **kwargs)
                    active_state["env"] = env
                    return env, injection

                def act(obs, active_policy=policy, active_state=active, active_origin=origin):
                    raw = active_policy(obs)
                    env = active_state["env"]
                    step = int(env.current_step)
                    if step in times:
                        snapshot = build_snapshot(env)
                        for storage in (-.1, 0., .1):
                            clone = copy.deepcopy(env)
                            if build_snapshot(clone) != snapshot:
                                raise ValueError("same-state planning inputs differ")
                            changed = raw.copy()
                            changed[-1] = storage
                            before = {str(t.task_id): t.remaining_work for t in clone.tasks
                                      if t.status != "not_arrived"}
                            _, reward, _, _, info = CorrectorWrapper(
                                clone, corrector_time_limit_s=.25).step(changed)
                            rows.append({
                                "origin": active_origin, "step": step,
                                "raw_storage": storage,
                                "exec_storage": float(info["exec_action"][-1]),
                                "actual_storage": (info["bess_discharge_power_kW"]
                                                   / clone.bess_discharge_power_max_kW
                                                   - info["bess_charge_power_kW"]
                                                   / clone.bess_charge_power_max_kW),
                                "charge_kw": info["bess_charge_power_kW"],
                                "discharge_kw": info["bess_discharge_power_kW"],
                                "purchase_cost_sgd": info["electricity_cost"],
                                "degradation_cost_sgd": info["bess_degradation_cost"],
                                "reward": float(reward),
                                "soc_before": env.bess_energy_kWh / env.bess_capacity_kWh,
                                "soc_after": clone.bess_energy_kWh / clone.bess_capacity_kWh,
                                "correction_reason": info["correction_reason"],
                                "task_progress": {
                                    str(t.task_id): before[str(t.task_id)] - t.remaining_work
                                    for t in clone.tasks if str(t.task_id) in before},
                                "reward_components": {k: float(v) for k, v in info.items()
                                                      if k.startswith("r_")},
                            })
                    return raw

                with patch.object(diagnostics, "build_train_env", builder):
                    result, _, _, _ = diagnostics.evaluate_origin(
                        config, origin, 0, act, run_id=args.run_id)
                episodes.append(result)
                provenance[origin] = result["injection_provenance"]
                print(f"origin={origin} recorded_probes={len(rows)} "
                      f"episode_failure={result['failure']}", flush=True)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    spreads = []
    for origin in origins:
        for step in times:
            current = [r for r in rows if r["origin"] == origin and r["step"] == step]
            if len(current) == 3:
                spreads.append({"origin": origin, "step": step,
                                "exec_range": float(np.ptp([r["exec_storage"] for r in current])),
                                "reward_range": float(np.ptp([r["reward"] for r in current]))})
    passed = (failure is None and len(rows) == 288 and len(episodes) == 24
              and all(all(diagnostics.inventory_episode_acceptance(e).values()) for e in episodes))
    report = {"passed": passed, "failure": failure, "origins": origins,
              "fixed_steps": list(times), "storage_proposals": [-.1, 0., .1],
              "episodes": episodes, "same_state_spreads": spreads,
              "noncollapsed_states": sum(s["exec_range"] > 1e-6 for s in spreads),
              "collapsed_states": sum(s["exec_range"] <= 1e-6 for s in spreads),
              "parameter_updates": 0, "validation_run": False, "test_run": False,
              "inventory_binding": binding, "instrumentation_sha256": sha(__file__),
              "baseline_proposal": "inventory-aware-causal-price-probe-v2 amplitude=.1",
              "planning_truth_access": False, "current_realization_used_only_by_environment": True}
    ledger = training_source_ledger(provenance)
    folder = write_run(
        args.run_id, config={"training": config, "fixed_steps": list(times),
                             "inventory_binding": binding},
        metrics=pd.DataFrame(rows), report=report, base_dir=str(ROOT / "runs"), seed=0,
        command="python -m scripts.m6p2b_storage_state_probe " + " ".join(sys.argv[1:]),
        status="success" if passed else "failed",
        failure_classification=None if passed else "state_probe_failed",
        **ledger)
    if (json.loads((folder / "report.json").read_text()) != report
            or len(pd.read_parquet(folder / "metrics.parquet")) != len(rows)):
        raise ValueError("state probe read-back differs")
    manifest = json.loads((folder / "manifest.json").read_text())
    saved_config = yaml.safe_load((folder / "config.yaml").read_text())
    if (any(manifest[k] != v for k, v in ledger.items())
            or saved_config["inventory_binding"] != binding
            or not (folder / "figures").is_dir()):
        raise ValueError("state probe provenance/config read-back differs")
    if not passed:
        raise RuntimeError("state probe failed; all evidence retained")
    print(json.dumps({"passed": passed, "noncollapsed_states": report["noncollapsed_states"]}))


if __name__ == "__main__":
    main()

"""Causal train-only full-day storage reward counterfactuals; no parameter updates."""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

from runs.writer import write_run
from safe_rl_v2.formal_train_loop import training_source_ledger
from safe_rl_v2.inventory_diagnostics import evaluate_origin
from scenario.inventory_release import (
    CONFIG_PATH,
    ROOT,
    diagnostic_candidate_config,
    load_config,
    semantics_binding,
    sha,
)
from scripts.calibrate_training_config import select_origins


def proposal(amplitude, obs_dim):
    thresholds = None

    def act(observation):
        nonlocal thresholds
        # 136 current features, then the 48-point causal price forecast channel.
        # The action reads only the same bounded observation available to PPO.
        prices = np.asarray(observation[136:184], dtype=np.float64)
        if len(observation) != obs_dim or obs_dim != 523:
            raise ValueError("counterfactual requires versioned terminal state observation")
        if thresholds is None:
            visible = prices[prices > 0.]
            thresholds = np.quantile(visible, [.25, .75]) if len(visible) else (0., 0.)
        storage = 0.
        if thresholds[0] < thresholds[1]:
            current_price = observation[1]  # Existing current price observable, no future truth.
            soc, target = observation[-3:-1]
            if current_price <= thresholds[0] and soc < target + .1:
                storage = -amplitude
            elif current_price >= thresholds[1] and soc > target - .1:
                storage = amplitude
        return np.asarray([1.] * 20 + [storage], dtype=np.float32)
    return act


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--origins-limit", type=int)
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    if (ROOT / "runs" / args.run_id).exists():
        raise FileExistsError("counterfactual run already exists")
    config = diagnostic_candidate_config() if args.candidate else load_config()
    origins = list(select_origins())
    if args.origins_limit is not None:
        if not 1 <= args.origins_limit <= 24:
            raise ValueError("diagnostic origin prefix must have 1..24 entries")
        origins = origins[:args.origins_limit]
    rows, pairs, provenance = [], [], {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for origin in origins:
                arms = []
                for amplitude in (0., .05, .1):
                    result, _steps, service, inventory = evaluate_origin(
                        config, origin, 0,
                        proposal(amplitude, config["training"]["policy"]["obs_dim"]),
                        run_id=args.run_id)
                    result.update(amplitude=amplitude, service_metrics=service.service.model_dump(),
                                  inventory_record=inventory.to_dict())
                    rows.append(result)
                    arms.append(result)
                    provenance[origin] = result["injection_provenance"]
                baseline = arms[0]
                for arm in arms[1:]:
                    eligible = all(r["episode_complete"] and r["service_qualified"]
                                   and r["inventory_qualified"]
                                   and r["physical_violation_count"] == 0
                                   and r["fallbacks"] == 0 for r in (baseline, arm))
                    bi, ai = baseline["inventory_record"], arm["inventory_record"]
                    terminal_difference = abs(bi["final_energy_kwh"] - ai["final_energy_kwh"])
                    eligible = (eligible and terminal_difference <= 1e-6
                                and bi["capacity_kwh"] == ai["capacity_kwh"]
                                and abs(bi["initial_energy_kwh"]
                                        - ai["initial_energy_kwh"]) <= 1e-6)
                    purchase_gain = baseline["purchase_cost_sgd"] - arm["purchase_cost_sgd"]
                    degradation_extra = (arm["degradation_cost_sgd"]
                                         - baseline["degradation_cost_sgd"])
                    cost_slope = arm["reward_units_per_purchase_sgd"]
                    degradation_slope = arm["reward_units_per_degradation_sgd"]
                    pair = {
                        "origin": origin, "amplitude": arm["amplitude"], "eligible": eligible,
                        "terminal_energy_difference_kwh": terminal_difference,
                        "purchase_gain_sgd": purchase_gain,
                        "degradation_extra_sgd": degradation_extra,
                        "net_money_gain_sgd": purchase_gain - degradation_extra,
                        "original_reward_gain": arm["measured_reward_sum"]
                        - baseline["measured_reward_sum"],
                        "reward_units_per_purchase_sgd": cost_slope,
                        "reward_units_per_degradation_sgd": degradation_slope,
                        "reward_component_differences": {
                            k: arm[k] - baseline[k] for k in arm if k.startswith("r_")
                            and k.endswith(("_sum", "_discounted"))},
                    }
                    pair["common_sgd_reward_arithmetic_gain"] = (
                        pair["original_reward_gain"]
                        - (arm["r_bess_degradation_sum"] - baseline["r_bess_degradation_sum"])
                        - cost_slope * degradation_extra)
                    pairs.append(pair)
                print(f"origin={origin} fair_pairs={sum(p['eligible'] for p in pairs[-2:])} "
                      f"net_money={[round(p['net_money_gain_sgd'], 6) for p in pairs[-2:]]}",
                      flush=True)
    except Exception as error:
        failure = {"scope": "train_only_candidate_reward_counterfactual", "status": "failed",
                   "failure": f"{type(error).__name__}: {error}", "parameter_updates": 0,
                   "candidate_semantics_binding": semantics_binding(),
                   "instrumentation_sha256": sha(__file__), "pairs": pairs,
                   "completed_episodes": len(rows), "formal_acceptance_claimed": False}
        write_run(args.run_id, config=config, metrics=pd.DataFrame(rows), report=failure,
                  base_dir=str(ROOT / "runs"), seed=0, status="failed",
                  failure_classification="candidate_diagnostic_failure",
                  command="python -m scripts.m6p2b_reward_counterfactual " + " ".join(sys.argv[1:]),
                  **training_source_ledger(provenance))
        raise
    report = {
        "scope": "train_only_unreleased_candidate_reward_counterfactual",
        "origins": origins, "full_24_origin_diagnostic": len(origins) == 24,
        "proposal_rule": "compute=1; current price against initial visible B6 price quartiles; "
        "charge/discharge only with current SOC below/above target +/- .1",
        "proposal_version": "inventory-aware-causal-price-probe-v2",
        "amplitudes": [0., .05, .1], "parameter_updates": 0,
        "original_reward_retained": True, "reward_arithmetic_only": True,
        "candidate_semantics_binding": semantics_binding(), "instrumentation_sha256": sha(__file__),
        "training_config_sha256": sha(ROOT / CONFIG_PATH) if not args.candidate else None,
        "pairs": pairs,
        "fair_profitable_pairs": sum(p["eligible"] and p["net_money_gain_sgd"] > 1e-6
                                     for p in pairs),
        "profitable_pairs_with_negative_original_reward": sum(
            p["eligible"] and p["net_money_gain_sgd"] > 1e-6 and p["original_reward_gain"] < 0
            for p in pairs),
        "validation_run": False, "test_run": False, "formal_acceptance_claimed": False,
        "proposed_formula": "r_degradation = -reward_cost_weight * degradation_SGD / cost_ref",
        "coefficient_basis": "Same marginal reward per SGD for grid purchase and degradation; "
        "frozen refs and all other reward terms unchanged; not implemented by this diagnostic",
    }
    folder = write_run(
        args.run_id, config=config, metrics=pd.DataFrame(rows), report=report,
        base_dir=str(ROOT / "runs"), seed=0,
        command="python -m scripts.m6p2b_reward_counterfactual " + " ".join(sys.argv[1:]),
        **training_source_ledger(provenance))
    if (json.loads((folder / "report.json").read_text()) != report
            or len(pd.read_parquet(folder / "metrics.parquet")) != 3 * len(origins)):
        raise ValueError("counterfactual artifacts differ after read-back")
    print(json.dumps({k: report[k] for k in (
        "fair_profitable_pairs", "profitable_pairs_with_negative_original_reward")}), flush=True)


if __name__ == "__main__":
    main()

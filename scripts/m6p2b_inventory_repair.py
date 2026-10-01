"""M6-P2b train-only terminal inventory diagnostics; never evaluates held-out data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from checkpointing import VersionedCheckpoint
from evaluation.controlled_run import deterministic_action
from runs.writer import write_run
from safe_rl.corrector_wrapper import CorrectorWrapper
from safe_rl_v2.formal_train_loop import (
    build_seeded_policy,
    build_train_env,
    load_frozen_training_config,
    training_source_ledger,
)
from safe_rl_v2.inventory_diagnostics import evaluate_origin
from scripts.calibrate_training_config import (
    asset_hashes,
    calibrate_budgets,
    multiplier_scaling,
    select_origins,
    training_config_candidate,
)

ROOT = Path(__file__).resolve().parent.parent


def historical_policy(seed):
    """Explicit historical-only loader: no optimizer restore, no new formal role."""
    config = load_frozen_training_config()
    path = ROOT / f"runs/m13gfck_formal_train_seed{seed}/checkpoint_final.pt"
    checkpoint = VersionedCheckpoint.load(
        path, expected_action_dim=21, expected_obs_dim=520,
        expected_schema_hash="m13gfck-formal-train-resume-v1")
    if checkpoint.state["artifact_role"] != "formal_training_resume":
        raise ValueError("historical role mismatch")
    if checkpoint.state["frozen_config"] != config:
        raise ValueError("historical frozen configuration mismatch")
    policy = build_seeded_policy(config, obs_dim=520, seed=seed)
    policy.load_state_dict(checkpoint.state["policy"], strict=True)
    policy.eval()
    return policy, path, hashlib.sha256(path.read_bytes()).hexdigest()


def replay(config, origins, run_id):
    rows, probes, provenance, sources = [], [], {}, []
    for seed in (0, 1, 2):
        policy, path, sha = historical_policy(seed)
        def action_fn(obs, active_policy=policy):
            return deterministic_action(active_policy, obs)
        for origin in origins:
            for terminal in (False, True):
                result, _steps, _record, _inv = evaluate_origin(
                    config, origin, 0, action_fn, terminal=terminal, run_id=run_id)
                result.update(policy_seed=seed, arm="repaired" if terminal else "historical")
                rows.append(result)
                provenance[origin] = result["injection_provenance"]
            env, _ = build_train_env(origin, master_seed=0, config=config)
            env.terminal_inventory_enabled = True
            obs, _ = env.reset(seed=0)
            raw = deterministic_action(policy, obs)
            for storage in (-1.0, 0.0, 1.0):
                clone = copy.deepcopy(env)
                proposal = raw.copy()
                proposal[-1] = storage
                _, reward, _, _, info = CorrectorWrapper(
                    clone, corrector_time_limit_s=0.25).step(proposal)
                probes.append({
                    "origin": origin, "policy_seed": seed, "step": 0,
                    "raw_storage": storage, "exec_storage": float(info["exec_action"][-1]),
                    "reward": reward, "r_cost": info["r_cost"],
                    "r_soc_final": info["r_soc_final"],
                    "electricity_cost_sgd": info["electricity_cost"],
                    "charge_kw": info["bess_charge_power_kW"],
                    "discharge_kw": info["bess_discharge_power_kW"],
                    "r_bess_degradation": info["r_bess_degradation"],
                    "r_bess_invalid_action": info["r_bess_invalid_action"],
                    "storage_execution_clipping_kw": abs(
                        info["desired_bess_charge_power_kW"] - info["bess_charge_power_kW"])
                        + abs(info["desired_bess_discharge_power_kW"]
                              - info["bess_discharge_power_kW"]),
                    "correction_reason": info["correction_reason"],
                })
            print(f"historical seed={seed} origin={origin} "
                  f"repaired_soc={rows[-1]['final_soc']:.6f} "
                  f"fallbacks={rows[-1]['fallbacks']}", flush=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("historical checkpoint changed")
        sources.append({"seed": seed, "path": str(path.relative_to(ROOT)), "sha256": sha,
                        "parameter_updates": 0})
    report = {"historical_sources": sources, "same_state_storage_probes": probes,
              "reward_decision": "original reward retained pending short-run evidence",
              "scope": "historical_train_only_diagnostic", "origins": list(origins)}
    return rows, report, provenance


def calibrate(config, origins, run_id):
    rows, summaries, provenance = [], [], {}
    for compute in (1.0, .5, .25):
        business_sum, carbon_sum, transitions = 0.0, 0.0, 0
        for origin in origins:
            def action_fn(obs, value=compute):
                return np.array([value] * 20 + [0.0], dtype=np.float32)
            result, steps, _record, _inventory = evaluate_origin(
                config, origin, 0, action_fn, run_id=run_id)
            result["compute_value"] = compute
            rows.append(result)
            provenance[origin] = result["injection_provenance"]
            business_sum += sum(r["sla_violation_count"] for r in steps)
            carbon_sum += sum(r["carbon_emission"] for r in steps)
            transitions += result["steps"]
            print(f"calibration compute={compute} origin={origin} "
                  f"service={result['service_qualified']} soc={result['final_soc']:.6f}",
                  flush=True)
        summaries.append({"compute_value": compute,
                          "business_mean": business_sum / transitions,
                          "carbon_mean": carbon_sum / transitions,
                          "transitions": transitions})
    passed = all(r["episode_complete"] and r["service_qualified"]
                 and r["inventory_qualified"] and r["physical_violation_count"] == 0
                 and r["fallbacks"] == 0 for r in rows)
    budgets = calibrate_budgets(summaries)
    multipliers = multiplier_scaling(summaries, budgets)
    candidate = training_config_candidate(budgets, multipliers, 520)
    return rows, {"passed": passed, "scope": "train_only_calibration", "origins": list(origins),
                  "proposal_summaries": summaries, "candidate": candidate,
                  "asset_hashes": asset_hashes(), "reward_changed": False}, provenance


def freeze_calibration(config, folder, report):
    from scenario.inventory_release import CONFIG_PATH, MATRIX_PATH, SEMANTICS
    if report["passed"] is not True or len(report["origins"]) != 24:
        raise ValueError("cannot freeze unqualified or incomplete calibration")
    output = ROOT / CONFIG_PATH
    matrix_path = ROOT / MATRIX_PATH
    if output.exists() or matrix_path.exists():
        raise FileExistsError("versioned configuration/matrix already exist")
    candidate = report["candidate"]
    training = {k: v for k, v in candidate.items() if k not in ("schema", "status", "note")}
    frozen = copy.deepcopy(config)
    frozen.update(schema="idc-training-config-v2", version="v2",
                  note="M6-P2b train-only re-calibration; original environment reward retained",
                  training=training, reward_semantics="original-env-reward-v1")
    frozen["training"]["corrector"]["inventory_version"] = SEMANTICS
    frozen["training"]["corrector"]["horizon_policy"] = "real_episode_remainder"
    frozen["frozen_decision"]["card"] = "M6-P2b"
    frozen["calibration"] = {
        "run_id": folder.name,
        "command": json.loads((folder / "manifest.json").read_text())["command"],
        "split": "train", "origins": report["origins"], "asset_hashes": report["asset_hashes"],
        "run_manifest_path": str((folder / "manifest.json").relative_to(ROOT)),
        "run_manifest_sha256": hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest(),
        "run_report_path": str((folder / "report.json").relative_to(ROOT)),
        "run_report_sha256": hashlib.sha256((folder / "report.json").read_bytes()).hexdigest(),
    }
    frozen.pop("candidate_source", None)
    output.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n")
    old_path = ROOT / "configs/experiments/m9_experiment_matrix_v3.json"
    matrix = json.loads(old_path.read_text())
    matrix.update(schema="m9-experiment-matrix-v4", version="v4",
                  note="M6-P2b: unchanged experimental design; inventory training v2 binding",
                  supersedes={"logical_path": str(old_path.relative_to(ROOT)),
                              "sha256": hashlib.sha256(old_path.read_bytes()).hexdigest()})
    matrix["sources"].pop("training_config_v1")
    matrix["sources"]["training_config_v2"] = {
        "logical_path": CONFIG_PATH, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    matrix_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n")


def short_gate():
    from checkpointing.inventory_eval_input import SHORT_SCHEMA
    from safe_rl_v2.inventory_train import verify_written_run
    from scenario.inventory_release import checkpoint_binding
    binding = checkpoint_binding()
    sources, failed = [], []
    provenance = {}
    for seed in (0, 1, 2):
        folder = ROOT / f"runs/m6p2b_short_seed{seed}_v2"
        report = verify_written_run(folder, binding)
        manifest = json.loads((folder / "manifest.json").read_text())
        checkpoint = VersionedCheckpoint.load(
            folder / "checkpoint_final.pt", expected_action_dim=21, expected_obs_dim=520,
            expected_schema_hash=SHORT_SCHEMA)
        state = checkpoint.state
        checks = {
            "successful_manifest": manifest["status"] == "success",
            "controlled_scope": report["scope"] == "controlled_short_run",
            "eight_batches": report["batches"] == 8,
            "counts": (report["transitions"], report["adam_steps"],
                       report["lagrangian_updates"]) == (1536, 128, 8),
            "binding": report["inventory_binding"] == binding,
            "checkpoint": report["checkpoint_sha256"] == hashlib.sha256(
                (folder / "checkpoint_final.pt").read_bytes()).hexdigest(),
            "checkpoint_state": (
                checkpoint.extras == {"inventory_binding": binding}
                and state["next_batch_index"] == 8
                and state["training_scope"] == "controlled_short_run"
                and state["artifact_role"] == "controlled_training_resume"
                and state["source_ledger"]["master_seed"] == seed
                and {int(s["step"]) for s in state["optimizer"]["state"].values()} == {128}
                and state["lagrangian"]["updates"] == 8),
            "episodes": len(report["inventory_episodes"]) == 32,
            "service_physics_inventory": all(
                e["episode_complete"] and e["service_qualified"] is True
                and e["physical_violation_count"] == 0 and e["inventory_qualified"]
                for e in report["inventory_episodes"]),
            "no_fallback": all(r["zero_action_fallback_steps"] == 0
                               for r in report["batch_records"]),
            "storage_gradient": all(r["storage_signal"]["head_grad_norm_mean"] > 0
                                    for r in report["batch_records"]),
            "feasible_terminal_target": all(
                e["final_planning_audit"].get("target_reachable") is not True
                or e["target_qualified"] for e in report["inventory_episodes"]),
        }
        failed.extend([f"seed{seed}:{k}" for k, value in checks.items() if not value])
        sources.append({"seed": seed, "checks": checks, "run_id": folder.name,
                        "checkpoint_sha256": report["checkpoint_sha256"],
                        "source_scenario_hash": manifest["scenario_hash"],
                        "source_manifest_sha256": hashlib.sha256(
                            (folder / "manifest.json").read_bytes()).hexdigest(),
                        "report_sha256": hashlib.sha256(
                            (folder / "report.json").read_bytes()).hexdigest(),
                        "target_qualified_episodes": sum(e["target_qualified"]
                                                         for e in report["inventory_episodes"])})
        for batch in report["batch_records"]:
            provenance.update({int(k): v for k, v in batch["origin_provenance"].items()})
    return sources, {"passed": not failed, "failed_checks": failed, "short_runs": sources,
                     "inventory_binding": binding,
                     "reward_decision": "keep original reward" if not failed
                     else "short-run failures require repair before formal training"}, provenance


def audit(config, run_id):
    from checkpointing.inventory_eval_input import export_policy
    from checkpointing.versioned import read_checkpoint_payload
    from safe_rl_v2.inventory_train import verify_written_run
    from scenario.inventory_release import checkpoint_binding, load_matrix
    matrix, binding = load_matrix(), checkpoint_binding()
    origins = matrix["training_schedule"]["origin_pool"]
    if len(origins) != 212:
        raise ValueError("train-only audit requires the frozen 212 origin pool")
    folder = ROOT / "runs" / run_id
    folder.mkdir(parents=True)
    rows, provenance, sources = [], {}, []
    for seed in (0, 1, 2):
        source_folder = ROOT / f"runs/m6p2b_formal_seed{seed}_v2"
        source = source_folder / "checkpoint_final.pt"
        report = verify_written_run(source_folder, binding)
        manifest = json.loads((source_folder / "manifest.json").read_text())
        state = read_checkpoint_payload(source)["state"]
        if (manifest["status"] != "success" or report["scope"] != "formal_training"
                or report["inventory_binding"] != binding or report["seed"] != seed
                or (report["batches"], report["transitions"], report["adam_steps"],
                    report["lagrangian_updates"]) != (512, 98304, 8192, 512)
                or state["next_batch_index"] != 512
                or {int(s["step"]) for s in state["optimizer"]["state"].values()} != {8192}
                or state["lagrangian"]["updates"] != 512
                or state["source_ledger"]["master_seed"] != seed
                or report["checkpoint_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest()):
            raise ValueError(f"seed {seed} formal artifact review failed")
        policy = export_policy(source, folder / f"policy_seed{seed}.pt")
        for origin in origins:
            def action_fn(obs, active_policy=policy):
                return deterministic_action(active_policy, obs)
            result, _steps, _record, _inventory = evaluate_origin(
                config, origin, 0, action_fn, run_id=run_id)
            result["policy_seed"] = seed
            result["simultaneously_qualified"] = (result["service_qualified"]
                                                  and result["inventory_qualified"])
            rows.append(result)
            provenance[int(origin)] = result["injection_provenance"]
            if len(rows) % 12 == 0:
                print(f"audit completed={len(rows)}/636 service={result['service_qualified']} "
                      f"soc={result['final_soc']:.6f}", flush=True)
        sources.append({"seed": seed, "run_id": source_folder.name,
                        "checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "report_sha256": hashlib.sha256(
                            (source_folder / "report.json").read_bytes()).hexdigest()})
    counts = {key: sum(r[key] is True for r in rows) for key in (
        "episode_complete", "service_qualified", "inventory_qualified", "target_qualified",
        "simultaneously_qualified")}
    counts["physics_qualified"] = sum(r["physical_violation_count"] == 0 for r in rows)
    missing = [m["method_id"] for m in matrix["methods"]
               if m["method_id"] != "safe_ppo_joint_rolling_corrector"]
    # Cost is deliberately absent from readiness. Unresolved failures keep this gate closed.
    ready = all(counts[k] == 636 for k in (
        "episode_complete", "simultaneously_qualified", "physics_qualified"))
    return rows, {"scope": "train_only_212x3_artifact_review", "formal_sources": sources,
                  "inventory_binding": binding, "episodes": len(rows),
                  "qualification_counts": counts,
                  "qualification_rates": {k: v / 636 for k, v in counts.items()},
                  "validation_readiness": ready, "cost_threshold_used": False,
                  "missing_method_seats": missing, "five_method_comparison_ready": False,
                  "fair_pairing_terminal_difference_kwh_max": 1e-6,
                  "failure_rows": [r for r in rows if not r["simultaneously_qualified"]
                                   or r["physical_violation_count"]]}, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("replay", "calibrate", "release", "gate", "audit"),
                        required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--origins-limit", type=int)
    args = parser.parse_args(argv)
    if (ROOT / "runs" / args.run_id).exists():
        raise FileExistsError("run-id exists; historical and failed runs are never overwritten")
    config = load_frozen_training_config()
    origins = select_origins()
    if args.origins_limit is not None:
        if args.phase != "replay" or not 1 <= args.origins_limit <= len(origins):
            raise ValueError("origin limit is only allowed for explicitly diagnostic replay")
        origins = origins[:args.origins_limit]
    command = "python -m scripts.m6p2b_inventory_repair " + " ".join(sys.argv[1:])
    provenance = {}
    try:
        if args.phase == "replay":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rows, report, provenance = replay(config, origins, args.run_id)
        elif args.phase == "calibrate":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rows, report, provenance = calibrate(config, origins, args.run_id)
        elif args.phase == "release":
            from scenario.inventory_release import RELEASE_PATH, build_release
            release = build_release()
            destination = ROOT / RELEASE_PATH
            if destination.exists():
                raise FileExistsError("inventory release v2 already exists")
            destination.write_text(json.dumps(release, ensure_ascii=False, indent=2) + "\n")
            rows, report = [], {"release": release}
            # Actual train-origin provenance for the release validation probe.
            _, injection = build_train_env(origins[0], master_seed=0, config=config)
            provenance = {origins[0]: injection.provenance_hash}
        elif args.phase == "gate":
            rows, report, provenance = short_gate()
        else:
            from scenario.inventory_release import load_config
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rows, report, provenance = audit(load_config(), args.run_id)
        ledger = training_source_ledger(provenance)
        if args.phase == "gate":
            ledger["scenario_hash"] = hashlib.sha256(json.dumps(
                [[s["seed"], s["source_scenario_hash"]] for s in rows],
                sort_keys=True).encode()).hexdigest()
            report["scenario_hash_scope"] = "sha256 of ordered seed/source scenario hashes"
        report.update(validation_run=False, test_run=False, parameter_updates=0)
        folder = write_run(args.run_id, config={
                  "phase": args.phase, "origins": list(origins),
                  "environment_training_config": config,
                  "inventory_semantics": "terminal-inventory-v1",
                  "reward_semantics": "original-env-reward-v1",
                  "global_corrector_budget_s": .25},
                  metrics=pd.DataFrame(rows), report=report, base_dir=str(ROOT / "runs"),
                  command=command, status="success" if report.get("passed", True) else "failed",
                  failure_classification=(None if report.get("passed", True)
                                          else "acceptance_failed"),
                  **ledger)
        if args.phase == "calibrate" and report["passed"]:
            freeze_calibration(config, folder, report)
        if report.get("passed", True) is False:
            return 1
        return 0
    except Exception as exc:
        ledger = {"dependency_lock_hash": hashlib.sha256(
                    (ROOT / "uv.lock").read_bytes()).hexdigest(),
                  "data_hash": None, "scenario_hash": None}
        if provenance:
            ledger = training_source_ledger(provenance)
        write_run(args.run_id, config={"phase": args.phase, "origins": list(origins)},
                  metrics=pd.DataFrame(), report={"failure": f"{type(exc).__name__}: {exc}"},
                  base_dir=str(ROOT / "runs"), command=command, status="failed",
                  failure_classification=type(exc).__name__, **ledger)
        raise


if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(main())

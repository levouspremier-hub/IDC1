"""M6-P2b train-only terminal inventory diagnostics; never evaluates held-out data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
import warnings
from pathlib import Path

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
from scripts.calibrate_training_config import select_origins

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
        else:
            raise NotImplementedError(
                f"phase {args.phase} not yet implemented; long training blocked")
        ledger = training_source_ledger(provenance)
        report.update(validation_run=False, test_run=False, parameter_updates=0)
        write_run(args.run_id, config={"phase": args.phase, "origins": list(origins)},
                  metrics=pd.DataFrame(rows), report=report, base_dir=str(ROOT / "runs"),
                  command=command, **ledger)
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

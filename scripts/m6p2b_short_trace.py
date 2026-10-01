"""Read-only train rollout replay with full service and solver failure evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from unittest.mock import patch

import pandas as pd

import safe_rl.corrector_wrapper as wrapper
import safe_rl_v2.formal_train_loop as loop
from checkpointing.inventory_eval_input import SHORT_SCHEMA
from checkpointing.versioned import read_checkpoint_payload
from evaluation.metrics import classify_tasks
from runs.writer import write_run
from safe_rl_v2.controlled_formal_train import _generators, apply_frozen_thread_setting
from safe_rl_v2.formal_train import preregistered_batches
from safe_rl_v2.inventory_train import verify_written_run
from safe_rl_v2.rollout import _json_safe
from scenario.inventory_release import ROOT, checkpoint_binding, load_config, load_matrix, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--batch-index", type=int, choices=(0, 2), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    folder = ROOT / "runs" / args.run_id
    if folder.exists():
        raise FileExistsError("trace run already exists")
    config, matrix, binding = load_config(), load_matrix(), checkpoint_binding()
    apply_frozen_thread_setting(config)
    source = ROOT / f"runs/m6p2b_short_seed{args.seed}_v2"
    reference = verify_written_run(source, binding)["batch_records"][args.batch_index]
    policy = loop.build_seeded_policy(config, obs_dim=520, seed=args.seed)
    sampling, shuffle = _generators(args.seed)
    checkpoint = None
    if args.batch_index == 2:
        checkpoint = source / "checkpoint_batch2.pt"
        if read_checkpoint_payload(checkpoint)["metadata"].get("inventory_binding") != binding:
            raise ValueError("trace checkpoint binding differs")
        restored = loop.load_resume_checkpoint(
            checkpoint, policy=policy, optimizer=loop.build_optimizer(config, policy),
            lagrangian=loop.build_lagrangian(config), sampling_generator=sampling,
            shuffle_generator=shuffle, config=config, expected_obs_dim=520,
            expected_schema=SHORT_SCHEMA, expected_scope="controlled_short_run",
            expected_role="controlled_training_resume")
        if restored["next_batch_index"] != 2:
            raise ValueError("trace restore boundary differs")
    environments, snapshots, infos = [], [], []
    original_build, original_snapshot, original_step = (
        loop.build_train_env, wrapper.build_snapshot, wrapper.CorrectorWrapper.step)

    def build(*a, **kw):
        result = original_build(*a, **kw)
        environments.append(result[0])
        return result

    def snapshot(env):
        result = original_snapshot(env)
        snapshots.append(result)
        return result

    def step(self, action):
        result = original_step(self, action)
        infos.append(result[-1])
        return result

    origins = preregistered_batches(matrix, args.seed)[args.batch_index]
    with patch.object(loop, "build_train_env", build), patch.object(
            wrapper, "build_snapshot", snapshot), patch.object(
            wrapper.CorrectorWrapper, "step", step):
        buffer, provenance, timing = loop._collect_with_config(
            policy, sampling, origins=origins, horizon=48, env_seed=args.seed,
            master_seed=args.seed, config=config, corrector_time_limit_s=.25)
    digest = loop._digest_arrays([
        loop._stack(buffer, field).numpy()
        for field in ("observation", "raw_action", "old_raw_log_prob")])
    service = []
    for origin, env in zip(origins, environments, strict=True):
        metrics, missing = classify_tasks(
            env.tasks, horizon=48,
            non_interruptible_interruptions=env.total_non_interruptible_interruption_count)
        service.append({"origin": origin, "metrics": metrics.model_dump(),
                        "missing": missing})
    rows = [{"origin": origins[i // 48], "step": i % 48, **_json_safe(info)}
            for i, info in enumerate(infos)]
    failures = [{"origin": origins[i // 48], "step": i % 48,
                 "snapshot": snap.model_dump(), "info": rows[i]}
                for i, snap in enumerate(snapshots)
                if infos[i]["correction_reason"] in (
                    "timeout", "solver_failure", "proposal_invalid", "base_shortage")]
    report = {
        "scope": "train_only_read_only_short_trace", "parameter_updates": 0,
        "seed": args.seed, "batch_index": args.batch_index, "origins": origins,
        "inventory_binding": binding, "source_report_sha256": sha(source / "report.json"),
        "checkpoint_sha256": sha(checkpoint) if checkpoint else None,
        "instrumentation_sha256": sha(__file__),
        "transition_digest": digest, "reference_digest": reference["batch_transition_digest"],
        "exact_transition_replay": digest == reference["batch_transition_digest"],
        "correction_counts": dict(Counter(r["correction_reason"] for r in rows)),
        "service": service, "inventory_episodes": timing["inventory_episodes"],
        "failure_snapshots": failures, "steps": rows,
        "validation_run": False, "test_run": False,
    }
    write_run(
        args.run_id, config=config,
        metrics=pd.DataFrame([{k: r[k] for k in (
            "origin", "step", "correction_reason", "correction_solve_time_s",
            "stage_a_status", "stage_b_status", "stage_a_solve_time_s",
            "stage_b_solve_time_s")} for r in rows]), report=report,
        seed=args.seed, command=f"python -m scripts.m6p2b_short_trace --seed {args.seed} "
        f"--batch-index {args.batch_index} --run-id {args.run_id}",
        base_dir=str(ROOT / "runs"), **loop.training_source_ledger(provenance))
    reread = json.loads((folder / "report.json").read_text())
    if reread != report or len(pd.read_parquet(folder / "metrics.parquet")) != 192:
        raise ValueError("trace read-back differs")
    print(json.dumps({k: report[k] for k in (
        "exact_transition_replay", "correction_counts", "service")}, indent=2), flush=True)


if __name__ == "__main__":
    main()

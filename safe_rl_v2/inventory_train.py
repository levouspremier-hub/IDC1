"""M6-P2b v2 train-only entry: 8 controlled batches or 512 formal batches."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
import yaml

from checkpointing import VersionedCheckpoint
from checkpointing.inventory_eval_input import FORMAL_SCHEMA, SHORT_SCHEMA
from checkpointing.versioned import read_checkpoint_payload
from runs.writer import git_revision, write_run
from safe_rl_v2.controlled_formal_train import _generators, apply_frozen_thread_setting
from safe_rl_v2.formal_train import preregistered_batches
from safe_rl_v2.formal_train_loop import (
    build_lagrangian,
    build_optimizer,
    build_seeded_policy,
    load_resume_checkpoint,
    run_training_batch,
    save_resume_checkpoint,
    training_source_ledger,
)
from scenario.inventory_release import (
    CONFIG_PATH,
    ROOT,
    checkpoint_binding,
    load_config,
    load_matrix,
    sha,
)


def save_bound(path, *, binding, **kwargs):
    target = Path(path)
    temporary = target.with_suffix(".tmp.pt")
    save_resume_checkpoint(temporary, **kwargs)
    payload = read_checkpoint_payload(temporary)
    payload["metadata"]["inventory_binding"] = binding
    torch.save(payload, temporary)
    os.replace(temporary, target)
    # Read the complete written artifact; never infer successful save from counters.
    read = VersionedCheckpoint.load(
        target, expected_action_dim=21, expected_obs_dim=520,
        expected_schema_hash=kwargs["schema"])
    if read.extras != {"inventory_binding": binding}:
        raise ValueError("checkpoint binding failed after writing")


def require_short_gate(binding):
    gate = ROOT / "runs/m6p2b_short_gate_v2"
    manifest = json.loads((gate / "manifest.json").read_text())
    report = json.loads((gate / "report.json").read_text())
    if (manifest["status"] != "success" or report.get("passed") is not True
            or report.get("inventory_binding") != binding):
        raise ValueError("three-seed short-run gate must pass before formal training")


def run(args):
    config, matrix = load_config(), load_matrix()
    binding = checkpoint_binding()
    apply_frozen_thread_setting(config)
    if not args.short:
        require_short_gate(binding)
    batches = preregistered_batches(matrix, args.seed)
    if args.short:
        batches = batches[:8]
    scope = "controlled_short_run" if args.short else "formal_training"
    role = "controlled_training_resume" if args.short else "formal_training_resume"
    schema = SHORT_SCHEMA if args.short else FORMAL_SCHEMA
    folder = ROOT / "runs" / args.run_id
    if folder.exists():
        raise FileExistsError("run-id already exists; choose a new run-id")
    folder.mkdir(parents=True)
    policy = build_seeded_policy(config, obs_dim=520, seed=args.seed)
    optimizer = build_optimizer(config, policy)
    lagrangian = build_lagrangian(config)
    sampling, shuffle = _generators(args.seed)
    records, provenance = [], {}
    next_batch = 0
    if args.resume_from:
        payload = read_checkpoint_payload(args.resume_from)
        if payload["metadata"].get("inventory_binding") != binding:
            raise ValueError("resume requires matching v2 inventory/reward/release semantics")
        state = payload["state"]
        if (state["source_ledger"]["master_seed"] != args.seed
                or state["origins"] != [o for group in batches for o in group]
                or not 0 <= state["next_batch_index"] <= len(batches)):
            raise ValueError("resume seed/order/boundary differs from registered schedule")
        restored = load_resume_checkpoint(
            args.resume_from, policy=policy, optimizer=optimizer, lagrangian=lagrangian,
            sampling_generator=sampling, shuffle_generator=shuffle, config=config,
            expected_obs_dim=520, expected_schema=schema, expected_scope=scope,
            expected_role=role)
        next_batch = restored["next_batch_index"]
        provenance = restored["origin_provenance"]
        journal = Path(args.resume_from).parent / "batches.jsonl"
        records = [json.loads(line) for line in journal.read_text().splitlines()
                   if json.loads(line)["batch_index"] < next_batch]
        if len(records) != next_batch:
            raise ValueError("resume journal does not match the checkpoint boundary")
    command = "python -m safe_rl_v2.inventory_train " + " ".join(sys.argv[1:])
    started = time.perf_counter()
    checkpoint_kwargs = {
        "policy": policy, "optimizer": optimizer, "lagrangian": lagrangian,
        "sampling_generator": sampling, "shuffle_generator": shuffle, "config": config,
        "origins": [o for group in batches for o in group], "obs_dim": 520,
        "master_seed": args.seed, "code_revision": git_revision(),
        "training_scope": scope, "artifact_role": role, "schema": schema,
    }
    try:
        for index in range(next_batch, len(batches)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                batch = run_training_batch(
                    policy, optimizer, lagrangian, sampling, shuffle, config=config,
                    origins=batches[index], batch_index=index, env_seed=args.seed,
                    corrector_time_limit_s=.25, master_seed=args.seed)
            provenance.update({int(k): v for k, v in batch["origin_provenance"].items()})
            records.append(batch)
            (folder / "batches.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records))
            save_bound(folder / "checkpoint_latest.pt", binding=binding,
                       origin_provenance=provenance, next_batch_index=index + 1,
                       **checkpoint_kwargs)
            if args.short and index == 1:
                save_bound(folder / "checkpoint_batch2.pt", binding=binding,
                           origin_provenance=provenance, next_batch_index=2,
                           **checkpoint_kwargs)
            if args.short or (index + 1) % 16 == 0:
                episodes = batch["inventory_episodes"]
                print(f"seed={args.seed} batch={index+1}/{len(batches)} "
                      f"inventory={sum(e['inventory_qualified'] for e in episodes)}/4 "
                      f"service={sum(e['service_qualified'] is True for e in episodes)}/4 "
                      f"fallbacks={batch['zero_action_fallback_steps']} "
                      f"elapsed={time.perf_counter()-started:.0f}s", flush=True)
        save_bound(folder / "checkpoint_final.pt", binding=binding,
                   origin_provenance=provenance, next_batch_index=len(batches),
                   **checkpoint_kwargs)
        last = records[-1]
        episodes = [{"batch_index": r["batch_index"], **e}
                    for r in records for e in r["inventory_episodes"]]
        report = {
            "scope": scope, "seed": args.seed, "batches": len(records),
            "episodes": len(episodes), "transitions": sum(r["transitions"] for r in records),
            "adam_steps": last["optimizer_steps_cumulative"],
            "lagrangian_updates": last["lagrangian_updates_cumulative"],
            "batch_records": records, "inventory_episodes": episodes,
            "inventory_binding": binding, "config_path": CONFIG_PATH,
            "batch_origin_list_digest": matrix["training_schedule"]["batch_origin_list_digest"],
            "checkpoint_sha256": sha(folder / "checkpoint_final.pt"),
            "elapsed_s": time.perf_counter() - started,
            "claims": {"convergence_claimed": False, "performance_evaluated": False},
            "validation_run": False, "test_run": False,
        }
        metrics = pd.DataFrame([{
            "batch_index": r["batch_index"], "transitions": r["transitions"],
            "adam_steps": r["optimizer_steps_cumulative"],
            "lagrangian_updates": r["lagrangian_updates_cumulative"],
            "fallbacks": r["zero_action_fallback_steps"],
            **r["storage_signal"],
        } for r in records])
        ledger = training_source_ledger(provenance)
        write_run(args.run_id, config=config, metrics=metrics, report=report,
                  base_dir=str(ROOT / "runs"), seed=args.seed, command=command, **ledger)
        # Complete finalization acceptance: reopen all required artifacts.
        reread_manifest = json.loads((folder / "manifest.json").read_text())
        if (len(pd.read_parquet(folder / "metrics.parquet")) != len(records)
                or json.loads((folder / "report.json").read_text()) != report
                or yaml.safe_load((folder / "config.yaml").read_text()) != config
                or not (folder / "figures").is_dir()
                or reread_manifest["status"] != "success"
                or any(reread_manifest[k] != v for k, v in ledger.items())):
            raise ValueError("final artifact read-back verification failed")
        return 0
    except Exception as exc:
        ledger = training_source_ledger(provenance) if provenance else {
            "dependency_lock_hash": sha(ROOT / "uv.lock"),
            "data_hash": None, "scenario_hash": None}
        write_run(args.run_id, config=config, metrics=pd.DataFrame(),
                  report={"failure": f"{type(exc).__name__}: {exc}",
                          "batches_completed": len(records), "inventory_binding": binding},
                  base_dir=str(ROOT / "runs"), seed=args.seed, command=command,
                  status="failed", failure_classification=type(exc).__name__, **ledger)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short", action="store_true")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-from")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

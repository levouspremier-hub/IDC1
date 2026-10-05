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
        target, expected_action_dim=21, expected_obs_dim=kwargs["obs_dim"],
        expected_schema_hash=kwargs["schema"])
    if read.extras != {"inventory_binding": binding}:
        raise ValueError("checkpoint binding failed after writing")


def require_short_gate(binding):
    from scenario.inventory_release import RUN_REVISION
    gate = ROOT / f"runs/m6p2b_short_gate_{RUN_REVISION}"
    manifest = json.loads((gate / "manifest.json").read_text())
    report = json.loads((gate / "report.json").read_text())
    if (manifest["status"] != "success" or report.get("passed") is not True
            or report.get("inventory_binding") != binding
            or report.get("formal_three_seed_gate") is not True):
        raise ValueError("three-seed short-run gate must pass before formal training")
    from scripts.m6p2b_inventory_repair import short_gate
    sources, live, _ = short_gate()
    if live["passed"] is not True or report["short_runs"] != sources:
        raise ValueError("live short-run artifacts no longer satisfy the recorded gate")



def require_seed0_launch_gate(binding, *, seed):
    """User-authorized seed 0 only; inherit verified r6 qualification, never weights."""
    if seed != 0:
        raise ValueError("single-seed formal authorization is restricted to seed 0")
    from safe_rl_v2.inventory_diagnostics import inventory_episode_acceptance
    from scenario.inventory_release import verify_release

    authorization = verify_release()["seed0_formal_authorization"]
    gate_folder = ROOT / authorization["gate_path"]
    for name, digest in authorization["gate_hashes"].items():
        if sha(gate_folder / name) != digest:
            raise ValueError("seed0 gate evidence hash mismatch")
    gate = json.loads((gate_folder / "report.json").read_text())
    old_binding = gate["inventory_binding"]
    # Only launch code and its release declaration changed. Physics, reward,
    # optimizer, policy, buffer, config and matrix must be byte-identical.
    ignored = {"release_path", "release_sha256", "semantics_binding"}
    if ({k: v for k, v in binding.items() if k not in ignored}
            != {k: v for k, v in old_binding.items() if k not in ignored}):
        raise ValueError("seed0 inherited configuration/reward binding differs")
    before, after = old_binding["semantics_binding"], binding["semantics_binding"]
    allowed = {"safe_rl_v2/inventory_train.py", "scenario/inventory_release.py"}
    if (set(before) != set(after)
            or any(before[p] != after[p] for p in before if p not in allowed)):
        raise ValueError("seed0 inheritance changed training/planning semantics")
    if (gate.get("passed") is not True or gate.get("seeds") != [0]
            or gate.get("formal_three_seed_gate") is not False
            or len(gate["short_runs"]) != 1):
        raise ValueError("seed0 requires qualified single-seed evidence")
    source = gate["short_runs"][0]
    folder = ROOT / "runs" / source["run_id"]
    report = verify_written_run(folder, old_binding)
    if (source["seed"] != 0 or not all(source["checks"].values())
            or sha(folder / "report.json") != source["report_sha256"]
            or sha(folder / "manifest.json") != source["source_manifest_sha256"]
            or sha(folder / "checkpoint_final.pt") != source["checkpoint_sha256"]
            or report["seed"] != 0 or report["scope"] != "controlled_short_run"
            or report["batches"] != 8 or report["transitions"] != 1536
            or report["adam_steps"] != 128 or report["lagrangian_updates"] != 8
            or len(report["inventory_episodes"]) != 32
            or not all(all(inventory_episode_acceptance(e).values())
                       for e in report["inventory_episodes"])
            or not all(b["zero_action_fallback_steps"] == 0
                       and b["storage_signal"]["head_grad_norm_mean"] > 0
                       for b in report["batch_records"])):
        raise ValueError("live seed0 short-run artifacts no longer qualify")
    return authorization


def verify_written_run(folder, binding):
    folder = Path(folder)
    receipt = json.loads((folder / "artifact_verification.json").read_text())
    if (receipt["inventory_binding"] != binding or receipt["passed"] is not True
            or set(receipt["hashes"]) != {"config.yaml", "metrics.parquet", "report.json",
                                         "manifest.json", "checkpoint_final.pt"}
            or any(sha(folder / path) != digest for path, digest in receipt["hashes"].items())
            or not (folder / "figures").is_dir()):
        raise ValueError("written run artifact receipt does not match live files")
    return json.loads((folder / "report.json").read_text())


def _run(args):
    candidate_path = getattr(args, "runtime_candidate", None)
    runtime_release_path = getattr(args, "runtime_release", None)
    if candidate_path and runtime_release_path:
        raise ValueError("select either a runtime candidate or a qualified runtime release")
    if runtime_release_path:
        from scenario.runtime_release import (
            runtime_checkpoint_binding,
            verify_candidate,
            verify_runtime_release,
        )
        release = verify_runtime_release(runtime_release_path)
        if args.short or args.seed not in release["allowed_seeds"]:
            raise ValueError("runtime formal launch requires a release-authorized seed; "
                             "old resume bindings are rejected")
        candidate_path = str(ROOT / release["candidate_path"])
        config = verify_candidate(candidate_path)["config"]
        matrix = load_matrix()
        binding = runtime_checkpoint_binding(runtime_release_path)
    elif candidate_path:
        from scenario.runtime_release import (
            candidate_binding,
            require_candidate_scope,
            verify_candidate,
        )
        require_candidate_scope(short=args.short)
        config = verify_candidate(candidate_path)["config"]
        matrix = load_matrix()
        binding = candidate_binding(candidate_path)
    else:
        config, matrix = load_config(), load_matrix()
        binding = checkpoint_binding()
    apply_frozen_thread_setting(config)
    seed0_formal = getattr(args, "seed0_formal", False)
    if seed0_formal and (args.short or args.seed != 0):
        raise ValueError("seed0 formal authorization requires seed 0 and full training")
    authorization = None
    if not args.short:
        if runtime_release_path:
            authorization = release
        elif seed0_formal:
            authorization = require_seed0_launch_gate(binding, seed=args.seed)
        else:
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
    obs_dim = int(config["training"]["policy"]["obs_dim"])
    policy = build_seeded_policy(config, obs_dim=obs_dim, seed=args.seed)
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
            expected_obs_dim=obs_dim, expected_schema=schema, expected_scope=scope,
            expected_role=role)
        next_batch = restored["next_batch_index"]
        provenance = restored["origin_provenance"]
        journal = Path(args.resume_from).parent / "batches.jsonl"
        records = ([json.loads(line) for line in journal.read_text().splitlines()
                    if json.loads(line)["batch_index"] < next_batch] if next_batch else [])
        if len(records) != next_batch:
            raise ValueError("resume journal does not match the checkpoint boundary")
    command = "python -m safe_rl_v2.inventory_train " + " ".join(sys.argv[1:])
    started = time.perf_counter()
    checkpoint_kwargs = {
        "policy": policy, "optimizer": optimizer, "lagrangian": lagrangian,
        "sampling_generator": sampling, "shuffle_generator": shuffle, "config": config,
        "origins": [o for group in batches for o in group], "obs_dim": obs_dim,
        "master_seed": args.seed, "code_revision": git_revision(),
        "training_scope": scope, "artifact_role": role, "schema": schema,
    }
    try:
        write_run(args.run_id, config=config, metrics=pd.DataFrame([
                      {"batch_index": r["batch_index"], "transitions": r["transitions"],
                       **r["storage_signal"]} for r in records]),
                  report={"status": "running", "scope": scope, "seed": args.seed,
                          "target_batches": len(batches), "completed_batches": next_batch,
                          "inventory_binding": binding,
                          "formal_launch_authorization": authorization},
                  base_dir=str(ROOT / "runs"), seed=args.seed, command=command,
                  status="running", dependency_lock_hash=sha(ROOT / "uv.lock"))
        for index in range(next_batch, len(batches)):
            save_bound(folder / "checkpoint_before_batch.pt", binding=binding,
                       origin_provenance=provenance, next_batch_index=index,
                       **checkpoint_kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                batch = run_training_batch(
                    policy, optimizer, lagrangian, sampling, shuffle, config=config,
                    origins=batches[index], batch_index=index, env_seed=args.seed,
                    corrector_time_limit_s=float(config["training"]["corrector"]["time_limit_s"]),
                    master_seed=args.seed)
            provenance.update({int(k): v for k, v in batch["origin_provenance"].items()})
            records.append(batch)
            (folder / "batches.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records))
            save_bound(folder / "checkpoint_latest.pt", binding=binding,
                       origin_provenance=provenance, next_batch_index=index + 1,
                       **checkpoint_kwargs)
            if (index + 1) % 16 == 0:
                save_bound(folder / f"checkpoint_batch_{index + 1:04d}.pt", binding=binding,
                           origin_provenance=provenance, next_batch_index=index + 1,
                           **checkpoint_kwargs)
            if args.short and index == 1:
                save_bound(folder / "checkpoint_batch2.pt", binding=binding,
                           origin_provenance=provenance, next_batch_index=2,
                           **checkpoint_kwargs)
            if index == 0 or (index + 1) % 16 == 0:
                write_run(args.run_id, config=config,
                          metrics=pd.DataFrame([{
                              "batch_index": r["batch_index"],
                              "transitions": r["transitions"],
                              **r["storage_signal"]} for r in records]),
                          report={"status": "running", "scope": scope, "seed": args.seed,
                                  "target_batches": len(batches),
                                  "completed_batches": index + 1,
                                  "transitions": sum(r["transitions"] for r in records),
                                  "inventory_binding": binding,
                                  "formal_launch_authorization": authorization,
                                  "checkpoint_latest_sha256": sha(folder / "checkpoint_latest.pt")},
                          base_dir=str(ROOT / "runs"), seed=args.seed, command=command,
                          status="running", **training_source_ledger(provenance))
            if args.short or index == 0 or (index + 1) % 16 == 0:
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
            "formal_launch_authorization": authorization,
            "batches_newly_run": len(batches) - next_batch,
            "resumed_from": args.resume_from,
            "resume_source_sha256": sha(args.resume_from) if args.resume_from else None,
            "episodes": len(episodes), "transitions": sum(r["transitions"] for r in records),
            "adam_steps": last["optimizer_steps_cumulative"],
            "lagrangian_updates": last["lagrangian_updates_cumulative"],
            "batch_records": records, "inventory_episodes": episodes,
            "inventory_binding": binding, "config_path": candidate_path or CONFIG_PATH,
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
        (folder / "artifact_verification.json").write_text(json.dumps({
            "passed": True, "inventory_binding": binding,
            "hashes": {path: sha(folder / path) for path in (
                "config.yaml", "metrics.parquet", "report.json", "manifest.json",
                "checkpoint_final.pt")}}, indent=2) + "\n")
        verify_written_run(folder, binding)
        return 0
    except Exception as exc:
        evidence = dict(getattr(exc, "evidence", {"reason": str(exc)}))
        evidence["adam_steps_at_failure"] = max(
            (int(s["step"]) for s in optimizer.state.values() if "step" in s), default=0)
        evidence["multiplier_updates_at_failure"] = int(getattr(lagrangian, "_updates", 0))
        (folder / "failed_batch.json").write_text(json.dumps(evidence, indent=2) + "\n")
        provenance.update({int(k): v for k, v in evidence.get("origin_provenance", {}).items()})
        ledger = training_source_ledger(provenance) if provenance else {
            "dependency_lock_hash": sha(ROOT / "uv.lock"),
            "data_hash": None, "scenario_hash": None}
        write_run(args.run_id, config=config, metrics=pd.DataFrame([
                      {"batch_index": r["batch_index"], "transitions": r["transitions"],
                       **r["storage_signal"]} for r in records]),
                  report={"failure": f"{type(exc).__name__}: {exc}",
                          "failed_batch_evidence": "failed_batch.json",
                          "last_safe_checkpoint": "checkpoint_before_batch.pt",
                          "batches_completed": len(records), "inventory_binding": binding},
                  base_dir=str(ROOT / "runs"), seed=args.seed, command=command,
                  status="failed", failure_classification=type(exc).__name__, **ledger)
        raise


def run(args):
    folder = ROOT / "runs" / args.run_id
    if folder.exists():
        raise FileExistsError("run-id already exists; choose a new run-id")
    try:
        return _run(args)
    except Exception as exc:
        # Preflight/recovery errors also keep the five required artifacts.
        # Unknown actual-use hashes stay null; do not invent injected scenarios.
        if not (folder / "manifest.json").exists():
            write_run(args.run_id, config={"short": args.short, "seed": args.seed},
                      metrics=pd.DataFrame(), base_dir=str(ROOT / "runs"), seed=args.seed,
                      command="python -m safe_rl_v2.inventory_train " + " ".join(sys.argv[1:]),
                      report={"failure": f"{type(exc).__name__}: {exc}",
                              "failure_phase": "preflight_or_restore", "rollout_started": False},
                      status="failed", failure_classification=type(exc).__name__,
                      dependency_lock_hash=sha(ROOT / "uv.lock"))
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short", action="store_true")
    parser.add_argument("--seed0-formal", action="store_true",
                        help="explicit seed0-only launch using verified inherited qualification")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--runtime-candidate", help="signed train-only qualification candidate")
    parser.add_argument("--runtime-release", help="frozen host-qualified runtime release")
    args = parser.parse_args(argv)
    if args.seed0_formal and (args.short or args.seed != 0):
        parser.error("--seed0-formal requires --seed 0 and full training")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

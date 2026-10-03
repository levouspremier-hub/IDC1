"""Preregistered train-only final-policy diagnosis; no training updates.

Runtime observers preserve release-bound source bytes and call arguments/results.
Timing includes observer overhead; no profiler, tracing, or model-line callbacks.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from checkpointing.inventory_eval_input import export_policy, load_policy
from evaluation.controlled_run import deterministic_action
from runs.writer import write_run
from safe_rl_v2.controlled_formal_train import apply_frozen_thread_setting
from safe_rl_v2.formal_train_loop import peak_rss_bytes, training_source_ledger
from safe_rl_v2.inventory_diagnostics import evaluate_origin, inventory_episode_acceptance
from scenario.inventory_release import load_config, semantics_binding, sha, verify_release

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/m6p2b_formal_train_seed0_v2_r7"
CHECKPOINT_SHA = "472b42467448a8c3ebe00a57a4f44000aea7a98574b43ba3316d14eded5ff935"
ORIGINS = (5040, 5088, 5136, 5184, 5568, 6576)
MODES = ("deterministic", "sample_0", "sample_1", "sample_2")
FALLBACKS = ("timeout", "solver_failure", "proposal_invalid", "base_shortage")


def case_grid():
    return [(origin, mode) for origin in ORIGINS for mode in MODES]


def validate_case(origin, mode):
    if origin not in ORIGINS or mode not in MODES:
        raise ValueError("Only preregistered train dates and action modes are allowed")


def assert_unchanged(before, after, name):
    if before != after:
        raise ValueError(f"{name} changed during fixed-policy diagnosis")


def parameter_hash(policy):
    h = hashlib.sha256()
    for key, value in sorted(policy.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def failure_stage(result, events):
    if result.failure_class not in FALLBACKS:
        return "none"
    if not events:
        return "before_first_solver_or_between_stages"
    last = events[-1]
    return last["stage"] if last["status"] != 0 else "after_" + last["stage"]


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    return value


def dump(path, value):
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False))


def forecast_evidence(snapshot):
    bundle = snapshot.forecast
    if bundle.split != "train" or not bundle.forecast_provenance:
        raise ValueError("diagnosis requires train forecast provenance")
    return {
        "split": bundle.split, "generated_at": jsonable(bundle.generated_at),
        "sources": jsonable(bundle.forecast_provenance),
        "start": bundle.start, "forecast_cutoff": bundle.forecast_cutoff,
        "visible_mask": snapshot.planning_forecast.visible_mask,
        "assumed_mask": snapshot.planning_forecast.assumed_mask,
        "extension_policy": snapshot.planning_forecast.extension_policy,
    }


class Observer:
    """Small function wrappers only; serialization occurs after correct() returns."""

    def __init__(self, folder):
        self.folder = folder
        self.rows = []
        self.previous = None
        self.first_failure = False
        self.current = {}
        self.snapshot = None
        self.proposal = None

    @contextlib.contextmanager
    def installed(self):
        import scipy.optimize as optimize

        import planning.corrector as corrector
        import planning.model as model
        import planning.service_guard as guard
        import safe_rl.corrector_wrapper as wrapper

        original_snapshot = wrapper.build_snapshot
        original_guard = guard.build_service_guard
        original_model = corrector.solve_time_indexed_mip_raw_projection
        original_milp = optimize.milp
        original_certificate = model._base_only_terminal_certificate
        original_validate = model.validate_inventory_snapshot
        original_empty = model._projection_empty

        def empty(*args, **kwargs):
            caller = inspect.currentframe().f_back
            local = caller.f_locals
            if "res_b" in local:
                phase = "B"
            elif "res_a" in local:
                phase = "A" if local["res_a"].status != 0 else "after_A_before_B"
            elif "res_inventory" in local:
                phase = "R" if local["res_inventory"].status != 0 else "after_R_before_A"
            elif "t_inventory" in local:
                phase = "R_certificate_before_solver"
            else:
                phase = "validation_or_build_before_R"
            self.current["observed_failure_phase"] = phase
            del caller, local
            return original_empty(*args, **kwargs)

        def make_snapshot(env):
            self.current = {"solver_calls": [], "pid": os.getpid()}
            started = time.perf_counter()
            result = original_snapshot(env)
            self.current["snapshot_including_guard_s"] = time.perf_counter() - started
            self.snapshot = result
            return result

        def make_guard(*args, **kwargs):
            started = time.perf_counter()
            result = original_guard(*args, **kwargs)
            self.current["guard_s"] = time.perf_counter() - started
            return result

        def validate(*args, **kwargs):
            started = time.perf_counter()
            result = original_validate(*args, **kwargs)
            self.current["validation_before_deadline_s"] = time.perf_counter() - started
            return result

        def certificate(*args, **kwargs):
            self.current["r_enter_perf"] = time.perf_counter()
            result = original_certificate(*args, **kwargs)
            self.current["certificate_function_s"] = (
                time.perf_counter() - self.current["r_enter_perf"])
            return result

        def milp(*args, **kwargs):
            caller = inspect.currentframe().f_back
            local = caller.f_locals
            objective = kwargs.get("c")
            stage = next((label for key, label in (
                ("c_inventory", "R"), ("c_off", "A"), ("c_econ", "B"))
                if objective is local.get(key)), "diagnostic")
            deadline = local.get("deadline")
            event = {
                "stage": stage,
                "passed_time_limit_s": kwargs.get("options", {}).get("time_limit"),
                "remaining_at_observer_entry_s": None if deadline is None else
                deadline - time.monotonic(),
                "model_variables": int(local.get("n_vars", 0)),
                "model_constraints": len(local.get("rows", [])),
            }
            for key in ("elapsed_inventory", "tA", "t_inventory", "tA0", "tB0"):
                if key in local:
                    self.current[key] = float(local[key])
            del local, caller
            start = time.perf_counter()
            result = original_milp(*args, **kwargs)
            end = time.perf_counter()
            event.update(status=int(result.status), message=str(result.message),
                         wall_s=end - start, start_perf=start, end_perf=end)
            self.current["solver_calls"].append(event)
            return result

        def solve(snapshot, proposal, **kwargs):
            self.snapshot, self.proposal = snapshot, proposal
            self.current.setdefault("solver_calls", [])
            start = time.perf_counter()
            result = original_model(snapshot, proposal, **kwargs)
            end = time.perf_counter()
            events = self.current["solver_calls"]
            timing = self.current
            timing.update(model_call_s=end - start, budget_s=kwargs["time_limit_s"],
                          failure_stage=timing.get("observed_failure_phase",
                                                   failure_stage(result, events)),
                          variables=result.n_variables, constraints=result.n_constraints,
                          integer_variables=result.n_integer_variables)
            r_start = timing.get("r_enter_perf")
            if r_start is not None:
                timing["pre_r_build_including_validation_s"] = r_start - start
            timing["r_total_s"] = result.inventory_audit.get("reachability_solve_time_s")
            timing["a_call_s"] = result.stage_a_solve_time_s
            timing["b_call_s"] = result.stage_b_solve_time_s
            timing["post_last_solver_s"] = end - events[-1]["end_perf"] if events else None
            timing["non_solver_total_s"] = end - start - sum(e["wall_s"] for e in events)
            return result

        original_step = wrapper.CorrectorWrapper.step

        def step(env, action):
            output = original_step(env, action)
            info = output[-1]
            self.record(info)
            return output

        with contextlib.ExitStack() as stack:
            for obj, name, replacement in (
                (wrapper, "build_snapshot", make_snapshot),
                (guard, "build_service_guard", make_guard),
                (model, "validate_inventory_snapshot", validate),
                (model, "_base_only_terminal_certificate", certificate),
                (model, "_projection_empty", empty),
                (optimize, "milp", milp),
                (corrector, "solve_time_indexed_mip_raw_projection", solve),
                (wrapper.CorrectorWrapper, "step", step),
            ):
                stack.enter_context(patch.object(obj, name, replacement))
            yield

    def record(self, info):
        snapshot = self.snapshot
        guard = snapshot.service_guard
        keys = ("raw_action", "exec_action", "correction_reason", "correction_detail",
                "correction_solve_time_s", "stage_a_status", "stage_b_status",
                "stage_a_solve_time_s", "stage_b_solve_time_s", "reward_total",
                "bess_charge_power_kW", "bess_discharge_power_kW", "sla_violation_count")
        row = {key: jsonable(info[key]) for key in keys}
        row.update(step=int(snapshot.step), soc_before_kwh=snapshot.soc_kwh,
                   task_count=len(snapshot.tasks),
                   tasks=[t.model_dump(mode="json") for t in snapshot.tasks],
                   known_task_started=guard.known_task_started,
                   known_task_interruptible=guard.known_task_interruptible,
                   horizon=snapshot.planning_horizon_steps,
                   forecast_provenance=forecast_evidence(snapshot),
                   timing=jsonable(self.current), peak_rss_bytes=peak_rss_bytes())
        audit = info["inventory_audit"]
        row["inventory_audit"] = {k: v for k, v in audit.items()
                                  if k != "service_guard"}
        capture = (snapshot, self.proposal)
        if not self.rows:
            self.save_capture("initial", capture)
        if row["correction_reason"] in FALLBACKS and not self.first_failure:
            self.save_capture("first_failure", capture)
            if self.previous is not None:
                self.save_capture("before_first_failure", self.previous)
            self.first_failure = True
        self.previous = capture
        self.rows.append(row)
        with (self.folder / "steps.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    def save_capture(self, name, capture):
        snapshot, proposal = capture
        dump(self.folder / f"{name}.json", {
            "snapshot": snapshot, "proposal": proposal,
            "source_checkpoint_sha256": CHECKPOINT_SHA, "budget_s": .25,
            "scope": "new_final_policy_diagnostic_not_historical_transition"})


def historical_summary():
    report = json.loads((SOURCE / "report.json").read_text())
    result = []
    for index in (450, 502, 503, 511):
        batch = report["batch_records"][index]
        result.append({k: batch[k] for k in (
            "batch_index", "origins", "zero_action_fallback_steps", "env_seeds",
            "timing_s", "corrector_solve_stats", "storage_signal", "step_records")}
            | {"episodes": [{k: e[k] for k in (
                "origin", "fallbacks", "service_qualified", "target_gap_kwh",
                "inventory_unproven_steps")} for e in batch["inventory_episodes"]]})
    return result


def episode(folder, origin, mode, policy_path):
    validate_case(origin, mode)
    folder.mkdir(parents=True, exist_ok=False)
    config = load_config()
    apply_frozen_thread_setting(config)
    policy = load_policy(policy_path)
    policy.eval()
    before = parameter_hash(policy)
    sources = semantics_binding()
    generator = torch.Generator(device="cpu").manual_seed(
        0 if mode == "deterministic" else int(mode[-1]))

    def action(obs):
        with torch.inference_mode():
            if mode == "deterministic":
                return deterministic_action(policy, obs)
            raw, _, _ = policy.act(torch.as_tensor(obs, dtype=torch.float32), generator)
            return raw.cpu().numpy().astype(np.float32)

    observer = Observer(folder)
    with observer.installed(), torch.inference_mode():
        result, _, _, _ = evaluate_origin(config, origin, 0, action, run_id=folder.name)
    assert_unchanged(before, parameter_hash(policy), "policy parameters")
    assert_unchanged(sources, semantics_binding(), "release sources")
    assert_unchanged(CHECKPOINT_SHA, sha(SOURCE / "checkpoint_final.pt"), "checkpoint")
    result.pop("final_planning_audit", None)
    result.update(mode=mode, parameter_hash_before=before,
                  parameter_hash_after=parameter_hash(policy), parameter_updates=0,
                  pid=os.getpid(), acceptance=inventory_episode_acceptance(result),
                  failure_counts=dict(Counter(r["correction_reason"] for r in observer.rows)),
                  stage_failure_counts=dict(Counter(r["timing"]["failure_stage"]
                                                    for r in observer.rows)),
                  peak_rss_bytes=peak_rss_bytes())
    dump(folder / "episode.json", result)
    print(json.dumps({k: result[k] for k in (
        "origin", "mode", "fallbacks", "service_qualified", "pid")}), flush=True)
    return result


def receipt(folder):
    hashes = {str(path.relative_to(folder)): sha(path)
              for path in sorted(folder.rglob("*")) if path.is_file()
              and path.name not in ("artifact_verification.json", "console.log")}
    dump(folder / "artifact_verification.json", {"hashes": hashes})


def run(run_id):
    folder = ROOT / "runs" / run_id
    if folder.exists():
        raise FileExistsError("run-id exists; evidence must not be overwritten")
    config = {"scope": "train_only_fixed_final_policy", "origins": list(ORIGINS),
              "modes": list(MODES), "process_arms": ["fresh", "shared"],
              "episodes": 48, "steps": 2304, "environment_seed": 0,
              "parameter_updates": 0, "solver_budget_s": .25,
              "observer": "runtime wrappers; source bytes and call arguments unchanged",
              "timing_caveat": "times include observer overhead; fresh/shared is not a 3h soak",
              "checkpoint_sha256": CHECKPOINT_SHA}
    rows = []
    ledger = {"dependency_lock_hash": sha(ROOT / "uv.lock")}
    report = {"validation_run": False, "test_run": False, "parameter_updates": 0}
    command = " ".join(sys.argv)

    def save(status, error=None):
        write_run(run_id, config=config, metrics=pd.DataFrame([
            {k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows]),
            report={**report, "episodes": rows, "failure": error},
            base_dir=str(ROOT / "runs"), command=command,
            status=status, failure_classification=error, **ledger)

    save("running")
    try:
        verify_release()
        assert_unchanged(CHECKPOINT_SHA, sha(SOURCE / "checkpoint_final.pt"), "checkpoint")
        report["source_hashes_before"] = semantics_binding()
        report["historical_comparison"] = historical_summary()
        policy_path = folder / "final_policy.pt"
        export_policy(SOURCE / "checkpoint_final.pt", policy_path)
        for arm in ("fresh", "shared"):
            for origin, mode in case_grid():
                output = folder / arm / f"{origin}_{mode}"
                if arm == "fresh":
                    subprocess.run([sys.executable, "-m", "scripts.m6p2b_seed0_diagnosis",
                                    "--worker", str(output),
                                    "--origin", str(origin), "--mode", mode,
                                    "--policy", str(policy_path)], cwd=ROOT, check=True)
                    row = json.loads((output / "episode.json").read_text())
                else:
                    row = episode(output, origin, mode, policy_path)
                row["process_arm"] = arm
                rows.append(row)
                ledger = training_source_ledger({r["origin"]: r["injection_provenance"]
                                                 for r in rows})
                save("running")
        if len(rows) != 48 or sum(r["steps"] for r in rows) != 2304:
            raise ValueError("incomplete preregistered experiment")
        report["source_hashes_after"] = semantics_binding()
        assert_unchanged(report["source_hashes_before"], report["source_hashes_after"], "sources")
        report["all_episode_acceptance"] = all(all(r["acceptance"].values()) for r in rows)
        report["checkpoint_sha256_after"] = sha(SOURCE / "checkpoint_final.pt")
        save("success")
    except BaseException as exc:
        save("failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--origin", type=int)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    if args.worker:
        episode(args.worker, args.origin, args.mode, args.policy)
    elif args.run_id:
        run(args.run_id)
    else:
        parser.error("--run-id required")


if __name__ == "__main__":
    main()

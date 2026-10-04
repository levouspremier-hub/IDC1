"""Authenticated physical solver capsule; no policy load or model qualification."""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import platform
import resource
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct
from runs.writer import write_run
from safe_rl_v2.controlled_formal_train import apply_frozen_thread_setting
from scripts.m6p2b_seed0_diagnosis import (
    CHECKPOINT_SHA,
    ROOT,
    SOURCE,
    Observer,
    assert_unchanged,
    forecast_evidence,
    receipt,
    semantics_binding,
    sha,
)
from scripts.m6p2b_seed0_runtime import CAPTURE, CAPTURE_SHA

BASELINE = ROOT / "runs/m6p2b_seed0_soak_v1/failure_replay/replay.json"
BASELINE_SHA = "6397c5623eb752dbf3fc1ff0429dcd0429c18a86e03ac7db3984f88a793b9de9"
RELEASE = ROOT / "configs/release/idc_formal_train_release_v2_r7.json"
RELEASE_SHA = "412057ca837cbea5b4e08017317eae6c1d797897f32f7389fa38553354ddf22f"


def assert_digest(path, expected):
    if sha(path) != expected:
        raise ValueError(f"capsule authentication failed: {path.name}")


def authenticate():
    assert_digest(RELEASE, RELEASE_SHA)
    assert_digest(CAPTURE, CAPTURE_SHA)
    assert_digest(BASELINE, BASELINE_SHA)
    assert_digest(SOURCE / "checkpoint_final.pt", CHECKPOINT_SHA)
    release = json.loads(RELEASE.read_text())
    assert_unchanged(release["semantics_binding"], semantics_binding(), "bound source")
    for asset in release["assets"].values():
        assert_digest(ROOT / asset["path"], asset["sha256"])
    capture = json.loads(CAPTURE.read_text())
    if capture["source_checkpoint_sha256"] != CHECKPOINT_SHA or capture["budget_s"] != .25:
        raise ValueError("capsule origin/budget mismatch")
    snapshot = InventorySnapshot.model_validate(capture["snapshot"])
    proposal = DispatchProposal.model_validate(capture["proposal"])
    forecast_evidence(snapshot)
    baseline = json.loads(BASELINE.read_text())[0]["results"][0]
    if baseline["failure"] != "none":
        raise ValueError("capsule control is not successful")
    return snapshot, proposal, baseline["exec"]


MODES = ("legacy", "frozen_gc_on", "frozen_gc_off")


@contextlib.contextmanager
def runtime_configuration(mode):
    if mode not in MODES:
        raise ValueError("unregistered capsule runtime mode")
    before_gc, before_threads = gc.isenabled(), torch.get_num_threads()
    metadata = {"mode": mode, "gc_before": before_gc,
                "torch_threads_before": before_threads}
    try:
        if mode != "legacy":
            release = json.loads(RELEASE.read_text())
            asset = release["assets"]["training_config"]
            path = ROOT / asset["path"]
            assert_digest(path, asset["sha256"])
            metadata.update(apply_frozen_thread_setting(json.loads(path.read_text())))
            if mode == "frozen_gc_off":
                gc.disable()
            else:
                gc.enable()
        metadata.update(gc_effective=gc.isenabled(),
                        effective_torch_num_threads=torch.get_num_threads())
        yield metadata
    finally:
        if before_gc:
            gc.enable()
        else:
            gc.disable()
        if torch.get_num_threads() != before_threads:
            torch.set_num_threads(before_threads)


def run(run_id, runtime_mode="legacy"):
    folder = ROOT / "runs" / run_id
    if folder.exists():
        raise FileExistsError("run-id exists")
    rows = []
    config = {"scope": "contract_only_no_policy_load", "repeats": 128,
              "budget_s": .25, "capture_sha256": CAPTURE_SHA,
              "baseline_sha256": BASELINE_SHA, "platform": platform.platform(),
              "runtime_mode": runtime_mode,
              "gc_off_is_diagnostic_counterfactual": runtime_mode == "frozen_gc_off"}
    report = {"scope": config["scope"], "full_policy_qualification": False,
              "parameter_updates": 0, "train_only": True}

    def save(status, error=None):
        write_run(run_id, config=config, report={**report, "failure": error},
                  metrics=pd.DataFrame([{k: v for k, v in row.items()
                                         if not isinstance(v, (list, dict))} for row in rows]),
                  status=status, failure_classification=error, command=" ".join(sys.argv),
                  data_hash=CAPTURE_SHA, scenario_hash=sha(RELEASE),
                  dependency_lock_hash=sha(ROOT / "uv.lock"))

    save("running")
    try:
        snapshot, proposal, baseline = authenticate()
        with runtime_configuration(runtime_mode) as runtime:
            report["runtime"] = runtime
            for index in range(128):
                observer = Observer(folder)
                before = resource.getrusage(resource.RUSAGE_SELF)
                wall, cpu, thread = time.perf_counter(), time.process_time(), time.thread_time()
                with observer.installed(), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result = correct(snapshot, proposal, time_limit_s=.25)
                wall, cpu, thread = (time.perf_counter() - wall, time.process_time() - cpu,
                                     time.thread_time() - thread)
                after = resource.getrusage(resource.RUSAGE_SELF)
                executed = [*result.exec_compute_actions, result.exec_storage_action]
                row = {"index": index, "failure": str(result.failure), "wall_s": wall,
                       "cpu_s": cpu, "thread_cpu_s": thread, "exec": executed,
                       "timing": observer.current,
                       "involuntary_switches": after.ru_nivcsw - before.ru_nivcsw,
                       "control_exec_max_difference": float(np.max(np.abs(
                           np.array(executed) - np.array(baseline))))}
                rows.append(row)
                with (folder / "solves.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
        authenticate()
        failed = [r for r in rows if r["failure"] != "none"]
        report.update(completed_solves=len(rows), failed_solves=len(failed),
                      sources_and_assets_unchanged=True,
                      optimal_exec_max_difference=max(
                          (r["control_exec_max_difference"] for r in rows
                           if r["failure"] == "none"), default=None))
        save("failed" if failed else "success", "solver_failure_observed" if failed else None)
        return 1 if failed else 0
    except BaseException as exc:
        save("failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runtime-mode", choices=MODES, default="legacy")
    args = parser.parse_args()
    raise SystemExit(run(args.run_id, args.runtime_mode))

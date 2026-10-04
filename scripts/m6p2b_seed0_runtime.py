"""Bounded captured-input CPU/wall/HiGHS-pool diagnosis; no policy update."""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct
from runs.writer import write_run
from scripts.m6p2b_seed0_diagnosis import (
    CHECKPOINT_SHA,
    ROOT,
    SOURCE,
    Observer,
    assert_unchanged,
    dump,
    forecast_evidence,
    receipt,
    semantics_binding,
    sha,
    verify_release,
)

CAPTURE = ROOT / "runs/m6p2b_seed0_soak_v1/shared/000440_5136_deterministic/first_failure.json"
CAPTURE_SHA = "f80d0ce944a333319ddbab4e574149048312e16e9fd0755c6264388f4a106586"
ARMS = ("published", "single_thread")


def options_for_arm(base, arm):
    if arm not in ARMS:
        raise ValueError("unregistered thread arm")
    return {**base, **({"threads": 1} if arm == "single_thread" else {})}


def thread_count():
    return len(subprocess.check_output(
        ["ps", "-M", "-p", str(os.getpid())], text=True).splitlines()) - 1


def worker(folder, arm, repeats):
    import scipy.optimize as optimize

    import planning.model as model

    verify_release()
    assert_unchanged(CAPTURE_SHA, sha(CAPTURE), "captured input")
    assert_unchanged(CHECKPOINT_SHA, sha(SOURCE / "checkpoint_final.pt"), "checkpoint")
    sources = semantics_binding()
    capture = json.loads(CAPTURE.read_text())
    snapshot = InventorySnapshot.model_validate(capture["snapshot"])
    proposal = DispatchProposal.model_validate(capture["proposal"])
    forecast_evidence(snapshot)
    folder.mkdir(parents=True)
    original_options = model.deterministic_mip_options
    rows = []
    gc_events = []
    gc_start = {}

    def gc_event(phase, info):
        if phase == "start":
            gc_start[info["generation"]] = time.perf_counter()
        else:
            gc_events.append({**info, "wall_s": time.perf_counter()
                              - gc_start.pop(info["generation"], time.perf_counter())})

    gc.callbacks.append(gc_event)
    try:
        with patch.object(model, "deterministic_mip_options",
                          lambda **kw: options_for_arm(original_options(**kw), arm)):
            for index in range(repeats):
                observer = Observer(folder)
                calls = []
                with observer.installed():
                    wrapped = optimize.milp

                    def measured(*args, _wrapped=wrapped, _calls=calls, **kwargs):
                        wall, cpu, thread = (time.perf_counter(), time.process_time(),
                                             time.thread_time())
                        result = _wrapped(*args, **kwargs)
                        _calls.append({"wall_s": time.perf_counter() - wall,
                                      "cpu_s": time.process_time() - cpu,
                                      "thread_cpu_s": time.thread_time() - thread,
                                      "options": kwargs["options"]})
                        return result

                    before = resource.getrusage(resource.RUSAGE_SELF)
                    wall, cpu, thread = (time.perf_counter(), time.process_time(),
                                         time.thread_time())
                    with patch.object(optimize, "milp", measured), warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        result = correct(snapshot, proposal, time_limit_s=.25)
                    wall, cpu, thread = (time.perf_counter() - wall,
                                         time.process_time() - cpu,
                                         time.thread_time() - thread)
                    after = resource.getrusage(resource.RUSAGE_SELF)
                row = {"arm": arm, "index": index, "pid": os.getpid(),
                       "failure": str(result.failure), "wall_s": wall, "cpu_s": cpu,
                       "thread_cpu_s": thread, "calls": calls, "timing": observer.current,
                       "voluntary_switches": after.ru_nvcsw - before.ru_nvcsw,
                       "involuntary_switches": after.ru_nivcsw - before.ru_nivcsw,
                       "minor_faults": after.ru_minflt - before.ru_minflt,
                       "gc_events": list(gc_events), "rss": after.ru_maxrss,
                       "exec": [*result.exec_compute_actions, result.exec_storage_action]}
                gc_events.clear()
                if index % 64 == 0:
                    row["thread_count"] = thread_count()
                rows.append(row)
                with (folder / "solves.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                if row["failure"] != "none" and repeats > 1:
                    # Immediate isolated same-input control, original options, one solve.
                    subprocess.run([sys.executable, "-m", "scripts.m6p2b_seed0_runtime",
                                    "--worker", str(folder / f"fresh_{index}"),
                                    "--arm", "published", "--repeats", "1"],
                                   cwd=ROOT, check=True, timeout=30)
                if index % 64 == 0:
                    print(json.dumps({k: row[k] for k in (
                        "arm", "index", "failure", "wall_s", "cpu_s", "thread_count")}),
                        flush=True)
        assert_unchanged(sources, semantics_binding(), "bound source")
        dump(folder / "result.json", {"rows": rows, "sources_unchanged": True})
    finally:
        gc.callbacks.remove(gc_event)


def run(run_id):
    folder = ROOT / "runs" / run_id
    if folder.exists():
        raise FileExistsError("run-id exists")
    rows = []
    report = {"parameter_updates": 0, "train_only": True,
              "diagnostic_variant": "isolated threads=1 arm; published arm unchanged"}
    config = {"capture_sha256": CAPTURE_SHA, "arms": ARMS, "repeats": 512,
              "budget_s": .25, "parameter_updates": 0}

    def save(status, error=None):
        write_run(run_id, config=config, report={**report, "failure": error},
                  metrics=pd.DataFrame([{k: v for k, v in r.items()
                                         if not isinstance(v, (list, dict))} for r in rows]),
                  status=status, failure_classification=error, command=" ".join(sys.argv),
                  data_hash=sha(CAPTURE), scenario_hash=sha(SOURCE / "manifest.json"),
                  dependency_lock_hash=sha(ROOT / "uv.lock"))

    save("running")
    try:
        for arm in ARMS:
            output = folder / arm
            subprocess.run([sys.executable, "-m", "scripts.m6p2b_seed0_runtime",
                            "--worker", str(output), "--arm", arm, "--repeats", "512"],
                           cwd=ROOT, check=True, timeout=280)
            arm_rows = json.loads((output / "result.json").read_text())["rows"]
            rows.extend(arm_rows)
            values = [r["exec"] for r in rows if r["failure"] == "none"]
            report["optimal_exec_max_difference"] = float(
                np.max(np.abs(np.array(values) - np.array(values[0]))))
            report["completed_solves"] = len(rows)
            save("running")
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
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--repeats", type=int, default=512)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.arm, args.repeats)
    elif args.run_id:
        run(args.run_id)
    else:
        parser.error("--run-id required")


if __name__ == "__main__":
    main()

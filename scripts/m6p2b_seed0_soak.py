"""Registered four-hour train-only fixed final-policy process-duration probe."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.m6p2b_seed0_diagnosis import (
    CHECKPOINT_SHA,
    ROOT,
    SOURCE,
    assert_unchanged,
    case_grid,
    episode,
    export_policy,
    load_policy,
    parameter_hash,
    receipt,
    semantics_binding,
    sha,
    training_source_ledger,
    verify_release,
    write_run,
)

DURATION_S = 14400


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def keep_running(start, current):
    return current - start < DURATION_S


def episode_ok(row):
    return (bool(row["acceptance"]) and all(row["acceptance"].values())
            and row["parameter_updates"] == 0)


def check_host(row):
    if row["errors"] or not row["ac"] or row["free_bytes"] < 5 * 2**30:
        raise RuntimeError(f"host condition invalid: {row}")


def host_sample():
    result = {"at": now(), "monotonic": time.monotonic(), "pid": os.getpid(),
              "free_bytes": shutil.disk_usage(ROOT).free, "load": list(os.getloadavg()),
              "errors": []}
    for key, cmd in (
        ("battery", ["pmset", "-g", "batt"]),
        ("thermal", ["pmset", "-g", "therm"]),
        ("process", ["ps", "-p", str(os.getpid()), "-o", "pid=,rss=,%cpu=,etime="]),
        ("power_settings", ["pmset", "-g", "custom"]),
    ):
        try:
            result[key] = subprocess.check_output(cmd, text=True, timeout=5)
        except Exception as exc:
            result["errors"].append(f"{key}: {exc}")
    result["ac"] = "Now drawing from 'AC Power'" in result.get("battery", "")
    return result


class HostMonitor:
    def __init__(self, folder):
        self.folder = folder
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.failure = None
        self.power_settings = None
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        with self.lock:
            row = host_sample()
            with (self.folder / "host.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            try:
                check_host(row)
                if self.power_settings is None:
                    self.power_settings = row["power_settings"]
                if row["power_settings"] != self.power_settings:
                    raise RuntimeError("power settings changed during diagnosis")
            except RuntimeError as exc:
                self.failure = str(exc)
            if self.failure:
                raise RuntimeError(self.failure)

    def loop(self):
        while not self.stop.wait(60):
            try:
                self.sample()
            except Exception as exc:
                self.failure = str(exc)
                return

    def close(self):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=25)


def run(run_id, *, metadata=None):
    folder = ROOT / "runs" / run_id
    if folder.exists():
        raise FileExistsError("run-id exists; no overwrite or implicit resume")
    config = {"scope": "train_only_fixed_final_policy", "shared_duration_s": DURATION_S,
              "cases": case_grid(), "environment_seed": 0, "parameter_updates": 0,
              "checkpoint_sha256": CHECKPOINT_SHA, "budget_s": .25,
              "host_sample_interval_s": 60, "ac_required": True,
              "boundary": "finish current episode; fresh 24 before and after",
              "shared_policy": "one strictly loaded object retained for entire shared phase",
              "diagnostic_metadata": metadata or {}}
    rows = []
    report = {"started_at": now(), "pid": os.getpid(), "phase": "preflight",
              "validation_run": False, "test_run": False, "parameter_updates": 0}
    ledger = {"dependency_lock_hash": sha(ROOT / "uv.lock")}
    monitor = HostMonitor(folder)

    def save(status, error=None):
        write_run(run_id, config=config, metrics=pd.DataFrame([
            {k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows]),
            report={**report, "episodes": rows, "failure": error},
            base_dir=str(ROOT / "runs"), command=" ".join(sys.argv),
            status=status, failure_classification=error, **ledger)

    def record(row, arm, ordinal):
        nonlocal ledger
        assert_unchanged(report["observer_source_hashes"], observer_hashes(), "observers")
        row.update(process_arm=arm, ordinal=ordinal, recorded_at=now())
        rows.append(row)
        ledger = training_source_ledger({r["origin"]: r["injection_provenance"] for r in rows})
        report["completed_episodes"] = len(rows)
        report["completed_steps"] = sum(r["steps"] for r in rows)
        save("running")

    def fresh(arm):
        report["phase"] = arm
        save("running")
        for index, (origin, mode) in enumerate(case_grid()):
            monitor.sample()
            output = folder / arm / f"{origin}_{mode}"
            subprocess.run([sys.executable, "-m", "scripts.m6p2b_seed0_diagnosis",
                            "--worker", str(output), "--origin", str(origin),
                            "--mode", mode, "--policy", str(policy_path)],
                           cwd=ROOT, check=True, timeout=600)
            row = json.loads((output / "episode.json").read_text())
            record(row, arm, index)

    def observer_hashes():
        names = ["m6p2b_seed0_soak.py", "m6p2b_seed0_diagnosis.py"]
        if metadata:
            names.extend(metadata.get("additional_observer_files", []))
        return {name: sha(ROOT / "scripts" / name) for name in names}

    save("running")
    try:
        report["observer_source_hashes"] = observer_hashes()
        verify_release()
        assert_unchanged(CHECKPOINT_SHA, sha(SOURCE / "checkpoint_final.pt"), "checkpoint")
        sources = semantics_binding()
        report["source_hashes_before"] = sources
        monitor.sample()
        monitor.thread.start()
        policy_path = folder / "final_policy.pt"
        export_policy(SOURCE / "checkpoint_final.pt", policy_path)
        fresh("fresh_before")
        if not all(episode_ok(r) for r in rows):
            raise RuntimeError("pre-control acceptance failed; long phase not started")
        policy = load_policy(policy_path)
        policy.eval()
        before = parameter_hash(policy)
        report.update(phase="shared", shared_started_at=now(), parameter_hash_before=before)
        start = time.monotonic()
        report["shared_start_monotonic"] = start
        save("running")
        index = 0
        failed_capture = None
        while keep_running(start, time.monotonic()):
            monitor.sample()
            origin, mode = case_grid()[index % 24]
            output = folder / "shared" / f"{index:06d}_{origin}_{mode}"
            row = episode(output, origin, mode, policy_path, policy=policy)
            assert_unchanged(before, parameter_hash(policy), "persistent policy")
            report["shared_elapsed_s"] = time.monotonic() - start
            record(row, "shared", index)
            index += 1
            if not episode_ok(row):
                failed_capture = output / "first_failure.json"
                report["early_stop"] = "episode_acceptance_failure"
                break
        report["shared_ended_at"] = now()
        report["shared_elapsed_s"] = time.monotonic() - start
        report["shared_episodes"] = index
        report["parameter_hash_after"] = parameter_hash(policy)
        monitor.sample()
        if failed_capture is not None and failed_capture.exists():
            report["phase"] = "failure_replay"
            save("running")
            subprocess.run([sys.executable, "-m", "scripts.m6p2b_seed0_diagnosis",
                            "--replay-worker", str(failed_capture), "--replay-out",
                            str(folder / "failure_replay")], cwd=ROOT, check=True, timeout=120)
        fresh("fresh_after")
        monitor.sample()
        report["source_hashes_after"] = semantics_binding()
        assert_unchanged(sources, report["source_hashes_after"], "sources")
        assert_unchanged(CHECKPOINT_SHA, sha(SOURCE / "checkpoint_final.pt"), "checkpoint")
        report["all_episode_acceptance"] = all(episode_ok(r) for r in rows)
        if report["shared_elapsed_s"] < DURATION_S or not report["all_episode_acceptance"]:
            raise RuntimeError("incomplete duration or failed acceptance; evidence retained")
        report.update(phase="complete", finished_at=now())
        monitor.close()
        save("success")
    except BaseException as exc:
        report["finished_at"] = now()
        monitor.close()
        save("failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        monitor.close()
        receipt(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise RuntimeError(f"interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run(args.run_id)


if __name__ == "__main__":
    main()

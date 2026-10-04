"""Original-options soak with contemporaneous CPU-timed old/fresh failure replay."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from scripts import m6p2b_seed0_soak as soak
from scripts.m6p2b_seed0_diagnosis import Observer, dump, replay_capture


def run(run_id):
    original_record = Observer.record
    original_host = soak.host_sample
    probed = False

    def host():
        row = original_host()
        try:
            output = subprocess.check_output(
                ["ps", "-Ao", "pid=,pcpu=,rss=,comm="], text=True, timeout=5)
            # Local-only resource evidence; publish aggregate CPU metrics, not app activity.
            lines = sorted(output.splitlines(),
                           key=lambda line: float(line.split()[1]), reverse=True)
            row["top_cpu_processes_local_only"] = lines[:12]
        except Exception as exc:
            row["errors"].append(f"process sampling: {exc}")
        return row

    def record(self, info):
        nonlocal probed
        original_record(self, info)
        if not probed and self.rows[-1]["correction_reason"] == "timeout":
            probed = True
            capture = self.folder / "first_failure.json"
            # Original output and first-failure snapshot were written first. No env reset
            # or extra budget for this action; these additional solves are separate probes.
            with self.unobserved():
                dump(self.folder / "contemporaneous_probe_started.json", {
                    "at": soak.now(), "old_pid": os.getpid(), "original_action_unchanged": True})
                replay_capture(capture, self.folder / "contemporaneous_old")
                subprocess.run([sys.executable, "-m", "scripts.m6p2b_seed0_diagnosis",
                                "--replay-worker", str(capture), "--replay-out",
                                str(self.folder / "contemporaneous_fresh")],
                               cwd=soak.ROOT, check=True, timeout=120)
                dump(self.folder / "contemporaneous_probe_finished.json", {"at": soak.now()})

    with patch.object(Observer, "record", record), patch.object(soak, "host_sample", host):
        soak.run(run_id, metadata={
            "additional_observer_files": [Path(__file__).name],
            "solver_options": "original published; no single-thread override",
            "on_first_timeout": "persist original output then old/fresh same-input probes",
            "post_failure_caveat": "later steps are affected by diagnostic probe runtime",
            "policy_updates": 0,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    def interrupted(signum, frame):
        raise RuntimeError(f"interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run(parser.parse_args().run_id)


if __name__ == "__main__":
    main()

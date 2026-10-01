"""Run repository or related quality checks with durable native evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET

import pandas as pd

from evaluation.sources import canonical_source_digests
from runs.writer import write_run
from scenario.inventory_release import ROOT, checkpoint_binding, semantics_binding, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    folder = ROOT / "runs" / args.run_id
    if folder.exists():
        raise FileExistsError(f"quality run already exists: {folder}")
    binding = checkpoint_binding()
    test_hashes = {str(p.relative_to(ROOT)): sha(p)
                   for p in sorted((ROOT / "tests").rglob("*.py"))}
    code = semantics_binding()
    source_rows = [[d.role, d.logical_path, d.sha256] for d in canonical_source_digests()]
    data_hash = hashlib.sha256(json.dumps(source_rows, sort_keys=True).encode()).hexdigest()
    scenario_hash = hashlib.sha256(json.dumps(
        [test_hashes, code], sort_keys=True).encode()).hexdigest()
    ledger = dict(dependency_lock_hash=sha(ROOT / "uv.lock"), data_hash=data_hash,
                  scenario_hash=scenario_hash)
    junit = folder / "pytest.xml"
    if args.full:
        command = ["make", "check"]
    else:
        tests = sorted(str(p.relative_to(ROOT))
                       for p in (ROOT / "tests").glob("test_m6p2b_*.py"))
        command = ["uv", "run", "pytest", "-q", *tests]
    config = {"scope": "repository_quality_gate" if args.full else "related_quality_gate",
              "command": command, "inventory_binding": binding,
              "test_source_hashes": test_hashes, "semantic_source_hashes": code,
              "instrumentation_sha256": sha(__file__), "heldout_evaluation_run": False}
    write_run(args.run_id, config=config, metrics=pd.DataFrame(), report={"status": "running"},
              base_dir=str(ROOT / "runs"), command=" ".join(command), status="running", **ledger)
    started = time.monotonic()
    environment = dict(os.environ)
    environment["PYTEST_ADDOPTS"] = "--junitxml=" + str(junit)
    with (folder / "console.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                stdout=log, stderr=subprocess.STDOUT)
    suites = list(ET.parse(junit).getroot().iter("testsuite")) if junit.exists() else []
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    unchanged = (code == semantics_binding() and all(
        sha(ROOT / path) == digest for path, digest in test_hashes.items()))
    report = {"passed": result.returncode == 0 and counts["tests"] > 0 and unchanged,
              "exit_code": result.returncode, "pytest_counts": counts,
              "elapsed_s": time.monotonic() - started, "sources_unchanged": unchanged,
              "inventory_binding": binding, "semantic_source_hashes": code,
              "test_source_hashes": test_hashes, "console_sha256": sha(folder / "console.log"),
              "junit_sha256": sha(junit) if junit.exists() else None,
              "instrumentation_sha256": sha(__file__),
              "validation_run": False, "test_run": False, "quality_tests_run": True}
    write_run(args.run_id, config=config, metrics=pd.DataFrame([counts]), report=report,
              base_dir=str(ROOT / "runs"), command=" ".join(command),
              status="success" if report["passed"] else "failed",
              failure_classification=None if report["passed"] else "quality_gate_failed", **ledger)
    assert json.loads((folder / "report.json").read_text()) == report
    assert len(pd.read_parquet(folder / "metrics.parquet")) == 1
    receipt = {name: sha(folder / name) for name in (
        "config.yaml", "metrics.parquet", "report.json", "manifest.json", "console.log")}
    (folder / "artifact_verification.json").write_text(json.dumps(
        {"file_hashes": receipt, "reread_verified": True,
         "figures_exists": (folder / "figures").is_dir()}, indent=2))
    print(json.dumps({"run_id": args.run_id, "passed": report["passed"], **counts}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

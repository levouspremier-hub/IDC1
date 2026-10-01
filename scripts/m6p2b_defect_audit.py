"""Read-only audit of historical training and residual inventory defects."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_report(run):
    return json.loads((ROOT / "runs" / run / "report.json").read_text())


def controlled_schedule_case(compute, margin):
    """Perfect constant exogenous fixture, no unseen tasks, unchanged executor."""
    from envs.idc_price_env import IDCPriceEnv20D
    from idc_model.task import Task
    from planning.snapshot_adapter import build_snapshot
    from safe_rl.corrector_wrapper import CorrectorWrapper

    env = IDCPriceEnv20D(
        horizon=3, forecast_cutoff=3, delta_t_hours=.5,
        bess_soc_init=.481, access_limit_kw=1000., base_load=.3,
        price_t=np.full(3, .2), pv_t=np.zeros(3), wt_t=np.zeros(3),
        carbon_factor_t=np.full(3, .4), T_amb=np.full(3, 25.),
        server_seed=0, task_seed=0, forecast_seed=0)
    env.terminal_inventory_enabled = True
    env.reset(seed=0)
    env.terminal_service_guard_version = "arrived-service-reserve-v1"
    env.terminal_service_temperature_margin_c = margin
    env.task_arrival_forecast[:] = 0.
    env.access_limit_kw = env._idc_power_kw(
        np.full(env.model.N, env.base_load), 25. + margin) + 2.
    task = Task(1, "A", "known interruptible", 0, 1, np.array([.1]),
                2., 3, 3., True, False)
    task.status = "waiting"
    env.tasks = [task]
    wrapper = CorrectorWrapper(env, corrector_time_limit_s=.25)
    raw = np.asarray([compute] * 20 + [0.], dtype=np.float32)
    rows = []
    for step in range(3):
        snapshot = build_snapshot(env)
        _, reward, done, _, info = wrapper.step(raw)
        rows.append({
            "step": step, "snapshot": snapshot.model_dump(mode="json"),
            "raw_action": raw.tolist(), "exec_action": info["exec_action"].tolist(),
            "inventory_audit": info["inventory_audit"],
            "remaining_work_after": task.remaining_work,
            "energy_after_kwh": env.bess_energy_kWh,
            "reward": float(reward), "episode_done": bool(done),
            "access_curtailment_work": info["access_curtailment_work"],
            "correction_reason": info["correction_reason"],
            "stage_a_status": info["stage_a_status"],
            "stage_b_status": info["stage_b_status"],
            "business_gap": info["business_gap"],
            "deadline_shortfall_work": info["deadline_shortfall_work"],
        })
    return {"compute_proposal": compute, "temperature_margin_c": margin,
            "fixture": "constant perfect visible exogenous; no unseen tasks; diagnostic only",
            "global_budget_s": .25, "steps": rows,
            "initial_snapshot_sha256": hashlib.sha256(json.dumps(
                rows[0]["snapshot"], sort_keys=True).encode()).hexdigest(),
            "terminal_gap_kwh": abs(env.bess_energy_kWh - 50.),
            "end_remaining_work": task.remaining_work}


def aggregate_backlog_case():
    from planning.service_guard import build_service_guard

    env = SimpleNamespace(
        terminal_service_guard_version="arrived-service-reserve-v1",
        terminal_inventory_enabled=True, terminal_service_temperature_margin_c=0.,
        current_step=0, tasks=[], horizon=3,
        model=SimpleNamespace(C_server=np.array([2., 2.])), delta_t_hours=.5,
        base_load=0., access_limit_kw=4., bess_charge_power_max_kW=4.,
        _idc_power_kw=lambda loads, temp: float(np.sum(loads)),
    )
    capacity = [1., 1.]
    arrival = [0., 3., 0.]
    guard = build_service_guard(env, [], capacity, temperature=[25.] * 3, arrival=arrival)
    # Accounting comparator carries aggregate forecast work, without task instances.
    queue, expected = 0., []
    for amount in arrival:
        queue += amount
        served = min(queue, sum(capacity))
        queue -= served
        expected.append(4. - served)
    return {"arrival_work": arrival, "capacity_work_per_step": capacity,
            "observed_charge_limits_kw": guard.charge_limits_kw,
            "carryover_accounting_charge_limits_kw": expected,
            "omitted_backlog_work_at_last_step": 1.,
            "reproduced": guard.charge_limits_kw[-1] > expected[-1],
            "scope": "synthetic forecast overload; does not assert observed train overload"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="m6p2b_defect_audit_v2_r1")
    args = parser.parse_args()
    folder = ROOT / "runs" / args.run_id
    if folder.exists():
        raise FileExistsError(f"audit run already exists: {folder}")
    from evaluation.sources import canonical_source_digests
    from runs.writer import write_run
    from scenario.inventory_release import checkpoint_binding
    from scripts.audit_m6p2a_formal_checkpoints import FORMAL_RUN_IDS, audit_formal_run
    from scripts.m6p2b_inventory_repair import short_gate

    sources = {}
    source_runs = [*FORMAL_RUN_IDS, "m6p2a_formal_trainonly_212x3_v1",
                   "m6p2b_short_gate_v2_r2",
                   *[f"m6p2b_short_seed{s}_v2_r2" for s in range(3)]]
    for run in source_runs:
        for name in ("config.yaml", "metrics.parquet", "report.json", "manifest.json",
                     "checkpoint_final.pt"):
            path = ROOT / "runs" / run / name
            if path.is_file():
                sources[str(path.relative_to(ROOT))] = digest(path)
    source_code = {name: digest(ROOT / name) for name in (
        "scripts/m6p2b_defect_audit.py", "scripts/audit_m6p2a_formal_checkpoints.py",
        "scripts/m6p2b_inventory_repair.py", "planning/service_guard.py",
        "planning/model.py", "safe_rl/corrector_wrapper.py",
        "tests/test_m6p2b_terminal_inventory.py")}
    data = [[d.role, d.logical_path, d.sha256] for d in canonical_source_digests()]
    ledger = {"dependency_lock_hash": digest(ROOT / "uv.lock"),
              "data_hash": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
              "scenario_hash": hashlib.sha256(json.dumps(
                  [sources, source_code], sort_keys=True).encode()).hexdigest()}
    config = {"scope": "read_only_historical_and_train_artifact_audit",
              "runtime_binding": checkpoint_binding(), "source_artifact_hashes": sources,
              "source_code_hashes": source_code,
              "controlled_cases": {"horizon": 3, "soc": .481, "known_work": 2.,
                                   "compute_proposals": [0., 1.], "margins_c": [0., 4.4],
                                   "solver_global_budget_s": .25},
              "new_training": False, "validation_run": False, "test_run": False}
    command = f"uv run python -m scripts.m6p2b_defect_audit --run-id {args.run_id}"
    write_run(args.run_id, config=config, metrics=pd.DataFrame(), report={"status": "running"},
              base_dir=str(ROOT / "runs"), command=command, status="running", **ledger)
    rows, report = [], {"scope": config["scope"], "new_training": False,
                        "validation_run": False, "test_run": False, "errors": []}
    try:
        formal = []
        for run in FORMAL_RUN_IDS:
            item = audit_formal_run(run)
            formal.append(item)
            rows.append({"kind": "historical_formal_integrity", "run_id": run,
                         "passed": item["all_pass"]})
            print(f"historical {run}: {item['all_pass']}", flush=True)
        report["historical_formal_integrity"] = formal
        old = pd.read_parquet(ROOT / "runs/m6p2a_formal_trainonly_212x3_v1/metrics.parquet")
        report["historical_636"] = {
            "episodes": len(old), "service_pass": int(old.service_qualified.sum()),
            "inventory_band_pass": int(old.final_soc_within_env_tolerance.sum()),
            "physical_violation_steps": int(old[[
                "access_limit_violation_steps", "soc_violation_steps",
                "charge_discharge_exclusion_violations", "energy_conservation_violations"]]
                .to_numpy().sum()),
            "final_soc_min": float(old.final_soc.min()),
            "final_soc_max": float(old.final_soc.max()),
            "correction_timeout_count": int(old.correction_timeout_count.sum()),
            "correction_zero_action_fallback_count": int(
                old.correction_zero_action_fallback_count.sum()),
            "storage_abs_delta_mean": float(old.storage_abs_delta_mean.mean()),
            "per_seed_summary": read_report("m6p2a_formal_trainonly_212x3_v1")["per_seed_summary"],
            "evidence_type": "reread retained full train-only diagnosis; no new replay"}
        _, gate, _ = short_gate()
        report["current_live_short_gate"] = gate
        shorts = []
        for seed in range(3):
            r = read_report(f"m6p2b_short_seed{seed}_v2_r2")
            episodes, batches = r["inventory_episodes"], r["batch_records"]
            signals = pd.DataFrame([b["storage_signal"] for b in batches])
            item = {"seed": seed, "episodes": len(episodes), "batches": r["batches"],
                    "transitions": r["transitions"],
                    "service_pass": sum(e["service_qualified"] for e in episodes),
                    "inventory_band_pass": sum(e["inventory_qualified"] for e in episodes),
                    "target_pass": sum(e["target_qualified"] for e in episodes),
                    "physics_violations": sum(e["physical_violation_count"] for e in episodes),
                    "final_target_reachable_true": sum(
                        e["final_planning_audit"].get("target_reachable") is True
                        for e in episodes),
                    "final_target_reachable_unknown": sum(
                        e["final_planning_audit"].get("target_reachable") is None
                        for e in episodes),
                    "final_soc_min": min(e["final_soc"] for e in episodes),
                    "final_soc_max": max(e["final_soc"] for e in episodes),
                    "charge_kwh": sum(e["charge_kwh"] for e in episodes),
                    "discharge_kwh": sum(e["discharge_kwh"] for e in episodes),
                    "storage_signal_batch_mean": signals.mean().to_dict(),
                    "terminal_reward_sum": float(signals.terminal_reward_sum.sum()),
                    "fallback_steps": sum(b["zero_action_fallback_steps"] for b in batches)}
            shorts.append(item)
            rows.append({"kind": "controlled_short_artifact", **{
                k: v for k, v in item.items() if k != "storage_signal_batch_mean"}})
        report["current_short_summary"] = shorts
        cases = []
        for margin in (0., 4.4):
            for compute in (0., 1.):
                case = controlled_schedule_case(compute, margin)
                cases.append(case)
                rows.append({"kind": "controlled_schedule_case", "margin_c": margin,
                             "compute_proposal": compute,
                             "terminal_gap_kwh": case["terminal_gap_kwh"],
                             "remaining_work": case["end_remaining_work"]})
                print(f"schedule margin={margin} raw={compute} gap={case['terminal_gap_kwh']}",
                      flush=True)
        report["controlled_schedule_cases"] = cases
        for first, second in zip(cases[::2], cases[1::2], strict=True):
            assert first["initial_snapshot_sha256"] == second["initial_snapshot_sha256"]
        report["controlled_pairs_identical_initial_snapshots"] = True
        report["preceding_exploratory_run"] = {
            "run_id": "m6p2b_defect_audit_v2_r1",
            "limitation": "unseeded demo server construction; replaced with fixed-seed pairs"}
        report["aggregate_backlog_case"] = aggregate_backlog_case()
        report["static_gate_findings"] = {
            "short_target_condition_only_checks_last_snapshot": True,
            "short_last_unreachable_or_unknown_bypasses_target_test": True,
            "validation_readiness_does_not_check_target_or_fallback": True,
            "validation_readiness_does_not_check_unreachable_explanations": True,
            "five_method_ready_is_explicitly_false": True,
            "source": "scripts/m6p2b_inventory_repair.py:short_gate/audit"}
        check_dir = ROOT / "runs/m6p2b_checks_v2_r2_retry1"
        report["full_make_check"] = {
            "retained_report": json.loads((check_dir / "report.json").read_text()),
            "log_tail": (check_dir / "console.log").read_text()[-1300:],
            "log_sha256": digest(check_dir / "console.log"),
            "passed_confirmed": False,
            "observed_state": "stale running report; original session and process absent"}
    except Exception as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    unchanged = all(digest(ROOT / p) == h for p, h in sources.items())
    report["source_artifacts_unchanged"] = unchanged
    report["validation_readiness"] = False
    report["all_exposed_problems_resolved"] = False
    report["code_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    success = not report["errors"] and unchanged
    report["audit_execution_success"] = success
    write_run(args.run_id, config=config, metrics=pd.DataFrame(rows), report=report,
              base_dir=str(ROOT / "runs"), command=command,
              status="success" if success else "failed",
              failure_classification=None if success else "defect_audit_incomplete", **ledger)
    assert json.loads((folder / "report.json").read_text()) == report
    assert len(pd.read_parquet(folder / "metrics.parquet")) == len(rows)
    receipt = {name: digest(folder / name) for name in (
        "config.yaml", "metrics.parquet", "report.json", "manifest.json")}
    (folder / "artifact_verification.json").write_text(json.dumps(
        {"file_hashes": receipt, "figures_exists": (folder / "figures").is_dir(),
         "reread_verified": True}, indent=2))
    print(json.dumps({"run_id": args.run_id, "success": success,
                      "unchanged": unchanged, "errors": report["errors"]}), flush=True)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

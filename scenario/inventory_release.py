"""M6-P2b v2 release: bind the full training/corrector semantics, retain v1 assets."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

from evaluation.sources import ROLE_LOGICAL_PATHS
from scenario.env_release import load_verified_env_release

ROOT = Path(__file__).resolve().parent.parent
RUN_REVISION = "v2_r4"
CONFIG_PATH = "configs/training/idc_training_config_v2_r4.json"
MATRIX_PATH = "configs/experiments/m9_experiment_matrix_v4_r4.json"
RELEASE_PATH = "configs/release/idc_formal_train_release_v2_r4.json"
SEMANTICS = "terminal-inventory-v1"
SOURCE_PATHS = (
    "contracts/inventory.py", "planning/model.py", "planning/snapshot_adapter.py",
    "planning/corrector.py", "safe_rl/corrector_wrapper.py",
    "safe_rl_v2/inventory_train.py", "safe_rl_v2/formal_train_loop.py",
    "safe_rl_v2/rollout.py", "safe_rl_v2/policy.py", "safe_rl_v2/buffer.py",
    "safe_rl_v2/ppo_update.py", "safe_rl_v2/ppo_objective.py", "safe_rl_v2/lagrangian.py",
    "safe_rl_v2/models.py", "checkpointing/versioned.py",
    "safe_rl_v2/inventory_diagnostics.py", "scenario/inventory_release.py",
    "checkpointing/inventory_eval_input.py", "envs/idc_price_env.py",
    "scripts/m6p2b_inventory_repair.py",
    "evaluation/adapter.py",
    "planning/service_guard.py", "idc_model/allocation.py",
    "evaluation/inventory.py", "evaluation/metrics.py",
)
ASSET_PATHS = {**ROLE_LOGICAL_PATHS, "training_config": CONFIG_PATH,
               "experiment_matrix": MATRIX_PATH}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def semantics_binding():
    return {path: sha(ROOT / path) for path in SOURCE_PATHS}


def observation_spec():
    from safe_rl.corrector_wrapper import (
        INVENTORY_OBSERVATION_FIELDS,
        INVENTORY_OBSERVATION_VERSION,
    )
    return {"version": INVENTORY_OBSERVATION_VERSION, "base_dimension": 520,
            "appended_fields": list(INVENTORY_OBSERVATION_FIELDS), "dimension": 523,
            "normalization": "declared physical fractions; refs_v4 unchanged",
            "shared_by_all_methods": True}


def reward_repair_spec():
    return {"version": "common-sgd-degradation-v1",
            "formula": "r_degradation = -(reward_cost_weight / cost_ref) * degradation_SGD",
            "coefficient_basis": "same marginal reward per SGD; original cost/carbon/service terms",
            "equivalent_weight": 1. / 600., "frozen_refs_unchanged": True,
            "evidence_path": "runs/m6p2b_reward_counterfactual_v2_r3/report.json",
            "evidence_sha256": "72cf0cc0d06db7d8dc7e1d8b391a7acd0268abd5f25d2caa0286c5150a19a278"}


def temperature_reserve_spec():
    return {"version": "train-temperature-upper-reserve-v1", "margin_c": 4.4,
            "formula": "ceil(max(train_realized_c - signed_B6_forecast_c, 0) * 10) / 10",
            "evidence_path": "runs/m6p2b_temperature_calibration_v2_r1/report.json",
            "evidence_sha256": "386a5020d78e8684385e3e07ef869f7bb93b4e64645348892d9e97940f745fb7",
            "physical_error_bound_proven": False, "forecast_arrays_modified": False}


def inventory_acceptance_spec():
    return {"version": "inventory-progress-audit-v1", "unexplained_gap_blocks": True,
            "unproven_reachability_blocks": True, "planner_consistency_loss_blocks": True,
            "fair_pairing_terminal_difference_kwh_max": 1e-6,
            "known_service_and_inventory_share_feasible_domain": True}


def method_design(methods):
    mutable = {"status", "runnable_now", "blocking_reasons", "blocking_reasons_note"}
    return [{k: v for k, v in method.items() if k not in mutable} for method in methods]


def refresh_method_readiness(matrix):
    """Update stale readiness text; preserve every preregistered method design field."""
    matrix["formal_training_ready"] = True
    matrix["release_readiness"] = {"formal_env_ready": True, "formal_training_ready": True}
    for method in matrix["methods"]:
        if method["method_id"] == "safe_ppo_joint_rolling_corrector":
            method["status"] = "training_release_ready_evaluation_pending"
            method["runnable_now"] = False
            method["blocking_reasons"] = [
                "r4训练配置和独立发布已接线；受控验收与完整门禁仍须通过",
                "新版正式checkpoint与完整train诊断尚未完成，不能进入正式评估"]
        elif method.get("trains_ppo"):
            method["blocking_reasons"] = [
                "尚无该方法自己的冻结训练配置与语义发布",
                "尚无该方法已审核的新21维正式checkpoint与诊断证据"]



def diagnostic_candidate_config(*, reward_semantics="common-sgd-degradation-v1"):
    """Explicit unreleased semantics for calibration/probes, never formal restore."""
    from safe_rl_v2.formal_train_loop import load_frozen_training_config
    config = copy.deepcopy(load_frozen_training_config())
    config.update(schema="idc-training-config-v2", version="v2",
                  configuration_revision="r4", status="candidate_diagnostic_only",
                  reward_semantics=reward_semantics)
    if reward_semantics == "common-sgd-degradation-v1":
        config["reward_repair"] = reward_repair_spec()
    elif reward_semantics != "original-env-reward-v1":
        raise ValueError("unregistered candidate reward semantics")
    config["service_temperature_reserve"] = temperature_reserve_spec()
    config["inventory_acceptance"] = inventory_acceptance_spec()
    config["training"]["corrector"].update(
        inventory_version=SEMANTICS, horizon_policy="real_episode_remainder",
        observation_version=observation_spec()["version"], solver_feasibility_tolerance=1e-8,
        service_guard_version="arrived-service-reserve-v3",
        service_temperature_margin_c=temperature_reserve_spec()["margin_c"])
    config["training"]["policy"]["obs_dim"] = observation_spec()["dimension"]
    config["candidate_source"] = {
        "path": "configs/training/idc_training_config_v1.json",
        "sha256": sha(ROOT / "configs/training/idc_training_config_v1.json")}
    return config


def load_config():
    config = json.loads((ROOT / CONFIG_PATH).read_text())
    if (config.get("schema") != "idc-training-config-v2"
            or config.get("status") != "frozen" or config.get("version") != "v2"):
        raise ValueError("frozen inventory training config v2 is required")
    corrector = config["training"]["corrector"]
    from safe_rl.corrector_wrapper import INVENTORY_OBSERVATION_VERSION
    if (config.get("configuration_revision") != "r4"
            or corrector.get("observation_version") != INVENTORY_OBSERVATION_VERSION
            or config["training"]["policy"]["obs_dim"] != 523
            or corrector.get("solver_feasibility_tolerance") != 1e-8):
        raise ValueError("r4 requires registered terminal state observation semantics")
    if config.get("inventory_acceptance") != inventory_acceptance_spec():
        raise ValueError("r4 inventory progress acceptance binding mismatch")
    if corrector.get("service_guard_version") != "arrived-service-reserve-v3":
        raise ValueError("r4 requires registered causal arrived service reserves")
    if (corrector["inventory_version"] != SEMANTICS or corrector["time_limit_s"] != .25
            or config["reward_semantics"] != "common-sgd-degradation-v1"
            or config.get("reward_repair") != reward_repair_spec()):
        raise ValueError("unregistered inventory/reward/budget semantics")
    reserve = config.get("service_temperature_reserve")
    if (reserve != temperature_reserve_spec()
            or corrector.get("service_temperature_margin_c") != reserve["margin_c"]
            or sha(ROOT / reserve["evidence_path"]) != reserve["evidence_sha256"]):
        raise ValueError("registered train temperature reserve binding mismatch")
    evidence = config["reward_repair"]
    if sha(ROOT / evidence["evidence_path"]) != evidence["evidence_sha256"]:
        raise ValueError("reward repair train-only evidence hash mismatch")
    from safe_rl_v2.formal_train_loop import live_asset_hash_check
    from scripts.calibrate_training_config import select_origins
    calibration = config["calibration"]
    if calibration["split"] != "train" or calibration["origins"] != list(select_origins()):
        raise ValueError("v2 calibration must use the frozen 24 train origins")
    for kind in ("manifest", "report"):
        if sha(ROOT / calibration[f"run_{kind}_path"]) != calibration[f"run_{kind}_sha256"]:
            raise ValueError(f"v2 calibration {kind} hash mismatch")
    report = json.loads((ROOT / calibration["run_report_path"]).read_text())
    if report["passed"] is not True:
        raise ValueError("v2 calibration must pass service and inventory acceptance")
    for key, value in report["candidate"].items():
        if key in ("schema", "status", "note"):
            continue
        actual = config["training"].get(key)
        if key == "backend":
            actual = {k: v for k, v in actual.items() if k != "note"}
            value = {k: v for k, v in value.items() if k != "note"}
        if actual != value:
            raise ValueError(f"v2 config differs from measured calibration: {key}")
    if any(not row["match"] for row in live_asset_hash_check(config).values()):
        raise ValueError("v2 calibration live asset mismatch")
    return config


def load_matrix():
    matrix = json.loads((ROOT / MATRIX_PATH).read_text())
    old = json.loads((ROOT / "configs/experiments/m9_experiment_matrix_v3.json").read_text())
    if matrix.get("schema") != "m9-experiment-matrix-v4":
        raise ValueError("inventory experiment matrix v4 is required")
    if matrix.get("policy_observation") != observation_spec():
        raise ValueError("v4 r4 requires the shared versioned terminal state observation")
    # All substantive preregistered experimental design remains unchanged.
    for key in ("split_freeze", "scenario_freeze", "training_schedule", "evaluation_schedule",
                "pairing", "statistics", "fair_cost_carbon_pairing"):
        if key in old and matrix.get(key) != old[key]:
            raise ValueError(f"v4 cannot change frozen experimental design: {key}")
    if method_design(matrix["methods"]) != method_design(old["methods"]):
        raise ValueError("v4 cannot change frozen experimental design: methods")
    binding = matrix["sources"]["training_config_v2"]
    if binding != {"logical_path": CONFIG_PATH, "sha256": sha(ROOT / CONFIG_PATH)}:
        raise ValueError("v4 config binding mismatch")
    for role, source in matrix["sources"].items():
        if sha(ROOT / source["logical_path"]) != source["sha256"]:
            raise ValueError(f"v4 live asset mismatch: {role}")
    return matrix


def build_release():
    load_verified_env_release()
    load_config()
    load_matrix()
    if _git("status", "--porcelain", "--", *SOURCE_PATHS):
        raise ValueError("inventory release source closure must be committed")
    return {
        "schema": "idc-formal-train-release-v2", "inventory_version": SEMANTICS,
        "reward_semantics": "common-sgd-degradation-v1", "approved_decision_id": "M6-P2b",
        "release_iteration": "r4",
        "supersedes": {"path": "configs/release/idc_formal_train_release_v2_r2.json",
                       "sha256": sha(ROOT / "configs/release/idc_formal_train_release_v2_r2.json")},
        "readiness": {"formal_env_ready": True, "formal_training_ready": True},
        "assets": {role: {"path": path, "sha256": sha(ROOT / path)}
                   for role, path in ASSET_PATHS.items()},
        "semantics_binding": semantics_binding(),
        "release_revision": _git("log", "-1", "--format=%H", "--", *SOURCE_PATHS),
    }


def verify_release():
    recorded = json.loads((ROOT / RELEASE_PATH).read_text())
    if recorded != build_release():
        raise ValueError("inventory release v2 does not match live code/assets")
    return recorded


def checkpoint_binding():
    release = verify_release()
    config = load_config()
    return {"release_path": RELEASE_PATH, "release_sha256": sha(ROOT / RELEASE_PATH),
            "inventory_version": SEMANTICS, "reward_semantics": release["reward_semantics"],
            "service_guard_version": config["training"]["corrector"]["service_guard_version"],
            "observation_version": config["training"]["corrector"]["observation_version"],
            "observation_dimension": config["training"]["policy"]["obs_dim"],
            "training_config_sha256": sha(ROOT / CONFIG_PATH),
            "experiment_matrix_sha256": sha(ROOT / MATRIX_PATH),
            "semantics_binding": release["semantics_binding"]}

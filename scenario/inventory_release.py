"""M6-P2b v2 release: bind the full training/corrector semantics, retain v1 assets."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from evaluation.sources import ROLE_LOGICAL_PATHS
from scenario.env_release import load_verified_env_release

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = "configs/training/idc_training_config_v2.json"
MATRIX_PATH = "configs/experiments/m9_experiment_matrix_v4.json"
RELEASE_PATH = "configs/release/idc_formal_train_release_v2.json"
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
)
ASSET_PATHS = {**ROLE_LOGICAL_PATHS, "training_config": CONFIG_PATH,
               "experiment_matrix": MATRIX_PATH}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def semantics_binding():
    return {path: sha(ROOT / path) for path in SOURCE_PATHS}


def load_config():
    config = json.loads((ROOT / CONFIG_PATH).read_text())
    if (config.get("schema") != "idc-training-config-v2"
            or config.get("status") != "frozen" or config.get("version") != "v2"):
        raise ValueError("frozen inventory training config v2 is required")
    corrector = config["training"]["corrector"]
    if (corrector["inventory_version"] != SEMANTICS or corrector["time_limit_s"] != .25
            or config["reward_semantics"] != "original-env-reward-v1"):
        raise ValueError("unregistered inventory/reward/budget semantics")
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
        if key in ("schema", "status", "note", "corrector"):
            continue
        if config["training"].get(key) != value:
            raise ValueError(f"v2 config differs from measured calibration: {key}")
    if any(not row["match"] for row in live_asset_hash_check(config).values()):
        raise ValueError("v2 calibration live asset mismatch")
    return config


def load_matrix():
    matrix = json.loads((ROOT / MATRIX_PATH).read_text())
    old = json.loads((ROOT / "configs/experiments/m9_experiment_matrix_v3.json").read_text())
    if matrix.get("schema") != "m9-experiment-matrix-v4":
        raise ValueError("inventory experiment matrix v4 is required")
    # All substantive preregistered experimental design remains unchanged.
    for key in ("split_freeze", "scenario_freeze", "training_schedule", "evaluation_schedule",
                "pairing", "statistics", "methods", "fair_cost_carbon_pairing"):
        if key in old and matrix.get(key) != old[key]:
            raise ValueError(f"v4 cannot change frozen experimental design: {key}")
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
        "reward_semantics": "original-env-reward-v1", "approved_decision_id": "M6-P2b",
        "supersedes": {"path": "configs/release/idc_formal_train_release_v1.json",
                       "sha256": sha(ROOT / "configs/release/idc_formal_train_release_v1.json")},
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
    return {"release_path": RELEASE_PATH, "release_sha256": sha(ROOT / RELEASE_PATH),
            "inventory_version": SEMANTICS, "reward_semantics": release["reward_semantics"],
            "training_config_sha256": sha(ROOT / CONFIG_PATH),
            "experiment_matrix_sha256": sha(ROOT / MATRIX_PATH),
            "semantics_binding": release["semantics_binding"]}

"""New formal evaluation input binds inventory/reward/training release semantics."""

from __future__ import annotations

from pathlib import Path

from checkpointing import VersionedCheckpoint
from checkpointing.versioned import read_checkpoint_payload
from safe_rl_v2.formal_train_loop import build_seeded_policy
from scenario.inventory_release import checkpoint_binding, load_config

SCHEMA = "m6p2b-eval-input-v2"
FORMAL_SCHEMA = "m6p2b-formal-train-resume-v2"
SHORT_SCHEMA = "m6p2b-controlled-resume-v2"


def load_policy(path, *, formal=True):
    config = load_config()
    ckpt = VersionedCheckpoint.load(
        path, expected_action_dim=21, expected_obs_dim=520, expected_schema_hash=SCHEMA)
    role = "formal_training_policy" if formal else "controlled_short_run_eval_input"
    if ckpt.extras != {"inventory_binding": checkpoint_binding(), "artifact_role": role}:
        raise ValueError("evaluation input inventory semantics/role mismatch")
    if set(ckpt.state) != {"policy"}:
        raise ValueError("evaluation input must contain only policy weights")
    policy = build_seeded_policy(config, obs_dim=520, seed=0)
    policy.load_state_dict(ckpt.state["policy"], strict=True)
    policy.eval()
    return policy


def export_policy(source, destination, *, formal=True):
    expected_schema = FORMAL_SCHEMA if formal else SHORT_SCHEMA
    ckpt = VersionedCheckpoint.load(
        source, expected_action_dim=21, expected_obs_dim=520,
        expected_schema_hash=expected_schema)
    binding = checkpoint_binding()
    if ckpt.extras != {"inventory_binding": binding}:
        raise ValueError("training checkpoint lacks matching inventory semantics")
    if ckpt.state["frozen_config"] != load_config():
        raise ValueError("training checkpoint config mismatch")
    scope = "formal_training" if formal else "controlled_short_run"
    source_role = "formal_training_resume" if formal else "controlled_training_resume"
    if (ckpt.state.get("training_scope") != scope
            or ckpt.state.get("artifact_role") != source_role
            or ckpt.state.get("code_revision") != ckpt.code_revision):
        raise ValueError("training checkpoint scope/role/revision mismatch")
    role = "formal_training_policy" if formal else "controlled_short_run_eval_input"
    output = Path(destination)
    if output.exists():
        raise FileExistsError("evaluation input cannot overwrite an existing artifact")
    VersionedCheckpoint(
        contract_version_id=ckpt.contract_version_id, action_dim=21, obs_dim=520,
        schema_hash=SCHEMA, code_revision=ckpt.code_revision,
        state={"policy": ckpt.state["policy"]},
        extras={"inventory_binding": binding, "artifact_role": role}).save(output)
    loaded = load_policy(output, formal=formal)
    original = read_checkpoint_payload(source)["state"]["policy"]
    if any(not original[k].equal(v) for k, v in loaded.state_dict().items()):
        raise ValueError("export changed policy weights")
    return loaded

"""M2.3 版本化 checkpoint 测试：round-trip、版本/维度/schema 拒绝。"""

import pytest
import torch

from checkpointing import (
    CURRENT_ACTION_DIM,
    CURRENT_CONTRACT_VERSION,
    CheckpointVersionError,
    VersionedCheckpoint,
)

_EXPECTED_OBS_DIM = 128
_EXPECTED_SCHEMA = "schema-abc"


def _ckpt(
    action_dim: int = CURRENT_ACTION_DIM, schema_hash: str = _EXPECTED_SCHEMA
) -> VersionedCheckpoint:
    return VersionedCheckpoint(
        contract_version_id=CURRENT_CONTRACT_VERSION,
        action_dim=action_dim,
        obs_dim=_EXPECTED_OBS_DIM,
        schema_hash=schema_hash,
        code_revision="test-rev",
        state={
            "weights": {"w": torch.tensor([1.0, 2.0, 3.0])},
            "multipliers": {"business": 0.5, "carbon": 1.5},
            "rng": torch.tensor([42, 43]),
        },
    )


def _load(
    path,
    *,
    action_dim: int = CURRENT_ACTION_DIM,
    obs_dim: int = _EXPECTED_OBS_DIM,
    schema: str = _EXPECTED_SCHEMA,
) -> VersionedCheckpoint:
    return VersionedCheckpoint.load(
        path,
        expected_action_dim=action_dim,
        expected_obs_dim=obs_dim,
        expected_schema_hash=schema,
    )


def test_roundtrip_preserves_state(tmp_path):
    path = tmp_path / "ckpt.pt"
    _ckpt().save(path)
    loaded = _load(path)
    assert loaded.contract_version_id == CURRENT_CONTRACT_VERSION
    assert torch.equal(loaded.state["weights"]["w"], torch.tensor([1.0, 2.0, 3.0]))
    assert loaded.state["multipliers"] == {"business": 0.5, "carbon": 1.5}
    assert torch.equal(loaded.state["rng"], torch.tensor([42, 43]))


def test_rejects_missing_metadata(tmp_path):
    path = tmp_path / "no_meta.pt"
    torch.save({"state": {}}, str(path))
    with pytest.raises(CheckpointVersionError, match="无版本"):
        _load(path)


def test_rejects_old_23_dim(tmp_path):
    path = tmp_path / "old.pt"
    _ckpt(action_dim=23).save(path)
    with pytest.raises(CheckpointVersionError, match="action_dim"):
        _load(path)


def test_rejects_schema_mismatch(tmp_path):
    path = tmp_path / "bad_schema.pt"
    _ckpt(schema_hash="other").save(path)
    with pytest.raises(CheckpointVersionError, match="schema_hash"):
        _load(path)


def test_rejects_wrong_version(tmp_path):
    path = tmp_path / "wrong_ver.pt"
    ckpt = _ckpt()
    ckpt.contract_version_id = "old-version"
    ckpt.save(path)
    with pytest.raises(CheckpointVersionError, match="contract_version_id"):
        _load(path)

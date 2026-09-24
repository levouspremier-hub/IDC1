"""M6-P1：评估输入 checkpoint 契约（保存 / 读取 / 拒绝）。

**先红**：`checkpointing/eval_input.py` 尚不存在（`ModuleNotFoundError`），
且现有 `VersionedCheckpoint` 只承载
`contract_version_id / action_dim / obs_dim / schema_hash / code_revision`，
**没有**动作模式、策略配置、资产来源、train origin 与种子 ⇒ 无法证明
「评估可读 checkpoint 的版本、21 维动作、观测维度、policy 权重、配置与资产来源、
train origin、种子、代码 revision」已被稳定。

本文件覆盖两件事：

1. **可读**：21 维 checkpoint 能保存、读取，并**复现固定动作**；
2. **拒绝**：23 维 / 版本 / obs 维度 / schema / 来源缺失或畸形 / 非 train origin /
   元数据缺键或多键 / 未知状态键 / two-batch 恢复格式冒充，一律 **fail closed**。
"""

import importlib

import numpy as np
import pytest
import torch

from checkpointing import CURRENT_CONTRACT_VERSION, CheckpointVersionError, VersionedCheckpoint

OBS_DIM = 12
ACTION_DIM = 21
SCHEMA = "m6p1-eval-input-v1"

SOURCE_ROLES = (
    "canonical_parquet",
    "canonical_manifest",
    "truth_split_manifest",
    "forecast_policy_v3",
    "b6_arrival_policy",
    "exogenous_v3_manifest",
    "exogenous_v3_source_manifest",
    "exogenous_v3_parquet",
    "refs_v4",
    "formal_split_manifest_train",
    "arrival_mapper_policy",
    "env_release",
)
_SHA = "a" * 64


def _module():
    return importlib.import_module("checkpointing.eval_input")


def _policy(seed: int = 0, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM):
    from safe_rl_v2.policy import SafePPOPolicy

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return SafePPOPolicy(obs_dim=obs_dim, action_dim=action_dim)


def _sources(role_count: int = len(SOURCE_ROLES)):
    return [
        {"role": role, "logical_path": f"data/manifest/{role}.json", "sha256": _SHA}
        for role in SOURCE_ROLES[:role_count]
    ]


_UNSET = object()


def _save_kwargs(policy=_UNSET, **overrides):
    # 哨兵而非 `None`：`policy=None` 必须**原样**传给保存路径（用于拒绝用例），
    # 不能被 helper 悄悄替换成一份默认权重。
    policy = _policy() if policy is _UNSET else policy
    kwargs = {
        "policy": policy,
        "obs_dim": OBS_DIM,
        "code_revision": "0" * 40,
        "artifact_role": _module().CONTROLLED_ROLE,
        "action_mode": "deterministic_mean",
        "policy_config": {
            "hidden": 64,
            "hidden_layers": 1,
            "activation": "Tanh",
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
        },
        "train_split": "train",
        "train_origin": 48,
        "train_start": "2024-01-02T00:00:00+08:00",
        "seeds": {"task": 0, "server": 1, "forecast": 300000},
        "sources": _sources(),
    }
    kwargs.update(overrides)
    return kwargs


def _save(tmp_path, name="eval_input.pt", **overrides):
    path = tmp_path / name
    _module().save_evaluation_checkpoint(path, **_save_kwargs(**overrides))
    return path


def _load(path, **kwargs):
    return _module().load_evaluation_checkpoint(path, **kwargs)


# =============================================================================
# 1. 可读：保存 / 读取 / 复现固定动作
# =============================================================================

def test_roundtrip_saves_and_reads_the_contract(tmp_path):
    path = _save(tmp_path)
    ckpt = _load(path)

    assert ckpt.envelope.contract_version_id == CURRENT_CONTRACT_VERSION
    assert ckpt.envelope.action_dim == ACTION_DIM
    assert ckpt.envelope.obs_dim == OBS_DIM
    assert ckpt.envelope.schema_hash == SCHEMA
    assert ckpt.artifact_role == _module().CONTROLLED_ROLE
    assert ckpt.action_mode == "deterministic_mean"
    assert ckpt.train_split == "train"
    assert ckpt.train_origin == 48
    assert ckpt.seeds == {"task": 0, "server": 1, "forecast": 300000}
    assert len(ckpt.sources) == len(SOURCE_ROLES)
    assert {d.role for d in ckpt.sources} == set(SOURCE_ROLES)


def test_loaded_checkpoint_reproduces_fixed_actions(tmp_path):
    """同一条 obs 上，两次独立读取给出**逐位相同**的固定动作。"""
    path = _save(tmp_path)
    obs = np.linspace(-1.0, 1.0, OBS_DIM, dtype=np.float32)

    a1 = _load(path).policy.act_mean(torch.as_tensor(obs)).detach().numpy()
    a2 = _load(path).policy.act_mean(torch.as_tensor(obs)).detach().numpy()

    assert a1.shape == (ACTION_DIM,)
    assert np.array_equal(a1, a2), "读取后的固定动作必须可复现"
    # 非空洞性：换一份权重必须给出不同动作
    other = _load(_save(tmp_path, "other.pt", policy=_policy(seed=7)))
    a3 = other.policy.act_mean(torch.as_tensor(obs)).detach().numpy()
    assert not np.array_equal(a1, a3)


def test_envelope_is_a_versioned_checkpoint(tmp_path):
    ckpt = _load(_save(tmp_path))
    assert isinstance(ckpt.envelope, VersionedCheckpoint)


def test_declares_itself_as_a_controlled_short_run_input(tmp_path):
    ckpt = _load(_save(tmp_path))
    assert ckpt.is_controlled_short_run is True
    assert ckpt.artifact_role != "formal_training_policy"
    assert _load(_save(tmp_path, "formal.pt",
                       artifact_role="formal_training_policy")).is_controlled_short_run is False


# =============================================================================
# 2. 拒绝：维度 / 版本 / schema
# =============================================================================

def test_rejects_old_23_dim_action_space(tmp_path):
    """旧 23 维动作空间一律拒绝：写入侧不得落盘，读取侧不得放行。"""
    with pytest.raises((CheckpointVersionError, ValueError), match="action_dim"):
        _save(tmp_path, "old23.pt", policy=_policy(action_dim=23),
              policy_config={"hidden": 64, "hidden_layers": 1, "activation": "Tanh",
                             "obs_dim": OBS_DIM, "action_dim": 23})
    assert not (tmp_path / "old23.pt").exists()

    path = _save(tmp_path, "tampered23.pt")
    payload = torch.load(str(path), weights_only=False)
    payload["metadata"]["action_dim"] = 23
    torch.save(payload, str(path))
    with pytest.raises((CheckpointVersionError, ValueError), match="action_dim"):
        _load(path)


def test_rejects_wrong_obs_dim(tmp_path):
    """写入侧与读取侧都必须拒绝 obs 维度不符（不得写出读不回来的件）。"""
    with pytest.raises((CheckpointVersionError, ValueError), match="obs_dim"):
        _save(tmp_path, "obs.pt", obs_dim=OBS_DIM + 1)
    assert not (tmp_path / "obs.pt").exists()

    # 读取侧：把一个**合法**件的 obs_dim 改坏，必须拒绝
    path = _save(tmp_path, "tampered.pt")
    payload = torch.load(str(path), weights_only=False)
    payload["metadata"]["obs_dim"] = OBS_DIM + 1
    torch.save(payload, str(path))
    with pytest.raises((CheckpointVersionError, ValueError), match="obs_dim"):
        _load(path)


def test_rejects_wrong_contract_version(tmp_path):
    path = tmp_path / "old_ver.pt"
    VersionedCheckpoint(
        contract_version_id=CURRENT_CONTRACT_VERSION, action_dim=ACTION_DIM, obs_dim=OBS_DIM,
        schema_hash=SCHEMA, code_revision="0" * 40, state={"policy": {}},
    ).save(path)

    payload = torch.load(str(path), weights_only=False)
    payload["metadata"]["contract_version_id"] = "contract-v8"
    torch.save(payload, str(path))
    with pytest.raises((CheckpointVersionError, ValueError), match="contract_version_id"):
        _load(path)


def test_rejects_non_evaluation_schema(tmp_path):
    """schema_hash 必须是评估输入 schema，不能拿别的格式冒充。"""
    path = tmp_path / "bad_schema.pt"
    VersionedCheckpoint(
        contract_version_id=CURRENT_CONTRACT_VERSION, action_dim=ACTION_DIM, obs_dim=OBS_DIM,
        schema_hash="something-else-v1", code_revision="0" * 40, state={"policy": {}},
    ).save(path)
    with pytest.raises((CheckpointVersionError, ValueError), match="schema_hash"):
        _load(path)


def test_rejects_unversioned_payload(tmp_path):
    path = tmp_path / "nover.pt"
    torch.save({"state": {"policy": {}}}, str(path))
    with pytest.raises((CheckpointVersionError, ValueError), match="无版本"):
        _load(path)


def test_rejects_two_batch_resume_format(tmp_path):
    """`safe_rl_v2/ppo_two_batch.py` 的恢复格式**不得**被当作评估输入。"""
    from safe_rl_v2.ppo_two_batch import CHECKPOINT_SCHEMA, CHECKPOINT_STATE_KEYS

    path = tmp_path / "two_batch.pt"
    VersionedCheckpoint(
        contract_version_id=CURRENT_CONTRACT_VERSION, action_dim=ACTION_DIM, obs_dim=OBS_DIM,
        schema_hash=CHECKPOINT_SCHEMA, code_revision="0" * 40,
        state={key: {} for key in CHECKPOINT_STATE_KEYS},
    ).save(path)
    with pytest.raises((CheckpointVersionError, ValueError)):
        _load(path)


# =============================================================================
# 3. 拒绝：元数据键集合、动作模式、角色
# =============================================================================

def test_rejects_missing_metadata_key(tmp_path):
    path = _save(tmp_path)
    payload = torch.load(str(path), weights_only=False)
    del payload["metadata"]["train_origin"]
    torch.save(payload, str(path))
    with pytest.raises(ValueError, match="train_origin"):
        _load(path)


def test_rejects_extra_metadata_key(tmp_path):
    path = _save(tmp_path)
    payload = torch.load(str(path), weights_only=False)
    payload["metadata"]["bonus"] = 1
    torch.save(payload, str(path))
    with pytest.raises(ValueError, match="bonus"):
        _load(path)


def test_rejects_unknown_artifact_role(tmp_path):
    path = tmp_path / "role.pt"
    with pytest.raises(ValueError, match="artifact_role"):
        _module().save_evaluation_checkpoint(
            path, **_save_kwargs(artifact_role="looks_legit"))


def test_rejects_unknown_action_mode(tmp_path):
    path = tmp_path / "mode.pt"
    with pytest.raises(ValueError, match="action_mode"):
        _module().save_evaluation_checkpoint(
            path, **_save_kwargs(action_mode="whatever"))


def test_rejects_non_train_split(tmp_path):
    """评估输入**只能**来自 train split；validation/test 一律拒绝。"""
    for split in ("validation", "test"):
        path = tmp_path / f"{split}.pt"
        with pytest.raises(ValueError, match="train"):
            _module().save_evaluation_checkpoint(path, **_save_kwargs(train_split=split))
        assert not path.exists()


def test_rejects_bool_or_non_int_seed(tmp_path):
    for seeds in ({"task": True, "server": 1, "forecast": 300000},
                  {"task": 0, "server": 1.0, "forecast": 300000},
                  {"task": 0, "server": 1, "forecast": "300000"}):
        path = tmp_path / "seeds.pt"
        with pytest.raises(ValueError, match="seed"):
            _module().save_evaluation_checkpoint(path, **_save_kwargs(seeds=seeds))
        assert not path.exists()


def test_rejects_missing_seed_key(tmp_path):
    path = tmp_path / "seeds2.pt"
    with pytest.raises(ValueError, match="seed"):
        _module().save_evaluation_checkpoint(
            path, **_save_kwargs(seeds={"task": 0, "server": 1}))


# =============================================================================
# 4. 拒绝：资产来源
# =============================================================================

def test_rejects_missing_required_source_role(tmp_path):
    path = tmp_path / "src.pt"
    with pytest.raises(ValueError, match="env_release"):
        _module().save_evaluation_checkpoint(
            path, **_save_kwargs(sources=_sources(len(SOURCE_ROLES) - 1)))


def test_rejects_duplicate_source_role(tmp_path):
    dup = _sources()
    dup[-1] = dict(dup[0])
    path = tmp_path / "dup.pt"
    with pytest.raises(ValueError, match="role"):
        _module().save_evaluation_checkpoint(path, **_save_kwargs(sources=dup))


def test_rejects_non_canonical_source_path(tmp_path):
    for bad in ("/abs/path.json", "data/../x.json", "./x.json", "data//x.json", " lead.json"):
        sources = _sources()
        sources[0] = dict(sources[0], logical_path=bad)
        path = tmp_path / "path.pt"
        with pytest.raises(ValueError, match="logical_path"):
            _module().save_evaluation_checkpoint(path, **_save_kwargs(sources=sources))
        assert not path.exists()


def test_rejects_malformed_source_sha256(tmp_path):
    for bad in ("", "0" * 63, "A" * 64, "z" * 64):
        sources = _sources()
        sources[0] = dict(sources[0], sha256=bad)
        path = tmp_path / "sha.pt"
        with pytest.raises(ValueError, match="sha256"):
            _module().save_evaluation_checkpoint(path, **_save_kwargs(sources=sources))
        assert not path.exists()


def test_rejects_empty_sources(tmp_path):
    path = tmp_path / "empty.pt"
    with pytest.raises(ValueError, match="sources"):
        _module().save_evaluation_checkpoint(path, **_save_kwargs(sources=[]))


# =============================================================================
# 5. 拒绝：策略状态与配置
# =============================================================================

def test_rejects_missing_policy_state(tmp_path):
    path = tmp_path / "nopolicy.pt"
    with pytest.raises(ValueError, match="policy"):
        _module().save_evaluation_checkpoint(
            path, **_save_kwargs(policy=None))


def test_rejects_unknown_state_key(tmp_path):
    path = _save(tmp_path)
    payload = torch.load(str(path), weights_only=False)
    payload["state"]["surprise"] = torch.tensor([1.0])
    torch.save(payload, str(path))
    with pytest.raises(ValueError, match="surprise"):
        _load(path)


def test_rejects_policy_config_that_contradicts_the_state_shapes(tmp_path):
    path = tmp_path / "cfg.pt"
    with pytest.raises(ValueError, match="hidden"):
        _module().save_evaluation_checkpoint(path, **_save_kwargs(
            policy_config={"hidden": 128, "hidden_layers": 1, "activation": "Tanh",
                           "obs_dim": OBS_DIM, "action_dim": ACTION_DIM}))


def test_rejects_policy_shapes_that_do_not_match_the_obs_dim(tmp_path):
    """权重张量与声明的 obs_dim 不符时必须拒绝（不得靠 `strict=False` 蒙混）。"""
    path = _save(tmp_path)
    payload = torch.load(str(path), weights_only=False)
    payload["state"]["policy"]["actor.0.weight"] = torch.zeros(64, OBS_DIM + 5)
    torch.save(payload, str(path))
    with pytest.raises(ValueError):
        _load(path)


def test_saved_file_layout_is_metadata_plus_state(tmp_path):
    path = _save(tmp_path)
    payload = torch.load(str(path), weights_only=False)
    assert set(payload) == {"metadata", "state"}
    assert set(payload["state"]) == {"policy"}

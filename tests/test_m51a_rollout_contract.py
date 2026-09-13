"""M5.1a 测试：版本化 RolloutBuffer 契约（唯一训练样本格式）。

不测 PPO 数学；只测数据契约、防御性复制、严格序列化与版本拒绝。
"""

import json

import numpy as np
import pytest

from contracts import CONTRACT_VERSION_ID
from safe_rl_v2.buffer import ACTION_DIM, CONTRACT_VERSION, RolloutBuffer, Transition

OBS_DIM = 280


def _obs(seed: int = 0) -> np.ndarray:
    return np.full(OBS_DIM, seed, dtype=np.float32)


def _raw(v: float = 0.5) -> np.ndarray:
    a = np.full(ACTION_DIM, v, dtype=np.float32)
    a[20] = 0.0
    return a


def _transition(**over) -> Transition:
    kwargs = dict(
        observation=_obs(1),
        next_observation=_obs(2),
        raw_action=_raw(0.5),
        old_raw_log_prob=-1.25,
        exec_action=_raw(0.4),
        reward=0.1,
        business_cost=0.5,
        carbon_cost=0.3,
        terminated=False,
        truncated=False,
        correction_info={"reason": "access_limit", "offset": 0.01},
    )
    kwargs.update(over)
    return Transition(**kwargs)


# --- 1. raw / exec 分离 ---

def test_raw_and_exec_stored_separately_and_verbatim():
    buf = RolloutBuffer()
    raw, exec_a = _raw(0.5), _raw(0.4)
    buf.add(_transition(raw_action=raw, exec_action=exec_a))
    t = buf.transitions[0]
    assert not np.allclose(t.raw_action, t.exec_action)
    np.testing.assert_allclose(t.raw_action, raw, atol=0.0)
    np.testing.assert_allclose(t.exec_action, exec_a, atol=0.0)
    assert t.old_raw_log_prob == -1.25


def test_no_exec_log_prob_anywhere():
    buf = RolloutBuffer()
    buf.add(_transition())
    payload = buf.to_dict()
    assert "exec_log_prob" not in json.dumps(payload)
    for t in buf.transitions:
        assert not hasattr(t, "exec_log_prob")
    import inspect

    import safe_rl_v2.buffer as mod

    assert "exec_log_prob" not in inspect.getsource(mod)


# --- 2. 防御性复制 ---

def test_defensive_copy_of_arrays_and_correction_info():
    buf = RolloutBuffer()
    raw, exec_a, obs, nobs = _raw(0.5), _raw(0.4), _obs(1), _obs(2)
    info = {"reason": "access_limit", "nested": {"k": [1, 2]}}
    buf.add(
        _transition(
            observation=obs, next_observation=nobs, raw_action=raw,
            exec_action=exec_a, correction_info=info,
        )
    )
    # 调用方随后修改输入
    raw[:] = 9.0
    exec_a[:] = 9.0
    obs[:] = 9.0
    nobs[:] = 9.0
    info["reason"] = "MUTATED"
    info["nested"]["k"].append(3)
    t = buf.transitions[0]
    assert t.raw_action[0] != 9.0
    assert t.exec_action[0] != 9.0
    assert t.observation[0] != 9.0
    assert t.next_observation[0] != 9.0
    assert t.correction_info["reason"] == "access_limit"
    assert t.correction_info["nested"]["k"] == [1, 2]


# --- 3. 严格序列化往返 ---

def test_serialization_roundtrip_strict():
    buf = RolloutBuffer()
    buf.add(_transition(terminated=True, truncated=False))
    buf.add(_transition(terminated=False, truncated=True, reward=0.2))
    restored = RolloutBuffer.from_dict(json.loads(json.dumps(buf.to_dict())))
    assert len(restored) == 2
    for a, b in zip(buf.transitions, restored.transitions, strict=True):
        np.testing.assert_allclose(a.raw_action, b.raw_action, atol=0.0)
        np.testing.assert_allclose(a.exec_action, b.exec_action, atol=0.0)
        np.testing.assert_allclose(a.observation, b.observation, atol=0.0)
        np.testing.assert_allclose(a.next_observation, b.next_observation, atol=0.0)
        assert (a.terminated, a.truncated) == (b.terminated, b.truncated)
        assert a.correction_info == b.correction_info
        assert a.old_raw_log_prob == b.old_raw_log_prob


def test_payload_is_json_serializable():
    buf = RolloutBuffer()
    buf.add(_transition())
    assert isinstance(json.dumps(buf.to_dict()), str)


# --- 4. 严格拒绝 ---

def test_rejects_wrong_action_dim():
    buf = RolloutBuffer()
    with pytest.raises(ValueError, match="raw_action"):
        buf.add(_transition(raw_action=np.zeros(23, dtype=np.float32)))


def test_rejects_inconsistent_observation_dim():
    buf = RolloutBuffer()
    with pytest.raises(ValueError, match="observation"):
        buf.add(_transition(next_observation=np.zeros(OBS_DIM + 1, dtype=np.float32)))


def test_rejects_non_finite_values():
    buf = RolloutBuffer()
    bad = _raw(0.5)
    bad[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        buf.add(_transition(raw_action=bad))
    with pytest.raises(ValueError, match="finite"):
        buf.add(_transition(old_raw_log_prob=float("nan")))


def test_rejects_top_level_and_entry_version_mismatch():
    buf = RolloutBuffer()
    buf.add(_transition())
    payload = buf.to_dict()
    payload["contract_version"] = "contract-v5"
    with pytest.raises(ValueError, match="version"):
        RolloutBuffer.from_dict(payload)


def test_rejects_old_v5_payload():
    buf = RolloutBuffer()
    buf.add(_transition())
    payload = buf.to_dict()
    payload["contract_version"] = "contract-v5"
    for t in payload["transitions"]:
        t["contract_version"] = "contract-v5"
    with pytest.raises(ValueError, match="version"):
        RolloutBuffer.from_dict(payload)


def test_rejects_missing_version():
    buf = RolloutBuffer()
    buf.add(_transition())
    payload = buf.to_dict()
    del payload["contract_version"]
    with pytest.raises((ValueError, KeyError)):
        RolloutBuffer.from_dict(payload)


def test_rejects_missing_fields_without_defaults():
    buf = RolloutBuffer()
    buf.add(_transition())
    payload = buf.to_dict()
    del payload["transitions"][0]["next_observation"]
    with pytest.raises((ValueError, KeyError)):
        RolloutBuffer.from_dict(payload)


def test_rejects_non_serializable_correction_info():
    buf = RolloutBuffer()
    import torch

    with pytest.raises((TypeError, ValueError)):
        buf.add(_transition(correction_info={"t": torch.zeros(2)}))


# --- 5. 版本与常量 ---

def test_contract_version_is_v6():
    assert CONTRACT_VERSION_ID == "contract-v6"
    assert CONTRACT_VERSION == CONTRACT_VERSION_ID


def test_transition_dimensions_and_flags():
    t = _transition(terminated=True, truncated=True)
    assert t.observation.shape == (OBS_DIM,)
    assert t.next_observation.shape == (OBS_DIM,)
    assert t.raw_action.shape == (ACTION_DIM,)
    assert t.exec_action.shape == (ACTION_DIM,)
    assert t.terminated is True and t.truncated is True

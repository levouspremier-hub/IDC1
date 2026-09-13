"""M5.1 测试：buffer 同存 raw/exec，21 维，序列化往返。"""

import numpy as np
import pytest

from safe_rl_v2.buffer import RolloutBuffer, Transition


def _obs() -> np.ndarray:
    return np.zeros(280, dtype=np.float32)


def _transition(raw: np.ndarray, exec_a: np.ndarray) -> Transition:
    return Transition(
        observation=_obs(),
        raw_action=raw,
        raw_log_prob=-1.5,
        reward=0.1,
        business_cost=0.5,
        carbon_cost=0.3,
        exec_action=exec_a,
        correction_info={"reason": "access_limit"},
    )


def test_buffer_stores_raw_exec_separately():
    buf = RolloutBuffer()
    raw = np.concatenate([np.full(20, 0.5), [0.3]])
    exec_a = np.concatenate([np.full(20, 0.4), [0.2]])
    buf.add(_transition(raw, exec_a))
    t = buf.transitions[0]
    assert not np.allclose(t.raw_action, t.exec_action)  # exec 不覆盖 raw
    assert t.raw_action.shape == (21,)
    assert t.exec_action.shape == (21,)
    assert t.contract_version == "contract-v3"


def test_buffer_rejects_wrong_dim():
    buf = RolloutBuffer()
    raw23 = np.zeros(23)
    exec21 = np.zeros(21)
    with pytest.raises(ValueError):
        buf.add(_transition(raw23, exec21))


def test_buffer_serialization_roundtrip():
    buf = RolloutBuffer()
    raw = np.concatenate([np.full(20, 0.5), [0.3]])
    exec_a = np.concatenate([np.full(20, 0.4), [0.2]])
    buf.add(_transition(raw, exec_a))
    restored = RolloutBuffer.from_dict(buf.to_dict())
    assert len(restored) == 1
    t = restored.transitions[0]
    assert np.allclose(t.raw_action, raw)
    assert np.allclose(t.exec_action, exec_a)
    assert t.raw_log_prob == -1.5

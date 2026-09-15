"""M5.1 测试（M5.1a 迁移）：buffer 同存 raw/exec、21 维、严格序列化往返。

M5.1a 起 `Transition` 新增 next_observation / terminated / truncated / electricity_cost_sgd；
**M1.3e 起**全仓唯一契约版本为 `contract-v8`（本文件只把版本断言随之迁移，
不自带版本字面量）；严格契约细节由 `tests/test_m51a_rollout_contract.py` 覆盖。
"""

import numpy as np
import pytest

from contracts import CONTRACT_VERSION_ID
from safe_rl_v2.buffer import ACTION_DIM, RolloutBuffer, Transition

OBS_DIM = 280


def _transition(raw: np.ndarray, exec_a: np.ndarray) -> Transition:
    return Transition(
        observation=np.zeros(OBS_DIM, dtype=np.float32),
        next_observation=np.ones(OBS_DIM, dtype=np.float32),
        raw_action=raw,
        old_raw_log_prob=-1.5,
        exec_action=exec_a,
        reward=0.1,
        business_cost=3.0,  # 违规计数
        carbon_cost=12.5,  # kgCO2e
        electricity_cost_sgd=120.0,  # SGD
        terminated=False,
        truncated=False,
        correction_info={"reason": "access_limit"},
    )


def test_buffer_stores_raw_exec_separately():
    buf = RolloutBuffer()
    raw = np.concatenate([np.full(20, 0.5), [0.3]])
    exec_a = np.concatenate([np.full(20, 0.4), [0.2]])
    buf.add(_transition(raw, exec_a))
    t = buf.transitions[0]
    assert not np.allclose(t.raw_action, t.exec_action)  # exec 不覆盖 raw
    assert t.raw_action.shape == (ACTION_DIM,)
    assert t.exec_action.shape == (ACTION_DIM,)
    assert t.contract_version == CONTRACT_VERSION_ID


def test_buffer_rejects_wrong_dim():
    buf = RolloutBuffer()
    with pytest.raises(ValueError):
        buf.add(_transition(np.zeros(23), np.zeros(21)))


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
    assert t.old_raw_log_prob == -1.5

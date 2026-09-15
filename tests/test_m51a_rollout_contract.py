"""M5.1a 测试：版本化 RolloutBuffer 契约（唯一训练样本格式）。

不测 PPO 数学；只测数据契约、防御性复制、严格序列化与版本拒绝。
"""

import dataclasses
import inspect
import json
from typing import Any

import numpy as np
import pytest

from contracts import CONTRACT_VERSION_ID
from safe_rl_v2 import buffer as buffer_mod
from safe_rl_v2.buffer import ACTION_DIM, CONTRACT_VERSION, RolloutBuffer, Transition

OBS_DIM = 280

# v5 payload 字面量（M5.1a 前的真实落盘格式）：无 next_observation/terminated/truncated，
# log-prob 字段名为 raw_log_prob，顶层无 action_dim/units。用于拒绝证据。
V5_PAYLOAD = {
    "contract_version": "contract-v5",
    "transitions": [
        {
            "observation": [0.0] * OBS_DIM,
            "raw_action": [0.5] * ACTION_DIM,
            "raw_log_prob": -1.25,
            "reward": 0.1,
            "business_cost": 0.5,
            "carbon_cost": 0.3,
            "exec_action": [0.4] * ACTION_DIM,
            "correction_info": {"reason": "access_limit"},
            "contract_version": "contract-v5",
        }
    ],
}


def _obs(seed: int = 0) -> np.ndarray:
    return np.full(OBS_DIM, seed, dtype=np.float32)


def _raw(v: float = 0.5) -> np.ndarray:
    a = np.full(ACTION_DIM, v, dtype=np.float32)
    a[20] = 0.0
    return a


def _transition(**over: Any) -> Transition:
    kwargs: dict[str, Any] = dict(
        observation=_obs(1),
        next_observation=_obs(2),
        raw_action=_raw(0.5),
        old_raw_log_prob=-1.25,
        exec_action=_raw(0.4),
        reward=0.1,
        business_cost=3.0,
        carbon_cost=12.5,
        electricity_cost_sgd=120.0,
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


def _exec_prob_like(names) -> list[str]:
    """「执行动作的概率」类字段名：同时含 exec 与 prob 的标识符。"""
    return [n for n in names if "exec" in n.lower() and "prob" in n.lower()]


def test_execution_action_carries_no_log_prob_field():
    """exec 只审计：dataclass、payload、运行时 API 均不得存在 exec 侧 log-prob 字段。

    不扫描源码文本——文档/注释需要说明该禁止项时不应误报。
    """
    # 1) dataclass 字段
    fields = [f.name for f in dataclasses.fields(Transition)]
    assert "old_raw_log_prob" in fields
    assert _exec_prob_like(fields) == []

    # 2) payload 键（顶层与每条 transition，含嵌套）
    buf = RolloutBuffer()
    buf.add(_transition())
    payload = buf.to_dict()
    assert _exec_prob_like(payload) == []
    for entry in payload["transitions"]:
        assert _exec_prob_like(entry) == []
        assert "old_raw_log_prob" in entry

    # 3) 运行时 API（模块、类、实例）
    for obj in (buffer_mod, Transition, RolloutBuffer, buf, buf.transitions[0]):
        assert _exec_prob_like(dir(obj)) == []

    # 4) 构造签名不得声明该字段（静态可查，不依赖运行期报错文本）
    assert _exec_prob_like(inspect.signature(Transition).parameters) == []


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

def test_contract_version_is_the_current_single_source():
    assert CONTRACT_VERSION_ID == "contract-v8"
    assert CONTRACT_VERSION_ID != "contract-v7"
    assert CONTRACT_VERSION == CONTRACT_VERSION_ID


def test_transition_dimensions_and_flags():
    t = _transition(terminated=True, truncated=True)
    assert t.observation.shape == (OBS_DIM,)
    assert t.next_observation.shape == (OBS_DIM,)
    assert t.raw_action.shape == (ACTION_DIM,)
    assert t.exec_action.shape == (ACTION_DIM,)
    assert t.terminated is True and t.truncated is True


# --- 6. terminated / truncated 严格布尔（禁止静默转换）---

@pytest.mark.parametrize(
    "bad",
    [
        "false",  # bool("false") is True —— 最危险的静默转换
        "true",
        "",
        1,
        0,
        2,
        1.0,
        0.0,
        None,
        np.int64(1),
        np.float32(0.0),
        [True],
    ],
)
def test_rejects_non_bool_terminated(bad):
    buf = RolloutBuffer()
    with pytest.raises((TypeError, ValueError), match="terminated"):
        buf.add(_transition(terminated=bad))


@pytest.mark.parametrize("bad", ["false", 1, 0, None, np.int64(0)])
def test_rejects_non_bool_truncated(bad):
    buf = RolloutBuffer()
    with pytest.raises((TypeError, ValueError), match="truncated"):
        buf.add(_transition(truncated=bad))


def test_accepts_numpy_bool_and_normalizes_to_python_bool():
    buf = RolloutBuffer()
    buf.add(_transition(terminated=np.bool_(True), truncated=np.bool_(False)))
    t = buf.transitions[0]
    assert t.terminated is True
    assert t.truncated is False
    assert isinstance(t.terminated, bool) and isinstance(t.truncated, bool)


def test_flags_survive_roundtrip_as_python_bool():
    buf = RolloutBuffer()
    buf.add(_transition(terminated=np.bool_(True), truncated=np.bool_(True)))
    restored = RolloutBuffer.from_dict(json.loads(json.dumps(buf.to_dict())))
    t = restored.transitions[0]
    assert t.terminated is True and t.truncated is True


# --- 7. 观测数组形状/维度严格 ---

def test_rejects_non_1d_observation():
    buf = RolloutBuffer()
    with pytest.raises(ValueError, match="observation"):
        buf.add(_transition(observation=np.zeros((2, OBS_DIM), dtype=np.float32)))


def test_rejects_non_1d_next_observation():
    buf = RolloutBuffer()
    with pytest.raises(ValueError, match="next_observation"):
        buf.add(_transition(next_observation=np.zeros((2, OBS_DIM), dtype=np.float32)))


def test_rejects_empty_observation():
    buf = RolloutBuffer()
    with pytest.raises(ValueError, match="observation"):
        buf.add(_transition(observation=np.zeros(0), next_observation=np.zeros(0)))


def test_rejects_non_finite_next_observation():
    buf = RolloutBuffer()
    nobs = _obs(2)
    nobs[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        buf.add(_transition(next_observation=nobs))


def test_rejects_inconsistent_observation_dim_within_buffer():
    """同一 buffer 内所有记录的观测维度必须一致。"""
    buf = RolloutBuffer()
    buf.add(_transition())
    with pytest.raises(ValueError, match="observation"):
        buf.add(
            _transition(
                observation=np.zeros(OBS_DIM + 1, dtype=np.float32),
                next_observation=np.zeros(OBS_DIM + 1, dtype=np.float32),
            )
        )


def test_observation_dim_guard_survives_roundtrip():
    """from_dict 重建后仍拒绝不同观测维度（状态跟着数据走，不是跟着实例走）。"""
    src = RolloutBuffer()
    src.add(_transition())
    restored = RolloutBuffer.from_dict(json.loads(json.dumps(src.to_dict())))
    with pytest.raises(ValueError, match="observation"):
        restored.add(
            _transition(
                observation=np.zeros(OBS_DIM + 1, dtype=np.float32),
                next_observation=np.zeros(OBS_DIM + 1, dtype=np.float32),
            )
        )


# --- 8. payload action_dim 严格为 21 ---

def test_payload_declares_action_dim_21():
    buf = RolloutBuffer()
    buf.add(_transition())
    assert buf.to_dict()["action_dim"] == ACTION_DIM == 21


def test_rejects_payload_with_wrong_action_dim():
    buf = RolloutBuffer()
    buf.add(_transition())
    payload = buf.to_dict()
    payload["action_dim"] = 23  # 旧 23 维动作不得被接受
    with pytest.raises(ValueError, match="action_dim"):
        RolloutBuffer.from_dict(payload)


def test_rejects_payload_missing_action_dim():
    buf = RolloutBuffer()
    buf.add(_transition())
    payload = buf.to_dict()
    del payload["action_dim"]
    with pytest.raises((ValueError, KeyError), match="action_dim"):
        RolloutBuffer.from_dict(payload)


def test_rejects_payload_swapping_action_dim_for_non_int():
    buf = RolloutBuffer()
    buf.add(_transition())
    payload = buf.to_dict()
    payload["action_dim"] = 21.0  # 非整数类型不得冒充通过
    with pytest.raises((TypeError, ValueError), match="action_dim"):
        RolloutBuffer.from_dict(payload)


# --- 9. 单位不混：业务违规量 / 碳排放量 / 电费 ---

def test_payload_declares_units_for_all_three_quantities():
    buf = RolloutBuffer()
    buf.add(_transition())
    units = buf.to_dict()["units"]
    assert units["business_cost"] == "violation_count"
    assert units["carbon_cost"] == "kgCO2e"
    assert units["electricity_cost_sgd"] == "SGD"
    # 三类量纲互不相同：违规计数 / 排放质量 / 货币
    assert len({units["business_cost"], units["carbon_cost"], units["electricity_cost_sgd"]}) == 3


def test_three_quantities_are_stored_separately():
    buf = RolloutBuffer()
    buf.add(_transition(business_cost=3.0, carbon_cost=12.5, electricity_cost_sgd=77.25))
    t = buf.transitions[0]
    assert t.business_cost == 3.0
    assert t.carbon_cost == 12.5
    assert t.electricity_cost_sgd == 77.25
    restored = RolloutBuffer.from_dict(json.loads(json.dumps(buf.to_dict())))
    r = restored.transitions[0]
    assert (r.business_cost, r.carbon_cost, r.electricity_cost_sgd) == (3.0, 12.5, 77.25)


def test_electricity_cost_must_be_its_own_finite_field():
    buf = RolloutBuffer()
    with pytest.raises(ValueError, match="electricity_cost_sgd"):
        buf.add(_transition(electricity_cost_sgd=float("inf")))


def test_rejects_payload_with_tampered_units():
    buf = RolloutBuffer()
    buf.add(_transition())
    payload = buf.to_dict()
    payload["units"]["carbon_cost"] = "SGD"  # 把排放量标成货币 = 混单位
    with pytest.raises(ValueError, match="units"):
        RolloutBuffer.from_dict(payload)


def test_rejects_payload_missing_units():
    buf = RolloutBuffer()
    buf.add(_transition())
    payload = buf.to_dict()
    del payload["units"]
    with pytest.raises((ValueError, KeyError), match="units"):
        RolloutBuffer.from_dict(payload)


# --- 10. 标量数值严格 ---

@pytest.mark.parametrize("bad", ["-1.25", True, None, [1.0], {"v": 1.0}])
def test_rejects_non_numeric_scalars(bad):
    buf = RolloutBuffer()
    with pytest.raises((TypeError, ValueError), match="old_raw_log_prob"):
        buf.add(_transition(old_raw_log_prob=bad))


# --- 11. v5 → v6 拒绝证据 ---

def test_rejects_literal_v5_payload():
    """真实 v5 落盘 payload 必须显式报错。

    v5 无 next_observation/terminated/truncated，顶层无 action_dim/units。
    """
    with pytest.raises(ValueError, match="contract_version"):
        RolloutBuffer.from_dict(json.loads(json.dumps(V5_PAYLOAD)))


def test_v5_payload_is_rejected_even_if_version_forwarded():
    """只把版本号改成**当前**版本也不得被接受：v5 缺字段必须仍然报错，不得填默认值。"""
    payload = json.loads(json.dumps(V5_PAYLOAD))
    payload["contract_version"] = CONTRACT_VERSION_ID
    for entry in payload["transitions"]:
        entry["contract_version"] = CONTRACT_VERSION_ID
    with pytest.raises((ValueError, KeyError)):
        RolloutBuffer.from_dict(payload)


def test_v5_log_prob_field_name_is_not_aliased():
    """旧字段名 raw_log_prob 不得作为 old_raw_log_prob 的别名被接受。"""
    buf = RolloutBuffer()
    buf.add(_transition())
    payload = buf.to_dict()
    entry = payload["transitions"][0]
    entry["raw_log_prob"] = entry.pop("old_raw_log_prob")
    with pytest.raises((ValueError, KeyError), match="old_raw_log_prob"):
        RolloutBuffer.from_dict(payload)


def test_unversioned_payload_rejected():
    payload = RolloutBuffer().to_dict()
    del payload["contract_version"]
    with pytest.raises((ValueError, KeyError)):
        RolloutBuffer.from_dict(payload)

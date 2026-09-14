"""M5.3e 测试：约束信号的非负物理域。

business = violation_task_steps（违规任务·步计数）、carbon = kgCO2e（排放质量），
两者物理上不可能为负；负值必须被**明确拒绝**（不得裁剪为 0）。

本卡**不提升** contract 版本：序列化 schema 不变，仍为 contract-v7。
"""

import copy
import json

import numpy as np
import pytest

from safe_rl_v2 import lagrangian as lag_mod
from safe_rl_v2.lagrangian import (
    UNIT_KG_CO2E,
    UNIT_VIOLATION_TASK_STEPS,
    ConstraintSpec,
    Lagrangian,
)

SPECS = (
    ConstraintSpec(
        name="business", budget=5.0, unit=UNIT_VIOLATION_TASK_STEPS,
        learning_rate=0.1, max_multiplier=100.0,
    ),
    ConstraintSpec(
        name="carbon", budget=3.0, unit=UNIT_KG_CO2E,
        learning_rate=0.2, max_multiplier=100.0,
    ),
)


def fresh() -> Lagrangian:
    return Lagrangian(SPECS)


def seed_state() -> Lagrangian:
    """已被更新过的实例（用于验证失败时不得回退/前进）。"""
    lag = fresh()
    lag.update({"business": [9.0], "carbon": [13.0]})
    return lag


# --- 0. 前置：版本与 schema 不变 -------------------------------------------

def test_contract_version_is_still_v7():
    from contracts import CONTRACT_VERSION_ID

    assert CONTRACT_VERSION_ID == "contract-v7"
    assert lag_mod.CONTRACT_VERSION == CONTRACT_VERSION_ID


def test_validate_helper_is_public_and_shared():
    """update() 与训练预检必须共用同一份规则（此处只固定它是公开纯函数）。"""
    assert hasattr(lag_mod, "validate_constraint_signals")
    import inspect

    params = list(inspect.signature(lag_mod.validate_constraint_signals).parameters)
    assert params[:2] == ["batch_signals", "expected"]


# --- 1/2. update() 必须逐 transition 拒绝负信号 ----------------------------

@pytest.mark.parametrize(
    "batch",
    [
        {"business": [-1.0], "carbon": [1.0]},
        {"business": [-1.0, 1.0], "carbon": [1.0, 1.0]},      # 均值恰为 0
        {"business": [0.0, -1.0e-12], "carbon": [1.0, 1.0]},  # 极小负值
        {"business": [1.0, 1.0], "carbon": [-2.0, 2.0]},      # 均值恰为 0
        {"business": [1.0, 1.0], "carbon": [1.0, -0.5]},
    ],
)
def test_update_rejects_negative_signals(batch):
    lag = fresh()
    with pytest.raises(ValueError, match="负|non-negative|>= 0"):
        lag.update(batch)


def test_mean_zero_sequence_is_rejected_elementwise():
    """[-1, 1] 的均值为 0：任何只查均值的实现都会漏掉它。"""
    lag = fresh()
    assert float(np.mean([-1.0, 1.0])) == 0.0
    with pytest.raises(ValueError):
        lag.update({"business": [-1.0, 1.0], "carbon": [1.0, 1.0]})


def test_error_names_the_constraint_index_and_value():
    lag = fresh()
    with pytest.raises(ValueError) as excinfo:
        lag.update({"business": [1.0, -3.5, 2.0], "carbon": [1.0, 1.0, 1.0]})
    message = str(excinfo.value)
    assert "business" in message
    assert "1" in message  # 下标
    assert "-3.5" in message  # 具体值


# --- 3. update() 失败必须原子 ----------------------------------------------

@pytest.mark.parametrize(
    "batch",
    [
        {"business": [-1.0], "carbon": [1.0]},
        {"business": [1.0], "carbon": [-1.0]},
        {"business": [float("nan")], "carbon": [1.0]},
        {"business": [float("inf")], "carbon": [1.0]},
        {"business": [], "carbon": [1.0]},
        {"business": [1.0]},                                   # 键集不符
        {"business": [1.0], "carbon": [1.0], "electricity": [1.0]},
        {"business": 1.0, "carbon": [1.0]},                    # 标量
        {"business": [[1.0]], "carbon": [1.0]},                # 非一维
    ],
)
def test_update_failure_leaves_every_field_untouched(batch):
    """失败时两个约束的 estimate/multiplier/updates/log 与全局 updates 都必须不变。"""
    lag = seed_state()
    before = copy.deepcopy(lag.state_dict())

    with pytest.raises((ValueError, TypeError)):
        lag.update(batch)

    assert lag.state_dict() == before
    assert lag.multipliers() == {
        name: before["constraints"][name]["multiplier"] for name in ("business", "carbon")
    }
    for name in ("business", "carbon"):
        assert lag.constraints[name].estimate == before["constraints"][name]["estimate"]
        assert lag.constraints[name].log == before["constraints"][name]["log"]
        assert lag.constraints[name].updates == before["constraints"][name]["updates"]


def test_shared_validator_is_pure():
    """共享校验函数本身不得改动传入的约束状态。"""
    lag = seed_state()
    before = copy.deepcopy(lag.state_dict())
    means = lag_mod.validate_constraint_signals(
        {"business": [2.0, 4.0], "carbon": [1.0]}, lag.constraints
    )
    assert means == {"business": 3.0, "carbon": 1.0}
    assert lag.state_dict() == before


# --- 4. load_state_dict() 必须拒绝负 estimate ------------------------------

def test_load_rejects_negative_estimate():
    state = seed_state().state_dict()
    state["constraints"]["business"]["estimate"] = -5.0
    with pytest.raises(ValueError, match="estimate"):
        fresh().load_state_dict(state)


def test_load_rejects_negative_estimate_on_any_constraint():
    for name in ("business", "carbon"):
        state = seed_state().state_dict()
        state["constraints"][name]["estimate"] = -1.0e-9
        with pytest.raises(ValueError, match="estimate"):
            fresh().load_state_dict(state)


def test_load_rejects_non_finite_estimate():
    for bad in (float("nan"), float("inf"), float("-inf")):
        state = seed_state().state_dict()
        state["constraints"]["carbon"]["estimate"] = bad
        with pytest.raises(ValueError):
            fresh().load_state_dict(state)


def test_load_rejection_is_atomic():
    target = seed_state()
    before = copy.deepcopy(target.state_dict())

    state = seed_state().state_dict()
    state["constraints"]["carbon"]["estimate"] = -2.0
    with pytest.raises(ValueError):
        target.load_state_dict(state)

    assert target.state_dict() == before


# --- 5. 合法路径不得回归 ----------------------------------------------------

@pytest.mark.parametrize("signals", [[0.0], [1.0], [0.0, 0.0], [1.0e-12], [1.0e6]])
def test_zero_and_positive_signals_are_accepted(signals):
    lag = fresh()
    lag.update({"business": list(signals), "carbon": list(signals)})
    assert lag.constraints["business"].estimate == pytest.approx(float(np.mean(signals)))


def test_zero_signals_are_not_confused_with_rejection():
    """0 是合法的（无违规、无排放），必须被接受而不是当作「缺失」。"""
    lag = fresh()
    lag.update({"business": [0.0, 0.0], "carbon": [0.0, 0.0]})
    assert lag.constraints["business"].estimate == 0.0
    assert lag.constraints["carbon"].estimate == 0.0
    assert lag.multipliers() == {"business": 0.0, "carbon": 0.0}
    assert lag.state_dict()["updates"] == 1


def test_capped_history_and_json_roundtrip_still_work():
    lag = fresh()
    for _ in range(200):
        lag.update({"business": [1.0e6], "carbon": [1.0e6]})
    assert lag.multipliers() == {"business": 100.0, "carbon": 100.0}

    on_disk = json.loads(json.dumps(lag.state_dict()))
    restored = fresh()
    restored.load_state_dict(on_disk)
    assert restored.state_dict() == lag.state_dict()


def test_existing_valid_v7_states_still_load():
    lag = fresh()
    lag.update({"business": [9.0], "carbon": [13.0]})
    lag.update({"business": [7.0, 8.0], "carbon": [2.0, 4.0]})
    state = lag.state_dict()
    restored = fresh()
    restored.load_state_dict(copy.deepcopy(state))
    assert restored.state_dict() == state


def test_update_state_load_continue_is_bitwise_identical():
    batches = [
        {"business": [9.0], "carbon": [13.0]},
        {"business": [7.0, 8.0], "carbon": [2.0, 4.0]},
        {"business": [5.0], "carbon": [3.0]},
    ]
    straight = fresh()
    for batch in batches:
        straight.update(batch)

    paused = fresh()
    paused.update(batches[0])
    resumed = fresh()
    resumed.load_state_dict(paused.state_dict())
    for batch in batches[1:]:
        resumed.update(batch)

    assert resumed.state_dict() == straight.state_dict()

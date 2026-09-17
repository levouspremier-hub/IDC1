"""M5.3e 测试：约束信号的非负物理域。

business = violation_task_steps（违规任务·步计数）、carbon = kgCO2e（排放质量），
两者物理上不可能为负；负值必须被**明确拒绝**（不得裁剪为 0）。

本卡**不提升** contract 版本：序列化 schema 不变。
**M1.3g-0 迁移**：全仓唯一契约版本升至 `contract-v9`，本文件只把版本断言随之迁移。
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

def test_contract_version_is_still_the_single_current_version():
    from contracts import CONTRACT_VERSION_ID

    assert CONTRACT_VERSION_ID == "contract-v9"
    assert CONTRACT_VERSION_ID != "contract-v7"
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


# --- 6. 回归：接受域恰等于物理合法域，且 update 与预检共用同一规则 ----------

def _valid(values) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    return bool(arr.ndim == 1 and arr.size > 0 and np.all(np.isfinite(arr)) and np.all(arr >= 0.0))


def _random_batches(rng, count: int) -> list[dict[str, list[float]]]:
    """构造混合批：含合法、含负、含非有限、含空。"""
    batches = []
    for _ in range(count):
        batch = {}
        for name in ("business", "carbon"):
            kind = rng.integers(0, 5)
            size = int(rng.integers(1, 5))
            if kind == 0:
                values = rng.uniform(0.0, 50.0, size=size)
            elif kind == 1:
                values = rng.uniform(-50.0, 0.0, size=size)          # 全负
            elif kind == 2:
                values = rng.uniform(-50.0, 50.0, size=size)          # 混合
            elif kind == 3:
                values = np.zeros(size)                               # 全零（合法）
            else:
                values = rng.uniform(0.0, 50.0, size=size)
                values[0] = float("nan")                              # 非有限
            batch[name] = values.tolist()
        batches.append(batch)
    return batches


def test_update_accepts_exactly_the_physically_valid_batches():
    """随机批：update() 成功 ⟺ 两个约束的序列都满足物理域（非负、有限、一维、非空）。"""
    rng = np.random.default_rng(20240914)
    accepted = rejected = 0
    for batch in _random_batches(rng, 120):
        should_accept = _valid(batch["business"]) and _valid(batch["carbon"])
        lag = fresh()
        before = copy.deepcopy(lag.state_dict())
        try:
            lag.update(batch)
        except (ValueError, TypeError):
            got_accept = False
        else:
            got_accept = True
        assert got_accept is should_accept, f"接受域与物理合法域不一致：{batch}"
        if got_accept:
            accepted += 1
            assert lag.constraints["business"].estimate >= 0.0
            assert lag.constraints["carbon"].estimate >= 0.0
        else:
            rejected += 1
            assert lag.state_dict() == before, "被拒绝的批次不得留下任何痕迹"
    assert accepted > 0 and rejected > 0, f"样本必须同时覆盖接受与拒绝：{accepted}/{rejected}"


def test_update_and_preflight_share_the_same_acceptance_rule():
    """update() 与 validate_constraint_signals 的接受/拒绝必须完全一致（同一份规则）。"""
    rng = np.random.default_rng(7)
    for batch in _random_batches(rng, 120):
        lag = fresh()
        try:
            means = lag_mod.validate_constraint_signals(batch, lag.constraints)
        except (ValueError, TypeError) as exc:
            preflight = ("reject", type(exc).__name__)
        else:
            preflight = ("accept", means)

        lag2 = fresh()
        try:
            updated = lag2.update(batch)
        except (ValueError, TypeError) as exc:
            via_update = ("reject", type(exc).__name__)
        else:
            via_update = ("accept", updated)

        assert preflight[0] == via_update[0], f"接受域不一致：{batch}"
        if preflight[0] == "reject":
            assert preflight[1] == via_update[1], f"拒绝类型不一致：{batch}"


def test_preflight_never_mutates_and_matches_the_estimates_update_would_use():
    rng = np.random.default_rng(11)
    lag = fresh()
    for batch in _random_batches(rng, 60):
        snapshot = copy.deepcopy(lag.state_dict())
        try:
            means = lag_mod.validate_constraint_signals(batch, lag.constraints)
        except (ValueError, TypeError):
            assert lag.state_dict() == snapshot
            continue
        assert lag.state_dict() == snapshot, "预检不得改动状态"
        lag.update(batch)
        for name in ("business", "carbon"):
            assert lag.constraints[name].estimate == pytest.approx(means[name])


def test_rejection_message_is_specific_for_every_failure_kind():
    """异常必须指出具体约束与失败原因（供训练预检报错使用）。"""
    cases = [
        ({"business": [-1.0], "carbon": [1.0]}, "business"),
        ({"business": [1.0], "carbon": [-1.0]}, "carbon"),
        ({"business": [], "carbon": [1.0]}, "business"),
        ({"business": [float("nan")], "carbon": [1.0]}, "business"),
        ({"business": [1.0], "carbon": [float("inf")]}, "carbon"),
        ({"business": [[1.0]], "carbon": [1.0]}, "business"),
        ({"business": 1.0, "carbon": [1.0]}, "business"),
    ]
    for batch, expected_name in cases:
        with pytest.raises((ValueError, TypeError)) as excinfo:
            fresh().update(batch)
        assert expected_name in str(excinfo.value), f"{batch} 的报错未指明 {expected_name}"

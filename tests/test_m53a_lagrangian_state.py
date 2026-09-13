"""M5.3a 测试：多约束拉格朗日状态、单位、聚合口径与版本化持久化。

本卡不测 actor objective：把 multiplier 接入损失属 M5.3b。
"""

import numpy as np
import pytest

from safe_rl_v2 import lagrangian as lag_mod
from safe_rl_v2.lagrangian import (
    AGGREGATION_PER_TRANSITION_MEAN,
    REQUIRED_CONSTRAINTS,
    UNIT_KG_CO2E,
    UNIT_VIOLATION_TASK_STEPS,
    ConstraintSpec,
    Lagrangian,
)

SPECS = (
    ConstraintSpec(
        name="business", budget=5.0, unit=UNIT_VIOLATION_TASK_STEPS,
        learning_rate=0.1, max_multiplier=10.0,
    ),
    ConstraintSpec(
        name="carbon", budget=3.0, unit=UNIT_KG_CO2E,
        learning_rate=0.2, max_multiplier=20.0,
    ),
)


def make_lagrangian(specs=SPECS) -> Lagrangian:
    return Lagrangian(specs)


# --- 1. 约束必须显式定义五要素 ---

def test_constraint_spec_carries_all_five_fields():
    spec = SPECS[0]
    assert (spec.name, spec.budget, spec.unit, spec.learning_rate, spec.max_multiplier) == (
        "business", 5.0, UNIT_VIOLATION_TASK_STEPS, 0.1, 10.0,
    )


def test_constraint_spec_requires_every_field():
    with pytest.raises(TypeError):
        ConstraintSpec(name="business", budget=5.0, unit=UNIT_VIOLATION_TASK_STEPS)


@pytest.mark.parametrize("name", ["business", "carbon"])
def test_units_are_fixed_per_constraint(name):
    units = {"business": UNIT_VIOLATION_TASK_STEPS, "carbon": UNIT_KG_CO2E}
    wrong = "SGD" if name == "business" else UNIT_VIOLATION_TASK_STEPS
    with pytest.raises(ValueError, match="unit"):
        ConstraintSpec(
            name=name, budget=1.0, unit=wrong, learning_rate=0.1, max_multiplier=1.0
        )
    # 正确单位可用
    ConstraintSpec(
        name=name, budget=1.0, unit=units[name], learning_rate=0.1, max_multiplier=1.0
    )


def test_electricity_cannot_become_a_constraint():
    """电费不是约束：既不能作为新约束加入，也不能顶替 business/carbon 的单位。"""
    assert "electricity" not in REQUIRED_CONSTRAINTS
    assert "electricity_cost_sgd" not in REQUIRED_CONSTRAINTS
    with pytest.raises(ValueError):
        Lagrangian((
            SPECS[0],
            SPECS[1],
            ConstraintSpec(
                name="electricity_cost", budget=1.0, unit="SGD",
                learning_rate=0.1, max_multiplier=1.0,
            ),
        ))
    # 用 SGD 顶替 business 单位同样被拒
    with pytest.raises(ValueError, match="unit"):
        Lagrangian((
            ConstraintSpec(
                name="business", budget=5.0, unit="SGD",
                learning_rate=0.1, max_multiplier=10.0,
            ),
            SPECS[1],
        ))


def test_constraint_set_must_be_exactly_business_and_carbon():
    with pytest.raises(ValueError):
        Lagrangian((SPECS[0],))  # 缺 carbon
    with pytest.raises(ValueError):
        Lagrangian((SPECS[1],))  # 缺 business
    with pytest.raises(ValueError):
        Lagrangian(())  # 空集合


def test_rejects_duplicate_constraint_names():
    with pytest.raises(ValueError):
        Lagrangian((SPECS[0], SPECS[0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget": -1.0},
        {"budget": float("nan")},
        {"learning_rate": 0.0},
        {"learning_rate": -0.1},
        {"learning_rate": float("inf")},
        {"max_multiplier": 0.0},
        {"max_multiplier": -2.0},
        {"max_multiplier": float("nan")},
    ],
)
def test_rejects_invalid_numeric_fields(kwargs):
    base = dict(
        name="business", budget=5.0, unit=UNIT_VIOLATION_TASK_STEPS,
        learning_rate=0.1, max_multiplier=10.0,
    )
    base.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        ConstraintSpec(**base)


# --- 2. 聚合口径固定为每 transition mean ---

def test_update_uses_per_transition_mean():
    lag = make_lagrangian()
    lag.update({"business": [1.0, 2.0, 3.0], "carbon": [0.0, 0.0, 0.0]})
    assert lag.constraints["business"].estimate == pytest.approx(2.0)
    assert lag.constraints["carbon"].estimate == pytest.approx(0.0)
    # multiplier = 0 + 0.1 * (2 - 5) < 0 -> 截断到 0
    assert lag.multipliers()["business"] == 0.0


def test_update_rejects_scalar_signals():
    """标量不得直接进 update：聚合口径不能被调用方绕过。"""
    lag = make_lagrangian()
    with pytest.raises((TypeError, ValueError)):
        lag.update({"business": 7.0, "carbon": 3.0})


@pytest.mark.parametrize(
    "batch",
    [
        {"business": [], "carbon": [1.0]},
        {"business": [1.0], "carbon": []},
        {"business": [float("nan")], "carbon": [1.0]},
        {"business": [float("inf")], "carbon": [1.0]},
        {"business": [1.0, float("-inf")], "carbon": [1.0]},
    ],
)
def test_update_rejects_empty_or_non_finite_batches(batch):
    lag = make_lagrangian()
    with pytest.raises(ValueError):
        lag.update(batch)


def test_update_rejects_wrong_signal_key_set():
    lag = make_lagrangian()
    with pytest.raises(ValueError):
        lag.update({"business": [1.0]})
    with pytest.raises(ValueError):
        lag.update({"business": [1.0], "carbon": [1.0], "electricity": [1.0]})


def test_aggregation_label_is_recorded():
    lag = make_lagrangian()
    assert lag.aggregation == AGGREGATION_PER_TRANSITION_MEAN
    assert lag.state_dict()["aggregation"] == AGGREGATION_PER_TRANSITION_MEAN
    with pytest.raises(ValueError):
        Lagrangian(SPECS, aggregation="sum")


# --- 3. 乘子独立、非负、有限、有上限 ---

def test_multipliers_are_independent():
    lag = make_lagrangian()
    lag.update({"business": [9.0], "carbon": [0.0]})
    assert lag.multipliers()["business"] == pytest.approx(0.4)  # 0.1*(9-5)
    assert lag.multipliers()["carbon"] == 0.0

    lag2 = make_lagrangian()
    lag2.update({"business": [0.0], "carbon": [13.0]})
    assert lag2.multipliers()["business"] == 0.0
    assert lag2.multipliers()["carbon"] == pytest.approx(2.0)  # 0.2*(13-3)


def test_multiplier_never_negative():
    lag = make_lagrangian()
    for _ in range(5):
        lag.update({"business": [0.0], "carbon": [0.0]})
    assert lag.multipliers()["business"] == 0.0
    assert lag.multipliers()["carbon"] == 0.0


def test_multiplier_is_capped_and_finite():
    lag = make_lagrangian()
    for _ in range(200):
        lag.update({"business": [1.0e6], "carbon": [1.0e6]})
    assert lag.multipliers()["business"] == pytest.approx(10.0)  # max_multiplier
    assert lag.multipliers()["carbon"] == pytest.approx(20.0)
    assert all(np.isfinite(v) for v in lag.multipliers().values())


def test_update_returns_the_new_multipliers():
    lag = make_lagrangian()
    returned = lag.update({"business": [9.0], "carbon": [13.0]})
    assert returned == lag.multipliers()


def test_updates_counter_and_log_per_constraint():
    lag = make_lagrangian()
    lag.update({"business": [9.0], "carbon": [0.0]})
    lag.update({"business": [9.0], "carbon": [0.0]})
    business = lag.constraints["business"]
    assert business.updates == 2
    assert len(business.log) == 2
    assert business.log[-1] == business.multiplier
    assert lag.constraints["carbon"].updates == 2  # 同步计数，但数值互不影响


# --- 4. 版本化持久化 ---

def test_contract_version_is_v7():
    from contracts import CONTRACT_VERSION_ID

    assert CONTRACT_VERSION_ID == "contract-v7"
    assert lag_mod.CONTRACT_VERSION == CONTRACT_VERSION_ID


def test_state_dict_records_version_units_aggregation_and_index():
    lag = make_lagrangian()
    lag.update({"business": [9.0], "carbon": [13.0]})
    state = lag.state_dict()

    assert state["contract_version"] == lag_mod.CONTRACT_VERSION
    assert state["aggregation"] == AGGREGATION_PER_TRANSITION_MEAN
    assert state["updates"] == 1
    assert set(state["constraints"]) == set(REQUIRED_CONSTRAINTS)
    assert state["constraints"]["business"]["unit"] == UNIT_VIOLATION_TASK_STEPS
    assert state["constraints"]["carbon"]["unit"] == UNIT_KG_CO2E
    for name in REQUIRED_CONSTRAINTS:
        entry = state["constraints"][name]
        assert set(entry) >= {
            "name", "budget", "unit", "learning_rate", "max_multiplier",
            "estimate", "multiplier", "updates", "log",
        }


def test_state_roundtrip_preserves_everything():
    lag = make_lagrangian()
    lag.update({"business": [9.0, 11.0], "carbon": [13.0, 17.0]})
    lag.update({"business": [4.0], "carbon": [1.0]})

    restored = make_lagrangian()
    restored.load_state_dict(lag.state_dict())

    assert restored.multipliers() == lag.multipliers()
    assert restored.aggregation == lag.aggregation
    for name in REQUIRED_CONSTRAINTS:
        assert restored.constraints[name].estimate == lag.constraints[name].estimate
        assert restored.constraints[name].updates == lag.constraints[name].updates
        assert restored.constraints[name].log == lag.constraints[name].log


def test_restored_state_produces_identical_further_updates():
    lag = make_lagrangian()
    lag.update({"business": [9.0], "carbon": [13.0]})

    restored = make_lagrangian()
    restored.load_state_dict(lag.state_dict())

    shared = {"business": [7.0, 8.0], "carbon": [2.0, 4.0]}
    assert lag.update(shared) == restored.update(shared)
    assert lag.state_dict() == restored.state_dict()


# --- 5. 旧版本 / 缺字段 / 单位不符 / 约束集合不符 必须显式拒绝 ---

def _valid_state() -> dict:
    lag = make_lagrangian()
    lag.update({"business": [9.0], "carbon": [13.0]})
    return lag.state_dict()


def test_load_rejects_missing_or_old_version():
    state = _valid_state()

    del state["contract_version"]
    with pytest.raises((ValueError, KeyError)):
        make_lagrangian().load_state_dict(state)

    state = _valid_state()
    state["contract_version"] = "contract-v6"
    with pytest.raises(ValueError, match="version"):
        make_lagrangian().load_state_dict(state)

    state = _valid_state()
    state["contract_version"] = "contract-v99"
    with pytest.raises(ValueError, match="version"):
        make_lagrangian().load_state_dict(state)


def test_load_rejects_legacy_v6_shaped_state():
    """M5.4 占位实现落盘的旧形态：{name: {budget, estimate, multiplier, log}}。"""
    legacy = {
        "business": {"budget": 5.0, "estimate": 9.0, "multiplier": 0.4, "log": [0.4]},
        "carbon": {"budget": 3.0, "estimate": 13.0, "multiplier": 2.0, "log": [2.0]},
    }
    with pytest.raises((ValueError, KeyError)):
        make_lagrangian().load_state_dict(legacy)
    with pytest.raises(TypeError):
        make_lagrangian().load_state_dict("not-a-dict")


def test_load_rejects_missing_fields():
    for missing in ("aggregation", "updates", "constraints"):
        state = _valid_state()
        del state[missing]
        with pytest.raises((ValueError, KeyError)):
            make_lagrangian().load_state_dict(state)

    for missing in ("unit", "learning_rate", "max_multiplier", "log", "updates"):
        state = _valid_state()
        del state["constraints"]["business"][missing]
        with pytest.raises((ValueError, KeyError)):
            make_lagrangian().load_state_dict(state)


def test_load_rejects_wrong_aggregation():
    state = _valid_state()
    state["aggregation"] = "sum"
    with pytest.raises(ValueError, match="aggregation"):
        make_lagrangian().load_state_dict(state)


def test_load_rejects_wrong_constraint_set():
    state = _valid_state()
    del state["constraints"]["carbon"]
    with pytest.raises(ValueError, match="constraint"):
        make_lagrangian().load_state_dict(state)

    state = _valid_state()
    state["constraints"]["electricity"] = dict(state["constraints"]["carbon"])
    with pytest.raises(ValueError, match="constraint"):
        make_lagrangian().load_state_dict(state)


def test_load_rejects_unit_mismatch():
    state = _valid_state()
    state["constraints"]["business"]["unit"] = "SGD"
    with pytest.raises(ValueError, match="unit"):
        make_lagrangian().load_state_dict(state)


def test_load_rejects_spec_mismatch():
    """budget / learning_rate / max_multiplier 与当前 spec 不符也必须拒绝。"""
    other = (
        ConstraintSpec(
            name="business", budget=99.0, unit=UNIT_VIOLATION_TASK_STEPS,
            learning_rate=0.1, max_multiplier=10.0,
        ),
        SPECS[1],
    )
    with pytest.raises(ValueError):
        Lagrangian(other).load_state_dict(_valid_state())

    other_lr = (
        ConstraintSpec(
            name="business", budget=5.0, unit=UNIT_VIOLATION_TASK_STEPS,
            learning_rate=0.5, max_multiplier=10.0,
        ),
        SPECS[1],
    )
    with pytest.raises(ValueError):
        Lagrangian(other_lr).load_state_dict(_valid_state())


@pytest.mark.parametrize("bad", [-0.1, 11.0, float("nan"), float("inf")])
def test_load_rejects_invalid_multiplier(bad):
    state = _valid_state()
    state["constraints"]["business"]["multiplier"] = bad
    with pytest.raises(ValueError):
        make_lagrangian().load_state_dict(state)


def test_load_rejects_bad_log():
    state = _valid_state()
    state["constraints"]["business"]["log"] = [0.4, 0.5]  # 与 updates 不符
    with pytest.raises(ValueError):
        make_lagrangian().load_state_dict(state)

    state = _valid_state()
    state["constraints"]["business"]["log"] = [float("nan")]
    with pytest.raises(ValueError):
        make_lagrangian().load_state_dict(state)

    state = _valid_state()
    state["constraints"]["business"]["log"] = "not-a-list"
    with pytest.raises((TypeError, ValueError)):
        make_lagrangian().load_state_dict(state)


def test_load_rejects_negative_or_non_finite_updates():
    for bad in (-1, 1.5, "1"):
        state = _valid_state()
        state["updates"] = bad
        with pytest.raises((TypeError, ValueError)):
            make_lagrangian().load_state_dict(state)


def test_failed_load_does_not_mutate_existing_state():
    """拒绝必须是原子的：失败后原状态不变。"""
    lag = make_lagrangian()
    lag.update({"business": [9.0], "carbon": [13.0]})
    before = lag.state_dict()

    bad = _valid_state()
    bad["constraints"]["business"]["unit"] = "SGD"
    with pytest.raises(ValueError):
        lag.load_state_dict(bad)

    assert lag.state_dict() == before

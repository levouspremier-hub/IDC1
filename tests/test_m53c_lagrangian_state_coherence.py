"""M5.3c 测试：拒绝「形态合法但 update() 不可能产生」的 contract-v7 乘子状态。

本卡**不提升** contract 版本：state_dict() 的 schema 完全不变，
只是补上状态内部自洽性的强制拒绝（此前会被静默接受）。
"""

import copy
from typing import Any

import pytest

from safe_rl_v2.lagrangian import (
    UNIT_KG_CO2E,
    UNIT_VIOLATION_TASK_STEPS,
    ConstraintSpec,
    Lagrangian,
)

BUSINESS_MAX = 10.0
CARBON_MAX = 20.0

SPECS = (
    ConstraintSpec(
        name="business", budget=5.0, unit=UNIT_VIOLATION_TASK_STEPS,
        learning_rate=0.1, max_multiplier=BUSINESS_MAX,
    ),
    ConstraintSpec(
        name="carbon", budget=3.0, unit=UNIT_KG_CO2E,
        learning_rate=0.2, max_multiplier=CARBON_MAX,
    ),
)


def fresh() -> Lagrangian:
    return Lagrangian(SPECS)


def valid_state() -> dict:
    """由真实 update() 产生的合法状态：business λ=0.4、carbon λ=2.0，均 updates=1。"""
    lag = fresh()
    lag.update({"business": [9.0], "carbon": [13.0]})
    return lag.state_dict()


def multi_update_state(rounds: int = 3) -> dict:
    lag = fresh()
    for _ in range(rounds):
        lag.update({"business": [9.0], "carbon": [13.0]})
    return lag.state_dict()


def load(state: dict) -> Lagrangian:
    lag = fresh()
    lag.load_state_dict(state)
    return lag


def assert_rejected(state: dict, match: str | None = None) -> None:
    with pytest.raises(ValueError, match=match):
        load(state)


# --- 0. 前置：schema 未变，合法状态仍被接受 -------------------------------

def test_contract_version_is_still_v7():
    from contracts import CONTRACT_VERSION_ID
    from safe_rl_v2 import lagrangian as lag_mod

    assert CONTRACT_VERSION_ID == "contract-v7"
    assert lag_mod.CONTRACT_VERSION == CONTRACT_VERSION_ID
    assert valid_state()["contract_version"] == "contract-v7"


def test_state_dict_shape_is_unchanged():
    """本卡只加校验，序列化 schema 的字段集不得变化。"""
    state = valid_state()
    assert set(state) == {"contract_version", "aggregation", "updates", "constraints"}
    for entry in state["constraints"].values():
        assert set(entry) == {
            "name", "budget", "unit", "learning_rate", "max_multiplier",
            "estimate", "multiplier", "updates", "log",
        }


def test_legal_states_still_load():
    restored = load(valid_state())
    assert restored.multipliers() == {"business": 0.4, "carbon": 2.0}
    # 每轮同类批次下乘子线性累加：5 轮 -> business 5×0.4 = 2.0、carbon 5×2.0 = 10.0
    assert load(multi_update_state(5)).multipliers() == pytest.approx(
        {"business": 2.0, "carbon": 10.0}
    )


# --- 1. log 值必须有限且落在 [0, max_multiplier] --------------------------

@pytest.mark.parametrize("bad_log", [-0.5, -1.0e-9, BUSINESS_MAX + 1.0, 999.0])
def test_rejects_log_values_outside_range(bad_log):
    state = valid_state()
    state["constraints"]["business"]["log"] = [bad_log]
    assert_rejected(state, match="log")


@pytest.mark.parametrize("bad_log", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_log_values(bad_log):
    state = valid_state()
    state["constraints"]["business"]["log"] = [bad_log]
    assert_rejected(state)


def test_rejects_out_of_range_value_in_the_middle_of_a_valid_history():
    """历史中间的越界值同样必须被拒（不能只看最后一项）。"""
    state = multi_update_state(3)
    assert len(state["constraints"]["business"]["log"]) == 3
    state["constraints"]["business"]["log"][1] = BUSINESS_MAX + 0.5
    assert_rejected(state, match="log")


def test_accepts_log_values_exactly_at_bounds():
    """边界值 0 与 max_multiplier 是合法历史值。"""
    state = valid_state()
    state["constraints"]["business"]["log"] = [0.0, BUSINESS_MAX]
    state["constraints"]["business"]["updates"] = 2
    state["constraints"]["business"]["multiplier"] = BUSINESS_MAX  # log[-1] 必须等于它
    state["updates"] = 2
    state["constraints"]["carbon"]["updates"] = 2
    state["constraints"]["carbon"]["log"] = [0.0, 0.0]
    state["constraints"]["carbon"]["multiplier"] = 0.0
    restored = load(state)
    assert restored.multipliers() == {"business": BUSINESS_MAX, "carbon": 0.0}


# --- 2. updates > 0 时 log[-1] 必须严格等于 multiplier ---------------------

def test_rejects_last_log_entry_not_equal_to_multiplier():
    state = valid_state()
    state["constraints"]["business"]["log"] = [0.5]  # multiplier 仍为 0.4
    assert_rejected(state, match="log")


def test_rejects_multiplier_not_matching_last_log_entry():
    """反向：改 multiplier 而不改 log[-1]，同样必须拒绝。"""
    state = valid_state()
    state["constraints"]["business"]["multiplier"] = 1.5
    state["constraints"]["business"]["log"] = [0.4]
    assert_rejected(state, match="log")


def test_rejects_mismatch_on_any_constraint():
    state = valid_state()
    state["constraints"]["carbon"]["log"] = [0.0]  # carbon multiplier 为 2.0
    assert_rejected(state, match="log")


# --- 3. 逐约束 updates 必须等于顶层 updates --------------------------------

def test_rejects_constraint_updates_differing_from_outer():
    state = valid_state()
    state["updates"] = 5
    state["constraints"]["carbon"]["updates"] = 1
    state["constraints"]["carbon"]["log"] = [2.0]  # log 长度对得上，仍必须拒绝
    assert_rejected(state, match="updates")


def test_rejects_outer_updates_differing_from_constraint():
    state = multi_update_state(3)
    state["updates"] = 2  # 各约束仍是 3
    assert_rejected(state, match="updates")


def test_rejects_any_single_constraint_out_of_step():
    for name in ("business", "carbon"):
        state = multi_update_state(2)
        state["constraints"][name]["updates"] = 1
        state["constraints"][name]["log"] = [state["constraints"][name]["log"][-1]]
        assert_rejected(state, match="updates")


# --- 4. updates == 0 必须与构造后的零状态一致 ------------------------------

def test_rejects_nonzero_multiplier_at_zero_updates():
    state = valid_state()
    state["updates"] = 0
    for name in ("business", "carbon"):
        state["constraints"][name]["updates"] = 0
        state["constraints"][name]["log"] = []
        state["constraints"][name]["estimate"] = 0.0
    # multiplier 仍为 0.4 / 2.0
    assert_rejected(state, match="multiplier")


def test_rejects_nonzero_estimate_at_zero_updates():
    state = valid_state()
    state["updates"] = 0
    for name in ("business", "carbon"):
        state["constraints"][name]["updates"] = 0
        state["constraints"][name]["log"] = []
        state["constraints"][name]["multiplier"] = 0.0
    # estimate 仍为 9.0 / 13.0
    assert_rejected(state, match="estimate")


def test_rejects_nonempty_log_at_zero_updates():
    """零更新却带历史（长度校验之外的自洽性）。"""
    state = valid_state()
    state["updates"] = 0
    for name in ("business", "carbon"):
        state["constraints"][name]["updates"] = 0
        state["constraints"][name]["multiplier"] = 0.0
        state["constraints"][name]["estimate"] = 0.0
    # log 保留 [0.4] / [2.0] -> 长度与 updates=0 不符，必须拒绝
    assert_rejected(state)


def test_accepts_a_freshly_constructed_zero_state():
    """真正的零状态必须继续被接受。"""
    zero = fresh().state_dict()
    restored = load(zero)
    assert restored.multipliers() == {"business": 0.0, "carbon": 0.0}
    assert restored.state_dict() == zero


# --- 5. 拒绝必须原子 --------------------------------------------------------

INCOHERENT_STATES = {
    "log_out_of_range": lambda s: s["constraints"]["business"].__setitem__("log", [-0.5]),
    "log_last_mismatch": lambda s: s["constraints"]["business"].__setitem__("log", [0.5]),
    "updates_out_of_step": lambda s: (
        s.__setitem__("updates", 5),
        s["constraints"]["carbon"].__setitem__("updates", 1),
        s["constraints"]["carbon"].__setitem__("log", [2.0]),
    ),
    "zero_updates_nonzero_multiplier": lambda s: (
        s.__setitem__("updates", 0),
        [s["constraints"][n].update({"updates": 0, "log": [], "estimate": 0.0})
         for n in ("business", "carbon")],
    ),
}


@pytest.mark.parametrize("name", sorted(INCOHERENT_STATES))
def test_rejection_is_atomic(name):
    """失败前后 target.state_dict() 必须逐位一致。"""
    target = fresh()  # 实例（非 state dict）
    for _ in range(2):
        target.update({"business": [9.0], "carbon": [13.0]})
    before = copy.deepcopy(target.state_dict())

    state = valid_state()
    INCOHERENT_STATES[name](state)
    with pytest.raises(ValueError):
        target.load_state_dict(state)

    assert target.state_dict() == before


# --- 6. 合法路径不得回归 ----------------------------------------------------

def test_update_state_load_continue_roundtrip_is_bitwise_identical():
    """真 update → state_dict → load → 继续 update，必须与不中断路径逐位一致。"""
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
    resumed = load(paused.state_dict())
    for batch in batches[1:]:
        resumed.update(batch)

    assert resumed.state_dict() == straight.state_dict()


def test_capped_history_is_accepted():
    """乘子被 max_multiplier 截断后的历史（含重复的上限值）必须合法。"""
    lag = fresh()
    for _ in range(200):
        lag.update({"business": [1.0e6], "carbon": [1.0e6]})
    state = lag.state_dict()
    assert set(state["constraints"]["business"]["log"]) == {BUSINESS_MAX}
    restored = load(copy.deepcopy(state))
    assert restored.state_dict() == state


# --- 7. 回归：任何被接受的状态都必须是不动点 --------------------------------

MUTANTS: list[object] = [
    0.0, -0.0, 1.0, -1.0, 1.0e-9, -1.0e-9, 0.4, 2.0, 9.0, 13.0,
    BUSINESS_MAX, BUSINESS_MAX + 1.0, CARBON_MAX, CARBON_MAX + 1.0,
    float("nan"), float("inf"), float("-inf"), None, "x", [], {}, True, 3,
]


def _leaf_paths(state: dict) -> list[tuple[str, ...]]:
    paths: list[tuple[str, ...]] = [("updates",)]
    for name, entry in state["constraints"].items():
        for field in entry:
            paths.append(("constraints", name, field))
    return paths


def _get(state: dict, path: tuple[str, ...]) -> object:
    node = state
    for key in path:
        node = node[key]
    return node


def _set(state: dict, path: tuple[str, ...], value: object) -> None:
    node: Any = state
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def test_any_accepted_state_is_a_fixed_point_under_single_field_mutation():
    """对每个字段做穷举单点变异：要么被明确拒绝，要么必须是 state_dict() 的不动点。

    这条不变量同时覆盖了 M5.3a 的静态规则与本卡新增的自洽性规则：
    任何「被接受但 state_dict() 与其不等」的状态都意味着存在未被拒绝的
    不可能状态。
    """
    base = multi_update_state(2)
    accepted = 0
    rejected = 0

    for path in _leaf_paths(base):
        for mutant in MUTANTS:
            if _get(base, path) == mutant:
                continue  # 与合法值相同，视为未变异
            state = copy.deepcopy(base)
            _set(state, path, mutant)
            try:
                restored = load(copy.deepcopy(state))
            except (ValueError, TypeError):
                rejected += 1
                continue
            accepted += 1
            assert restored.state_dict() == state, (
                f"{path} = {mutant!r} 被接受但不是不动点："
                f"state_dict() 与之不相等"
            )

    assert rejected > 0, "变异集必须至少触发一些拒绝"
    assert accepted > 0, "变异集必须至少包含一些合法变异"
    assert rejected + accepted == sum(
        1 for path in _leaf_paths(base) for mutant in MUTANTS if _get(base, path) != mutant
    )


def test_valid_state_loads_then_reloads_identically():
    """load → state_dict → load 必须幂等。"""
    first = load(copy.deepcopy(valid_state()))
    snapshot = first.state_dict()
    second = load(copy.deepcopy(snapshot))
    assert second.state_dict() == snapshot


def test_multi_update_state_round_trips_exactly():
    for rounds in (1, 2, 3, 7):
        state = multi_update_state(rounds)
        restored = load(copy.deepcopy(state))
        assert restored.state_dict() == state
        assert restored.state_dict()["updates"] == rounds


def test_coherence_rules_do_not_reject_real_update_histories():
    """真实 update() 产生的任意长度历史都必须被接受（不得误杀）。"""
    lag = fresh()
    for round_index in range(1, 8):
        lag.update({
            "business": [0.0, 9.0, 1.0e6][: (round_index % 3) + 1],
            "carbon": [0.0, 13.0, 1.0e6][: (round_index % 3) + 1],
        })
        state = lag.state_dict()
        assert load(copy.deepcopy(state)).state_dict() == state

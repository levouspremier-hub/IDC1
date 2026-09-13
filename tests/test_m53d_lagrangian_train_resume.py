"""M5.3d 测试：恢复出的乘子状态在下一轮 actor objective 中与连续路径逐位等价。

用确定性假 collector + **同权重**全新 policy 隔离差异源：
两条路径唯一可能的分叉只能来自乘子状态的传递方式。

**边界声明**：本卡**不**覆盖 policy/optimizer 状态，也不构成完整训练断点恢复；
完整训练恢复（训练 CLI、checkpoint 总入口、policy/optimizer/RNG 状态）属 M5.4/M5.5。
"""

import copy
import json
from typing import Any

import numpy as np
import pytest
import torch

from safe_rl_v2 import models as models_mod
from safe_rl_v2 import train as train_mod
from safe_rl_v2.buffer import ACTION_DIM, CONTRACT_VERSION, Transition
from safe_rl_v2.lagrangian import (
    UNIT_KG_CO2E,
    UNIT_VIOLATION_TASK_STEPS,
    ConstraintSpec,
    Lagrangian,
)
from safe_rl_v2.policy import SafePPOPolicy
from safe_rl_v2.train import dry_run_update

OBS_DIM = 8
HEADS = ("reward", "business", "carbon")

BUSINESS_MAX = 100.0
CARBON_MAX = 100.0

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

# 每批的 (reward, business, carbon) 逐 transition 信号
BATCHES: list[tuple[list[float], list[float], list[float]]] = [
    ([1.0, -0.5], [6.0, 8.0], [4.0, 6.0]),    # BATCH0: mean b=7.0, c=5.0
    ([2.0, 0.5], [9.0, 11.0], [1.0, 3.0]),    # BATCH1: mean b=10.0, c=2.0
    ([1.5], [5.0], [9.0]),                    # BATCH2: mean b=5.0, c=9.0
]

# 跑完 BATCH0 后的期望乘子（非零，故约束项在 actor objective 中非平凡）
LAMBDA_AFTER_BATCH0 = {"business": 0.2, "carbon": 0.4}
# 跑完 BATCH1 后的期望乘子
LAMBDA_AFTER_BATCH1 = {"business": 0.7, "carbon": 0.2}


def fresh() -> Lagrangian:
    return Lagrangian(SPECS)


def make_policy(seed: int = 0) -> SafePPOPolicy:
    torch.manual_seed(seed)
    return SafePPOPolicy(obs_dim=OBS_DIM)


def batch_transitions(batch_index: int) -> list[Transition]:
    rewards, business, carbon = BATCHES[batch_index]
    transitions = []
    for index, (reward, biz, car) in enumerate(
        zip(rewards, business, carbon, strict=True)
    ):
        transitions.append(
            Transition(
                observation=np.full(OBS_DIM, 0.1 * (index + 1) + 0.05 * batch_index,
                                    dtype=np.float32),
                next_observation=np.full(OBS_DIM, 0.2 * (index + 1) + 0.05 * batch_index,
                                         dtype=np.float32),
                raw_action=np.full(ACTION_DIM, 0.5, dtype=np.float32),
                old_raw_log_prob=-1.0,
                exec_action=np.full(ACTION_DIM, 0.5, dtype=np.float32),
                reward=reward,
                business_cost=biz,
                carbon_cost=car,
                electricity_cost_sgd=7.0,
                terminated=False,
                truncated=False,
                correction_info={},
            )
        )
    return transitions


def install_fake_collector(monkeypatch, transitions: list[Transition]) -> None:
    def fake(env, policy, buffer, *, steps, seed=0, corrector_on=False,
             corrector_time_limit_s=None, generator=None):
        for t in transitions:
            buffer.add(t)
        return {
            "corrector_on": corrector_on,
            "corrector_time_limit_s": corrector_time_limit_s,
            "steps_requested": steps,
            "transitions": len(transitions),
            "raw_exec_difference_count": 0,
            "terminated_count": 0,
            "truncated_count": 0,
            "contract_version": CONTRACT_VERSION,
            "action_dim": ACTION_DIM,
            "env_seed": seed,
            "policy_rng_source": (
                "explicit_generator" if generator is not None else "global_torch_rng"
            ),
        }

    monkeypatch.setattr(train_mod, "collect_rollout", fake)


def run_round(
    monkeypatch, lagrangian: Lagrangian, batch_index: int, *, policy_seed: int = 0
) -> dict:
    """跑一批：假 collector + **同种子**全新 policy（权重逐位相同）。"""
    transitions = batch_transitions(batch_index)
    install_fake_collector(monkeypatch, transitions)
    policy = make_policy(policy_seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    return dry_run_update(
        None, policy, lagrangian, optimizer,
        steps=len(transitions), seed=0, corrector_on=False, generator=None,
    )


def independent_effective_advantage(
    policy: SafePPOPolicy, batch_index: int, multipliers: dict[str, float]
) -> np.ndarray:
    """用给定 λ 独立重算 A_eff = A_r − λ_b·A_b − λ_c·A_c。"""
    transitions = batch_transitions(batch_index)
    obs = torch.as_tensor(
        np.stack([t.observation for t in transitions]), dtype=torch.float32
    )
    next_obs = torch.as_tensor(
        np.stack([t.next_observation for t in transitions]), dtype=torch.float32
    )
    with torch.no_grad():
        _, current_v = policy.forward(obs)
        _, next_v = policy.forward(next_obs)
    targets = models_mod.compute_three_value_targets(
        np.array([t.reward for t in transitions]),
        np.array([t.business_cost for t in transitions]),
        np.array([t.carbon_cost for t in transitions]),
        {h: current_v[:, i].numpy() for i, h in enumerate(HEADS)},
        {h: next_v[:, i].numpy() for i, h in enumerate(HEADS)},
        terminated=np.zeros(len(transitions), dtype=bool),
        truncated=np.zeros(len(transitions), dtype=bool),
        gamma=train_mod.GAMMA,
        lam=train_mod.LAM,
    )
    return (
        targets["reward"][0]
        - multipliers["business"] * targets["business"][0]
        - multipliers["carbon"] * targets["carbon"][0]
    )


COMPARED_KEYS = (
    "multipliers_pre_update",
    "multipliers_post_update",
    "effective_advantage",
    "effective_advantage_terms",
    "actor_loss",
    "critic_loss",
    "constraint_means",
)


def assert_rounds_equivalent(left: dict, right: dict) -> None:
    for key in COMPARED_KEYS:
        assert left[key] == right[key], f"{key} 在两条路径上不一致"


# --- 1. 恢复路径与连续路径在下一轮逐项等价 --------------------------------

def test_resumed_multiplier_state_is_equivalent_in_the_next_actor_round(monkeypatch):
    # 连续路径：BATCH0 -> 存检查点 -> BATCH1
    lag_cont = fresh()
    run_round(monkeypatch, lag_cont, 0)
    checkpoint = copy.deepcopy(lag_cont.state_dict())
    assert lag_cont.multipliers() == pytest.approx(LAMBDA_AFTER_BATCH0)

    cont = run_round(monkeypatch, lag_cont, 1)

    # 恢复路径：新实例 load 检查点 -> BATCH1
    lag_resumed = fresh()
    lag_resumed.load_state_dict(checkpoint)
    assert lag_resumed.state_dict() == checkpoint
    resumed = run_round(monkeypatch, lag_resumed, 1)

    assert_rounds_equivalent(cont, resumed)
    assert lag_resumed.state_dict() == lag_cont.state_dict()
    assert lag_cont.multipliers() == pytest.approx(LAMBDA_AFTER_BATCH1)


def test_resume_equivalence_holds_for_the_whole_downstream_history(monkeypatch):
    """检查点之后连跑两轮，两条路径的最终状态仍须逐位一致。"""
    lag_cont = fresh()
    run_round(monkeypatch, lag_cont, 0)
    run_round(monkeypatch, lag_cont, 1)
    checkpoint = copy.deepcopy(lag_cont.state_dict())

    cont = [run_round(monkeypatch, lag_cont, 2)]

    lag_resumed = fresh()
    lag_resumed.load_state_dict(checkpoint)
    resumed = [run_round(monkeypatch, lag_resumed, 2)]

    assert_rounds_equivalent(cont[0], resumed[0])
    assert lag_resumed.state_dict() == lag_cont.state_dict()


def test_resume_is_equivalent_at_every_round_boundary(monkeypatch):
    """在每一个轮次边界落盘并恢复，都必须与连续路径一致。"""
    batches = [0, 1, 2]
    continuous: list[dict] = []
    states: list[dict] = []

    lag_cont = fresh()
    for batch in batches:
        continuous.append(run_round(monkeypatch, lag_cont, batch))
        states.append(copy.deepcopy(lag_cont.state_dict()))

    for boundary in range(len(batches) - 1):
        lag_resumed = fresh()
        lag_resumed.load_state_dict(copy.deepcopy(states[boundary]))
        for offset, batch in enumerate(batches[boundary + 1 :], start=1):
            resumed = run_round(monkeypatch, lag_resumed, batch)
            assert_rounds_equivalent(continuous[boundary + offset], resumed)
        assert lag_resumed.state_dict() == lag_cont.state_dict()


# --- 2. actor 用的是恢复出的 pre-update λ ---------------------------------

def test_actor_uses_restored_pre_update_multipliers(monkeypatch):
    lag = fresh()
    run_round(monkeypatch, lag, 0)
    checkpoint = copy.deepcopy(lag.state_dict())

    lag_resumed = fresh()
    lag_resumed.load_state_dict(checkpoint)
    result = run_round(monkeypatch, lag_resumed, 1)

    pre = result["multipliers_pre_update"]
    post = result["multipliers_post_update"]
    assert pre == pytest.approx(LAMBDA_AFTER_BATCH0)
    assert post == pytest.approx(LAMBDA_AFTER_BATCH1)
    assert pre != pytest.approx(post), "本测试要求 pre/post 可区分"

    # 用 pre 的 λ 独立重算，必须与上报值吻合
    policy = make_policy(seed=0)
    expected_pre = independent_effective_advantage(policy, 1, pre)
    assert result["effective_advantage"]["mean"] == pytest.approx(float(expected_pre.mean()))

    # 用 post 的 λ 重算必须**不**吻合（证明用的不是本轮更新后的 λ）
    wrong_post = independent_effective_advantage(policy, 1, post)
    assert result["effective_advantage"]["mean"] != pytest.approx(float(wrong_post.mean()))

    # 三项分解也必须由 pre 的 λ 决定
    assert result["effective_advantage_terms"]["business_mean"] == pytest.approx(
        float((-pre["business"] * models_mod_advantage(policy, 1, "business")).mean())
    )


def models_mod_advantage(policy: SafePPOPolicy, batch_index: int, head: str) -> np.ndarray:
    """独立取某一头的优势（供上面的分解断言使用）。"""
    transitions = batch_transitions(batch_index)
    obs = torch.as_tensor(
        np.stack([t.observation for t in transitions]), dtype=torch.float32
    )
    next_obs = torch.as_tensor(
        np.stack([t.next_observation for t in transitions]), dtype=torch.float32
    )
    with torch.no_grad():
        _, current_v = policy.forward(obs)
        _, next_v = policy.forward(next_obs)
    targets = models_mod.compute_three_value_targets(
        np.array([t.reward for t in transitions]),
        np.array([t.business_cost for t in transitions]),
        np.array([t.carbon_cost for t in transitions]),
        {h: current_v[:, i].numpy() for i, h in enumerate(HEADS)},
        {h: next_v[:, i].numpy() for i, h in enumerate(HEADS)},
        terminated=np.zeros(len(transitions), dtype=bool),
        truncated=np.zeros(len(transitions), dtype=bool),
        gamma=train_mod.GAMMA,
        lam=train_mod.LAM,
    )
    return targets[head][0]


# --- 3. 比较必须有齿 --------------------------------------------------------

def test_comparison_is_sensitive_to_a_different_restored_state(monkeypatch):
    """恢复成另一份不同状态必须产生不同结果——证明等价性不是恒真。"""
    lag_cont = fresh()
    run_round(monkeypatch, lag_cont, 0)
    checkpoint = copy.deepcopy(lag_cont.state_dict())
    cont = run_round(monkeypatch, lag_cont, 1)

    other = fresh()
    run_round(monkeypatch, other, 0)
    run_round(monkeypatch, other, 0)  # 多跑一轮 -> 乘子更大
    lag_other = fresh()
    lag_other.load_state_dict(other.state_dict())
    other_result = run_round(monkeypatch, lag_other, 1)

    assert other_result["multipliers_pre_update"] != pytest.approx(
        checkpoint["constraints"]["business"]["multiplier"]
    ) or other_result["multipliers_pre_update"] != pytest.approx(
        cont["multipliers_pre_update"]
    )
    assert other_result["effective_advantage"] != cont["effective_advantage"]


def test_mutated_checkpoint_changes_the_next_round(monkeypatch):
    """把检查点里的 λ 改掉，下一轮结果必须随之改变。"""
    lag = fresh()
    run_round(monkeypatch, lag, 0)
    baseline_state = copy.deepcopy(lag.state_dict())
    baseline = run_round(monkeypatch, lag, 1)

    mutated_state = copy.deepcopy(baseline_state)
    for name in ("business", "carbon"):
        entry = mutated_state["constraints"][name]
        entry["multiplier"] = entry["multiplier"] + 0.5
        entry["log"][-1] = entry["multiplier"]  # 保持自洽（M5.3c 规则）
    lag_mutated = fresh()
    lag_mutated.load_state_dict(mutated_state)
    mutated = run_round(monkeypatch, lag_mutated, 1)

    assert mutated["effective_advantage"] != baseline["effective_advantage"]
    assert mutated["actor_loss"] != pytest.approx(baseline["actor_loss"])


# --- 4. 边界声明：本卡不覆盖 policy/optimizer ------------------------------

def test_lagrangian_state_does_not_carry_policy_or_optimizer_state(monkeypatch):
    """同一份乘子状态 + 不同 policy 权重必须给出不同 actor_loss。

    这正面说明 Lagrangian state **不携带** policy/optimizer 状态，
    因此本卡**不构成**完整训练断点恢复。
    """
    lag = fresh()
    run_round(monkeypatch, lag, 0)
    checkpoint = copy.deepcopy(lag.state_dict())

    results = []
    for seed in (0, 1):
        lag_r = fresh()
        lag_r.load_state_dict(copy.deepcopy(checkpoint))
        results.append(run_round(monkeypatch, lag_r, 1, policy_seed=seed))

    assert results[0]["actor_loss"] != pytest.approx(results[1]["actor_loss"])


def test_round_metrics_expose_no_policy_or_optimizer_state(monkeypatch):
    """返回指标不得声称携带 policy/optimizer/RNG 状态。"""
    lag = fresh()
    result = run_round(monkeypatch, lag, 0)

    forbidden = ("policy_state", "optimizer_state", "rng_state", "checkpoint")
    offenders = [key for key in result if any(token in key.lower() for token in forbidden)]
    assert offenders == []
    assert result["claims"] == {
        "trained": False,
        "performance_evaluated": False,
        "convergence_claimed": False,
    }
    # 诊断结果可 JSON 序列化（buffer/stats/数组型 target 除外）
    scalar = {
        key: value
        for key, value in result.items()
        if key not in {"buffer", "stats", "critic_targets"}
    }
    assert isinstance(json.dumps(scalar), str)


def test_resume_path_is_deterministic_across_repeats(monkeypatch):
    """同一检查点重复恢复，结果必须逐位一致（无隐藏状态）。"""
    lag = fresh()
    run_round(monkeypatch, lag, 0)
    checkpoint = copy.deepcopy(lag.state_dict())

    outputs: list[tuple[dict[str, Any], dict]] = []
    for _ in range(2):
        lag_r = fresh()
        lag_r.load_state_dict(copy.deepcopy(checkpoint))
        outputs.append((run_round(monkeypatch, lag_r, 1), lag_r.state_dict()))

    assert_rounds_equivalent(outputs[0][0], outputs[1][0])
    assert outputs[0][1] == outputs[1][1]


# --- 5. 回归：真实持久化路径（磁盘往返）下的恢复等价性 ----------------------

def test_resume_through_json_roundtrip_is_equivalent(monkeypatch):
    """检查点经 JSON 序列化往返（真实落盘形态）后，恢复仍须与连续路径一致。

    内存里直接传 dict 与「写盘再读回」在数值类型上可能不同
    （int/float 保真、键序、NaN 等），真实 checkpoint 走的是后者。
    """
    lag_cont = fresh()
    run_round(monkeypatch, lag_cont, 0)
    on_disk = json.loads(json.dumps(lag_cont.state_dict()))
    cont = run_round(monkeypatch, lag_cont, 1)

    lag_resumed = fresh()
    lag_resumed.load_state_dict(on_disk)
    resumed = run_round(monkeypatch, lag_resumed, 1)

    assert_rounds_equivalent(cont, resumed)
    assert lag_resumed.state_dict() == lag_cont.state_dict()
    # 往返后的 state 必须与连续侧逐位一致（含 updates 的整数类型）
    assert isinstance(on_disk["updates"], int)
    assert json.loads(json.dumps(lag_cont.state_dict())) == lag_cont.state_dict()


def test_resume_from_a_capped_history_is_equivalent(monkeypatch):
    """乘子被 max_multiplier 截断后的检查点，恢复语义同样必须等价。"""
    capped_specs = (
        ConstraintSpec(
            name="business", budget=0.0, unit=UNIT_VIOLATION_TASK_STEPS,
            learning_rate=1.0, max_multiplier=1.0,
        ),
        ConstraintSpec(
            name="carbon", budget=0.0, unit=UNIT_KG_CO2E,
            learning_rate=1.0, max_multiplier=1.0,
        ),
    )
    lag_cont = Lagrangian(capped_specs)
    for _ in range(3):
        run_round(monkeypatch, lag_cont, 0)
    assert lag_cont.multipliers() == {"business": 1.0, "carbon": 1.0}
    checkpoint = json.loads(json.dumps(lag_cont.state_dict()))

    cont = run_round(monkeypatch, lag_cont, 1)

    lag_resumed = Lagrangian(capped_specs)
    lag_resumed.load_state_dict(checkpoint)
    resumed = run_round(monkeypatch, lag_resumed, 1)

    assert_rounds_equivalent(cont, resumed)
    assert lag_resumed.state_dict() == lag_cont.state_dict()
    assert lag_resumed.multipliers() == {"business": 1.0, "carbon": 1.0}  # 仍在上限

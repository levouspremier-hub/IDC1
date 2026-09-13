"""M5.2d 测试：dry_run_update 逐 transition 的 next-value bootstrap。

用 monkeypatch 替换 collect_rollout，构造一个**两 episode** buffer：
t=0 截断且其 next_observation 与 t=1 的 observation 故意不同。
证明 target 用的是 t=0 自己的 next value，而不是下一条 transition 的。
"""

import copy
from typing import Any

import numpy as np
import pytest
import torch

from safe_rl_v2 import train as train_mod
from safe_rl_v2.buffer import ACTION_DIM, CONTRACT_VERSION, Transition
from safe_rl_v2.lagrangian import Lagrangian
from safe_rl_v2.policy import SafePPOPolicy
from safe_rl_v2.train import dry_run_update

OBS_DIM = 8
HEADS = ("reward", "business", "carbon")

# 两条 transition 的观测常量
OBS_0 = 0.10
NEXT_OBS_0 = 0.90  # t=0 的 next_observation
OBS_1 = 0.20  # t=1 的 observation（下一条 episode 的首步）
NEXT_OBS_1 = 0.30


def make_policy(seed: int = 0, obs_dim: int = OBS_DIM) -> SafePPOPolicy:
    torch.manual_seed(seed)
    return SafePPOPolicy(obs_dim=obs_dim)


def make_transition(value: float, next_value: float, **over) -> Transition:
    kwargs: dict[str, Any] = dict(
        observation=np.full(OBS_DIM, value, dtype=np.float32),
        next_observation=np.full(OBS_DIM, next_value, dtype=np.float32),
        raw_action=np.full(ACTION_DIM, 0.5, dtype=np.float32),
        old_raw_log_prob=-1.0,
        exec_action=np.full(ACTION_DIM, 0.5, dtype=np.float32),
        reward=1.0,
        business_cost=1.0,
        carbon_cost=2.0,
        electricity_cost_sgd=3.0,
        terminated=False,
        truncated=False,
        correction_info={},
    )
    kwargs.update(over)
    return Transition(**kwargs)


def two_episode_transitions(obs_1: float = OBS_1) -> list[Transition]:
    """t=0 截断、t=1 属下一 episode，且 V(next_obs[0]) 与 V(obs[1]) 必然不同。"""
    return [
        make_transition(OBS_0, NEXT_OBS_0, truncated=True),
        make_transition(obs_1, NEXT_OBS_1),
    ]


def install_fake_collector(monkeypatch, transitions: list[Transition]) -> None:
    """替换 collector：直接填充给定 transition，不接触真实环境。"""

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
            "terminated_count": sum(t.terminated for t in transitions),
            "truncated_count": sum(t.truncated for t in transitions),
            "contract_version": CONTRACT_VERSION,
            "action_dim": ACTION_DIM,
            "env_seed": seed,
            "policy_rng_source": (
                "explicit_generator" if generator is not None else "global_torch_rng"
            ),
        }

    monkeypatch.setattr(train_mod, "collect_rollout", fake)


def critic_values(policy: SafePPOPolicy, obs_value: float) -> dict[str, float]:
    """独立前向：某个常量观测的 critic 估计（逐头）。"""
    obs = torch.tensor([[obs_value] * OBS_DIM], dtype=torch.float32)
    with torch.no_grad():
        _, values = policy.forward(obs)
    return {head: float(values[0, index]) for index, head in enumerate(HEADS)}


def run_dry(policy, lagrangian, optimizer, *, env=None):
    return dry_run_update(
        env, policy, lagrangian, optimizer,
        steps=2, seed=0, corrector_on=False, generator=None,
    )


def capture_target_call(monkeypatch, policy, lagrangian, optimizer) -> dict:
    recorded: list[dict[str, Any]] = []
    real = train_mod.compute_three_value_targets

    def spy(*args, **kwargs):
        recorded.append({"args": args, "kwargs": kwargs})
        return real(*args, **kwargs)

    monkeypatch.setattr(train_mod, "compute_three_value_targets", spy)
    result = run_dry(policy, lagrangian, optimizer)
    assert len(recorded) == 1, "必须恰好调用一次 M5.2c 的三套 target API"
    result["_call"] = recorded[0]
    return result


# --- 1. 逐 transition 的 next value 必须来自本条 next_observation ---

def test_target_api_receives_per_transition_next_values(monkeypatch):
    install_fake_collector(monkeypatch, two_episode_transitions())
    policy = make_policy()
    snapshot = copy.deepcopy(policy)
    lagrangian = Lagrangian({"business": 5.0, "carbon": 3.0})
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    result = capture_target_call(monkeypatch, policy, lagrangian, optimizer)
    call = result["_call"]

    # 参数形态：两个 dict，逐头长度均为 T（不是 T+1）
    current_values, next_values = call["args"][3], call["args"][4]
    assert set(current_values) == set(next_values) == set(HEADS)
    for head in HEADS:
        assert len(current_values[head]) == 2, "current_values 必须长度 T=2"
        assert len(next_values[head]) == 2, "next_values 必须长度 T=2"

    expected_next0 = critic_values(snapshot, NEXT_OBS_0)  # V(next_obs[0])
    expected_cur1 = critic_values(snapshot, OBS_1)  # V(obs[1])
    assert expected_next0["reward"] != expected_cur1["reward"], "本测试必须可区分"

    for head in HEADS:
        assert next_values[head][0] == pytest.approx(expected_next0[head]), (
            f"{head}: next_values[0] 必须来自 t=0 自己的 next_observation"
        )
        assert current_values[head][1] == pytest.approx(expected_cur1[head]), (
            f"{head}: current_values[1] 必须来自 t=1 的 observation"
        )
        assert next_values[head][0] != pytest.approx(expected_cur1[head]), (
            "旧的下标 t+1 语义会误用 V(obs[1])"
        )


def test_bootstrap_metric_source_is_per_transition(monkeypatch):
    install_fake_collector(monkeypatch, two_episode_transitions())
    policy = make_policy()
    lagrangian = Lagrangian({"business": 5.0, "carbon": 3.0})
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    result = run_dry(policy, lagrangian, optimizer)
    boot = result["bootstrap"]
    assert boot["source"] == "buffer_transition_next_observations"
    assert boot["source"] != "buffer_last_next_observation"
    assert boot["truncated_count"] == 1
    assert boot["terminated_count"] == 0
    assert boot["bootstrapped_count"] == 2  # 两条都不是 terminated


def test_train_forwards_both_batches(monkeypatch):
    """critic 必须对 observation batch 与 next_observation batch 分别前向。"""
    install_fake_collector(monkeypatch, two_episode_transitions())
    policy = make_policy()
    lagrangian = Lagrangian({"business": 5.0, "carbon": 3.0})
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    forwards: list[np.ndarray] = []
    real_forward = SafePPOPolicy.forward

    def spy(self, obs):
        forwards.append(np.asarray(obs.detach().cpu().numpy()))
        return real_forward(self, obs)

    monkeypatch.setattr(SafePPOPolicy, "forward", spy)
    run_dry(policy, lagrangian, optimizer)

    def has_row(value: float) -> bool:
        return any(
            arr.ndim == 2 and arr.shape[0] == 2 and np.allclose(arr[0], value) for arr in forwards
        )

    assert has_row(OBS_0), "缺少 observation batch 的前向"
    assert has_row(NEXT_OBS_0), "缺少 next_observation batch 的前向"


# --- 2. 下一 episode 不得污染本条 episode 的 bootstrap ---

def test_second_episode_observation_does_not_pollute_first_episode_bootstrap(monkeypatch):
    """改动 t=1 的 observation，t=0 的截断 bootstrap 与其 target 必须不变。"""
    policy = make_policy()
    snapshot = copy.deepcopy(policy)

    def run(obs_1: float) -> tuple[dict, dict, dict]:
        with monkeypatch.context() as ctx:
            install_fake_collector(ctx, two_episode_transitions(obs_1=obs_1))
            lagrangian = Lagrangian({"business": 5.0, "carbon": 3.0})
            optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
            return capture_target_call(ctx, policy, lagrangian, optimizer)

    base = run(OBS_1)
    changed = run(0.55)

    # next_values[0] 只依赖 t=0 的 next_observation，必须逐头不变
    for head in HEADS:
        assert (
            base["_call"]["args"][4][head][0] == changed["_call"]["args"][4][head][0]
        ), f"{head}: t=0 的 bootstrap 被下一条 episode 污染"
        # current_values[1] 依赖 t=1 的 observation，必须改变
        assert (
            base["_call"]["args"][3][head][1] != changed["_call"]["args"][3][head][1]
        ), f"{head}: t=1 的 current value 应随其 observation 改变"

    # 端到端：t=0 的 target 不变（截断中断递推），t=1 的 target 改变
    for head in HEADS:
        np.testing.assert_allclose(
            base["critic_targets"][head][0], changed["critic_targets"][head][0]
        )
        assert not np.allclose(
            base["critic_targets"][head][1], changed["critic_targets"][head][1]
        )
    assert critic_values(snapshot, OBS_1) != critic_values(snapshot, 0.55)


def test_truncated_step_target_uses_its_own_next_value_end_to_end(monkeypatch):
    """端到端手算：t=0 截断时 target[0] = signal[0] + gamma * V(next_obs[0])。

    （delta = signal + gamma*V(next) - V(obs)，target = delta + V(obs)，两项相消；
    carry 被截断切断，故不继承 t=1。）
    """
    install_fake_collector(monkeypatch, two_episode_transitions())
    policy = make_policy()
    snapshot = copy.deepcopy(policy)
    lagrangian = Lagrangian({"business": 5.0, "carbon": 3.0})
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    result = run_dry(policy, lagrangian, optimizer)
    gamma = result["gae"]["gamma"]

    next0 = critic_values(snapshot, NEXT_OBS_0)
    expected_reward_target0 = 1.0 + gamma * next0["reward"]
    assert result["critic_targets"]["reward"][0] == pytest.approx(expected_reward_target0)

    # business signal = 1.0（夹具），carbon signal = 2.0
    assert result["critic_targets"]["business"][0] == pytest.approx(1.0 + gamma * next0["business"])
    assert result["critic_targets"]["carbon"][0] == pytest.approx(2.0 + gamma * next0["carbon"])


# --- 3. exec_action 不得进入任何 target ---

def test_exec_action_never_enters_targets(monkeypatch):
    """把 exec_action 改成荒谬值，三套 target 必须逐元素不变。"""
    policy = make_policy()

    def run(exec_value: float) -> dict:
        transitions = two_episode_transitions()
        for t in transitions:
            t.exec_action = np.full(ACTION_DIM, exec_value, dtype=np.float32)
        with monkeypatch.context() as ctx:
            install_fake_collector(ctx, transitions)
            lagrangian = Lagrangian({"business": 5.0, "carbon": 3.0})
            optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
            return run_dry(policy, lagrangian, optimizer)

    base, mutated = run(0.5), run(-1.0)
    for head in HEADS:
        np.testing.assert_array_equal(
            base["critic_targets"][head], mutated["critic_targets"][head]
        )


# --- 4. 单位与未训练声明保持不变 ---

def test_units_and_claims_unchanged(monkeypatch):
    install_fake_collector(monkeypatch, two_episode_transitions())
    policy = make_policy()
    lagrangian = Lagrangian({"business": 5.0, "carbon": 3.0})
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    result = run_dry(policy, lagrangian, optimizer)
    assert result["units"] == {
        "reward": "dimensionless",
        "business_violations": "violation_task_steps",
        "carbon_emissions": "kgCO2e",
        "electricity_cost_sgd": "SGD",
    }
    assert result["claims"]["trained"] is False
    assert result["targets_source"] == "compute_three_value_targets(terminated=, truncated=)"


def test_rollout_buffer_helper_is_unused_by_train(monkeypatch):
    """train 不得再构造 T+1 数组：target API 收到的必须是两个长度 T 的 dict。"""
    install_fake_collector(monkeypatch, two_episode_transitions())
    policy = make_policy()
    lagrangian = Lagrangian({"business": 5.0, "carbon": 3.0})
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    result = capture_target_call(monkeypatch, policy, lagrangian, optimizer)
    args = result["_call"]["args"]
    assert isinstance(args[3], dict) and isinstance(args[4], dict)
    assert len(result["buffer"]) == 2
    for head in HEADS:
        assert len(args[3][head]) == len(result["buffer"])
        assert len(args[4][head]) == len(result["buffer"])

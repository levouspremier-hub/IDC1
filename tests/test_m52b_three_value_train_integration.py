"""M5.2b 测试：dry_run_update 使用 M5.2a 的终端感知三套 target，输入全部来自 v6 buffer。

本卡不测 PPO 数学，也不声称训练有效：dry-run 仍只是「闭环可更新」的冒烟验证。
"""

import ast
import json
import pathlib
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from envs.idc_price_env import IDCPriceEnv20D
from safe_rl_v2 import models as models_mod
from safe_rl_v2 import train as train_mod
from safe_rl_v2.lagrangian import Lagrangian
from safe_rl_v2.policy import SafePPOPolicy
from safe_rl_v2.train import dry_run_update

ENV_SEED_KWARGS = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}
CORRECTOR_TIME_LIMIT_S = 0.05
STEPS = 4
HEADS = ("reward", "business", "carbon")


def make_env(**over):
    kwargs = dict(ENV_SEED_KWARGS)
    kwargs.update(over)
    return IDCPriceEnv20D(**kwargs)


def make_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def make_policy(env, seed: int = 0) -> SafePPOPolicy:
    torch.manual_seed(seed)  # 仅测试夹具：两次构造权重逐元素相同
    return SafePPOPolicy(obs_dim=env.obs_dim)


def make_setup(**over):
    env = make_env(**over)
    policy = make_policy(env)
    lagrangian = Lagrangian({"business": 5.0, "carbon": 3.0})
    return env, policy, lagrangian, torch.optim.Adam(policy.parameters(), lr=1e-3)


def run_dry(env, policy, lagrangian, optimizer, **over):
    kwargs = dict(steps=STEPS, seed=0, corrector_on=False, generator=make_generator(0))
    kwargs.update(over)
    return dry_run_update(env, policy, lagrangian, optimizer, **kwargs)


class _ForwardingWrapper(gym.Wrapper):
    def __getattr__(self, name: str):
        if name == "env" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def obs_dim(self) -> int:
        env: Any = self.env
        return int(env.obs_dim)


class _TruncatingEnv(_ForwardingWrapper):
    """在指定步把 terminated 改写成 truncated，用于构造真实截断 transition。"""

    def __init__(self, env, at_step: int):
        super().__init__(env)
        self._at = at_step
        self._n = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        index, self._n = self._n, self._n + 1
        if index == self._at:
            terminated, truncated = False, True
        return obs, reward, terminated, truncated, info


# --- 1. 必须调用 M5.2a 的唯一 target API ---

def test_train_calls_m52a_target_api_with_buffer_masks(monkeypatch):
    env, policy, lag, opt = make_setup()
    recorded: list[dict[str, Any]] = []
    real_targets = models_mod.compute_three_value_targets

    def spy(*args, **kwargs):
        recorded.append({"args": args, "kwargs": kwargs})
        return real_targets(*args, **kwargs)

    monkeypatch.setattr(train_mod, "compute_three_value_targets", spy)
    result = run_dry(env, policy, lag, opt)

    assert len(recorded) == 1, "必须恰好调用一次 M5.2a 的三套 target API"
    assert train_mod.compute_three_value_targets is not None
    call = recorded[0]
    assert set(call["kwargs"]) >= {"terminated", "truncated", "gamma", "lam"}

    buffer = result["buffer"]
    rewards = np.array([t.reward for t in buffer.transitions], dtype=np.float64)
    business = np.array([t.business_cost for t in buffer.transitions], dtype=np.float64)
    carbon = np.array([t.carbon_cost for t in buffer.transitions], dtype=np.float64)
    terminals = np.array([t.terminated for t in buffer.transitions], dtype=bool)
    truncations = np.array([t.truncated for t in buffer.transitions], dtype=bool)

    np.testing.assert_array_equal(call["args"][0], rewards)
    np.testing.assert_array_equal(call["args"][1], business)
    np.testing.assert_array_equal(call["args"][2], carbon)
    np.testing.assert_array_equal(call["kwargs"]["terminated"], terminals)
    np.testing.assert_array_equal(call["kwargs"]["truncated"], truncations)
    assert call["kwargs"]["terminated"].dtype == np.bool_
    assert call["kwargs"]["truncated"].dtype == np.bool_
    assert result["targets_source"] == "compute_three_value_targets(terminated=, truncated=)"


def test_train_does_not_reference_legacy_maskless_gae():
    tree = ast.parse(pathlib.Path(train_mod.__file__).read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "compute_gae" not in names | attrs, "旧的无掩码 GAE 不得再被引用"


def test_actor_likelihood_still_uses_m52a_independent_path(monkeypatch):
    """三套 target API 与 actor likelihood 是两个独立调用，不得互相替代。"""
    env, policy, lag, opt = make_setup()
    seen: list[str] = []
    real_eval = SafePPOPolicy.evaluate_raw_actions
    real_targets = models_mod.compute_three_value_targets

    def eval_spy(self, obs, raw_action):
        seen.append("evaluate_raw_actions")
        return real_eval(self, obs, raw_action)

    def target_spy(*args, **kwargs):
        seen.append("three_value_targets")
        return real_targets(*args, **kwargs)

    monkeypatch.setattr(SafePPOPolicy, "evaluate_raw_actions", eval_spy)
    monkeypatch.setattr(train_mod, "compute_three_value_targets", target_spy)
    run_dry(env, policy, lag, opt)

    assert seen.count("three_value_targets") == 1
    assert seen.count("evaluate_raw_actions") >= 1


# --- 2. target 输入只来自 buffer ---

def test_bootstrap_reads_only_last_next_observation(monkeypatch):
    env, policy, lag, opt = make_setup()
    forwards: list[np.ndarray] = []
    real_forward = SafePPOPolicy.forward

    def spy(self, obs):
        forwards.append(np.asarray(obs.detach().cpu().numpy()))
        return real_forward(self, obs)

    monkeypatch.setattr(SafePPOPolicy, "forward", spy)
    result = run_dry(env, policy, lag, opt)

    expected = np.asarray(result["buffer"].transitions[-1].next_observation, dtype=np.float32)
    assert any(
        arr.dtype == np.float32 and arr.shape == expected.shape and np.array_equal(arr, expected)
        for arr in forwards
    ), "必须有一次 forward 的输入恰为 buffer 最后一条 next_observation"


@pytest.mark.parametrize("corrector_on", [False, True])
def test_critic_targets_come_from_buffer_named_signals(corrector_on):
    env, policy, lag, opt = make_setup()
    result = run_dry(
        env, policy, lag, opt, corrector_on=corrector_on,
        corrector_time_limit_s=CORRECTOR_TIME_LIMIT_S if corrector_on else None,
    )
    buffer = result["buffer"]
    # target = advantage + value，故长度与 transition 数一致
    for head in HEADS:
        assert np.asarray(result["critic_targets"][head]).shape == (len(buffer),)
    # 奖励 target 必须真的依赖 buffer 的 reward（而非电费或别的量）
    rewards = np.array([t.reward for t in buffer.transitions])
    assert not np.allclose(result["critic_targets"]["reward"], np.zeros_like(rewards))


def test_electricity_cost_never_enters_any_target(monkeypatch):
    env, policy, lag, opt = make_setup()
    recorded: list[dict[str, Any]] = []
    real_targets = models_mod.compute_three_value_targets

    def spy(*args, **kwargs):
        recorded.append({"args": args, "kwargs": kwargs})
        return real_targets(*args, **kwargs)

    monkeypatch.setattr(train_mod, "compute_three_value_targets", spy)
    result = run_dry(env, policy, lag, opt)

    buffer = result["buffer"]
    electricity = np.array([t.electricity_cost_sgd for t in buffer.transitions], dtype=np.float64)
    passed = [np.asarray(a, dtype=np.float64) for a in recorded[0]["args"][:3]]
    for arr in passed:
        if arr.shape == electricity.shape:
            assert not np.allclose(arr, electricity), "电费不得作为任何一套 signal"

    # 电费在返回口径中仍与业务/碳完全分离，且单位各自标注
    units = result["units"]
    assert "electricity_cost_sgd" in units
    distinct = {
        units["business_violations"],
        units["carbon_emissions"],
        units["electricity_cost_sgd"],
    }
    assert len(distinct) == 3, "三类量纲必须互不相同"
    assert result["electricity_cost_sum_sgd"] > 0.0


# --- 3. 三种 bootstrap 行为 ---

def test_terminal_buffer_does_not_bootstrap():
    env, policy, lag, opt = make_setup(horizon=3)
    result = run_dry(env, policy, lag, opt, steps=50)
    buffer = result["buffer"]

    assert buffer.transitions[-1].terminated is True
    assert result["bootstrap"]["terminal"] is True
    assert result["bootstrap"]["used"] is False
    assert result["bootstrap"]["source"] == "buffer_last_next_observation"


def test_truncated_buffer_bootstraps_without_being_terminal():
    env = _TruncatingEnv(make_env(), at_step=1)
    policy = make_policy(env)
    lag = Lagrangian({"business": 5.0, "carbon": 3.0})
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    result = run_dry(env, policy, lag, opt, steps=10)
    buffer = result["buffer"]

    assert len(buffer) == 2, "第 1 步被截断后采集必须停止"
    assert buffer.transitions[-1].truncated is True
    assert buffer.transitions[-1].terminated is False
    assert result["bootstrap"]["terminal"] is False
    assert result["bootstrap"]["used"] is True, "截断步仍允许 bootstrap"


def test_non_terminal_runout_bootstraps():
    env, policy, lag, opt = make_setup(horizon=24)
    result = run_dry(env, policy, lag, opt, steps=3)
    assert len(result["buffer"]) == 3
    assert result["bootstrap"] == {
        "terminal": False,
        "source": "buffer_last_next_observation",
        "used": True,
    }


# --- 4. 三头 critic target 相互独立 ---

def _critic_losses(**over) -> dict:
    env, policy, lag, opt = make_setup()
    return run_dry(env, policy, lag, opt, **over)["critic_loss_by_head"]


def test_changing_carbon_target_leaves_reward_and_business_losses_untouched(monkeypatch):
    baseline = _critic_losses()

    real_targets = models_mod.compute_three_value_targets

    def shifted(*args, **kwargs):
        out = real_targets(*args, **kwargs)
        adv, tgt = out["carbon"]
        out["carbon"] = (adv, tgt + 100.0)
        return out

    with monkeypatch.context() as ctx:
        ctx.setattr(train_mod, "compute_three_value_targets", shifted)
        shifted_losses = _critic_losses()

    assert shifted_losses["carbon"] != pytest.approx(baseline["carbon"])
    assert shifted_losses["reward"] == pytest.approx(baseline["reward"])
    assert shifted_losses["business"] == pytest.approx(baseline["business"])


def test_critic_loss_is_the_sum_of_its_three_heads():
    env, policy, lag, opt = make_setup()
    result = run_dry(env, policy, lag, opt)
    by_head = result["critic_loss_by_head"]
    assert set(by_head) == set(HEADS)
    assert sum(by_head.values()) == pytest.approx(result["critic_loss"])


# --- 5. actor likelihood 仍只见 raw ---

def test_actor_likelihood_sees_raw_action_not_exec(monkeypatch):
    env, policy, lag, opt = make_setup()
    recorded: list[np.ndarray] = []
    real_eval = SafePPOPolicy.evaluate_raw_actions

    def spy(self, obs, raw_action):
        value = raw_action.detach().cpu().numpy() if torch.is_tensor(raw_action) else np.asarray(
            raw_action
        )
        recorded.append(np.asarray(value, dtype=np.float64))
        return real_eval(self, obs, raw_action)

    monkeypatch.setattr(SafePPOPolicy, "evaluate_raw_actions", spy)
    result = run_dry(
        env, policy, lag, opt,
        corrector_on=True, corrector_time_limit_s=CORRECTOR_TIME_LIMIT_S,
    )

    buffer = result["buffer"]
    raw_batch = np.stack([t.raw_action for t in buffer.transitions])
    exec_batch = np.stack([t.exec_action for t in buffer.transitions])
    assert not np.allclose(raw_batch, exec_batch), "corrector 未改变动作，本测试无法区分"

    batched = [r for r in recorded if r.ndim == 2]
    assert any(np.allclose(r, raw_batch) for r in batched)
    assert not any(np.allclose(r, exec_batch) for r in batched)


# --- 6. 泄漏回归：未来真值不得影响 target ---

@pytest.mark.leakage
def test_targets_unaffected_by_future_truth_mutation():
    def run(mutate: bool) -> dict:
        env = make_env(forecast_cutoff=2)
        if mutate:
            env.price_t[20] = 9999.0
            env.pv_t[20] = 9999.0
            env.carbon_factor_t[20] = 9999.0
        policy = make_policy(env, seed=3)
        lag = Lagrangian({"business": 5.0, "carbon": 3.0})
        opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        return run_dry(env, policy, lag, opt, steps=4)

    clean, mutated = run(False), run(True)

    for head in HEADS:
        np.testing.assert_array_equal(
            clean["critic_targets"][head], mutated["critic_targets"][head]
        )
    np.testing.assert_array_equal(
        clean["buffer"].transitions[0].raw_action, mutated["buffer"].transitions[0].raw_action
    )
    assert clean["critic_loss"] == pytest.approx(mutated["critic_loss"])


# --- 7. 指标必须标注单位与未训练 ---

def test_metrics_declare_units_and_claims():
    env, policy, lag, opt = make_setup()
    result = run_dry(env, policy, lag, opt)

    units = result["units"]
    assert units["business_violations"] == "violation_task_steps"
    assert units["carbon_emissions"] == "kgCO2e"
    assert units["electricity_cost_sgd"] == "SGD"
    assert units["reward"] == "dimensionless"

    assert result["claims"] == {
        "trained": False,
        "performance_evaluated": False,
        "convergence_claimed": False,
    }
    assert set(result["gae"]) == {"gamma", "lam"}
    # 除 buffer / stats / 数组型 target 外，返回指标必须可 JSON 序列化
    scalar_metrics = {
        k: v for k, v in result.items() if k not in {"buffer", "stats", "critic_targets"}
    }
    assert isinstance(json.dumps(scalar_metrics), str)


def test_metrics_make_no_performance_claim_by_key_name():
    forbidden = ("converg", "performance", "improve", "gain", "accuracy", "return_mean")
    env, policy, lag, opt = make_setup()
    result = run_dry(env, policy, lag, opt)
    offenders = [
        key
        for key in result
        if key != "claims" and any(s in key.lower() for s in forbidden)
    ]
    assert offenders == []

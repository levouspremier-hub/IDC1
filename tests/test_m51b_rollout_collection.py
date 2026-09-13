"""M5.1b 测试：真实 raw/exec rollout 采集接线。

本卡**不**实现也不测试 PPO ratio/clip、GAE、actor loss、熵项或拉格朗日乘子更新；
只测「有界 raw 采样 → 环境/修正器 → v6 transition」这条采集链的完整性与可复现性。
"""

import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from safe_rl_v2.rollout import (
    BUSINESS_VIOLATION_INFO_KEY,
    CARBON_EMISSION_INFO_KEY,
    ELECTRICITY_COST_INFO_KEY,
    MissingEnvInfoError,
    collect_rollout,
)

from envs.idc_price_env import IDCPriceEnv20D
from safe_rl_v2.buffer import ACTION_DIM, CONTRACT_VERSION, RolloutBuffer
from safe_rl_v2.policy import SafePPOPolicy

CORRECTOR_TIME_LIMIT_S = 0.05
STEPS = 3
N_COMPUTE = ACTION_DIM - 1


def _policy(env, seed: int = 0) -> SafePPOPolicy:
    torch.manual_seed(seed)
    return SafePPOPolicy(obs_dim=env.obs_dim)


def _obs_tensor(env) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.zeros(env.obs_dim, dtype=torch.float32)


class _DroppingEnv(gym.Wrapper):
    """把某个 info 键删掉，模拟环境信息缺失。"""

    def __init__(self, env, key: str):
        super().__init__(env)
        self._key = key

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.pop(self._key, None)
        return obs, reward, terminated, truncated, info


# --- 1. 有界 raw 动作采样 ---

def test_act_returns_bounded_raw_action():
    env = IDCPriceEnv20D()
    policy = _policy(env)
    obs = _obs_tensor(env)
    for _ in range(25):
        raw, log_prob, _ = policy.act(obs)
        arr = raw.detach().numpy()
        assert arr.shape == (ACTION_DIM,)
        assert np.all(arr[:N_COMPUTE] >= 0.0), "前 20 维必须 >= 0"
        assert np.all(arr[:N_COMPUTE] <= 1.0), "前 20 维必须 <= 1"
        assert np.all(arr[N_COMPUTE:] >= -1.0), "储能维必须 >= -1"
        assert np.all(arr[N_COMPUTE:] <= 1.0), "储能维必须 <= 1"
        assert np.isfinite(float(log_prob.detach()))


def test_act_returns_float32_compatible_with_env_action_space():
    env = IDCPriceEnv20D()
    policy = _policy(env)
    raw, _, _ = policy.act(_obs_tensor(env))
    arr = raw.detach().numpy().astype(np.float32)
    assert env.action_space.contains(arr)


# --- 2. log-prob 严格绑定最终 raw 动作 ---

def test_evaluate_raw_actions_reproduces_act_log_prob():
    """`evaluate_raw_actions` 必须复现 `act()` 对同一 raw 样本的 log-prob。"""
    env = IDCPriceEnv20D()
    policy = _policy(env)
    obs = _obs_tensor(env)
    for _ in range(10):
        raw, log_prob, _ = policy.act(obs)
        recomputed = policy.evaluate_raw_actions(obs, raw)
        assert float(recomputed.detach()) == pytest.approx(float(log_prob.detach()), abs=1e-6)


def test_evaluate_raw_actions_is_pure():
    env = IDCPriceEnv20D()
    policy = _policy(env)
    obs = _obs_tensor(env)
    raw, _, _ = policy.act(obs)
    snapshot = raw.detach().numpy().copy()
    policy.evaluate_raw_actions(obs, raw)
    np.testing.assert_array_equal(raw.detach().numpy(), snapshot)


def test_stored_log_prob_corresponds_to_stored_raw_action():
    """落库的 old_raw_log_prob 必须对应落库的 raw_action（禁止 clip 后沿用旧概率）。"""
    env = IDCPriceEnv20D()
    policy = _policy(env)
    buf = RolloutBuffer()
    collect_rollout(env, policy, buf, steps=STEPS, seed=0, corrector_on=False)

    for t in buf.transitions:
        obs_t = torch.as_tensor(t.observation, dtype=torch.float32)
        recomputed = policy.evaluate_raw_actions(obs_t, torch.as_tensor(t.raw_action))
        assert float(recomputed.detach()) == pytest.approx(t.old_raw_log_prob, abs=1e-5)


def test_raw_action_is_stored_verbatim_without_clipping():
    env = IDCPriceEnv20D()
    policy = _policy(env)
    obs = torch.zeros(env.obs_dim, dtype=torch.float32)

    # 固定 torch RNG 状态，使手工采样与采集器内首次采样是同一样本
    rng_state = torch.get_rng_state()
    raw, log_prob, _ = policy.act(obs)
    expected = raw.detach().numpy().astype(np.float64)

    torch.set_rng_state(rng_state)
    buf = RolloutBuffer()
    collect_rollout(env, policy, buf, steps=1, seed=0, corrector_on=False)
    stored = buf.transitions[0].raw_action

    np.testing.assert_allclose(stored, expected, atol=0.0)
    assert buf.transitions[0].old_raw_log_prob == pytest.approx(float(log_prob.detach()), abs=1e-6)


# --- 3. corrector 关闭：完整 transition ---

def test_collect_rollout_off_writes_complete_transition():
    env = IDCPriceEnv20D()
    policy = _policy(env)
    buf = RolloutBuffer()
    stats = collect_rollout(env, policy, buf, steps=STEPS, seed=0, corrector_on=False)

    assert stats["transitions"] == len(buf) == STEPS
    assert stats["corrector_on"] is False
    assert stats["contract_version"] == CONTRACT_VERSION

    for t in buf.transitions:
        assert t.observation.shape == (env.obs_dim,)
        assert t.next_observation.shape == (env.obs_dim,)
        assert t.raw_action.shape == (ACTION_DIM,)
        assert t.exec_action.shape == (ACTION_DIM,)
        # 无修正器时 exec 必须逐元素等于 raw
        np.testing.assert_allclose(t.exec_action, t.raw_action, atol=0.0)
        assert isinstance(t.terminated, bool)
        assert isinstance(t.truncated, bool)
        assert t.contract_version == CONTRACT_VERSION


def test_collect_rollout_off_records_named_quantities():
    env = IDCPriceEnv20D()
    policy = _policy(env)
    buf = RolloutBuffer()
    collect_rollout(env, policy, buf, steps=STEPS, seed=0, corrector_on=False)
    t = buf.transitions[0]
    # 具名业务违规量（计数）/ 具名碳排放量（kgCO2e）/ 电费（SGD）三者互不覆盖
    assert isinstance(t.business_cost, float) and t.business_cost >= 0.0
    assert isinstance(t.carbon_cost, float) and t.carbon_cost > 0.0
    assert isinstance(t.electricity_cost_sgd, float) and t.electricity_cost_sgd > 0.0
    assert len({t.business_cost, t.carbon_cost, t.electricity_cost_sgd}) >= 2


# --- 4. corrector 开启：raw 原样送入修正器 ---

def test_collect_rollout_on_sends_raw_action_verbatim_to_corrector():
    env = IDCPriceEnv20D()
    policy = _policy(env)
    buf = RolloutBuffer()
    stats = collect_rollout(
        env, policy, buf, steps=STEPS, seed=0,
        corrector_on=True, corrector_time_limit_s=CORRECTOR_TIME_LIMIT_S,
    )

    assert stats["corrector_on"] is True
    assert stats["transitions"] == len(buf) == STEPS

    for t in buf.transitions:
        # CorrectorWrapper 把它**收到**的动作写回 info["raw_action"]：
        # 两者逐元素相等即证明采集器没有 clip / 改写采样动作。
        assert "raw_action" in t.correction_info
        received = np.asarray(t.correction_info["raw_action"], dtype=np.float32)
        np.testing.assert_array_equal(received, t.raw_action.astype(np.float32))
        assert "correction_reason" in t.correction_info
        assert "correction_solve_time_s" in t.correction_info


def test_collect_rollout_on_requires_explicit_time_limit():
    env = IDCPriceEnv20D()
    policy = _policy(env)
    with pytest.raises(ValueError, match="corrector_time_limit_s"):
        collect_rollout(env, policy, RolloutBuffer(), steps=1, seed=0, corrector_on=True)


def test_collect_rollout_off_rejects_unused_time_limit():
    env = IDCPriceEnv20D()
    policy = _policy(env)
    with pytest.raises(ValueError, match="corrector_time_limit_s"):
        collect_rollout(
            env, policy, RolloutBuffer(), steps=1, seed=0,
            corrector_on=False, corrector_time_limit_s=CORRECTOR_TIME_LIMIT_S,
        )


# --- 5. 环境信息缺失必须报错，不得伪造零 ---

@pytest.mark.parametrize(
    "key",
    [BUSINESS_VIOLATION_INFO_KEY, CARBON_EMISSION_INFO_KEY, ELECTRICITY_COST_INFO_KEY],
)
def test_missing_env_info_raises_instead_of_faking_zero(key):
    env = _DroppingEnv(IDCPriceEnv20D(), key)
    policy = _policy(env)
    with pytest.raises(MissingEnvInfoError, match=key):
        collect_rollout(env, policy, RolloutBuffer(), steps=1, seed=0, corrector_on=False)


def test_missing_correction_audit_field_raises():
    """corrector 开启时审计字段缺失也必须报错，不得用默认值补齐。"""
    env = _DroppingEnv(IDCPriceEnv20D(), "correction_reason")
    policy = _policy(env)
    with pytest.raises(MissingEnvInfoError, match="correction_reason"):
        collect_rollout(
            env, policy, RolloutBuffer(), steps=1, seed=0,
            corrector_on=True, corrector_time_limit_s=CORRECTOR_TIME_LIMIT_S,
        )


# --- 6. correction_info 必须 JSON 安全 ---

@pytest.mark.parametrize("corrector_on", [False, True])
def test_correction_info_is_json_safe(corrector_on):
    env = IDCPriceEnv20D()
    policy = _policy(env)
    buf = RolloutBuffer()
    collect_rollout(
        env, policy, buf, steps=STEPS, seed=0,
        corrector_on=corrector_on,
        corrector_time_limit_s=CORRECTOR_TIME_LIMIT_S if corrector_on else None,
    )
    for t in buf.transitions:
        json.dumps(t.correction_info)  # 不抛异常即 JSON 安全
    assert isinstance(json.dumps(buf.to_dict()), str)


def test_buffer_payload_from_rollout_roundtrips():
    env = IDCPriceEnv20D()
    policy = _policy(env)
    buf = RolloutBuffer()
    collect_rollout(env, policy, buf, steps=STEPS, seed=0, corrector_on=False)
    restored = RolloutBuffer.from_dict(json.loads(json.dumps(buf.to_dict())))
    assert len(restored) == len(buf)
    for a, b in zip(buf.transitions, restored.transitions, strict=True):
        np.testing.assert_allclose(a.raw_action, b.raw_action, atol=0.0)
        assert a.old_raw_log_prob == b.old_raw_log_prob


# --- 7. 终止语义与确定性 ---

def test_rollout_stops_at_terminal_step():
    env = IDCPriceEnv20D(horizon=4)
    policy = _policy(env)
    buf = RolloutBuffer()
    stats = collect_rollout(env, policy, buf, steps=50, seed=0, corrector_on=False)

    assert len(buf) == 4, "env 在 horizon 处终止，采集必须停止"
    assert stats["terminated_count"] + stats["truncated_count"] == 1
    assert buf.transitions[-1].terminated or buf.transitions[-1].truncated


def test_rollout_is_deterministic_given_seed():
    def raws(corrector_on: bool) -> np.ndarray:
        env = IDCPriceEnv20D()
        policy = _policy(env, seed=7)
        buf = RolloutBuffer()
        collect_rollout(
            env, policy, buf, steps=STEPS, seed=0,
            corrector_on=corrector_on,
            corrector_time_limit_s=CORRECTOR_TIME_LIMIT_S if corrector_on else None,
        )
        return np.array([t.raw_action for t in buf.transitions])

    np.testing.assert_array_equal(raws(False), raws(False))
    np.testing.assert_array_equal(raws(True), raws(True))


# --- 8. 本卡不训练 ---

@pytest.mark.parametrize("corrector_on", [False, True])
def test_collect_rollout_does_not_touch_policy_parameters(corrector_on):
    env = IDCPriceEnv20D()
    policy = _policy(env)
    before = [p.detach().clone() for p in policy.parameters()]
    collect_rollout(
        env, policy, RolloutBuffer(), steps=STEPS, seed=0,
        corrector_on=corrector_on,
        corrector_time_limit_s=CORRECTOR_TIME_LIMIT_S if corrector_on else None,
    )
    for old, new in zip(before, policy.parameters(), strict=True):
        assert torch.equal(old, new.detach()), "本卡不得更新任何策略参数"


def test_rollout_module_exposes_no_ppo_update_api():
    """本卡不提供 ratio/clip/GAE/actor-loss/乘子更新入口。"""
    from safe_rl_v2 import rollout as mod

    forbidden = ("ppo", "ratio", "clip", "gae", "advantage", "actor_loss", "lagrang")
    public = [n for n in dir(mod) if not n.startswith("_")]
    assert [n for n in public if any(f in n.lower() for f in forbidden)] == []

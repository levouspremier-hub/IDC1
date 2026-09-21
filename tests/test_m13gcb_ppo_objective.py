"""M1.3g-f-c-b：PPO clipped actor objective 核心。

**只**验证数值计算，不接训练入口、不运行正式训练、不写 checkpoint。

覆盖：`ratio = exp(new − old)`、三头优势
`A_reward − λ_business·A_business − λ_carbon·A_carbon`、
正负优势下的 `min(ratio×A, clip(ratio,1−ε,1+ε)×A)`、
**新 log-prob 只能由 `raw_action` 求得**（corrector 使 raw≠exec 时结果必须不同）、
以及 `clip_epsilon` 必须由调用方显式传入。

**改前缺陷（本文件对应先红）**：`safe_rl_v2/ppo_objective.py` 尚不存在。
"""

import importlib
import pathlib

import numpy as np
import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE = "safe_rl_v2.ppo_objective"


def ppo():
    return importlib.import_module(MODULE)


# =============================================================================
# 1. ratio
# =============================================================================

def test_ratio_is_exp_of_the_log_prob_difference():
    m = ppo()
    old = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float64)
    new = torch.tensor([0.0, 1.0 + float(np.log(2.0)), -2.0 + float(np.log(0.5))],
                       dtype=torch.float64)
    ratio = m.compute_ratio(new, old)
    assert torch.allclose(ratio, torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64),
                          atol=1e-12)
    # ratio == 1 边界：new 与 old 完全相同时必须**精确**为 1
    same = m.compute_ratio(old, old)
    assert torch.equal(same, torch.ones_like(old))


def test_ratio_requires_matching_shapes():
    m = ppo()
    with pytest.raises((ValueError, RuntimeError)):
        m.compute_ratio(torch.zeros(3), torch.zeros(4))


# =============================================================================
# 2. 三头优势语义
# =============================================================================

def test_effective_advantage_uses_the_three_head_semantics():
    m = ppo()
    adv_r = torch.tensor([1.0, -1.0], dtype=torch.float64)
    adv_b = torch.tensor([2.0, 0.0], dtype=torch.float64)
    adv_c = torch.tensor([0.0, 4.0], dtype=torch.float64)
    lam_b, lam_c = 0.5, 0.25
    got = m.effective_advantage(adv_r, adv_b, adv_c,
                                lambda_business=lam_b, lambda_carbon=lam_c)
    expected = adv_r - lam_b * adv_b - lam_c * adv_c
    assert torch.allclose(got, expected)
    assert torch.allclose(got, torch.tensor([1.0 - 1.0, -1.0 - 1.0], dtype=torch.float64))
    # 非空洞性：三个乘子都必须真的参与（缺任一项结果不同）
    assert not torch.allclose(got, adv_r - lam_c * adv_c)
    assert not torch.allclose(got, adv_r - lam_b * adv_b)


# =============================================================================
# 3. clipped surrogate：正 / 负优势 × 上下越界
# =============================================================================

@pytest.mark.parametrize("ratio_value", [1.0, 0.5, 1.5, 0.1, 10.0])
@pytest.mark.parametrize("advantage_value", [2.0, -2.0])
def test_clipped_surrogate_matches_the_hand_computed_min(ratio_value, advantage_value):
    m = ppo()
    eps = 0.2
    ratio = torch.tensor([ratio_value], dtype=torch.float64)
    advantage = torch.tensor([advantage_value], dtype=torch.float64)

    got = m.clipped_surrogate(ratio, advantage, clip_epsilon=eps)

    unclipped = ratio_value * advantage_value
    clipped_ratio = min(max(ratio_value, 1.0 - eps), 1.0 + eps)
    expected = min(unclipped, clipped_ratio * advantage_value)
    assert float(got[0]) == pytest.approx(expected, abs=1e-12)
    # 手算锚点：ratio=10, A=-2 → min(-20, 1.2×-2=-2.4) = -20（**不**被 clip 救回）
    if ratio_value == 10.0 and advantage_value == -2.0:
        assert float(got[0]) == pytest.approx(-20.0)
    # 手算锚点：ratio=10, A=+2 → min(20, 2.4) = 2.4（被上界 clip）
    if ratio_value == 10.0 and advantage_value == 2.0:
        assert float(got[0]) == pytest.approx(2.4)


def test_clipping_only_bites_outside_the_epsilon_band():
    m = ppo()
    eps = 0.2
    inside = torch.tensor([0.9, 1.0, 1.1], dtype=torch.float64)
    adv = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64)
    got = m.clipped_surrogate(inside, adv, clip_epsilon=eps)
    assert torch.allclose(got, inside * adv), "ε 带内不得被 clip 影响"


def test_clip_epsilon_must_be_a_valid_probability_like_number():
    m = ppo()
    ratio = torch.tensor([1.5], dtype=torch.float64)
    adv = torch.tensor([1.0], dtype=torch.float64)
    for bad in (0.0, -0.1, 1.0, 1.5, float("nan")):
        with pytest.raises((ValueError, TypeError)):
            m.clipped_surrogate(ratio, adv, clip_epsilon=bad)


# =============================================================================
# 4. clip_epsilon 必须显式传入（不擅自冻结超参数）
# =============================================================================

def test_clip_epsilon_has_no_default():
    m = ppo()
    ratio = torch.tensor([1.5], dtype=torch.float64)
    adv = torch.tensor([1.0], dtype=torch.float64)
    with pytest.raises(TypeError):
        m.clipped_surrogate(ratio, adv)  # type: ignore[call-arg]


# =============================================================================
# 5. 新 log-prob 只能由 raw_action 求得（绝不以 exec_action 替代）
# =============================================================================

def _tiny_policy(obs_dim: int = 5, seed: int = 0):
    from safe_rl_v2.policy import SafePPOPolicy

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return SafePPOPolicy(obs_dim=obs_dim)


def test_objective_recomputes_log_prob_from_the_raw_action_only():
    """传 `raw_action` 与传 `exec_action` 必须给出**不同**结果（raw ≠ exec 时）。"""
    m = ppo()
    policy = _tiny_policy()
    obs = torch.zeros(2, 5, dtype=torch.float32)
    raw = torch.tensor([[0.1] * 21, [-0.2] * 21], dtype=torch.float32)
    exec_ = torch.tensor([[0.9] * 21, [-0.8] * 21], dtype=torch.float32)
    assert not torch.equal(raw, exec_), "本用例要求 raw ≠ exec"

    old = policy.evaluate_raw_actions(obs, raw).detach()
    adv = {"reward": torch.tensor([1.0, 1.0]), "business": torch.tensor([0.0, 0.0]),
           "carbon": torch.tensor([0.0, 0.0])}

    from_raw = m.ppo_clipped_actor_objective(
        policy, observation=obs, raw_action=raw, old_raw_log_prob=old,
        adv_reward=adv["reward"], adv_business=adv["business"], adv_carbon=adv["carbon"],
        lambda_business=0.5, lambda_carbon=0.25, clip_epsilon=0.2)
    # 错误做法：拿 exec 当 raw。结果必须不同 —— 证明实现确实用了传入的 raw_action。
    wrong = m.ppo_clipped_actor_objective(
        policy, observation=obs, raw_action=exec_, old_raw_log_prob=old,
        adv_reward=adv["reward"], adv_business=adv["business"], adv_carbon=adv["carbon"],
        lambda_business=0.5, lambda_carbon=0.25, clip_epsilon=0.2)

    assert not torch.allclose(from_raw["new_raw_log_prob"], wrong["new_raw_log_prob"])
    assert not torch.allclose(from_raw["ratio"], wrong["ratio"])
    assert float(from_raw["loss"]) != pytest.approx(float(wrong["loss"]))
    # 正确调用必须等于对 raw_action 的直接重算
    assert torch.allclose(
        from_raw["new_raw_log_prob"],
        policy.evaluate_raw_actions(obs, raw).detach(), atol=1e-6)


def test_objective_api_has_no_exec_action_parameter():
    """**结构性守卫**：该 API 不得提供任何 `exec_action` 入参。"""
    import inspect

    m = ppo()
    for name in ("ppo_clipped_actor_objective", "ppo_actor_objective_from_buffer"):
        params = inspect.signature(getattr(m, name)).parameters
        assert not any("exec" in p for p in params), (name, list(params))


# =============================================================================
# 6. 与现有 buffer 契约对接
# =============================================================================

def _formal_buffer(steps: int = 4):
    """从**已验证的正式链**采集一小段真实 buffer（复用 g-e-d/c-a 的既有能力）。"""
    ei = importlib.import_module("scenario.env_injection")
    env_cls = importlib.import_module("envs.idc_price_env").IDCPriceEnv20D
    inj = ei.build_verified_formal_env_injection(
        "train", start="2024-01-02T00:00:00+08:00", horizon=8, forecast_cutoff=4)
    env = env_cls(horizon=8, task_seed=0, server_seed=0, forecast_seed=300000,
                  delta_t_hours=0.5, formal_injection=inj)
    policy = _tiny_policy(obs_dim=env.obs_dim)
    from safe_rl_v2.buffer import RolloutBuffer
    from safe_rl_v2.rollout import collect_rollout

    generator = torch.Generator()
    generator.manual_seed(0)
    buffer = RolloutBuffer()
    collect_rollout(env, policy, buffer, steps=steps, seed=0, generator=generator)
    return buffer, policy


def test_objective_consumes_a_real_collected_buffer():
    """用真实 buffer 的 `observation` / `raw_action` / `old_raw_log_prob` 计算。"""
    m = ppo()
    buffer, policy = _formal_buffer(steps=4)
    n = len(buffer)
    assert n > 0

    obs = torch.as_tensor(np.stack([t.observation for t in buffer.transitions]),
                          dtype=torch.float32)
    raw = torch.as_tensor(np.stack([t.raw_action for t in buffer.transitions]),
                          dtype=torch.float32)
    old = torch.as_tensor([t.old_raw_log_prob for t in buffer.transitions],
                          dtype=torch.float32)
    adv_r = torch.as_tensor([t.reward for t in buffer.transitions], dtype=torch.float32)
    adv_b = torch.as_tensor([t.business_cost for t in buffer.transitions],
                            dtype=torch.float32)
    adv_c = torch.as_tensor([t.carbon_cost for t in buffer.transitions],
                            dtype=torch.float32)

    out = m.ppo_clipped_actor_objective(
        policy, observation=obs, raw_action=raw, old_raw_log_prob=old,
        adv_reward=adv_r, adv_business=adv_b, adv_carbon=adv_c,
        lambda_business=0.5, lambda_carbon=0.25, clip_epsilon=0.2)

    assert out["ratio"].shape == (n,)
    assert out["new_raw_log_prob"].shape == (n,)
    assert torch.isfinite(out["loss"])
    # 采集时 `old_raw_log_prob` 就是该 raw_action 下的真实概率 ⇒ ratio ≈ 1
    assert torch.allclose(out["ratio"], torch.ones(n), atol=1e-5)


def test_buffer_helper_reads_the_transition_contract_fields():
    m = ppo()
    buffer, policy = _formal_buffer(steps=3)
    out = m.ppo_actor_objective_from_buffer(
        policy, buffer,
        adv_reward=[float(t.reward) for t in buffer.transitions],
        adv_business=[float(t.business_cost) for t in buffer.transitions],
        adv_carbon=[float(t.carbon_cost) for t in buffer.transitions],
        lambda_business=0.5, lambda_carbon=0.25, clip_epsilon=0.2)
    assert out["num_transitions"] == len(buffer)
    assert len(out["ratio"]) == len(buffer)
    assert torch.isfinite(out["loss"])
    # 非空洞性：ratio 必须接近 1（同一策略、同一动作）
    assert float(torch.max(torch.abs(out["ratio"] - 1.0))) < 1e-4

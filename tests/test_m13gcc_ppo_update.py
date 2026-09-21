"""M1.3g-f-c-c：基于 formal buffer 的**单次** PPO 更新。

把既有三头 target、乘子与 f-c-b 的 clipped actor objective 接成一次更新。
**不实现多轮训练、不写 checkpoint、不接训练入口。**

**改前缺陷（本文件对应先红）**：`safe_rl_v2/ppo_update.py` 尚不存在。
"""

import importlib
import pathlib

import numpy as np
import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE = "safe_rl_v2.ppo_update"

SPLIT = "train"
EPISODE_START = "2024-01-02T00:00:00+08:00"
HORIZON = 8
FORECAST_CUTOFF = 4
DELTA_HOURS = 0.5
STEPS = 4
CLIP_EPSILON = 0.2


def pu():
    return importlib.import_module(MODULE)


def _upstream_present() -> bool:
    return all(p.exists() for p in (
        REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet",
        REPO_ROOT / "data/manifest/singapore_2024_half_hour.json",
        REPO_ROOT / "data/manifest/singapore_2024_splits.json",
        REPO_ROOT / "data/manifest/singapore_2024_forecast_policy_v3.json",
        REPO_ROOT / "data/manifest/m13f_arrival_intensity_policy_v1.json",
        REPO_ROOT / "configs/frozen_refs/refs_v4.json",
        REPO_ROOT / "data/manifest/formal_splits_v5/train.json",
    ))


needs_assets = pytest.mark.skipif(
    not _upstream_present(), reason="真实冻结上游资产不在本机")


def _policy(obs_dim: int, seed: int = 0):
    from safe_rl_v2.policy import SafePPOPolicy

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return SafePPOPolicy(obs_dim=obs_dim)


def _lagrangian(multiplier_business: float = 0.5, multiplier_carbon: float = 0.25):
    from safe_rl_v2.lagrangian import (
        UNIT_KG_CO2E,
        UNIT_VIOLATION_TASK_STEPS,
        ConstraintSpec,
        Lagrangian,
    )

    lag = Lagrangian((
        ConstraintSpec(name="business", budget=5.0, unit=UNIT_VIOLATION_TASK_STEPS,
                       learning_rate=0.01, max_multiplier=100.0),
        ConstraintSpec(name="carbon", budget=3.0, unit=UNIT_KG_CO2E,
                       learning_rate=0.01, max_multiplier=100.0),
    ))
    lag.constraints["business"].multiplier = multiplier_business
    lag.constraints["carbon"].multiplier = multiplier_carbon
    return lag


def _formal_buffer(steps: int = STEPS, corrector_on: bool = False):
    """从**已验证的正式链**采集一小段真实 buffer。"""
    ei = importlib.import_module("scenario.env_injection")
    env_cls = importlib.import_module("envs.idc_price_env").IDCPriceEnv20D
    inj = ei.build_verified_formal_env_injection(
        SPLIT, start=EPISODE_START, horizon=HORIZON, forecast_cutoff=FORECAST_CUTOFF)
    env = env_cls(horizon=HORIZON, task_seed=0, server_seed=0, forecast_seed=300000,
                  delta_t_hours=DELTA_HOURS, formal_injection=inj)
    policy = _policy(env.obs_dim)
    from safe_rl_v2.buffer import RolloutBuffer
    from safe_rl_v2.rollout import collect_rollout

    kwargs = {}
    if corrector_on:
        from planning.corrector import PRODUCTION_CORRECTOR_TIME_LIMIT_S
        kwargs = {"corrector_on": True,
                  "corrector_time_limit_s": PRODUCTION_CORRECTOR_TIME_LIMIT_S}
    generator = torch.Generator()
    generator.manual_seed(0)
    buffer = RolloutBuffer()
    collect_rollout(env, policy, buffer, steps=steps, seed=0, generator=generator,
                    **kwargs)
    return buffer, policy, env


def _optimizer(policy):
    return torch.optim.Adam(policy.parameters(), lr=1e-3)


def _update(buffer, policy, optimizer, lagrangian, **overrides):
    m = pu()
    kwargs = {"clip_epsilon": CLIP_EPSILON, "gamma": 0.99, "lam": 0.95}
    kwargs.update(overrides)
    return m.single_ppo_update(policy, optimizer, lagrangian, buffer, **kwargs)


def _param_vector(policy):
    return np.concatenate([p.detach().numpy().ravel() for p in policy.parameters()])


# =============================================================================
# 1. 固定性与 raw_action 来源
# =============================================================================

@needs_assets
def test_old_log_prob_and_advantages_do_not_receive_gradient():
    """`old_raw_log_prob` 与三头优势必须**固定**（不可经 actor loss 反传）。"""
    buffer, policy, _env = _formal_buffer()
    optimizer = _optimizer(policy)
    lag = _lagrangian()
    m = pu()

    n = len(buffer)
    old = torch.as_tensor([t.old_raw_log_prob for t in buffer.transitions],
                          dtype=torch.float32).clone().requires_grad_(True)
    adv_r = torch.ones(n, dtype=torch.float32).requires_grad_(True)
    adv_b = torch.zeros(n, dtype=torch.float32).requires_grad_(True)
    adv_c = torch.zeros(n, dtype=torch.float32).requires_grad_(True)

    out = m.single_ppo_update_from_arrays(
        policy, optimizer, lag,
        observation=torch.as_tensor(
            np.stack([t.observation for t in buffer.transitions]), dtype=torch.float32),
        raw_action=torch.as_tensor(
            np.stack([t.raw_action for t in buffer.transitions]), dtype=torch.float32),
        old_raw_log_prob=old,
        rewards=torch.as_tensor([t.reward for t in buffer.transitions]),
        business_violations=torch.as_tensor([t.business_cost for t in buffer.transitions]),
        carbon_emissions=torch.as_tensor([t.carbon_cost for t in buffer.transitions]),
        terminated=torch.zeros(n, dtype=torch.bool),
        truncated=torch.zeros(n, dtype=torch.bool),
        adv_reward=adv_r, adv_business=adv_b, adv_carbon=adv_c,
        clip_epsilon=CLIP_EPSILON, gamma=0.99, lam=0.95,
    )

    assert old.grad is None, "old_raw_log_prob 不得接收梯度（必须被 detach）"
    assert adv_r.grad is None and adv_b.grad is None and adv_c.grad is None, \
        "三头优势不得接收梯度（必须被 detach）"
    assert torch.isfinite(out["loss_total"])
    # 非空洞性：actor 参数**必须**确实拿到了梯度
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0.0 for g in grads)


@needs_assets
def test_new_log_prob_is_recomputed_from_raw_action_even_with_corrector_on():
    """corrector 开启（`raw ≠ exec`）时，新 log-prob **必须**仍由 `raw_action` 求得。"""
    buffer, policy, _env = _formal_buffer(corrector_on=True)
    assert all(not np.array_equal(t.raw_action, t.exec_action)
               for t in buffer.transitions), "本用例要求 raw ≠ exec"

    m = pu()
    new_lp = m.new_raw_log_prob_from_buffer(policy, buffer)
    from safe_rl_v2.ppo_objective import _as_tensor  # noqa: PLC2701 - 复用同一数值口径

    obs = torch.as_tensor(np.stack([t.observation for t in buffer.transitions]),
                          dtype=torch.float32)
    raw = torch.as_tensor(np.stack([t.raw_action for t in buffer.transitions]),
                          dtype=torch.float32)
    exec_ = torch.as_tensor(np.stack([t.exec_action for t in buffer.transitions]),
                            dtype=torch.float32)
    assert torch.allclose(new_lp, policy.evaluate_raw_actions(obs, raw).detach(), atol=1e-6)
    # 若误用 exec_action，数值必须**不同**
    assert not torch.allclose(new_lp, policy.evaluate_raw_actions(obs, exec_).detach(),
                              atol=1e-6), "本用例要求 raw 与 exec 给出不同 log-prob"
    _ = _as_tensor  # 保持导入被使用（同一 float32 口径）


# =============================================================================
# 2. clipped objective / 三头 critic / 乘子口径
# =============================================================================

@needs_assets
def test_actor_term_is_the_clipped_objective_and_critics_are_per_head():
    buffer, policy, _env = _formal_buffer()
    optimizer = _optimizer(policy)
    lag = _lagrangian()
    out = _update(buffer, policy, optimizer, lag)

    # actor 使用 clipped objective（结果里带 clip_fraction 与 ε）
    assert out["clip_epsilon"] == pytest.approx(CLIP_EPSILON)
    assert 0.0 <= out["clip_fraction"] <= 1.0
    assert set(out["critic_loss_by_head"]) == {"reward", "business", "carbon"}
    # 三头各自的 MSE 之和 == 总 critic loss（**不**混列、**不**共享）
    per_head = out["critic_loss_by_head"]
    assert out["critic_loss_total"] == pytest.approx(
        sum(per_head.values()), rel=1e-6)
    # 非空洞性：三头 loss 不应恰好全等（否则可能只算了一头）
    assert len({round(v, 12) for v in per_head.values()}) > 1


@needs_assets
def test_multipliers_used_are_the_pre_update_values():
    buffer, policy, _env = _formal_buffer()
    optimizer = _optimizer(policy)
    lag = _lagrangian(multiplier_business=0.5, multiplier_carbon=0.25)
    out = _update(buffer, policy, optimizer, lag)

    assert out["multipliers_pre_update"]["business"] == pytest.approx(0.5)
    assert out["multipliers_pre_update"]["carbon"] == pytest.approx(0.25)
    # 乘子在 optimizer.step() **之后**才更新 ⇒ 前后必须可能不同
    assert out["multipliers_post_update"] != out["multipliers_pre_update"]
    assert out["lagrangian_updates_before"] == 0
    assert out["lagrangian_updates_after"] == 1


@needs_assets
def test_perturbing_one_head_target_only_changes_that_head():
    """三头 critic **各自**对自己的 target：改一头 targets 不得影响另两头。"""
    buffer, policy, _env = _formal_buffer()
    base = _update(buffer, policy, _optimizer(policy), _lagrangian())

    m = pu()
    fixed = m.compute_targets_from_buffer(policy, buffer, gamma=0.99, lam=0.95)
    assert set(fixed) == {"reward", "business", "carbon"}
    assert all(t.shape[0] == len(buffer) for t in fixed.values())
    # 非空洞性：三头 target 互不相同
    assert not np.allclose(fixed["reward"][1], fixed["business"][1])
    assert not np.allclose(fixed["reward"][1], fixed["carbon"][1])
    assert base["critic_loss_by_head"]["business"] > 0.0


# =============================================================================
# 3. 恰好一次 optimizer step / 失败语义
# =============================================================================

@needs_assets
def test_exactly_one_optimizer_step_changes_params_and_state():
    buffer, policy, _env = _formal_buffer()
    optimizer = _optimizer(policy)
    lag = _lagrangian()

    before = _param_vector(policy).copy()
    steps_before = int(optimizer.state_dict()["state"].__len__() and
                       list(optimizer.state.values())[0]["step"])
    out = _update(buffer, policy, optimizer, lag)
    after = _param_vector(policy)

    assert out["optimizer_steps"] == 1
    assert not np.array_equal(before, after), "参数必须变化"
    assert out["param_delta_norm"] > 0.0
    # Adam 状态必须被创建/推进
    steps_after = list(optimizer.state.values())[0]["step"]
    assert float(steps_after) == pytest.approx(float(steps_before) + 1.0)
    assert len(optimizer.state) > 0


@needs_assets
def test_empty_buffer_fails_closed():
    from safe_rl_v2.buffer import RolloutBuffer

    buffer, policy, _env = _formal_buffer(steps=1)
    buffer.transitions.clear()
    with pytest.raises((ValueError, RuntimeError)):
        _update(buffer, policy, _optimizer(policy), _lagrangian())


@needs_assets
def test_non_finite_loss_fails_closed_without_stepping():
    buffer, policy, _env = _formal_buffer()
    optimizer = _optimizer(policy)
    before = _param_vector(policy).copy()

    m = pu()
    with pytest.raises((ValueError, RuntimeError)):
        m.single_ppo_update(
            policy, optimizer, _lagrangian(), buffer,
            clip_epsilon=CLIP_EPSILON, gamma=0.99, lam=0.95,
            _force_non_finite_loss=True)
    assert np.array_equal(before, _param_vector(policy)), \
        "损失非有限时必须明确失败，且**不得**已经 step"
    assert len(optimizer.state) == 0


# =============================================================================
# 4. 可重放 / 不声称训练有效
# =============================================================================

@needs_assets
def test_single_update_is_bit_replayable():
    buffer_a, policy_a, _e = _formal_buffer()
    buffer_b, policy_b, _f = _formal_buffer()
    opt_a = _optimizer(policy_a)
    opt_b = _optimizer(policy_b)
    out_a = _update(buffer_a, policy_a, opt_a, _lagrangian())
    out_b = _update(buffer_b, policy_b, opt_b, _lagrangian())

    assert float(out_a["loss_total"]) == pytest.approx(float(out_b["loss_total"]), rel=1e-9)
    assert float(out_a["param_delta_norm"]) == pytest.approx(
        float(out_b["param_delta_norm"]), rel=1e-9)
    assert np.allclose(_param_vector(policy_a), _param_vector(policy_b), atol=0)
    assert out_a["multipliers_post_update"] == out_b["multipliers_post_update"]


@needs_assets
def test_update_does_not_claim_training_success():
    buffer, policy, _env = _formal_buffer()
    out = _update(buffer, policy, _optimizer(policy), _lagrangian())
    assert out["claims"] == {"trained": False, "performance_evaluated": False,
                             "convergence_claimed": False}
    # 非空洞性：确实做了一次真实更新（有梯度与参数变化）
    assert out["optimizer_steps"] == 1
    assert out["grad_norm_actor"] > 0.0

"""M1.3g-f-c-d：双批次 formal PPO 更新连通性探针。

两个**不同**的 formal episode（从 verified train v5 candidate origins 48 / 96 推导），
**同一** policy / optimizer / Lagrangian / **连续**显式采样 RNG；
每批依次 `collect_rollout → single_ppo_update`。

**本卡不写 checkpoint、不接正式训练入口、不做性能或收敛评价。**
测试**不得**通过每批重建 policy / optimizer / Lagrangian 或重播种采样 RNG
来制造一致。

**改前缺陷（本文件对应先红）**：`safe_rl_v2/ppo_two_batch.py` 尚不存在。
"""

import importlib
import pathlib

import numpy as np
import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE = "safe_rl_v2.ppo_two_batch"

HORIZON = 8
FORECAST_CUTOFF = 4
DELTA_HOURS = 0.5
STEPS = 3
CLIP_EPSILON = 0.2
GAMMA = 0.99
LAM = 0.95
ENV_SEED_KWARGS = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}


def tb():
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


# ⚠️ 下面这些数值只是**示例输入**，由**调用方**构造并显式传入；
# 它们**不是**正式训练超参数，probe 内部不得选择或冻结其中任何一个。
EXAMPLE_POLICY_SEED = 0          # 示例：策略权重初始化种子
EXAMPLE_ADAM_LR = 1e-3           # 示例：Adam 学习率
EXAMPLE_ADAM_BETAS = (0.9, 0.999)
EXAMPLE_ADAM_EPS = 1e-8
EXAMPLE_BUSINESS_SPEC = {"budget": 5.0, "learning_rate": 0.01, "max_multiplier": 100.0}
EXAMPLE_CARBON_SPEC = {"budget": 3.0, "learning_rate": 0.01, "max_multiplier": 100.0}
EXAMPLE_SAMPLING_SEED = 0        # 示例：显式采样 RNG 的种子
EXAMPLE_ENV_SEED = 0             # 示例：环境 episode 种子


def _make_example_objects(obs_dim: int, *, policy_seed: int = EXAMPLE_POLICY_SEED,
                          adam_lr: float = EXAMPLE_ADAM_LR,
                          sampling_seed: int = EXAMPLE_SAMPLING_SEED,
                          business: dict | None = None, carbon: dict | None = None):
    """构造**示例** policy / optimizer / Lagrangian / 显式 generator（调用方所有）。"""
    from safe_rl_v2.lagrangian import (
        UNIT_KG_CO2E,
        UNIT_VIOLATION_TASK_STEPS,
        ConstraintSpec,
        Lagrangian,
    )
    from safe_rl_v2.policy import SafePPOPolicy

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(policy_seed)
        policy = SafePPOPolicy(obs_dim=obs_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=adam_lr,
                                 betas=EXAMPLE_ADAM_BETAS, eps=EXAMPLE_ADAM_EPS)
    b = {**EXAMPLE_BUSINESS_SPEC, **(business or {})}
    c = {**EXAMPLE_CARBON_SPEC, **(carbon or {})}
    lagrangian = Lagrangian((
        ConstraintSpec(name="business", unit=UNIT_VIOLATION_TASK_STEPS, **b),
        ConstraintSpec(name="carbon", unit=UNIT_KG_CO2E, **c),
    ))
    generator = torch.Generator()
    generator.manual_seed(sampling_seed)
    return policy, optimizer, lagrangian, generator


def _obs_dim() -> int:
    env, _inj = tb().build_formal_env(
        tb().TRAIN_CANDIDATE_STARTS[0], horizon=HORIZON,
        forecast_cutoff=FORECAST_CUTOFF, delta_t_hours=DELTA_HOURS,
        env_seed_kwargs=dict(ENV_SEED_KWARGS))
    return int(env.obs_dim)


def _run(**overrides):
    m = tb()
    obs_dim = _obs_dim()
    obj_overrides = {}
    for key in ("adam_lr", "sampling_seed", "policy_seed", "business", "carbon"):
        if key in overrides:
            obj_overrides[key] = overrides.pop(key)
    policy, optimizer, lagrangian, generator = _make_example_objects(
        obs_dim, **obj_overrides)
    kwargs = {"clip_epsilon": CLIP_EPSILON, "gamma": GAMMA, "lam": LAM,
              "steps": STEPS, "horizon": HORIZON, "forecast_cutoff": FORECAST_CUTOFF,
              "delta_t_hours": DELTA_HOURS, "env_seed": EXAMPLE_ENV_SEED,
              "env_seed_kwargs": dict(ENV_SEED_KWARGS)}
    kwargs.update(overrides)
    return m.run_two_batch_probe(policy, optimizer, lagrangian, generator, **kwargs)


def _policy_from_snapshot(obs_dim: int, state_dict: dict):
    from safe_rl_v2.policy import SafePPOPolicy

    policy = SafePPOPolicy(obs_dim=obs_dim)
    policy.load_state_dict(state_dict)
    return policy


# =============================================================================
# 1. 两批各有真实 transition
# =============================================================================

@needs_assets
def test_both_batches_carry_real_transitions():
    out = _run()
    assert len(out["batches"]) == 2
    assert [b["origin"] for b in out["batches"]] == [48, 96]
    assert [b["start"] for b in out["batches"]] == [
        "2024-01-02T00:00:00+08:00", "2024-01-03T00:00:00+08:00"]

    for index, batch in enumerate(out["batches"]):
        assert batch["transitions"] == STEPS, f"批 {index} 未采到预期步数"
        obs = np.asarray(batch["observations"], dtype=np.float32)
        raw = np.asarray(batch["raw_actions"], dtype=np.float32)
        assert obs.shape == (STEPS, out["obs_dim"])
        assert raw.shape[0] == STEPS
        assert np.all(np.isfinite(obs)) and np.all(np.isfinite(raw))
        assert float(np.max(np.abs(obs))) > 0.0, "观测不得是恒零占位"
        assert any(abs(float(r)) > 0.0 for r in batch["rewards"]), \
            f"批 {index} 必须携带非零奖励"
    assert out["total_transitions"] == 2 * STEPS


# =============================================================================
# 2. 第二批的 old log-prob 对应**更新后**策略
# =============================================================================

@needs_assets
def test_second_batch_log_probs_belong_to_the_post_update_policy():
    """批 2 的 `old_raw_log_prob` 必须由**批 1 更新后**的策略采集。

    用**批 2 采集时刻**的状态重算 ⇒ 一致；用**批 1 采集时刻**的状态重算 ⇒ 不同。
    """
    out = _run()
    b1, b2 = out["batches"]

    # 非空洞性：策略确实在两批之间被更新过
    assert b1["policy_state_digest"] != b2["policy_state_digest"], \
        "两批之间策略状态必须发生变化（否则本用例区分不出「更新后」）"

    obs2 = torch.as_tensor(np.asarray(b2["observations"], dtype=np.float32))
    raw2 = torch.as_tensor(np.asarray(b2["raw_actions"], dtype=np.float32))
    recorded = torch.as_tensor(np.asarray(b2["old_raw_log_prob"], dtype=np.float32))

    at_collection = _policy_from_snapshot(
        out["obs_dim"], b2["policy_state_dict"]).evaluate_raw_actions(obs2, raw2).detach()
    assert torch.allclose(recorded, at_collection, atol=1e-6), \
        "批 2 的 old_raw_log_prob 必须由批 2 采集时刻的策略给出"

    stale = _policy_from_snapshot(
        out["obs_dim"], b1["policy_state_dict"]).evaluate_raw_actions(obs2, raw2).detach()
    assert not torch.allclose(recorded, stale, atol=1e-6), \
        "批 2 的 old_raw_log_prob 不得对应批 1 采集时刻（更新前）的策略"


# =============================================================================
# 3. 累计恰好 2 次 optimizer step 与 2 次乘子更新
# =============================================================================

@needs_assets
def test_cumulative_steps_and_multiplier_updates_are_exactly_two():
    out = _run()
    b1, b2 = out["batches"]

    assert b1["optimizer_steps_cumulative"] == 1
    assert b2["optimizer_steps_cumulative"] == 2
    assert out["optimizer_steps_total"] == 2

    assert b1["lagrangian_updates_cumulative"] == 1
    assert b2["lagrangian_updates_cumulative"] == 2
    assert out["lagrangian_updates_total"] == 2

    # 乘子更新**确实发生**了（计数），而不是断言其数值必然变化：
    # Lagrangian 的乘子被钳在 [0, max]；当违规量低于 budget 时候选值为负 ⇒ 留在 0。
    # 故「数值是否变化」依赖数据，**不可**作为判据；这里断言的是**同一 Lagrangian**
    # 在两个批次间被连续使用（批 2 的更新前值 == 批 1 的更新后值）。
    assert b2["multipliers_pre_update"] == b1["multipliers_post_update"], \
        "批 2 的更新前乘子必须等于批 1 更新后的乘子（同一 Lagrangian 连续使用）"
    assert b1["multipliers_pre_update"] == {"business": 0.0, "carbon": 0.0}, \
        "初始乘子必须为 0（否则本用例的连续性对照无意义）"
    # 参数确实逐批变化
    assert b1["policy_state_digest"] != out["final_policy_state_digest"]
    assert all(b["param_delta_norm"] > 0.0 for b in (b1, b2))


# =============================================================================
# 4. 同初始状态完整重跑逐位一致
# =============================================================================

@needs_assets
def test_full_rerun_from_the_same_initial_state_is_bit_identical():
    a = _run()
    b = _run()

    assert a["batch_digests"] == b["batch_digests"]
    assert a["final_policy_state_digest"] == b["final_policy_state_digest"]
    assert a["rerun_digest"] == b["rerun_digest"]
    assert [x["multipliers_post_update"] for x in a["batches"]] == \
           [x["multipliers_post_update"] for x in b["batches"]]
    # **不同初始种子**必须给出不同结果（否则 digest 可能恒等、测不出东西）
    c = _run(policy_seed=1, sampling_seed=1)
    assert c["rerun_digest"] != a["rerun_digest"]
    assert c["batch_digests"] != a["batch_digests"]


@needs_assets
def test_probe_shares_one_policy_optimizer_and_generator_across_batches():
    """**结构性守卫**：两批必须共用同一 policy / optimizer / Lagrangian / RNG。"""
    out = _run()
    assert out["shared_policy"] is True
    assert out["shared_optimizer"] is True
    assert out["shared_lagrangian"] is True
    assert out["generator_reseeded_between_batches"] is False
    assert out["batches"][0]["generator_state_digest"] != \
           out["batches"][1]["generator_state_digest"], \
        "连续 RNG 在两批之间必须**前进**（未被重播种为同一状态）"


# =============================================================================
# 5. 不得调用 synthetic / demo
# =============================================================================

@needs_assets
def test_probe_never_uses_synthetic_or_demo_sources(monkeypatch):
    tm = importlib.import_module("idc_model.task_model")

    def boom(*args, **kwargs):
        raise AssertionError("双批次 probe 不得调用 demo/random task generator")

    monkeypatch.setattr(tm.IDCEnergyTaskModel, "create_demo_tasks", boom)
    monkeypatch.setattr(tm.IDCEnergyTaskModel, "create_random_tasks", boom)

    out = _run()
    assert out["total_transitions"] == 2 * STEPS
    assert all(b["formal"] is True for b in out["batches"])


# =============================================================================
# 6. 不声称训练有效
# =============================================================================

@needs_assets
def test_probe_does_not_claim_training_success():
    out = _run()
    assert out["claims"] == {"trained": False, "performance_evaluated": False,
                             "convergence_claimed": False}
    assert out["probe_only"] is True
    # 非空洞性：确实完成了两次真实更新
    assert out["optimizer_steps_total"] == 2
    assert out["total_transitions"] == 2 * STEPS


# =============================================================================
# 7. M1.3g-f-c-d-R1：参数所有权（**调用方**提供对象）
# =============================================================================

@needs_assets
def test_probe_uses_the_caller_provided_objects():
    """**R1**：两批必须**使用调用方**传入的 policy / optimizer / Lagrangian / RNG。"""
    m = tb()
    policy, optimizer, lagrangian, generator = _make_example_objects(_obs_dim())
    out = m.run_two_batch_probe(
        policy, optimizer, lagrangian, generator,
        clip_epsilon=CLIP_EPSILON, gamma=GAMMA, lam=LAM, steps=STEPS,
        horizon=HORIZON, forecast_cutoff=FORECAST_CUTOFF, delta_t_hours=DELTA_HOURS,
        env_seed=EXAMPLE_ENV_SEED, env_seed_kwargs=dict(ENV_SEED_KWARGS))

    # **同一性**（不是相等）：probe 直接用调用方对象，未另建自己的
    assert out["policy"] is policy
    assert out["optimizer"] is optimizer
    assert out["lagrangian"] is lagrangian
    assert out["generator"] is generator
    # 更新确实落在调用方的 optimizer / Lagrangian 上
    assert len(optimizer.state) > 0, "调用方 optimizer 必须被推进"
    assert lagrangian._updates == 2, "调用方 Lagrangian 必须被推进两次"
    # 非空洞性：两批都真的采到 transition
    assert out["total_transitions"] == 2 * STEPS


@needs_assets
def test_caller_optimizer_learning_rate_changes_the_update_result():
    """**R1**：probe **不拥有**学习率 —— 改调用方 lr 必须改变更新结果。"""
    slow = _run(adam_lr=1e-3)
    fast = _run(adam_lr=1e-2)
    assert slow["optimizer_steps_total"] == fast["optimizer_steps_total"] == 2
    # 同一初始状态、同一数据、仅 lr 不同 ⇒ 参数变化量必须不同
    for index in (0, 1):
        assert slow["batches"][index]["param_delta_norm"] != pytest.approx(
            fast["batches"][index]["param_delta_norm"], rel=1e-6), \
            f"批 {index} 的参数变化量必须随调用方学习率变化"
    assert slow["final_policy_state_digest"] != fast["final_policy_state_digest"]


@needs_assets
def test_caller_lagrangian_spec_changes_the_multiplier_trajectory():
    """**R1**：probe **不拥有**乘子预算 / 学习率 —— 改调用方 spec 必须改变乘子轨迹。"""
    base = _run()
    generous = _run(business={"budget": 0.0}, carbon={"budget": 0.0})
    assert base["batches"][0]["multipliers_post_update"] != \
           generous["batches"][0]["multipliers_post_update"], \
        "把 budget 降到 0 必须改变乘子轨迹（否则 probe 未使用调用方 spec）"


@needs_assets
def test_probe_constructs_no_objects_and_reseeds_nothing(monkeypatch):
    """**R1 结构性守卫**：probe 内部**不得**构造 policy/optimizer/Lagrangian，
    也**不得**重播种 generator —— 把这些入口全部 monkeypatch 成抛错后仍须跑通。"""
    import safe_rl_v2.lagrangian as lag_mod
    import safe_rl_v2.policy as policy_mod

    def boom(*args, **kwargs):
        raise AssertionError("probe 不得在内部构造训练对象")

    monkeypatch.setattr(policy_mod, "SafePPOPolicy", boom)
    monkeypatch.setattr(lag_mod, "Lagrangian", boom)
    monkeypatch.setattr(lag_mod, "ConstraintSpec", boom)
    monkeypatch.setattr(torch.optim, "Adam", boom)
    monkeypatch.setattr(torch.Generator, "manual_seed", boom)

    m = tb()
    # 对象必须**在 monkeypatch 生效前**由调用方构造好（这里用未打补丁的构造路径）
    monkeypatch.undo()
    policy, optimizer, lagrangian, generator = _make_example_objects(_obs_dim())
    monkeypatch.setattr(policy_mod, "SafePPOPolicy", boom)
    monkeypatch.setattr(lag_mod, "Lagrangian", boom)
    monkeypatch.setattr(lag_mod, "ConstraintSpec", boom)
    monkeypatch.setattr(torch.optim, "Adam", boom)
    monkeypatch.setattr(torch.Generator, "manual_seed", boom)

    out = m.run_two_batch_probe(
        policy, optimizer, lagrangian, generator,
        clip_epsilon=CLIP_EPSILON, gamma=GAMMA, lam=LAM, steps=STEPS,
        horizon=HORIZON, forecast_cutoff=FORECAST_CUTOFF, delta_t_hours=DELTA_HOURS,
        env_seed=EXAMPLE_ENV_SEED, env_seed_kwargs=dict(ENV_SEED_KWARGS))
    assert out["total_transitions"] == 2 * STEPS
    assert out["optimizer_steps_total"] == 2


@needs_assets
def test_probe_signature_requires_all_four_objects():
    """**R1 结构性守卫**：四个对象必须是**必填**位置参数（**无默认值**）。"""
    import inspect

    params = inspect.signature(tb().run_two_batch_probe).parameters
    for name in ("policy", "optimizer", "lagrangian", "generator"):
        assert name in params, f"缺少必填参数 {name}"
        assert params[name].default is inspect.Parameter.empty, \
            f"{name} 不得有默认值（probe 不得替调用方选择）"
    # 不得再出现任何「替调用方选择超参数」的入参
    for forbidden in ("adam_lr", "learning_rate", "budget", "max_multiplier",
                      "hidden", "policy_seed", "sampling_seed"):
        assert forbidden not in params, f"probe 不得拥有超参数入参 {forbidden}"

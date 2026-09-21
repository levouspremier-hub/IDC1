"""M1.3g-f-c-a：formal 真实 rollout 闭环验证。

用**已验证的正式链**跑一次短的真实数据 rollout，并证明闭环可信：

```text
verified train v5 candidate origin → build_verified_formal_env_injection
  → IDCPriceEnv20D(formal_injection=…) → collect_rollout → RolloutBuffer
```

**本卡为只读验证卡**：不修改 `train.py`、发布产物 v1、冻结资产或 PPO 算法。
若本文件暴露必须修改 `envs/`、`scenario/env_injection.py` 或 `train.py` 的缺陷，
应**先提交失败证据并停止**，报告修复范围与发布产物升级范围，**不**偷偷重物化 v1。
"""

import importlib
import json
import pathlib

import numpy as np
import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

SPLIT = "train"
# 已验证的 train v5 candidate origin（本地行 48）。
EPISODE_START = "2024-01-02T00:00:00+08:00"
HORIZON = 8
FORECAST_CUTOFF = 4
DELTA_HOURS = 0.5
STEPS = 6

ENV_SEED_KWARGS = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}


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


def _injection(horizon: int = HORIZON):
    return importlib.import_module("scenario.env_injection") \
        .build_verified_formal_env_injection(
            SPLIT, start=EPISODE_START, horizon=horizon, forecast_cutoff=FORECAST_CUTOFF)


def _env(injection=None, horizon: int = HORIZON):
    inj = injection if injection is not None else _injection(horizon)
    env_cls = importlib.import_module("envs.idc_price_env").IDCPriceEnv20D
    return env_cls(horizon=horizon, delta_t_hours=DELTA_HOURS,
                   formal_injection=inj, **ENV_SEED_KWARGS), inj


def _policy(env, seed: int = 0):
    from safe_rl_v2.policy import SafePPOPolicy

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return SafePPOPolicy(obs_dim=env.obs_dim)


def _buffer_digest(buffer) -> str:
    """buffer 全部 transition 的**逐位**指纹（供重放比对）。"""
    import hashlib

    h = hashlib.sha256()
    for t in buffer.transitions:
        for arr in (t.observation, t.next_observation, t.raw_action, t.exec_action):
            h.update(np.asarray(arr, dtype=np.float32).tobytes())
        for scalar in (t.old_raw_log_prob, t.reward, t.business_cost,
                       t.carbon_cost, t.electricity_cost_sgd):
            h.update(repr(float(scalar)).encode())
        h.update(b"1" if t.terminated else b"0")
        h.update(b"1" if t.truncated else b"0")
    return h.hexdigest()


def _collect(env, policy, steps: int = STEPS, seed: int = 0):
    from safe_rl_v2.buffer import RolloutBuffer
    from safe_rl_v2.rollout import collect_rollout

    generator = torch.Generator()
    generator.manual_seed(seed)
    buffer = RolloutBuffer()
    stats = collect_rollout(env, policy, buffer, steps=steps, seed=seed,
                            generator=generator)
    return buffer, stats


# =============================================================================
# 0. 正式注入实际进入 env（回归守卫：改前已绿）
# =============================================================================

@needs_assets
def test_formal_injection_actually_enters_the_env():
    env, inj = _env()
    assert env.formal is True
    assert env.formal_injection is inj
    env.reset(seed=0)
    # realized exogenous 逐位等于 verified injection
    for k in range(inj.horizon):
        assert float(env.price_t[k]) == inj.price_sgd_per_kwh[k]
        assert float(env.pv_t[k]) == inj.local_pv_kw[k]
        assert float(env.wt_t[k]) == inj.wind_generation_kw[k]
        assert float(env.carbon_factor_t[k]) == inj.carbon_kg_per_kwh[k]
    # causal forecast 亦逐位进入
    assert np.array_equal(env.price_forecast_t,
                          np.asarray(inj.causal_forecasts["price_forecast"]))
    # 非空洞性：该 episode 必须有非零任务与工作量
    assert inj.ledger_micro and sum(inj.ledger_micro) > 0
    assert env.initial_Q > 0.0


# =============================================================================
# 1. buffer 来自**真实 transition**
# =============================================================================

@needs_assets
def test_collected_buffer_holds_real_transitions():
    env, inj = _env()
    policy = _policy(env)
    buffer, stats = _collect(env, policy)

    assert stats["transitions"] == STEPS == len(buffer)
    assert stats["steps_requested"] == STEPS
    assert stats["env_seed"] == 0
    assert stats["policy_rng_source"] == "explicit_generator"
    assert stats["contract_version"]
    # 非空洞性：必须真的采集到 transition
    assert len(buffer.transitions) > 0

    from safe_rl_v2.buffer import ACTION_DIM

    for t in buffer.transitions:
        assert np.asarray(t.observation).shape == (env.obs_dim,)
        assert np.asarray(t.next_observation).shape == (env.obs_dim,)
        assert np.asarray(t.raw_action).shape == (ACTION_DIM,)
        assert np.asarray(t.exec_action).shape == (ACTION_DIM,)
        # 无修正器时 exec 必须与 raw 逐元素相同（不得被 clip 掩盖）
        assert np.array_equal(t.raw_action, t.exec_action)
        assert np.all(np.isfinite(t.observation))
        assert np.all(np.isfinite(t.raw_action))
        # 三类约束信号与 reward 都是**有限实数**（不是占位 0）
        for name in ("reward", "business_cost", "carbon_cost", "electricity_cost_sgd"):
            assert np.isfinite(float(getattr(t, name))), name
        # observation 不得是恒零占位
        assert float(np.max(np.abs(t.observation))) > 0.0

    # 至少有一条 transition 携带非零奖励（证明闭环真的在跑）
    assert any(abs(float(t.reward)) > 0.0 for t in buffer.transitions)
    # 碳排放为正（formal 链注入了 real carbon factor）
    assert any(float(t.carbon_cost) > 0.0 for t in buffer.transitions)


@needs_assets
def test_raw_log_prob_matches_a_real_recomputation():
    """`old_raw_log_prob` 必须是**该 observation/raw_action** 下的真实对数概率。"""
    env, _inj = _env()
    policy = _policy(env)
    buffer, _stats = _collect(env, policy)

    for t in buffer.transitions:
        obs_t = torch.as_tensor(np.asarray(t.observation, dtype=np.float32))
        raw_t = torch.as_tensor(np.asarray(t.raw_action, dtype=np.float32))
        recomputed = float(policy.evaluate_raw_actions(obs_t, raw_t).detach())
        assert recomputed == pytest.approx(float(t.old_raw_log_prob), abs=1e-6)
    # 非空洞性：对数概率不得全部相同（否则可能是占位常量）
    probs = [float(t.old_raw_log_prob) for t in buffer.transitions]
    assert len(set(probs)) > 1


@needs_assets
def test_reward_and_constraint_signals_come_from_the_env_step():
    """对照：buffer 的当步 reward/约束信号必须等于 env.step 的返回值。"""
    env, _inj = _env()
    policy = _policy(env)
    from safe_rl_v2.rollout import (
        BUSINESS_VIOLATION_INFO_KEY,
        CARBON_EMISSION_INFO_KEY,
        ELECTRICITY_COST_INFO_KEY,
    )

    # 在**另一个**同配置 env 上手工重放同一条轨迹（collect_rollout 内部已 reset）
    probe_env, _j = _env()
    probe_policy = _policy(probe_env)
    buffer, _stats = _collect(env, policy)

    obs, _ = probe_env.reset(seed=0)
    generator = torch.Generator()
    generator.manual_seed(0)

    replayed = 0
    for t in buffer.transitions:
        assert np.allclose(np.asarray(t.observation, dtype=np.float32),
                           np.asarray(obs, dtype=np.float32))
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        raw_t, _lp, _ = probe_policy.act(obs_t, generator=generator)
        raw_action = raw_t.detach().numpy().astype(np.float32)
        next_obs, reward, _term, _trunc, info = probe_env.step(raw_action)
        assert float(t.reward) == pytest.approx(float(reward), abs=1e-9)
        assert float(t.business_cost) == pytest.approx(
            float(info[BUSINESS_VIOLATION_INFO_KEY]), abs=1e-9)
        assert float(t.carbon_cost) == pytest.approx(
            float(info[CARBON_EMISSION_INFO_KEY]), abs=1e-9)
        assert float(t.electricity_cost_sgd) == pytest.approx(
            float(info[ELECTRICITY_COST_INFO_KEY]), abs=1e-9)
        obs = next_obs
        replayed += 1
    # 非空洞性：必须真的逐步重放了全部 transition
    assert replayed == STEPS == len(buffer.transitions)


# =============================================================================
# 2. 可重放
# =============================================================================

@needs_assets
def test_same_seed_and_action_path_replays_bit_for_bit():
    env_a, _i = _env()
    env_b, _j = _env()
    buf_a, stats_a = _collect(env_a, _policy(env_a), seed=0)
    buf_b, stats_b = _collect(env_b, _policy(env_b), seed=0)

    assert stats_a == stats_b
    assert _buffer_digest(buf_a) == _buffer_digest(buf_b)
    # 非空洞性：digest 必须是真的 64 位十六进制，且不同 seed 会给出不同 digest
    assert len(_buffer_digest(buf_a)) == 64
    env_c, _k = _env()
    buf_c, _s = _collect(env_c, _policy(env_c, seed=1), seed=1)
    assert _buffer_digest(buf_c) != _buffer_digest(buf_a)


# =============================================================================
# 3. 不调用 synthetic / demo 来源
# =============================================================================

@needs_assets
def test_formal_rollout_never_uses_synthetic_or_demo_sources(monkeypatch):
    tm = importlib.import_module("idc_model.task_model")

    def boom(*args, **kwargs):
        raise AssertionError("formal rollout 不得调用 demo/random task generator")

    monkeypatch.setattr(tm.IDCEnergyTaskModel, "create_demo_tasks", boom)
    monkeypatch.setattr(tm.IDCEnergyTaskModel, "create_random_tasks", boom)

    env, _inj = _env()
    buffer, stats = _collect(env, _policy(env))
    assert stats["transitions"] == STEPS
    # 非空洞性：确实采到了真实 transition
    assert len(buffer.transitions) == STEPS


# =============================================================================
# 4. 未来真值仍不可见（保留既有约束）
# =============================================================================

@needs_assets
@pytest.mark.leakage
def test_future_realized_exogenous_stays_invisible_to_the_observation():
    """改**未来** realized exogenous → 当前 observation **不变**。"""
    env, _inj = _env()
    env.reset(seed=0)
    base = env._get_forecast_features().copy()

    t = int(env.current_step)
    lo = t + env.forecast_cutoff
    assert lo < env.horizon, "窗口外区间必须非空，否则本用例空洞"
    victim = env.price_t[lo:].copy()
    env.price_t[lo:] += 1000.0
    assert not np.array_equal(victim, env.price_t[lo:]), "mutation 未生效"

    assert np.array_equal(base, env._get_forecast_features()), \
        "未来 realized exogenous 不得影响当前 observation"


@needs_assets
@pytest.mark.leakage
def test_visible_causal_forecast_still_changes_the_observation():
    """**反向控制**：改可见窗口内 causal forecast → observation **必变**。"""
    env, _inj = _env()
    env.reset(seed=0)
    base = env._get_forecast_features().copy()

    assert env.forecast_cutoff > 0
    env.price_forecast_t[:env.forecast_cutoff] += 5.0
    assert not np.array_equal(base, env._get_forecast_features()), \
        "可见窗口内 forecast 变了，observation 却不变 —— 断言过强"


@needs_assets
def test_formal_step_info_does_not_expose_full_future_truth():
    env, _inj = _env()
    _obs, info = env.reset(seed=0)
    assert "true_task_arrival_profile" not in info
    assert "task_arrival_forecast" not in info
    # info 的任务数不得含未来任务（g-e-d-R1 的约束）
    arrived = sum(1 for x in env.tasks if int(x.arrival_time) <= int(env.current_step))
    assert info["total_task_count"] == arrived
    assert len(env.tasks) > arrived


# =============================================================================
# 5. probe 与正式链一致（机器可读证据）
# =============================================================================

@needs_assets
def test_probe_reports_the_same_digest_as_the_test_harness():
    probe = importlib.import_module("scripts.probe_formal_rollout")
    evidence = probe.run_probe(steps=STEPS)
    env, _i = _env()
    buffer, _s = _collect(env, _policy(env))
    assert evidence["buffer_digest"] == _buffer_digest(buffer)
    assert evidence["steps_collected"] == STEPS
    assert evidence["formal"] is True
    # 证据必须可 JSON 序列化（供验收记录引用）
    json.dumps(evidence)


# =============================================================================
# 6. 补充验证路径（**回归守卫**，改前已绿 —— 不冒充先红）
# =============================================================================

@needs_assets
def test_corrector_on_runs_the_real_milp_over_formal_data():
    """`corrector_on=True` 时 exec 与 raw **真实分离**（修正器真的在正式数据上求解）。"""
    from planning.corrector import PRODUCTION_CORRECTOR_TIME_LIMIT_S
    from safe_rl_v2.buffer import RolloutBuffer
    from safe_rl_v2.rollout import collect_rollout

    env, _inj = _env()
    policy = _policy(env)
    generator = torch.Generator()
    generator.manual_seed(0)
    buffer = RolloutBuffer()
    stats = collect_rollout(
        env, policy, buffer, steps=4, seed=0, corrector_on=True,
        corrector_time_limit_s=PRODUCTION_CORRECTOR_TIME_LIMIT_S, generator=generator,
    )
    assert stats["transitions"] == 4
    assert stats["corrector_on"] is True
    # 非空洞性：修正器必须在**正式**数据上真的改变了动作（否则该路径未被触及）
    assert stats["raw_exec_difference_count"] > 0, (
        "corrector_on 下 exec 必须与 raw 分离；差异为 0 说明修正器没真正生效")
    for t in buffer.transitions:
        assert np.asarray(t.exec_action).shape == (21,)


@needs_assets
def test_rollout_stops_early_on_termination_without_losing_transitions():
    """请求步数超过 episode 长度时，buffer 应**提前停止**且不丢 transition。"""
    from safe_rl_v2.buffer import RolloutBuffer
    from safe_rl_v2.rollout import collect_rollout

    env, inj = _env()
    policy = _policy(env)
    generator = torch.Generator()
    generator.manual_seed(0)
    buffer = RolloutBuffer()
    stats = collect_rollout(env, policy, buffer, steps=inj.horizon + 12, seed=0,
                            generator=generator)
    assert stats["terminated_count"] == 1, "长 rollout 必须真实终止一次"
    assert stats["transitions"] == len(buffer) == inj.horizon
    assert stats["transitions"] < stats["steps_requested"], "必须提前停止"

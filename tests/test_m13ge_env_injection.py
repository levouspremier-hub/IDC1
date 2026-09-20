"""M1.3g-e-c：B6 mapper → 正式环境注入的**先红**回归。

改前缺陷（本文件在实现前必须为红）：

- `scenario.env_injection` 模块尚不存在（`ModuleNotFoundError`）；
- 环境**没有** formal injection 入口，`reset()` 恒走
  `model.create_demo_tasks`（demo/random 生成器）；
- `planned_capacity_vec` **未**按 `delta_t_hours` 折算成 work/step；
- observation 的 arrival 通道读的是**当步 realized truth**；
- `reset` 的 `info` 暴露**整段** arrival truth。

**本卡只做 env 接线**：不接训练入口、不改 readiness。
"""

import importlib
import pathlib

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
INJECTION_MODULE = "scenario.env_injection"

WORK_UNIT_SCALE = 1_000_000
B6_RATE_WORK_PER_HOUR = 63.988
B6_RATE_WORK_PER_STEP = 31.994  # work / 半小时
DELTA_HOURS = 0.5

CANONICAL_PARQUET = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
CANONICAL_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"
SPLIT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_splits.json"
POLICY_V3 = REPO_ROOT / "data/manifest/singapore_2024_forecast_policy_v3.json"
B6_POLICY = REPO_ROOT / "data/manifest/m13f_arrival_intensity_policy_v1.json"
REFS_V4 = REPO_ROOT / "configs/frozen_refs/refs_v4.json"
V5_DIR = REPO_ROOT / "data/manifest/formal_splits_v5"

STARTS = {
    "train": "2024-01-02T00:00:00+08:00",
    "validation": "2024-08-01T00:00:00+08:00",
    "test": "2024-10-01T00:00:00+08:00",
}


def injection_module():
    return importlib.import_module(INJECTION_MODULE)


def _upstream_present() -> bool:
    return all(p.exists() for p in (
        CANONICAL_PARQUET, CANONICAL_MANIFEST, SPLIT_MANIFEST, POLICY_V3,
        B6_POLICY, REFS_V4, V5_DIR / "train.json",
    ))


needs_assets = pytest.mark.skipif(
    not _upstream_present(), reason="真实冻结上游资产不在本机")


def _build(split: str = "train", horizon: int = 8, cutoff: int = 4):
    return injection_module().build_verified_formal_env_injection(
        split, start=STARTS[split], horizon=horizon, forecast_cutoff=cutoff)


def _env(split: str = "train", horizon: int = 8, cutoff: int = 4, seed: int = 7):
    from envs.idc_price_env import IDCPriceEnv20D

    inj = _build(split, horizon, cutoff)
    # formal 链**必须**显式声明半小时步长（环境会校验它与注入一致）
    return IDCPriceEnv20D(horizon=horizon, task_seed=seed, server_seed=0,
                          forecast_seed=seed, delta_t_hours=0.5,
                          formal_injection=inj), inj


# --- 1. formal 入口存在且可接收 verified injection ------------------------------

@needs_assets
def test_formal_injection_entry_exists_and_verifies_the_chain():
    inj = _build()
    assert inj.split == "train"
    assert inj.delta_t_hours == DELTA_HOURS
    assert inj.horizon == 8
    assert inj.ledger_micro and len(inj.ledger_micro) == 8
    assert len(inj.provenance_hash) == 64
    assert inj.lambda_ref_work_per_step == B6_RATE_WORK_PER_STEP


@needs_assets
def test_formal_injection_rejects_bad_inputs():
    m = injection_module()
    for split, start in (("train", "2024-01-01"),
                         ("train", "2024-01-01T00:00:00+08:00"),
                         ("nope", STARTS["train"])):
        with pytest.raises((ValueError, m.FormalInjectionError)):
            m.build_verified_formal_env_injection(
                split, start=start, horizon=4, forecast_cutoff=4)


# --- 2. formal reset 用 mapper 任务，不用 demo/random ---------------------------

@needs_assets
@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_formal_reset_uses_mapper_tasks(monkeypatch, split):
    import idc_model.task_model as tm

    def boom(*a, **k):
        raise AssertionError("formal reset 不得调用 demo/random task generator")

    monkeypatch.setattr(tm.IDCEnergyTaskModel, "create_demo_tasks", boom)
    monkeypatch.setattr(tm.IDCEnergyTaskModel, "create_random_tasks", boom)
    env, inj = _env(split)
    env.reset(seed=7)
    assert len(env.tasks) >= len(inj.tasks)


@needs_assets
def test_env_task_stream_matches_the_public_mapper_output():
    env, inj = _env()
    env.reset(seed=7)
    mapped = {t.task_id: t for t in inj.tasks}
    matched = [t for t in env.tasks if t.task_id in mapped]
    assert len(matched) == len(inj.tasks)
    for task in matched:
        ref = mapped[task.task_id]
        assert task.profile_key == ref.profile_key
        assert task.duration == ref.duration
        assert task.deadline == ref.deadline
        assert task.workload == pytest.approx(ref.workload)


@needs_assets
def test_env_ledger_conserves_every_slot_and_the_whole_episode():
    env, inj = _env()
    env.reset(seed=7)
    ledger = {t.task_id: t.workload for t in env.tasks}
    total = 0.0
    for slot, micro in zip(inj.slots, inj.ledger_micro, strict=True):
        ids = [t.task_id for t in slot.tasks]
        got = sum(ledger[i] for i in ids if i in ledger)
        assert got == pytest.approx(micro / WORK_UNIT_SCALE)
        total += micro
    assert total == sum(inj.ledger_micro)
    assert int(round(total)) % WORK_UNIT_SCALE == 0


# --- 3. 0.5 小时单位 ------------------------------------------------------------

@needs_assets
def test_formal_env_delta_is_half_an_hour():
    env, _ = _env()
    assert env.delta_t_hours == DELTA_HOURS


@needs_assets
def test_planned_capacity_is_converted_to_work_per_step():
    """`C_server` 是 work/hour rate；每步可执行量必须乘 `delta_t_hours`。"""
    env, _ = _env()
    env.reset(seed=7)
    C = float(np.asarray(env.model.C_server).sum())
    full_action = np.ones(env.action_dim, dtype=np.float64)
    env.step(full_action)
    expected_rate = env.max_task_load_per_server * C
    expected_per_step = expected_rate * DELTA_HOURS
    assert env.last_planned_capacity_rate_work_per_hour == pytest.approx(expected_rate)
    assert env.last_planned_capacity_per_step == pytest.approx(expected_per_step)
    # 若**漏乘** delta_t_hours，工程量会恰好差 2 倍
    assert env.last_planned_capacity_per_step != pytest.approx(expected_rate)


@needs_assets
def test_lambda_ref_normalises_to_work_per_step():
    env, _ = _env()
    env.reset(seed=7)
    # 冻结 rate 是 63.988 work/hour；按半小时归一化必须用 31.994 work/step
    assert env.lambda_ref == pytest.approx(B6_RATE_WORK_PER_STEP)
    assert env.lambda_ref != pytest.approx(B6_RATE_WORK_PER_HOUR)


@needs_assets
def test_queue_refs_are_stock_and_not_scaled_by_the_step_length():
    env, _ = _env()
    assert env.queue_ref == pytest.approx(6000.0)
    assert env.queue_capacity_ref == pytest.approx(6000.0)


# --- 4. observation 用 causal forecast，不读 truth ------------------------------

@needs_assets
def test_observation_lambda_channel_uses_the_causal_forecast():
    env, inj = _env()
    obs, _ = env.reset(seed=7)
    # 第 2 个分量是 lambda 通道（T, price, lambda, Q, sin, cos）
    assert obs[2] == pytest.approx(
        float(inj.arrival_forecast[0]) / env.lambda_ref, rel=1e-6)


@needs_assets
@pytest.mark.leakage
def test_observation_is_invariant_to_future_truth_mutation(monkeypatch):
    env, inj = _env(horizon=8)
    base_obs, _ = env.reset(seed=7)
    truth = np.asarray(env.true_task_arrival_profile, dtype=np.float64).copy()
    mutated = truth.copy()
    mutated[env.current_step + env.forecast_cutoff:] += 1000.0
    env.true_task_arrival_profile = mutated
    after_obs, _ = env.reset(seed=7)
    # 只有**当步**通道可能不同；未来段不得影响任何观测分量
    assert mutated[0] == truth[0]
    assert np.allclose(base_obs, after_obs)


@needs_assets
@pytest.mark.leakage
def test_observation_changes_when_the_visible_forecast_changes():
    """**防假绿**：改**可见窗口内**的 forecast，observation 必须变化。"""
    env, _ = _env(horizon=8)
    env.reset(seed=7)
    base_obs = env._get_obs()
    env.task_arrival_forecast = np.asarray(
        env.task_arrival_forecast, dtype=np.float64) + 5.0
    assert not np.allclose(base_obs, env._get_obs())


@needs_assets
@pytest.mark.leakage
def test_observation_is_invariant_to_changes_outside_the_visible_window():
    """反向控制：只改**可见窗口之外**的 forecast，observation **不得**变化。"""
    env, _ = _env(horizon=8, cutoff=4)
    env.reset(seed=7)
    base_obs = env._get_obs()
    forecast = np.asarray(env.task_arrival_forecast, dtype=np.float64).copy()
    forecast[4:] += 1000.0  # 窗口是 [0, 4)
    env.task_arrival_forecast = forecast
    assert np.allclose(base_obs, env._get_obs())


# --- 5. info 不含完整未来 truth ------------------------------------------------

@needs_assets
def test_formal_reset_info_does_not_expose_full_future_truth():
    env, _ = _env()
    _, info = env.reset(seed=7)
    assert "true_task_arrival_profile" not in info
    assert "task_arrival_forecast" not in info
    assert info["formal"] is True
    assert info["split"] == "train"
    assert len(info["provenance_hash"]) == 64


@needs_assets
def test_formal_step_info_does_not_expose_full_future_truth():
    env, _ = _env()
    env.reset(seed=7)
    _, _, _, _, info = env.step(np.full(env.action_dim, 0.5, dtype=np.float64))
    assert "true_task_arrival_profile" not in info
    assert "task_arrival_forecast" not in info


# --- 6. repeated reset 一致 -----------------------------------------------------

@needs_assets
def test_repeated_reset_gives_an_identical_task_stream_and_observation():
    env, _ = _env()
    obs_a, _ = env.reset(seed=7)
    ids_a = [t.task_id for t in env.tasks]
    obs_b, _ = env.reset(seed=7)
    ids_b = [t.task_id for t in env.tasks]
    assert ids_a == ids_b
    assert np.allclose(obs_a, obs_b)


# --- 7. fail closed + legacy 不受影响 -------------------------------------------

@needs_assets
def test_formal_injection_refuses_a_legacy_delta():
    from envs.idc_price_env import IDCPriceEnv20D

    inj = _build()
    with pytest.raises(ValueError):
        IDCPriceEnv20D(horizon=8, task_seed=7, server_seed=0, forecast_seed=7,
                       delta_t_hours=1.0, formal_injection=inj)


@needs_assets
def test_legacy_env_behaviour_is_unchanged():
    from envs.idc_price_env import IDCPriceEnv20D

    env = IDCPriceEnv20D(horizon=24, task_seed=0, server_seed=0, forecast_seed=300000)
    assert env.delta_t_hours == 1.0
    assert env.lambda_ref == pytest.approx(1000.0)
    obs, info = env.reset(seed=0)
    assert "true_task_arrival_profile" in info
    assert len(env.tasks) > 0

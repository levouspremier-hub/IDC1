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


def _env_cls():
    """**动态**导入环境类：mypy 不跟进，避免把 `idc_model.task_model` 的既有
    类型错误（不在 `make check` 扫描范围内）经 `envs/` 拉进门禁。"""
    return importlib.import_module("envs.idc_price_env").IDCPriceEnv20D


def _env(split: str = "train", horizon: int = 8, cutoff: int = 4, seed: int = 7):
    IDCPriceEnv20D = _env_cls()

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
    # 动态导入：mypy 不跟进 `idc_model.task_model`（该文件含既有类型错误且不在
    # `make check` 的扫描范围内），避免把它拉进门禁。
    tm = importlib.import_module("idc_model.task_model")

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
    IDCPriceEnv20D = _env_cls()

    inj = _build()
    with pytest.raises(ValueError):
        IDCPriceEnv20D(horizon=8, task_seed=7, server_seed=0, forecast_seed=7,
                       delta_t_hours=1.0, formal_injection=inj)


@needs_assets
def test_legacy_env_behaviour_is_unchanged():
    IDCPriceEnv20D = _env_cls()

    env = IDCPriceEnv20D(horizon=24, task_seed=0, server_seed=0, forecast_seed=300000)
    assert env.delta_t_hours == 1.0
    assert env.lambda_ref == pytest.approx(1000.0)
    obs, info = env.reset(seed=0)
    assert "true_task_arrival_profile" in info
    assert len(env.tasks) > 0


# =============================================================================
# M1.3g-e-c-R1：formal exogenous / refs 接线 + 真正的 reset 重放
# =============================================================================

# 一个**日照时段**的起点（PV / 风电非零），避免「夜间全零」假绿。
DAYLIGHT_START = "2024-04-16T10:00:00+08:00"

# refs_v4 中环境实际消费的正式参考值（人工批准 / train 冻结）
FORMAL_REFS = {
    "price_ref": 4.5,
    "lambda_ref": 31.994,          # = 63.988 work/hour × delta_t_hours(0.5)
    "queue_ref": 6000.0,
    "queue_capacity_ref": 6000.0,
    "pv_ref_kw": 350.9073696124661,
    "wind_ref_kw": 262.3178613166015,
    "carbon_factor_ref": 0.402,
    "cost_ref": 60.0,
    "carbon_ref": 15.0,
    "grid_power_limit_kW": 18.0,
    "peak_power_ref_kW": 10.0,
    "peak_power_threshold_kW": 18.0,
    "sla_penalty_ref": 50.0,
}


def _daylight_env(horizon: int = 8, cutoff: int = 4):
    from envs.idc_price_env import IDCPriceEnv20D

    inj = injection_module().build_verified_formal_env_injection(
        "train", start=DAYLIGHT_START, horizon=horizon, forecast_cutoff=cutoff)
    return IDCPriceEnv20D(horizon=horizon, task_seed=7, server_seed=0,
                          forecast_seed=7, delta_t_hours=0.5,
                          formal_injection=inj), inj


# --- R1-1. realized exogenous 逐位接线 ------------------------------------------

@needs_assets
def test_formal_realized_exogenous_is_wired_bit_for_bit():
    env, inj = _daylight_env()
    env.reset(seed=7)
    # 非空洞性：该窗口的 PV / 风电必须**非零**
    assert max(inj.local_pv_kw) > 1.0, "测试窗口必须含非零 PV"
    assert max(inj.wind_generation_kw) > 0.0, "测试窗口必须含非零风电"
    for k in range(inj.horizon):
        assert float(env.price_t[k]) == inj.price_sgd_per_kwh[k]
        assert float(env.T_amb[k]) == inj.temperature_deg_c[k]
        assert float(env.pv_t[k]) == inj.local_pv_kw[k]
        assert float(env.wt_t[k]) == inj.wind_generation_kw[k]
        assert float(env.carbon_factor_t[k]) == inj.carbon_kg_per_kwh[k]


@needs_assets
def test_formal_realized_exogenous_differs_from_the_legacy_defaults():
    """**防假绿**：接线后的值与 legacy 默认曲线**必须不同**。"""
    env, inj = _daylight_env()
    env.reset(seed=7)
    assert float(env.pv_t[0]) != 0.0          # legacy 默认全零
    assert float(env.carbon_factor_t[0]) != 0.70  # legacy 日曲线峰值段
    assert float(env.T_amb[0]) != pytest.approx(
        25 + 5 * np.sin(np.pi * (0 - 8) / 12), rel=1e-3)  # legacy 正弦


# --- R1-2. forecast observation 用 B6 causal -----------------------------------

@needs_assets
def test_formal_forecast_arrays_are_the_b6_causal_forecasts():
    env, inj = _daylight_env()
    env.reset(seed=7)
    for attr in ("price_forecast_t", "temperature_forecast_t", "pv_forecast_t",
                 "wind_forecast_t", "carbon_forecast_t"):
        assert hasattr(env, attr), attr
    # realized 与 forecast **不得**是同一套数组
    assert not np.allclose(env.pv_forecast_t, env.pv_t)
    assert not np.allclose(env.price_forecast_t, env.price_t)
    # 注意：carbon 当前 **不能** 这样断言 —— canonical carbon 与它的 causal forecast
    # 都是冻结常量 0.402，`np.allclose` 恒等，断言会**空过**（见 R2-1/R2-2）。


@needs_assets
@pytest.mark.leakage
def test_forecast_observation_uses_causal_forecasts_not_realized_truth():
    """把 realized 改成不同值 → forecast observation **不变**；反之必变。"""
    env, _ = _daylight_env()
    env.reset(seed=7)
    base = env._get_forecast_features().copy()
    env.price_t = np.asarray(env.price_t, dtype=np.float64) + 10.0
    env.pv_t = np.asarray(env.pv_t, dtype=np.float64) + 100.0
    assert np.allclose(base, env._get_forecast_features())
    env.price_forecast_t = np.asarray(env.price_forecast_t, dtype=np.float64) + 10.0
    assert not np.allclose(base, env._get_forecast_features())


# --- R1-3. reset 重放：全新 Task 对象 -------------------------------------------

@needs_assets
def test_reset_after_steps_replays_the_initial_task_state():
    env, inj = _daylight_env()
    env.reset(seed=7)
    before = [(t.task_id, t.status, t.remaining_work, t.start_time, t.finish_time)
              for t in env.tasks]
    ids_before = [t.task_id for t in env.tasks]

    for _ in range(3):
        env.step(np.full(env.action_dim, 0.9, dtype=np.float64))
    mid = [(t.task_id, t.status, t.remaining_work) for t in env.tasks]
    assert mid != before, "执行后任务状态必须发生变化（否则重放无意义）"

    obs_after, _ = env.reset(seed=7)
    after = [(t.task_id, t.status, t.remaining_work, t.start_time, t.finish_time)
             for t in env.tasks]
    assert after == before
    assert [t.task_id for t in env.tasks] == ids_before


@needs_assets
def test_reset_creates_fresh_task_objects():
    env, inj = _env()
    env.reset(seed=7)
    first = env.tasks
    for _ in range(2):
        env.step(np.full(env.action_dim, 0.9, dtype=np.float64))
    env.reset(seed=7)
    assert [id(t) for t in env.tasks] != [id(t) for t in first]
    # injection 内的 pristine 模板**不得**被就地执行污染
    assert all(t.status == "not_arrived" for t in inj.tasks)


@needs_assets
def test_reset_replay_restores_the_initial_observation():
    env, _ = _daylight_env()
    obs_before, _ = env.reset(seed=7)
    for _ in range(3):
        env.step(np.full(env.action_dim, 0.9, dtype=np.float64))
    obs_after, _ = env.reset(seed=7)
    assert np.allclose(obs_before, obs_after)


# --- R1-4. refs_v4 全量接线 -----------------------------------------------------

@needs_assets
@pytest.mark.parametrize("attr,expected", sorted(FORMAL_REFS.items()))
def test_formal_env_consumes_every_reference_from_refs_v4(attr, expected):
    env, _ = _env()
    assert float(getattr(env, attr)) == pytest.approx(expected)


@needs_assets
def test_formal_refs_come_from_the_verified_refs_v4_file():
    import json

    refs = json.loads(REFS_V4.read_text(encoding="utf-8"))["references"]
    env, _ = _env()
    for name in ("price_ref", "pv_ref_kw", "wind_ref_kw", "carbon_factor_ref",
                 "cost_ref", "carbon_ref", "grid_power_limit_kW",
                 "peak_power_ref_kW", "peak_power_threshold_kW", "sla_penalty_ref",
                 "queue_ref", "queue_capacity_ref"):
        assert float(getattr(env, name)) == pytest.approx(float(refs[name]["value"]))
    # arrival 每步归一化 = 冻结 rate × delta_t_hours
    assert float(env.lambda_ref) == pytest.approx(
        float(refs["lambda_ref"]["value"]) * 0.5)


# --- R1-5. forecast_cutoff 一致性 -----------------------------------------------

@needs_assets
def test_forecast_cutoff_must_match_the_injection():
    from envs.idc_price_env import IDCPriceEnv20D

    inj = _build(horizon=8, cutoff=4)
    with pytest.raises(ValueError):
        IDCPriceEnv20D(horizon=8, task_seed=7, server_seed=0, forecast_seed=7,
                       delta_t_hours=0.5, forecast_cutoff=2,
                       formal_injection=inj)
    ok = IDCPriceEnv20D(horizon=8, task_seed=7, server_seed=0, forecast_seed=7,
                        delta_t_hours=0.5, forecast_cutoff=4,
                        formal_injection=inj)
    assert ok.forecast_cutoff == inj.forecast_cutoff


# --- R1-6. provenance 覆盖面 ----------------------------------------------------

@needs_assets
def test_provenance_hash_covers_realized_forecast_refs_and_tasks():
    m = injection_module()
    inj = _build(horizon=8, cutoff=4)
    assert len(inj.provenance_hash) == 64
    # 覆盖 realized exogenous
    assert inj.price_sgd_per_kwh and inj.local_pv_kw and inj.wind_generation_kw
    assert inj.carbon_kg_per_kwh and inj.temperature_deg_c
    # 覆盖 causal forecast
    assert inj.causal_forecasts and "price_forecast" in inj.causal_forecasts
    # 覆盖 refs
    assert inj.formal_refs and "price_ref" in inj.formal_refs
    # 覆盖任务业务内容（不只是 id/ledger）
    assert inj.task_specs
    changed = m._provenance_hash(
        type(inj)(**{**inj.__dict__, "price_sgd_per_kwh": tuple(
            v + 1.0 for v in inj.price_sgd_per_kwh)}))
    assert changed != inj.provenance_hash


# =============================================================================
# M1.3g-e-c-R2：formal carbon 观测通道分离（+ P3 清理）
#
# 改前缺陷（本段在实现前必须为红）：
# - `_get_forecast_features` 的 carbon 分组读 **realized** `carbon_factor_t`，
#   而读入了正确 causal 值的 `carbon_forecast_t` **全仓库无人读取**（死代码）；
# - 死属性 `arrival_forecast_t` 从未被读取。
#
# ⚠️ canonical carbon 与它的 causal forecast **当前恒等**（同为冻结常量 0.402），
# 因此本段**不得**依赖「二者天然不同」；一律在 reset 后对目标数组做**明确 mutation**
# 来检验读取路径，并附**非空洞性断言**证明 mutation 确实生效。
# =============================================================================

# `_get_forecast_features` 的固定分组顺序（M3.10b）。
FORECAST_GROUP_ORDER = ("price", "temperature", "arrival", "pv", "wind", "carbon",
                        "sin", "cos")


def _forecast_groups(features, horizon: int) -> dict:
    """按固定顺序把 forecast observation 切成 8 组，便于逐组比较。"""
    assert features.shape[0] == len(FORECAST_GROUP_ORDER) * horizon
    return {name: features[k * horizon:(k + 1) * horizon]
            for k, name in enumerate(FORECAST_GROUP_ORDER)}


# --- R2-1. realized carbon mutation 不得影响 formal forecast observation ---------

@needs_assets
@pytest.mark.leakage
def test_formal_carbon_observation_ignores_realized_carbon_mutation():
    """formal 下改 **realized** carbon → forecast observation **逐位不变**。"""
    env, _ = _daylight_env()
    env.reset(seed=7)
    base = env._get_forecast_features().copy()

    realized_before = np.asarray(env.carbon_factor_t, dtype=np.float64).copy()
    # 只改**可见窗口内**的 realized carbon
    env.carbon_factor_t[:env.forecast_cutoff] += 5.0
    # --- 非空洞性：mutation 必须真的改到目标数组 ---
    assert not np.array_equal(realized_before, env.carbon_factor_t), \
        "mutation 未生效：realized carbon 数组没变，本用例无意义"
    assert float(env.carbon_factor_t[0]) == float(realized_before[0]) + 5.0

    assert np.array_equal(base, env._get_forecast_features()), \
        "formal forecast observation 不得读取 realized carbon_factor_t"


# --- R2-2. causal carbon forecast mutation 必须影响 carbon 分组 ------------------

@needs_assets
@pytest.mark.leakage
def test_formal_carbon_observation_follows_the_causal_carbon_forecast():
    """formal 下改**可见窗口内** causal carbon → carbon 分组必变，其余分组不变。"""
    env, _ = _daylight_env()
    env.reset(seed=7)
    horizon = env.horizon
    base = env._get_forecast_features().copy()
    base_groups = _forecast_groups(base, horizon)

    causal_before = np.asarray(env.carbon_forecast_t, dtype=np.float64).copy()
    env.carbon_forecast_t[:env.forecast_cutoff] += 5.0
    # --- 非空洞性：mutation 必须真的改到目标数组 ---
    assert not np.array_equal(causal_before, env.carbon_forecast_t), \
        "mutation 未生效：causal carbon 数组没变，本用例无意义"
    assert float(env.carbon_forecast_t[0]) == float(causal_before[0]) + 5.0

    after_groups = _forecast_groups(env._get_forecast_features(), horizon)

    assert not np.allclose(base_groups["carbon"], after_groups["carbon"]), \
        "carbon 观测分组必须读 causal carbon_forecast_t（当前读的是 realized 死值）"
    for name in ("price", "temperature", "arrival", "pv", "wind", "sin", "cos"):
        assert np.allclose(base_groups[name], after_groups[name]), \
            f"{name} 分组不得受 carbon forecast mutation 影响"


# --- R2-3. 窗口外 carbon forecast mutation 不影响当前 observation（反向控制）-----

@needs_assets
@pytest.mark.leakage
def test_formal_carbon_observation_ignores_out_of_window_carbon_forecast():
    """只改可见窗口**之外**的 causal carbon → 当前 observation 不变。"""
    env, _ = _daylight_env()          # horizon=8, cutoff=4 → 可见窗口 [0, 4)
    env.reset(seed=7)
    base = env._get_forecast_features().copy()

    t = int(env.current_step)
    lo, hi = t + env.forecast_cutoff, env.horizon
    assert lo > t, "可见窗口为空 → 本用例无意义"
    assert hi > lo, "可见窗口之外的区间为空 → 反向控制无意义"

    target_before = np.asarray(env.carbon_forecast_t, dtype=np.float64).copy()
    env.carbon_forecast_t[lo:hi] += 1000.0
    # --- 非空洞性：mutation 必须真的改到**窗口外**那一段 ---
    target_after = np.asarray(env.carbon_forecast_t, dtype=np.float64)
    assert not np.array_equal(target_before, target_after), \
        "mutation 未生效：窗口外 carbon forecast 没变，本用例无意义"
    assert np.array_equal(target_after[lo:hi], target_before[lo:hi] + 1000.0)

    assert np.array_equal(base, env._get_forecast_features()), \
        "可见窗口之外的 carbon forecast 不得影响当前 observation"


# --- R2-4. P3 清理：死属性 arrival_forecast_t 必须删除 --------------------------

@needs_assets
def test_arrival_forecast_has_no_dead_attribute():
    """arrival 的 causal forecast 唯一来源是 `task_arrival_forecast`。"""
    env, inj = _daylight_env()
    env.reset(seed=7)
    assert not hasattr(env, "arrival_forecast_t"), \
        "arrival_forecast_t 从未被读取，必须删除；观测只走 task_arrival_forecast"
    assert [float(v) for v in env.task_arrival_forecast] == list(inj.arrival_forecast)

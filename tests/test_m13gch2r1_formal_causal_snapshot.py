"""M1.3g-f-c-h2-R1：正式 corrector 规划输入的**因果隔离**回归 + 探针碳累加。

**改前缺陷（本文件对应先红）**：`planning/snapshot_adapter.py` 对 **formal** env
也把 `env.price_t` / `env.T_amb` / `env.pv_t` / `env.wt_t` / `env.carbon_factor_t`
当作「可见预测」，取 `[t, t+cutoff)` —— 这些是 **realized 真值**，故规划输入含
`t+1 .. t+cutoff-1` 的**未来真值**；`snapshot.forecast` 也恒为 `mode="oracle_debug"`。
审核方实测：origin 48 只改 `price_t[1]`，快照价格随之改动。

**目标**：formal 的规划输入与 forecast **只**来自**已验签的 B6 因果预测**通道
（`env.*_forecast_t` / `env.task_arrival_forecast`，由
`build_verified_formal_env_injection` → `build_formal_scenario_b6` 注入），
且 provenance 能通过 `purpose="training"` 门禁。**legacy / oracle_debug 逐字不变。**
"""

import importlib
import pathlib

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

HORIZON = 24
CUTOFF = 4
DELTA_T_HOURS = 0.5
START = "2024-01-02T00:00:00+08:00"
SEED_KWARGS = {"task_seed": 0, "server_seed": 1, "forecast_seed": 300000}

# 因果通道（formal 规划**唯一**允许的来源）
CAUSAL_ATTRS = {
    "price": "price_forecast_t",
    "pv": "pv_forecast_t",
    "wind": "wind_forecast_t",
    "temperature": "temperature_forecast_t",
    "carbon": "carbon_forecast_t",
}
# realized 真值通道（formal 规划**不得**读取）
REALIZED_ATTRS = {
    "price": "price_t",
    "pv": "pv_t",
    "wind": "wt_t",
    "temperature": "T_amb",
    "carbon": "carbon_factor_t",
}


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


pytestmark = pytest.mark.skipif(not _upstream_present(),
                                reason="真实冻结上游资产不在本机")


def _env_cls():
    return importlib.import_module("envs.idc_price_env").IDCPriceEnv20D


def _build_snapshot(env):
    return importlib.import_module("planning.snapshot_adapter").build_snapshot(env)


def _formal_env(*, horizon: int = HORIZON, cutoff: int = CUTOFF, t: int = 0):
    """真实正式链上的环境（B6 causal forecast 通道已注入）。"""
    inj = importlib.import_module("scenario.env_injection") \
        .build_verified_formal_env_injection(
            "train", start=START, horizon=horizon, forecast_cutoff=cutoff)
    env = _env_cls()(horizon=horizon, forecast_cutoff=cutoff,
                     delta_t_hours=DELTA_T_HOURS, formal_injection=inj,
                     **SEED_KWARGS)
    env.reset(seed=0)
    action = np.zeros(env.action_dim, dtype=np.float64)
    action[: env.model.N] = 0.5
    for _ in range(t):
        env.step(action)
    assert env.formal is True
    return env


def _legacy_env(*, horizon: int = HORIZON, cutoff: int = CUTOFF, t: int = 0):
    env = _env_cls()(horizon=horizon, forecast_cutoff=cutoff, **SEED_KWARGS)
    env.reset(seed=0)
    action = np.zeros(env.action_dim, dtype=np.float64)
    action[: env.model.N] = 0.5
    for _ in range(t):
        env.step(action)
    assert env.formal is False
    return env


# =============================================================================
# 1. 因果隔离：只扰动未来 realized，规划输入不得变化
# =============================================================================

@pytest.mark.leakage
def test_future_realized_price_mutation_at_origin_does_not_change_formal_snapshot():
    """审核方的**逐字复现**：origin 48（`t=0`、`cutoff=48`）只改 `price_t[1] = 123`。

    `cutoff == horizon == 48` 时旧实现的可见窗口覆盖整个 horizon，
    故 `price_t[1]` 直接进入规划；修复后规划只读 causal forecast。
    """
    env = _formal_env(horizon=48, cutoff=48, t=0)
    before = _build_snapshot(env)

    env.price_t[1] = 123.0

    after = _build_snapshot(env)
    assert before.planning_forecast.model_dump() == after.planning_forecast.model_dump(), \
        "改动未来 realized price_t[1] 不得改变 formal 的规划输入"
    assert before.forecast.model_dump() == after.forecast.model_dump()


@pytest.mark.leakage
@pytest.mark.parametrize("cutoff", [4, 48])
def test_future_realized_mutation_does_not_change_formal_snapshot(cutoff):
    """`[t+1, horizon)` 的 realized 真值整体被改，规划快照逐字节不变。"""
    env = _formal_env(cutoff=cutoff, t=1)
    before = _build_snapshot(env)

    t = env.current_step
    for idx in range(t + 1, env.horizon):
        for attr in REALIZED_ATTRS.values():
            getattr(env, attr)[idx] = 9999.0
        env.true_task_arrival_profile[idx] = 9999.0

    after = _build_snapshot(env)
    assert before.planning_forecast.model_dump() == after.planning_forecast.model_dump()
    assert before.forecast.model_dump() == after.forecast.model_dump()
    assert before.base_idc_power_forecast_kw == after.base_idc_power_forecast_kw


@pytest.mark.leakage
def test_formal_base_idc_power_does_not_read_realized_temperature():
    """`base_idc_power` 由温度推得：必须只跟 causal 温度走。"""
    env = _formal_env(t=1)
    before = _build_snapshot(env).planning_forecast.base_idc_power

    env.T_amb[:] = -50.0  # realized 温度整体改废

    assert _build_snapshot(env).planning_forecast.base_idc_power == before


# =============================================================================
# 2. 反向控制：改动可见窗口内的 causal forecast ⇒ 对应字段必须变化
# =============================================================================

@pytest.mark.parametrize("name", sorted(CAUSAL_ATTRS))
def test_visible_causal_forecast_change_moves_the_matching_planning_field(name):
    env = _formal_env(t=1, cutoff=CUTOFF)
    before = _build_snapshot(env)

    attr = CAUSAL_ATTRS[name]
    series = getattr(env, attr)
    series[1] = float(series[1]) + 7.5  # 可见窗口 [t, t+cutoff) 内

    after = _build_snapshot(env)
    before_vec = list(getattr(before.planning_forecast, name))
    after_vec = list(getattr(after.planning_forecast, name))
    assert before_vec != after_vec, f"改动可见 causal forecast 必须改变 {name}"
    # 源索引 1（= 该步可见）对应规划索引 `1 - t` = 0
    assert after_vec[1 - env.current_step] != before_vec[1 - env.current_step]


def test_visible_causal_temperature_moves_base_idc_power():
    env = _formal_env(t=1, cutoff=CUTOFF)
    before = list(_build_snapshot(env).planning_forecast.base_idc_power)

    env.temperature_forecast_t[1] = float(env.temperature_forecast_t[1]) + 7.5

    after = list(_build_snapshot(env).planning_forecast.base_idc_power)
    assert after != before
    assert after[1 - env.current_step] != before[1 - env.current_step]


def test_visible_causal_arrival_moves_the_arrival_field():
    env = _formal_env(t=1, cutoff=CUTOFF)
    before = list(_build_snapshot(env).planning_forecast.arrival)

    env.task_arrival_forecast[1] = float(env.task_arrival_forecast[1]) + 12.0

    after = list(_build_snapshot(env).planning_forecast.arrival)
    assert after != before
    assert after[1 - env.current_step] != before[1 - env.current_step]


@pytest.mark.leakage
def test_causal_change_outside_the_visible_window_is_ignored():
    """窗口外策略不变：`[t+cutoff, horizon)` 的 causal 值不得进入规划。"""
    env = _formal_env(t=1, cutoff=CUTOFF)
    before = _build_snapshot(env)

    for attr in CAUSAL_ATTRS.values():
        series = getattr(env, attr)
        for idx in range(env.current_step + CUTOFF, env.horizon):
            series[idx] = 8888.0

    assert _build_snapshot(env).planning_forecast.model_dump() == \
        before.planning_forecast.model_dump()


# =============================================================================
# 3. 来源可验证：formal provenance + training purpose gate + 真实时间窗口
# =============================================================================

def test_formal_snapshot_bundle_is_mode_formal():
    snap = _build_snapshot(_formal_env(t=1))
    assert snap.forecast.mode == "formal", (
        "formal env 的快照 forecast 必须是 mode=formal（不得把 oracle_debug 改名）")


def test_formal_snapshot_bundle_passes_the_training_purpose_gate():
    from contracts.validators import validate_forecast_purpose

    snap = _build_snapshot(_formal_env(t=1))
    # 不抛异常即通过；再显式断言一次门禁语义。
    validate_forecast_purpose(snap.forecast, purpose="training")


def test_oracle_debug_bundle_would_be_rejected_for_training():
    """**反向**：oracle_debug 快照必须被 training 门禁拒绝（本卡不得放宽门禁）。"""
    from contracts.validators import validate_forecast_purpose

    legacy = _build_snapshot(_legacy_env(t=1))
    assert legacy.forecast.mode == "oracle_debug"
    with pytest.raises(ValueError, match="只接受 mode=formal"):
        validate_forecast_purpose(legacy.forecast, purpose="training")


def test_formal_snapshot_provenance_is_real_and_not_the_dev_anchor():
    from datetime import datetime

    snap = _build_snapshot(_formal_env(t=1))
    bundle = snap.forecast

    assert bundle.forecast_provenance.price_forecast.code_revision != ""
    assert len(bundle.forecast_provenance.price_forecast.code_revision) == 40
    for name in ("price_forecast", "pv_forecast", "wind_forecast",
                 "temperature_forecast", "carbon_forecast", "arrival_forecast"):
        entry = getattr(bundle.forecast_provenance, name)
        assert entry.sources, f"{name} 必须带来源 hash"
        assert entry.source_kind not in ("oracle_debug", "synthetic", "unavailable")
        assert entry.generated_at == bundle.generated_at
        # 时间窗口必须可解析且有序（真实窗口，不是 dev 锚点）
        assert datetime.fromisoformat(entry.target_start) <= \
            datetime.fromisoformat(entry.target_end_exclusive)

    adapter = importlib.import_module("planning.snapshot_adapter")
    assert bundle.generated_at != adapter.ORACLE_DEBUG_ANCHOR, \
        "formal 不得复用 dev 锚点时间"


def test_formal_planning_window_slices_the_verified_bundle():
    """规划窗口必须**逐位**等于已验签 bundle 在 `t` 处的切片。"""
    env = _formal_env(t=2, cutoff=CUTOFF)
    snap = _build_snapshot(env)
    t = env.current_step
    n_visible = min(CUTOFF, env.horizon - t)

    for k in range(n_visible):
        assert snap.planning_forecast.price[k] == \
            pytest.approx(snap.forecast.price_forecast[t + k])
        assert snap.planning_forecast.pv[k] == \
            pytest.approx(snap.forecast.pv_forecast[t + k])
        assert snap.planning_forecast.wind[k] == \
            pytest.approx(snap.forecast.wind_forecast[t + k])
        assert snap.planning_forecast.temperature[k] == \
            pytest.approx(snap.forecast.temperature_forecast[t + k])
        assert snap.planning_forecast.carbon[k] == \
            pytest.approx(snap.forecast.carbon_forecast[t + k])
        assert snap.planning_forecast.arrival[k] == \
            pytest.approx(snap.forecast.arrival_forecast[t + k])


def test_formal_snapshot_passes_the_full_validator():
    from contracts.validators import validate_snapshot

    validate_snapshot(_build_snapshot(_formal_env(t=1)))


# =============================================================================
# 4. legacy / oracle_debug 路径**逐字不变**
# =============================================================================

def test_legacy_snapshot_still_reads_the_realized_window():
    """legacy 语义保留：窗口内的 realized 真值仍进入规划（原测试依赖此行为）。"""
    env = _legacy_env(t=1, cutoff=CUTOFF)
    before = list(_build_snapshot(env).planning_forecast.price)

    env.price_t[1] = float(env.price_t[1]) + 3.25

    after = list(_build_snapshot(env).planning_forecast.price)
    assert after != before
    # 源索引 1（= 当前步，属可见窗口）对应规划索引 0
    assert after[1 - env.current_step] != before[1 - env.current_step]


def test_legacy_snapshot_keeps_the_oracle_debug_bundle_shape():
    env = _legacy_env(t=1, cutoff=CUTOFF)
    snap = _build_snapshot(env)
    assert snap.forecast.mode == "oracle_debug"
    assert len(snap.forecast.price_forecast) == CUTOFF


def test_legacy_causal_forecast_arrays_are_not_a_planner_source():
    """legacy 下 `*_forecast_t` 是 realized 的别名，不是独立因果通道。"""
    env = _legacy_env(t=1, cutoff=CUTOFF)
    assert np.array_equal(env.price_forecast_t, env.price_t)


# =============================================================================
# 5. P2：`replay_corrector_service.py` 的 carbon_total 必须**逐步累加**
# =============================================================================

class _FakeModel:
    N = 2


class _FakeEnv:
    """只提供 `run()` 需要的最小接口（不触真实环境）。"""

    def __init__(self, **_kwargs):
        self.action_dim = 3
        self.model = _FakeModel()
        self.initial_Q = 0.0


class _FakeInjection:
    ledger_micro = (0,) * 4


class _FakeWrapper:
    """按脚本预设的逐 step 碳排返回 info（**至少两步取值不同**）。"""

    def __init__(self, env, *, corrector_time_limit_s):
        self.env = env
        self.corrector_time_limit_s = corrector_time_limit_s
        self._step = 0

    def reset(self, seed=None):
        return np.zeros(3, dtype=np.float32), {}

    def step(self, _action):
        carbon = self.CARBON_SEQUENCE[self._step]
        self._step += 1
        # 序列走完即终止 episode：让脚本的 `for _ in range(HORIZON)` 只跑序列长度步。
        terminated = self._step >= len(self.CARBON_SEQUENCE)
        info = {
            "completed_work": 1.0,
            "sla_violation_count": 0,
            "carbon_emission": carbon,
            "correction_reason": "none",
        }
        return np.zeros(3, dtype=np.float32), 0.0, terminated, False, info


def _patch_replay(monkeypatch, carbon_sequence):
    """把重放脚本的三个外部依赖替换为离线桩（**不动**脚本自身的累加逻辑）。"""
    replay = importlib.import_module("scripts.replay_corrector_service")
    inj_mod = importlib.import_module("scenario.env_injection")
    env_mod = importlib.import_module("envs.idc_price_env")
    wrapper_mod = importlib.import_module("safe_rl.corrector_wrapper")

    class _Wrapper(_FakeWrapper):
        CARBON_SEQUENCE = tuple(carbon_sequence)

    monkeypatch.setattr(replay, "start_for_origin",
                        lambda origin: "2024-01-02T00:00:00+08:00")
    monkeypatch.setattr(inj_mod, "build_verified_formal_env_injection",
                        lambda *a, **k: _FakeInjection())
    monkeypatch.setattr(env_mod, "IDCPriceEnv20D", _FakeEnv)
    monkeypatch.setattr(wrapper_mod, "CorrectorWrapper", _Wrapper)
    return replay


def test_replay_carbon_total_accumulates_every_step(monkeypatch):
    """三步碳排 1.0 / 2.5 / 4.0 ⇒ 必须累加为 7.5（旧实现只取最后一步 4.0）。"""
    replay = _patch_replay(monkeypatch, (1.0, 2.5, 4.0))
    row = replay.run(48, 0.25)
    assert row["steps"] == 3
    assert len(set((1.0, 2.5, 4.0))) == 3, "用例本身必须含至少两步不同的碳排"
    assert row["carbon_total"] == pytest.approx(7.5), (
        "carbon_total 必须是**逐步累加**，不是最后一步的值")


def test_replay_carbon_total_is_not_the_last_step(monkeypatch):
    """两步碳排 2.0 / 5.0：若仍取最后一步会得到 5.0（缺陷值），累加应为 7.0。"""
    replay = _patch_replay(monkeypatch, (2.0, 5.0))
    row = replay.run(48, 0.25)
    assert row["steps"] == 2
    assert row["carbon_total"] == pytest.approx(7.0)
    assert row["carbon_total"] != pytest.approx(5.0)

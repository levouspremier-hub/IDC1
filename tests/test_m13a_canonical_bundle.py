"""M1.3a 测试：scenario provider 与 contracts.ScenarioBundle 类型统一。"""

from pathlib import Path

import numpy as np
import pytest

import contracts
import scenario
from contracts.validators import validate_scenario
from envs.idc_price_env import IDCPriceEnv20D, visible_window_slice
from planning.snapshot_adapter import build_snapshot
from scenario import build_scenario, build_scenario_from_true
from scenario.scenario import SERIES_KEYS

SIX_FORECAST_FIELDS = (
    "price_forecast",
    "load_forecast",
    "pv_forecast",
    "wind_forecast",
    "temperature_forecast",
    "carbon_forecast",
)


def _true(horizon: int = 24) -> dict[str, np.ndarray]:
    t = np.arange(horizon, dtype=float)
    return {
        "price": 0.2 + 0.01 * t,
        "load": 6000.0 + 10.0 * t,
        "pv": np.clip(np.sin(t / 4.0), 0.0, None),
        "wind": np.clip(np.cos(t / 5.0), 0.0, None),
        "temperature": 28.0 + np.zeros(horizon),
        "carbon": 0.5 + 0.01 * t,
        "arrival": 80.0 + 5.0 * t,
    }


# --- 1. 类型统一 ---

def test_provider_returns_contract_type():
    b = build_scenario("train", "2026-01-01T00:00:00", 24, 6, synthetic=True, seed=42)
    assert isinstance(b, contracts.ScenarioBundle)


def test_no_local_duplicate_class():
    """scenario 模块不得再定义自己的 ScenarioBundle。"""
    import scenario.scenario as mod

    assert mod.ScenarioBundle is contracts.ScenarioBundle
    assert scenario.ScenarioBundle is contracts.ScenarioBundle
    src = Path("scenario/scenario.py").read_text(encoding="utf-8")
    assert "class ScenarioBundle" not in src, "本地重复类仍存在"
    assert "build_scenario_from_true" in src  # 模块仍有实际内容


# --- 2. 六类预测字段与元数据完整 ---

@pytest.mark.parametrize("cutoff", [2, 6])
def test_all_six_forecast_fields_present(cutoff):
    b = build_scenario("train", "s", 24, cutoff, synthetic=True, seed=1)
    for field in SIX_FORECAST_FIELDS:
        values = getattr(b, field)
        assert isinstance(values, list)
        assert len(values) == cutoff, f"{field} 长度 {len(values)} != cutoff {cutoff}"
    assert b.forecast_cutoff == cutoff
    assert b.split == "train"
    assert b.start == "s"
    assert b.horizon == 24


def test_metadata_not_lost():
    b = build_scenario("train", "s", 24, 4, synthetic=True, seed=3)
    assert b.synthetic is True
    assert isinstance(b.source_hashes, dict) and b.source_hashes, "source_hashes 不可为空"
    assert "generator" in b.source_hashes


def test_from_true_requires_six_series():
    b = build_scenario_from_true("train", "s", 24, 4, _true(24), synthetic=True)
    assert isinstance(b, contracts.ScenarioBundle)
    assert len(b.carbon_forecast) == 4
    assert set(SERIES_KEYS) == {"price", "load", "pv", "wind", "temperature", "carbon", "arrival"}


# --- 3. contracts.validators 显式拒绝 ---

def test_provider_output_passes_validator():
    validate_scenario(build_scenario("train", "s", 24, 4, synthetic=True, seed=5))


def test_validator_rejects_bad_window_length():
    b = build_scenario("train", "s", 24, 4, synthetic=True, seed=5)
    bad = b.model_copy(update={"carbon_forecast": [0.1, 0.2]})  # 长度 2 != cutoff 4
    with pytest.raises(ValueError):
        validate_scenario(bad)


def test_validator_rejects_missing_source_hashes():
    b = build_scenario("train", "s", 24, 4, synthetic=True, seed=5)
    bad = b.model_copy(update={"source_hashes": {}})
    with pytest.raises(ValueError, match="来源"):
        validate_scenario(bad)


# --- 4. 未来真值不泄漏（[t, t+cutoff) 语义） ---

@pytest.mark.leakage
def test_future_truth_mutation_does_not_change_bundle():
    true = _true(24)
    before = build_scenario_from_true("train", "s", 24, 4, true, synthetic=True)
    for k in SERIES_KEYS:
        true[k][10] = 9999.0  # 超出 cutoff=4
    after = build_scenario_from_true("train", "s", 24, 4, true, synthetic=True)
    for field in SIX_FORECAST_FIELDS:
        assert getattr(before, field) == getattr(after, field)
    assert before.content_hash() == after.content_hash()


@pytest.mark.leakage
def test_visible_truth_mutation_changes_bundle():
    true = _true(24)
    before = build_scenario_from_true("train", "s", 24, 4, true, synthetic=True)
    true["carbon"][2] = 9999.0  # 在 cutoff 内
    after = build_scenario_from_true("train", "s", 24, 4, true, synthetic=True)
    assert before.content_hash() != after.content_hash()


# --- 5. 与 snapshot adapter 的 cutoff 语义一致 ---

def test_provider_and_snapshot_share_cutoff_semantics():
    cutoff = 4
    env = IDCPriceEnv20D(horizon=24, forecast_cutoff=cutoff, access_limit_kw=1000.0)
    env.reset(seed=0)
    snap = build_snapshot(env)
    start, end = visible_window_slice(env.current_step, cutoff, env.horizon)
    # adapter 的窗口长度固定为 cutoff，且可见元素个数与 [t, t+cutoff) 一致
    assert len(snap.forecast.price_forecast) == cutoff
    assert len(snap.forecast.carbon_forecast) == cutoff
    assert (end - start) <= cutoff


# --- 6. 数据边界：合成 vs 真实 ---

def test_synthetic_flagged_and_real_mode_blocked():
    b = build_scenario("train", "s", 24, 4, synthetic=True, seed=9)
    assert b.synthetic is True
    # 正式模式（无 M1.2 数据）必须显式失败，不得回退到合成
    with pytest.raises(FileNotFoundError):
        build_scenario("train", "s", 24, 4, synthetic=False, manifest_dir="data/manifest")

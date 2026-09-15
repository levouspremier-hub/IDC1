"""M1.3a 测试：scenario provider 与 contracts.ScenarioBundle 类型统一。

**M1.3e 迁移**：契约升到 `contract-v8`，`synthetic: bool` 由显式的 `mode` 取代、
自由 dict `source_hashes` 退役为结构化 `forecast_provenance`；
`build_scenario_from_true` 改名为 `build_oracle_debug_scenario_from_truth`
（必须显式 `oracle_debug=True`）。

原来那两条「可见 truth 改变 bundle」的断言**没有删除、也没有反转**：它们改名为
**oracle-debug** 语义——把 `[t, t+cutoff)` 真值当作预测**正是 oracle-debug 的定义**。
正式 causal forecast 的 leakage 回归在 `tests/test_m13e_forecast_provenance.py`。
"""

from pathlib import Path

import numpy as np
import pytest

import contracts
import scenario
from contracts.models import ScenarioForecastProvenance
from contracts.validators import validate_scenario
from envs.idc_price_env import IDCPriceEnv20D, visible_window_slice
from planning.snapshot_adapter import build_snapshot
from scenario import build_oracle_debug_scenario_from_truth, build_scenario
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


def _oracle_debug(true: dict[str, np.ndarray], horizon: int = 24, cutoff: int = 4):
    return build_oracle_debug_scenario_from_truth(
        "train", "s", horizon, cutoff, true, oracle_debug=True
    )


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
    assert "build_oracle_debug_scenario_from_truth" in src  # 模块仍有实际内容


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
    """元数据不再放在自由 dict 里：改为结构化 provenance。"""
    b = build_scenario("train", "s", 24, 4, synthetic=True, seed=3)
    assert b.mode == "synthetic"
    assert isinstance(b.forecast_provenance, ScenarioForecastProvenance)
    assert b.generated_at
    for field in SIX_FORECAST_FIELDS:
        entry = getattr(b.forecast_provenance, field)
        assert entry.series_name == field
        assert entry.sources, f"{field} 的来源 digest 不可为空"
        assert entry.method


def test_oracle_debug_helper_covers_all_seven_series():
    b = _oracle_debug(_true(24))
    assert isinstance(b, contracts.ScenarioBundle)
    assert b.mode == "oracle_debug"
    assert len(b.carbon_forecast) == 4
    assert set(SERIES_KEYS) == {"price", "load", "pv", "wind", "temperature", "carbon",
                               "arrival"}


# --- 3. contracts.validators 显式拒绝 ---

def test_provider_output_passes_validator():
    validate_scenario(build_scenario("train", "s", 24, 4, synthetic=True, seed=5))


def test_validator_rejects_bad_window_length():
    b = build_scenario("train", "s", 24, 4, synthetic=True, seed=5)
    bad = b.model_copy(update={"carbon_forecast": [0.1, 0.2]})  # 长度 2 != cutoff 4
    with pytest.raises(ValueError):
        validate_scenario(bad)


def test_validator_rejects_missing_source_provenance():
    """`source_hashes` 自由 dict 已退役；**没有来源 digest** 仍然必须失败。"""
    b = build_scenario("train", "s", 24, 4, synthetic=True, seed=5)
    payload = b.model_dump()
    payload["forecast_provenance"]["carbon_forecast"]["sources"] = []
    with pytest.raises(ValueError, match="来源"):
        contracts.ScenarioBundle(**payload)


# --- 4. oracle-debug 语义下的可见/未来窗口（原 M1.3a 断言，改名保留） ---

@pytest.mark.leakage
def test_future_truth_mutation_does_not_change_bundle():
    true = _true(24)
    before = _oracle_debug(true)
    for k in SERIES_KEYS:
        true[k][10] = 9999.0  # 超出 cutoff=4
    after = _oracle_debug(true)
    for field in SIX_FORECAST_FIELDS:
        assert getattr(before, field) == getattr(after, field)
    assert before.content_hash() == after.content_hash()


@pytest.mark.leakage
def test_oracle_debug_visible_truth_mutation_changes_bundle():
    """**oracle-debug 定义**：窗口内真值就是预测，所以它改变 bundle。

    这条断言**不是**正式 forecast 的语义——正式 causal provider 只读
    `[origin-48, origin)` 历史，其 future-truth mutation 回归见
    `tests/test_m13e_forecast_provenance.py`（`@pytest.mark.leakage`）。
    """
    true = _true(24)
    before = _oracle_debug(true)
    assert before.mode == "oracle_debug"
    true["carbon"][2] = 9999.0  # 在 cutoff 内
    after = _oracle_debug(true)
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


# --- 6. 数据边界：合成 / oracle-debug vs 正式 ---

def test_synthetic_flagged_and_real_mode_blocked():
    b = build_scenario("train", "s", 24, 4, synthetic=True, seed=9)
    assert b.mode == "synthetic"
    # 正式模式（无 M1.3 split/forecast）必须显式失败，不得回退到合成
    with pytest.raises(FileNotFoundError):
        build_scenario("train", "s", 24, 4, synthetic=False, manifest_dir="data/manifest")

"""M1.3 场景提供器测试：可见预测 + 未来信息泄漏回归。"""

import numpy as np
import pytest

from scenario import ScenarioBundle, build_scenario, build_scenario_from_true


def _true_arrays(horizon: int = 24) -> dict[str, np.ndarray]:
    t = np.arange(horizon, dtype=float)
    return {
        "price": 0.2 + 0.01 * t,
        "load": 6000.0 + 10.0 * t,
        "pv": np.clip(np.sin(t / 4.0), 0.0, None),
        "wind": np.zeros(horizon),
        "temperature": 28.0 + np.zeros(horizon),
    }


def test_synthetic_bundle_shape():
    b = build_scenario("train", "2026-01-01T00:00:00", 24, 6, synthetic=True, seed=42)
    assert isinstance(b, ScenarioBundle)
    assert b.horizon == 24 and b.forecast_cutoff == 6
    assert set(b.current) == {"price", "load", "pv", "wind", "temperature"}
    assert all(len(v) == 6 for v in b.forecast.values())
    assert b.synthetic is True
    assert len(b.hash) == 64


@pytest.mark.leakage
def test_future_truth_mutation_does_not_change_decision_input():
    true = _true_arrays(24)
    before = build_scenario_from_true("train", "start", 24, 4, true)
    # 篡改未来真值 t=20（超出 forecast_cutoff=4），不得影响当前决策输入
    true["price"][20] = 9999.0
    true["load"][20] = 0.0
    after = build_scenario_from_true("train", "start", 24, 4, true)
    assert before.decision_input() == after.decision_input()
    assert before.hash == after.hash


def test_visible_forecast_mutation_does_change_decision_input():
    true = _true_arrays(24)
    before = build_scenario_from_true("train", "start", 24, 4, true)
    true["price"][2] = 9999.0  # 在 forecast_cutoff 内，可见，应改变输入
    after = build_scenario_from_true("train", "start", 24, 4, true)
    assert before.hash != after.hash


def test_same_input_same_hash():
    b1 = build_scenario("train", "s", 24, 4, synthetic=True, seed=7)
    b2 = build_scenario("train", "s", 24, 4, synthetic=True, seed=7)
    assert b1.hash == b2.hash
    assert b1.decision_input() == b2.decision_input()


def test_formal_mode_without_data_raises():
    with pytest.raises(FileNotFoundError):
        build_scenario("train", "s", 24, 4, synthetic=False, manifest_dir="data/manifest")


def test_invalid_forecast_cutoff_raises():
    with pytest.raises(ValueError):
        build_scenario("train", "s", 24, 25, synthetic=True)

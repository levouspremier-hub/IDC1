"""M1.3 场景提供器测试：可见预测 + 未来信息泄漏回归。

**M1.3e 迁移**：契约升到 `contract-v8`；`build_scenario_from_true` 改名为
`build_oracle_debug_scenario_from_truth`（必须显式 `oracle_debug=True`），
其可见窗口断言保留为 **oracle-debug** 语义（未删除、未反转）。
正式 causal forecast 的 leakage 回归在 `tests/test_m13e_forecast_provenance.py`。
"""

import numpy as np
import pytest

import contracts
from scenario import ScenarioBundle, build_oracle_debug_scenario_from_truth, build_scenario

FORECAST_FIELDS = (
    "price_forecast",
    "load_forecast",
    "pv_forecast",
    "wind_forecast",
    "temperature_forecast",
    "carbon_forecast",
)


def _true_arrays(horizon: int = 24) -> dict[str, np.ndarray]:
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
        "train", "start", horizon, cutoff, true, oracle_debug=True
    )


def test_scenario_type_is_contract():
    assert ScenarioBundle is contracts.ScenarioBundle


def test_synthetic_bundle_shape():
    b = build_scenario("train", "2026-01-01T00:00:00", 24, 6, synthetic=True, seed=42)
    assert isinstance(b, contracts.ScenarioBundle)
    assert b.horizon == 24 and b.forecast_cutoff == 6
    for field in FORECAST_FIELDS:
        assert len(getattr(b, field)) == 6
    assert b.mode == "synthetic"
    assert len(b.content_hash()) == 64


@pytest.mark.leakage
def test_future_truth_mutation_does_not_change_forecast():
    true = _true_arrays(24)
    before = _oracle_debug(true)
    # 篡改未来真值 t=20（超出 forecast_cutoff=4），不得影响可见预测
    true["price"][20] = 9999.0
    true["load"][20] = 0.0
    after = _oracle_debug(true)
    for field in FORECAST_FIELDS:
        assert getattr(before, field) == getattr(after, field)
    assert before.content_hash() == after.content_hash()


def test_oracle_debug_visible_forecast_mutation_does_change_forecast():
    """**oracle-debug 定义**：窗口内真值即预测，因此它改变 bundle。

    这不代表正式 forecast 的语义；正式 causal provider 的 future-truth
    mutation 回归见 `tests/test_m13e_forecast_provenance.py`。
    """
    true = _true_arrays(24)
    before = _oracle_debug(true)
    assert before.mode == "oracle_debug"
    true["price"][2] = 9999.0  # 在 forecast_cutoff 内，可见
    after = _oracle_debug(true)
    assert before.content_hash() != after.content_hash()


def test_same_input_same_hash():
    b1 = build_scenario("train", "s", 24, 4, synthetic=True, seed=7)
    b2 = build_scenario("train", "s", 24, 4, synthetic=True, seed=7)
    assert b1.content_hash() == b2.content_hash()
    assert b1.model_dump() == b2.model_dump()


def test_formal_mode_without_data_raises():
    with pytest.raises(FileNotFoundError):
        build_scenario("train", "s", 24, 4, synthetic=False, manifest_dir="data/manifest")


def test_invalid_forecast_cutoff_raises():
    with pytest.raises(ValueError):
        build_scenario("train", "s", 24, 25, synthetic=True)

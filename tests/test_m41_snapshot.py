"""M4.1 测试：snapshot adapter 构建受限 SystemSnapshot，不访问未来真值。"""

import pytest

from envs.idc_price_env import IDCPriceEnv20D
from planning.snapshot_adapter import build_snapshot


def test_snapshot_has_required_fields():
    env = IDCPriceEnv20D()
    env.reset(seed=0)
    snap = build_snapshot(env)
    assert snap.schema_version is not None
    assert len(snap.group_work_capacity) == env.model.N
    assert snap.soc_kwh is not None
    assert snap.access_limit_kw is not None
    assert isinstance(snap.tasks, list)
    assert snap.forecast is not None
    assert snap.budget_remaining_sgd is not None


@pytest.mark.leakage
def test_snapshot_does_not_leak_future():
    env = IDCPriceEnv20D(forecast_cutoff=2)
    env.reset(seed=0)
    before = build_snapshot(env)

    env.price_t[10] = 9999.0
    env.pv_t[10] = 9999.0
    env.T_amb[10] = 9999.0
    after = build_snapshot(env)

    assert before.forecast.price_forecast == after.forecast.price_forecast
    assert before.forecast.pv_forecast == after.forecast.pv_forecast
    assert before.forecast.temperature_forecast == after.forecast.temperature_forecast

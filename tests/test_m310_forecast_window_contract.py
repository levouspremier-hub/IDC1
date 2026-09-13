"""M3.10a 测试：env 与 snapshot 共用同一可见窗口定义（[t, t+cutoff)）。

定义：forecast_cutoff = 当前时刻起、含当前时刻在内的可见预测点数；
时刻 t 可见区间 = [t, t + forecast_cutoff)，首个不可见下标 = t + forecast_cutoff。
"""

import numpy as np
import pytest

from envs.idc_price_env import IDCPriceEnv20D
from planning.snapshot_adapter import build_snapshot

HORIZON = 24


def _env(cutoff: int, t: int = 0) -> IDCPriceEnv20D:
    """构造真值全正的环境，便于用「非零」判定可见性。"""
    hours = np.arange(HORIZON, dtype=np.float64)
    env = IDCPriceEnv20D(
        horizon=HORIZON,
        forecast_cutoff=cutoff,
        access_limit_kw=1000.0,
        price_t=(hours + 1.0) * 0.1,
        pv_t=hours + 1.0,
        wt_t=hours + 1.0,
        carbon_factor_t=hours + 1.0,
    )
    env.reset(seed=0)
    action = np.concatenate([np.full(20, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)])
    for _ in range(t):
        env.step(action)
    return env


def _expected_window(t: int, cutoff: int, horizon: int = HORIZON) -> tuple[int, int]:
    return max(t, 0), min(t + cutoff, horizon)


def _env_visible_indices(env) -> set[int]:
    """从 observation 的 price 分组判定当前可见下标集合。"""
    obs = np.asarray(env._get_obs(), dtype=np.float64)
    forecast_block = obs[env.current_obs_dim :]
    price_group = forecast_block[: env.horizon]
    return {i for i in range(env.horizon) if price_group[i] != 0.0}


def _snapshot_visible_indices(snap, t: int) -> set[int]:
    window = snap.forecast.price_forecast
    return {t + k for k, v in enumerate(window) if v != 0.0 and t + k < HORIZON}


@pytest.mark.parametrize("cutoff", [2, 4])
@pytest.mark.parametrize("t", [0, 1, 5])
def test_env_and_snapshot_expose_same_range(cutoff, t):
    env = _env(cutoff, t)
    snap = build_snapshot(env)
    start, end = _expected_window(t, cutoff)
    expected = set(range(start, end))
    assert _env_visible_indices(env) == expected
    assert _snapshot_visible_indices(snap, t) == expected


@pytest.mark.parametrize("cutoff", [2, 4])
@pytest.mark.parametrize("t", [0, 1, 5])
def test_first_invisible_index_is_t_plus_cutoff(cutoff, t):
    """t + cutoff 必须不可见；t + cutoff - 1 必须可见（off-by-one 锁定）。"""
    env = _env(cutoff, t)
    _, end = _expected_window(t, cutoff)
    idx_beyond = t + cutoff
    idx_last = t + cutoff - 1

    if idx_beyond < HORIZON:
        env.price_t[idx_beyond] = 9999.0
        env.pv_t[idx_beyond] = 9999.0
        env.wt_t[idx_beyond] = 9999.0
        env.T_amb[idx_beyond] = 9999.0
        env.carbon_factor_t[idx_beyond] = 9999.0
        assert _env_visible_indices(env) == set(range(*_expected_window(t, cutoff)))
        snap = build_snapshot(env)
        assert idx_beyond not in _snapshot_visible_indices(snap, t)

    if idx_last < HORIZON and idx_last >= 0:
        env2 = _env(cutoff, t)
        before = np.asarray(env2._get_obs(), dtype=np.float64)
        env2.price_t[idx_last] = 8888.0
        after = np.asarray(env2._get_obs(), dtype=np.float64)
        assert not np.allclose(before, after), f"下标 {idx_last} 应可见"
        assert idx_last in _snapshot_visible_indices(build_snapshot(env2), t)


@pytest.mark.parametrize("cutoff", [2, 4])
def test_future_truth_mutation_does_not_change_obs_or_snapshot(cutoff):
    env = _env(cutoff, t=1)
    obs_before = np.asarray(env._get_obs(), dtype=np.float64)
    snap_before = build_snapshot(env).forecast

    idx = 1 + cutoff  # 首个不可见下标
    env.price_t[idx] = 9999.0
    env.pv_t[idx] = 9999.0
    env.wt_t[idx] = 9999.0
    env.T_amb[idx] = 9999.0
    env.carbon_factor_t[idx] = 9999.0
    env.true_task_arrival_profile[idx] = 9999.0

    assert np.allclose(obs_before, np.asarray(env._get_obs(), dtype=np.float64))
    snap_after = build_snapshot(env).forecast
    assert snap_before.price_forecast == snap_after.price_forecast
    assert snap_before.pv_forecast == snap_after.pv_forecast
    assert snap_before.wind_forecast == snap_after.wind_forecast
    assert snap_before.temperature_forecast == snap_after.temperature_forecast


@pytest.mark.parametrize("cutoff", [2, 4])
def test_tail_window_fixed_length_zero_padded(cutoff):
    """horizon 尾部：窗口长度固定为 cutoff，超出 horizon 的部分零填充。"""
    env = _env(cutoff, t=HORIZON - 1)  # 最后一步
    snap = build_snapshot(env)
    window = snap.forecast.price_forecast
    assert len(window) == cutoff
    start, end = _expected_window(HORIZON - 1, cutoff)
    visible_count = end - start
    assert visible_count == 1  # 尾部仅剩 1 个可见点
    assert sum(1 for v in window if v != 0.0) == visible_count
    assert all(v == 0.0 for v in window[visible_count:])  # 缺失部分显式零填充
    assert end == HORIZON


def test_shared_window_helper_exists_and_matches_definition():
    from envs.idc_price_env import visible_window_slice

    for t, cutoff in [(0, 2), (1, 4), (10, 3), (23, 4)]:
        start, end = visible_window_slice(t, cutoff, HORIZON)
        assert (start, end) == _expected_window(t, cutoff)


@pytest.mark.parametrize("cutoff", [2, 4])
def test_no_future_leak_for_all_series(cutoff):
    """price/pv/wind/temperature/carbon/任务到达 的未来真值均不得进入 observation。"""
    env = _env(cutoff, t=2)
    obs_before = np.asarray(env._get_obs(), dtype=np.float64)
    for idx in range(2 + cutoff, HORIZON):
        env.price_t[idx] = 9999.0
        env.pv_t[idx] = 9999.0
        env.wt_t[idx] = 9999.0
        env.T_amb[idx] = 9999.0
        env.carbon_factor_t[idx] = 9999.0
        env.true_task_arrival_profile[idx] = 9999.0
    assert np.allclose(obs_before, np.asarray(env._get_obs(), dtype=np.float64))

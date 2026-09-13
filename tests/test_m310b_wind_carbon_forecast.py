"""M3.10b 测试：wind/carbon 可见预测纳入 observation 与 ScenarioBundle。"""

import numpy as np
import pytest
import torch

from checkpointing import CURRENT_CONTRACT_VERSION, CheckpointVersionError, VersionedCheckpoint
from contracts.models import ScenarioBundle
from envs.idc_price_env import IDCPriceEnv20D, visible_window_slice
from planning.snapshot_adapter import build_snapshot

HORIZON = 24
N_GROUPS = 20
CURRENT_OBS_DIM = 6 + 10 + 6 * N_GROUPS  # 136
FORECAST_GROUPS = 8
EXPECTED_OBS_DIM = CURRENT_OBS_DIM + FORECAST_GROUPS * HORIZON  # 328
FEATURE_ORDER = ("price", "temperature", "arrival", "pv", "wind", "carbon", "sin", "cos")


def _env(cutoff: int = 4, t: int = 0) -> IDCPriceEnv20D:
    h = HORIZON
    env = IDCPriceEnv20D(
        horizon=h,
        forecast_cutoff=cutoff,
        access_limit_kw=1000.0,
        price_t=(np.arange(h) + 1.0) * 0.1,
        pv_t=np.arange(h) + 1.0,
        wt_t=np.arange(h) + 1.0,
        carbon_factor_t=(np.arange(h) + 1.0) * 0.1,
        T_amb=np.arange(h, dtype=np.float64) + 20.0,
    )
    env.reset(seed=0)
    action = np.concatenate([np.full(20, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)])
    for _ in range(t):
        env.step(action)
    return env


def _group(env, name: str) -> np.ndarray:
    """按固定顺序取出某个预测特征组（长度 horizon）。"""
    obs = np.asarray(env._get_obs(), dtype=np.float64)
    block = obs[env.current_obs_dim :]
    idx = FEATURE_ORDER.index(name)
    return block[idx * env.horizon : (idx + 1) * env.horizon]


# --- 1. 维度与顺序 ---

def test_observation_dimension_updated():
    env = _env()
    assert env.forecast_feature_groups == FORECAST_GROUPS
    assert env.obs_dim == EXPECTED_OBS_DIM
    assert env.observation_space.shape == (EXPECTED_OBS_DIM,)
    assert env._get_obs().shape == (EXPECTED_OBS_DIM,)


def test_feature_order_documented_and_stable():
    env = _env(t=3)
    # wind 组应等于归一化后的 wt_t（窗口内），carbon 组应等于归一化后的 carbon_factor_t
    start, end = visible_window_slice(3, env.forecast_cutoff, env.horizon)
    wind = _group(env, "wind")
    expected_wind = np.asarray(env.wt_t, dtype=np.float64) / env.wind_ref_kw
    expected_wind = np.clip(expected_wind[start:end], -1.5, 1.5)
    assert np.allclose(wind[start:end], expected_wind)
    assert np.all(wind[:start] == 0.0)


def test_wind_carbon_refs_are_declared_and_frozen():
    import json
    from pathlib import Path

    refs = json.loads(Path("configs/frozen_refs/refs.json").read_text(encoding="utf-8"))
    for key in ("wind_ref_kw", "carbon_factor_ref"):
        assert key in refs["references"], f"冻结参考值缺少 {key}"
    assert "units" in refs, "冻结参考值需带单位"
    assert "wind_ref_kw" in refs["units"] and "carbon_factor_ref" in refs["units"]


# --- 2. ScenarioBundle carbon_forecast ---

def test_scenario_bundle_has_carbon_forecast():
    env = _env(cutoff=2)
    snap = build_snapshot(env)
    assert hasattr(snap.forecast, "carbon_forecast")
    assert len(snap.forecast.carbon_forecast) == 2


def test_snapshot_wind_carbon_share_window_and_padding():
    env = _env(cutoff=4, t=HORIZON - 1)  # 尾部
    snap = build_snapshot(env)
    for field in ("wind_forecast", "carbon_forecast", "price_forecast", "pv_forecast"):
        window = getattr(snap.forecast, field)
        assert len(window) == 4
        assert sum(1 for v in window if v != 0.0) == 1  # 尾部仅 1 个可见点
        assert window[1:] == [0.0, 0.0, 0.0]


# --- 3. 窗口内变更影响 / 窗口外不影响 ---

@pytest.mark.parametrize("cutoff", [2, 4])
def test_in_window_wind_carbon_change_affects_obs_and_snapshot(cutoff):
    env = _env(cutoff, t=1)
    before_obs = np.asarray(env._get_obs(), dtype=np.float64).copy()
    before_snap = build_snapshot(env).forecast

    idx = 1 + cutoff - 1  # 窗口内最后一个点
    env.wt_t[idx] = 777.0
    env.carbon_factor_t[idx] = 7.7

    after_obs = np.asarray(env._get_obs(), dtype=np.float64)
    assert not np.allclose(before_obs, after_obs)
    after_snap = build_snapshot(env).forecast
    assert before_snap.wind_forecast != after_snap.wind_forecast
    assert before_snap.carbon_forecast != after_snap.carbon_forecast


@pytest.mark.parametrize("cutoff", [2, 4])
def test_out_of_window_wind_carbon_change_ignored(cutoff):
    env = _env(cutoff, t=1)
    before_obs = np.asarray(env._get_obs(), dtype=np.float64).copy()
    before_snap = build_snapshot(env).forecast

    for idx in range(1 + cutoff, HORIZON):
        env.wt_t[idx] = 777.0
        env.carbon_factor_t[idx] = 7.7

    assert np.allclose(before_obs, np.asarray(env._get_obs(), dtype=np.float64))
    after_snap = build_snapshot(env).forecast
    assert before_snap.wind_forecast == after_snap.wind_forecast
    assert before_snap.carbon_forecast == after_snap.carbon_forecast


# --- 4. 契约版本与旧 schema 拒绝 ---

def test_contract_version_bumped():
    assert CURRENT_CONTRACT_VERSION == "contract-v2"


def test_old_and_unversioned_checkpoints_rejected(tmp_path):
    # 旧版本
    old = tmp_path / "old.pt"
    torch.save(
        {
            "metadata": {
                "contract_version_id": "contract-v1",
                "action_dim": 21,
                "obs_dim": 280,
                "schema_hash": "h",
                "code_revision": "r",
            },
            "state": {},
        },
        str(old),
    )
    with pytest.raises(CheckpointVersionError):
        VersionedCheckpoint.load(
            old, expected_action_dim=21, expected_obs_dim=EXPECTED_OBS_DIM, expected_schema_hash="h"
        )

    # 无版本
    noversion = tmp_path / "noversion.pt"
    torch.save({"state": {}}, str(noversion))
    with pytest.raises(CheckpointVersionError, match="无版本"):
        VersionedCheckpoint.load(
            noversion,
            expected_action_dim=21,
            expected_obs_dim=EXPECTED_OBS_DIM,
            expected_schema_hash="h",
        )


def test_old_obs_dim_not_mixable(tmp_path):
    """旧 obs_dim(280) checkpoint 不得与新 obs_dim(328) 混用。"""
    ckpt = VersionedCheckpoint(
        contract_version_id=CURRENT_CONTRACT_VERSION,
        action_dim=21,
        obs_dim=280,
        schema_hash="h",
        code_revision="r",
        state={},
    )
    path = tmp_path / "oldshape.pt"
    ckpt.save(path)
    with pytest.raises(CheckpointVersionError, match="obs_dim"):
        VersionedCheckpoint.load(
            path,
            expected_action_dim=21,
            expected_obs_dim=EXPECTED_OBS_DIM,
            expected_schema_hash="h",
        )


def test_scenario_contract_schema_version_bumped():
    s = ScenarioBundle(
        split="train",
        start="0",
        horizon=24,
        forecast_cutoff=2,
        price_forecast=[0.1, 0.2],
        load_forecast=[0.0, 0.0],
        pv_forecast=[1.0, 2.0],
        wind_forecast=[1.0, 2.0],
        temperature_forecast=[20.0, 21.0],
        carbon_forecast=[0.5, 0.6],
        source_hashes={"a": "b"},
    )
    assert s.schema_version == "contract-v2"

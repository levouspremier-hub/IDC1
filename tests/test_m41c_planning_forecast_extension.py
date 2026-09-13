"""M4.1c 测试：规划时域外生量展开（窗口外保守/持久化假设，绝不读未来真值）。"""

import numpy as np
import pytest
import torch

from checkpointing import CURRENT_CONTRACT_VERSION, CheckpointVersionError, VersionedCheckpoint
from contracts import CONTRACT_VERSION_ID
from contracts.models import PlanningExogenousForecast, SystemSnapshot
from contracts.validators import validate_snapshot
from envs.idc_price_env import IDCPriceEnv20D, visible_window_slice
from planning.snapshot_adapter import build_snapshot

HORIZON = 24
CUTOFF = 4
N_GROUP = 20
EXPECTED_OBS_DIM = 6 + 10 + 6 * N_GROUP + 8 * HORIZON  # 328
VECTORS = (
    "price", "pv", "wind", "temperature", "carbon", "arrival", "base_idc_power",
)


def _env(cutoff: int = CUTOFF, t: int = 0, horizon: int = HORIZON) -> IDCPriceEnv20D:
    h = horizon
    # 真值随时间线性变化，便于区分「窗口内真值」与「窗口外持久化」
    env = IDCPriceEnv20D(
        horizon=h, forecast_cutoff=cutoff, access_limit_kw=1000.0,
        price_t=0.1 + 0.01 * np.arange(h),
        pv_t=1.0 + 0.1 * np.arange(h),
        wt_t=2.0 + 0.1 * np.arange(h),
        carbon_factor_t=0.5 + 0.01 * np.arange(h),
        T_amb=20.0 + 0.5 * np.arange(h),
    )
    env.reset(seed=0)
    action = np.concatenate(
        [np.full(N_GROUP, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)]
    )
    for _ in range(t):
        env.step(action)
    return env


# --- 1. 长度与来源 ---

@pytest.mark.parametrize("t", [0, 3, 10])
def test_all_planning_vectors_length_equal_horizon(t):
    snap = build_snapshot(_env(t=t))
    pf = snap.planning_forecast
    assert pf.horizon_steps == snap.planning_horizon_steps
    for name in VECTORS:
        assert len(getattr(pf, name)) == pf.horizon_steps, name
    assert len(pf.visible_mask) == pf.horizon_steps
    assert len(pf.assumed_mask) == pf.horizon_steps
    assert pf.extension_policy


def test_units_declared_for_planning_forecast():
    for name in VECTORS:
        assert name in PlanningExogenousForecast.UNITS, name
    assert PlanningExogenousForecast.UNITS["base_idc_power"] == "kW"
    assert PlanningExogenousForecast.UNITS["arrival"] == "work-units/step"


# --- 2. 窗口内与可见 forecast 一致 ---

def test_in_window_values_match_visible_forecast():
    env = _env(t=2)
    snap = build_snapshot(env)
    pf = snap.planning_forecast
    t = env.current_step
    start, end = visible_window_slice(t, CUTOFF, env.horizon)
    n_visible = end - start
    for k in range(n_visible):
        assert pf.visible_mask[k] is True
        assert pf.assumed_mask[k] is False
        assert pf.price[k] == pytest.approx(snap.forecast.price_forecast[k])
        assert pf.pv[k] == pytest.approx(snap.forecast.pv_forecast[k])
        assert pf.wind[k] == pytest.approx(snap.forecast.wind_forecast[k])
        assert pf.temperature[k] == pytest.approx(snap.forecast.temperature_forecast[k])
        assert pf.carbon[k] == pytest.approx(snap.forecast.carbon_forecast[k])
        assert pf.arrival[k] == pytest.approx(snap.forecast.arrival_forecast[k])


# --- 3. 窗口外规则 ---

def test_out_of_window_follows_declared_rules():
    env = _env(t=2)
    snap = build_snapshot(env)
    pf = snap.planning_forecast
    t = env.current_step
    start, end = visible_window_slice(t, CUTOFF, env.horizon)
    n_visible = end - start
    last_visible_price = pf.price[n_visible - 1]
    last_visible_carbon = pf.carbon[n_visible - 1]
    last_visible_temp = pf.temperature[n_visible - 1]

    for k in range(n_visible, pf.horizon_steps):
        assert pf.visible_mask[k] is False
        assert pf.assumed_mask[k] is True
        # PV / wind / arrival 保守为 0
        assert pf.pv[k] == 0.0
        assert pf.wind[k] == 0.0
        assert pf.arrival[k] == 0.0
        # price / carbon / temperature 持久化
        assert pf.price[k] == pytest.approx(last_visible_price)
        assert pf.carbon[k] == pytest.approx(last_visible_carbon)
        assert pf.temperature[k] == pytest.approx(last_visible_temp)


def test_base_idc_power_uses_persisted_temperature():
    env = _env(t=2)
    snap = build_snapshot(env)
    pf = snap.planning_forecast
    base_load = np.clip(env.base_load, 0.0, 1.0)
    for k in range(pf.horizon_steps):
        expected = env._idc_power_kw(base_load, pf.temperature[k])
        assert pf.base_idc_power[k] == pytest.approx(expected, rel=1e-9)


def test_base_idc_power_matches_snapshot_field():
    """规划契约的 base_idc_power 必须与 SystemSnapshot 既有字段一致（防静默分叉）。"""
    snap = build_snapshot(_env(t=3))
    assert snap.planning_forecast.base_idc_power == pytest.approx(
        snap.base_idc_power_forecast_kw
    )


# --- 4. 不读未来真值 ---

@pytest.mark.leakage
@pytest.mark.parametrize("cutoff", [2, 4])
def test_out_of_window_truth_mutation_does_not_change_planning(cutoff):
    env = _env(cutoff=cutoff, t=1)
    before = build_snapshot(env).planning_forecast
    t = env.current_step
    for idx in range(t + cutoff, HORIZON):
        env.price_t[idx] = 9999.0
        env.pv_t[idx] = 9999.0
        env.wt_t[idx] = 9999.0
        env.T_amb[idx] = 9999.0
        env.carbon_factor_t[idx] = 9999.0
        env.true_task_arrival_profile[idx] = 9999.0
    after = build_snapshot(env).planning_forecast
    assert before.model_dump() == after.model_dump()


def test_arrival_out_of_window_is_zero_not_truth():
    env = _env(t=1)
    snap = build_snapshot(env)
    pf = snap.planning_forecast
    n_visible = min(CUTOFF, env.horizon - env.current_step)
    for k in range(n_visible, pf.horizon_steps):
        assert pf.arrival[k] == 0.0, "未来到达量不得凭真值进入规划"


# --- 5. mask / 尾部截断 / cutoff=0 ---

def test_masks_are_complementary_and_counted():
    env = _env(t=18)  # 接近尾部
    snap = build_snapshot(env)
    pf = snap.planning_forecast
    assert all(v != a for v, a in zip(pf.visible_mask, pf.assumed_mask, strict=True))
    expected_visible = min(CUTOFF, env.horizon - env.current_step)
    assert sum(pf.visible_mask) == expected_visible


def test_tail_truncated_to_planning_horizon():
    env = _env(t=HORIZON - 1)  # 只剩 1 步
    snap = build_snapshot(env)
    assert snap.planning_horizon_steps == 1
    assert snap.planning_forecast.horizon_steps == 1
    for name in VECTORS:
        assert len(getattr(snap.planning_forecast, name)) == 1
    assert snap.planning_forecast.visible_mask == [True]
    assert snap.planning_forecast.assumed_mask == [False]


@pytest.mark.parametrize("cutoff", [0, -1])
def test_zero_cutoff_explicitly_rejected(cutoff):
    env = _env()
    env.forecast_cutoff = cutoff
    with pytest.raises(ValueError, match="forecast_cutoff"):
        build_snapshot(env)


# --- 6. validator ---

def test_adapter_output_passes_validator():
    validate_snapshot(build_snapshot(_env(t=5)))


def test_validator_rejects_planning_length_mismatch():
    snap = build_snapshot(_env())
    bad = snap.planning_forecast.model_copy(update={"price": [0.2, 0.3]})
    kwargs = snap.model_dump()
    kwargs["planning_forecast"] = bad
    with pytest.raises(ValueError, match="planning_forecast"):
        validate_snapshot(SystemSnapshot(**kwargs))


def test_validator_rejects_assumed_pv_nonzero():
    snap = build_snapshot(_env(t=1))
    pf = snap.planning_forecast
    n_visible = min(CUTOFF, 24 - 1)
    pv = list(pf.pv)
    pv[n_visible] = 5.0  # 假设段 PV 必须为 0
    kwargs = snap.model_dump()
    kwargs["planning_forecast"] = pf.model_copy(update={"pv": pv})
    with pytest.raises(ValueError, match="pv"):
        validate_snapshot(SystemSnapshot(**kwargs))


# --- 7. contract-v5 与旧版本拒绝 ---

def test_contract_version_is_v5():
    assert CONTRACT_VERSION_ID == "contract-v5"
    assert CURRENT_CONTRACT_VERSION == CONTRACT_VERSION_ID


@pytest.mark.parametrize("version", ["contract-v4", "contract-v3", "contract-v2", "contract-v1"])
def test_old_versions_rejected(tmp_path, version):
    path = tmp_path / "c.pt"
    torch.save(
        {"metadata": {"contract_version_id": version, "action_dim": 21,
                      "obs_dim": EXPECTED_OBS_DIM, "schema_hash": "h", "code_revision": "r"},
         "state": {}},
        str(path),
    )
    with pytest.raises(CheckpointVersionError, match="contract_version_id"):
        VersionedCheckpoint.load(
            path,
            expected_action_dim=21,
            expected_obs_dim=EXPECTED_OBS_DIM,
            expected_schema_hash="h",
        )


def test_unversioned_and_legacy_payload_rejected(tmp_path):
    noversion = tmp_path / "n.pt"
    torch.save({"state": {}}, str(noversion))
    with pytest.raises(CheckpointVersionError, match="无版本"):
        VersionedCheckpoint.load(
            noversion,
            expected_action_dim=21,
            expected_obs_dim=EXPECTED_OBS_DIM,
            expected_schema_hash="h",
        )

    legacy = tmp_path / "l.pt"
    torch.save(
        {"metadata": {"contract_version_id": "contract-v4", "action_dim": 21,
                      "obs_dim": EXPECTED_OBS_DIM, "schema_hash": "h", "code_revision": "r"},
         "state": {"group_work_capacity": [1.0] * N_GROUP}},
        str(legacy),
    )
    with pytest.raises(CheckpointVersionError):
        VersionedCheckpoint.load(
            legacy,
            expected_action_dim=21,
            expected_obs_dim=EXPECTED_OBS_DIM,
            expected_schema_hash="h",
        )

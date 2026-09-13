"""M4.1a 测试：SystemSnapshot 作为滚动 24h 联合可行性规划的完整输入契约。"""

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from checkpointing import CURRENT_CONTRACT_VERSION, CheckpointVersionError, VersionedCheckpoint
from contracts import CONTRACT_VERSION_ID
from contracts.models import ScenarioBundle, SystemSnapshot, TaskState
from contracts.validators import validate_snapshot
from envs.idc_price_env import IDCPriceEnv20D, visible_window_slice
from planning.snapshot_adapter import POWER_APPROXIMATION_NOTE, build_snapshot

HORIZON = 24
CUTOFF = 4
EXPECTED_OBS_DIM = 6 + 10 + 6 * 20 + 8 * HORIZON  # 328

TIME_FIELDS = ("step", "delta_t_hours", "planning_horizon_steps")
BESS_FIELDS = (
    "soc_kwh", "soc_min_kwh", "soc_max_kwh", "soc_capacity_kwh",
    "bess_charge_power_max_kw", "bess_discharge_power_max_kw",
    "bess_charge_efficiency", "bess_discharge_efficiency",
    "bess_degradation_cost_per_kwh",
)
ACCESS_ONLY_FIELDS = ("access_limit_kw", "base_idc_power_forecast_kw")
GROUP_FIELDS = ("group_work_capacity", "group_power_coeff_kw_per_work", "group_power_upper_kw")


def _env(cutoff: int = CUTOFF, t: int = 0, horizon: int = HORIZON) -> IDCPriceEnv20D:
    h = horizon
    env = IDCPriceEnv20D(
        horizon=h,
        forecast_cutoff=cutoff,
        access_limit_kw=1000.0,
        price_t=(np.arange(h) + 1.0) * 0.1,
        pv_t=np.linspace(0.1, 1.0, h),
        wt_t=np.linspace(0.1, 1.0, h),
        carbon_factor_t=np.linspace(0.01, 0.1, h),
        T_amb=np.linspace(20.0, 30.0, h),
    )
    env.reset(seed=0)
    action = np.concatenate([np.full(20, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)])
    for _ in range(t):
        env.step(action)
    return env


# --- 1. 字段完整性与单位 ---

def test_all_planning_fields_present():
    snap = build_snapshot(_env())
    for field in TIME_FIELDS + BESS_FIELDS + ACCESS_ONLY_FIELDS + GROUP_FIELDS:
        assert hasattr(snap, field), f"缺少规划字段 {field}"


def test_units_declared_for_new_fields():
    for field in TIME_FIELDS + BESS_FIELDS + ACCESS_ONLY_FIELDS + GROUP_FIELDS:
        assert field in SystemSnapshot.UNITS, f"{field} 未声明单位"
    assert SystemSnapshot.UNITS["delta_t_hours"] == "h"
    assert SystemSnapshot.UNITS["bess_charge_efficiency"] == "fraction"
    assert SystemSnapshot.UNITS["group_work_capacity"] == "work-units"
    assert SystemSnapshot.UNITS["group_power_coeff_kw_per_work"] == "kW/work-unit"
    assert SystemSnapshot.UNITS["group_power_upper_kw"] == "kW"


def test_power_approximation_note_present():
    snap = build_snapshot(_env())
    assert snap.power_approximation_note == POWER_APPROXIMATION_NOTE
    assert "规划近似" in snap.power_approximation_note
    assert "复核" in snap.power_approximation_note


def test_task_state_has_max_rate_and_units():
    assert "max_rate_work_per_step" in TaskState.model_fields
    assert TaskState.UNITS["max_rate_work_per_step"] == "work-units/step"


# --- 2. validator 显式拒绝 ---

def _valid_snapshot_kwargs() -> dict:
    env = _env()
    snap = build_snapshot(env)
    return snap.model_dump()


def test_validator_accepts_adapter_output():
    validate_snapshot(build_snapshot(_env()))


@pytest.mark.parametrize(
    "update, match",
    [
        ({"bess_charge_efficiency": -0.1}, "efficiency"),
        ({"bess_discharge_efficiency": 1.5}, "efficiency"),
        ({"soc_kwh": 1e9}, "soc_kwh"),
        ({"bess_charge_power_max_kw": -5.0}, "power"),
        ({"bess_discharge_power_max_kw": -5.0}, "power"),
        ({"group_power_coeff_kw_per_work": [-1.0] * 20}, "coeff"),
    ],
)
def test_validator_rejects_bad_snapshot(update, match):
    kwargs = _valid_snapshot_kwargs()
    kwargs.update(update)
    snap = SystemSnapshot(**kwargs)
    with pytest.raises(ValueError, match=match):
        validate_snapshot(snap)


def test_validator_rejects_missing_field():
    kwargs = {k: v for k, v in _valid_snapshot_kwargs().items() if k != "delta_t_hours"}
    with pytest.raises(ValidationError):
        SystemSnapshot(**kwargs)


# --- 3. 基础负载预测与功耗模型一致 ---

def test_base_idc_power_forecast_matches_model():
    env = _env(t=3)
    snap = build_snapshot(env)
    t = env.current_step
    for k in range(CUTOFF):
        expected = env._idc_power_kw(np.clip(env.base_load, 0.0, 1.0), env.T_amb[t + k])
        assert snap.base_idc_power_forecast_kw[k] == pytest.approx(expected, rel=1e-9)


def test_base_idc_forecast_length_is_planning_horizon():
    env = _env(t=3)
    snap = build_snapshot(env)
    assert len(snap.base_idc_power_forecast_kw) == snap.planning_horizon_steps
    assert snap.planning_horizon_steps == min(24, env.horizon - env.current_step)


# --- 4. 无未来真值泄漏 ---

@pytest.mark.leakage
def test_future_series_mutation_does_not_change_snapshot():
    env = _env(t=1)
    before = build_snapshot(env)
    t = env.current_step
    for idx in range(t + CUTOFF, HORIZON):
        env.price_t[idx] = 9999.0
        env.pv_t[idx] = 9999.0
        env.wt_t[idx] = 9999.0
        env.T_amb[idx] = 9999.0
        env.carbon_factor_t[idx] = 9999.0
        env.true_task_arrival_profile[idx] = 9999.0
    after = build_snapshot(env)
    assert before.forecast.model_dump() == after.forecast.model_dump()
    assert before.base_idc_power_forecast_kw == after.base_idc_power_forecast_kw


@pytest.mark.leakage
def test_future_real_tasks_not_read():
    """未来真实任务（status=not_arrived）不得进入 snapshot.tasks。"""
    env = _env(t=1)
    snap = build_snapshot(env)
    for task in snap.tasks:
        assert task.status != "not_arrived"
    assert len(snap.tasks) == sum(1 for t in env.tasks if t.status != "not_arrived")


def test_forecast_uses_visible_window_and_padding():
    env = _env(cutoff=CUTOFF, t=2)
    snap = build_snapshot(env)
    start, end = visible_window_slice(env.current_step, CUTOFF, env.horizon)
    assert (start, end) == (2, 6)
    for field in ("price_forecast", "arrival_forecast"):
        assert len(getattr(snap.forecast, field)) == CUTOFF


# --- 5. active task 的 max-rate 与环境定义一致 ---

def test_active_task_max_rate_matches_env_definition():
    env = _env(t=0)
    snap = build_snapshot(env)
    by_id = {str(t.task_id): t for t in env.tasks if t.status != "not_arrived"}
    for task in snap.tasks:
        src = by_id[task.task_id]
        expected = float(src.workload / max(int(src.duration), 1))
        assert task.max_rate_work_per_step == pytest.approx(expected)
        assert task.remaining_work == pytest.approx(float(src.remaining_work))
        assert task.deadline == int(src.latest_finish_time)
        assert task.priority == pytest.approx(float(src.priority))


# --- 6. 契约版本 v3 与旧 checkpoint 拒绝 ---

def test_contract_version_is_v4():
    assert CONTRACT_VERSION_ID == "contract-v4"
    assert CURRENT_CONTRACT_VERSION == CONTRACT_VERSION_ID


@pytest.mark.parametrize("version", ["contract-v2", "contract-v1"])
def test_old_version_checkpoints_rejected(tmp_path, version):
    path = tmp_path / "c.pt"
    torch.save(
        {"metadata": {"contract_version_id": version, "action_dim": 21, "obs_dim": 328,
                      "schema_hash": "h", "code_revision": "r"}, "state": {}},
        str(path),
    )
    with pytest.raises(CheckpointVersionError, match="contract_version_id"):
        VersionedCheckpoint.load(
            path, expected_action_dim=21, expected_obs_dim=328, expected_schema_hash="h"
        )


def test_unversioned_checkpoint_rejected(tmp_path):
    path = tmp_path / "n.pt"
    torch.save({"state": {}}, str(path))
    with pytest.raises(CheckpointVersionError, match="无版本"):
        VersionedCheckpoint.load(
            path,
            expected_action_dim=21,
            expected_obs_dim=EXPECTED_OBS_DIM,
            expected_schema_hash="h",
        )


def test_old_obs_dim_rejected(tmp_path):
    path = tmp_path / "o.pt"
    torch.save(
        {"metadata": {"contract_version_id": CONTRACT_VERSION_ID, "action_dim": 21,
                      "obs_dim": 280, "schema_hash": "h", "code_revision": "r"}, "state": {}},
        str(path),
    )
    with pytest.raises(CheckpointVersionError, match="obs_dim"):
        VersionedCheckpoint.load(
            path, expected_action_dim=21, expected_obs_dim=328, expected_schema_hash="h"
        )


def test_scenario_bundle_has_arrival_forecast():
    b = build_snapshot(_env()).forecast
    assert isinstance(b, ScenarioBundle)
    assert len(b.arrival_forecast) == CUTOFF

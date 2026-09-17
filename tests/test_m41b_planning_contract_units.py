"""M4.1b 测试：规划契约单位加固（work/kW 消歧 + 结构不变量）。"""

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from checkpointing import CURRENT_CONTRACT_VERSION, CheckpointVersionError, VersionedCheckpoint
from contracts import CONTRACT_VERSION_ID
from contracts.models import SystemSnapshot
from contracts.validators import validate_snapshot
from envs.idc_price_env import IDCPriceEnv20D
from planning.snapshot_adapter import build_snapshot

HORIZON = 24
CUTOFF = 4
N_GROUP = 20
EXPECTED_OBS_DIM = 6 + 10 + 6 * N_GROUP + 8 * HORIZON  # 328


def _env(t: int = 0) -> IDCPriceEnv20D:
    h = HORIZON
    env = IDCPriceEnv20D(
        horizon=h, forecast_cutoff=CUTOFF, access_limit_kw=1000.0,
        T_amb=np.linspace(20.0, 30.0, h),
    )
    env.reset(seed=0)
    action = np.concatenate(
        [np.full(N_GROUP, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)]
    )
    for _ in range(t):
        env.step(action)
    return env


def _kwargs() -> dict:
    return build_snapshot(_env()).model_dump()


# --- 1. 字段迁移：无旧字段、无 alias ---

def test_group_work_capacity_field_exists_with_unit():
    assert "group_work_capacity" in SystemSnapshot.model_fields
    assert SystemSnapshot.UNITS["group_work_capacity"] == "work-units"


def test_legacy_group_capacity_kw_removed():
    assert "group_capacity_kw" not in SystemSnapshot.model_fields
    assert "group_capacity_kw" not in SystemSnapshot.UNITS


def test_legacy_field_cannot_construct_v4_snapshot():
    with pytest.raises(ValidationError):
        SystemSnapshot(**_kwargs(), group_capacity_kw=[1.0] * N_GROUP)


def test_no_alias_or_dual_field():
    """不得保留同值双字段：两个名字不能同时存在于模型中。"""
    names = set(SystemSnapshot.model_fields)
    assert ("group_work_capacity" in names) ^ ("group_capacity_kw" in names)


def test_adapter_populates_new_field_with_env_capability():
    env = _env()
    snap = build_snapshot(env)
    assert snap.group_work_capacity == pytest.approx(
        [float(c) for c in np.asarray(env.model.C_server, dtype=np.float64)]
    )


# --- 2. 规划近似字段保留且语义明确 ---

def test_power_approximation_fields_preserved():
    snap = build_snapshot(_env())
    assert len(snap.group_power_coeff_kw_per_work) == N_GROUP
    assert len(snap.group_power_upper_kw) == N_GROUP
    assert SystemSnapshot.UNITS["group_power_coeff_kw_per_work"] == "kW/work-unit"
    assert SystemSnapshot.UNITS["group_power_upper_kw"] == "kW"
    assert "规划近似" in snap.power_approximation_note
    assert "复核" in snap.power_approximation_note


# --- 3. 新增结构不变量 ---

def test_adapter_output_passes_validator():
    validate_snapshot(build_snapshot(_env()))


@pytest.mark.parametrize(
    "update, match",
    [
        ({"delta_t_hours": 0.0}, "delta_t_hours"),
        ({"delta_t_hours": -1.0}, "delta_t_hours"),
        ({"planning_horizon_steps": 0}, "planning_horizon_steps"),
        ({"planning_horizon_steps": -3}, "planning_horizon_steps"),
        ({"power_approximation_note": ""}, "power_approximation_note"),
        ({"soc_max_kwh": 1e9}, "soc"),
        ({"soc_min_kwh": -1.0}, "soc"),
    ],
)
def test_validator_rejects_bad_scalars(update, match):
    kwargs = _kwargs()
    kwargs.update(update)
    with pytest.raises(ValueError, match=match):
        validate_snapshot(SystemSnapshot(**kwargs))


def test_soc_max_must_not_exceed_capacity():
    kwargs = _kwargs()
    kwargs["soc_capacity_kwh"] = 50.0
    kwargs["soc_max_kwh"] = 90.0
    kwargs["soc_kwh"] = 50.0
    with pytest.raises(ValueError, match="soc"):
        validate_snapshot(SystemSnapshot(**kwargs))


@pytest.mark.parametrize(
    "field, value",
    [
        ("bess_charge_efficiency", 0.0),
        ("bess_charge_efficiency", -0.1),
        ("bess_charge_efficiency", 1.5),
        ("bess_discharge_efficiency", 0.0),
        ("bess_discharge_efficiency", 2.0),
    ],
)
def test_validator_rejects_illegal_efficiency(field, value):
    kwargs = _kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match="efficiency"):
        validate_snapshot(SystemSnapshot(**kwargs))


@pytest.mark.parametrize(
    "field, value",
    [
        ("group_work_capacity", [1.0] * 3),
        ("group_power_coeff_kw_per_work", [0.01] * 3),
        ("group_power_upper_kw", [0.7] * 3),
    ],
)
def test_three_group_vectors_must_match_length(field, value):
    kwargs = _kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match="长度"):
        validate_snapshot(SystemSnapshot(**kwargs))


def test_base_load_forecast_length_must_match_planning_horizon():
    kwargs = _kwargs()
    kwargs["base_idc_power_forecast_kw"] = [14.0] * 3
    with pytest.raises(ValueError, match="base_idc_power_forecast_kw"):
        validate_snapshot(SystemSnapshot(**kwargs))


def test_active_task_negative_values_rejected():
    kwargs = _kwargs()
    task = dict(kwargs["tasks"][0])
    task["max_rate_work_per_step"] = -1.0
    kwargs["tasks"] = [task]
    with pytest.raises(ValueError, match="max_rate"):
        validate_snapshot(SystemSnapshot(**kwargs))

    kwargs2 = _kwargs()
    task2 = dict(kwargs2["tasks"][0])
    task2["remaining_work"] = -5.0
    kwargs2["tasks"] = [task2]
    with pytest.raises(ValueError, match="remaining_work"):
        validate_snapshot(SystemSnapshot(**kwargs2))


# --- 4. contract-v4 与旧 payload 拒绝 ---

def test_contract_version_is_the_current_single_source():
    assert CONTRACT_VERSION_ID == "contract-v9"
    assert CONTRACT_VERSION_ID != "contract-v7"
    assert CURRENT_CONTRACT_VERSION == CONTRACT_VERSION_ID


@pytest.mark.parametrize("version", ["contract-v3", "contract-v2", "contract-v1"])
def test_old_versions_rejected(tmp_path, version):
    path = tmp_path / "c.pt"
    torch.save(
        {"metadata": {"contract_version_id": version, "action_dim": 21, "obs_dim": EXPECTED_OBS_DIM,
                      "schema_hash": "h", "code_revision": "r"}, "state": {}},
        str(path),
    )
    with pytest.raises(CheckpointVersionError, match="contract_version_id"):
        VersionedCheckpoint.load(
            path,
            expected_action_dim=21,
            expected_obs_dim=EXPECTED_OBS_DIM,
            expected_schema_hash="h",
        )


def test_unversioned_rejected(tmp_path):
    path = tmp_path / "n.pt"
    torch.save({"state": {}}, str(path))
    with pytest.raises(CheckpointVersionError, match="无版本"):
        VersionedCheckpoint.load(
            path,
            expected_action_dim=21,
            expected_obs_dim=EXPECTED_OBS_DIM,
            expected_schema_hash="h",
        )


def test_old_payload_with_legacy_field_rejected(tmp_path):
    """旧字段 payload（含 group_capacity_kw）不得被当作 v4 加载。"""
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "metadata": {"contract_version_id": "contract-v3", "action_dim": 21,
                         "obs_dim": EXPECTED_OBS_DIM, "schema_hash": "h", "code_revision": "r"},
            "state": {"group_capacity_kw": [1.0] * N_GROUP},
        },
        str(path),
    )
    with pytest.raises(CheckpointVersionError):
        VersionedCheckpoint.load(
            path,
            expected_action_dim=21,
            expected_obs_dim=EXPECTED_OBS_DIM,
            expected_schema_hash="h",
        )

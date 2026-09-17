"""M3.10c 测试：运行时冻结参考值 + 唯一契约版本源。"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from checkpointing import CURRENT_CONTRACT_VERSION, CheckpointVersionError, VersionedCheckpoint
from contracts import CONTRACT_VERSION_ID
from envs.idc_price_env import IDCPriceEnv20D
from safe_rl_v2.buffer import CONTRACT_VERSION as BUFFER_CONTRACT_VERSION

HORIZON = 24
N_GROUPS = 20
EXPECTED_OBS_DIM = 6 + 10 + 6 * N_GROUPS + 8 * HORIZON  # 328
REFS_PATH = Path("configs/frozen_refs/refs.json")
FROZEN_RUNTIME_REFS = ("pv_ref_kw", "wind_ref_kw", "carbon_factor_ref")


def _frozen_reference_value(name: str) -> float:
    """M1.3g-d：v2 refs 的逐值 entry 是**结构化对象**，标量在 `.value`。"""
    entry = json.loads(REFS_PATH.read_text(encoding="utf-8"))["references"][name]
    return float(entry["value"])


def _env(peak: float, cutoff: int = 4, t: int = 0) -> IDCPriceEnv20D:
    h = HORIZON
    env = IDCPriceEnv20D(
        horizon=h,
        forecast_cutoff=cutoff,
        access_limit_kw=1000.0,
        pv_t=np.full(h, peak),
        wt_t=np.full(h, peak),
        carbon_factor_t=np.full(h, peak / 100.0),
        # M1.3g-d：把**冻结** refs 显式传进 env，使「运行时 refs == 冻结 refs」
        # 这条断言真正在检验 env 不按序列峰值重算（而不是在检验两侧的默认值）。
        pv_ref_kw=_frozen_reference_value("pv_ref_kw"),
        wind_ref_kw=_frozen_reference_value("wind_ref_kw"),
        carbon_factor_ref=_frozen_reference_value("carbon_factor_ref"),
    )
    env.reset(seed=0)
    action = np.concatenate([np.full(20, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)])
    for _ in range(t):
        env.step(action)
    return env


# --- 1. 唯一版本源 ---

def test_contract_version_single_source():
    assert CONTRACT_VERSION_ID == "contract-v9"
    assert CONTRACT_VERSION_ID != "contract-v7"
    assert CURRENT_CONTRACT_VERSION == CONTRACT_VERSION_ID
    assert BUFFER_CONTRACT_VERSION == CONTRACT_VERSION_ID


def test_no_hardcoded_version_literal_outside_contracts():
    """checkpointing 与 buffer 不得各自硬编码版本字面量（含旧版本）。"""
    for path in ("checkpointing/versioned.py", "safe_rl_v2/buffer.py"):
        text = Path(path).read_text(encoding="utf-8")
        assert (
            f'"{CONTRACT_VERSION_ID}"' not in text
            and "'contract-v7'" not in text
            and '"contract-v7"' not in text
            and "'contract-v2'" not in text
        ), f"{path} 仍硬编码版本字面量"
        assert "CONTRACT_VERSION_ID" in text, f"{path} 未从 contracts 导入唯一版本源"


def test_checkpoint_uses_canonical_version(tmp_path):
    ckpt = VersionedCheckpoint(
        contract_version_id=CONTRACT_VERSION_ID,
        action_dim=21,
        obs_dim=EXPECTED_OBS_DIM,
        schema_hash="h",
        code_revision="r",
        state={},
    )
    path = tmp_path / "ok.pt"
    ckpt.save(path)
    loaded = VersionedCheckpoint.load(
        path, expected_action_dim=21, expected_obs_dim=EXPECTED_OBS_DIM, expected_schema_hash="h"
    )
    assert loaded.contract_version_id == CONTRACT_VERSION_ID


@pytest.mark.parametrize(
    "meta",
    [
        {"contract_version_id": "contract-v1", "action_dim": 21, "obs_dim": 280,
         "schema_hash": "h", "code_revision": "r"},
        {"contract_version_id": CONTRACT_VERSION_ID, "action_dim": 21, "obs_dim": 280,
         "schema_hash": "h", "code_revision": "r"},
    ],
)
def test_old_checkpoints_rejected(tmp_path, meta):
    path = tmp_path / "c.pt"
    torch.save({"metadata": meta, "state": {}}, str(path))
    with pytest.raises(CheckpointVersionError):
        VersionedCheckpoint.load(
            path,
            expected_action_dim=21,
            expected_obs_dim=EXPECTED_OBS_DIM,
            expected_schema_hash="h",
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


# --- 2. 冻结参考值不随序列峰值变化 ---

@pytest.mark.parametrize("ref_name", FROZEN_RUNTIME_REFS)
def test_runtime_refs_ignore_sequence_peak(ref_name):
    low = _env(peak=1.0)
    high = _env(peak=9999.0)
    assert getattr(low, ref_name) == getattr(high, ref_name), (
        f"{ref_name} 随序列峰值变化（应为冻结值）"
    )


def test_runtime_refs_match_frozen_json():
    refs = json.loads(REFS_PATH.read_text(encoding="utf-8"))
    env = _env(peak=1.0)
    for name in FROZEN_RUNTIME_REFS:
        assert name in refs["references"], f"refs.json 缺少 {name}"
        assert getattr(env, name) == pytest.approx(
            refs["references"][name]["value"]), name
        assert name in refs["units"], f"refs.json 缺少 {name} 的单位"


def test_normalization_invariant_across_sequences():
    """相同声明 reference 下，即使另一环境峰值更高，当前相同输入的归一化结果一致。"""
    low = _env(peak=1.0, t=1)
    high = _env(peak=9999.0, t=1)

    def group(env, obs, idx):
        block = obs[env.current_obs_dim :]
        return block[idx * env.horizon : (idx + 1) * env.horizon]

    # 两环境在写回相同输入后，pv/wind/carbon 归一化结果必须完全相同
    # （若 ref 曾按序列峰值重算，high 环境的归一化会被压到 1/9999 量级而不同）
    same = 0.5
    for e in (low, high):
        e.pv_t[:] = same
        e.wt_t[:] = same
        e.carbon_factor_t[:] = same / 100.0
    o1 = np.asarray(low._get_obs(), dtype=np.float64)
    o2 = np.asarray(high._get_obs(), dtype=np.float64)
    # 顺序：price(0) temperature(1) arrival(2) pv(3) wind(4) carbon(5) sin(6) cos(7)
    for idx in (3, 4, 5):
        assert np.allclose(group(low, o1, idx), group(high, o2, idx))


# --- 3. 裁剪需记录，且不改参考值 ---

def test_clipping_recorded_when_input_exceeds_ref():
    env = _env(peak=9999.0)  # 远超冻结 ref=1.0
    env.reset(seed=0)
    action = np.concatenate([np.full(20, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)])
    _, _, _, _, info = env.step(action)
    assert "forecast_clipping" in info
    clip = info["forecast_clipping"]
    assert clip["pv"] > 0 and clip["wind"] > 0 and clip["carbon"] > 0
    # 参考值不得因裁剪而改变（比对**冻结**值，而不是硬编码的旧声明值）
    for name in FROZEN_RUNTIME_REFS:
        assert getattr(env, name) == pytest.approx(_frozen_reference_value(name)), name


def test_no_clipping_when_input_within_ref():
    env = _env(peak=0.5)
    env.reset(seed=0)
    action = np.concatenate([np.full(20, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)])
    _, _, _, _, info = env.step(action)
    clip = info["forecast_clipping"]
    assert clip["pv"] == 0 and clip["wind"] == 0 and clip["carbon"] == 0


# --- 4. 参考值可审计 ---

def test_refs_auditable_in_info():
    env = _env(peak=1.0)
    env.reset(seed=0)
    action = np.concatenate([np.full(20, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)])
    _, _, _, _, info = env.step(action)
    assert "normalization_refs" in info
    refs = info["normalization_refs"]
    for name in FROZEN_RUNTIME_REFS:
        assert name in refs
    assert refs["pv_ref_kw"] == env.pv_ref_kw
    assert refs["source"] == "declared_frozen"


# --- 5. 文档修正 ---

def test_forecast_and_obs_docstrings_updated():
    forecast_doc = IDCPriceEnv20D._get_forecast_features.__doc__ or ""
    obs_doc = IDCPriceEnv20D._get_obs.__doc__ or ""
    assert "8 组" in forecast_doc or "8 * horizon" in forecast_doc
    assert "280" not in obs_doc
    assert "328" in obs_doc

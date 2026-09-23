"""M1.3g-f-c-h1：业务敏感性诊断 + 配置口径补全。

覆盖：诊断矩阵的定义（三个 origin、三档 compute、on/off、cutoff 4/48）、
最早分叉函数、`business_gap` 的**适用性**（off 时**不适用**，不是 0）、
候选 JSON 的 `forecast_cutoff` 与**预算来源**。

**本文件不跑全量诊断**（3 origin × 11 episode × 48 步、corrector on，约 10 分钟）。
"""

import importlib
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE = "scripts.diagnose_business_sensitivity"
CANDIDATE = REPO_ROOT / "configs/training/idc_training_config_candidate_v1.json"
CALIB = REPO_ROOT / "scripts/calibrate_training_config.py"


def diag():
    return importlib.import_module(MODULE)


def _upstream_present() -> bool:
    return all(p.exists() for p in (
        REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet",
        REPO_ROOT / "data/manifest/singapore_2024_half_hour.json",
        REPO_ROOT / "data/manifest/singapore_2024_splits.json",
        REPO_ROOT / "data/manifest/singapore_2024_forecast_policy_v3.json",
        REPO_ROOT / "data/manifest/m13f_arrival_intensity_policy_v1.json",
        REPO_ROOT / "configs/frozen_refs/refs_v4.json",
        REPO_ROOT / "data/manifest/formal_splits_v5/train.json",
    ))


needs_assets = pytest.mark.skipif(
    not _upstream_present(), reason="真实冻结上游资产不在本机")


# =============================================================================
# 1. 诊断矩阵的定义
# =============================================================================

def test_diagnostic_origins_are_the_three_required_ones():
    m = diag()
    assert tuple(m.DIAGNOSTIC_ORIGINS) == (48, 4848, 10176)


@needs_assets
def test_diagnostic_origins_are_all_verified_train_origins():
    m = diag()
    from scenario.b6_split_manifests import load_verified_split_manifest_v5, local_origin_from_start

    payload = load_verified_split_manifest_v5(expected_split="train")
    start = int(payload["candidate_origins"]["start"])
    end = int(payload["split_rows"]["end_exclusive"])
    for origin in m.DIAGNOSTIC_ORIGINS:
        assert start <= origin < end
        assert origin + m.HORIZON <= end
        assert local_origin_from_start("train", m.start_for_origin(origin)) == origin


def test_compute_levels_and_cutoffs_are_the_required_sets():
    m = diag()
    assert tuple(m.COMPUTE_LEVELS) == (0.0, 0.25, 1.0)
    assert tuple(m.FORECAST_CUTOFFS) == (4, 48)


def test_storage_action_is_zero():
    m = diag()
    action = m.fixed_action(0.5, action_dim=21, n_groups=20)
    assert action[20] == 0.0, "储能维必须为 0"


def test_fixed_seed_is_single_and_explicit():
    m = diag()
    assert m.FIXED_SEED == 0
    assert set(m.SEED_KWARGS) == {"task_seed", "server_seed", "forecast_seed"}


# =============================================================================
# 2. 最早分叉函数
# =============================================================================

def _ep(steps):
    return {"steps": [{"step": i, "v": v} for i, v in enumerate(steps)]}


def test_first_divergence_returns_the_earliest_differing_step():
    m = diag()
    a = _ep([1, 1, 1, 2, 3])
    b = _ep([1, 1, 1, 9, 9])
    assert m.first_divergence(a, b, "v") == 3


def test_first_divergence_is_none_when_identical():
    m = diag()
    a = _ep([1, 2, 3])
    assert m.first_divergence(a, _ep([1, 2, 3]), "v") is None


def test_first_divergence_detects_step_zero():
    m = diag()
    assert m.first_divergence(_ep([5, 5]), _ep([7, 5]), "v") == 0


# =============================================================================
# 3. business_gap 的适用性（off 时**不适用**）
# =============================================================================

def test_off_is_explicitly_not_a_baseline():
    """`corrector=off` 只用于定位动作作用 —— 声明必须随报告落盘。"""
    m = diag()
    assert "off" in m.OFF_BASELINE_DISCLAIMER
    assert "不得" in m.OFF_BASELINE_DISCLAIMER
    assert "基线" in m.OFF_BASELINE_DISCLAIMER


def test_business_gap_note_states_it_is_not_applicable_when_off():
    m = diag()
    assert "不适用" in m.BUSINESS_GAP_NOTE
    assert "corrector_wrapper" in m.BUSINESS_GAP_NOTE


@needs_assets
def test_business_gap_is_absent_when_the_corrector_is_off():
    """`business_gap` 由 corrector 提供；off 时必须**不适用**，而不是 0。"""
    m = diag()
    ep_off = m.run_episode(48, compute=0.5, corrector_on=False,
                           forecast_cutoff=48, budget=None)
    assert all(s["business_gap"] is None for s in ep_off["steps"]), \
        "corrector=off 时 business_gap 必须记为不适用（None），不得记 0"
    assert all(s["deadline_shortfall"] is None for s in ep_off["steps"])
    # 非空洞性：off 时 exec 必须与 raw 相同
    assert all(s["exec_equals_raw"] for s in ep_off["steps"])


# =============================================================================
# 4. 配置口径
# =============================================================================

@pytest.mark.skipif(not CANDIDATE.exists(), reason="候选尚未物化")
def test_candidate_records_forecast_cutoff_and_its_source():
    d = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    s = d["sampling"]
    assert s["forecast_cutoff"] == 48, "正式候选必须记录 forecast_cutoff=48"
    assert "causal" in s["forecast_source"].lower() or "causal" in s["forecast_source"]
    note = s["forecast_cutoff_note"]
    assert "train.py" in note and "4" in note and "48" in note, \
        "必须写明 train.py 预检仍为 4、接线须改为同值 48"


@pytest.mark.skipif(not CANDIDATE.exists(), reason="候选尚未物化")
def test_candidate_corrector_source_is_the_resolver_output():
    d = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    from planning.corrector import PRODUCTION_CORRECTOR_TIME_LIMIT_S, resolve_corrector_budget

    value, source = resolve_corrector_budget(None, enabled=True)
    assert d["corrector"]["time_limit_s"] == pytest.approx(value)
    assert d["corrector"]["source"] == source == "production_default"
    assert value == pytest.approx(PRODUCTION_CORRECTOR_TIME_LIMIT_S)


def test_calibration_script_has_no_hardcoded_budget():
    """标定脚本**不得**本地硬编码 0.25，必须经解析器取来源。"""
    src = CALIB.read_text(encoding="utf-8")
    assert "resolve_corrector_budget" in src, "必须使用既有预算解析器"
    assert "PRODUCTION_CORRECTOR_TIME_LIMIT_S" in src, "必须导入生产常量"
    assert "CORRECTOR_TIME_LIMIT_S = 0.25" not in src, "不得本地硬编码 0.25"


def test_calibration_script_imports_the_budget_instead_of_defining_it():
    m = importlib.import_module("scripts.calibrate_training_config")
    from planning.corrector import PRODUCTION_CORRECTOR_TIME_LIMIT_S

    assert m.CORRECTOR_TIME_LIMIT_S == pytest.approx(PRODUCTION_CORRECTOR_TIME_LIMIT_S)
    assert m.CORRECTOR_TIME_LIMIT_SOURCE == "production_default"

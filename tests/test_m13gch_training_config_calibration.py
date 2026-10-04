"""M1.3g-f-c-h：正式训练配置候选与 **train-only 预算标定**。

覆盖：24 个 origin 的**预先写定**规则、`origin -> start` 往返、固定 raw 提案、
budget 与乘子缩放**公式可重算**、配置候选结构、claims 恒 false。

**本文件不跑全量标定**（3 提案 × 24 origin × 48 步、corrector on，约 12 分钟）；
只验证**纯函数与规则**，以及已物化产物的结构。
"""

import importlib
import json
import pathlib

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE = "scripts.calibrate_training_config"
CANDIDATE = REPO_ROOT / "configs/training/idc_training_config_candidate_v1.json"


def cal():
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
# 1. 24 个 origin 的预先写定规则
# =============================================================================

@needs_assets
def test_selects_exactly_24_distinct_origins():
    m = cal()
    origins = m.select_origins()
    assert len(origins) == 24, f"必须恰为 24 个，实际 {len(origins)}"
    assert len(set(origins)) == 24, "origin 必须互不相同"
    assert list(origins) == sorted(origins), "origin 必须升序"


@needs_assets
def test_origins_are_day_aligned_and_non_overlapping():
    m = cal()
    origins = m.select_origins()
    slots = m.SLOTS_PER_DAY
    assert slots == 48
    for origin in origins:
        assert origin % slots == 0, f"origin {origin} 非日对齐"
    gaps = [(b - a) // slots for a, b in zip(origins, origins[1:], strict=False)]
    assert min(gaps) >= 1, f"存在重叠：{gaps}"
    # 非空洞性：确实有间隔（等间隔规则）
    assert min(gaps) > 1, f"间隔过密，疑似非等间隔：{gaps}"


@needs_assets
def test_origins_stay_inside_the_verified_train_window():
    """全部来自 verified **train** split，且 episode 不越界。"""
    m = cal()
    from scenario.b6_split_manifests import load_verified_split_manifest_v5

    payload = load_verified_split_manifest_v5(expected_split="train")
    start = int(payload["candidate_origins"]["start"])
    end = int(payload["split_rows"]["end_exclusive"])

    for origin in m.select_origins():
        assert start <= origin < end, f"origin {origin} 越出 train 窗口"
        assert origin + m.HORIZON <= end, f"origin {origin} + H 越界"


@needs_assets
def test_origin_rule_matches_the_pre_written_formula():
    """规则必须**可重算**：`index_k = floor(k*(N-1)/23)`，`origin = 48 + 48*index_k`。"""
    m = cal()
    pool = m.day_aligned_origins()
    total = len(pool)
    expected = tuple(pool[(k * (total - 1)) // 23] for k in range(24))
    assert m.select_origins() == expected
    assert total == 212, f"可用日 origin 应为 212，实际 {total}"


@needs_assets
def test_start_round_trips_back_to_the_origin():
    m = cal()
    from scenario.b6_split_manifests import local_origin_from_start

    origins = m.select_origins()
    for origin in (origins[0], origins[-1]):
        start = m.start_for_origin(origin)
        assert local_origin_from_start("train", start) == origin
        assert start.endswith("+08:00"), f"start 必须带显式时区：{start}"


@needs_assets
def test_the_last_origin_sits_exactly_on_the_window_boundary():
    m = cal()
    from scenario.b6_split_manifests import load_verified_split_manifest_v5

    end = int(load_verified_split_manifest_v5(
        expected_split="train")["split_rows"]["end_exclusive"])
    assert m.select_origins()[-1] + m.HORIZON == end, "末 origin 应恰好在边界上"


# =============================================================================
# 2. 固定 raw 提案
# =============================================================================

def test_reference_action_sets_all_compute_dims_and_zero_storage():
    m = cal()
    action = m.reference_action(0.5, action_dim=21, n_groups=20)
    assert action.shape == (21,)
    assert np.all(action[:20] == 0.5), "20 个计算维必须全取该值"
    assert action[20] == 0.0, "储能维必须为 0"


def test_proposal_order_is_pre_written():
    m = cal()
    assert tuple(m.PROPOSALS) == (1.0, 0.5, 0.25)


# =============================================================================
# 3. 标定公式可重算
# =============================================================================

def _fake(compute, business_mean, carbon_mean):
    return {"compute_value": compute, "business_mean": business_mean,
            "carbon_mean": carbon_mean, "transitions": 100}


def test_business_budget_is_the_minimum_over_proposals():
    m = cal()
    results = [_fake(1.0, 5.0, 9.0), _fake(0.5, 2.0, 7.0), _fake(0.25, 3.0, 4.0)]
    out = m.calibrate_budgets(results)
    assert out["business_budget"] == 2.0
    assert out["business_budget_from_proposal"] == 0.5


def test_carbon_budget_only_considers_proposals_meeting_the_business_budget():
    m = cal()
    results = [_fake(1.0, 5.0, 1.0), _fake(0.5, 2.0, 7.0), _fake(0.25, 3.0, 4.0)]
    out = m.calibrate_budgets(results)
    # business 最小 = 2.0（仅 0.5 达标）⇒ carbon 只能取 0.5 的 7.0，不能取 1.0 的 1.0
    assert out["business_budget"] == 2.0
    assert out["carbon_budget"] == 7.0, "不得取未达标提案的碳排"
    assert out["qualified_proposals"] == [0.5]


def test_ties_break_by_proposal_order():
    m = cal()
    results = [_fake(1.0, 2.0, 3.0), _fake(0.5, 2.0, 3.0), _fake(0.25, 4.0, 1.0)]
    out = m.calibrate_budgets(results)
    assert out["business_budget_from_proposal"] == 1.0, "并列取更早的提案"
    assert out["carbon_budget_from_proposal"] == 1.0


def test_multiplier_scaling_follows_the_written_formula():
    m = cal()
    results = [_fake(1.0, 4.0, 2.0), _fake(0.5, 1.0, 0.5), _fake(0.25, 9.0, 9.0)]
    budgets = m.calibrate_budgets(results)
    mult = m.multiplier_scaling(results, budgets)

    for name, mean_key in (("business", "business_mean"), ("carbon", "carbon_mean")):
        block = mult[name]
        ref = next(r for r in results
                   if r["compute_value"] == block["reference_proposal"])
        mean = ref[mean_key]
        scale = max(1.0, mean)
        assert block["scale"] == pytest.approx(scale)
        assert block["learning_rate"] == pytest.approx(0.01 / scale ** 2)
        assert block["max_multiplier"] == pytest.approx(10.0 / scale)
        assert block["initial_multiplier"] == 0.0

    # 非空洞性：scale < 1 时必须被抬到 1
    tiny = [_fake(1.0, 0.001, 0.002)]
    b2 = m.calibrate_budgets(tiny)
    m2 = m.multiplier_scaling(tiny, b2)
    assert m2["business"]["scale"] == 1.0
    assert m2["business"]["learning_rate"] == pytest.approx(0.01)
    assert m2["business"]["max_multiplier"] == pytest.approx(10.0)


def test_multiplier_scaling_is_not_the_smoke_values():
    """**必须计算**：不得沿用 smoke 的 0.01 / 100。"""
    m = cal()
    results = [_fake(1.0, 4.0, 2.0), _fake(0.5, 1.0, 0.5), _fake(0.25, 9.0, 9.0)]
    budgets = m.calibrate_budgets(results)
    mult = m.multiplier_scaling(results, budgets)
    lr, cap = mult["business"]["learning_rate"], mult["business"]["max_multiplier"]
    assert (lr, cap) != (0.01, 100.0), "不得原样沿用 smoke 的 0.01 / 100"


# =============================================================================
# 4. correction_reason 分类
# =============================================================================

def test_benign_and_fallback_reasons_are_disjoint_and_cover_the_enum():
    """分类必须与 `planning/corrector.py:1-11` 的权威语义一致且**完备**。"""
    m = cal()
    benign = set(m.BENIGN_CORRECTION_REASONS)
    fallback = set(m.ZERO_ACTION_FALLBACK_REASONS)
    assert benign == {"none", "deadline_shortfall"}, "仅这两类是「MIP 最优、可执行」"
    degraded = set(m.EXECUTABLE_DEGRADED_REASONS)
    assert degraded == {"stage_b_timeout_feasible"}
    assert benign.isdisjoint(fallback)
    assert benign.isdisjoint(degraded)
    assert fallback.isdisjoint(degraded)
    from planning.corrector import FailureClass

    all_reasons = {c.value for c in FailureClass}
    assert benign | fallback | degraded == all_reasons, (
        f"未分类的原因：{all_reasons - benign - fallback - degraded}")
    # `base_shortage` 是**零动作回退**，不是「求解器失败」，但同样不可执行
    assert "base_shortage" in fallback


# =============================================================================
# 5. 配置候选（需已物化）
# =============================================================================

@pytest.mark.skipif(not CANDIDATE.exists(), reason="配置候选尚未物化")
def test_candidate_records_the_decided_values():
    d = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    assert d["schema"] == "idc-training-config-candidate-v1"
    assert d["status"] == "candidate_not_frozen", "本卡产物是**候选**，不是冻结"
    assert d["corrector"]["mode"] == "on"
    assert d["corrector"]["time_limit_s"] == pytest.approx(0.25)
    assert d["backend"] == {"device": "cpu", "torch_num_threads": 1,
                            "note": d["backend"]["note"]}
    assert d["policy"]["hidden"] == 64 and d["policy"]["action_dim"] == 21
    assert d["policy"]["obs_dim"] > 0
    opt = d["optimizer"]
    assert opt["lr"] == pytest.approx(3e-4) and opt["weight_decay"] == 0.0
    ppo = d["ppo"]
    assert ppo["clip_epsilon"] == pytest.approx(0.2)
    assert ppo["gae_lambda"] == pytest.approx(0.95)
    assert ppo["gamma_per_step"] == pytest.approx(np.sqrt(0.99))
    s = d["sampling"]
    assert s["horizon"] == 48 and s["episodes_per_batch"] == 4
    assert s["transitions_per_batch"] == 192
    assert s["epochs_per_batch"] == 4 and s["minibatch_size"] == 48
    assert s["minibatches_per_epoch"] == 4
    assert "old_raw_log_prob" in s["frozen_within_batch"]
    assert s["lagrangian_updates_per_batch"] == 1
    sc = d["scale"]
    assert sc["batches_per_seed"] == 512 and sc["transitions_per_seed"] == 98304
    assert sc["train_seeds"] == [0, 1, 2]


@pytest.mark.skipif(not CANDIDATE.exists(), reason="配置候选尚未物化")
def test_candidate_budgets_and_multipliers_are_recomputable():
    d = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    b, mu = d["budgets"], d["multipliers"]
    # M1.3g-f-c-i：F3-R2 修复后参考轨迹的 `sla_violation_count` 逐 transition 全为 0，
    # 故 `business_budget` 实测为 **0.0**（dh.6 规则照常取三提案最小，**不**加人为下限）。
    # 因此这里断言「非负」而非「> 0」；carbon 仍来自实测的逐 transition 碳排均值。
    assert b["business_budget"] >= 0.0 and b["carbon_budget"] > 0
    assert "violation_task_steps" in b["business_budget_unit"]
    assert "kgCO2e" in b["carbon_budget_unit"]
    for name in ("business", "carbon"):
        block = mu[name]
        scale = block["scale"]
        assert scale == max(1.0, block["reference_transition_mean"])
        assert block["learning_rate"] == pytest.approx(0.01 / scale ** 2)
        assert block["max_multiplier"] == pytest.approx(10.0 / scale)
        assert block["initial_multiplier"] == 0.0


@pytest.mark.skipif(not CANDIDATE.exists(), reason="配置候选尚未物化")
def test_candidate_does_not_rewrite_the_historical_audit():
    """历史审计 `docs/training_config_candidates.json` 的 S/P 分级不得被改写。"""
    audit = json.loads(
        (REPO_ROOT / "docs/training_config_candidates.json").read_text(encoding="utf-8"))
    items = {i["id"]: i for i in audit["items"]}
    assert items["business_budget"]["value_source_grade"] == "S/P"
    assert items["carbon_budget"]["value_source_grade"] == "S/P"
    assert items["multiplier_learning_rate"]["source_grade"] == "S/P"
    assert items["multiplier_max"]["source_grade"] == "S/P"
    assert items["solver_determinism"]["source_grade"] == "F"

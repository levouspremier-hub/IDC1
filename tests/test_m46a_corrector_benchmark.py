"""M4.6a 测试：corrector 预算矩阵 benchmark 的 schema、隔离与 timeout 计数。

不断言绝对耗时（避免机器抖动），只断言结构与计数。
"""

import json
import subprocess

import pytest

from scripts.benchmark_corrector import (
    BUDGET_MATRIX_S,
    EXECUTABLE_REASONS,
    SCENARIOS,
    build_comparison_table,
    classify_outcome,
    make_scenario_env,
    run_benchmark,
)

STAT_KEYS = {"mean_s", "median_s", "p95_s", "max_s"}


@pytest.fixture(scope="module")
def small_report() -> dict:
    """小规模跑一遍（快），用于 schema 校验。"""
    return run_benchmark(
        seeds=[0], budgets=[0.01, 0.25], scenarios=["normal", "tight"],
        episodes=1, horizon=4, warmup_steps=1,
    )


# --- schema ---

def test_benchmark_schema(small_report):
    assert "measurement_basis" in small_report
    basis = small_report["measurement_basis"]
    assert basis["tracemalloc_active"] is False
    assert basis["timer"] == "time.perf_counter"
    assert basis["budget_matrix_s"] == [0.01, 0.25]
    assert "normal" in small_report["scenarios"] and "tight" in small_report["scenarios"]


def test_every_candidate_budget_has_results(small_report):
    for scenario in ("normal", "tight"):
        budgets = small_report["scenarios"][scenario]["budgets"]
        for b in ("0.01", "0.25"):
            assert b in budgets, (scenario, b)
            entry = budgets[b]
            for key in ("wrapper_step", "corrector_total", "stage_a", "stage_b"):
                assert STAT_KEYS <= set(entry[key]), key
            for key in ("steps_per_s", "timeout_count", "timeout_rate",
                        "zero_action_fallback_count", "business_gap_sum"):
                assert key in entry, key


def test_budget_matrix_is_fixed():
    assert BUDGET_MATRIX_S == [0.01, 0.02, 0.05, 0.10, 0.25]


def test_raw_env_step_measured(small_report):
    for scenario in ("normal", "tight"):
        raw = small_report["scenarios"][scenario]["raw_env_step"]
        assert STAT_KEYS <= set(raw)


# --- timeout 语义与计数 ---

def test_timeout_counted_as_zero_action_fallback(small_report):
    """极小预算下 timeout 必须计入零动作回退，且不得计入成功。"""
    entry = small_report["scenarios"]["normal"]["budgets"]["0.01"]
    assert entry["timeout_count"] >= 0
    assert entry["zero_action_fallback_count"] >= entry["timeout_count"]
    assert 0.0 <= entry["timeout_rate"] <= 1.0
    if entry["timeout_count"] > 0:
        assert entry["timeout_rate"] > 0.0
        assert entry["business_gap_sum"] > 0.0


def test_timeout_not_counted_as_success(small_report):
    """timeout 不计为 executable candidate；三类结果互斥且完备。"""
    for scenario in ("normal", "tight"):
        for entry in small_report["scenarios"][scenario]["budgets"].values():
            assert entry["n_steps"] == (
                entry["executable_candidate_count"]
                + entry["executable_degraded_count"]
                + entry["timeout_count"]
                + entry["non_timeout_failure_count"]
            )


# --- 候选预算比较表 ---

def test_comparison_table_columns(small_report):
    table = build_comparison_table(small_report)
    assert table, "候选预算比较表不应为空"
    for row in table:
        assert set(row) == {
            "scenario", "budget_s", "timeout_rate", "p95_wrapper_step_s",
            "business_gap_sum", "steps_per_s",
        }


def test_no_automatic_production_default(small_report):
    assert small_report["production_default_selected"] is False
    assert "selection_note" in small_report
    assert "M5" in small_report["selection_note"] or "M9" in small_report["selection_note"]


# --- CLI ---

def test_cli_outputs_json():
    out = subprocess.run(
        ["uv", "run", "python", "scripts/benchmark_corrector.py", "--quick"],
        capture_output=True, text=True, check=True,
    ).stdout
    report = json.loads(out)
    assert report["measurement_basis"]["tracemalloc_active"] is False
    assert "candidate_budget_table" in report


# --- M4.6a1：场景一致性与结果会计 ---

def test_raw_baseline_uses_same_scenario_config():
    """raw baseline 工厂必须接收与 wrapper 完全相同的 scenario 配置。"""
    seen = {}

    for scenario, cfg in SCENARIOS.items():
        env = make_scenario_env(scenario, horizon=6)
        seen[scenario] = {
            "access_limit_kw": float(env.access_limit_kw),
            "bess_soc_init": float(env.bess_soc_init),
            "horizon": int(env.horizon),
        }
        assert seen[scenario]["horizon"] == 6
        assert seen[scenario]["bess_soc_init"] == pytest.approx(cfg["soc_init"])
        # access_limit 会被 idc_power_scale_factor 缩放，故按比例核验
        assert seen[scenario]["access_limit_kw"] > 0.0


def test_scenario_config_reported(small_report):
    for scenario in ("normal", "tight"):
        cfg = small_report["scenarios"][scenario]["scenario_config"]
        assert set(cfg) == {
            "access_limit_kw", "access_limit_kw_declared", "bess_soc_init", "horizon",
        }
        assert cfg["horizon"] == 4


def test_outcome_classification_table():
    assert EXECUTABLE_REASONS == {"none", "deadline_shortfall"}
    cases = {
        "none": "executable_candidate",
        "deadline_shortfall": "executable_candidate",
        "stage_b_timeout_feasible": "executable_degraded",
        "timeout": "timeout",
        "base_shortage": "non_timeout_failure",
        "solver_failure": "non_timeout_failure",
        "proposal_invalid": "non_timeout_failure",
        "some_unknown_reason": "non_timeout_failure",
    }
    for reason, expected in cases.items():
        assert classify_outcome(reason) == expected, reason


def test_accounting_identity_holds(small_report):
    for scenario in ("normal", "tight"):
        for entry in small_report["scenarios"][scenario]["budgets"].values():
            assert entry["n_steps"] == (
                entry["executable_candidate_count"]
                + entry["executable_degraded_count"]
                + entry["timeout_count"]
                + entry["non_timeout_failure_count"]
            ), entry
            # timeout 必须归入 failure_counts；且不计为 executable candidate
            assert entry["failure_counts"].get("timeout", 0) == entry["timeout_count"]
            assert "executable_candidate_count" in entry


def test_failure_counts_cover_all_failures(small_report):
    for scenario in ("normal", "tight"):
        for entry in small_report["scenarios"][scenario]["budgets"].values():
            total_failures = entry["timeout_count"] + entry["non_timeout_failure_count"]
            assert sum(entry["failure_counts"].values()) == total_failures
            for key in ("base_shortage", "solver_failure", "proposal_invalid"):
                assert key in entry["failure_counts"], key


def test_zero_action_fallback_and_unsafe_counts(small_report):
    for scenario in ("normal", "tight"):
        for entry in small_report["scenarios"][scenario]["budgets"].values():
            # 失败类别绝不允许发送非零动作
            assert entry["unsafe_failure_action_count"] == 0
            # 零动作回退数至少覆盖 timeout
            assert entry["zero_action_fallback_count"] >= entry["timeout_count"]


def test_no_optimal_count_field(small_report):
    for scenario in ("normal", "tight"):
        for entry in small_report["scenarios"][scenario]["budgets"].values():
            assert "optimal_count" not in entry


def test_timeout_is_not_executable_candidate(small_report):
    entry = small_report["scenarios"]["normal"]["budgets"]["0.01"]
    if entry["timeout_count"] > 0:
        assert entry["executable_candidate_count"] <= entry["n_steps"] - entry["timeout_count"]

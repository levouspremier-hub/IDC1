"""M4.6a 测试：corrector 预算矩阵 benchmark 的 schema、隔离与 timeout 计数。

不断言绝对耗时（避免机器抖动），只断言结构与计数。
"""

import json
import subprocess

import pytest

from scripts.benchmark_corrector import BUDGET_MATRIX_S, build_comparison_table, run_benchmark

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
    for scenario in ("normal", "tight"):
        for entry in small_report["scenarios"][scenario]["budgets"].values():
            assert entry["timeout_count"] + entry["optimal_count"] <= entry["n_steps"]


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

"""M5.4h 测试：corrector 预算耗尽的可审计归因与节点预算标定。

本卡**不修改**生产求解语义：`mip_max_nodes` 只在探针内通过 runtime wrapper 注入，
默认 corrector 配置与默认预算一律不动。
"""

import ast
import json
import pathlib
import shlex
import subprocess
import sys

import pytest
import yaml

from scripts import probe_corrector_repro as probe

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REQUIRED_BUDGETS = (0.05, 0.10, 0.25, 0.50, 2.0)
REQUIRED_STAGE_FIELDS = (
    "options", "status", "success", "message", "elapsed_s",
    "remaining_deadline_s", "mip_node_count", "mip_dual_bound", "mip_gap",
    "corrector_failure", "corrector_reason", "digest",
)


def run_probe(*args: str, timeout: int = 1200) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/probe_corrector_repro.py", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )


# --- 1. 显式 --corrector-time-limit 且写入三处 ------------------------------

def test_parser_accepts_explicit_corrector_time_limit():
    parser = probe.build_parser()
    args = parser.parse_args(["--corrector-time-limit", "0.25"])
    assert args.corrector_time_limit == pytest.approx(0.25)

    default = parser.parse_args([])
    assert default.corrector_time_limit == pytest.approx(0.05), "默认必须仍是 0.05"


def test_parser_accepts_matrix_load_and_node_cap_options():
    parser = probe.build_parser()
    args = parser.parse_args([
        "--time-limits", "0.05", "0.10", "--runs", "6", "--steps", "8",
        "--load", "hogs", "--load-concurrency", "4", "--node-caps", "100", "1000",
    ])
    assert args.time_limits == [0.05, 0.10]
    assert args.load == "hogs"
    assert args.load_concurrency == 4
    assert args.node_caps == [100, 1000]


@pytest.mark.slow
def test_explicit_budget_is_recorded_in_config_manifest_and_report(tmp_path):
    result = run_probe("--modes", "off", "--runs", "1", "--steps", "1",
                       "--corrector-time-limit", "0.25",
                       "--base-dir", str(tmp_path), "--run-id", "budget")
    assert result.returncode == 0, result.stderr
    run_dir = tmp_path / "budget"

    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert config["corrector_time_limit_s"] == pytest.approx(0.25)
    assert report["corrector_time_limit_s"] == pytest.approx(0.25)
    assert "--corrector-time-limit" in shlex.split(manifest["command"])


# --- 2. 逐步、逐阶段的求解器事实 --------------------------------------------

def test_child_records_every_stage_field():
    """子进程探针必须产出逐步 × 逐阶段的完整字段。"""
    entry = probe.run_once("on", steps=2, budget=0.25, record_stages=True)
    assert entry["steps"], "必须记录每一步"
    assert len(entry["steps"]) == 2
    for step in entry["steps"]:
        assert set(step) >= {"step", "digest", "corrector_failure", "corrector_reason", "calls"}
        assert step["calls"], f"step {step['step']} 没有记录任何阶段调用"
        for call in step["calls"]:
            assert set(call) >= set(REQUIRED_STAGE_FIELDS), sorted(call)
            assert call["stage"] in {"A", "B", "extra"}
            assert isinstance(call["options"], dict) and call["options"]
            assert call["elapsed_s"] >= 0.0
            assert call["remaining_deadline_s"] is None or call["remaining_deadline_s"] >= 0.0


def test_two_stages_are_recorded_per_step_when_the_solver_converges():
    entry = probe.run_once("on", steps=1, budget=0.25, record_stages=True)
    stages = [call["stage"] for call in entry["steps"][0]["calls"]]
    assert stages[:2] == ["A", "B"], f"阶段顺序必须是 A、B，实际 {stages}"


def test_unavailable_solver_fields_are_null_with_a_reason():
    """字段不可用必须写 null **并给出原因**，不得伪造数值。"""
    entry = probe.run_once("on", steps=1, budget=0.25, record_stages=True)
    for call in entry["steps"][0]["calls"]:
        for field in ("mip_node_count", "mip_dual_bound", "mip_gap"):
            if call[field] is None:
                assert field in call["unavailable_fields"], f"{field} 为 null 但未说明原因"
            else:
                assert field not in call["unavailable_fields"]


def test_remaining_deadline_matches_the_time_limit_actually_passed():
    entry = probe.run_once("on", steps=2, budget=0.25, record_stages=True)
    for step in entry["steps"]:
        for call in step["calls"]:
            assert call["remaining_deadline_s"] == pytest.approx(
                call["options"]["time_limit"], abs=1e-9
            )


def test_reported_options_carry_the_deterministic_flags():
    """记录的是**实际传入**的 options —— 必须含 M5.4g 接线的确定性选项。"""
    entry = probe.run_once("on", steps=1, budget=0.25, record_stages=True)
    for call in entry["steps"][0]["calls"]:
        assert call["options"]["random_seed"] == 0
        assert call["options"]["parallel"] is False


# --- 3. 结论分类（纯函数，数据驱动，三选一） --------------------------------

def _matrix(*, distinct_by_budget, processes=6, steps=8):
    return {
        "budgets": {
            str(budget): {"distinct": distinct, "processes": processes, "steps": steps}
            for budget, distinct in distinct_by_budget.items()
        }
    }


def test_classify_wall_clock_dominant_when_largest_budget_is_stable():
    out = probe.classify(_matrix(distinct_by_budget={
        0.05: 3, 0.10: 2, 0.25: 1, 0.50: 1, 2.0: 1,
    }))
    assert out["conclusion"] == "wall_clock_budget_dominant"
    assert out["conclusion"] in probe.ALLOWED_CONCLUSIONS


def test_classify_node_or_solve_path_when_largest_budget_is_also_unstable():
    out = probe.classify(_matrix(distinct_by_budget={
        0.05: 3, 0.10: 2, 0.25: 2, 0.50: 2, 2.0: 2,
    }))
    assert out["conclusion"] == "node_or_solve_path_problem"


def test_classify_insufficient_evidence_when_no_instability_reproduced():
    out = probe.classify(_matrix(distinct_by_budget={
        0.05: 1, 0.10: 1, 0.25: 1, 0.50: 1, 2.0: 1,
    }))
    assert out["conclusion"] == "insufficient_evidence"


@pytest.mark.parametrize(
    "broken",
    [
        {"0.05": 3, "0.10": 1, "0.25": 1, "0.50": 1},                 # 缺 2.0 档
        {"0.05": 3, "0.10": 1, "0.25": 1, "0.50": 1, "2.0": 1},       # 完整但下面调小样本
    ],
)
def test_classify_insufficient_evidence_when_matrix_is_incomplete(broken):
    matrix = {"budgets": {k: {"distinct": v, "processes": 6, "steps": 8}
                          for k, v in broken.items()}}
    if len(broken) == 5:
        matrix["budgets"]["0.05"]["processes"] = 3  # 进程数不足
    out = probe.classify(matrix)
    assert out["conclusion"] == "insufficient_evidence"
    assert out["reason"]


def test_classify_never_returns_anything_outside_the_three_allowed():
    for distincts in ([1, 1, 1, 1, 1], [3, 2, 1, 1, 1], [3, 2, 2, 2, 2], [], [1, 1]):
        matrix = {"budgets": {
            str(b): {"distinct": d, "processes": 6, "steps": 8}
            for b, d in zip(REQUIRED_BUDGETS, distincts, strict=False)
        }}
        out = probe.classify(matrix)
        assert out["conclusion"] in probe.ALLOWED_CONCLUSIONS
        assert set(probe.ALLOWED_CONCLUSIONS) == {
            "wall_clock_budget_dominant",
            "node_or_solve_path_problem",
            "insufficient_evidence",
        }


# --- 4. 禁止事项的结构性保证 ------------------------------------------------

def test_probe_never_touches_planning_defaults():
    """探针不得写入 planning 的默认配置；节点上限只能 runtime 注入。"""
    source = pathlib.Path(probe.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                rendered = ast.unparse(target)
                assert not rendered.startswith("planning."), (
                    f"探针不得给 planning 赋值：{rendered}"
                )
    # 不得改默认预算常量
    assert probe.DEFAULT_CORRECTOR_TIME_LIMIT_S == pytest.approx(0.05)


def test_node_cap_is_only_injected_at_runtime():
    """`mip_max_nodes` 只能出现在 runtime wrapper 里，不得成为默认选项。"""
    source = pathlib.Path(probe.__file__).read_text(encoding="utf-8")
    assert "mip_max_nodes" in source
    # 默认确定性选项里不得含节点上限
    from planning.model import deterministic_mip_options

    assert "mip_max_nodes" not in deterministic_mip_options(time_limit_s=0.05)
    assert "mip_max_nodes" not in deterministic_mip_options(time_limit_s=None)
    assert probe.DEFAULT_NODE_CAP is None

"""M5.4h1 测试：诊断产物真实性与节点上限语义修正。

本卡只改 probe：审计账本必须完整（含 skipped 行）、负载事实必须自洽、
`attribution` 与 `release_gate` 必须分离、节点上限必须另设「替代语义」臂。
**绝不改变生产 corrector 行为。**
"""

import json
import pathlib
import shlex
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from scripts import probe_corrector_repro as probe

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REQUIRED_AUDIT_COLUMNS = (
    "options_json", "status", "success", "message", "elapsed_s",
    "remaining_deadline_s", "mip_node_count", "mip_dual_bound", "mip_gap",
    "unavailable_fields", "corrector_failure", "corrector_reason", "digest",
    "skipped", "skip_reason", "wall_clock_disabled_in_probe",
)
DEFAULT_BUDGET = 0.05


def run_probe(*args: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/probe_corrector_repro.py", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )


# --- 1. 逐阶段全字段 + 显式 skipped 行 --------------------------------------

def test_stage_rows_carry_every_required_column():
    rows = probe.build_stage_rows(
        entry={
            "steps_detail": [{
                "step": 0, "digest": "d0", "corrector_reason": "none",
                "calls": [{
                    "stage": "A", "options": {"time_limit": 0.25, "random_seed": 0},
                    "status": 0, "success": True, "message": "ok", "elapsed_s": 0.01,
                    "remaining_deadline_s": 0.25, "mip_node_count": 1.0,
                    "mip_dual_bound": 0.0, "mip_gap": 0.0,
                    "unavailable_fields": {}, "corrector_failure": "none",
                    "corrector_reason": "none", "node_cap_injected": None,
                }],
            }],
        },
        budget=0.25, pid=1, node_cap=None, wall_clock_disabled=False,
    )
    assert rows, "必须产出审计行"
    for row in rows:
        for column in REQUIRED_AUDIT_COLUMNS:
            assert column in row, f"审计行缺少列 {column}"


def test_missing_stage_b_produces_an_explicit_skipped_row():
    """某步只跑了 Stage A 时，Stage B 必须有一条显式 skipped 行，不得静默少行。"""
    rows = probe.build_stage_rows(
        entry={
            "steps_detail": [{
                "step": 0, "digest": "d0", "corrector_reason": "timeout",
                "calls": [{
                    "stage": "A", "options": {"time_limit": 0.05},
                    "status": 1, "success": False, "message": "time limit",
                    "elapsed_s": 0.05, "remaining_deadline_s": 0.0,
                    "mip_node_count": None, "mip_dual_bound": None, "mip_gap": None,
                    "unavailable_fields": {"mip_node_count": "超时未提供"},
                    "corrector_failure": "timeout", "corrector_reason": "timeout",
                    "node_cap_injected": None,
                }],
            }],
        },
        budget=0.05, pid=1, node_cap=None, wall_clock_disabled=False,
    )
    stages = {row["stage"]: row for row in rows if row["step"] == 0}
    assert set(stages) == {"A", "B"}, f"每步必须有 A、B 两行，实际 {sorted(stages)}"
    assert stages["A"]["skipped"] is False
    assert stages["B"]["skipped"] is True
    assert stages["B"]["skip_reason"], "skipped 行必须写明原因"
    assert stages["B"]["status"] is None
    assert stages["B"]["digest"] == "d0"


def test_step_with_no_calls_marks_both_stages_skipped():
    rows = probe.build_stage_rows(
        entry={"steps_detail": [{
            "step": 0, "digest": "d0", "corrector_reason": "proposal_invalid", "calls": [],
        }]},
        budget=0.05, pid=1, node_cap=None, wall_clock_disabled=False,
    )
    assert [row["stage"] for row in rows] == ["A", "B"]
    assert all(row["skipped"] is True for row in rows)
    assert all(row["skip_reason"] for row in rows)


# --- 2. 负载事实统一 --------------------------------------------------------

def test_load_injected_is_derived_not_hardcoded():
    parser = probe.build_parser()
    none_args = parser.parse_args([])
    hogs_args = parser.parse_args(["--load", "hogs", "--load-concurrency", "4"])
    assert probe.load_info(none_args)["load_injected"] is False
    assert probe.load_info(hogs_args)["load_injected"] is True


def test_provenance_note_matches_the_load_condition():
    parser = probe.build_parser()
    none_note = probe.provenance_note(probe.load_info(parser.parse_args([])))
    hogs_note = probe.provenance_note(
        probe.load_info(parser.parse_args(["--load", "hogs", "--load-concurrency", "4"]))
    )
    assert "未注入" in none_note
    assert "已注入受控 CPU 负载" in hogs_note
    assert "concurrency=4" in hogs_note or "4" in hogs_note


# --- 3. release_gate 与 attribution 分离 ------------------------------------

def _obs(distinct, *, processes=6, steps=8, source="matrix.0.05"):
    return {"source": source, "distinct": distinct, "processes": processes, "steps": steps}


def test_release_gate_blocks_when_default_budget_is_unstable():
    gate = probe.evaluate_release_gate([_obs(2)])
    assert gate["blocked"] is True
    assert gate["default_budget_s"] == DEFAULT_BUDGET
    assert gate["all_qualifying"] is True


def test_release_gate_passes_when_default_budget_is_stable():
    gate = probe.evaluate_release_gate([_obs(1)])
    assert gate["blocked"] is False


def test_release_gate_is_not_evaluated_without_a_default_budget_measurement():
    """只测了 corrector 关闭时，不得凭空判 blocked（也不得声称已放行）。"""
    gate = probe.evaluate_release_gate([])
    assert gate["evaluated"] is False
    assert gate["blocked"] is False
    assert gate["reason"]


def test_release_gate_flags_non_qualifying_observations():
    gate = probe.evaluate_release_gate([_obs(1, processes=2, steps=3)])
    assert gate["all_qualifying"] is False


def test_attribution_and_release_gate_are_separate_concepts():
    """归因为 wall_clock_budget_dominant **不能**让 release_gate 放行。"""
    assert hasattr(probe, "evaluate_release_gate")
    assert hasattr(probe, "classify")
    matrix = {"budgets": {
        "0.05": {"distinct": 2, "processes": 6, "steps": 8},
        "0.10": {"distinct": 1, "processes": 6, "steps": 8},
        "0.25": {"distinct": 1, "processes": 6, "steps": 8},
        "0.50": {"distinct": 1, "processes": 6, "steps": 8},
        "2.0": {"distinct": 1, "processes": 6, "steps": 8},
    }}
    attribution = probe.classify(matrix)
    gate = probe.evaluate_release_gate([_obs(2)])
    assert attribution["conclusion"] == "wall_clock_budget_dominant"
    assert gate["blocked"] is True, "归因成立不得让发布门禁放行"


# --- 4. 节点上限「替代语义」臂 ----------------------------------------------

def _call_milp_once(real_milp, *, options: dict) -> None:
    """通过已被 wrapper 替换的 `scipy.optimize.milp` 真正调用一次，随后还原。"""
    import numpy as np
    import scipy.optimize as scipy_optimize
    from scipy.optimize import Bounds, LinearConstraint, milp

    try:
        milp(
            c=np.array([1.0, 1.0]),
            constraints=[LinearConstraint(
                np.array([[1.0, 1.0]]), np.array([1.0]), np.array([1.0])
            )],
            integrality=np.array([1.0, 1.0]),
            bounds=Bounds(lb=[0, 0], ub=[1, 1]),
            options=options,
        )
    finally:
        scipy_optimize.milp = real_milp

def test_wrapper_can_disable_wall_clock_for_the_alternative_arm():
    """替代语义臂必须**移除** time_limit，而不是叠加节点上限。"""
    records, real_milp = probe._stage_recorder(node_cap=100, wall_clock_disabled=True)
    _call_milp_once(real_milp, options={"time_limit": 0.05, "random_seed": 0})

    assert records, "必须记录一次调用"
    options = records[0]["options"]
    assert options["mip_max_nodes"] == 100
    assert "time_limit" not in options, f"替代语义臂必须移除 time_limit：{options}"
    assert records[0]["wall_clock_disabled_in_probe"] is True
    assert records[0]["remaining_deadline_s"] is None


def test_wall_clock_arm_keeps_time_limit():
    records, real_milp = probe._stage_recorder(node_cap=100, wall_clock_disabled=False)
    _call_milp_once(real_milp, options={"time_limit": 0.05, "random_seed": 0})

    assert records[0]["options"]["mip_max_nodes"] == 100
    assert records[0]["options"]["time_limit"] == pytest.approx(0.05)
    assert records[0]["wall_clock_disabled_in_probe"] is False


def test_argument_names_separate_the_two_node_cap_arms():
    parser = probe.build_parser()
    args = parser.parse_args(["--node-cap-arm", "alternative_semantics", "--node-caps", "100"])
    assert args.node_cap_arm == "alternative_semantics"
    default = parser.parse_args([])
    assert default.node_cap_arm == "with_wall_clock"


# --- 5. 禁止事项 ------------------------------------------------------------

def test_probe_never_writes_planning_defaults():
    """节点上限只允许存在于探针；planning 的默认选项不得含它。"""
    source = pathlib.Path(probe.__file__).read_text(encoding="utf-8")
    assert "mip_max_nodes" in source
    from planning.model import deterministic_mip_options

    for options in (deterministic_mip_options(time_limit_s=0.05),
                    deterministic_mip_options(time_limit_s=None)):
        assert "mip_max_nodes" not in options
    assert probe.DEFAULT_CORRECTOR_TIME_LIMIT_S == pytest.approx(DEFAULT_BUDGET)


# --- 6. 真实产物（slow） ----------------------------------------------------

@pytest.mark.slow
def test_ledger_persists_skipped_rows_and_load_facts(tmp_path):
    result = run_probe(
        "--modes", "on", "--time-limits", "0.05", "--runs", "1", "--steps", "2",
        "--load", "hogs", "--load-concurrency", "2",
        "--base-dir", str(tmp_path), "--run-id", "m54h1_ledger",
    )
    assert result.returncode in (0, 1), result.stderr
    run_dir = tmp_path / "m54h1_ledger"

    table = pd.read_parquet(run_dir / "summary.parquet")
    for column in REQUIRED_AUDIT_COLUMNS:
        assert column in table.columns, f"审计表缺少列 {column}"

    # 每步必须恰好 A、B 两行
    per_step = table[table.node_cap.isna()].groupby("step").size() if "node_cap" in table else None
    for step, count in (per_step.items() if per_step is not None else []):
        assert count == 2, f"step {step} 应有 A、B 两行，实际 {count}"

    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert config["load_injected"] is True
    assert report["load_injected"] is True
    assert summary["load_injected"] is True
    assert "已注入受控 CPU 负载" in report["provenance_note"]
    assert "--load" in shlex.split(manifest["command"])


@pytest.mark.slow
def test_unstable_default_budget_forces_a_failed_manifest(tmp_path):
    """0.05 s 不稳定时，即使归因成立，manifest 也必须是 failed。"""
    result = run_probe(
        "--modes", "on", "--time-limits", "0.05", "0.25",
        "--runs", "6", "--steps", "8",
        "--load", "hogs", "--load-concurrency", "8",
        "--base-dir", str(tmp_path), "--run-id", "m54h1_gate",
    )
    run_dir = tmp_path / "m54h1_gate"
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    gate = report["release_gate"]

    assert gate["default_budget_s"] == pytest.approx(DEFAULT_BUDGET)
    if gate["blocked"]:
        assert manifest["status"] == "failed", "release_gate.blocked 必须反映为 failed manifest"
        assert result.returncode != 0
    else:
        assert manifest["status"] == "success"


# --- 7. 回归：候选值表述必须如实 --------------------------------------------

def test_candidate_notes_never_claim_cross_machine_guarantees():
    notes = probe.candidate_notes(
        {"conclusion": "wall_clock_budget_dominant"}, {"blocked": True}, {}
    )
    assert notes["machine_local"] is True
    assert "本机" in notes["disclaimer"]
    assert "跨机器" in notes["disclaimer"]
    assert notes["default_budget_still_blocked"] is True
    assert "候选" in notes["wall_clock_candidate"]


def test_node_cap_is_a_candidate_only_when_the_alternative_arm_is_stable():
    unstable = probe.candidate_notes(
        None, {"blocked": True},
        {"100": {"arm": "alternative_semantics", "wall_clock_disabled_in_probe": True,
                 "distinct": 2}},
    )
    assert unstable["node_cap_alternative_semantics"]["candidate"] is False
    assert "不得" in unstable["node_cap_alternative_semantics"]["note"]

    stable = probe.candidate_notes(
        None, {"blocked": True},
        {"100": {"arm": "alternative_semantics", "wall_clock_disabled_in_probe": True,
                 "distinct": 1}},
    )
    assert stable["node_cap_alternative_semantics"]["candidate"] is True
    assert stable["node_cap_alternative_semantics"]["stable_caps"] == ["100"]


def test_wall_clock_arm_alone_does_not_make_the_node_cap_a_candidate():
    """只做了 with_wall_clock 臂时，不得把节点上限说成候选。"""
    notes = probe.candidate_notes(
        None, {"blocked": True},
        {"100": {"arm": "with_wall_clock", "wall_clock_disabled_in_probe": False,
                 "distinct": 1}},
    )
    assert "node_cap_alternative_semantics" not in notes


def test_alternative_arm_notes_are_absent_when_no_node_cap_was_tested():
    notes = probe.candidate_notes(None, {"blocked": True}, {})
    assert "node_cap_alternative_semantics" not in notes
    assert "wall_clock_candidate" not in notes


# --- 8. 回归：审计表结构完整性（纯函数） ------------------------------------

def test_every_step_gets_exactly_two_stage_rows_even_with_skips():
    rows = probe.build_stage_rows(
        entry={"steps_detail": [
            {"step": 0, "digest": "d0", "corrector_reason": "none", "calls": [
                {"stage": "A", "options": {}, "status": 0, "success": True, "message": "",
                 "elapsed_s": 0.0, "remaining_deadline_s": None, "mip_node_count": None,
                 "mip_dual_bound": None, "mip_gap": None, "unavailable_fields": {},
                 "corrector_failure": "none", "corrector_reason": "none",
                 "node_cap_injected": None},
                {"stage": "B", "options": {}, "status": 0, "success": True, "message": "",
                 "elapsed_s": 0.0, "remaining_deadline_s": None, "mip_node_count": None,
                 "mip_dual_bound": None, "mip_gap": None, "unavailable_fields": {},
                 "corrector_failure": "none", "corrector_reason": "none",
                 "node_cap_injected": None},
            ]},
            {"step": 1, "digest": "d1", "corrector_reason": "timeout", "calls": [
                {"stage": "A", "options": {}, "status": 1, "success": False, "message": "",
                 "elapsed_s": 0.0, "remaining_deadline_s": 0.0, "mip_node_count": None,
                 "mip_dual_bound": None, "mip_gap": None, "unavailable_fields": {},
                 "corrector_failure": "timeout", "corrector_reason": "timeout",
                 "node_cap_injected": None},
            ]},
        ]},
        budget=0.05, pid=1, node_cap=None, wall_clock_disabled=False,
    )
    counts = {}
    for row in rows:
        counts[row["step"]] = counts.get(row["step"], 0) + 1
    assert counts == {0: 2, 1: 2}, f"每步必须恰好 A、B 两行：{counts}"
    assert sum(1 for r in rows if r["skipped"]) == 1


def test_skipped_rows_carry_null_measurements_not_zeros():
    """skipped 行的数值必须是 null —— 不得写成 0 冒充「测到了 0」。"""
    rows = probe.build_stage_rows(
        entry={"steps_detail": [{"step": 0, "digest": "d", "corrector_reason": "x",
                                 "calls": []}]},
        budget=0.05, pid=1, node_cap=None, wall_clock_disabled=False,
    )
    for row in rows:
        assert row["skipped"] is True
        for column in ("status", "success", "elapsed_s", "remaining_deadline_s",
                       "mip_node_count", "mip_dual_bound", "mip_gap", "options_json"):
            assert row[column] is None, f"skipped 行的 {column} 必须是 null"

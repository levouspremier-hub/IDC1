"""M5.4h2 测试：诊断账本语义最终勘误。

三项缺陷（改前均已在冻结历史产物上实测复现）：

1. `summary.parquet` 写 `options_json`，而约定字段名是 `options`（规范 JSON 字符串）；
2. `overall` 由 `attribution` 推导，导致 `release_gate.blocked=true` 与
   `overall.blocked=false` 并存，`summary.json` 也与 `manifest.status="failed"` 冲突；
3. 节点上限从未绑定时，`candidate_notes` 仍给出 `candidate=true`。

语义分层固定为：`release_gate` 决定 M5.4 阶段是否放行；`overall` **只**反映
`release_gate` 的最终阶段状态；`attribution` 只解释不稳定原因，**绝不**决定放行。
**本卡不改变 corrector 求解结果、rollout action、`time_limit` 或默认预算。**
"""

import json
import pathlib
import subprocess
import sys

import pandas as pd
import pytest

from scripts import probe_corrector_repro as probe

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_BUDGET = 0.05
EXPECTED_OPTIONS_COLUMN = "options"
RETIRED_OPTIONS_COLUMN = "options_json"


def run_probe(*args: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/probe_corrector_repro.py", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )


def canonical(value) -> str:
    """约定编码：`json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`。"""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# --- 1. summary.parquet 的字段名与编码 ---------------------------------------

def _stage_call(stage: str, *, options: dict | None = None) -> dict:
    return {
        "stage": stage,
        "options": {"time_limit": 0.25, "random_seed": 0, "parallel": False}
                   if options is None else options,
        "status": 0, "success": True, "message": "ok", "elapsed_s": 0.01,
        "remaining_deadline_s": 0.25, "mip_node_count": 1.0,
        "mip_dual_bound": 0.0, "mip_gap": 0.0, "unavailable_fields": {},
        "corrector_failure": "none", "corrector_reason": "none",
        "node_cap_injected": None,
    }


def _executed_step(calls: list[dict]) -> dict:
    return {"steps_detail": [{
        "step": 0, "digest": "d0", "corrector_failure": "none",
        "corrector_reason": "none", "calls": calls,
    }]}


def test_stage_rows_expose_options_and_drop_the_retired_options_json_column():
    """约定字段名是 `options`；**不得**同时保留 `options_json` 兼容列。"""
    rows = probe.build_stage_rows(
        entry=_executed_step([_stage_call("A"), _stage_call("B")]),
        budget=0.25, pid=1, node_cap=None, wall_clock_disabled=False,
    )
    assert rows, "必须产出审计行"
    for row in rows:
        assert EXPECTED_OPTIONS_COLUMN in row, f"审计行缺少 {EXPECTED_OPTIONS_COLUMN}"
        assert RETIRED_OPTIONS_COLUMN not in row, "不得保留 options_json 兼容列"


def test_executed_stage_options_use_the_fixed_canonical_encoding():
    """编码规则固定，且必须对键排序、去空白、保留非 ASCII。"""
    options = {"z_last": 1, "a_first": 2, "note": "中文候选"}
    rows = probe.build_stage_rows(
        entry=_executed_step([_stage_call("A", options=options)]),
        budget=0.25, pid=1, node_cap=None, wall_clock_disabled=False,
    )
    encoded = rows[0][EXPECTED_OPTIONS_COLUMN]
    assert isinstance(encoded, str)
    assert encoded == canonical(options)
    assert json.loads(encoded) == options


def test_skipped_stage_rows_carry_null_options_and_null_measurements():
    """未执行的阶段允许 `options=null`；测量字段仍必须是 null，不得填零。"""
    rows = probe.build_stage_rows(
        entry={"steps_detail": [{
            "step": 0, "digest": "d0", "corrector_failure": "timeout",
            "corrector_reason": "timeout", "calls": [],
        }]},
        budget=0.05, pid=1, node_cap=None, wall_clock_disabled=False,
    )
    assert [row["stage"] for row in rows] == ["A", "B"]
    for row in rows:
        assert row["skipped"] is True
        assert row[EXPECTED_OPTIONS_COLUMN] is None
        for column in ("status", "success", "elapsed_s", "remaining_deadline_s",
                       "mip_node_count", "mip_dual_bound", "mip_gap"):
            assert row[column] is None, f"skipped 行的 {column} 必须是 null"


def test_every_step_still_gets_exactly_two_stage_rows():
    """保留逐 step × Stage A/B 恰好两行的审计语义。"""
    rows = probe.build_stage_rows(
        entry={"steps_detail": [
            {"step": 0, "digest": "d0", "corrector_failure": "none",
             "corrector_reason": "none", "calls": [_stage_call("A"), _stage_call("B")]},
            {"step": 1, "digest": "d1", "corrector_failure": "timeout",
             "corrector_reason": "timeout", "calls": [_stage_call("A")]},
        ]},
        budget=0.05, pid=1, node_cap=None, wall_clock_disabled=False,
    )
    counts: dict[int, int] = {}
    for row in rows:
        counts[row["step"]] = counts.get(row["step"], 0) + 1
    assert counts == {0: 2, 1: 2}, f"每步必须恰好 A、B 两行：{counts}"


# --- 2. overall 必须反映 release_gate，attribution 绝不决定放行 ---------------

def _obs(distinct: int, *, processes: int = 6, steps: int = 8,
         source: str = "matrix.0.05") -> dict:
    return {"source": source, "distinct": distinct, "processes": processes, "steps": steps}


def _blocked_gate() -> dict:
    return probe.evaluate_release_gate([_obs(2)])


def _clear_gate() -> dict:
    return probe.evaluate_release_gate([_obs(1)])


def test_phase_status_cannot_contradict_a_blocked_release_gate():
    gate = _blocked_gate()
    phase = probe.phase_status(release_gate=gate)
    assert phase["overall"]["blocked"] == gate["blocked"]
    assert phase["overall"]["blocked"] is True
    assert phase["overall"]["conclusion"] == "blocked"


def test_attribution_never_decides_the_phase_status():
    """归因为 wall_clock_budget_dominant **不能**把 overall 或 manifest 写成成功。"""
    phase = probe.phase_status(
        release_gate=_blocked_gate(),
        attribution={"conclusion": "wall_clock_budget_dominant"},
    )
    assert phase["overall"]["blocked"] is True, "归因成立不得让 overall 变成成功"
    assert phase["overall"]["conclusion"] == "blocked"
    assert phase["status"] == "failed"
    assert phase["exit_code"] != 0


def test_phase_status_only_releases_when_the_gate_is_clear():
    phase = probe.phase_status(
        release_gate=_clear_gate(),
        attribution={"conclusion": "insufficient_evidence"},
    )
    assert phase["overall"]["blocked"] is False
    assert phase["overall"]["conclusion"] == "released"
    assert phase["status"] == "success"
    assert phase["exit_code"] == 0


def test_phase_status_does_not_claim_release_without_a_default_budget_measurement():
    """未测默认预算时不得声称已放行（也不得判 blocked）。"""
    gate = probe.evaluate_release_gate([])
    phase = probe.phase_status(release_gate=gate)
    assert gate["evaluated"] is False
    assert phase["overall"]["blocked"] is False
    assert phase["overall"]["conclusion"] != "released"


def test_phase_status_is_decided_by_the_release_gate_alone():
    phase = probe.phase_status(release_gate=_blocked_gate())
    assert phase["overall"]["decided_by"] == "release_gate"


# --- 3. 节点上限：未绑定就绝不是候选 ----------------------------------------

def _node_row(*, node_count: float | None, cap: int, skipped: bool = False) -> dict:
    return {
        "budget_s": 0.05, "node_cap": cap, "pid": 1, "step": 0, "stage": "A",
        "skipped": skipped, "skip_reason": None, "options": "{}",
        "status": None if skipped else 0, "success": None if skipped else True,
        "message": None, "elapsed_s": None, "remaining_deadline_s": None,
        "mip_node_count": node_count, "mip_dual_bound": None, "mip_gap": None,
        "unavailable_fields": "{}", "corrector_failure": "none",
        "corrector_reason": "none", "digest": "d0",
        "node_cap_injected": cap, "wall_clock_disabled_in_probe": True,
    }


def test_node_cap_binding_is_derived_from_observed_solve_counts():
    rows = [_node_row(node_count=1.0, cap=100), _node_row(node_count=1.0, cap=100)]
    unbound = probe.node_cap_binding(rows, cap=100)
    assert unbound["cap_bound"] is False
    assert unbound["max_mip_node_count"] == pytest.approx(1.0)

    bound = probe.node_cap_binding(rows, cap=1)
    assert bound["cap_bound"] is True, "观测到 mip_node_count 触及 cap 即为绑定"


def test_node_cap_binding_ignores_skipped_rows():
    rows = [_node_row(node_count=None, cap=100, skipped=True),
            _node_row(node_count=1.0, cap=100)]
    binding = probe.node_cap_binding(rows, cap=100)
    assert binding["observed_solves"] == 1
    assert binding["cap_bound"] is False


def test_candidate_notes_always_expose_machine_readable_node_cap_fields():
    notes = probe.candidate_notes(None, {"blocked": True}, {})
    assert notes["node_cap_effective"] is False
    assert notes["node_cap_candidate"] is False


def test_unbound_node_cap_is_never_a_candidate():
    """替代语义臂稳定，但 cap 从未绑定 -> 不是候选，且必须给出机器可读原因。"""
    notes = probe.candidate_notes(
        {"conclusion": "wall_clock_budget_dominant"}, {"blocked": True},
        {"100": {"arm": "alternative_semantics", "wall_clock_disabled_in_probe": True,
                 "distinct": 1, "processes": 6, "cap_bound": False,
                 "max_mip_node_count": 1.0}},
    )
    assert notes["node_cap_effective"] is False
    assert notes["node_cap_candidate"] is False
    alternative = notes["node_cap_alternative_semantics"]
    assert alternative["node_cap_candidate"] is False
    assert alternative["stable_caps"] == ["100"], "稳定性仍须如实记录"
    assert alternative["cap_bound_caps"] == []

    reasons = notes["node_cap_reasons"]
    assert "observed_mip_node_count_below_cap" in reasons
    assert "stability_attributable_to_wall_clock_disabled_in_probe" in reasons
    assert "mip_max_nodes_not_a_production_candidate" in reasons


def test_node_cap_candidate_requires_an_observed_binding():
    notes = probe.candidate_notes(
        None, {"blocked": True},
        {"100": {"arm": "alternative_semantics", "wall_clock_disabled_in_probe": True,
                 "distinct": 1, "processes": 6, "cap_bound": True,
                 "max_mip_node_count": 100.0}},
    )
    assert notes["node_cap_effective"] is True
    assert notes["node_cap_candidate"] is True


def test_bound_but_unstable_node_cap_is_not_a_candidate():
    notes = probe.candidate_notes(
        None, {"blocked": True},
        {"100": {"arm": "alternative_semantics", "wall_clock_disabled_in_probe": True,
                 "distinct": 2, "processes": 6, "cap_bound": True,
                 "max_mip_node_count": 100.0}},
    )
    assert notes["node_cap_effective"] is True
    assert notes["node_cap_candidate"] is False


def test_wall_clock_candidate_stays_machine_local():
    notes = probe.candidate_notes(
        {"conclusion": "wall_clock_budget_dominant"}, {"blocked": True}, {}
    )
    assert notes["machine_local"] is True
    assert "本机" in notes["disclaimer"]
    assert "跨机器" in notes["disclaimer"]
    assert "候选" in notes["wall_clock_candidate"]
    assert notes["default_budget_still_blocked"] is True


def test_node_cap_is_not_described_as_a_candidate_without_evidence():
    """只做了 with_wall_clock 臂时不得出现替代语义候选块。"""
    notes = probe.candidate_notes(
        None, {"blocked": True},
        {"100": {"arm": "with_wall_clock", "wall_clock_disabled_in_probe": False,
                 "distinct": 1, "processes": 6}},
    )
    assert "node_cap_alternative_semantics" not in notes
    assert notes["node_cap_candidate"] is False


# --- 4. 真实产物：report / summary.json / manifest 三者一致（slow） ----------

@pytest.mark.slow
def test_artifacts_agree_on_the_release_status(tmp_path):
    result = run_probe(
        "--modes", "on", "--time-limits", "0.05", "0.25",
        "--runs", "6", "--steps", "8",
        "--base-dir", str(tmp_path), "--run-id", "m54h2_gate",
    )
    run_dir = tmp_path / "m54h2_gate"
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    gate = report["release_gate"]

    assert report["overall"]["blocked"] == gate["blocked"]
    assert report["overall"]["conclusion"] == (
        "blocked" if gate["blocked"] else report["overall"]["conclusion"]
    )
    assert summary["overall"] == report["overall"], "summary 的阶段状态必须与 report 一致"
    assert summary["release_gate"]["blocked"] == gate["blocked"]
    assert summary["status"] == manifest["status"]
    assert manifest["status"] == ("failed" if gate["blocked"] else "success")
    assert (result.returncode != 0) == gate["blocked"]


@pytest.mark.slow
def test_new_run_ledger_uses_the_options_column(tmp_path):
    result = run_probe(
        "--modes", "on", "--time-limits", "0.05", "0.25",
        "--runs", "1", "--steps", "2",
        "--base-dir", str(tmp_path), "--run-id", "m54h2_ledger",
    )
    assert result.returncode in (0, 1), result.stderr
    table = pd.read_parquet(tmp_path / "m54h2_ledger" / "summary.parquet")

    assert EXPECTED_OPTIONS_COLUMN in table.columns
    assert RETIRED_OPTIONS_COLUMN not in table.columns

    executed = table.loc[table["skipped"] == False, EXPECTED_OPTIONS_COLUMN]  # noqa: E712
    assert len(executed) > 0, "必须记录已执行的阶段"
    for value in executed:
        assert isinstance(value, str) and json.loads(value) is not None

    skipped = table.loc[table["skipped"] == True, EXPECTED_OPTIONS_COLUMN]  # noqa: E712
    assert len(skipped) > 0
    assert skipped.isna().all(), "skipped 阶段允许 options=null"


@pytest.mark.slow
def test_node_cap_run_records_an_unbound_cap_as_not_effective(tmp_path):
    result = run_probe(
        "--modes", "on", "--node-caps", "100", "10000",
        "--node-cap-arm", "alternative_semantics",
        "--runs", "6", "--steps", "8",
        "--base-dir", str(tmp_path), "--run-id", "m54h2_nodealt",
    )
    assert result.returncode in (0, 1), result.stderr
    run_dir = tmp_path / "m54h2_nodealt"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    notes = summary["candidate_notes"]

    table = pd.read_parquet(run_dir / "summary.parquet")
    observed = table["mip_node_count"].dropna()
    assert len(observed) > 0
    assert (observed < 100).all(), "本配置下 cap 100 不得被绑定"

    assert notes["node_cap_effective"] is False
    assert notes["node_cap_candidate"] is False


# --- 5. 禁止事项的结构性保证 ------------------------------------------------

def test_default_budget_and_time_limit_semantics_are_untouched():
    """本卡不得改默认预算，也不得让 probe 改变求解器停止条件默认值。"""
    assert probe.DEFAULT_CORRECTOR_TIME_LIMIT_S == pytest.approx(0.05)
    assert probe.MODE_BUDGETS["on"] == pytest.approx(0.05)
    assert probe.DEFAULT_NODE_CAP is None

    from planning.model import deterministic_mip_options

    default = deterministic_mip_options(time_limit_s=0.05)
    assert default["time_limit"] == pytest.approx(0.05)
    assert "mip_max_nodes" not in default

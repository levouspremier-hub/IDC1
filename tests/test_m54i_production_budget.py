"""M5.4i 测试：0.25 s 作为 corrector **生产默认**预算。

改前缺陷（本文件在实现前必须为红）：

1. 没有唯一生产默认常量：四个入口各自硬编码 `DEFAULT_CORRECTOR_TIME_LIMIT_S = 0.05`；
2. `--corrector on` 未给预算时 train **直接失败**，而不是解析为生产默认；
3. 四个入口都不记录 provenance 三字段；
4. release gate 无法区分「生产默认观测」与「显式 override 观测」。

本卡**不**改变 corrector 的求解语义：`correct()` 仍要求调用方显式传入 `time_limit_s`，
Stage A/B 仍共享同一个总 deadline。
"""

import inspect
import json
import pathlib
import re
import subprocess
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PRODUCTION_DEFAULT = 0.25
OVERRIDE_BUDGET = 0.05

ENTRY_FILES = (
    "safe_rl_v2/train.py",
    "scripts/smoke_main_chain.py",
    "scripts/probe_rollout_deterministic.py",
    "scripts/probe_corrector_repro.py",
)

PROVENANCE_FIELDS = (
    "production_corrector_time_limit_s",
    "effective_corrector_time_limit_s",
    "corrector_time_limit_source",
)


def run_cli(*args: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )


# --- 1. 唯一生产默认来源 -----------------------------------------------------

def test_the_production_default_lives_in_planning_corrector():
    from planning.corrector import PRODUCTION_CORRECTOR_TIME_LIMIT_S

    assert PRODUCTION_CORRECTOR_TIME_LIMIT_S == pytest.approx(PRODUCTION_DEFAULT)


def test_the_production_default_is_defined_exactly_once_in_the_repo():
    pattern = re.compile(r"^PRODUCTION_CORRECTOR_TIME_LIMIT_S\s*=", re.M)
    hits = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in REPO_ROOT.rglob("*.py")
        if ".venv" not in path.parts and pattern.search(path.read_text(encoding="utf-8"))
    )
    assert hits == ["planning/corrector.py"], f"生产默认常量必须只有一处定义，实际 {hits}"


@pytest.mark.parametrize("rel", ENTRY_FILES)
def test_every_entry_imports_the_single_source(rel):
    source = (REPO_ROOT / rel).read_text(encoding="utf-8")
    assert "PRODUCTION_CORRECTOR_TIME_LIMIT_S" in source, f"{rel} 必须使用唯一生产默认常量"
    assert re.search(
        r"^PRODUCTION_CORRECTOR_TIME_LIMIT_S\s*=", source, re.M
    ) is None, f"{rel} 不得自行定义生产默认常量"


@pytest.mark.parametrize("rel", ENTRY_FILES)
def test_no_entry_still_hardcodes_the_old_default(rel):
    source = (REPO_ROOT / rel).read_text(encoding="utf-8")
    assert not re.search(
        r"DEFAULT_CORRECTOR_TIME_LIMIT_S\s*=\s*0\.05", source
    ), f"{rel} 仍硬编码旧的 0.05 默认"


def test_core_correct_still_requires_an_explicit_time_limit():
    """红线：不得给核心 `correct()` 增加隐式默认参数。"""
    import planning.corrector as corrector

    parameter = inspect.signature(corrector.correct).parameters["time_limit_s"]
    assert parameter.default is inspect.Parameter.empty, "correct() 的 time_limit_s 不得有默认值"


# --- 2. provenance 三字段 ----------------------------------------------------

def test_provenance_fields_are_recorded_by_every_entry():
    """四个入口都必须能写出三字段（结构性检查：常量已被引用）。"""
    for rel in ENTRY_FILES:
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for field in PROVENANCE_FIELDS:
            assert field in source, f"{rel} 未记录 {field}"


def test_source_is_decided_by_whether_a_budget_was_given_not_by_its_value():
    """显式给出 0.25 也必须记成 explicit_override（source 描述来源，不是数值）。"""
    from scripts import probe_corrector_repro as probe

    assert probe.resolve_budget_source(None) == "production_default"
    assert probe.resolve_budget_source(0.05) == "explicit_override"
    assert probe.resolve_budget_source(0.25) == "explicit_override"
    assert probe.resolve_budget_source(None, disabled=True) == "disabled"
    assert probe.resolve_budget_source(0.05, disabled=True) == "disabled"


def test_default_resolution_reports_the_production_default():
    from scripts import probe_corrector_repro as probe

    effective, source = probe.resolve_effective_budget(None)
    assert effective == pytest.approx(PRODUCTION_DEFAULT)
    assert source == "production_default"


def test_explicit_budget_resolution_keeps_the_override_value():
    from scripts import probe_corrector_repro as probe

    effective, source = probe.resolve_effective_budget(OVERRIDE_BUDGET)
    assert effective == pytest.approx(OVERRIDE_BUDGET), "显式 0.05 绝不能被静默替换为 0.25"
    assert source == "explicit_override"


# --- 3. release gate 只认生产默认证据 ---------------------------------------

def _obs(*, source, distinct=1, processes=6, steps=8, budget_source="production_default"):
    return {
        "source": source, "distinct": distinct, "processes": processes, "steps": steps,
        "corrector_time_limit_source": budget_source,
    }


def test_release_gate_ignores_explicit_override_observations():
    """显式 0.05 的观测**不得**算作生产默认证据。"""
    from scripts import probe_corrector_repro as probe

    gate = probe.evaluate_release_gate(
        [_obs(source="matrix.0.05", distinct=1, budget_source="explicit_override")]
    )
    assert gate["evaluated"] is False, "override 观测不能充当生产默认证据"
    assert gate["passed"] is False
    assert gate["observations"] == []


def test_release_gate_uses_production_default_observations():
    from scripts import probe_corrector_repro as probe

    gate = probe.evaluate_release_gate(
        [_obs(source="matrix.0.25", distinct=1, budget_source="production_default")]
    )
    assert gate["evaluated"] is True
    assert gate["all_qualifying"] is True
    assert gate["blocked"] is False
    assert gate["passed"] is True


def test_override_observations_are_recorded_separately_and_never_decide():
    from scripts import probe_corrector_repro as probe

    gate = probe.evaluate_release_gate(
        [_obs(source="matrix.0.25", distinct=1, budget_source="production_default")],
        override_observations=[
            _obs(source="matrix.0.05", distinct=3, budget_source="explicit_override")
        ],
    )
    assert gate["passed"] is True, "override 的不稳定不得污染生产默认判定"
    assert [o["source"] for o in gate["override_observations"]] == ["matrix.0.05"]
    assert all(
        o["corrector_time_limit_source"] == "explicit_override"
        for o in gate["override_observations"]
    )


def test_release_gate_still_fails_closed():
    """M5.4h2 的 fail-closed 规则必须保留。"""
    from scripts import probe_corrector_repro as probe

    nothing = probe.evaluate_release_gate([])
    assert nothing["evaluated"] is False and nothing["passed"] is False

    underpowered = probe.evaluate_release_gate(
        [_obs(source="modes.on", processes=2, steps=3)]
    )
    assert underpowered["all_qualifying"] is False and underpowered["passed"] is False

    unstable = probe.evaluate_release_gate([_obs(source="modes.on", distinct=2)])
    assert unstable["blocked"] is True and unstable["passed"] is False


# --- 4. Stage A/B 共享同一个 0.25 s deadline（runtime spy） ------------------

@pytest.mark.slow
def test_stages_a_and_b_share_one_deadline_at_the_production_default():
    """runtime spy：记录**真正传给** scipy.optimize.milp 的 options。"""
    from scripts import probe_corrector_repro as probe

    entry = probe.run_once("on", steps=3, budget=PRODUCTION_DEFAULT, record_stages=True)
    steps = entry["steps_detail"]
    assert steps, "必须记录每一步"
    assert len(steps) == 3

    checked = 0
    for step in steps:
        by_stage = {call["stage"]: call for call in step["calls"]}
        calls = [by_stage.get("A"), by_stage.get("B")]
        assert all(c is not None for c in calls), f"step {step['step']} 必须同时记录 A、B"
        a, b = calls
        assert "time_limit" in a["options"], "Stage A 必须收到 time_limit"
        assert "time_limit" in b["options"], "Stage B 必须收到 time_limit"
        a_limit, b_limit = a["options"]["time_limit"], b["options"]["time_limit"]
        assert 0.0 < a_limit <= PRODUCTION_DEFAULT, (
            f"Stage A 初始剩余预算必须 <= {PRODUCTION_DEFAULT}，实际 {a_limit}"
        )
        assert b_limit <= a_limit, (
            f"Stage B 不得重新分配完整预算：B={b_limit} 必须 <= A={a_limit}"
        )
        checked += 1
    assert checked == 3


@pytest.mark.slow
def test_stage_b_does_not_get_a_fresh_full_budget():
    """若 A、B 各自拿到完整 0.25，本断言必须失败。"""
    from scripts import probe_corrector_repro as probe

    entry = probe.run_once("on", steps=3, budget=PRODUCTION_DEFAULT, record_stages=True)
    pairs = [
        (call["options"]["time_limit"] for call in (step["calls"][0], step["calls"][1]))
        for step in entry["steps_detail"]
    ]
    for a_limit, b_limit in pairs:
        assert not (a_limit == pytest.approx(PRODUCTION_DEFAULT)
                    and b_limit == pytest.approx(PRODUCTION_DEFAULT)), (
            "Stage A/B 不得各自获得完整 0.25 s —— 它们共享一个总 deadline"
        )


# --- 5. 既有拒绝语义不得变化 -------------------------------------------------

def test_the_21_dim_action_contract_is_unchanged():
    from safe_rl_v2.buffer import ACTION_DIM

    assert ACTION_DIM == 21


def test_deterministic_options_are_untouched_by_this_card():
    """`planning/model.py` 的 options 构造不得被本卡改动。"""
    from planning.model import deterministic_mip_options

    options = deterministic_mip_options(time_limit_s=PRODUCTION_DEFAULT)
    assert options["time_limit"] == pytest.approx(PRODUCTION_DEFAULT)
    assert options["random_seed"] == 0
    assert options["parallel"] is False
    assert "mip_max_nodes" not in options, "不得启用生产节点上限"


# --- 6. 真实入口（slow） ----------------------------------------------------

def _read_json(run_dir: pathlib.Path, name: str) -> dict:
    return json.loads((run_dir / name).read_text(encoding="utf-8"))


@pytest.mark.slow
def test_train_corrector_on_without_a_budget_uses_the_production_default(tmp_path):
    """改前该路径**直接失败**；改后必须解析为生产默认 0.25。"""
    result = run_cli(
        "-m", "safe_rl_v2.train", "--synthetic-smoke", "--steps", "2", "--seed", "0",
        "--corrector", "on", "--base-dir", str(tmp_path), "--run-id", "prod_default",
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    run_dir = tmp_path / "prod_default"
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    report = _read_json(run_dir, "report.json")

    assert config["production_corrector_time_limit_s"] == pytest.approx(PRODUCTION_DEFAULT)
    assert config["effective_corrector_time_limit_s"] == pytest.approx(PRODUCTION_DEFAULT)
    assert config["corrector_time_limit_source"] == "production_default"
    assert report["corrector_time_limit_source"] == "production_default"


@pytest.mark.slow
def test_train_explicit_override_keeps_005_and_its_source(tmp_path):
    result = run_cli(
        "-m", "safe_rl_v2.train", "--synthetic-smoke", "--steps", "2", "--seed", "0",
        "--corrector", "on", "--corrector-time-limit-s", str(OVERRIDE_BUDGET),
        "--base-dir", str(tmp_path), "--run-id", "override",
    )
    assert result.returncode == 0, result.stderr
    run_dir = tmp_path / "override"
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))

    assert config["effective_corrector_time_limit_s"] == pytest.approx(OVERRIDE_BUDGET)
    assert config["corrector_time_limit_source"] == "explicit_override"
    assert config["production_corrector_time_limit_s"] == pytest.approx(PRODUCTION_DEFAULT)


@pytest.mark.slow
def test_train_command_ledger_does_not_fabricate_a_budget(tmp_path):
    """未传预算时，账本不得伪造 --corrector-time-limit-s。"""
    run_cli("-m", "safe_rl_v2.train", "--synthetic-smoke", "--steps", "1", "--seed", "0",
            "--corrector", "on", "--base-dir", str(tmp_path), "--run-id", "ledger_default")
    manifest = _read_json(tmp_path / "ledger_default", "manifest.json")
    assert "--corrector-time-limit-s" not in manifest["command"], manifest["command"]


@pytest.mark.slow
def test_train_command_ledger_records_an_explicit_budget(tmp_path):
    run_cli("-m", "safe_rl_v2.train", "--synthetic-smoke", "--steps", "1", "--seed", "0",
            "--corrector", "on", "--corrector-time-limit-s", str(OVERRIDE_BUDGET),
            "--base-dir", str(tmp_path), "--run-id", "ledger_override")
    manifest = _read_json(tmp_path / "ledger_override", "manifest.json")
    assert "--corrector-time-limit-s" in manifest["command"], manifest["command"]


@pytest.mark.slow
def test_smoke_defaults_to_the_production_budget_with_provenance(tmp_path):
    result = run_cli("scripts/smoke_main_chain.py", "--base-dir", str(tmp_path),
                     "--run-id", "smoke_default")
    assert result.returncode == 0, result.stderr
    run_dir = tmp_path / "smoke_default"
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    report = _read_json(run_dir, "report.json")

    for obj in (config, report):
        assert obj["production_corrector_time_limit_s"] == pytest.approx(PRODUCTION_DEFAULT)
        assert obj["effective_corrector_time_limit_s"] == pytest.approx(PRODUCTION_DEFAULT)
        assert obj["corrector_time_limit_source"] == "production_default"


@pytest.mark.slow
def test_rollout_probe_records_the_production_default(tmp_path):
    result = run_cli("scripts/probe_rollout_deterministic.py", "--steps", "2",
                     "--base-dir", str(tmp_path), "--run-id", "rollout_default")
    assert result.returncode == 0, result.stderr
    run_dir = tmp_path / "rollout_default"
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert config["effective_corrector_time_limit_s"] == pytest.approx(PRODUCTION_DEFAULT)
    assert config["corrector_time_limit_source"] == "production_default"


@pytest.mark.slow
def test_corrector_probe_defaults_to_the_production_budget(tmp_path):
    from scripts import probe_corrector_repro as probe

    parser = probe.build_parser()
    default_args = parser.parse_args([])
    effective, source = probe.resolve_effective_budget(default_args.corrector_time_limit)
    assert effective == pytest.approx(PRODUCTION_DEFAULT)
    assert source == "production_default"

    explicit_args = parser.parse_args(["--corrector-time-limit", str(OVERRIDE_BUDGET)])
    effective, source = probe.resolve_effective_budget(explicit_args.corrector_time_limit)
    assert effective == pytest.approx(OVERRIDE_BUDGET)
    assert source == "explicit_override"


@pytest.mark.slow
def test_corrector_probe_override_run_cannot_pass_the_production_gate(tmp_path):
    """显式 0.05 诊断 run 必须如实记录，但**不得**通过生产 release gate。"""
    result = run_cli(
        "scripts/probe_corrector_repro.py", "--modes", "on",
        "--corrector-time-limit", str(OVERRIDE_BUDGET),
        "--runs", "6", "--steps", "8",
        "--base-dir", str(tmp_path), "--run-id", "override_diag",
    )
    assert result.returncode != 0, "override run 不得通过生产 release gate"
    run_dir = tmp_path / "override_diag"
    report = _read_json(run_dir, "report.json")
    summary = _read_json(run_dir, "summary.json")
    manifest = _read_json(run_dir, "manifest.json")

    assert report["corrector_time_limit_source"] == "explicit_override"
    assert report["effective_corrector_time_limit_s"] == pytest.approx(OVERRIDE_BUDGET)
    assert report["release_gate"]["passed"] is False
    assert report["release_gate"]["observations"] == []
    assert manifest["status"] == "failed"
    assert summary["status"] == "failed"


# --- 7. 返修：manifest 也必须记录 provenance --------------------------------

MANIFEST_PROVENANCE_FIELDS = (
    "production_corrector_time_limit_s",
    "effective_corrector_time_limit_s",
    "corrector_time_limit_source",
)


@pytest.mark.slow
def test_train_records_provenance_in_config_report_and_manifest(tmp_path):
    """三处必须**恒等**（首轮审核：manifest 缺这三个字段）。"""
    run_cli("-m", "safe_rl_v2.train", "--synthetic-smoke", "--steps", "1", "--seed", "0",
            "--corrector", "on", "--base-dir", str(tmp_path), "--run-id", "prov")
    run_dir = tmp_path / "prov"
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    for field in MANIFEST_PROVENANCE_FIELDS:
        assert field in manifest, f"manifest 缺少 {field}"
        assert manifest[field] == config[field] == report[field], field
    assert manifest["corrector_time_limit_source"] == "production_default"


@pytest.mark.slow
def test_train_disabled_source_is_recorded_in_all_three(tmp_path):
    run_cli("-m", "safe_rl_v2.train", "--synthetic-smoke", "--steps", "1", "--seed", "0",
            "--corrector", "off", "--base-dir", str(tmp_path), "--run-id", "disabled")
    run_dir = tmp_path / "disabled"
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    for obj in (config, report, manifest):
        assert obj["corrector_time_limit_source"] == "disabled"
        assert obj["effective_corrector_time_limit_s"] is None


@pytest.mark.slow
def test_smoke_and_rollout_record_provenance_in_their_manifests(tmp_path):
    run_cli("scripts/smoke_main_chain.py", "--base-dir", str(tmp_path),
            "--run-id", "smoke_prov")
    run_cli("scripts/probe_rollout_deterministic.py", "--steps", "2",
            "--base-dir", str(tmp_path), "--run-id", "rollout_prov")
    for run_id in ("smoke_prov", "rollout_prov"):
        run_dir = tmp_path / run_id
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        for field in MANIFEST_PROVENANCE_FIELDS:
            assert field in manifest, f"{run_id} 的 manifest 缺少 {field}"
            assert manifest[field] == config[field] == report[field], (run_id, field)


@pytest.mark.slow
def test_corrector_probe_records_provenance_in_its_manifest(tmp_path):
    run_cli("scripts/probe_corrector_repro.py", "--modes", "on", "--runs", "6",
            "--steps", "8", "--base-dir", str(tmp_path), "--run-id", "probe_prov")
    run_dir = tmp_path / "probe_prov"
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for field in MANIFEST_PROVENANCE_FIELDS:
        assert field in manifest, f"manifest 缺少 {field}"
        assert manifest[field] == config[field] == report[field], field

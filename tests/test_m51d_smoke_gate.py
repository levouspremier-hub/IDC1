"""M5.1d 测试：`make smoke` 主链门禁（非训练健康检查）。

测试在 tmp_path 内以子进程运行脚本，避免污染仓库 `runs/`，并验证产物与拒绝语义。
"""

import json
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "smoke_main_chain.py"
ARTIFACTS = ("config.yaml", "metrics.parquet", "report.json", "figures", "manifest.json")
MANIFEST_KEYS = (
    "run_id",
    "revision",
    "dependency_lock_hash",
    "data_hash",
    "scenario_hash",
    "seed",
    "command",
    "status",
)
# 非训练门禁不得出现的结论性字段名
FORBIDDEN_KEY_SUBSTRINGS = (
    "converg",
    "performance",
    "accuracy",
    "improve",
    "gain",
    "return_mean",
    "policy_loss",
    "reward_mean",
    "trained_model",
)


def run_smoke(base_dir: pathlib.Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--base-dir", str(base_dir), *extra],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _only_run_dir(base_dir: pathlib.Path) -> pathlib.Path:
    dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    assert len(dirs) == 1, f"期望恰好一个 run 目录，实际 {sorted(p.name for p in dirs)}"
    return dirs[0]


@pytest.fixture(scope="module")
def gate_run(tmp_path_factory) -> pathlib.Path:
    base = tmp_path_factory.mktemp("smoke_gate")
    result = run_smoke(base, "--run-id", "gate_check", "--seed", "0")
    assert result.returncode == 0, (
        f"smoke 必须退出 0\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    return base / "gate_check"


# --- 1. 退出码与五类产物 ---

def test_smoke_exits_zero_and_writes_all_artifacts(gate_run):
    for name in ARTIFACTS:
        assert (gate_run / name).exists(), f"缺少产物 {name}"
    assert (gate_run / "figures").is_dir()


# --- 2. manifest 字段完整 ---

def test_manifest_records_full_provenance(gate_run):
    manifest = json.loads((gate_run / "manifest.json").read_text(encoding="utf-8"))
    for key in MANIFEST_KEYS:
        assert key in manifest, f"manifest 缺少字段 {key}"
    assert manifest["status"] == "success"
    assert manifest["run_id"] == "gate_check"
    assert manifest["seed"] == 0
    assert manifest["revision"] and manifest["revision"] != "unknown"
    assert manifest["dependency_lock_hash"], "必须记录 uv.lock 的 hash"
    assert manifest["data_hash"], "必须记录数据来源 hash"
    assert manifest["scenario_hash"], "必须记录场景 hash"
    assert "smoke_main_chain" in manifest["command"]


def test_config_declares_data_hash_basis(gate_run):
    """data_hash 的取值依据必须可复现地登记（本 smoke 不使用外部数据）。"""
    import yaml

    config = yaml.safe_load((gate_run / "config.yaml").read_text(encoding="utf-8"))
    assert config["data_provenance"]["external_dataset"] is False
    assert config["data_provenance"]["env_seed_kwargs"] == {
        "task_seed": 0,
        "server_seed": 0,
        "forecast_seed": 300000,
    }
    assert config["corrector_time_limit_s"] > 0.0


# --- 3. report 不得含训练/性能结论 ---

def test_report_makes_no_training_or_performance_claims(gate_run):
    report = json.loads((gate_run / "report.json").read_text(encoding="utf-8"))
    assert report["trained"] is False
    assert report["claims"] == {
        "trained": False,
        "performance_evaluated": False,
        "convergence_claimed": False,
    }
    offenders = [
        key
        for key in report
        if key != "trained" and any(s in key.lower() for s in FORBIDDEN_KEY_SUBSTRINGS)
    ]
    assert offenders == [], f"report 不得含训练/性能结论字段：{offenders}"


def test_report_lists_named_health_checks(gate_run):
    report = json.loads((gate_run / "report.json").read_text(encoding="utf-8"))
    checks = report["checks"]
    for name in ("raw_exec_dims", "access_limit", "soc_bounds", "energy_balance"):
        assert name in checks, f"缺少具名检查 {name}"
        assert all(entry["passed"] for entry in checks[name])
    assert report["steps_executed"] >= 1


# --- 4. run_id 语义 ---

def test_second_write_with_same_explicit_run_id_is_rejected(tmp_path):
    first = run_smoke(tmp_path, "--run-id", "dup_check", "--seed", "0")
    assert first.returncode == 0

    manifest_path = tmp_path / "dup_check" / "manifest.json"
    before = manifest_path.read_text(encoding="utf-8")

    second = run_smoke(tmp_path, "--run-id", "dup_check", "--seed", "0")
    assert second.returncode != 0, "同 id 的第二次成功写入必须被拒绝"
    assert manifest_path.read_text(encoding="utf-8") == before, "旧产物不得被覆盖"


def test_default_run_id_can_run_repeatedly(tmp_path):
    """不带 --run-id 时必须每次生成唯一 id：`make smoke` 可连续运行。"""
    for _ in range(2):
        result = run_smoke(tmp_path)
        assert result.returncode == 0, result.stderr
    run_dirs = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert len(run_dirs) == 2, f"两次默认运行必须产生两个 run 目录，实际 {run_dirs}"
    for name in run_dirs:
        assert (tmp_path / name / "manifest.json").exists()


def test_script_exists_so_make_smoke_gate_is_real():
    assert SCRIPT.exists(), "make smoke 声明的脚本必须真实存在"


# --- 5. 回归：失败路径、可重复性与 Makefile 接线 ---

def test_failed_check_writes_failed_manifest_and_exits_nonzero(tmp_path, monkeypatch):
    """检查失败必须写 status=failed 的 manifest 并退出非 0，不得假成功。"""
    import scripts.smoke_main_chain as smoke

    # 把接入上限容差置为不可满足，强制 access_limit 检查失败
    monkeypatch.setattr(smoke, "ACCESS_LIMIT_TOL_KW", -1.0e6)

    exit_code = smoke.main(["--base-dir", str(tmp_path), "--run-id", "fail_check", "--seed", "0"])
    assert exit_code == 1

    run_dir = tmp_path / "fail_check"
    assert (run_dir / "manifest.json").exists(), "失败的 run 也必须有 manifest"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure_classification"] == "smoke_check_failed"

    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["failure"], "失败原因必须落盘"
    assert report["checks"]["access_limit"][-1]["passed"] is False


def test_same_seed_reproduces_identical_report_and_metrics(tmp_path):
    """同种子、同场景的两次运行必须产生相同的报告与指标（可重复执行）。"""
    import pandas as pd

    first = tmp_path / "a"
    second = tmp_path / "b"
    assert run_smoke(first, "--run-id", "rep", "--seed", "0").returncode == 0
    assert run_smoke(second, "--run-id", "rep", "--seed", "0").returncode == 0

    report_a = json.loads((first / "rep" / "report.json").read_text(encoding="utf-8"))
    report_b = json.loads((second / "rep" / "report.json").read_text(encoding="utf-8"))
    assert report_a == report_b

    metrics_a = pd.read_parquet(first / "rep" / "metrics.parquet")
    metrics_b = pd.read_parquet(second / "rep" / "metrics.parquet")
    assert metrics_a.equals(metrics_b)
    assert list(metrics_a["raw_exec_differs"])  # 修正器确实在做事


def test_makefile_smoke_target_is_wired_and_its_guard_now_passes():
    """Makefile 的 smoke 目标必须调用本脚本，且其存在性前置条件现已满足。

    Makefile 未被本卡修改：它原有 `test -f scripts/smoke_main_chain.py` 守卫，
    交付脚本后该守卫自然通过。
    """
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    body = makefile.split("smoke:", 1)[1].split("\n\n", 1)[0]
    assert "scripts/smoke_main_chain.py" in body, "smoke 目标必须调用本脚本"
    assert SCRIPT.exists(), "存在性守卫必须已满足，否则 make smoke 仍会失败"

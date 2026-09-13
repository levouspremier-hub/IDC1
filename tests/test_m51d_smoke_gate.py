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

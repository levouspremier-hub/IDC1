"""M5.4a 测试：可复现的训练 CLI、合成 dry-run 与运行产物。

本卡不写 checkpoint、不实现 PPO ratio/clip/熵项、不启动正式训练。
"""

import ast
import json
import pathlib
import re
import subprocess
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAIN_MODULE = REPO_ROOT / "safe_rl_v2" / "train.py"
ARTIFACTS = ("config.yaml", "metrics.parquet", "report.json", "figures", "manifest.json")
MAKEFILE = REPO_ROOT / "Makefile"


def run_cli(*args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "safe_rl_v2.train", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )


def train_target_body() -> str:
    text = MAKEFILE.read_text(encoding="utf-8")
    return text.split("train:", 1)[1].split("\n\n", 1)[0]


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory) -> pathlib.Path:
    base = tmp_path_factory.mktemp("m54a")
    result = run_cli(
        "--synthetic-smoke", "--steps", "3", "--seed", "0",
        "--corrector", "off", "--base-dir", str(base), "--run-id", "synthetic_smoke",
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    return base / "synthetic_smoke"


# --- 1. 入口形态 ------------------------------------------------------------

def test_train_has_a_main_entry_point():
    """必须可作为 `python -m safe_rl_v2.train` 运行（而非直接跑文件）。"""
    source = TRAIN_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "main" in functions, "train.py 必须提供 main()"
    assert re.search(r'if\s+__name__\s*==\s*"__main__"', source), "train.py 必须有 __main__ 入口"


def test_makefile_runs_train_as_a_module():
    """Makefile 不得再直接执行文件路径（那会导致 sys.path 缺仓库根）。"""
    body = train_target_body()
    assert "-m safe_rl_v2.train" in body, body
    assert "python safe_rl_v2/train.py" not in body, "Makefile 仍在直接执行文件路径"
    assert "TRAIN_ARGS" in body, "Makefile 必须支持 TRAIN_ARGS 透传"


def test_module_invocation_exits_zero_with_help():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "--synthetic-smoke" in result.stdout


# --- 2. 默认必须因 M1.2 阻塞明确失败，且不回退合成数据 ----------------------

def test_default_invocation_fails_with_m12_blocker(tmp_path):
    result = run_cli("--base-dir", str(tmp_path), "--run-id", "blocked")
    assert result.returncode != 0, "默认路径不得成功"
    combined = result.stdout + result.stderr
    assert "M1.2" in combined, f"必须明确指出 M1.2 阻塞：{combined}"
    assert "manifest" in combined, combined


def test_default_invocation_writes_a_failed_manifest(tmp_path):
    run_cli("--base-dir", str(tmp_path), "--run-id", "blocked")
    manifest_path = tmp_path / "blocked" / "manifest.json"
    assert manifest_path.exists(), "失败也必须写 manifest"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure_classification"]


def test_default_invocation_does_not_fall_back_to_synthetic(tmp_path):
    """阻塞路径不得写出任何带 synthetic=true 的产物，也不得产生 success run。"""
    run_cli("--base-dir", str(tmp_path), "--run-id", "blocked2")
    run_dir = tmp_path / "blocked2"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] != "success"
    report_path = run_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report.get("synthetic") is not True
    assert not (run_dir / "metrics.parquet").exists() or manifest["status"] == "failed"


# --- 3. 合成 dry-run 的自我标注 --------------------------------------------

def test_synthetic_run_writes_all_artifacts(synthetic_run):
    for name in ARTIFACTS:
        assert (synthetic_run / name).exists(), f"缺少产物 {name}"
    assert (synthetic_run / "figures").is_dir()


def test_report_declares_synthetic_and_dry_run(synthetic_run):
    report = json.loads((synthetic_run / "report.json").read_text(encoding="utf-8"))
    assert report["synthetic"] is True
    assert report["dry_run_only"] is True
    assert report["claims"] == {
        "trained": False,
        "performance_evaluated": False,
        "convergence_claimed": False,
    }


def test_report_is_not_dressed_up_as_a_real_experiment(synthetic_run):
    report = json.loads((synthetic_run / "report.json").read_text(encoding="utf-8"))
    forbidden = ("converg", "performance", "improve", "gain", "accuracy", "return_mean")
    offenders = [
        key
        for key in report
        if key not in {"claims", "synthetic", "dry_run_only"}
        and any(token in key.lower() for token in forbidden)
    ]
    assert offenders == []
    # 必须显式声明它不代表正式训练/评估/论文结果
    statement = report.get("statement", "")
    assert "不" in statement and ("训练" in statement or "实验" in statement)


def test_config_records_everything_required_for_reproducibility(synthetic_run):
    config = yaml.safe_load((synthetic_run / "config.yaml").read_text(encoding="utf-8"))
    assert config["action_dim"] == 21
    assert config["contract_version"] == "contract-v7"

    seeds = config["env_seed_kwargs"]
    for key in ("task_seed", "server_seed", "forecast_seed"):
        assert seeds[key] is not None, f"{key} 不得为 None（env 会用熵源）"

    assert config["policy_seed"] is not None
    assert config["corrector_on"] is False
    assert config["corrector_time_limit_s"] is None
    assert config["steps"] == 3
    assert config["code_revision"] and config["code_revision"] != "unknown"
    assert config["scenario_type"]


def test_manifest_links_seed_command_and_revision(synthetic_run):
    manifest = json.loads((synthetic_run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["run_id"] == "synthetic_smoke"
    assert manifest["seed"] == 0
    assert manifest["revision"] and manifest["revision"] != "unknown"
    assert "-m safe_rl_v2.train" in manifest["command"]


def test_corrector_on_requires_explicit_time_limit(tmp_path):
    result = run_cli(
        "--synthetic-smoke", "--steps", "1", "--corrector", "on",
        "--base-dir", str(tmp_path), "--run-id", "on_without_limit",
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "corrector_time_limit_s" in combined or "time_limit" in combined, combined


def test_corrector_on_with_limit_is_recorded(tmp_path):
    result = run_cli(
        "--synthetic-smoke", "--steps", "1", "--seed", "0", "--corrector", "on",
        "--corrector-time-limit-s", "0.05",
        "--base-dir", str(tmp_path), "--run-id", "on_with_limit",
    )
    assert result.returncode == 0, result.stderr
    config_path = tmp_path / "on_with_limit" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["corrector_on"] is True
    assert config["corrector_time_limit_s"] == pytest.approx(0.05)


# --- 4. 参数错误与失败产物 --------------------------------------------------

@pytest.mark.parametrize(
    "args",
    [
        ("--synthetic-smoke", "--steps", "0"),
        ("--synthetic-smoke", "--steps", "-3"),
        ("--synthetic-smoke", "--corrector", "maybe"),
        ("--synthetic-smoke", "--corrector", "off", "--corrector-time-limit-s", "0"),
        ("--synthetic-smoke", "--corrector", "off", "--corrector-time-limit-s", "-1"),
    ],
)
def test_invalid_arguments_fail_explicitly(tmp_path, args):
    result = run_cli(*args, "--base-dir", str(tmp_path), "--run-id", "bad_args")
    assert result.returncode != 0, f"非法参数必须失败：{args}"


def test_argument_failure_after_run_id_still_writes_failed_manifest(tmp_path):
    """已获得 run_id 时，失败也必须写失败 manifest（便于事后定位）。"""
    run_cli("--synthetic-smoke", "--steps", "-3",
            "--base-dir", str(tmp_path), "--run-id", "bad_steps")
    manifest_path = tmp_path / "bad_steps" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure_classification"]


def test_successful_run_is_not_overwritten(tmp_path):
    first = run_cli("--synthetic-smoke", "--steps", "1", "--seed", "0",
                    "--base-dir", str(tmp_path), "--run-id", "dup")
    assert first.returncode == 0, first.stderr
    manifest_path = tmp_path / "dup" / "manifest.json"
    before = manifest_path.read_text(encoding="utf-8")

    second = run_cli("--synthetic-smoke", "--steps", "1", "--seed", "0",
                     "--base-dir", str(tmp_path), "--run-id", "dup")
    assert second.returncode != 0, "同 id 的第二次成功写入必须被拒绝"
    assert manifest_path.read_text(encoding="utf-8") == before


def test_default_run_id_is_unique_per_invocation(tmp_path):
    for _ in range(2):
        result = run_cli("--synthetic-smoke", "--steps", "1", "--seed", "0",
                         "--base-dir", str(tmp_path))
        assert result.returncode == 0, result.stderr
    dirs = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert len(dirs) == 2, f"两次默认运行必须产生两个 run 目录，实际 {dirs}"


# --- 5. 本卡不越界 ----------------------------------------------------------

def test_train_module_has_no_checkpoint_or_ppo_math():
    tree = ast.parse(TRAIN_MODULE.read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    literals = {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    for token in ("ratio", "clip", "entropy", "save_checkpoint", "state_dict_file"):
        assert token not in names | attrs, f"train.py 不得出现 {token}"
        assert token not in literals, f"train.py 不得出现 {token}"


def test_entry_delegates_to_the_existing_dry_run_update():
    """入口必须复用 dry_run_update，不得另写一套训练循环。"""
    tree = ast.parse(TRAIN_MODULE.read_text(encoding="utf-8"))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "dry_run_update" in called


# --- 6. 文档纠正（要求 7） --------------------------------------------------

def test_docstring_does_not_claim_preflight_precedes_sampling_forward():
    """预检在 rollout 收集**之后**，不得声称它在采样前向之前。"""
    source = TRAIN_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree) or ""
    assert "任何 forward" not in docstring
    assert "rollout" in docstring and "预检" in docstring

    # 行内注释同样不得声称「任何 forward 之前」
    assert "在任何** forward" not in source
    assert "在任何 forward" not in source


def test_docstring_describes_the_actual_preflight_position():
    docstring = ast.get_docstring(ast.parse(TRAIN_MODULE.read_text(encoding="utf-8"))) or ""
    assert "收集" in docstring or "rollout" in docstring
    assert "更新" in docstring

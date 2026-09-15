"""M5.4a 测试：可复现的训练 CLI、合成 dry-run 与运行产物。

本卡不写 checkpoint、不实现 PPO ratio/clip/熵项、不启动正式训练。
"""

import ast
import json
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from contracts import CONTRACT_VERSION_ID

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
    assert config["contract_version"] == CONTRACT_VERSION_ID

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


def test_corrector_on_without_a_budget_resolves_to_the_production_default(tmp_path):
    """M5.4i 迁移：`--corrector on` 未给预算**不再报错**，而是解析为生产默认 0.25。

    改前该路径抛 `TrainEntryError`；新语义见 `docs/task_cards/M5.4i.md`。
    """
    result = run_cli(
        "--synthetic-smoke", "--steps", "1", "--corrector", "on",
        "--base-dir", str(tmp_path), "--run-id", "on_without_limit",
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    config = yaml.safe_load(
        (tmp_path / "on_without_limit" / "config.yaml").read_text(encoding="utf-8")
    )
    assert config["effective_corrector_time_limit_s"] == pytest.approx(0.25)
    assert config["corrector_time_limit_source"] == "production_default"


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


def test_failure_after_run_id_allocation_writes_failed_manifest(tmp_path):
    """已获得 run_id 的运行期失败，也必须写失败 manifest（便于事后定位）。

    注意：argparse 层面的参数错误发生在 run_id 分配**之前**，此时不存在 run_id，
    故不写 manifest（其非 0 退出由 test_invalid_arguments_fail_explicitly 覆盖）。
    """
    # M5.4i 迁移：改用**正式数据路径**（M1.3 未完成）作为 run_id 分配后的运行期失败。
    # 原先把「corrector on 缺预算」当作失败源，该路径现已解析为生产默认、不再失败。
    result = run_cli(
        "--steps", "1", "--corrector", "off",
        "--base-dir", str(tmp_path), "--run-id", "real_blocked_manifest",
    )
    assert result.returncode != 0
    manifest_path = tmp_path / "real_blocked_manifest" / "manifest.json"
    assert manifest_path.exists(), "run_id 已分配后的失败必须落失败 manifest"
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
    """预检在 rollout 收集**之后**：不得声称它在采样前向之前。"""
    source = TRAIN_MODULE.read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source)) or ""

    # 若出现「任何 forward」，必须带上「更新阶段」这一限定
    if "任何 forward" in docstring:
        assert "更新阶段" in docstring, "不得笼统声称预检在任何 forward 之前"
    # 必须显式点明它**不**在采样前向之前
    assert "不在采样前向之前" in docstring or "采样前向" in docstring, docstring
    assert "collect_rollout" in docstring
    # 行内注释同样必须限定为更新阶段
    assert "在任何** forward / backward / optimizer.step 之前" not in source
    assert "更新阶段**的 forward" in source


def test_docstring_describes_the_actual_preflight_position():
    docstring = ast.get_docstring(ast.parse(TRAIN_MODULE.read_text(encoding="utf-8"))) or ""
    assert "rollout 收集之后" in docstring, docstring
    assert "更新阶段" in docstring, docstring
    # 必须说明它保证的是「更新阶段零副作用」
    assert "零副作用" in docstring


# --- 7. 回归：全局 RNG 还原、无 checkpoint、metrics 形状 ---------------------

def test_global_torch_rng_is_restored_after_a_synthetic_dry_run():
    """权重初始化借用全局 RNG，但必须用 fork_rng 还原：调用方状态不受影响。"""
    from safe_rl_v2 import train as train_mod

    args = train_mod.build_parser().parse_args(
        ["--synthetic-smoke", "--steps", "1", "--seed", "3", "--corrector", "off"]
    )
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    train_mod._run_synthetic_dry_run(
        args, "rng_probe", {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}
    )
    assert torch.equal(before, torch.get_rng_state()), "全局 Torch RNG 必须被还原"


def test_same_seed_produces_the_same_weights_across_calls():
    """fork_rng + manual_seed 的意义：同 seed 的权重初始化必须逐位一致。"""
    from safe_rl_v2 import train as train_mod

    args = train_mod.build_parser().parse_args(
        ["--synthetic-smoke", "--steps", "1", "--seed", "5", "--corrector", "off"]
    )
    seeds = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}
    first = train_mod._run_synthetic_dry_run(args, "a", seeds)
    second = train_mod._run_synthetic_dry_run(args, "b", seeds)
    np.testing.assert_allclose(
        first["buffer"].transitions[0].raw_action, second["buffer"].transitions[0].raw_action
    )


def test_synthetic_run_writes_no_checkpoint_artifacts(synthetic_run):
    """本卡不写 checkpoint：产物目录里不得出现模型权重文件。"""
    offenders = [
        path.name
        for path in synthetic_run.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".bin"}
    ]
    assert offenders == [], f"本卡不得写 checkpoint：{offenders}"


def test_metrics_parquet_has_one_row_per_transition(synthetic_run):
    import pandas as pd

    metrics = pd.read_parquet(synthetic_run / "metrics.parquet")
    assert len(metrics) == 3  # fixture 用 --steps 3
    for column in (
        "reward", "business_violations", "carbon_emissions_kg",
        "electricity_cost_sgd", "old_raw_log_prob", "raw_exec_differs",
        "terminated", "truncated",
    ):
        assert column in metrics.columns, column


def test_config_and_report_agree_on_scenario_type(synthetic_run):
    config = yaml.safe_load((synthetic_run / "config.yaml").read_text(encoding="utf-8"))
    report = json.loads((synthetic_run / "report.json").read_text(encoding="utf-8"))
    assert config["scenario_type"] == report["scenario_type"]
    assert config["synthetic"] is True and report["synthetic"] is True


def test_blocked_default_run_never_writes_a_success_manifest(tmp_path):
    run_cli("--base-dir", str(tmp_path), "--run-id", "blocked3")
    manifest = json.loads((tmp_path / "blocked3" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure_classification"] == "TrainEntryError"

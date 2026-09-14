"""M5.4e 测试：探针必须记录它**真正执行**的参数。

M5.4d 的探针把 `--steps` 写进 config，却没把它传给子进程（子进程用默认 8），
于是 config 说 1 步、metrics/provenance 却是 8 步；`--steps 0` 还会写出
「声称跑 0 步的成功 run」。本卡修的是**参数保真**。
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
STEPS_CASES = (1, 3, 8)
ARTIFACTS = ("config.yaml", "metrics.parquet", "report.json", "figures", "manifest.json")


def run_probe(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/probe_corrector_repro.py", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )


def assert_exit_code_is_explained_by_the_gate(
    result: subprocess.CompletedProcess, run_dir: pathlib.Path, run_id: str
) -> dict:
    """M5.4h2 返修迁移（范围外，已登记）。

    退出码现在由 `release_gate.passed` 决定，不再是恒 0：本文件的 run 只测
    corrector 关闭（`--modes off`）或样本量不足，门禁因此 fail closed -> 退出码 1。
    本文件的主题是**参数保真**，与门禁判定无关，故改为**更严**的断言：

    - 产物必须真的写出来（探针崩溃时不会有 `report.json`，仍会被抓住）；
    - 退出码必须**恰好**等于门禁的判定，不能由任何其他失败解释。
    """
    report_path = run_dir / "report.json"
    assert report_path.exists(), (
        f"run {run_id} 未产出 report.json（探针可能崩溃）："
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert (run_dir / "manifest.json").exists(), f"run {run_id} 未产出 manifest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    passed = report["release_gate"]["passed"]
    assert result.returncode == (0 if passed else 1), (
        f"退出码必须由 release_gate.passed 决定：passed={passed} "
        f"returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    return report


def probe_run(tmp_path: pathlib.Path, run_id: str, steps: int, *extra: str):
    result = run_probe(
        "--modes", "off", "--runs", "1", "--steps", str(steps),
        "--base-dir", str(tmp_path), "--run-id", run_id, "--emit-provenance", *extra,
    )
    run_dir = tmp_path / run_id
    assert_exit_code_is_explained_by_the_gate(result, run_dir, run_id)
    return run_dir


# --- 1. steps 必须一致地记录在每一处 ----------------------------------------

@pytest.mark.parametrize("steps", STEPS_CASES)
def test_every_artifact_records_the_same_actual_steps(tmp_path, steps):
    run_dir = probe_run(tmp_path, f"steps{steps}", steps)

    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    metrics = pd.read_parquet(run_dir / "metrics.parquet")
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))

    assert config["steps"] == steps
    assert int(metrics["steps"].iloc[0]) == steps
    assert provenance[0]["steps"] == steps
    assert report["steps"] == steps
    assert {config["steps"], int(metrics["steps"].iloc[0]),
            provenance[0]["steps"], report["steps"]} == {steps}


@pytest.mark.parametrize("steps", STEPS_CASES)
def test_run_once_honours_the_requested_steps(steps):
    entry = probe.run_once("off", steps=steps)
    assert entry["steps"] == steps


def test_run_once_requires_explicit_steps():
    """不得有静默默认：调用方必须显式给出步数。"""
    with pytest.raises(TypeError):
        probe.run_once("off")  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", [0, -1, -3])
def test_run_once_rejects_non_positive_steps(bad):
    with pytest.raises(ValueError):
        probe.run_once("off", steps=bad)


def test_run_in_subprocess_passes_the_steps_through():
    entry = probe.run_in_subprocess("off", steps=2)
    assert entry["steps"] == 2


# --- 2. manifest.command 必须是完整账本 -------------------------------------

@pytest.mark.parametrize("steps", STEPS_CASES)
def test_command_ledger_records_every_effective_parameter(tmp_path, steps):
    run_dir = probe_run(tmp_path, f"cmd{steps}", steps)
    command = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["command"]
    parts = shlex.split(command)

    for flag in ("--modes", "--runs", "--steps", "--base-dir", "--run-id", "--emit-provenance"):
        assert flag in parts, f"command 缺少 {flag}：{command}"

    def value_of(flag: str) -> str:
        return parts[parts.index(flag) + 1]

    assert value_of("--steps") == str(steps)
    assert value_of("--base-dir") == str(tmp_path)
    assert value_of("--run-id") == f"cmd{steps}"
    assert value_of("--modes") == "off"
    assert value_of("--runs") == "1"


def test_emit_provenance_is_omitted_from_the_ledger_when_not_requested(tmp_path):
    result = run_probe("--modes", "off", "--runs", "1", "--steps", "1",
                       "--base-dir", str(tmp_path), "--run-id", "nochar")
    assert_exit_code_is_explained_by_the_gate(result, tmp_path / "nochar", "nochar")
    command = json.loads((tmp_path / "nochar" / "manifest.json").read_text(encoding="utf-8"))[
        "command"
    ]
    assert "--emit-provenance" not in shlex.split(command)
    assert not (tmp_path / "nochar" / "provenance.json").exists()


def test_manifest_command_can_be_replayed(tmp_path):
    """账本必须真的可重放：用它的 argv 重跑，config 除 revision 外应一致。"""
    first_base = tmp_path / "first"
    run_dir = probe_run(first_base, "replay", 3)
    command = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["command"]

    argv = shlex.split(command)
    argv[argv.index("--base-dir") + 1] = str(tmp_path / "second")
    replay = subprocess.run([sys.executable, *argv[1:]], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=600)
    assert_exit_code_is_explained_by_the_gate(
        replay, tmp_path / "second" / "replay", "replay"
    )

    first_cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    second_cfg = yaml.safe_load(
        (tmp_path / "second" / "replay" / "config.yaml").read_text(encoding="utf-8")
    )
    assert first_cfg == second_cfg, "重放产生了不同的 config"


# --- 3. 非法 --steps 必须明确失败且不写产物 ---------------------------------

@pytest.mark.parametrize("bad", ["0", "-3"])
def test_non_positive_steps_fail_without_writing_artifacts(tmp_path, bad):
    result = run_probe("--modes", "off", "--runs", "1", "--steps", bad,
                       "--base-dir", str(tmp_path), "--run-id", f"bad{bad}")
    assert result.returncode != 0, "非法 --steps 必须失败"
    assert not (tmp_path / f"bad{bad}").exists(), "非法参数不得写出任何产物"
    combined = result.stdout + result.stderr
    assert "steps" in combined


def test_non_positive_steps_do_not_write_a_success_manifest(tmp_path):
    run_probe("--modes", "off", "--runs", "1", "--steps", "0",
              "--base-dir", str(tmp_path), "--run-id", "zero")
    assert not (tmp_path / "zero" / "manifest.json").exists()


# --- 4. 默认步数仍然是可用的（回归） ----------------------------------------

def test_default_steps_still_works(tmp_path):
    result = run_probe("--modes", "off", "--runs", "1",
                       "--base-dir", str(tmp_path), "--run-id", "default")
    assert_exit_code_is_explained_by_the_gate(result, tmp_path / "default", "default")
    config = yaml.safe_load((tmp_path / "default" / "config.yaml").read_text(encoding="utf-8"))
    metrics = pd.read_parquet(tmp_path / "default" / "metrics.parquet")
    assert config["steps"] == probe.STEPS == int(metrics["steps"].iloc[0])


# --- 5. 回归：**每个**记录下来的参数都必须与实际执行一致 --------------------

def _config_of(tmp_path, result):
    return yaml.safe_load((result / "config.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("modes", [("off",), ("on",), ("off", "on")])
@pytest.mark.parametrize("runs", [1, 2])
def test_modes_and_runs_match_the_actual_metrics_rows(tmp_path, modes, runs):
    """config 里的 modes / runs_per_mode 必须等于 metrics 里真实的行数。"""
    run_id = f"mr_{'_'.join(modes)}_{runs}"
    result = run_probe("--modes", *modes, "--runs", str(runs), "--steps", "1",
                       "--base-dir", str(tmp_path), "--run-id", run_id)

    run_dir = tmp_path / run_id
    assert_exit_code_is_explained_by_the_gate(result, run_dir, run_id)
    config = _config_of(tmp_path, run_dir)
    metrics = pd.read_parquet(run_dir / "metrics.parquet")
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))

    assert config["modes"] == list(modes)
    assert config["runs_per_mode"] == runs
    # M5.4i 迁移：默认路径现在也会产出**逐阶段审计行**（生产默认档），
    # 它们没有 `mode` 列值。因此「modes × runs」的核对必须在**模式行**子集上做，
    # 断言强度不变（config 仍必须等于真实执行的模式行数）。
    mode_rows = metrics[metrics["mode"].notna()]
    assert len(mode_rows) == runs * len(modes), "模式行数必须等于 modes × runs"
    assert sorted(mode_rows["mode"].unique()) == sorted(modes)
    for mode in modes:
        assert (mode_rows["mode"] == mode).sum() == runs
        assert report["modes"][mode]["runs"] == runs
        assert len(report["modes"][mode]["digests"]) == runs


def test_probe_has_no_silent_default_for_steps_anywhere():
    """结构性保证：源码里不得出现 `steps=` 的默认值。"""
    import ast

    tree = ast.parse(pathlib.Path(probe.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_once":
            steps_arg = next(a for a in node.args.kwonlyargs if a.arg == "steps")
            default = node.args.kw_defaults[node.args.kwonlyargs.index(steps_arg)]
            assert default is None, "run_once 的 steps 不得有默认值"
        if isinstance(node, ast.FunctionDef) and node.name == "run_in_subprocess":
            steps_arg = next(a for a in node.args.kwonlyargs if a.arg == "steps")
            default = node.args.kw_defaults[node.args.kwonlyargs.index(steps_arg)]
            assert default is None, "run_in_subprocess 的 steps 不得有默认值"


def test_recorded_parameters_are_the_ones_actually_executed(tmp_path):
    """端到端：changed 参数必须同时出现在产物里 —— 不能只写不执行。"""
    for steps in STEPS_CASES:
        run_id = f"exec_{steps}"
        result = run_probe("--modes", "off", "--runs", "1", "--steps", str(steps),
                           "--base-dir", str(tmp_path), "--run-id", run_id,
                           "--emit-provenance")
        run_dir = tmp_path / run_id
        assert_exit_code_is_explained_by_the_gate(result, run_dir, run_id)
        metrics = pd.read_parquet(run_dir / "metrics.parquet")
        provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        # metrics/provenance 的 steps 来自**实际采集到的 transition 数**，
        # 而题设 `--steps` 必须与之相等 —— 这正是「写的就是跑的」。
        assert int(metrics["steps"].iloc[0]) == steps
        assert provenance[0]["steps"] == steps
        # 每个模式的 digest 数也要与 runs 一致
        assert len(provenance) == 1

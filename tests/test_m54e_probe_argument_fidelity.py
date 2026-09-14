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


def probe_run(tmp_path: pathlib.Path, run_id: str, steps: int, *extra: str):
    result = run_probe(
        "--modes", "off", "--runs", "1", "--steps", str(steps),
        "--base-dir", str(tmp_path), "--run-id", run_id, "--emit-provenance", *extra,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    return tmp_path / run_id


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
    assert result.returncode == 0, result.stderr
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
    assert replay.returncode == 0, replay.stderr

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
    assert result.returncode == 0, result.stderr
    config = yaml.safe_load((tmp_path / "default" / "config.yaml").read_text(encoding="utf-8"))
    metrics = pd.read_parquet(tmp_path / "default" / "metrics.parquet")
    assert config["steps"] == probe.STEPS == int(metrics["steps"].iloc[0])

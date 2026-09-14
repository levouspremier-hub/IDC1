"""M5.4c 测试：运行产物防覆盖与可复现命令账本。

本卡不触碰 env / planning / corrector / 动作空间 / buffer / PPO 数学 / 契约版本。
"""

import json
import pathlib
import shlex
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from runs.writer import write_run
from safe_rl_v2 import train as train_mod

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

ARTIFACT_NAMES = ("config.yaml", "metrics.parquet", "report.json", "manifest.json")


def run_cli(*args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "safe_rl_v2.train", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )


def artifact_bytes(run_dir: pathlib.Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(run_dir.iterdir())
        if path.is_file()
    }


def make_success(base: pathlib.Path, run_id: str = "X", **over) -> pathlib.Path:
    kwargs = dict(
        config={"marker": "success"}, metrics=pd.DataFrame([{"a": 1}]),
        report={"synthetic": True, "statement": "ok"}, base_dir=str(base),
        seed=0, command="first success", status="success",
    )
    kwargs.update(over)
    return write_run(run_id, **kwargs)


# --- 1. 已有成功 run 不得被任何状态覆盖 --------------------------------------

def test_failed_run_cannot_overwrite_a_successful_run(tmp_path):
    run_dir = make_success(tmp_path)
    before = artifact_bytes(run_dir)

    with pytest.raises(FileExistsError):
        write_run(
            "X", config={"marker": "failed"}, metrics=pd.DataFrame(),
            report={"statement": "failure"}, base_dir=str(tmp_path),
            seed=0, command="failed retry", status="failed",
            failure_classification="Boom",
        )

    assert artifact_bytes(run_dir) == before, "成功 run 的产物被改写"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["command"] == "first success"


def test_successful_run_cannot_be_overwritten_by_success(tmp_path):
    run_dir = make_success(tmp_path)
    before = artifact_bytes(run_dir)
    with pytest.raises(FileExistsError):
        make_success(tmp_path, command="second success")
    assert artifact_bytes(run_dir) == before


@pytest.mark.parametrize("status", ["success", "failed", "running", "aborted"])
def test_no_status_may_overwrite_a_success(tmp_path, status):
    run_dir = make_success(tmp_path)
    before = artifact_bytes(run_dir)
    with pytest.raises(FileExistsError):
        write_run(
            "X", config={}, metrics=pd.DataFrame(), report={},
            base_dir=str(tmp_path), status=status,
        )
    assert artifact_bytes(run_dir) == before


def test_failed_run_may_be_retried_and_may_become_success(tmp_path):
    """失败 run 不是永久封印：同 id 允许重试，修好后也允许转为成功。"""
    write_run("Y", config={}, metrics=pd.DataFrame(), report={},
              base_dir=str(tmp_path), status="failed", failure_classification="Boom")
    write_run("Y", config={}, metrics=pd.DataFrame(), report={},
              base_dir=str(tmp_path), status="failed", failure_classification="Boom2")
    write_run("Y", config={"ok": True}, metrics=pd.DataFrame(), report={},
              base_dir=str(tmp_path), status="success")

    manifest = json.loads((tmp_path / "Y" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"


# --- 2. train 失败路径遇到同名成功 run 时必须保留并明确报告 ------------------

def test_train_failure_preserves_an_existing_successful_run(tmp_path):
    """先成功创建 run_id=X，再以同一 X 触发运行期失败：成功产物逐字节保留。"""
    ok = run_cli("--synthetic-smoke", "--steps", "1", "--seed", "0",
                 "--corrector", "off", "--base-dir", str(tmp_path), "--run-id", "X")
    assert ok.returncode == 0, ok.stderr
    run_dir = tmp_path / "X"
    before = artifact_bytes(run_dir)

    # 同一 id、运行期失败（corrector on 缺预算）
    bad = run_cli("--synthetic-smoke", "--steps", "1", "--seed", "0",
                  "--corrector", "on", "--base-dir", str(tmp_path), "--run-id", "X")
    assert bad.returncode != 0

    assert artifact_bytes(run_dir) == before, "失败复跑销毁了成功 run 的产物"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"


def test_train_failure_reports_the_conflict_explicitly(tmp_path):
    """不得把「保留了成功结果」写成「失败 manifest 已写入」。"""
    run_cli("--synthetic-smoke", "--steps", "1", "--seed", "0", "--corrector", "off",
            "--base-dir", str(tmp_path), "--run-id", "X")
    bad = run_cli("--synthetic-smoke", "--steps", "1", "--seed", "0", "--corrector", "on",
                  "--base-dir", str(tmp_path), "--run-id", "X")
    combined = bad.stdout + bad.stderr
    assert "已存在成功" in combined or "冲突" in combined, combined
    assert bad.returncode == 3, f"run_id 冲突必须返回专门退出码，实际 {bad.returncode}"


def test_failed_run_is_still_written_when_there_is_no_conflict(tmp_path):
    """无冲突时失败 manifest 仍必须写（本卡不允许靠不写来避免覆盖）。"""
    bad = run_cli("--synthetic-smoke", "--steps", "1", "--corrector", "on",
                  "--base-dir", str(tmp_path), "--run-id", "fresh_fail")
    assert bad.returncode != 0
    manifest = json.loads((tmp_path / "fresh_fail" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure_classification"]


# --- 3. manifest.command 必须是完整可重放的调用账本 --------------------------

COMMAND_ARGV_SCRIPT = """
import json, sys
from safe_rl_v2.train import build_parser, _effective_argv

args = build_parser().parse_args(sys.argv[1:])
print(json.dumps(_effective_argv(args, args.run_id or "generated_run_id")))
"""


def _effective_argv(cli_args: list[str]) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", COMMAND_ARGV_SCRIPT, *cli_args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_command_ledger_covers_every_cli_action():
    """结构性保证：命令账本清单必须覆盖 build_parser() 的每一个 action。"""
    parser = train_mod.build_parser()
    declared = {flag for flag, _dest, _kind in train_mod.COMMAND_ARGV_SPEC}
    actual = {
        flag
        for action in parser._actions
        for flag in action.option_strings
        if action.dest != "help"
    }
    assert declared == actual, f"命令账本缺少 {actual - declared}，多出 {declared - actual}"


def test_effective_argv_round_trips_through_the_parser():
    """生成的 argv 必须能被同一个 parser 重新解析为等价 namespace。"""
    cli_args = [
        "--synthetic-smoke", "--steps", "5", "--seed", "7", "--corrector", "on",
        "--corrector-time-limit-s", "0.25", "--base-dir", "runs_x", "--run-id", "r1",
        "--task-seed", "1", "--server-seed", "2", "--forecast-seed", "3", "--horizon", "12",
    ]
    argv = _effective_argv(cli_args)
    assert argv[:3] == ["python", "-m", "safe_rl_v2.train"]

    parser = train_mod.build_parser()
    rep = parser.parse_args(argv[3:])
    original = parser.parse_args(cli_args)
    for key, value in vars(original).items():
        assert getattr(rep, key) == value, f"{key} 未被账本完整保留"


def test_effective_argv_is_shell_safe():
    """含空格等字符时必须可被 shlex.split 还原（安全转义，不是裸拼）。"""
    argv = _effective_argv(["--synthetic-smoke", "--base-dir", "dir with spaces",
                            "--run-id", "id;rm -rf /", "--steps", "1"])
    joined = shlex.join(argv)
    restored = shlex.split(joined)
    assert restored == argv, "command 未经安全转义"
    assert restored[restored.index("--base-dir") + 1] == "dir with spaces"
    assert restored[restored.index("--run-id") + 1] == "id;rm -rf /"


def test_manifest_command_records_every_parameter(tmp_path):
    result = run_cli(
        "--synthetic-smoke", "--steps", "2", "--seed", "3", "--corrector", "off",
        "--base-dir", str(tmp_path), "--run-id", "ledger",
        "--task-seed", "11", "--server-seed", "12", "--forecast-seed", "13", "--horizon", "6",
    )
    assert result.returncode == 0, result.stderr
    command = json.loads((tmp_path / "ledger" / "manifest.json").read_text(encoding="utf-8"))[
        "command"
    ]
    parts = shlex.split(command)
    for flag in ("--synthetic-smoke", "--steps", "--seed", "--corrector", "--base-dir",
                 "--run-id", "--task-seed", "--server-seed", "--forecast-seed", "--horizon"):
        assert flag in parts, f"command 缺少 {flag}：{command}"

    def value_of(flag: str) -> str:
        return parts[parts.index(flag) + 1]

    assert value_of("--steps") == "2"
    assert value_of("--seed") == "3"
    assert value_of("--base-dir") == str(tmp_path)
    assert value_of("--run-id") == "ledger"
    assert value_of("--task-seed") == "11"
    assert value_of("--server-seed") == "12"
    assert value_of("--forecast-seed") == "13"
    assert value_of("--horizon") == "6"


def test_manifest_command_can_be_replayed(tmp_path):
    """账本必须真的可重放：用 command 的 argv 重跑同一配置，产物配置应一致。"""
    base = tmp_path / "first"
    result = run_cli("--synthetic-smoke", "--steps", "2", "--seed", "0", "--corrector", "off",
                     "--base-dir", str(base), "--run-id", "a")
    assert result.returncode == 0, result.stderr
    command = json.loads((base / "a" / "manifest.json").read_text(encoding="utf-8"))["command"]

    argv = shlex.split(command)
    argv[argv.index("--base-dir") + 1] = str(tmp_path / "second")
    replay = subprocess.run([sys.executable, "-m", "safe_rl_v2.train", *argv[3:]],
                            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert replay.returncode == 0, replay.stderr

    first_cfg = yaml.safe_load((base / "a" / "config.yaml").read_text(encoding="utf-8"))
    second_cfg = yaml.safe_load(
        (tmp_path / "second" / "a" / "config.yaml").read_text(encoding="utf-8")
    )
    assert {k: v for k, v in first_cfg.items() if k != "code_revision"} == {
        k: v for k, v in second_cfg.items() if k != "code_revision"
    }


# --- 4. 默认（真实）路径不得使用 synthetic 前缀 ------------------------------

def test_default_blocked_run_id_is_not_labelled_synthetic(tmp_path):
    result = run_cli("--base-dir", str(tmp_path))  # 无 --run-id，默认真实路径
    assert result.returncode != 0
    run_dirs = [p.name for p in tmp_path.iterdir() if p.is_dir()]
    assert run_dirs, "失败也必须落一个目录"
    for name in run_dirs:
        assert not name.startswith("train_synthetic"), f"默认真实路径不得叫 synthetic：{name}"
        assert name.startswith("train_real"), f"默认真实路径应以 train_real 前缀：{name}"


def test_synthetic_run_id_keeps_its_prefix(tmp_path):
    result = run_cli("--synthetic-smoke", "--steps", "1", "--seed", "0",
                     "--corrector", "off", "--base-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    run_dirs = [p.name for p in tmp_path.iterdir() if p.is_dir()]
    assert all(name.startswith("train_synthetic") for name in run_dirs), run_dirs


def test_explicit_run_id_is_used_verbatim_on_both_paths(tmp_path):
    ok = run_cli("--synthetic-smoke", "--steps", "1", "--corrector", "off",
                 "--base-dir", str(tmp_path), "--run-id", "explicit_ok")
    assert ok.returncode == 0
    bad = run_cli("--base-dir", str(tmp_path), "--run-id", "explicit_blocked")
    assert bad.returncode != 0
    assert (tmp_path / "explicit_ok" / "manifest.json").exists()
    blocked = json.loads(
        (tmp_path / "explicit_blocked" / "manifest.json").read_text(encoding="utf-8")
    )
    assert blocked["status"] == "failed"

"""M5.4i 发布验收：跨批聚合的机器可读判据。

单次抽样可能是幸运的（0.05 s 的不稳定是间歇的，见 M5.4h2 §9.9），因此 M5.4 发布
证据必须按负载做**多批聚合**：每个单批 `distinct=1` **且** 跨批 `aggregate distinct=1`
**且** 零 `time_limit` **且** 全部批次确实来自**生产默认**（不得用显式 override 伪造默认）。

本文件的慢速用例会真实跑批次；快速用例只验证聚合器的**判据**是否如实。
"""

import json
import pathlib
import subprocess
import sys

import pandas as pd
import pytest

from scripts import probe_corrector_repro as probe

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PRODUCTION_DEFAULT = 0.25


def _fake_run(root: pathlib.Path, run_id: str, *, digests, passed=True,
              source="production_default", budget=PRODUCTION_DEFAULT,
              manifest_status=None, statuses=(0,)) -> pathlib.Path:
    """构造一个最小 run 目录（只含聚合器需要的产物），用于判据测试。"""
    run_dir = root / run_id
    (run_dir).mkdir(parents=True, exist_ok=True)
    report = {
        "effective_corrector_time_limit_s": budget,
        "corrector_time_limit_source": source,
        "release_gate": {
            "passed": passed,
            "observations": [{
                "source": "modes.on", "digests": list(digests),
                "distinct": len(set(digests)), "processes": len(digests), "steps": 8,
            }],
        },
    }
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id,
                    "status": manifest_status or ("success" if passed else "failed")}),
        encoding="utf-8",
    )
    pd.DataFrame([{"status": s} for s in statuses]).to_parquet(run_dir / "summary.parquet")
    return run_dir


# --- 1. 聚合判据（纯函数） ---------------------------------------------------

def test_aggregation_passes_only_when_every_batch_and_the_pool_agree(tmp_path):
    # 可复现的批次：同一批次内 6 个独立进程给出**同一个** digest。
    runs = [_fake_run(tmp_path, f"b{i}", digests=["a"] * 6) for i in range(3)]
    aggregated = probe.aggregate_release_batches(runs)
    assert aggregated["batches"] == 3
    assert aggregated["aggregate_processes"] == 18
    assert aggregated["aggregate_distinct"] == 1
    assert aggregated["every_batch_distinct_one"] is True
    assert aggregated["time_limit_failures"] == 0
    assert aggregated["passed"] is True


def test_aggregation_fails_when_any_single_batch_forks(tmp_path):
    runs = [
        _fake_run(tmp_path, "b0", digests=["a"] * 3),
        _fake_run(tmp_path, "b1", digests=["a", "a", "Z"]),  # 单批内部分叉
        _fake_run(tmp_path, "b2", digests=["a"] * 3),
    ]
    aggregated = probe.aggregate_release_batches(runs)
    assert aggregated["aggregate_distinct"] == 2
    assert aggregated["every_batch_distinct_one"] is False
    assert aggregated["passed"] is False


def test_aggregation_fails_when_batches_agree_internally_but_not_with_each_other(tmp_path):
    """每批各自 distinct=1，但批与批不同 —— 聚合仍必须失败。"""
    runs = [
        _fake_run(tmp_path, "b0", digests=["a", "a", "a"]),
        _fake_run(tmp_path, "b1", digests=["b", "b", "b"]),
    ]
    aggregated = probe.aggregate_release_batches(runs)
    assert aggregated["every_batch_distinct_one"] is True
    assert aggregated["aggregate_distinct"] == 2
    assert aggregated["passed"] is False


def test_aggregation_fails_on_any_time_limit_failure(tmp_path):
    runs = [
        _fake_run(tmp_path, "b0", digests=["a", "a"], statuses=(0, 0)),
        _fake_run(tmp_path, "b1", digests=["a", "a"], statuses=(1, 0)),  # 一次 time_limit
    ]
    aggregated = probe.aggregate_release_batches(runs)
    assert aggregated["time_limit_failures"] == 1
    assert aggregated["passed"] is False


def test_aggregation_requires_production_default_evidence(tmp_path):
    """显式 override 的批次**不得**充当生产默认发布证据。"""
    runs = [
        _fake_run(tmp_path, "b0", digests=["a", "a"]),
        _fake_run(tmp_path, "b1", digests=["a", "a"],
                  source="explicit_override", budget=0.05),
    ]
    aggregated = probe.aggregate_release_batches(runs)
    assert aggregated["production_default_only"] is False
    assert aggregated["passed"] is False, "显式 override 不得伪造生产默认通过"


def test_aggregation_requires_the_production_default_budget(tmp_path):
    runs = [_fake_run(tmp_path, "b0", digests=["a", "a"], budget=0.05)]
    aggregated = probe.aggregate_release_batches(runs)
    assert aggregated["effective_corrector_time_limit_s"] == [0.05]
    assert aggregated["passed"] is False


def test_aggregation_of_nothing_does_not_pass():
    assert probe.aggregate_release_batches([])["passed"] is False


# --- 2. 真实批次（slow） ----------------------------------------------------

def run_batch(base_dir: pathlib.Path, run_id: str, *extra: str,
              timeout: int = 3600) -> subprocess.CompletedProcess:
    """一个发布批次：`--modes on`（生产默认，不给任何显式预算）+ ≥6 进程 + 8 步。"""
    return subprocess.run(
        [sys.executable, "scripts/probe_corrector_repro.py",
         "--modes", "on", "--runs", "6", "--steps", "8",
         "--base-dir", str(base_dir), "--run-id", run_id, *extra],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )


@pytest.mark.slow
def test_a_default_batch_uses_the_production_default_and_writes_full_artifacts(tmp_path):
    result = run_batch(tmp_path, "m54i_batch")
    run_dir = tmp_path / "m54i_batch"
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert report["effective_corrector_time_limit_s"] == pytest.approx(PRODUCTION_DEFAULT)
    assert report["corrector_time_limit_source"] == "production_default"
    assert report["release_gate"]["observations"], "批次必须产出生产默认观测"
    assert all(
        o["corrector_time_limit_source"] == "production_default"
        for o in report["release_gate"]["observations"]
    ), "默认批次的观测必须全部来自生产默认"

    assert (result.returncode == 0) is report["release_gate"]["passed"]
    assert manifest["status"] == ("success" if report["release_gate"]["passed"] else "failed")
    assert summary["status"] == manifest["status"]
    for name in ("config.yaml", "metrics.parquet", "report.json", "figures",
                 "manifest.json", "summary.json", "summary.parquet"):
        assert (run_dir / name).exists(), f"缺少产物 {name}"


@pytest.mark.slow
def test_the_override_batch_is_recorded_and_excluded_from_the_release_gate(tmp_path):
    """显式 0.05 诊断 run：如实记录、可聚合，但**不**参与生产 release acceptance。"""
    result = run_batch(tmp_path, "m54i_override", "--corrector-time-limit", "0.05")
    run_dir = tmp_path / "m54i_override"
    # 先确认探针**确实**跑完并落了产物；否则后续 json 解析失败会掩盖真实原因。
    assert (run_dir / "report.json").exists(), (
        f"探针未产出 report.json（returncode={result.returncode}）\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))

    assert report["effective_corrector_time_limit_s"] == pytest.approx(0.05)
    assert report["corrector_time_limit_source"] == "explicit_override"
    assert report["release_gate"]["passed"] is False, "override 不得通过生产门禁"
    assert report["release_gate"]["observations"] == [], (
        f"override 观测混进了生产默认证据：{report['release_gate']['observations']}"
    )
    assert report["release_gate"]["override_observations"], "override 观测必须被如实登记"
    assert result.returncode != 0, (
        f"override run 不得退出 0\nstdout={result.stdout}\nstderr={result.stderr}"
    )

    aggregated = probe.aggregate_release_batches([run_dir])
    assert aggregated["passed"] is False
    assert aggregated["production_default_only"] is False

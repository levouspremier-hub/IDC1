"""M5.4i 发布验收：跨批聚合的**fail-closed** 判据。

首轮审核判定不通过：聚合器只要求 `batches` 非空，于是
「单批次」「同一 run 传三次」「三种负载混合」都能冒充三个独立同负载批次；
且聚合器读 `manifest` 却不校验它，`summary.parquet` 缺失时反而被当成零次 time-limit。

本文件固定返修后的判据：任何字段缺失、文件缺失、样本不足、批间不一致或解析异常
一律 **fail closed**，并给出机器可读的失败原因（具体批次 + 原因码）。
"""

import json
import pathlib
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from scripts import probe_corrector_repro as probe

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PRODUCTION_DEFAULT = 0.25
MIN_BATCHES = 3

REQUIRED_ARTIFACTS = (
    "config.yaml", "metrics.parquet", "report.json", "figures",
    "manifest.json", "summary.json", "summary.parquet",
)


def _write_batch(
    root: pathlib.Path,
    run_id: str,
    *,
    digests=("a", "a", "a", "a", "a", "a"),
    steps: int = 8,
    passed: bool = True,
    evaluated: bool = True,
    all_qualifying: bool = True,
    source: str = "production_default",
    budget: float = PRODUCTION_DEFAULT,
    manifest_status: str | None = None,
    revision: str | None = "rev-0001",
    load: dict | None = None,
    machine: dict | None = None,
    statuses=(0, 0, 0),
    with_parquet: bool = True,
    parquet_has_status: bool = True,
    provenance_in_manifest: bool = True,
    omit: tuple[str, ...] = (),
) -> pathlib.Path:
    """构造一个**完整**的标准 batch run（默认全部判据满足）。"""
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    load = load if load is not None else {
        "mode": "none", "concurrency": 0, "program": None, "load_injected": False,
    }
    machine = machine if machine is not None else {
        "platform": "test", "machine": "arm64", "python": "3.12", "cpu_count": 10,
    }
    observations = [] if not evaluated else [{
        "source": "modes.on",
        "digests": list(digests),
        "distinct": len(set(digests)),
        "processes": len(digests),
        "steps": steps,
        "effective_corrector_time_limit_s": budget,
        "corrector_time_limit_source": source,
    }]
    report = {
        "production_corrector_time_limit_s": PRODUCTION_DEFAULT,
        "effective_corrector_time_limit_s": budget,
        "corrector_time_limit_source": source,
        "load": load,
        "machine": machine,
        "release_gate": {
            "evaluated": evaluated,
            "all_qualifying": all_qualifying,
            "passed": passed,
            "observations": observations,
        },
    }
    manifest: dict = {
        "run_id": run_id,
        "revision": revision,
        "command": "python scripts/probe_corrector_repro.py",
        "status": manifest_status or ("success" if passed else "failed"),
    }
    if provenance_in_manifest:
        manifest |= {
            "production_corrector_time_limit_s": PRODUCTION_DEFAULT,
            "effective_corrector_time_limit_s": budget,
            "corrector_time_limit_source": source,
        }
    config = {
        "probe": "m54d_corrector_reproducibility",
        "production_corrector_time_limit_s": PRODUCTION_DEFAULT,
        "effective_corrector_time_limit_s": budget,
        "corrector_time_limit_source": source,
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    pd.DataFrame([{"step": i, "status": 0} for i in range(3)]).to_parquet(
        run_dir / "metrics.parquet"
    )
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    if with_parquet:
        columns = {"status": list(statuses)} if parquet_has_status else {"step": [0, 1]}
        pd.DataFrame(columns).to_parquet(run_dir / "summary.parquet")
    for name in omit:
        target = run_dir / name
        target.rmdir() if target.is_dir() else target.unlink()
    return run_dir


def _failures(aggregated: dict) -> set[str]:
    return {f["reason"] for f in aggregated["failures"]}


# --- 1. 批次数量与独立性 -----------------------------------------------------

def test_one_batch_is_not_cross_batch_evidence(tmp_path):
    aggregated = probe.aggregate_release_batches([_write_batch(tmp_path, "b0")])
    assert aggregated["passed"] is False
    assert "too_few_batches" in _failures(aggregated)


def test_two_batches_are_not_enough(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "b0"), _write_batch(tmp_path, "b1"),
    ])
    assert aggregated["passed"] is False
    assert "too_few_batches" in _failures(aggregated)


def test_the_same_path_three_times_is_not_three_independent_batches(tmp_path):
    same = _write_batch(tmp_path, "b0")
    aggregated = probe.aggregate_release_batches([same, same, same])
    assert aggregated["passed"] is False, "同一个 run 重复三次不得冒充三个独立批次"
    assert "duplicate_input_path" in _failures(aggregated)


def test_same_path_via_different_spellings_is_still_a_duplicate(tmp_path):
    same = _write_batch(tmp_path, "b0")
    spelled = pathlib.Path(str(same) + "/.")
    aggregated = probe.aggregate_release_batches([same, spelled, same])
    assert aggregated["passed"] is False
    assert "duplicate_input_path" in _failures(aggregated)


def test_duplicate_run_ids_across_different_dirs_fail(tmp_path):
    a = _write_batch(tmp_path / "x", "same_id")
    b = _write_batch(tmp_path / "y", "same_id")
    c = _write_batch(tmp_path / "z", "other_id")
    aggregated = probe.aggregate_release_batches([a, b, c])
    assert aggregated["passed"] is False
    assert "duplicate_run_id" in _failures(aggregated)


def test_mixed_loads_fail(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "nl", load={"mode": "none", "concurrency": 0,
                                           "program": None, "load_injected": False}),
        _write_batch(tmp_path, "h4", load={"mode": "hogs", "concurrency": 4,
                                           "program": "prog", "load_injected": True}),
        _write_batch(tmp_path, "h8", load={"mode": "hogs", "concurrency": 8,
                                           "program": "prog", "load_injected": True}),
    ])
    assert aggregated["passed"] is False, "三种负载混合不得通过"
    assert "inconsistent_load_signature" in _failures(aggregated)


def test_different_concurrency_under_the_same_mode_fails(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a", load={"mode": "hogs", "concurrency": 4,
                                          "program": "p", "load_injected": True}),
        _write_batch(tmp_path, "b", load={"mode": "hogs", "concurrency": 8,
                                          "program": "p", "load_injected": True}),
        _write_batch(tmp_path, "c", load={"mode": "hogs", "concurrency": 4,
                                          "program": "p", "load_injected": True}),
    ])
    assert aggregated["passed"] is False
    assert "inconsistent_load_signature" in _failures(aggregated)


def test_inconsistent_machine_signature_fails(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a"),
        _write_batch(tmp_path, "b", machine={"platform": "other", "machine": "x86",
                                             "python": "3.12", "cpu_count": 10}),
        _write_batch(tmp_path, "c"),
    ])
    assert aggregated["passed"] is False, "批间机器不一致不得声称为同一个本机候选"
    assert "inconsistent_machine_signature" in _failures(aggregated)


# --- 2. manifest 校验 --------------------------------------------------------

def test_a_failed_batch_manifest_fails_the_aggregate(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a"), _write_batch(tmp_path, "b"),
        _write_batch(tmp_path, "c", manifest_status="failed", passed=False),
    ])
    assert aggregated["passed"] is False
    assert "manifest_status_not_success" in _failures(aggregated)


def test_missing_revision_fails(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a"), _write_batch(tmp_path, "b"),
        _write_batch(tmp_path, "c", revision=None),
    ])
    assert aggregated["passed"] is False
    assert "manifest_revision_missing" in _failures(aggregated)


def test_inconsistent_revisions_fail(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a", revision="rev-A"),
        _write_batch(tmp_path, "b", revision="rev-B"),
        _write_batch(tmp_path, "c", revision="rev-A"),
    ])
    assert aggregated["passed"] is False
    assert "inconsistent_revision" in _failures(aggregated)


def test_missing_run_id_in_manifest_fails(tmp_path):
    run_dir = _write_batch(tmp_path, "a")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("run_id")
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    aggregated = probe.aggregate_release_batches([
        run_dir, _write_batch(tmp_path, "b"), _write_batch(tmp_path, "c"),
    ])
    assert aggregated["passed"] is False
    assert "manifest_run_id_missing" in _failures(aggregated)


# --- 3. 门禁与样本量 ---------------------------------------------------------

@pytest.mark.parametrize(
    "overrides,expected_reason",
    [
        ({"evaluated": False}, "release_gate_not_evaluated"),
        ({"all_qualifying": False}, "release_gate_not_all_qualifying"),
        ({"passed": False}, "release_gate_not_passed"),
    ],
)
def test_any_non_passing_release_gate_fails(tmp_path, overrides, expected_reason):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a"), _write_batch(tmp_path, "b"),
        _write_batch(tmp_path, "c", manifest_status="success", **overrides),
    ])
    assert aggregated["passed"] is False
    assert expected_reason in _failures(aggregated)


def test_underpowered_observations_fail(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a"), _write_batch(tmp_path, "b"),
        _write_batch(tmp_path, "c", digests=("a", "a")),  # 只有 2 个进程
    ])
    assert aggregated["passed"] is False
    assert "observation_underpowered" in _failures(aggregated)


def test_short_rollouts_fail(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a"), _write_batch(tmp_path, "b"),
        _write_batch(tmp_path, "c", steps=4),
    ])
    assert aggregated["passed"] is False
    assert "observation_underpowered" in _failures(aggregated)


def test_digest_count_must_match_processes(tmp_path):
    """`processes` 与实际 digest 个数不符 -> 证据不可信。"""
    run_dir = _write_batch(tmp_path, "a")
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    report["release_gate"]["observations"][0]["processes"] = 6
    report["release_gate"]["observations"][0]["digests"] = ["a", "a", "a"]
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    aggregated = probe.aggregate_release_batches([
        run_dir, _write_batch(tmp_path, "b"), _write_batch(tmp_path, "c"),
    ])
    assert aggregated["passed"] is False
    assert "digest_count_mismatch" in _failures(aggregated)


def test_empty_digests_fail(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a"), _write_batch(tmp_path, "b"),
        _write_batch(tmp_path, "c", digests=()),
    ])
    assert aggregated["passed"] is False
    assert "digests_empty" in _failures(aggregated)


def test_override_observations_cannot_be_release_evidence(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a"), _write_batch(tmp_path, "b"),
        _write_batch(tmp_path, "c", source="explicit_override"),
    ])
    assert aggregated["passed"] is False
    assert "override_observation_in_release_evidence" in _failures(aggregated)


def test_a_non_production_budget_fails(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a"), _write_batch(tmp_path, "b"),
        _write_batch(tmp_path, "c", budget=0.05),
    ])
    assert aggregated["passed"] is False
    assert "budget_not_production_default" in _failures(aggregated)


# --- 4. summary.parquet 与输入产物完整性（fail closed） ----------------------

def test_missing_summary_parquet_fails_closed(tmp_path):
    """缺 summary.parquet **不得**被当成「零次 time-limit」。"""
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a"), _write_batch(tmp_path, "b"),
        _write_batch(tmp_path, "c", with_parquet=False),
    ])
    assert aggregated["passed"] is False
    assert "missing_or_unreadable_artifact" in _failures(aggregated)


def test_summary_parquet_without_status_column_fails_closed(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a"), _write_batch(tmp_path, "b"),
        _write_batch(tmp_path, "c", parquet_has_status=False),
    ])
    assert aggregated["passed"] is False
    assert "summary_parquet_missing_status_column" in _failures(aggregated)


def test_any_time_limit_failure_fails_closed(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a"), _write_batch(tmp_path, "b"),
        _write_batch(tmp_path, "c", statuses=(1, 0, 0)),
    ])
    assert aggregated["passed"] is False
    assert "time_limit_failure" in _failures(aggregated)


@pytest.mark.parametrize("missing", REQUIRED_ARTIFACTS)
def test_missing_standard_artifact_fails(tmp_path, missing):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a"), _write_batch(tmp_path, "b"),
        _write_batch(tmp_path, "c", omit=(missing,)),
    ])
    assert aggregated["passed"] is False
    assert "missing_or_unreadable_artifact" in _failures(aggregated)


def test_unreadable_report_fails_closed(tmp_path):
    run_dir = _write_batch(tmp_path, "a")
    (run_dir / "report.json").write_text("{ not json", encoding="utf-8")
    aggregated = probe.aggregate_release_batches([
        run_dir, _write_batch(tmp_path, "b"), _write_batch(tmp_path, "c"),
    ])
    assert aggregated["passed"] is False
    assert "missing_or_unreadable_artifact" in _failures(aggregated)


def test_missing_provenance_in_manifest_fails(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a"), _write_batch(tmp_path, "b"),
        _write_batch(tmp_path, "c", provenance_in_manifest=False),
    ])
    assert aggregated["passed"] is False
    assert "manifest_provenance_missing" in _failures(aggregated)


def test_empty_input_does_not_pass():
    aggregated = probe.aggregate_release_batches([])
    assert aggregated["passed"] is False
    assert "too_few_batches" in _failures(aggregated)


# --- 5. 正例：三个独立同负载批次通过 ----------------------------------------

def test_three_independent_same_load_batches_pass(tmp_path):
    batches = [_write_batch(tmp_path, f"b{i}") for i in range(MIN_BATCHES)]
    aggregated = probe.aggregate_release_batches(batches)
    assert aggregated["passed"] is True, aggregated["failures"]
    assert aggregated["failures"] == []
    assert aggregated["batches"] == MIN_BATCHES
    assert aggregated["aggregate_distinct"] == 1
    assert aggregated["every_batch_distinct_one"] is True
    assert aggregated["time_limit_failures"] == 0


def test_intra_batch_fork_fails(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a"), _write_batch(tmp_path, "b"),
        _write_batch(tmp_path, "c", digests=("a", "a", "Z")),
    ])
    assert aggregated["passed"] is False
    assert "batch_distinct_not_one" in _failures(aggregated)


def test_failures_name_the_offending_batch(tmp_path):
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "good_a"), _write_batch(tmp_path, "good_b"),
        _write_batch(tmp_path, "bad_c", manifest_status="failed", passed=False),
    ])
    offenders = {f.get("batch") for f in aggregated["failures"]}
    assert "bad_c" in offenders, aggregated["failures"]


# --- 6. 真实批次与标准聚合 run（slow） ---------------------------------------

def run_batch(base_dir: pathlib.Path, run_id: str, *extra: str,
              timeout: int = 3600) -> subprocess.CompletedProcess:
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
    assert report["release_gate"]["observations"]
    assert all(
        o["corrector_time_limit_source"] == "production_default"
        for o in report["release_gate"]["observations"]
    )
    assert (result.returncode == 0) is report["release_gate"]["passed"]
    assert manifest["status"] == ("success" if report["release_gate"]["passed"] else "failed")
    assert summary["status"] == manifest["status"]
    for name in REQUIRED_ARTIFACTS:
        assert (run_dir / name).exists(), f"缺少产物 {name}"


@pytest.mark.slow
def test_a_real_batch_writes_provenance_into_config_report_and_manifest(tmp_path):
    run_batch(tmp_path, "m54i_prov")
    run_dir = tmp_path / "m54i_prov"
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    for obj in (config, report, manifest):
        assert obj["production_corrector_time_limit_s"] == pytest.approx(PRODUCTION_DEFAULT)
        assert obj["effective_corrector_time_limit_s"] == pytest.approx(PRODUCTION_DEFAULT)
        assert obj["corrector_time_limit_source"] == "production_default", obj


@pytest.mark.slow
def test_the_aggregate_command_writes_a_standard_run(tmp_path):
    for i in range(MIN_BATCHES):
        run_batch(tmp_path, f"m54i_b{i}")
    out = subprocess.run(
        [sys.executable, "scripts/probe_corrector_repro.py", "--aggregate-runs",
         *[str(tmp_path / f"m54i_b{i}") for i in range(MIN_BATCHES)],
         "--base-dir", str(tmp_path), "--run-id", "m54i_agg"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800,
    )
    agg_dir = tmp_path / "m54i_agg"
    for name in ("config.yaml", "metrics.parquet", "report.json", "figures", "manifest.json"):
        assert (agg_dir / name).exists(), f"聚合 run 缺少标准产物 {name}"

    manifest = json.loads((agg_dir / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((agg_dir / "report.json").read_text(encoding="utf-8"))
    assert manifest["corrector_time_limit_source"] == "production_default"
    assert manifest["status"] == ("success" if report["passed"] else "failed")
    assert (out.returncode == 0) is report["passed"]
    assert report["passed"] is True, report["failures"]


@pytest.mark.slow
def test_the_aggregate_command_refuses_to_overwrite_a_successful_aggregate(tmp_path):
    for i in range(MIN_BATCHES):
        run_batch(tmp_path, f"c{i}")
    args = [sys.executable, "scripts/probe_corrector_repro.py", "--aggregate-runs",
            *[str(tmp_path / f"c{i}") for i in range(MIN_BATCHES)],
            "--base-dir", str(tmp_path), "--run-id", "agg_dup"]
    first = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800)
    assert first.returncode == 0, first.stderr
    before = (tmp_path / "agg_dup" / "manifest.json").read_bytes()

    second = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800)
    assert second.returncode != 0, "不得覆盖既有成功聚合 run"
    assert (tmp_path / "agg_dup" / "manifest.json").read_bytes() == before


@pytest.mark.slow
def test_the_override_batch_is_recorded_and_excluded_from_the_release_gate(tmp_path):
    result = run_batch(tmp_path, "m54i_override", "--corrector-time-limit", "0.05")
    run_dir = tmp_path / "m54i_override"
    assert (run_dir / "report.json").exists(), (
        f"探针未产出 report.json（returncode={result.returncode}）\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))

    assert report["effective_corrector_time_limit_s"] == pytest.approx(0.05)
    assert report["corrector_time_limit_source"] == "explicit_override"
    assert report["release_gate"]["passed"] is False
    assert report["release_gate"]["observations"] == []
    assert report["release_gate"]["override_observations"]
    assert result.returncode != 0

    aggregated = probe.aggregate_release_batches([run_dir])
    assert aggregated["passed"] is False


# --- 7. 第二次审核返修：三方恒等、生产默认判据、类型 fail closed ------------

def _mutate(path: pathlib.Path, name: str, mutate) -> pathlib.Path:
    """按文件类型读写并改写一个批次产物。"""
    target = path / name
    if name.endswith(".json"):
        obj = json.loads(target.read_text(encoding="utf-8"))
        obj = mutate(obj)
        target.write_text(json.dumps(obj), encoding="utf-8")
    else:
        obj = yaml.safe_load(target.read_text(encoding="utf-8"))
        obj = mutate(obj)
        target.write_text(yaml.safe_dump(obj), encoding="utf-8")
    return path


def _set_provenance(path: pathlib.Path, name: str, **fields) -> pathlib.Path:
    return _mutate(path, name, lambda obj: (obj.update(fields), obj)[1])


@pytest.mark.parametrize("filename", ("config.yaml", "report.json", "manifest.json"))
@pytest.mark.parametrize("field", (
    "production_corrector_time_limit_s",
    "effective_corrector_time_limit_s",
    "corrector_time_limit_source",
))
def test_any_provenance_mismatch_in_any_file_fails(tmp_path, filename, field):
    """config/report/manifest 任意一处、任意字段不一致 -> 聚合必须失败。"""
    a, b, c = (_write_batch(tmp_path, n) for n in ("a", "b", "c"))
    tampered = {"effective_corrector_time_limit_s": 99.0}.get(field, "tampered")
    _set_provenance(a, filename, **{field: tampered})
    aggregated = probe.aggregate_release_batches([a, b, c])
    assert aggregated["passed"] is False, f"{filename}:{field} 不一致必须失败"
    assert "provenance_inconsistent" in _failures(aggregated)


def test_top_level_production_budget_must_be_the_default(tmp_path):
    a, b, c = (_write_batch(tmp_path, n) for n in ("a", "b", "c"))
    for name in ("config.yaml", "report.json", "manifest.json"):
        _set_provenance(a, name, production_corrector_time_limit_s=0.05)
    aggregated = probe.aggregate_release_batches([a, b, c])
    assert aggregated["passed"] is False
    assert "production_default_not_satisfied" in _failures(aggregated)


def test_top_level_effective_budget_must_be_the_default(tmp_path):
    a, b, c = (_write_batch(tmp_path, n) for n in ("a", "b", "c"))
    for name in ("config.yaml", "report.json", "manifest.json"):
        _set_provenance(a, name, effective_corrector_time_limit_s=0.05)
    aggregated = probe.aggregate_release_batches([a, b, c])
    assert aggregated["passed"] is False
    assert "production_default_not_satisfied" in _failures(aggregated)


def test_top_level_source_must_be_production_default(tmp_path):
    """即使观测伪装成 production_default，顶层 source=explicit_override 仍必须失败。"""
    a, b, c = (_write_batch(tmp_path, n) for n in ("a", "b", "c"))
    for name in ("config.yaml", "report.json", "manifest.json"):
        _set_provenance(a, name, corrector_time_limit_source="explicit_override")
    aggregated = probe.aggregate_release_batches([a, b, c])
    assert aggregated["passed"] is False
    assert aggregated["production_default_only"] is False
    assert "production_default_not_satisfied" in _failures(aggregated)


def test_production_default_only_false_alone_blocks_passing(tmp_path):
    """`production_default_only=false` 必须**直接**导致 passed=false 并给出 reason。"""
    aggregated = probe.aggregate_release_batches([
        _write_batch(tmp_path, "a"), _write_batch(tmp_path, "b"),
        _write_batch(tmp_path, "c", source="explicit_override", passed=False,
                     manifest_status="failed"),
    ])
    assert aggregated["passed"] is False
    assert aggregated["production_default_only"] is False
    assert "production_default_not_satisfied" in _failures(aggregated)


def test_all_batches_missing_load_fails(tmp_path):
    """三批**全部**缺 load 时，空签名互相相等也不得通过。"""
    batches = [_write_batch(tmp_path, n) for n in ("a", "b", "c")]
    for d in batches:
        _mutate(d, "report.json", lambda obj: (obj.pop("load", None), obj)[1])
    aggregated = probe.aggregate_release_batches(batches)
    assert aggregated["passed"] is False
    assert "load_signature_invalid" in _failures(aggregated)


def test_all_batches_missing_machine_fails(tmp_path):
    batches = [_write_batch(tmp_path, n) for n in ("a", "b", "c")]
    for d in batches:
        _mutate(d, "report.json", lambda obj: (obj.pop("machine", None), obj)[1])
    aggregated = probe.aggregate_release_batches(batches)
    assert aggregated["passed"] is False
    assert "machine_signature_invalid" in _failures(aggregated)


@pytest.mark.parametrize("key", ("mode", "concurrency", "program", "load_injected"))
def test_incomplete_load_keys_fail(tmp_path, key):
    batches = [_write_batch(tmp_path, n) for n in ("a", "b", "c")]
    for d in batches:
        _mutate(d, "report.json", lambda obj, k=key: (
            obj.__setitem__("load", {kk: vv for kk, vv in obj["load"].items() if kk != k}), obj
        )[1])
    aggregated = probe.aggregate_release_batches(batches)
    assert aggregated["passed"] is False
    assert "load_signature_invalid" in _failures(aggregated)


@pytest.mark.parametrize("key", ("platform", "machine", "python", "cpu_count"))
def test_incomplete_machine_keys_fail(tmp_path, key):
    batches = [_write_batch(tmp_path, n) for n in ("a", "b", "c")]
    for d in batches:
        _mutate(d, "report.json", lambda obj, k=key: (
            obj.__setitem__("machine",
                            {kk: vv for kk, vv in obj["machine"].items() if kk != k}), obj
        )[1])
    aggregated = probe.aggregate_release_batches(batches)
    assert aggregated["passed"] is False
    assert "machine_signature_invalid" in _failures(aggregated)


@pytest.mark.parametrize("filename", ("report.json", "manifest.json"))
def test_non_dict_top_level_json_fails_closed_without_raising(tmp_path, filename):
    """合法 JSON 但顶层不是 dict -> passed=false，**不得**抛异常。"""
    batches = [_write_batch(tmp_path, n) for n in ("a", "b", "c")]
    (batches[0] / filename).write_text("[]", encoding="utf-8")
    aggregated = probe.aggregate_release_batches(batches)  # 不得抛
    assert aggregated["passed"] is False
    assert "malformed_batch_artifact" in _failures(aggregated)


def test_non_dict_release_gate_fails_closed(tmp_path):
    batches = [_write_batch(tmp_path, n) for n in ("a", "b", "c")]
    _mutate(batches[0], "report.json", lambda obj: (
        obj.__setitem__("release_gate", "nope"), obj)[1])
    aggregated = probe.aggregate_release_batches(batches)
    assert aggregated["passed"] is False
    assert "malformed_batch_artifact" in _failures(aggregated)


def test_non_list_observations_fails_closed(tmp_path):
    batches = [_write_batch(tmp_path, n) for n in ("a", "b", "c")]
    _mutate(batches[0], "report.json", lambda obj: (
        obj["release_gate"].__setitem__("observations", "nope"), obj)[1])
    aggregated = probe.aggregate_release_batches(batches)
    assert aggregated["passed"] is False
    assert "malformed_batch_artifact" in _failures(aggregated)


def test_non_dict_observation_fails_closed(tmp_path):
    batches = [_write_batch(tmp_path, n) for n in ("a", "b", "c")]
    _mutate(batches[0], "report.json", lambda obj: (
        obj["release_gate"].__setitem__("observations", ["not-a-dict"]), obj)[1])
    aggregated = probe.aggregate_release_batches(batches)
    assert aggregated["passed"] is False
    assert "malformed_batch_artifact" in _failures(aggregated)


@pytest.mark.parametrize("bad", ("abc", [], {}))
def test_unconvertible_processes_fails_closed(tmp_path, bad):
    """`processes` 不可转换为整数 -> fail closed，**不得**抛 ValueError/TypeError。"""
    batches = [_write_batch(tmp_path, n) for n in ("a", "b", "c")]
    _mutate(batches[0], "report.json", lambda obj: (
        obj["release_gate"]["observations"][0].__setitem__("processes", bad), obj)[1])
    aggregated = probe.aggregate_release_batches(batches)
    assert aggregated["passed"] is False
    assert "malformed_batch_artifact" in _failures(aggregated)


@pytest.mark.parametrize("bad", ("abc", {}, []))
def test_unconvertible_steps_fails_closed(tmp_path, bad):
    batches = [_write_batch(tmp_path, n) for n in ("a", "b", "c")]
    _mutate(batches[0], "report.json", lambda obj: (
        obj["release_gate"]["observations"][0].__setitem__("steps", bad), obj)[1])
    aggregated = probe.aggregate_release_batches(batches)
    assert aggregated["passed"] is False
    assert "malformed_batch_artifact" in _failures(aggregated)


@pytest.mark.parametrize("bad", ("abc", [], {}))
def test_unconvertible_budget_fails_closed(tmp_path, bad):
    batches = [_write_batch(tmp_path, n) for n in ("a", "b", "c")]
    _mutate(batches[0], "report.json", lambda obj: (
        obj["release_gate"]["observations"][0].__setitem__(
            "effective_corrector_time_limit_s", bad), obj)[1])
    aggregated = probe.aggregate_release_batches(batches)
    assert aggregated["passed"] is False
    assert "malformed_batch_artifact" in _failures(aggregated)


@pytest.mark.parametrize("bad", ("abc", None, [], float("nan")))
def test_illegal_summary_parquet_status_fails_closed(tmp_path, bad):
    """`summary.parquet` 的 status 含非法值/NaN -> fail closed，不得抛 ValueError。"""
    batches = [_write_batch(tmp_path, n) for n in ("a", "b", "c")]
    pd.DataFrame({"status": [bad, 0.0]}).to_parquet(batches[0] / "summary.parquet")
    aggregated = probe.aggregate_release_batches(batches)
    assert aggregated["passed"] is False
    assert "summary_parquet_malformed_status" in _failures(aggregated)


@pytest.mark.parametrize("field", ("processes", "steps", "effective_corrector_time_limit_s"))
def test_missing_numeric_field_fails_closed(tmp_path, field):
    """字段缺失（None）也必须 fail closed，且**不得**抛异常。"""
    batches = [_write_batch(tmp_path, n) for n in ("a", "b", "c")]
    _mutate(batches[0], "report.json", lambda obj, f=field: (
        obj["release_gate"]["observations"][0].__setitem__(f, None), obj)[1])
    aggregated = probe.aggregate_release_batches(batches)
    assert aggregated["passed"] is False
    assert aggregated["failures"], "缺值必须产生失败原因"


def test_every_failure_carries_batch_reason_and_detail(tmp_path):
    batches = [_write_batch(tmp_path, n) for n in ("a", "b", "c")]
    _mutate(batches[0], "report.json", lambda obj: (obj.pop("load", None), obj)[1])
    aggregated = probe.aggregate_release_batches(batches)
    assert aggregated["failures"], "必须有失败原因"
    for failure in aggregated["failures"]:
        assert "batch" in failure and "reason" in failure and "detail" in failure, failure
        assert isinstance(failure["reason"], str) and failure["reason"]
        assert isinstance(failure["detail"], str), failure
    assert any(f["batch"] == "a" for f in aggregated["failures"])


def test_complete_independent_batches_still_pass(tmp_path):
    """正例不得被返修误伤。"""
    batches = [_write_batch(tmp_path, n) for n in ("a", "b", "c")]
    aggregated = probe.aggregate_release_batches(batches)
    assert aggregated["passed"] is True, aggregated["failures"]
    assert aggregated["failures"] == []
    assert aggregated["production_default_only"] is True


def test_batches_are_read_once_per_input(tmp_path, monkeypatch):
    """每个输入只应读取一次 report.json（避免二次读取产生不一致）。"""
    batches = [_write_batch(tmp_path, n) for n in ("a", "b", "c")]
    reads: list[str] = []
    original = pathlib.Path.read_text

    def counting_read_text(self, *args, **kwargs):
        if self.name == "report.json":
            reads.append(str(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", counting_read_text)
    probe.aggregate_release_batches(batches)
    monkeypatch.undo()
    assert len(reads) == len(set(reads)) == 3, f"report.json 被重复读取：{reads}"

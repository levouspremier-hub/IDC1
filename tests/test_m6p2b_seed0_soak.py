"""Soak orchestration invariants; no four-hour workload in unit tests."""
import pytest

from scripts import m6p2b_seed0_soak as soak


def test_fixed_protocol_and_monotonic_boundary():
    assert soak.DURATION_S == 14400
    assert len(soak.case_grid()) == 24
    assert soak.keep_running(100, 14499)
    assert not soak.keep_running(100, 14500)


def test_host_power_and_disk_fail_closed():
    soak.check_host({"ac": True, "free_bytes": 6 * 2**30, "errors": []})
    for row in [
        {"ac": False, "free_bytes": 6 * 2**30, "errors": []},
        {"ac": True, "free_bytes": 4 * 2**30, "errors": []},
        {"ac": True, "free_bytes": 6 * 2**30, "errors": ["timeout"]},
    ]:
        with pytest.raises(RuntimeError):
            soak.check_host(row)


def test_incomplete_or_bad_episode_cannot_pass():
    row = {"acceptance": {"completion": True}, "parameter_updates": 0}
    assert soak.episode_ok(row)
    assert not soak.episode_ok({**row, "parameter_updates": 1})
    assert not soak.episode_ok({**row, "acceptance": {"completion": False}})
    assert not soak.episode_ok({**row, "acceptance": {}})


def test_failure_preflight_writes_standard_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(soak, "ROOT", tmp_path)
    monkeypatch.setattr(soak, "sha", lambda p: "mock-hash")
    monkeypatch.setattr(soak, "verify_release", lambda: (_ for _ in ()).throw(ValueError("binding")))
    with pytest.raises(ValueError, match="binding"):
        soak.run("failed-preflight")
    folder = tmp_path / "runs/failed-preflight"
    import json
    assert json.loads((folder / "manifest.json").read_text())["status"] == "failed"
    for name in ("config.yaml", "metrics.parquet", "report.json", "figures", "artifact_verification.json"):
        assert (folder / name).exists()
    with pytest.raises(FileExistsError):
        soak.run("failed-preflight")

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
    def reject():
        raise ValueError("binding")

    monkeypatch.setattr(soak, "verify_release", reject)
    with pytest.raises(ValueError, match="binding"):
        soak.run("failed-preflight")
    folder = tmp_path / "runs/failed-preflight"
    import json
    assert json.loads((folder / "manifest.json").read_text())["status"] == "failed"
    for name in ("config.yaml", "metrics.parquet", "report.json", "figures",
                 "artifact_verification.json"):
        assert (folder / name).exists()
    with pytest.raises(FileExistsError):
        soak.run("failed-preflight")


def test_orchestrator_retains_policy_and_runs_both_controls(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    tick = [0]
    policy = SimpleNamespace(eval=lambda: None)
    policy_ids = []
    monkeypatch.setattr(soak, "ROOT", tmp_path)
    monkeypatch.setattr(soak, "DURATION_S", 2)
    monkeypatch.setattr(soak, "time", SimpleNamespace(monotonic=lambda: tick[0]))
    monkeypatch.setattr(soak, "sha", lambda p: soak.CHECKPOINT_SHA)
    monkeypatch.setattr(soak, "verify_release", lambda: None)
    monkeypatch.setattr(soak, "semantics_binding", lambda: {"bound": "unchanged"})
    monkeypatch.setattr(soak, "parameter_hash", lambda p: "frozen")
    monkeypatch.setattr(soak, "load_policy", lambda p: policy)
    monkeypatch.setattr(soak, "export_policy", lambda src, dst: dst.write_text("mock"))
    monkeypatch.setattr(soak, "training_source_ledger", lambda rows: {})
    monkeypatch.setattr(soak.HostMonitor, "sample", lambda self: None)

    def row(origin, mode):
        return {"origin": origin, "mode": mode, "steps": 48, "parameter_updates": 0,
                "acceptance": {"completion": True}, "injection_provenance": {}}

    def child(cmd, **kwargs):
        output = Path(cmd[cmd.index("--worker") + 1])
        output.mkdir(parents=True)
        data = row(int(cmd[cmd.index("--origin") + 1]), cmd[cmd.index("--mode") + 1])
        (output / "episode.json").write_text(json.dumps(data))

    def shared(folder, origin, mode, path, *, policy):
        policy_ids.append(id(policy))
        tick[0] += 1
        return row(origin, mode)

    monkeypatch.setattr(soak.subprocess, "run", child)
    monkeypatch.setattr(soak, "episode", shared)
    soak.run("mock-soak")
    result = json.loads((tmp_path / "runs/mock-soak/report.json").read_text())
    assert result["shared_elapsed_s"] == 2
    assert policy_ids == [id(policy), id(policy)]
    assert len(result["episodes"]) == 50
    assert [r["process_arm"] for r in result["episodes"]].count("fresh_before") == 24
    assert [r["process_arm"] for r in result["episodes"]].count("fresh_after") == 24

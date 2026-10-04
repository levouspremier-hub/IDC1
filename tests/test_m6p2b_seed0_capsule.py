"""Contract-only capsule authenticity and scope; no checkpoint policy load."""
import pytest

from scripts.m6p2b_seed0_capsule import assert_digest


def test_capsule_rejects_changed_input(tmp_path):
    from hashlib import sha256

    path = tmp_path / "capture.json"
    path.write_text("original")
    digest = sha256(path.read_bytes()).hexdigest()
    assert_digest(path, digest)
    path.write_text("changed")
    with pytest.raises(ValueError, match="capsule"):
        assert_digest(path, digest)


def test_failed_capsule_propagates_nonzero_after_preserving_artifacts(tmp_path, monkeypatch):
    import contextlib
    import json
    from types import SimpleNamespace

    from scripts import m6p2b_seed0_capsule as capsule

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(capsule, "ROOT", tmp_path)
    release = tmp_path / "release.json"
    release.write_text("{}")
    (tmp_path / "uv.lock").write_text("frozen")
    monkeypatch.setattr(capsule, "RELEASE", release)
    monkeypatch.setattr(capsule, "authenticate", lambda: (None, None, [0.0] * 21))
    monkeypatch.setattr(capsule, "Observer", lambda _: SimpleNamespace(
        installed=contextlib.nullcontext, current={}))
    monkeypatch.setattr(capsule, "correct", lambda *a, **kw: SimpleNamespace(
        failure="timeout", exec_compute_actions=[0.0] * 20, exec_storage_action=0.0))
    assert capsule.run("failed_probe") == 1
    folder = tmp_path / "runs" / "failed_probe"
    assert json.loads((folder / "report.json").read_text())["failed_solves"] == 128
    assert json.loads((folder / "manifest.json").read_text())["status"] == "failed"
    assert (folder / "artifact_verification.json").is_file()

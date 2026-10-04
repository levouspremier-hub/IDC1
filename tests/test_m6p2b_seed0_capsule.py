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

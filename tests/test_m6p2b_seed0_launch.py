"""Single-seed authorization cannot impersonate the three-seed gate."""

import pytest

from safe_rl_v2 import inventory_train as train


@pytest.mark.parametrize("seed", [1, 2])
def test_seed0_authorization_rejects_other_seeds(seed):
    with pytest.raises(ValueError, match="seed 0"):
        train.require_seed0_launch_gate({}, seed=seed)


def test_seed0_requires_a_verified_release(monkeypatch):
    from scenario import inventory_release as release

    def reject():
        raise ValueError("release rejected")

    monkeypatch.setattr(release, "verify_release", reject)
    with pytest.raises(ValueError, match="release rejected"):
        train.require_seed0_launch_gate({}, seed=0)


def test_seed0_cli_rejects_short_scope(monkeypatch):
    calls = []
    monkeypatch.setattr(train, "run", lambda args: calls.append(args))
    with pytest.raises(SystemExit):
        train.main(["--short", "--seed0-formal", "--seed", "0", "--run-id", "fixture"])
    assert not calls


def test_seed0_cli_has_explicit_authorization(monkeypatch):
    calls = []
    monkeypatch.setattr(train, "run", lambda args: calls.append(args) or 0)
    assert train.main(["--seed0-formal", "--seed", "0", "--run-id", "fixture"]) == 0
    assert calls[0].seed0_formal is True and calls[0].short is False

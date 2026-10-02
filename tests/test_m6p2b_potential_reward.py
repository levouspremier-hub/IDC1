"""Stability shaping must not change complete discounted policy preferences."""

import copy

import numpy as np
import pytest

from planning.corrector import Correction, FailureClass
from safe_rl.corrector_wrapper import CorrectorWrapper
from tests.test_m6p2b_terminal_inventory import _env

VERSION = "common-sgd-potential-smooth-v1"


def potential(env):
    return (-env.reward_load_smooth_weight * float(np.mean(np.abs(
        env.prev_loads - env.base_load)))
            - env.reward_action_smooth_weight * float(np.mean(np.abs(env.prev_action))))


def projection(*args, **kwargs):
    proposal = args[1]
    return Correction(exec_compute_actions=proposal.compute_actions,
                      exec_storage_action=proposal.storage_action,
                      business_gap=0., failure=FailureClass.NONE, reviewed=True)


@pytest.mark.parametrize("levels", [(1., .4, .8, .5), (.4, .8, .4, 1.)])
def test_complete_discounted_potential_is_constant_and_preserves_execution(monkeypatch, levels):
    monkeypatch.setattr("safe_rl.corrector_wrapper.correct", projection)
    base = _env(horizon=4)
    base.terminal_inventory_reward_version = "common-sgd-degradation-v1"
    revised = copy.deepcopy(base)
    revised.terminal_inventory_reward_version = VERSION
    revised.terminal_inventory_reward_gamma = .99
    original = CorrectorWrapper(base, corrector_time_limit_s=.25)
    candidate = CorrectorWrapper(revised, corrector_time_limit_s=.25)
    initial = potential(revised)
    discounted = 0.
    for k, level in enumerate(levels):
        raw = np.asarray([level] * 20 + [(-1.) ** k * .05], dtype=np.float32)
        previous = potential(revised)
        old_obs, old_reward, old_done, old_trunc, old = original.step(raw)
        obs, reward, done, trunc, info = candidate.step(raw)
        following = 0. if done or trunc else potential(revised)
        shape = .99 * following - previous
        assert info["r_potential_smooth"] == pytest.approx(shape, abs=1e-12)
        assert info["r_load_smooth"] == info["r_action_smooth"] == 0.
        assert reward == pytest.approx(
            old_reward - old["r_load_smooth"] - old["r_action_smooth"] + shape, abs=1e-12)
        assert info["reward_total"] == pytest.approx(reward, abs=1e-12)
        np.testing.assert_array_equal(obs, old_obs)
        np.testing.assert_array_equal(info["raw_action"], raw)
        np.testing.assert_array_equal(info["exec_action"], old["exec_action"])
        assert (done, trunc) == (old_done, old_trunc)
        assert revised.bess_soc == base.bess_soc
        assert revised.normalization_refs() == base.normalization_refs()
        for key in old:
            if key.startswith("r_") and key not in ("r_load_smooth", "r_action_smooth"):
                assert info[key] == old[key]
        discounted += .99 ** k * shape
    assert discounted == pytest.approx(-initial, abs=1e-12)


@pytest.mark.parametrize("gamma", [None, -1., 1.1, float("nan")])
def test_potential_reward_requires_explicit_valid_discount(gamma):
    env = _env(horizon=4)
    env.terminal_inventory_reward_version = VERSION
    if gamma is not None:
        env.terminal_inventory_reward_gamma = gamma
    with pytest.raises(ValueError, match="discount"):
        CorrectorWrapper(env, corrector_time_limit_s=.25)


@pytest.mark.parametrize("field", ["ppo", "shaping"])
def test_frozen_potential_discount_cannot_diverge_from_actual_ppo(tmp_path, monkeypatch, field):
    import json

    from scenario import inventory_release as release

    path = release.ROOT / release.CONFIG_PATH
    if not path.exists():
        pytest.skip("new potential calibration not yet frozen")
    config = json.loads(path.read_text())
    if field == "ppo":
        config["training"]["ppo"]["gamma_per_step"] = .5
    else:
        config["reward_shaping"]["gamma"] = .5
    local = tmp_path / "discount-mismatch.json"
    local.write_text(json.dumps(config))
    monkeypatch.setattr(release, "CONFIG_PATH", str(local))
    with pytest.raises(ValueError, match="frozen discount"):
        release.load_config()

"""Same-money reward scale without changing execution, SOC or frozen references."""

import copy

import numpy as np
import pytest

from planning.corrector import Correction, FailureClass
from safe_rl.corrector_wrapper import CorrectorWrapper
from tests.test_m6p2b_terminal_inventory import _env


def test_common_sgd_degradation_scale_changes_only_that_reward_term(monkeypatch):
    monkeypatch.setattr('safe_rl.corrector_wrapper.correct', lambda *args, **kw: Correction(
        exec_compute_actions=[1.] * 20, exec_storage_action=-.05,
        business_gap=0., failure=FailureClass.NONE, reviewed=True))
    base = _env(horizon=8)
    candidate = copy.deepcopy(base)
    candidate.terminal_inventory_reward_version = 'common-sgd-degradation-v1'
    original = CorrectorWrapper(base, corrector_time_limit_s=.25)
    revised = CorrectorWrapper(candidate, corrector_time_limit_s=.25)
    raw = np.array([1.] * 20 + [-.05], dtype=np.float32)
    old_obs, old_reward, old_done, old_trunc, old = original.step(raw)
    obs, reward, done, trunc, info = revised.step(raw)
    slope = candidate.reward_cost_weight / candidate.cost_ref
    expected = -slope * info['bess_degradation_cost']
    assert info['r_bess_degradation'] == pytest.approx(expected)
    assert reward == pytest.approx(old_reward - old['r_bess_degradation'] + expected)
    np.testing.assert_array_equal(obs, old_obs)
    np.testing.assert_array_equal(info['raw_action'], raw)
    np.testing.assert_array_equal(info['exec_action'], old['exec_action'])
    assert (done, trunc) == (old_done, old_trunc)
    assert candidate.bess_soc == base.bess_soc
    assert candidate.normalization_refs() == base.normalization_refs()
    for key in old:
        if key.startswith('r_') and key != 'r_bess_degradation':
            assert info[key] == old[key]
    assert info['reward_semantics_audit']['version'] == 'common-sgd-degradation-v1'
    assert (info['reward_semantics_audit']['original_degradation_reward']
            == old['r_bess_degradation'])


def test_unknown_reward_semantics_are_rejected():
    env = _env(horizon=8)
    env.terminal_inventory_reward_version = 'unknown'
    with pytest.raises(ValueError, match='reward semantics'):
        CorrectorWrapper(env, corrector_time_limit_s=.25)

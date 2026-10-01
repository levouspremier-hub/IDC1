"""PPO and evaluation must see the same current terminal inventory state."""

import numpy as np
import pytest

from safe_rl.corrector_wrapper import CorrectorWrapper
from tests.test_m6p2b_terminal_inventory import _env

VERSION = 'terminal-state-observation-v1'


def test_current_soc_changes_policy_observation_when_base_features_are_identical():
    env = _env()
    env.terminal_inventory_observation_version = VERSION
    wrapped = CorrectorWrapper(env, corrector_time_limit_s=.25)
    before = wrapped.reset(seed=0)[0]
    base_before = env._get_obs().copy()
    # Same controlled current state; only the battery coordinate is varied.
    # No transition/qualification is executed or erased by this diagnostic.
    env.bess_soc = .9
    env.bess_energy_kWh = .9 * env.bess_capacity_kWh
    np.testing.assert_array_equal(base_before, env._get_obs())
    after = wrapped.policy_observation()
    np.testing.assert_array_equal(before[:-3], after[:-3])
    assert after.shape == (env.obs_dim + 3,)
    np.testing.assert_allclose(after[-3:], [.9, .5, 1.])
    assert before[-3] != after[-3]


def test_unknown_inventory_observation_version_is_rejected():
    env = _env()
    env.terminal_inventory_observation_version = 'unversioned-observation'
    with pytest.raises(ValueError, match='observation version'):
        CorrectorWrapper(env, corrector_time_limit_s=.25)


def test_terminal_observation_preserves_actual_inventory_without_future_base_features():
    env = _env(horizon=1, soc=.1)
    env.terminal_inventory_observation_version = VERSION
    wrapped = CorrectorWrapper(env, corrector_time_limit_s=.25)
    observation, _, done, _, info = wrapped.step(np.zeros(21, dtype=np.float32))
    assert done
    np.testing.assert_array_equal(observation[:-3], np.zeros(env.obs_dim))
    assert observation[-3] == pytest.approx(info['inventory_audit']['actual_soc'])
    assert observation[-2] == pytest.approx(.5)
    assert observation[-1] == 0.


def test_evaluation_adapter_preserves_versioned_wrapper_observation():
    from evaluation.adapter import _observation
    env = _env()
    env.terminal_inventory_observation_version = VERSION
    wrapped = CorrectorWrapper(env, corrector_time_limit_s=.25)
    observed = _observation(wrapped)
    assert observed.shape == (env.obs_dim + 3,)
    np.testing.assert_array_equal(observed, wrapped.policy_observation())


@pytest.mark.leakage
def test_inventory_observation_does_not_expose_future_truth_or_task_details():
    from safe_rl_v2.formal_train_loop import build_train_env, load_frozen_training_config
    env, _ = build_train_env(48, master_seed=0, config=load_frozen_training_config())
    env.terminal_inventory_enabled = True
    env.terminal_inventory_observation_version = VERSION
    wrapped = CorrectorWrapper(env, corrector_time_limit_s=.25)
    wrapped.reset(seed=0)
    before = wrapped.policy_observation()
    for channel in (env.price_t, env.T_amb, env.pv_t, env.wt_t, env.carbon_factor_t):
        channel[1:] += 1000.
    for task in env.tasks:
        if task.status == 'not_arrived':
            task.workload *= 1000.
            task.remaining_work *= 1000.
    np.testing.assert_array_equal(before, wrapped.policy_observation())

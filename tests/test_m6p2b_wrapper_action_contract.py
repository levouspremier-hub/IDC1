"""Reject unsupported actions before planning or advancing the episode."""

import numpy as np
import pytest

from planning.corrector import Correction, FailureClass
from safe_rl.corrector_wrapper import CorrectorWrapper
from tests.test_m6p2b_terminal_inventory import _env


@pytest.mark.parametrize("dimension", [20, 22, 23])
def test_wrapper_rejects_wrong_raw_dimension_without_advancing_or_projecting(
        monkeypatch, dimension):
    calls = []

    def project(*args, **kwargs):
        calls.append(args)
        return Correction(exec_compute_actions=[0.] * 20, exec_storage_action=0.,
                          business_gap=0., failure=FailureClass.NONE, reviewed=True)

    monkeypatch.setattr("safe_rl.corrector_wrapper.correct", project)
    env = _env(horizon=2)
    env.terminal_inventory_reward_version = "common-sgd-potential-smooth-v1"
    env.terminal_inventory_reward_gamma = .99498743710662
    wrapped = CorrectorWrapper(env, corrector_time_limit_s=.25)
    before = (env.current_step, env.bess_energy_kWh,
              [(t.task_id, t.remaining_work, t.status) for t in env.tasks])
    with pytest.raises(ValueError, match="21"):
        wrapped.step(np.zeros(dimension, dtype=np.float32))
    assert calls == []
    assert before == (env.current_step, env.bess_energy_kWh,
                      [(t.task_id, t.remaining_work, t.status) for t in env.tasks])

"""Reuse verified train inputs without sharing mutable episode or task state."""

import numpy as np
import pytest

from safe_rl_v2 import formal_train_loop as loop
from scenario import env_injection
from scenario.inventory_release import diagnostic_candidate_config


def test_cached_verified_inputs_do_not_share_task_specs_arrays_or_soc(monkeypatch):
    loop.clear_verified_train_input_cache()
    original = env_injection.build_verified_formal_env_injection
    calls = []
    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(env_injection, 'build_verified_formal_env_injection', counted)
    config = diagnostic_candidate_config()
    first, pristine = loop.build_train_env(48, master_seed=0, config=config)
    obs, _ = first.reset(seed=0)
    first.price_t[:] = 999.
    first.bess_soc = .1
    first.formal_injection.formal_refs['cost_ref'] = 999.
    first.formal_injection.task_specs[0]['workload'] *= 999.
    second, injected = loop.build_train_env(48, master_seed=0, config=config)
    next_obs, _ = second.reset(seed=0)
    assert len(calls) == 1
    assert pristine.provenance_hash == injected.provenance_hash
    np.testing.assert_array_equal(obs, next_obs)
    assert second.bess_soc == .5
    assert second.cost_ref == 60.
    assert not np.shares_memory(first.price_t, second.price_t)
    assert first.formal_injection is not second.formal_injection
    assert first.tasks[1] is not second.tasks[1]


def test_changed_verified_source_fingerprint_cannot_return_stale_cached_inputs(monkeypatch):
    loop.clear_verified_train_input_cache()
    config = diagnostic_candidate_config()
    fingerprint = ['verified-source-a']
    monkeypatch.setattr(loop, '_train_input_fingerprint', lambda: fingerprint[0])
    original = env_injection.build_verified_formal_env_injection
    def verified(*args, **kwargs):
        if fingerprint[0] != 'verified-source-a':
            raise env_injection.FormalInjectionError('source verification rejected changed bytes')
        return original(*args, **kwargs)
    monkeypatch.setattr(env_injection, 'build_verified_formal_env_injection', verified)
    loop.build_train_env(48, master_seed=0, config=config)
    fingerprint[0] = 'changed-source-b'
    with pytest.raises(env_injection.FormalInjectionError, match='changed bytes'):
        loop.build_train_env(48, master_seed=0, config=config)

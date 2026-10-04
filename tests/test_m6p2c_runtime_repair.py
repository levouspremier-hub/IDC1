"""Real A solve; controlled B outcomes test semantics, not timing performance."""
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.optimize

from contracts.models import DispatchProposal
from planning import model
from planning.corrector import correct
from tests.test_m44_corrector import _snapshot


def _proposal():
    return DispatchProposal(compute_actions=[0.5, 0.5], storage_action=0.3)


def _outcomes(monkeypatch, *, b_status=1, corrupt=False):
    original = scipy.optimize.milp
    calls = []

    def solve(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            return SimpleNamespace(status=b_status, x=None)
        result = original(**kwargs)
        assert result.status == 0
        if corrupt:
            result.x[0] = np.nan
        return result

    monkeypatch.setattr(scipy.optimize, 'milp', solve)
    return calls


def test_b_timeout_retains_validated_a(monkeypatch):
    calls = _outcomes(monkeypatch)
    result = model.solve_time_indexed_mip_raw_projection(_snapshot(), _proposal())
    assert len(calls) == 2
    assert result.stage_b_status == 'time_limit'
    assert result.execution_source == 'stage_a'
    assert result.candidate_check['passed'] is True
    assert result.failure_class == 'stage_b_timeout_feasible'
    assert sum(result.exec_compute_actions) > 0
    assert result.max_constraint_residual <= 1e-6


def test_corrector_exposes_timeout_separately_from_executable_action(monkeypatch):
    _outcomes(monkeypatch)
    result = correct(_snapshot(), _proposal(), time_limit_s=2.0)
    assert str(result.failure) == 'stage_b_timeout_feasible'
    assert result.execution_source == 'stage_a'
    assert result.executable is True
    assert sum(result.exec_compute_actions) > 0


def test_invalid_a_is_not_executed(monkeypatch):
    _outcomes(monkeypatch, corrupt=True)
    result = model.solve_time_indexed_mip_raw_projection(_snapshot(), _proposal())
    assert result.failure_class == 'solver_failure'
    assert result.execution_source == 'none'
    assert not result.candidate_check['passed']


def test_b_solver_error_does_not_use_a(monkeypatch):
    _outcomes(monkeypatch, b_status=4)
    result = model.solve_time_indexed_mip_raw_projection(_snapshot(), _proposal())
    assert result.failure_class == 'solver_failure'
    assert result.execution_source == 'none'


def test_budget_exhausted_after_a_retains_a_without_calling_b(monkeypatch):
    original = scipy.optimize.milp
    clock = [0.0]
    calls = []
    monkeypatch.setattr(model, '_monotonic', lambda: clock[0])

    def solve(**kwargs):
        result = original(**kwargs)
        calls.append(result)
        assert result.status == 0
        clock[0] = 2.0
        return result

    monkeypatch.setattr(scipy.optimize, 'milp', solve)
    result = model.solve_time_indexed_mip_raw_projection(
        _snapshot(), _proposal(), time_limit_s=1.0)
    assert len(calls) == 1
    assert result.stage_b_status == 'not_run'
    assert result.execution_source == 'stage_a'
    assert result.failure_class == 'stage_b_timeout_feasible'


@pytest.mark.parametrize('kind', ['bound', 'integer', 'row', 'nan'])
def test_candidate_checker_rejects_invalid_vectors(kind):
    from scipy.sparse import csr_matrix
    x = {'bound': [2., 0.], 'integer': [.5, .5],
         'row': [1., 1.], 'nan': [float('nan'), 0.]}[kind]
    check = model.check_projection_candidate(
        np.array(x), np.zeros(2), np.ones(2), np.array([1, 0]),
        csr_matrix([[1., 1.]]), np.array([1.]), np.array([1.]))
    assert not check['passed']


def test_rejected_correction_stops_before_environment_step(monkeypatch):
    from safe_rl import corrector_wrapper as wrapper
    from tests.test_m51b_rollout_collection import make_env
    env = make_env(horizon=2)
    wrapped = wrapper.CorrectorWrapper(env, corrector_time_limit_s=.25)
    wrapped.stop_on_unexecutable = True
    wrapped.reset(seed=0)
    called = []
    monkeypatch.setattr(env, 'step', lambda a: called.append(a))
    monkeypatch.setattr(wrapper, 'correct', lambda *a, **kw: SimpleNamespace(
        executable=False, failure='solver_failure', reason='controlled failure',
        stage_a_status='solver_error', stage_b_status='not_run', audit={},
        inventory_audit={}, candidate_check={}))
    with pytest.raises(wrapper.UnexecutableCorrectionError) as caught:
        wrapped.step(np.zeros(21, dtype=np.float32))
    assert called == []
    assert caught.value.evidence['environment_step_executed'] is False


def test_failed_rollout_cannot_update_adam_or_multipliers(monkeypatch):
    import torch

    from safe_rl_v2 import formal_train_loop as loop
    from safe_rl_v2.buffer import RolloutBuffer
    from safe_rl_v2.rollout import UnsafeRolloutError
    from scenario.inventory_release import load_config
    config = load_config()
    policy = loop.build_seeded_policy(config, obs_dim=523, seed=0)
    optimizer = loop.build_optimizer(config, policy)
    lag = loop.build_lagrangian(config)
    before = lag.state_dict()
    params = [p.detach().clone() for p in policy.parameters()]

    def fail(*a, **kw):
        raise UnsafeRolloutError('no candidate', RolloutBuffer(), {})

    monkeypatch.setattr(loop, '_collect_with_config', fail)
    with pytest.raises(UnsafeRolloutError):
        loop.run_training_batch(policy, optimizer, lag, torch.Generator(), torch.Generator(),
                                config=config, origins=[5040]*4, batch_index=0,
                                env_seed=0, corrector_time_limit_s=.25, master_seed=0)
    assert len(optimizer.state) == 0
    assert lag.state_dict() == before
    assert all(torch.equal(p, b) for p, b in zip(policy.parameters(), params, strict=True))


def test_initial_prebatch_checkpoint_keeps_rng_and_zero_cursor(tmp_path):
    import torch

    from checkpointing.versioned import read_checkpoint_payload
    from safe_rl_v2 import formal_train_loop as loop
    from scenario.inventory_release import load_config
    config = load_config()
    policy = loop.build_seeded_policy(config, obs_dim=523, seed=0)
    sampling, shuffle = torch.Generator().manual_seed(17), torch.Generator().manual_seed(19)
    path = tmp_path / 'before.pt'
    loop.save_resume_checkpoint(
        path, policy=policy, optimizer=loop.build_optimizer(config, policy),
        lagrangian=loop.build_lagrangian(config), sampling_generator=sampling,
        shuffle_generator=shuffle, config=config, origins=[5040]*4, next_batch_index=0,
        obs_dim=523, origin_provenance={}, master_seed=0)
    state = read_checkpoint_payload(path)['state']
    assert state['next_batch_index'] == 0
    assert torch.equal(state['sampling_generator'], sampling.get_state())
    assert state['source_ledger']['scenario_hash'] is None


def test_nonfinite_gradient_blocks_adam():
    import torch

    from safe_rl_v2.formal_train_loop import build_optimizer, build_seeded_policy
    from safe_rl_v2.ppo_update import minibatch_ppo_step
    from scenario.inventory_release import load_config
    config = load_config()
    policy = build_seeded_policy(config, obs_dim=523, seed=0)
    opt = build_optimizer(config, policy)
    obs = torch.zeros((2, 523))
    raw, logp, _ = policy.act(obs, generator=torch.Generator().manual_seed(0))
    handle = next(policy.parameters()).register_hook(lambda g: torch.full_like(g, float('nan')))
    with pytest.raises(ValueError, match='nonfinite gradient'):
        minibatch_ppo_step(
            policy, opt, observation=obs, raw_action=raw.detach(),
            old_raw_log_prob=logp.detach(), adv_reward=torch.ones(2),
            adv_business=torch.zeros(2), adv_carbon=torch.zeros(2),
            critic_targets={k: torch.zeros(2) for k in ('reward', 'business', 'carbon')},
            lambda_business=0., lambda_carbon=0., clip_epsilon=.2)
    handle.remove()
    assert len(opt.state) == 0


def test_terminal_inventory_is_checked_when_a_is_retained(monkeypatch):
    from planning.snapshot_adapter import build_snapshot
    from tests.test_m6p2b_terminal_inventory import _env
    original = scipy.optimize.milp

    def solve(**kwargs):
        # One-step inventory uses a closed-form reachability certificate; A then B.
        solve.calls += 1
        if solve.calls == 2:
            return SimpleNamespace(status=1, x=None)
        return original(**kwargs)

    solve.calls = 0
    monkeypatch.setattr(scipy.optimize, 'milp', solve)
    snap = build_snapshot(_env(horizon=1))
    proposal = DispatchProposal(compute_actions=[0.]*20, storage_action=1.)
    result = model.solve_time_indexed_mip_raw_projection(snap, proposal)
    assert result.execution_source == 'stage_a'
    assert result.candidate_check['passed']
    assert result.soc_kwh[-1] == pytest.approx(50., abs=1e-6)

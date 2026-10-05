"""Runtime candidates cannot inherit formal readiness or change research settings."""
import copy

import pytest

from scenario import runtime_release as runtime


@pytest.mark.parametrize('budget', [.25, .5, 1.])
def test_registered_budget_preserves_research_config(monkeypatch, budget):
    original = {'training': {'corrector': {'time_limit_s': .25}}, 'status': 'frozen'}
    monkeypatch.setattr(runtime, 'load_config', lambda: copy.deepcopy(original))
    config = runtime.candidate_config(budget)
    assert config['training']['corrector']['time_limit_s'] == budget
    assert config['runtime_execution_version'] == runtime.VERSION
    assert config['status'] == 'qualification_candidate'
    assert original['training']['corrector']['time_limit_s'] == .25


@pytest.mark.parametrize('budget', [0, .3, 2., float('nan')])
def test_unregistered_budget_rejected(budget):
    with pytest.raises(ValueError, match='budget'):
        runtime.candidate_config(budget)


def test_candidate_never_grants_formal_training():
    with pytest.raises(ValueError, match='formal'):
        runtime.require_candidate_scope(short=False)


def test_candidate_verification_rejects_changed_closure(monkeypatch, tmp_path):
    import json
    expected = {'config': {'runtime_budget_s': .25}, 'sources': {'model': 'original'}}
    path = tmp_path / 'candidate.json'
    path.write_text(json.dumps(expected))
    monkeypatch.setattr(runtime, 'build_candidate', lambda budget: expected)
    assert runtime.verify_candidate(path) == expected
    path.write_text(json.dumps({**expected, 'sources': {'model': 'changed'}}))
    with pytest.raises(ValueError, match='binding'):
        runtime.verify_candidate(path)


def test_qualification_cli_imports():
    from scripts import runtime_qualification
    assert runtime_qualification.ORIGINS == (5040, 5088, 5136, 5184, 5568, 6576)


def test_source_closure_excludes_untracked_historical_runners(monkeypatch):
    monkeypatch.setattr(runtime, '_git', lambda *args: 'scenario/a.py\nscripts/b.py\n')
    assert 'scenario/a.py' in runtime.source_paths()
    assert 'runs/writer.py' in runtime.source_paths()
    assert not any('runner.py' in p for p in runtime.source_paths())


def test_cold_input_uses_factory_validation_without_duplicate_outer_chain(monkeypatch):
    from safe_rl_v2 import formal_train_loop as loop
    from scenario import arrival_mapper
    loop.clear_verified_train_input_cache()
    original = arrival_mapper.load_verified_mapper_chain
    calls = []
    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(arrival_mapper, 'load_verified_mapper_chain', counted)
    injection = loop._verified_train_input(5040, 48, 48, 'uncached-equivalence-test')
    assert injection.split == 'train'
    assert len(calls) == 2


def test_strict_recording_keeps_original_failure_and_partial_records(monkeypatch):
    import numpy as np

    from safe_rl_v2.inventory_diagnostics import RecordingWrapper
    from tests.test_m51b_rollout_collection import make_env
    env = make_env(horizon=2)
    wrapped = RecordingWrapper(env, strict=True)
    error = RuntimeError('controlled solver refusal')
    error.evidence = {'environment_step_executed': False}
    def fail(action):
        raise error
    monkeypatch.setattr(env, 'step', fail)
    with pytest.raises(RuntimeError):
        wrapped.step(np.zeros(21))
    assert wrapped.failure is error
    assert wrapped.records == []


def test_qualification_stops_after_three_failed_budgets_without_training(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from scripts import runtime_qualification as q
    monkeypatch.setattr(q, 'ROOT', tmp_path)
    monkeypatch.setattr(q, 'verify_candidate', lambda p: {'config': {'runtime_budget_s': p}})
    monkeypatch.setattr(q, 'sha', lambda p: 'test-hash')
    saves, calls = [], []
    monkeypatch.setattr(q, 'save', lambda *a, **kw: saves.append(a))
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1 if 'diagnose' in argv else 0)
    monkeypatch.setattr(q.subprocess, 'run', run)
    (tmp_path / 'runs').mkdir()
    assert q.qualify('qualify', [.25, .5, 1.]) == 1
    assert len(calls) == 5
    assert not any('safe_rl_v2.inventory_train' in c for c in calls)
    assert saves[-1][3]['formal_training_ready'] is False
    assert saves[-1][3]['selected_budget_s'] is None


def test_positive_clipped_ratio_overflow_has_zero_finite_actor_gradient():
    import torch

    from safe_rl_v2.ppo_objective import ppo_clipped_actor_objective
    class Policy:
        value = torch.tensor([0.], requires_grad=True)
        def evaluate_raw_actions(self, observation, raw_action):
            return self.value
    policy = Policy()
    out = ppo_clipped_actor_objective(
        policy, observation=torch.zeros((1, 523)), raw_action=torch.zeros((1, 21)),
        old_raw_log_prob=torch.tensor([-1e30]), adv_reward=torch.tensor([2.]),
        adv_business=torch.zeros(1), adv_carbon=torch.zeros(1),
        lambda_business=0., lambda_carbon=0., clip_epsilon=.2)
    assert out['loss'].item() == pytest.approx(-2.4)
    out['loss'].backward()
    assert torch.isfinite(policy.value.grad).all()
    assert policy.value.grad.item() == 0.


@pytest.mark.parametrize('sign', [-1., 0., 1.])
def test_stable_clipping_preserves_finite_loss_and_actor_gradient(sign):
    import torch

    from safe_rl_v2.ppo_objective import clipped_surrogate, ppo_clipped_actor_objective
    values = torch.tensor([-5., -.3, 0., .1, .3, 2.], requires_grad=True)
    advantage = torch.full((6,), sign)
    reference = -clipped_surrogate(values.exp(), advantage, clip_epsilon=.2).mean()
    expected_grad = torch.autograd.grad(reference, values)[0]
    class Policy:
        def evaluate_raw_actions(self, observation, raw_action):
            return values
    result = ppo_clipped_actor_objective(
        Policy(), observation=torch.zeros((6, 523)), raw_action=torch.zeros((6, 21)),
        old_raw_log_prob=torch.zeros(6), adv_reward=advantage,
        adv_business=torch.zeros(6), adv_carbon=torch.zeros(6),
        lambda_business=0., lambda_carbon=0., clip_epsilon=.2)
    actual_grad = torch.autograd.grad(result['loss'], values)[0]
    assert torch.equal(result['loss'], reference)
    assert torch.equal(actual_grad, expected_grad)


def test_formal_release_refuses_partial_qualification(monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(runtime, 'verify_candidate',
                        lambda p: {'config': {'runtime_budget_s': .25}})
    (tmp_path / 'report.json').write_text(json.dumps({
        'passed': True, 'qualification_complete': False, 'selected_budget_s': .25,
        'formal_512_started': False}))
    with pytest.raises(ValueError, match='complete'):
        runtime.qualification_evidence(tmp_path, 'candidate')


def test_formal_release_cannot_load_short_evaluation_role():
    from checkpointing.inventory_eval_input import load_policy
    with pytest.raises(ValueError, match='short'):
        load_policy('missing-policy', formal=False, runtime_release='unverified-release')


def test_benchmark_distinguishes_retained_a_from_optimal_and_failure():
    from scripts import benchmark_corrector as bench
    assert bench.classify_outcome('stage_b_timeout_feasible') == 'executable_degraded'
    assert bench.classify_outcome('none') == 'executable_candidate'
    assert bench.classify_outcome('timeout') == 'timeout'
    assert bench.classify_outcome('unregistered') == 'non_timeout_failure'


def test_semantic_probes_exclude_only_new_pure_wall_timers():
    from scripts import probe_corrector_repro, probe_rollout_deterministic
    expected = {'correction_solve_time_s', 'stage_a_solve_time_s',
                'stage_b_solve_time_s', 'correction_total_wall_s',
                'correction_snapshot_wall_s'}
    assert set(probe_corrector_repro.WALL_CLOCK_KEYS) == expected
    assert set(probe_rollout_deterministic.WALL_CLOCK_ONLY_KEYS) == expected

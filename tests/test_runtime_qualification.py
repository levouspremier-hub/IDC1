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

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

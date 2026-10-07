"""Numerical counterexamples must not become lexicographic bounds or releases."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from planning import model


def _certify(x, matrix=None, row_lb=None, row_ub=None):
    from planning.numeric_contract import certify_witness
    return certify_witness(np.asarray(x), np.zeros(2), np.ones(2), np.array([1, 0]),
        csr_matrix([[1., 0.]]) if matrix is None else matrix,
        np.array([0.]) if row_lb is None else row_lb,
        np.array([1.]) if row_ub is None else row_ub)


@pytest.mark.parametrize('ghost', [5.156e-9, 8.131518354187604e-10, 1e-10])
def test_physical_acceptance_is_not_an_integer_witness(ghost):
    matrix = csr_matrix([[-20., 1.]])
    x = np.array([ghost, ghost * 20])
    assert model.check_projection_candidate(x, np.zeros(2), np.ones(2),
        np.array([1, 0]), matrix, np.array([-np.inf]), np.zeros(1))['passed']
    witness, audit = _certify(x, matrix, np.array([-np.inf]), np.zeros(1))
    assert witness is None and not audit['passed']
    assert audit['reason'] == 'noncanonical_integer'


def test_machine_representation_can_be_canonicalized_without_mutating_input():
    x = np.array([1. - 2 * np.spacing(1.), .25])
    witness, audit = _certify(x)
    assert audit['passed'] and witness[0] == 1.
    assert witness[1] == x[1] and x[0] != 1.


def test_canonicalization_must_recheck_original_matrix():
    x = [1. - 2 * np.spacing(1.), .25]
    witness, audit = _certify(x, csr_matrix([[1e6, 0.]]),
        np.array([-np.inf]), np.array([1e6 - 2e-10]))
    assert witness is None and audit['reason'] == 'canonical_primal_infeasible'


@pytest.mark.parametrize('x', [[np.nan, 0.], [np.inf, 0.], [0.]])
def test_invalid_witness_cannot_reach_stage_b(x):
    witness, audit = _certify(x)
    assert witness is None and not audit['passed']


def test_subphysical_continuous_violation_cannot_define_a_primary_bound():
    witness, audit = _certify([0., 2e-9], csr_matrix([[0., 1.]]),
        np.zeros(1), np.zeros(1))
    assert witness is None and audit['reason'] == 'canonical_primal_infeasible'


@pytest.mark.parametrize('lower', [.8, 1.1, np.nan, None])
def test_solver_optimal_label_is_insufficient_for_primary_optimality(lower):
    from planning.numeric_contract import certify_primary_objective
    result = SimpleNamespace(status=0, fun=1., mip_dual_bound=lower)
    audit = certify_primary_objective(result, np.array([1., 0.]), np.array([1., 0.]),
                                     tolerance=1e-6)
    assert not audit['passed']


def test_attained_primary_value_and_lower_bound_certify_the_original_offset():
    from planning.numeric_contract import certify_primary_objective
    result = SimpleNamespace(status=0, fun=.5, mip_dual_bound=.5 - 2e-7)
    audit = certify_primary_objective(result, np.array([.5, 0.]), np.array([1., 0.]),
                                     tolerance=1e-6)
    assert audit['passed'] and audit['objective'] == .5


def test_real_inventory_handoff_rejects_a_ghost_before_calling_b(monkeypatch):
    import scipy.optimize

    from contracts.inventory import InventorySnapshot
    from contracts.models import DispatchProposal
    from planning.corrector import correct
    path = Path(__file__).parent / 'fixtures/m6p2c_seed1_origin9120_step18.json'
    case = json.loads(path.read_text())
    real = scipy.optimize.milp
    calls = []
    def solve(**kwargs):
        calls.append(kwargs)
        result = real(**kwargs)
        if len(calls) == 1:
            assert result.status == 0
            indices = np.flatnonzero(kwargs['integrality'])
            z = next(i for i in indices if result.x[i] == 0.)
            result.x[z] = 8.13e-10
        return result
    monkeypatch.setattr(scipy.optimize, 'milp', solve)
    result = correct(InventorySnapshot.model_validate(case['snapshot']),
        DispatchProposal.model_validate(case['proposal']), time_limit_s=2.)
    assert len(calls) == 1, 'an uncertified A must never constrain or launch B'
    assert not result.executable
    assert result.inventory_audit['stage_a_witness']['reason'] == 'noncanonical_integer'


def test_candidate_binds_the_numeric_contract_and_two_learning_cycles():
    from scenario.runtime_release import candidate_config
    config = candidate_config(.50)
    contract = config['numeric_contract']
    assert contract['integer_representation_ulps'] == 8
    assert contract['primal_witness_tolerance'] == 1e-10
    assert contract['learning_origin_cycles'] == 2
    assert config['training']['corrector']['primary_mip_rel_gap'] == 0.
    assert config['training']['corrector']['time_limit_s'] == .50


def test_learning_coverage_follows_two_complete_registered_origin_cycles():
    from collections import Counter

    from safe_rl_v2.formal_train import preregistered_batches
    from safe_rl_v2.inventory_train import coverage_batches
    from scenario.inventory_release import load_matrix
    matrix = load_matrix()
    for seed in (0, 1, 2):
        batches = coverage_batches(matrix, seed)
        assert batches == preregistered_batches(matrix, seed)[:len(batches)]
        counts = Counter(o for b in batches for o in b)
        assert set(counts) == set(matrix['training_schedule']['origin_pool'])
        assert min(counts.values()) >= 2
        assert len(batches) == 106


def test_coverage_mode_cannot_use_old_formal_authorization_or_implicit_config():
    from safe_rl_v2.inventory_train import _run
    for candidate, release, short in [(None, None, False), ('candidate', 'release', False),
                                     ('candidate', None, True)]:
        args = SimpleNamespace(coverage=True, short=short, seed0_formal=False,
            runtime_candidate=candidate, runtime_release=release)
        with pytest.raises(ValueError, match='coverage'):
            _run(args)


def test_new_qualification_requires_numeric_replay_and_three_learning_runs(monkeypatch, tmp_path):
    from scenario.runtime_release import candidate_config
    from scripts import runtime_qualification as q
    config = candidate_config(.50)
    monkeypatch.setattr(q, 'ROOT', tmp_path)
    monkeypatch.setattr(q, 'verify_candidate',
        lambda p: {'config': {**copy.deepcopy(config), 'runtime_budget_s': p}})
    monkeypatch.setattr(q, 'sha', lambda p: 'fixed-hash')
    monkeypatch.setattr(q, 'save', lambda *a, **kw: None)
    monkeypatch.setattr(q, 'short_run_qualified', lambda *a: True)
    monkeypatch.setattr(q, 'coverage_run_qualified', lambda *a: True, raising=False)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(q.subprocess, 'run', run)
    (tmp_path / 'runs').mkdir()
    assert q.qualify('coverage', [.25, .50, 1.], budget=.50) == 0
    assert any('numerics' in argv for argv in calls)
    coverage = [argv for argv in calls if '--coverage' in argv]
    assert [argv[argv.index('--seed') + 1] for argv in coverage] == ['0', '1', '2']
    assert all('--runtime-candidate' in argv and '--runtime-release' not in argv
               for argv in coverage)

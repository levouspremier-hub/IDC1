"""One-step end inventory implies storage bounds, including subnanowatt flows."""
import json
from pathlib import Path

import numpy as np
import pytest

from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct


def _case():
    return json.loads((Path(__file__).parent / 'fixtures' /
                      'm6p2c_seed0_origin6192_step47.json').read_text())


def test_recorded_last_step_failure_has_an_executable_integer_solution():
    case = _case()
    snapshot = InventorySnapshot.model_validate(case['snapshot'])
    result = correct(snapshot, DispatchProposal.model_validate(case['proposal']), time_limit_s=.50)
    assert result.stage_a_status == result.stage_b_status == 'optimal'
    assert result.executable and result.candidate_check['passed']
    assert result.candidate_check['integer_residual'] == 0.
    bounds = result.inventory_audit['one_step_storage_bounds']
    expected = ((snapshot.soc_kwh - snapshot.terminal_inventory.target_kwh)
                * snapshot.bess_discharge_efficiency / snapshot.delta_t_hours)
    assert bounds['discharge_lower_kw'] == bounds['discharge_upper_kw'] == expected
    assert expected > 0., 'subnanowatt discharge must not be rounded away'


@pytest.mark.parametrize('delta', [-1e-7, -1e-9, -1e-10, -np.spacing(50.), 0.,
                                  np.spacing(50.), 1e-10, 1e-9, 1e-7])
def test_terminal_roundoff_on_both_sides_retains_the_original_target(delta):
    case = _case()
    snapshot = InventorySnapshot.model_validate({**case['snapshot'], 'soc_kwh': 50. + delta})
    result = correct(snapshot, DispatchProposal.model_validate(case['proposal']), time_limit_s=.50)
    assert result.stage_a_status == result.stage_b_status == 'optimal'
    assert result.executable and result.candidate_check['integer_residual'] == 0.
    assert result.inventory_audit['stage_a_witness']['passed']
    assert result.inventory_audit['execution_witness']['passed']


@pytest.mark.parametrize('lower,upper', [(49.,49.), (49.,50.), (49.,51.), (50.,50.),
                                      (50.,51.), (51.,51.)])
def test_derived_bounds_preserve_both_exact_mutually_exclusive_domains(lower, upper):
    from planning.numeric_contract import one_step_storage_bounds
    derived = one_step_storage_bounds(50., lower, upper, .5, .95, .9)
    # Enumerate each original binary domain at endpoints and interior SOC values.
    for energy in np.linspace(lower, upper, 5):
        delta = energy - 50.
        charge = max(delta, 0.) / (.95 * .5)
        discharge = max(-delta, 0.) * .9 / .5
        z_values = (1.,) if charge > 0 else (0.,) if discharge > 0 else (0., 1.)
        assert derived['charge_lower_kw'] <= charge <= derived['charge_upper_kw']
        assert derived['discharge_lower_kw'] <= discharge <= derived['discharge_upper_kw']
        assert all(derived['mode_lower'] <= z <= derived['mode_upper'] for z in z_values)


def test_candidate_binds_tail_bound_derivation_and_the_new_original_failure():
    from scenario.runtime_release import NUMERIC_FIXTURES, candidate_config
    assert 'm6p2c_seed0_origin6192_step47.json' in NUMERIC_FIXTURES
    assert candidate_config(.50)['numeric_contract']['one_step_storage_bounds_version'] == (
        'soc-balance-exclusion-one-step-v1')


def test_qualification_revalidates_same_candidate_shorts_without_rerunning(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from scripts import runtime_qualification as q
    monkeypatch.setattr(q, 'ROOT', tmp_path)
    monkeypatch.setattr(q, 'verify_candidate', lambda p: {'config': {
        'runtime_budget_s': p, 'numeric_contract': {'version': 'fixed'}}})
    monkeypatch.setattr(q, 'sha', lambda p: 'bound-hash')
    monkeypatch.setattr(q, 'save', lambda *a, **kw: None)
    monkeypatch.setattr(q, 'coverage_run_qualified', lambda *a: True)
    verified, calls = [], []
    def reuse(folder, candidate, seed):
        verified.append((folder.name, candidate, seed))
        return True
    monkeypatch.setattr(q, 'reused_short_qualified', reuse, raising=False)
    def run(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(q.subprocess, 'run', run)
    (tmp_path / 'runs').mkdir()
    assert q.qualify('complete', [.25,.50,1.], budget=.50, reuse_short_prefix='fixed_short') == 0
    assert verified == [(f'fixed_short_seed{s}', .50, s) for s in (0,1,2)]
    assert not any('--short' in argv for argv in calls)
    assert sum('--coverage' in argv for argv in calls) == 3
    assert all(any(action in argv for argv in calls)
               for action in ['gate','resume-audit','numerics','diagnose','soak'])


def test_reused_short_failure_blocks_learning_and_formal_release(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from scripts import runtime_qualification as q
    monkeypatch.setattr(q, 'ROOT', tmp_path)
    monkeypatch.setattr(q, 'verify_candidate', lambda p: {'config': {
        'runtime_budget_s': p, 'numeric_contract': {'version': 'fixed'}}})
    monkeypatch.setattr(q, 'sha', lambda p: 'bound-hash')
    monkeypatch.setattr(q, 'save', lambda *a, **kw: None)
    monkeypatch.setattr(q, 'reused_short_qualified', lambda *a: False, raising=False)
    calls = []
    def run(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(q.subprocess, 'run', run)
    (tmp_path / 'runs').mkdir()
    assert q.qualify('bad', [.25,.50,1.], budget=.50, reuse_short_prefix='fixed_short') == 1
    assert not any('--coverage' in argv or 'soak' in argv for argv in calls)


def test_tiny_soc_perturbation_does_not_create_a_fractional_charging_mode():
    path = Path(__file__).parent / 'fixtures/m6p2c_seed1_origin1008_step18.json'
    case = json.loads(path.read_text())
    snapshot = InventorySnapshot.model_validate({**case['snapshot'],
        'soc_kwh': case['snapshot']['soc_kwh'] + 1e-9})
    result = correct(snapshot, DispatchProposal.model_validate(case['proposal']), time_limit_s=.50)
    assert result.stage_a_status == result.stage_b_status == 'optimal'
    assert result.executable and result.candidate_check['integer_residual'] == 0.
    upper = result.inventory_audit['coupled_charge_domain_upper_kw'][0]
    assert 0 < upper < snapshot.bess_charge_power_max_kw


@pytest.mark.parametrize('headroom,minimum,hardware', [(3.,1.,20.), (0.,0.,20.),
                                                     (1.,2.,20.), (30.,0.,20.)])
def test_charge_domain_cut_preserves_every_original_binary_domain(headroom, minimum, hardware):
    from planning.numeric_contract import coupled_charge_domain_upper
    upper = coupled_charge_domain_upper(hardware, headroom, minimum)
    for z in (0.,1.):
        for service in [minimum, minimum+1., minimum+10.]:
            for charge in [0., .01, 1., 2., 3., 20.]:
                original = charge <= hardware*z and (z==0 or charge+service <= headroom)
                strengthened = original and charge <= upper*z
                assert original == strengthened


def test_stage_b_does_not_misclassify_certified_a_after_one_ulp_soc_change():
    case = json.loads((Path(__file__).parent / 'fixtures' /
                       'm6p2c_seed1_origin1008_step18.json').read_text())
    base = case['snapshot']['soc_kwh']
    snapshot = InventorySnapshot.model_validate({**case['snapshot'], 'soc_kwh':
                                               base + np.spacing(base)})
    result = correct(snapshot, DispatchProposal.model_validate(case['proposal']), time_limit_s=.50)
    assert result.inventory_audit['stage_a_witness']['passed']
    assert result.stage_b_status == 'optimal'
    assert result.executable and result.inventory_audit['execution_witness']['passed']


def test_stage_b_translation_preserves_original_domain_rows_and_economic_order():
    from scipy.optimize import Bounds
    from scipy.sparse import csr_matrix
    from planning.numeric_contract import translate_continuous_origin
    matrix = csr_matrix([[1., -20., 2.], [-1., 3., 1.]])
    bounds = Bounds([0.,0.,-10.], [20.,1.,10.])
    anchor = np.array([3.,1.,-2.])
    row_lower, row_upper = np.array([-30.,-10.]), np.array([30.,10.])
    shift, translated, lower, upper = translate_continuous_origin(
        anchor, [0,1,0], bounds, matrix, row_lower, row_upper)
    assert np.array_equal(shift, [3.,0.,-2.])
    assert translated.lb[1] == bounds.lb[1] and translated.ub[1] == bounds.ub[1]
    objective = np.array([.2,0.,.7])
    for original in [np.array([0.,0.,0.]), anchor, np.array([20.,1.,10.])]:
        delta = original-shift
        assert np.array_equal(delta+shift, original)
        assert np.array_equal(matrix@delta-lower, matrix@original-row_lower)
        assert np.array_equal(upper-matrix@delta, row_upper-matrix@original)
        assert np.array_equal(delta-translated.lb, original-bounds.lb)
        assert np.array_equal(translated.ub-delta, bounds.ub-original)
        assert np.isclose(objective@delta+objective@shift, objective@original)


def test_host_short_near_integer_b_is_certified_in_original_primal_domain():
    case = json.loads((Path(__file__).parent / 'fixtures' /
                       'm6p2c_seed0_origin384_step36.json').read_text())
    snapshot = InventorySnapshot.model_validate(case['snapshot'])
    result = correct(snapshot, DispatchProposal.model_validate(case['proposal']), time_limit_s=.50)
    assert result.stage_a_status == result.stage_b_status == 'optimal'
    assert result.executable and result.candidate_check['integer_residual'] == 0.
    assert result.inventory_audit['execution_witness']['passed']


def test_polish_fixes_the_candidate_integer_mode_and_rechecks_original_rows():
    from scipy.sparse import csr_matrix
    from planning.numeric_contract import certify_or_polish_witness
    matrix = csr_matrix([[-20.,1.]])
    witness, audit = certify_or_polish_witness(
        [5e-11,1e-9], [0.,0.], [1.,20.], [1,0], matrix, [-np.inf], [0.],
        objective=[0.,1.], remaining=lambda: .5)
    assert audit['passed'] and audit['primal_polish']['attempted']
    assert np.array_equal(witness, [0.,0.])
    assert audit['integer_residual_after']==0 and audit['row_residual']<=1e-10


def test_polish_does_not_accept_a_noninteger_candidate_or_an_expired_deadline(monkeypatch):
    import scipy.optimize
    from scipy.sparse import csr_matrix
    from planning.numeric_contract import certify_or_polish_witness
    def never(**kw):
        raise AssertionError('no LP may be called')
    monkeypatch.setattr(scipy.optimize, 'linprog', never)
    for x, budget in [([1e-9,1e-8],.5), ([5e-11,1e-9],0.)]:
        witness, audit = certify_or_polish_witness(
            x,[0.,0.],[1.,20.],[1,0],csr_matrix([[-20.,1.]]),[-np.inf],[0.],
            objective=[0.,1.],remaining=lambda b=budget:b)
        assert witness is None and not audit['passed']

"""Original seed1 near-integer B must satisfy the unchanged primary offset."""
import json
from pathlib import Path

from contracts.inventory import InventorySnapshot
from contracts.models import DispatchProposal
from planning.corrector import correct


def test_recorded_origin2544_b_has_exact_original_feasible_witness():
    fixture = Path(__file__).parent / 'fixtures' / 'm6p2c_seed1_origin2544_step23.json'
    case = json.loads(fixture.read_text())
    result = correct(InventorySnapshot.model_validate(case['snapshot']),
                     DispatchProposal.model_validate(case['proposal']), time_limit_s=.50)
    assert result.inventory_audit['stage_a_witness']['passed']
    assert result.inventory_audit['primary_objective_certificate']['passed']
    assert result.executable and result.stage_b_status == 'optimal'
    assert result.candidate_check['integer_residual'] == 0.


def test_nearest_capacity_sufficient_does_not_force_ghost_mode(monkeypatch):
    import numpy as np
    import scipy.optimize
    from scipy.sparse import csr_matrix

    from planning.numeric_contract import certify_or_polish_witness

    captured = []

    def rejected_lp(**kwargs):
        captured.append(kwargs['bounds'])
        return scipy.optimize.OptimizeResult(status=2, message='controlled rejection')

    monkeypatch.setattr(scipy.optimize, 'linprog', rejected_lp)
    matrix = csr_matrix([[0., 1., 0., 0., 0., 0.]])
    certify_or_polish_witness(
        np.array([7e-11, 1.8e-10, 0., 1., .0254, 0.]),
        np.zeros(6), np.array([1., 2.5, 20., 1., .025426, 20.]),
        np.array([1, 0, 0, 1, 0, 0]), matrix, np.array([-np.inf]), np.array([0.]),
        objective=np.zeros(6), remaining=lambda: .5,
        storage_modes=([0, 3], [1, 4], [2, 5]), storage_requirements=(True, False),
        storage_required_power=(.025403, 0.))
    assert len(captured) == 1
    assert captured[0][0, 0] == captured[0][0, 1] == 0.


def test_nearest_capacity_insufficient_still_supports_fractional_mode(monkeypatch):
    import numpy as np
    import scipy.optimize
    from scipy.sparse import csr_matrix

    from planning.numeric_contract import certify_or_polish_witness

    captured = []

    def rejected_lp(**kwargs):
        captured.append(kwargs['bounds'])
        return scipy.optimize.OptimizeResult(status=2, message='controlled rejection')

    monkeypatch.setattr(scipy.optimize, 'linprog', rejected_lp)
    certify_or_polish_witness(
        np.array([7e-11, 1.8e-10, 0.]), np.zeros(3), np.array([1., 2.5, 20.]),
        np.array([1, 0, 0]), csr_matrix([[0., 1., 0.]]),
        np.array([-np.inf]), np.array([0.]), objective=np.zeros(3),
        remaining=lambda: .5, storage_modes=([0], [1], [2]),
        storage_requirements=(True, False), storage_required_power=(1.8e-10, 0.))
    assert len(captured) == 1
    assert captured[0][0, 0] == captured[0][0, 1] == 1.


def test_discharge_capacity_support_is_symmetric(monkeypatch):
    import numpy as np
    import scipy.optimize
    from scipy.sparse import csr_matrix

    from planning.numeric_contract import certify_or_polish_witness

    captured = []

    def rejected_lp(**kwargs):
        captured.append(kwargs['bounds'])
        return scipy.optimize.OptimizeResult(status=2, message='controlled rejection')

    monkeypatch.setattr(scipy.optimize, 'linprog', rejected_lp)
    for other_discharge_mode, expected in [(0., 1.), (1., 0.)]:
        certify_or_polish_witness(
            np.array([1. - 7e-11, 0., 1.8e-10, other_discharge_mode, 0., 0.]),
            np.zeros(6), np.array([1., 20., 2.5, 1., 20., .025426]),
            np.array([1, 0, 0, 1, 0, 0]), csr_matrix([[0., 0., 1., 0., 0., 0.]]),
            np.array([-np.inf]), np.array([0.]), objective=np.zeros(6),
            remaining=lambda: .5, storage_modes=([0, 3], [1, 4], [2, 5]),
            storage_requirements=(False, True), storage_required_power=(0., .025403))
        assert captured[-1][0, 0] == captured[-1][0, 1] == expected
    assert len(captured) == 2

"""Solver row copies must retain every finite nonzero coefficient and bound."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from planning.numeric_contract import scale_constraint_rows


def test_quarter_origin_is_a_bijection_and_preserves_integer_domains():
    from scipy.optimize import Bounds

    from planning.numeric_contract import translate_continuous_origin
    anchor = np.array([1., 20., 2e-10])
    bounds = Bounds(np.zeros(3), np.array([1., 30., 1.]))
    matrix = csr_matrix([[1., 2., -3.]])
    lo, hi = np.array([0.]), np.array([50.])
    shift, translated, lower, upper = translate_continuous_origin(
        anchor, np.array([1, 0, 0]), bounds, matrix, lo, hi, fraction=.25)
    assert np.array_equal(shift, [0., 5., 5e-11])
    assert translated.lb[0] == 0. and translated.ub[0] == 1.
    assert np.array_equal(translated.lb + shift, bounds.lb)
    assert np.array_equal(translated.ub + shift, bounds.ub)
    assert np.array_equal(lower + matrix @ shift, lo)
    assert np.array_equal(upper + matrix @ shift, hi)


def test_row_scaling_is_exactly_reversible_for_normal_binary64_values():
    matrix = csr_matrix([[50000.00000641581, -.024999987500006246, 1.],
                         [3.141592653589793, 2e-10, 0.]])
    lo = np.array([1.835836989154186, -np.inf])
    hi = np.array([1.835836989154186, 5.00000000001])
    scaled, lower, upper, audit = scale_constraint_rows(matrix, lo, hi)
    factors = np.array(audit['factors'])
    assert np.array_equal(scaled.toarray() / factors[:, None], matrix.toarray())
    assert np.array_equal(lower / factors, lo)
    assert np.array_equal(upper / factors, hi)


def test_latest_full_discharge_uses_the_original_objective_and_certificate():
    import json
    from pathlib import Path

    from contracts.inventory import InventorySnapshot
    from contracts.models import DispatchProposal
    from planning.corrector import correct
    case = json.loads((Path(__file__).parent / 'fixtures' /
                       'm6p2c_seed0_origin9744_step44.json').read_text())
    result = correct(InventorySnapshot.model_validate(case['snapshot']),
                     DispatchProposal(compute_actions=[1.] * 20, storage_action=1.),
                     time_limit_s=.50)
    assert result.executable and result.stage_b_status == 'optimal'
    assert result.inventory_audit['execution_witness']['passed']
    assert result.inventory_audit['primary_objective_certificate']['passed']


@pytest.mark.parametrize(('matrix', 'lower', 'upper'), [
    ([[np.inf]], [0.], [1.]), ([[np.nan]], [0.], [1.]),
    ([[1.]], [np.nan], [1.]), ([[1.]], [0.], [np.nan]),
    ([[1.]], [2.], [1.]), ([[1.]], [np.inf], [np.inf]),
    ([[1.]], [-np.inf], [-np.inf]), ([[1.]], [0., 0.], [1.]),
])
def test_invalid_row_transformation_is_explicitly_rejected(matrix, lower, upper):
    with pytest.raises(ValueError, match='row'):
        scale_constraint_rows(csr_matrix(matrix), np.array(lower), np.array(upper))


def test_underflow_never_drops_nonzero_coefficient_or_finite_bound():
    tiny = np.nextafter(0., 1.)
    original = csr_matrix([[50000., tiny], [50000., 0.]])
    lower = np.array([0., tiny])
    upper = np.array([1., 1.])
    scaled, lo, hi, audit = scale_constraint_rows(original, lower, upper)
    assert scaled[0, 1] != 0. and lo[1] != 0.
    assert audit['underflow_guarded_rows'] == 2
    assert audit['factors'] == [1., 1.]
    assert np.array_equal(scaled.toarray(), original.toarray())
    assert np.array_equal(lo, lower) and np.array_equal(hi, upper)


def test_empty_zero_and_infinite_bound_rows_have_reproducible_audit():
    for matrix, lo, hi in (
            (csr_matrix((0, 2)), np.array([]), np.array([])),
            (csr_matrix([[0., 0.]]), np.array([-np.inf]), np.array([np.inf]))):
        scaled, lower, upper, audit = scale_constraint_rows(matrix, lo, hi)
        assert scaled.shape == matrix.shape
        assert np.array_equal(lower, lo) and np.array_equal(upper, hi)
        assert audit['original_input_sha256'] and audit['scaled_input_sha256']
        assert audit['factors'] == [1.] * matrix.shape[0]


def test_original_infeasible_intervals_can_only_be_preserved_explicitly():
    lo = np.array([np.nextafter(27., np.inf)])
    hi = np.array([27.])
    scaled, lower, upper, audit = scale_constraint_rows(
        csr_matrix([[1.]]), lo, hi, preserve_infeasible_intervals=True)
    assert lower[0] > upper[0]
    assert np.array_equal(lo, lower) and np.array_equal(hi, upper)
    assert audit['inverted_intervals_preserved'] == 1
    assert scaled[0, 0] == 1.

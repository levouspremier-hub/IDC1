"""Solver row copies must retain every finite nonzero coefficient and bound."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from planning.numeric_contract import scale_constraint_rows


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

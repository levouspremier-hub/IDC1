"""Sparse assembly must preserve the existing projection feasible domain."""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

import planning.model as model


def test_direct_sparse_assembly_preserves_all_coefficients_and_zero_rows():
    rows = [{0: 1., 4: -1e-12}, {}, {1: 0., 3: 2.5}, {4: 7.}]
    legacy = lil_matrix((len(rows), 6))
    for r, row in enumerate(rows):
        for c, v in row.items():
            legacy[r, c] = v
    actual = model._sparse_rows(rows, 6)
    assert isinstance(actual, csr_matrix)
    np.testing.assert_array_equal(actual.toarray(), legacy.toarray())
    assert actual.shape == (4, 6)


def test_appended_stage_b_row_keeps_stage_a_constraints_byte_exact():
    rows = [{0: 1., 4: -1e-12}, {}, {1: 0., 3: 2.5}]
    a = model._sparse_rows(rows, 6)
    b = model._append_sparse_row(a, {1: .05, 5: 1.})
    np.testing.assert_array_equal(b[:-1].toarray(), a.toarray())
    np.testing.assert_array_equal(b[-1].toarray(), [[0., .05, 0., 0., 0., 1.]])

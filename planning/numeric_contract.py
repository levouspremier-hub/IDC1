"""Certificates for solver witnesses used by lexicographic inventory planning.

Physical acceptance is intentionally separate from numerical optimization
acceptance. A subphysical residual must not define the next stage's feasible set.
"""
from __future__ import annotations

import numpy as np

INTEGER_REPRESENTATION_ULPS = 8
PRIMAL_WITNESS_TOLERANCE = 1e-10
NUMERIC_CONTRACT_VERSION = 'canonical-integer-lexicographic-v1'


def certify_witness(x, lb, ub, integrality, matrix, row_lb, row_ub):
    """Certify an exact integer assignment against the unchanged original model.

Continuous variables, model rows and bounds are never repaired or relaxed.
The supplied vector is not mutated. Numerical ghosts are rejected, not rounded
    into a different integer solution under the physical acceptance tolerance.
    Solver arithmetic outside the machine representation bound needs a full
    exact-assignment certificate and must remain below the frozen solver tolerance.
"""
    audit = {'version': NUMERIC_CONTRACT_VERSION, 'passed': False,
             'primal_tolerance': PRIMAL_WITNESS_TOLERANCE,
             'integer_representation_ulps': INTEGER_REPRESENTATION_ULPS}
    if x is None or np.shape(x) != np.shape(lb) or not np.all(np.isfinite(x)):
        return None, {**audit, 'reason': 'missing_shape_or_nonfinite'}
    normalized = np.array(x, dtype=np.float64, copy=True)
    mask = np.asarray(integrality) != 0
    integers = normalized[mask]
    rounded = np.rint(integers)
    representation = INTEGER_REPRESENTATION_ULPS * np.spacing(np.maximum(abs(rounded), 1.))
    delta = np.abs(integers - rounded)
    audit['integer_residual_before'] = float(np.max(delta, initial=0.))
    audit['outside_machine_representation_bound'] = bool(np.any(delta > representation))
    if np.any((delta > representation) & (delta >= PRIMAL_WITNESS_TOLERANCE)):
        return None, {**audit, 'reason': 'noncanonical_integer'}
    normalized[mask] = rounded
    ax = np.asarray(matrix @ normalized).ravel()
    if not np.all(np.isfinite(ax)):
        return None, {**audit, 'reason': 'nonfinite_row_activity'}
    bound = float(max(np.max(np.maximum(lb - normalized, 0.), initial=0.),
                      np.max(np.maximum(normalized - ub, 0.), initial=0.)))
    row = float(max(np.max(np.maximum(row_lb - ax, 0.), initial=0.),
                    np.max(np.maximum(ax - row_ub, 0.), initial=0.)))
    audit.update(bound_residual=bound, row_residual=row, integer_residual_after=0.)
    if max(bound, row) > PRIMAL_WITNESS_TOLERANCE:
        return None, {**audit, 'reason': 'canonical_primal_infeasible'}
    return normalized, {**audit, 'passed': True, 'reason': 'certified'}


def certify_primary_objective(result, witness, objective, *, tolerance):
    """Use an attained feasible objective and independently check its lower bound."""
    value = float(np.asarray(objective) @ witness)
    reported = getattr(result, 'fun', None)
    lower = getattr(result, 'mip_dual_bound', None)
    audit = {'passed': False, 'objective': value, 'offset_tolerance': tolerance}
    if (reported is None or lower is None
            or not np.all(np.isfinite([value, reported, lower]))):
        return {**audit, 'reason': 'missing_or_nonfinite_objective_bound'}
    audit.update(reported_objective=float(reported), lower_bound=float(lower),
                 absolute_gap=float(value - lower))
    if abs(value - reported) > PRIMAL_WITNESS_TOLERANCE:
        return {**audit, 'reason': 'reported_objective_differs_from_witness'}
    if lower > value + PRIMAL_WITNESS_TOLERANCE or value - lower > tolerance:
        return {**audit, 'reason': 'unproven_primary_optimality'}
    return {**audit, 'passed': True, 'reason': 'certified'}

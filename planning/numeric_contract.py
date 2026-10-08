"""Certificates for solver witnesses used by lexicographic inventory planning.

Physical acceptance is intentionally separate from numerical optimization
acceptance. A subphysical residual must not define the next stage's feasible set.
"""
from __future__ import annotations

import numpy as np

INTEGER_REPRESENTATION_ULPS = 8
PRIMAL_WITNESS_TOLERANCE = 1e-10
NUMERIC_CONTRACT_VERSION = 'canonical-integer-lexicographic-v1'
ONE_STEP_STORAGE_BOUND_VERSION = 'soc-balance-exclusion-one-step-v1'
COUPLED_CHARGE_DOMAIN_VERSION = 'reserve-headroom-charge-domain-v1'
STAGE_B_COORDINATE_VERSION = 'certified-a-continuous-origin-v1'
FIXED_INTEGER_POLISH_VERSION = 'original-primal-fixed-integer-lp-v1'


def translate_continuous_origin(anchor, integrality, bounds, matrix, row_lower, row_upper):
    """Bijective x=y+anchor for continuous variables; leave integer domains intact."""
    from scipy.optimize import Bounds
    shift = np.asarray(anchor, dtype=float).copy()
    shift[np.asarray(integrality) != 0] = 0.
    row_shift = matrix @ shift
    return (shift, Bounds(bounds.lb - shift, bounds.ub - shift),
            np.asarray(row_lower) - row_shift, np.asarray(row_upper) - row_shift)


def coupled_charge_domain_upper(hardware_max, encoded_headroom, minimum_service_power):
    """Valid in both binary domains; round a new redundant upper bound outward."""
    available = encoded_headroom - minimum_service_power
    return min(hardware_max, float(np.nextafter(available, np.inf))) if available > 0. else 0.


def one_step_storage_bounds(initial_energy, final_lower, final_upper, dt, eta_c, eta_d):
    """Bounds implied by one-step SOC balance AND exact charging exclusion.

    In the charging domain, discharge=0 and delta_E=eta_c*dt*charge.
    In the discharging domain, charge=0 and delta_E=-dt/eta_d*discharge.
    Taking their union gives these bounds without changing the feasible domain.
    Use strict signs: a subnanowatt flow is physical input, not an epsilon zero.
    """
    lower = final_lower - initial_energy
    upper = final_upper - initial_energy
    return {'version': ONE_STEP_STORAGE_BOUND_VERSION,
            'initial_energy_kwh': initial_energy, 'final_lower_kwh': final_lower,
            'final_upper_kwh': final_upper,
            'charge_lower_kw': max(lower, 0.) / (eta_c * dt),
            'charge_upper_kw': max(upper, 0.) / (eta_c * dt),
            'discharge_lower_kw': max(-upper, 0.) * eta_d / dt,
            'discharge_upper_kw': max(-lower, 0.) * eta_d / dt,
            'mode_lower': 1. if lower > 0. else 0.,
            'mode_upper': 0. if upper < 0. else 1.}


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


def certify_or_polish_witness(x, lb, ub, integrality, matrix, row_lb, row_ub, *,
                              objective, remaining):
    """One fixed-integer LP for a near-integer witness; never relax the original model.

    Exact witnesses take no extra solve. An invalid integer assignment, nonfinite
    candidate, or expired shared deadline cannot trigger polishing. LP output
    must pass the same strict original-matrix certificate as every other witness.
    """
    import time

    import scipy.optimize
    from scipy.sparse import vstack

    witness, original = certify_witness(x, lb, ub, integrality, matrix, row_lb, row_ub)
    if witness is not None or original['reason'] != 'canonical_primal_infeasible':
        return witness, original
    if max(original['bound_residual'], original['row_residual']) > 1e-6:
        return None, original
    budget = remaining()
    if budget is not None and budget <= 0.:
        return None, original
    started = time.perf_counter()
    mask = np.asarray(integrality) != 0
    fixed = np.rint(np.asarray(x)[mask])
    lower, upper = np.array(lb, copy=True), np.array(ub, copy=True)
    if np.any(fixed < lower[mask]) or np.any(fixed > upper[mask]):
        return None, original
    lower[mask] = upper[mask] = fixed
    row_lower, row_upper = np.asarray(row_lb), np.asarray(row_ub)
    equal = np.isfinite(row_lower) & (row_lower == row_upper)
    finite_upper = np.isfinite(row_upper) & ~equal
    finite_lower = np.isfinite(row_lower) & ~equal
    inequalities = vstack([matrix[finite_upper], -matrix[finite_lower]], format='csr')
    inequality_rhs = np.r_[row_upper[finite_upper], -row_lower[finite_lower]]
    options = {'primal_feasibility_tolerance': PRIMAL_WITNESS_TOLERANCE,
               'dual_feasibility_tolerance': PRIMAL_WITNESS_TOLERANCE,
               'random_seed': 0, 'parallel': False}
    budget = remaining()  # Assembly shares the same deadline too.
    if budget is not None:
        if budget <= 0.:
            return None, original
        options['time_limit'] = budget
    result = scipy.optimize.linprog(
        c=np.asarray(objective), A_eq=matrix[equal], b_eq=row_lower[equal],
        A_ub=inequalities, b_ub=inequality_rhs, bounds=np.column_stack([lower, upper]),
        method='highs-ds', options=options)
    polish = {'version': FIXED_INTEGER_POLISH_VERSION, 'attempted': True,
              'fixed_integer_count': int(np.count_nonzero(mask)), 'options': options,
              'solver_status': int(result.status), 'message': result.message,
              'elapsed_s': time.perf_counter() - started,
              'original_certificate': original}
    if result.status != 0:
        return None, {**original, 'primal_polish': polish}
    polished, certificate = certify_witness(
        result.x, lb, ub, integrality, matrix, row_lb, row_ub)
    return polished, {**certificate, 'primal_polish': polish}

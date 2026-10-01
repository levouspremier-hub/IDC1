"""Sparse assembly must preserve the existing projection feasible domain."""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

import planning.model as model
from contracts.models import DispatchProposal
from planning.snapshot_adapter import build_snapshot
from tests.test_m6p2b_terminal_inventory import _env


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


def test_base_only_target_certificate_is_a_physical_witness_not_a_relaxation():
    snapshot = build_snapshot(_env(soc=.1))
    witness = model._base_only_terminal_certificate(snapshot)
    assert witness is not None
    assert witness['energy_kwh'][-1] == 50.
    energy = snapshot.soc_kwh
    for k, (charge, discharge) in enumerate(zip(
            witness['charge_kw'], witness['discharge_kw'], strict=True)):
        assert min(charge, discharge) == 0.
        assert 0 <= charge <= snapshot.bess_charge_power_max_kw + 1e-9
        assert 0 <= discharge <= snapshot.bess_discharge_power_max_kw + 1e-9
        pf = snapshot.planning_forecast
        assert 0 <= pf.base_idc_power[k] + charge - discharge
        assert pf.base_idc_power[k] + charge - discharge <= (
            snapshot.access_limit_kw + pf.pv[k] + pf.wind[k] + 1e-9)
        energy += snapshot.delta_t_hours * (
            charge * snapshot.bess_charge_efficiency
            - discharge / snapshot.bess_discharge_efficiency)
        assert snapshot.soc_min_kwh - 1e-9 <= energy <= snapshot.soc_max_kwh + 1e-9
    assert abs(energy - 50.) <= 1e-9


def test_no_certificate_does_not_claim_target_unreachable():
    snapshot = build_snapshot(_env(horizon=1, soc=.1))
    assert model._base_only_terminal_certificate(snapshot) is None


def test_invalid_certificate_is_rejected_by_full_physical_constraints(monkeypatch):
    snapshot = build_snapshot(_env(horizon=1))
    monkeypatch.setattr(model, '_base_only_terminal_certificate', lambda snapshot: {
        'energy_kwh': [50., 50.], 'charge_kw': [1.], 'discharge_kw': [1.]})
    result = model.solve_time_indexed_mip_raw_projection(
        snapshot, DispatchProposal(compute_actions=[0.] * 20, storage_action=0.))
    assert result.solver_status == 'optimal'
    assert result.inventory_audit['reachability_method'] == 'mip'


def test_certified_reachability_and_action_solver_timeout_are_separate(monkeypatch):
    from types import SimpleNamespace
    snapshot = build_snapshot(_env(horizon=1))
    monkeypatch.setattr('scipy.optimize.milp', lambda **kw: SimpleNamespace(status=1, x=None))
    result = model.solve_time_indexed_mip_raw_projection(
        snapshot, DispatchProposal(compute_actions=[0.] * 20, storage_action=0.),
        time_limit_s=.25)
    assert result.failure_class == 'timeout'
    assert result.inventory_audit['target_reachable'] is True
    assert result.inventory_audit['reachability_method'] == 'base_only_zero_gap_certificate'
    assert result.inventory_audit['predicted_terminal_kwh'] is None
